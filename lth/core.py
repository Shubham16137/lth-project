"""
Core of the Lottery Ticket Hypothesis implementation (Frankle & Carbin, ICLR 2019).

Everything that is *the paper's contribution* lives in this file and is written
from scratch: mask bookkeeping, magnitude pruning, weight reset (rewinding) and
the iterative-magnitude-pruning (IMP) loop.  Only standard PyTorch building
blocks (nn.Linear, nn.Conv2d, Adam, torchvision datasets) are used otherwise.

Key objects
-----------
masks : dict[str, Tensor]   binary mask (1 = weight alive) for every prunable weight
theta0: state_dict          the initial weights that surviving weights are reset to
"""
from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Models (standard blocks; architectures follow Appendix of the paper)
# --------------------------------------------------------------------------- #
class LeNet300100(nn.Module):
    """Fully-connected 784-300-100-10 network used for MNIST."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)
        self.reset_parameters()

    def reset_parameters(self):
        # Gaussian Glorot initialisation, as in the paper
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ConvNet(nn.Module):
    """Conv-2 / Conv-4 / Conv-6 from the paper (VGG-style, CIFAR-10).

    Conv-2: [64,64]-pool                       Conv-4: + [128,128]-pool
    Conv-6: + [256,256]-pool                   then FC 256-256-10 (dropout 0.5)
    """

    CFG = {2: [64, 64, "M"],
           4: [64, 64, "M", 128, 128, "M"],
           6: [64, 64, "M", 128, 128, "M", 256, 256, "M"]}

    def __init__(self, depth: int = 2, dropout: float = 0.5):
        super().__init__()
        layers, c_in = [], 3
        for v in self.CFG[depth]:
            if v == "M":
                layers.append(nn.MaxPool2d(2))
            else:
                layers += [nn.Conv2d(c_in, v, 3, padding=1), nn.ReLU()]
                c_in = v
        self.features = nn.Sequential(*layers)
        n_pool = self.CFG[depth].count("M")
        flat = c_in * (32 // 2 ** n_pool) ** 2
        self.classifier = nn.Sequential(
            nn.Linear(flat, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 10))
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def build_model(name: str) -> nn.Module:
    if name == "lenet":
        return LeNet300100()
    if name.startswith("conv"):
        return ConvNet(int(name[4:]))
    raise ValueError(name)


# --------------------------------------------------------------------------- #
# Masks and pruning  (the heart of the method)
# --------------------------------------------------------------------------- #
def prunable_weights(model: nn.Module) -> Dict[str, nn.Parameter]:
    """Weights of every Linear / Conv2d layer.  Biases are never pruned."""
    return {f"{n}.weight": m.weight for n, m in model.named_modules()
            if isinstance(m, (nn.Linear, nn.Conv2d))}


def ones_masks(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {n: torch.ones_like(w, dtype=torch.bool) for n, w in prunable_weights(model).items()}


@torch.no_grad()
def apply_masks(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    """Force pruned weights to be exactly zero.

    Called (i) once after a reset and (ii) after *every* optimiser step.
    Doing it after the step (rather than only masking gradients) is required
    with Adam: momentum/second-moment buffers would otherwise keep moving
    weights whose gradient we zeroed.
    """
    for n, w in prunable_weights(model).items():
        w.mul_(masks[n].to(w.dtype))


@torch.no_grad()
def magnitude_prune(model: nn.Module,
                    masks: Dict[str, torch.Tensor],
                    rates: Dict[str, float],
                    scope: str = "layerwise") -> Dict[str, torch.Tensor]:
    """Return NEW masks that additionally remove the lowest-|w| *surviving* weights.

    rates : fraction of the currently-surviving weights to remove.
            layerwise -> one rate per layer name;  global -> rates["__global__"].
    scope : 'layerwise' ranks weights inside each layer;
            'global'    ranks all surviving weights of the network together.
    """
    weights = {n: w.detach().abs() for n, w in prunable_weights(model).items()}
    new = {n: m.clone() for n, m in masks.items()}

    if scope == "layerwise":
        for n, mag in weights.items():
            alive = mag[masks[n]]                      # magnitudes of surviving weights
            k = int(round(rates[n] * alive.numel()))   # how many to remove
            if k <= 0:
                continue
            thr = torch.kthvalue(alive.flatten(), k).values
            new[n] = masks[n] & (mag > thr)
    elif scope == "global":
        alive = torch.cat([mag[masks[n]].flatten() for n, mag in weights.items()])
        k = int(round(rates["__global__"] * alive.numel()))
        if k > 0:
            thr = torch.kthvalue(alive, k).values
            for n, mag in weights.items():
                new[n] = masks[n] & (mag > thr)
    else:
        raise ValueError(scope)
    return new


def remaining_fraction(masks: Dict[str, torch.Tensor]) -> float:
    alive = sum(int(m.sum()) for m in masks.values())
    total = sum(m.numel() for m in masks.values())
    return alive / total


def layer_remaining(masks: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {n: float(m.float().mean()) for n, m in masks.items()}


# Per-layer pruning rates per round (Appendix of the paper: 20% FC, 10% output / conv 10-15%).
# These are DEFAULTS -- verify against the paper's appendix for your final report.
def default_rates(model_name: str, model: nn.Module) -> Dict[str, float]:
    names = list(prunable_weights(model).keys())
    rates = {}
    for i, n in enumerate(names):
        last = i == len(names) - 1
        is_conv = n.startswith("features")
        rates[n] = 0.10 if (last or is_conv) else 0.20
    return rates


# --------------------------------------------------------------------------- #
# Data  (whole dataset lives on the training device => fast, no DataLoader)
# --------------------------------------------------------------------------- #
@dataclass
class Data:
    xtr: torch.Tensor
    ytr: torch.Tensor
    xva: torch.Tensor
    yva: torch.Tensor
    xte: torch.Tensor
    yte: torch.Tensor


def load_data(name: str, device, root: str = "./data", val_size: int = 5000,
              subset: Optional[int] = None, seed: int = 0) -> Data:
    g = torch.Generator().manual_seed(1234)   # fixed split across runs
    if name == "synthetic":                   # smoke tests only (no download)
        n, shape = 6000, (1, 28, 28)
        x = torch.randn(n, *shape)
        y = (x.flatten(1)[:, :10].argmax(1))  # learnable toy rule
        xte, yte = x[:1000], y[:1000]
        x, y = x[1000:], y[1000:]
    else:
        import torchvision
        ds = {"mnist": torchvision.datasets.MNIST,
              "cifar10": torchvision.datasets.CIFAR10}[name]
        tr, te = ds(root, train=True, download=True), ds(root, train=False, download=True)
        to_t = lambda d: torch.as_tensor(np.asarray(d.data), dtype=torch.float32)
        x, xte = to_t(tr), to_t(te)
        y = torch.as_tensor(tr.targets if name == "cifar10" else tr.targets.numpy())
        yte = torch.as_tensor(te.targets if name == "cifar10" else te.targets.numpy())
        if name == "mnist":
            x, xte = x.unsqueeze(1) / 255.0, xte.unsqueeze(1) / 255.0
            mean, std = 0.1307, 0.3081
            x, xte = (x - mean) / std, (xte - mean) / std
        else:
            x, xte = x.permute(0, 3, 1, 2) / 255.0, xte.permute(0, 3, 1, 2) / 255.0
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
            std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
            x, xte = (x - mean) / std, (xte - mean) / std
    perm = torch.randperm(len(x), generator=g)
    x, y = x[perm], y[perm]
    xva, yva, xtr, ytr = x[:val_size], y[:val_size], x[val_size:], y[val_size:]
    if subset:
        xtr, ytr = xtr[:subset], ytr[:subset]
    return Data(*(t.to(device) for t in (xtr, ytr, xva, yva, xte, yte)))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class TrainCfg:
    iters: int = 5000
    lr: float = 1.2e-3
    batch: int = 60
    eval_every: int = 100
    start_iter: int = 0          # >0 when training resumes from a rewound checkpoint


@torch.no_grad()
def evaluate(model, x, y, bs: int = 5000):
    model.eval()
    loss, correct = 0.0, 0
    for i in range(0, len(x), bs):
        out = model(x[i:i + bs])
        loss += F.cross_entropy(out, y[i:i + bs], reduction="sum").item()
        correct += (out.argmax(1) == y[i:i + bs]).sum().item()
    return loss / len(x), correct / len(x)


def train(model: nn.Module, masks, data: Data, cfg: TrainCfg,
          snapshot_at: Optional[int] = None) -> dict:
    """Train with Adam, re-applying masks after each step.

    Early stopping criterion (as in the paper): the iteration with minimum
    *validation* loss.  Test accuracy is reported at that iteration, so the
    test set is never used to pick anything.
    Returns curves + (optionally) a snapshot of the weights at `snapshot_at`.
    """
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    apply_masks(model, masks)
    n = len(data.xtr)
    hist = {"it": [], "val_loss": [], "val_acc": [], "test_acc": []}
    snap = None
    order, ptr = torch.randperm(n, device=data.xtr.device), 0

    def log(it):
        vl, va = evaluate(model, data.xva, data.yva)
        _, ta = evaluate(model, data.xte, data.yte)
        hist["it"].append(it + cfg.start_iter)
        hist["val_loss"].append(vl); hist["val_acc"].append(va); hist["test_acc"].append(ta)

    log(0)
    for it in range(1, cfg.iters + 1):
        model.train()
        if ptr + cfg.batch > n:                           # new epoch
            order, ptr = torch.randperm(n, device=data.xtr.device), 0
        idx = order[ptr:ptr + cfg.batch]; ptr += cfg.batch
        loss = F.cross_entropy(model(data.xtr[idx]), data.ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        apply_masks(model, masks)                         # keep pruned weights at 0
        if snapshot_at is not None and it + cfg.start_iter == snapshot_at:
            snap = copy.deepcopy(model.state_dict())
        if it % cfg.eval_every == 0 or it == cfg.iters:
            log(it)

    best = int(np.argmin(hist["val_loss"]))
    return {"hist": hist,
            "early_stop_iter": hist["it"][best],
            "test_acc_at_es": hist["test_acc"][best],
            "final_test_acc": hist["test_acc"][-1],
            "final_val_acc": hist["val_acc"][-1],
            "snapshot": snap}


# --------------------------------------------------------------------------- #
# Iterative Magnitude Pruning  (Algorithm in Sec. 1 of the paper)
# --------------------------------------------------------------------------- #
@dataclass
class IMPCfg:
    model: str = "lenet"
    rounds: int = 15
    scope: str = "layerwise"        # 'layerwise' | 'global'
    schedule: str = "iterative"     # 'iterative' | 'oneshot'
    rewind_iter: int = 0            # 0 = reset to theta_0 (original paper); k>0 = late rewinding
    random_reinit: bool = True      # also train the random-reinit control
    global_rate: float = 0.20       # per-round rate used when scope == 'global'
    train: TrainCfg = field(default_factory=TrainCfg)


def run_imp(cfg: IMPCfg, data: Data, device, seed: int,
            progress: Optional[Callable[[dict], None]] = None) -> List[dict]:
    """Find winning tickets and (optionally) run the random-reinit control.

    Round 0 = dense network.  Round r keeps (1-p)^r of the weights.
    Returns one record per round.
    """
    set_seed(seed)
    model = build_model(cfg.model).to(device)
    rates = default_rates(cfg.model, model)
    theta0 = copy.deepcopy(model.state_dict())      # <-- theta_0, the "ticket" initialisation
    masks = ones_masks(model)
    theta_rewind = theta0                           # replaced if rewind_iter > 0
    records: List[dict] = []
    dense_trained = None                            # trained dense weights (for one-shot)

    for r in range(cfg.rounds + 1):
        # ---- 1. build mask for this round ------------------------------------
        if r > 0:
            if cfg.schedule == "iterative":
                # prune the *trained* weights of the previous round
                masks = magnitude_prune(model, masks, _round_rates(cfg, rates), cfg.scope)
            else:  # one-shot: prune the trained dense net once to the target sparsity
                model.load_state_dict(dense_trained)
                frac = {n: 1 - (1 - p) ** r for n, p in rates.items()}
                frac["__global__"] = 1 - (1 - cfg.global_rate) ** r
                masks = magnitude_prune(model, ones_masks(model), frac, cfg.scope)

        # ---- 2. reset surviving weights to theta_0 (or theta_k) -------------
        model.load_state_dict(theta_rewind)
        masks = {n: m.to(device) for n, m in masks.items()}
        apply_masks(model, masks)

        rec = {"round": r, "remaining": remaining_fraction(masks),
               "layer_remaining": layer_remaining(masks)}

        # ---- 3. train the winning-ticket candidate ---------------------------
        k = cfg.rewind_iter
        tcfg = copy.copy(cfg.train)
        tcfg.start_iter = k if r > 0 else 0
        tcfg.iters = cfg.train.iters - (k if r > 0 else 0)
        t0 = time.time()
        out = train(model, masks, data, tcfg, snapshot_at=(k if (r == 0 and k > 0) else None))
        rec["ticket"] = {kk: v for kk, v in out.items() if kk != "snapshot"}
        rec["seconds"] = time.time() - t0
        if r == 0:
            dense_trained = copy.deepcopy(model.state_dict())
            if k > 0:
                theta_rewind = out["snapshot"]       # theta_k from the dense run

        # ---- 4. control: same mask, fresh random initialisation -------------
        if cfg.random_reinit and r > 0:
            ctrl = build_model(cfg.model).to(device)   # new random init (global RNG advances)
            cout = train(ctrl, masks, data, cfg.train)
            rec["random_reinit"] = {kk: v for kk, v in cout.items() if kk != "snapshot"}
        records.append(rec)
        if progress:
            progress(rec)
        # `model` still holds this round's trained weights -> pruned at top of next loop
    return records


def _round_rates(cfg: IMPCfg, rates: Dict[str, float]) -> Dict[str, float]:
    out = dict(rates)
    out["__global__"] = cfg.global_rate
    return out
