"""Run IMP experiments over several seeds and save JSON results.

Examples
--------
# Main reproduction (LeNet-300-100 / MNIST), winning ticket vs random reinit
python run_experiments.py --name lenet_main --model lenet --dataset mnist \
    --rounds 15 --iters 5000 --seeds 0 1 2

# Ablations
python run_experiments.py --name lenet_global   --scope global    ...
python run_experiments.py --name lenet_oneshot  --schedule oneshot ...
python run_experiments.py --name lenet_rewind   --rewind-iter 500  ...

# Conv-2 on CIFAR-10 (Colab T4)
python run_experiments.py --name conv2_main --model conv2 --dataset cifar10 \
    --lr 2e-4 --iters 10000 --rounds 10 --seeds 0 1 2

# Fast smoke test, no downloads
python run_experiments.py --name smoke --dataset synthetic --rounds 3 --iters 300 --seeds 0
"""
import argparse
import json
import os
import time

import torch

from lth.core import IMPCfg, TrainCfg, load_data, run_imp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="lenet", choices=["lenet", "conv2", "conv4", "conv6"])
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10", "synthetic"])
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--iters", type=int, default=5000, help="training iterations per round")
    ap.add_argument("--lr", type=float, default=1.2e-3)
    ap.add_argument("--batch", type=int, default=60)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--scope", default="layerwise", choices=["layerwise", "global"])
    ap.add_argument("--global-rate", type=float, default=0.20)
    ap.add_argument("--schedule", default="iterative", choices=["iterative", "oneshot"])
    ap.add_argument("--rewind-iter", type=int, default=0)
    ap.add_argument("--no-random-reinit", action="store_true")
    ap.add_argument("--subset", type=int, default=None, help="use only N training examples")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    device = torch.device(a.device)
    data = load_data(a.dataset, device, subset=a.subset)
    cfg = IMPCfg(model=a.model, rounds=a.rounds, scope=a.scope, schedule=a.schedule,
                 rewind_iter=a.rewind_iter, random_reinit=not a.no_random_reinit,
                 global_rate=a.global_rate,
                 train=TrainCfg(iters=a.iters, lr=a.lr, batch=a.batch, eval_every=a.eval_every))

    all_runs = []
    for seed in a.seeds:
        t0 = time.time()

        def show(rec):
            line = (f"[seed {seed}] round {rec['round']:2d}  remaining {100*rec['remaining']:6.2f}%  "
                    f"ticket acc@ES {100*rec['ticket']['test_acc_at_es']:.2f} (iter {rec['ticket']['early_stop_iter']})")
            if "random_reinit" in rec:
                line += f" | random-reinit acc@ES {100*rec['random_reinit']['test_acc_at_es']:.2f}"
            print(line, flush=True)

        records = run_imp(cfg, data, device, seed, progress=show)
        all_runs.append({"seed": seed, "records": records})
        print(f"seed {seed} done in {time.time()-t0:.0f}s", flush=True)
        # save after each seed so a Colab disconnect does not lose everything
        with open(os.path.join(a.out, f"{a.name}.json"), "w") as f:
            json.dump({"args": vars(a), "runs": all_runs}, f)


if __name__ == "__main__":
    main()
