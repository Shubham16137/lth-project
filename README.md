# The Lottery Ticket Hypothesis — from-scratch PyTorch implementation

Re-implementation and reduced-scale reproduction of

> J. Frankle and M. Carbin, *The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks*, ICLR 2019. arXiv:1803.03635

**Claim being tested.** A dense, randomly initialised network contains a sparse subnetwork
(a *winning ticket*) that, trained in isolation **from its original initialisation θ₀**, matches the dense
network's test accuracy in at most the same number of iterations. The same sparse structure with a
*fresh random* initialisation does not.

## What is implemented from scratch (`lth/core.py`)

| Piece | Where |
|---|---|
| Binary mask bookkeeping for every Linear/Conv2d weight | `ones_masks`, `apply_masks` |
| Magnitude pruning (layer-wise **and** global, on *surviving* weights only) | `magnitude_prune` |
| Reset to θ₀ / rewind to θₖ, keeping the mask fixed | `run_imp` |
| Iterative magnitude pruning loop + random-reinit control | `run_imp` |
| Early stopping on **validation** loss, test accuracy reported at that iteration | `train` |

Standard PyTorch parts (allowed by the brief): `nn.Linear`, `nn.Conv2d`, Adam, torchvision datasets.

## Setup

```bash
git clone <this repo> && cd lth-project
pip install -r requirements.txt
pytest -q                      # mask-logic sanity tests
```

Or open `LTH_Colab.ipynb` in Google Colab (runtime: T4 GPU) and run all cells.

## How to run

```bash
# 1) Main reproduction: LeNet-300-100 on MNIST, 15 pruning rounds (down to ~3.5 % of weights)
python run_experiments.py --name lenet_main --model lenet --dataset mnist \
       --rounds 15 --iters 10000 --seeds 0 1 2
python plot_results.py results/lenet_main.json --out figures/lenet_main

# 2) Ablations (same settings, one factor changed)
python run_experiments.py --name lenet_global  --scope global     --rounds 15 --iters 10000 --seeds 0 1 2
python run_experiments.py --name lenet_oneshot --schedule oneshot --rounds 15 --iters 10000 --seeds 0 1 2
python run_experiments.py --name lenet_rewind  --rewind-iter 500  --rounds 15 --iters 10000 --seeds 0 1 2
python plot_results.py --compare results/lenet_main.json results/lenet_global.json \
       results/lenet_oneshot.json results/lenet_rewind.json \
       --labels "layer-wise IMP" "global IMP" "one-shot" "rewind k=500" --out figures/ablations

# 3) Optional second architecture (Colab T4)
python run_experiments.py --name conv2_main --model conv2 --dataset cifar10 \
       --lr 2e-4 --iters 10000 --rounds 10 --seeds 0 1 2
```

Seeds: `0 1 2` for every reported number (mean ± std over seeds). The train/validation split
(55 000 / 5 000 for MNIST) is fixed independently of the seed. Results are written to `results/<name>.json`
after each seed finishes.

Quick smoke test without downloading data: `python run_experiments.py --name smoke --dataset synthetic --rounds 3 --iters 300 --seeds 0`.

## Deviations from the paper (state these in the report)

* Training length: the paper trains LeNet far longer; defaults here are shorter (`--iters`). Reduce/increase to fit your compute.
* Early stopping uses a 5 000-image validation split taken from the training set (paper-style criterion: minimum validation loss).
* Per-round pruning rates default to 20 % (FC), 10 % (output layer), 10 % (conv) — **check against the paper's appendix** and edit `default_rates` if needed.
* Only 3 seeds.

## Results

> Fill this table from `plot_results.py` output after running (it prints a markdown table).

| Weights remaining (%) | Ticket acc (%) | Random reinit acc (%) | Ticket ES iter | Random ES iter |
|---|---|---|---|---|
| 100.0 | _96.7_ | – | _98.9_ | – |
| … | | | | |

Figures: `figures/lenet_main.png`, `figures/ablations.png`.

## References

1. J. Frankle, M. Carbin, ICLR 2019 (arXiv:1803.03635).
2. J. Frankle, G. K. Dziugaite, D. M. Roy, M. Carbin, *Stabilizing the Lottery Ticket Hypothesis*, 2019 (arXiv:1903.01611) — for the rewinding ablation.
3. *Repositories consulted:* _list any official/third-party repository you looked at here — the brief requires this._
