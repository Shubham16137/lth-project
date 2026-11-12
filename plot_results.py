"""Plot mean +- std over seeds and print a markdown results table.

python plot_results.py results/lenet_main.json --out figures/lenet_main
python plot_results.py --compare results/lenet_main.json results/lenet_global.json \
       --labels "layerwise" "global" --out figures/ablation_scope
"""
import argparse
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN round-0 control


def load(path):
    with open(path) as f:
        d = json.load(f)
    runs = d["runs"]
    rounds = len(runs[0]["records"])
    rem = np.array([r["remaining"] for r in runs[0]["records"]]) * 100

    def stack(key, field):
        rows = []
        for run in runs:
            rows.append([rec[key][field] if key in rec else np.nan for rec in run["records"]])
        return np.array(rows)          # (seeds, rounds)

    out = {"remaining": rem, "n_seeds": len(runs)}
    for key in ("ticket", "random_reinit"):
        for field in ("test_acc_at_es", "early_stop_iter", "final_test_acc"):
            out[(key, field)] = stack(key, field)
    return out


def band(ax, x, arr, label, **kw):
    m, s = np.nanmean(arr, 0), np.nanstd(arr, 0)
    ax.plot(x, m, marker="o", ms=3, label=label, **kw)
    ax.fill_between(x, m - s, m + s, alpha=0.2)


def style(ax, ylabel):
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Weights remaining (%)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, which="both")
    ax.legend()


def plot_single(path, out):
    d = load(path)
    x = d["remaining"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    band(ax[0], x, 100 * d[("ticket", "test_acc_at_es")], "winning ticket (reset to $\\theta_0$)")
    band(ax[0], x, 100 * d[("random_reinit", "test_acc_at_es")], "random reinit")
    style(ax[0], "Test accuracy at early stop (%)")
    band(ax[1], x, d[("ticket", "early_stop_iter")], "winning ticket")
    band(ax[1], x, d[("random_reinit", "early_stop_iter")], "random reinit")
    style(ax[1], "Early-stop iteration")
    fig.suptitle(f"{os.path.basename(path)}  (mean $\\pm$ std, {d['n_seeds']} seeds)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf")
    print("saved", out + ".png")
    table(d)


def plot_compare(paths, labels, out):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for p, lab in zip(paths, labels):
        d = load(p)
        band(ax[0], d["remaining"], 100 * d[("ticket", "test_acc_at_es")], lab)
        band(ax[1], d["remaining"], d[("ticket", "early_stop_iter")], lab)
    style(ax[0], "Test accuracy at early stop (%)")
    style(ax[1], "Early-stop iteration")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf")
    print("saved", out + ".png")


def table(d, every=3):
    print("\n| Weights remaining (%) | Ticket acc (%) | Random reinit acc (%) | Ticket ES iter | Random ES iter |")
    print("|---|---|---|---|---|")
    for r in range(0, len(d["remaining"]), every):
        f = lambda k, s=1: (f"{s*np.nanmean(d[k][:, r]):.2f} ± {s*np.nanstd(d[k][:, r]):.2f}"
                            if not np.all(np.isnan(d[k][:, r])) else "—")
        print(f"| {d['remaining'][r]:.1f} | {f(('ticket','test_acc_at_es'),100)} | "
              f"{f(('random_reinit','test_acc_at_es'),100)} | {f(('ticket','early_stop_iter'))} | "
              f"{f(('random_reinit','early_stop_iter'))} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="*")
    ap.add_argument("--compare", nargs="+")
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--out", default="figures/result")
    a = ap.parse_args()
    if a.compare:
        plot_compare(a.compare, a.labels or a.compare, a.out)
    else:
        plot_single(a.results[0], a.out)
