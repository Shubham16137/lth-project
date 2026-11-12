"""Sanity tests for the mask logic. Run:  pytest -q"""
import torch
from lth.core import (LeNet300100, ones_masks, magnitude_prune, apply_masks,
                      remaining_fraction, default_rates, prunable_weights)


def test_prune_rate_and_ranking():
    torch.manual_seed(0)
    m = LeNet300100()
    masks = ones_masks(m)
    rates = default_rates("lenet", m)
    new = magnitude_prune(m, masks, rates, "layerwise")
    for n, w in prunable_weights(m).items():
        kept = new[n].float().mean().item()
        assert abs(kept - (1 - rates[n])) < 0.01                    # ~rate pruned
        assert w.abs()[new[n]].min() >= w.abs()[~new[n]].max()      # smallest removed
    assert remaining_fraction(new) < 1.0


def test_pruned_weights_stay_zero_after_training_step():
    torch.manual_seed(0)
    m = LeNet300100()
    masks = magnitude_prune(m, ones_masks(m), default_rates("lenet", m), "layerwise")
    apply_masks(m, masks)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    for _ in range(5):
        loss = m(torch.randn(8, 784)).logsumexp(1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); apply_masks(m, masks)
    for n, w in prunable_weights(m).items():
        assert (w[~masks[n]] == 0).all()


def test_masks_are_nested_across_rounds():
    torch.manual_seed(0)
    m = LeNet300100()
    r = default_rates("lenet", m)
    m1 = magnitude_prune(m, ones_masks(m), r, "layerwise")
    m2 = magnitude_prune(m, m1, r, "layerwise")
    for n in m1:
        assert (m2[n] & ~m1[n]).sum() == 0     # a pruned weight never comes back


def test_global_scope():
    torch.manual_seed(0)
    m = LeNet300100()
    new = magnitude_prune(m, ones_masks(m), {"__global__": 0.5}, "global")
    assert abs(remaining_fraction(new) - 0.5) < 0.01
