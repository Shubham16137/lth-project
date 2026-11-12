# 10-minute presentation outline (8 slides + Q&A)

| # | Time | Slide | Content |
|---|---|---|---|
| 1 | 0:45 | Title + question | "Can we train a *small* network from the start?" Paper, authors, venue (ICLR 2019, best paper). |
| 2 | 1:15 | The hypothesis | Formal statement: ∃ mask m with f(x; m⊙θ₀) reaching accuracy a' ≥ a in j' ≤ j iterations, ‖m‖₀ ≪ \|θ\|. Winning ticket = structure **+** initialisation. |
| 3 | 1:45 | Method: IMP | Algorithm box (train → prune p% smallest surviving → reset to θ₀ → repeat). Remaining fraction (1−p)ⁿ. Diagram of one round. |
| 4 | 1:30 | My implementation | Mask dict, `apply_masks` after every Adam step (why!), nested masks, val-loss early stopping, random-reinit control. Show 10 lines of `magnitude_prune`. |
| 5 | 1:45 | Reproduction result | Main figure: accuracy & early-stop iteration vs % weights remaining, mean ± std over 3 seeds, ticket vs random reinit. State what was scaled down. Compare to paper. |
| 6 | 1:30 | Ablations | Global vs layer-wise, one-shot vs iterative, rewind k. One takeaway per ablation. |
| 7 | 0:45 | Failure cases / limits | What broke; tickets found only after dense training; no wall-clock speed-up on dense hardware. |
| 8 | 0:30 | Conclusion | 3 bullets. Then Q&A. |

Tips: put the code for the mask logic on a backup slide; rehearse deriving (1−p)ⁿ and the Adam-masking argument on a whiteboard.
