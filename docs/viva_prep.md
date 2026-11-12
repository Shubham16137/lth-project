# Viva preparation

## Line-by-line of the core module (`lth/core.py`)
* `ones_masks` — one boolean tensor per Linear/Conv2d weight; biases are not pruned.
* `apply_masks` — `w *= mask`. Called after reset and after **every** optimiser step.
  *Why not just zero the gradient?* Adam keeps momentum (m) and second-moment (v) buffers; with a zeroed gradient the update is still non-zero (m decays, not zero), so pruned weights would drift. Multiplying weights after the step is exact.
* `magnitude_prune` — rank only **surviving** weights (`mag[mask]`), take the k-th smallest as threshold (`kthvalue`), new mask = old mask AND (|w| > threshold). Hence masks are nested: a pruned weight never returns.
* `run_imp` — saves θ₀ at the start; each round: build mask from the *trained* weights of the previous round → `load_state_dict(theta0)` → apply mask → train. The control builds a brand-new model with the same mask.
* `train` — early stopping = argmin of validation loss; test accuracy is read at that iteration (test set never selects anything).

## Equations to derive on the board
1. **Sparsity after n rounds.** Each round keeps (1−p) of the *surviving* weights ⇒ remaining = (1−p)ⁿ. For 20 % per round: 0.8¹⁵ ≈ 3.5 %. To reach a target fraction f: n = ln f / ln(1−p). (Mixed rates: weight-count-weighted average over layers.)
2. **Hypothesis statement.** With mask m ∈ {0,1}^{|θ|}: training f(x; m⊙θ₀) reaches (j', a') with j' ≤ j, a' ≥ a, ‖m‖₀ ≪ |θ|, where (j, a) belong to the dense net f(x; θ₀).
3. **Why masked weights stay zero under SGD/Adam.** Gradient wrt w_ij of the masked network is m_ij·∂L/∂(m⊙w)_ij ⇒ zero when m_ij = 0 for SGD (so mask-by-gradient suffices); for Adam, bias-corrected m̂/(√v̂+ε) can be non-zero from history, so post-step masking is needed.

## Likely questions
* **Why reset to θ₀?** Paper's key finding: same mask with random reinit trains worse/slower — structure and initialisation are co-dependent.
* **What does magnitude pruning assume?** Small-magnitude weights contribute little after training (a heuristic, not a guarantee).
* **Why does late rewinding help in deep nets?** Frankle et al. 2019: for deeper nets tickets are only stable after a few early iterations (SGD noise sensitivity), so reset to θ_k instead of θ₀.
* **Layer-wise vs global?** Global lets the network allocate sparsity unevenly; layer-wise protects small layers (e.g. the 10-unit output layer) — hence lower rate there.
* **Why compare at early stopping?** The claim is about reaching a given accuracy *no later* than the dense net.
* **Limitations.** Tickets found only after training the dense net; unstructured sparsity needs sparse kernels for speed-up; small-scale evidence (MNIST/CIFAR-10) — larger nets need rewinding.
* **What failed in your runs?** (Prepare from your own results.)
* **What did you change vs the paper?** (Know your deviation list from the README.)
