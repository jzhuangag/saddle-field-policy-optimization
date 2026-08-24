# Stochastic-oracle method audit

## Accepted construction

Let `z_bar` be the behavior parameters that generated a finite trajectory and
let

`w_0:t(z,z_bar) = product_{j=0}^t
 pi_z(a_j|s_j) nu_z(b_j|s_j) /
 [pi_z_bar(a_j|s_j) nu_z_bar(b_j|s_j)]`.

The implemented sampled objective is the average of

`sum_t alpha^t w_0:t(z,z_bar)
 [r_t + epsilon H(pi_z(.|s_t)) - epsilon H(nu_z(.|s_t))]`.

For every finite horizon and every target `z`, the per-decision change-of-measure
identity makes its expectation under the behavior trajectories equal to the
target-policy finite-horizon regularized return.  Neural softmax policies have
full action support.  Because the state and action spaces and the rollout
horizon are finite, differentiation can be interchanged with expectation on
every bounded local parameter set.  Consequently, differentiating once gives a
valid stochastic finite-horizon game field, and differentiating twice gives a
valid stochastic Hessian estimator.  At `z = z_bar`, all numerical ratios equal
one and repeated autodifferentiation is the DiCE likelihood-ratio construction.

The implemented directions are

`F_hat = S grad J_hat_H` and
`G_hat = S Hessian(J_hat_H) F_hat`,

where `S = diag(-I,I)`.  Because the same batch supplies the Hessian and field,
`G_hat` is a consistent but generally biased product estimator: its expectation
contains a finite-batch covariance term.  This is intentional and falls under
the manuscript's biased-curvature-oracle assumption.  Rollout truncation is a
second explicit oracle bias relative to the infinite-horizon population field.

QP coefficient stencils and backtracking reuse the behavior batch with the
explicit prefix likelihood ratios.  Thus their off-behavior evaluations retain
the trajectory-distribution derivative.  Exact soft or hard best-response
solvers are never called during training; they are checkpoint evaluators only.

## Numerical sentinels

`sentinels_dice.py` checks:

1. exact batch reproducibility;
2. unit importance ratios at the behavior point;
3. agreement between the autograd Hessian-vector action and a same-batch
   central difference;
4. convergence in direction of averaged stochastic `F_hat` and `G_hat` to
   finite-horizon population counterparts;
5. positive-definite QP stabilization, box feasibility, accepted same-batch
   merit decrease, and dynamic-programming residuals.

The frozen journal artifacts retain the final CSV/JSON evidence and audit
reports; local logs and exploratory outputs are intentionally excluded from
the public repository.
