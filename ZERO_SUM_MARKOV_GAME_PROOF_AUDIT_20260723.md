# Zero-sum Markov-game proof audit (2026-07-23)

## Decision

The RL reframing is mathematically viable without changing the curvature
direction `G = DF F`.  The rigorous claim must be split into:

1. a policy-space performance bridge for finite discounted two-player zero-sum
   Markov games;
2. the existing local stochastic saddle-field drift theory for parameterized
   (including neural) policies; and
3. an explicit entropy/approximation residual when the smooth regularized gap
   is used in place of the unregularized best-response gap.

It is not valid to claim that arbitrary neural stationarity implies a global
Nash equilibrium.  The current manuscript already states this limitation.

## Exact performance bridge

For protagonist policy `pi`, adversary policy `nu`, and return `J(pi,nu)`, define

```
R_BR(pi) = inf_{nu'} J(pi,nu')
Gap_0(pi,nu) = sup_{pi'} J(pi',nu) - inf_{nu'} J(pi,nu').
```

Let the zero-sum game have value

```
v* = sup_pi inf_nu J(pi,nu) = inf_nu sup_pi J(pi,nu).
```

Then, for every policy pair,

```
0 <= v* - R_BR(pi) <= Gap_0(pi,nu).
```

Proof: `sup_{pi'} J(pi',nu) >= inf_nu sup_pi J(pi,nu) = v*` for every
`nu`.  Subtract `inf_{nu'} J(pi,nu')` from both sides.  Thus exploitability
controls protagonist robust-value suboptimality directly.

## Smooth entropy-regularized bridge

For finite action spaces, define the discounted causal entropies `H_pi` and
`H_nu` and the regularized zero-sum return

```
J_tau(pi,nu) = J(pi,nu) + tau H_pi(pi,nu) - tau H_nu(pi,nu).
```

The signs preserve concavity for protagonist maximization and convexity for
adversary minimization.  Define the regularized Nash gap

```
Gap_tau(pi,nu)
  = sup_{pi'} J_tau(pi',nu) - inf_{nu'} J_tau(pi,nu').
```

On the interior policy class, entropy makes the regularized best responses
unique under the standard finite-game conditions.  Their value envelopes are
smooth wherever the regularized Bellman solution is smooth.  With the entropy
sign convention above, the uniform bound is

```
Gap_0(pi,nu)
 <= Gap_tau(pi,nu)
    + tau [log |A| + log |B|] / (1-alpha).
```

This follows by bounding the protagonist supremum perturbation by
`tau log |B|/(1-alpha)` and the adversary infimum perturbation by
`tau log |A|/(1-alpha)`.  Adding the two bounds gives the displayed constant;
no extra factor of two is needed.

Combining the two displays gives

```
v* - R_BR(pi)
 <= Gap_tau(pi,nu)
    + tau [log |A| + log |B|] / (1-alpha).
```

This is the missing rigorous link between a smooth Lyapunov component and the
RARL performance metric.

## Lyapunov and unchanged direction

Use the regularized policy-gradient field `F_tau` (the entropy terms are part
of `J`) and retain exactly

```
G_tau(z) = D F_tau(z) F_tau(z).
```

The proposed RL-aware Lyapunov is

```
V(z) = lambda_F E_F(z)/kappa_F
     + lambda_Gap Gap_tau(z)/kappa_Gap
     + lambda_C C_critic(z)/kappa_C.
```

The existing two-dimensional drift model and box-QP apply to any nonnegative
`C^3` Lyapunov on the admissible region.  Therefore the definition of `G`, the
dominance identity, and the local stochastic drift analysis do not need to be
changed.  What changes is the RL interpretation of the second Lyapunov term.

For large or neural games, a fixed number of differentiable regularized
best-response inner steps realizes a smooth approximate gap.  Inner-solver,
trajectory-truncation, critic, and finite-batch errors must enter the existing
coefficient/oracle residual budget.  Held-out BR evaluation remains separate
and is never differentiated through for method selection.

## Scope of theorems

- Global policy-performance bridge: finite discounted zero-sum Markov game.
- Smooth exact gap: entropy-regularized tabular/interior policy setting.
- Neural policies: local admissible-region stationarity and Lyapunov drift,
  plus empirical held-out BR; no global-Nash claim.
- Physical RARL: a special case in which the adversary action perturbs the
  transition dynamics.

## Required manuscript changes

1. Add unregularized exploitability and robust-BR definitions after the Markov
   game formulation.
2. Add the exact inequality `v* - R_BR <= Gap_0`.
3. Add the entropy-regularized gap and its approximation-bias bound.
4. Clarify that the existing local proximal gap is an implementable local
   surrogate, while the global regularized gap is used for the policy-space
   performance statement and exact tabular experiment.
5. State explicitly that neural results are local and evaluated with an
   independently trained held-out best response.

## Go/no-go

GO for the zero-sum Markov-game RL reframing.  NO-GO for any claim that field
stationarity alone certifies robust performance under unrestricted neural
parameterization.
