# VI-C Full MuJoCo PPO-RARL Protocol (2026-07-19)

## Status and supersession

The earlier HalfCheetah-v4 result is a 50k-protagonist-step pilot. It is useful as
an integration check, but it is not the final VI-C convergence experiment. The
full suite below supersedes that pilot for any final paper claim.

## Environments and game construction

- Environments: Gymnasium `HalfCheetah-v4`, `Hopper-v4`, and `Walker2d-v4`.
- Task reward: unchanged native Gymnasium reward.
- Protagonist action: native MuJoCo actuator action.
- Adversary action: two-dimensional bounded action controlling horizontal and
  vertical external force on the torso through `data.xfrc_applied`.
- Maximum force magnitude per active coordinate: `adv_fraction=2.5`.
- No `u^T M w`, cross-action bonus, reward shaping, or method-dependent reward.
- Deliberate RARL design choices that must be disclosed: attacked body (`torso`),
  force coordinates (`x,z`), action bound/force scale (`2.5`), and alternating
  update schedule.

## Policies and training budget

- Both players use PPO with Gaussian MLP policies.
- Protagonist policy network: two 256-unit hidden layers.
- Adversary policy network: two 64-unit hidden layers.
- Alternation: `N_mu=5` protagonist phases and `N_nu=1` adversary phase.
- Update order: each outer iteration performs all five protagonist PPO
  rollout/update phases first, with the current adversary frozen, and then one
  adversary PPO rollout/update phase with the updated protagonist frozen.
- Warm-up: none. The runner leaves `adv_delay=-1`, so adversary optimization is
  enabled in outer iteration zero. The first protagonist block nevertheless
  occurs before the first adversary block (10,240 protagonist environment steps
  at the 2048-step rollout length); this ordering effect is not a separately
  trained protagonist warm-up.
- This is alternating optimization, not a simultaneous joint update. During a
  player's rollout, the opponent acts deterministically from its frozen current
  policy. The manual optimizer constructs its direction from that player's
  detached PPO minibatch objective; it does not differentiate a single joint
  protagonist--adversary computation graph.
- Rollout length: 2048 environment steps per phase for both players. This makes
  the realized sample ratio agree with the nominal 5:1 phase ratio.
- Full budget: 100 outer iterations, approximately 1,024,000 protagonist steps
  and 204,800 adversary steps per method and seed.
- Paired seeds: `{0,1,2,3,4}`.
- Evaluation every 25k protagonist steps with 20 episodes per checkpoint.
- Evaluation channels: training return, clean evaluation, and learned-adversary
  force evaluation. The force evaluation is the primary robust endpoint.
- At evaluation, learning is paused and both policies act deterministically.
  Each method's protagonist is paired with that method's own concurrently trained
  adversary. Thus the learned-adversary curve and QP+G-minus-noG paired gain
  measure co-trained-pair performance, not cross-play against a single shared
  adversary. Clean evaluation disables the adversarial force channel.

## Optimizer baselines

All rows use PPO; the following names identify the optimizer used inside each
player's PPO update:

- `sgd` -> GDA/SGD baseline.
- `egm` -> extra-gradient baseline.
- `ppm` -> proximal-point approximation, three fixed-point inner steps. The earlier
  two-step configuration is excluded because it is algebraically identical to the
  implemented EGM update and produced pointwise-identical curves.
- `proposed_noG` -> adaptive field-only ablation.
- `proposed_qp` -> adaptive field-plus-curvature method (QP+G).

These are not five different RL algorithms and must not be described as such.
Because PPO alternates detached on-policy batches, these optimizer labels are also
not identical to simultaneous exact-game GDA/EGM/PPM.

## Three-layer rotation audit

The audit is kept separate from PPO return evaluation.

1. Reward/one-step layer:
   record the mixed finite-difference norm of the native one-step MuJoCo reward
   with respect to protagonist action and torso-force action. Also record that the
   explicit bilinear reward term is absent.
2. Action-value layer:
   fit a smooth joint-action critic `Q(s,u,w)` and record `||Q_uw||`, `||Q_uu||`,
   `||Q_ww||`, and `A_act=||Q_uw||/sqrt(||Q_uu||||Q_ww||)`.
3. Parameter-field layer:
   use JVP/VJP products of the joint actor field to record
   `R_param=||WF||/||SF||` and the F-independent Hutchinson estimate
   `||W||_F/||S||_F`.
4. Algorithm layer (not a fourth rotation definition):
   record `d=-<grad V,G>`, the reduced-gradient proxy at the noG point, and the
   predicted QP-vs-noG decrease gap.

The critic/JVP diagnostic is an explanatory screen. The theorem-level rotation
quantity is the parameter-field metric, not `A_act` and not the one-step reward
mixed derivative.

## Current three-layer finding at 2.5 N

| Environment | A_act | R_param | Matrix ratio | A_act/R_param | Predicted gap |
|---|---:|---:|---:|---:|---:|
| HalfCheetah-v4 | 0.473 | 0.082 | 0.036 | 6.13 | 0.0264 |
| Hopper-v4 | 0.239 | 0.020 | 0.0049 | 12.56 | 0 |
| Walker2d-v4 | 0.323 | 0.034 | 0.0076 | 9.39 | 0.0143 |

All three parameter-field ratios are below one. Therefore a positive return gap,
if observed, cannot be presented as evidence that these native MuJoCo runs are
uniformly in the theorem's skew-dominated regime. It may still be consistent with
the composite-Lyapunov reduced-gradient mechanism, which is seed dependent here.

## Paper qualification rule

An environment qualifies for the final VI-C main figure only if:

- the 1M-step curves are finite and visibly reach a late-training plateau;
- QP+G has a positive paired robust endpoint and does not materially sacrifice
  clean return;
- the endpoint is not created by a single isolated checkpoint;
- robust AUC and late-window behavior are reported alongside the endpoint;
- uncertainty and seed wins are shown; and
- the rotation audit is reported without replacing `R_param` by `A_act` or gamma
  activation.

## Final one-million-step result

The five-method aggregation uses the corrected three-step PPM runs. Across all
1,200 matched checkpoint-evaluation rows, PPM and EGM have zero identical return
values; the earlier two-step duplicate is not used.

| Environment | Robust noG | Robust QP+G | Paired gain | Bootstrap 95% CI | Wins | Normalized robust-AUC gain | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| HalfCheetah-v4 | 4475.40 | 4629.09 | +153.69 | [-130.43, 440.90] | 3/5 | +51.58 | positive persistent endpoint, unresolved |
| Hopper-v4 | 993.26 | 993.33 | +0.07 | [-16.41, 16.55] | 3/5 | -15.68 | not positive / tie |
| Walker2d-v4 | 1643.71 | 713.86 | -929.85 | [-2175.87, -55.21] | 1/5 | -382.64 | negative |

HalfCheetah is the VI-C main figure because the QP+G-minus-noG smoothed robust
gain is positive throughout the late window and clean return is not sacrificed.
The uncertainty interval crosses zero and only two of five robust AUC gains are
positive, so the result is a mechanism ablation, not statistically resolved
uniform dominance. Corrected PPM has the highest HalfCheetah endpoint
(4851.75 robust, 4867.19 clean), so QP+G is not the overall baseline winner.
Hopper and Walker2d remain in the paper's cross-environment table as required
negative/generalization controls.

HalfCheetah QP diagnostics averaged over five seeds show nonzero curvature
coefficients on 25.6% of protagonist and 37.6% of adversary minibatch updates,
with mean curvature update-norm contributions of 44.3% and 26.8%. There are no
crashes, NaNs, update-cap events, or zero-update failures. Activation is not
treated as proof of benefit because it also occurs in Hopper and Walker2d.

## HPC execution

- Full suite Slurm job: `1627995`.
- Corrected three-step PPM Slurm job: `1628232` (completed, all 15 tasks exit 0).
- Three-layer diagnostic Slurm job: `1627964` (completed, all three tasks exit 0).
- Remote full-suite root:
  `/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719`.
- Remote diagnostic root:
  `/project/vincentlau/jzhuangag/rarl_vic_full_20260719/diagnostics`.
- Remote corrected aggregate:
  `/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/aggregate_corrected`.
- Local corrected aggregate:
  `C:\Users\jzhuangag\work\rarl\results\phaseVIC_full_suite_1m\aggregate_corrected`.

## Supplementary synchronized-update experiment

The alternating 5:1 experiment above remains the main RARL experiment. A new
supplementary experiment tests whether the optimizer conclusions survive a
recursion that is closer to the simultaneous saddle-point model. It must not be
called an exact joint-gradient PPO experiment: PPO uses separate on-policy
rollouts and detached advantage targets, so there is no differentiable shared
protagonist--adversary trajectory objective from which an exact joint vector
field can be obtained.

The implemented `synchronized_lagged` schedule uses `N_mu=N_nu=1`. At the start
of round `t`, the protagonist policy is snapshotted. The protagonist rollout and
PPO update use the round-start adversary, while the adversary rollout and PPO
update use the snapshotted round-start protagonist. The computations occur
sequentially, but neither player's data in round `t` depends on the other
player's newly updated parameters. This is therefore a lagged pseudo-simultaneous
recursion, not ordinary Gauss--Seidel alternation and not exact joint autograd.

Three schedules are pre-specified for the gate:

- alternating 5:1 with no warm-up (the existing protocol);
- synchronized-lagged 1:1 with no warm-up; and
- synchronized-lagged 1:1 with 25 clean protagonist rounds (51,200 protagonist
  steps), during which the adversary force is exactly zero and the adversary is
  not updated.

The gate uses HalfCheetah-v4, noG and QP+G, three paired seeds, and approximately
204,800 protagonist steps per run. The synchronized runs collect more adversary
steps than the 5:1 alternating comparator; cross-schedule return differences
are therefore diagnostic and cannot be interpreted as a sample-efficiency
ranking. Within each schedule, noG and QP+G have identical budgets. A schedule
that is finite, learns visibly, and has a persistent paired QP+G signal will be
expanded to one million protagonist steps, five paired seeds, and the complete
GDA/EGM/PPM/noG/QP+G comparison.

The supplementary report separates training and evaluation axes:

- collection convergence: episodic native task return from rollout collection,
  against actual total collection steps from both player phases;
- clean convergence: mean episodic native task return over frozen deterministic
  evaluation episodes with adversarial force disabled, against protagonist
  training steps;
- learned-adversary convergence: the same frozen evaluation, with each
  protagonist paired with its own concurrently trained frozen adversary;
- common-attacker cross-play: frozen protagonists evaluated against a fixed bank
  of final adversaries, so the compared protagonists face identical attackers;
- dynamics generalization: frozen clean protagonists under pre-specified torso/body
  mass and contact-friction multiplier sweeps, plotted as return-versus-parameter
  curves rather than endpoint bar charts.

An audit of the first gate (`1628567`) found that the inner adversarial wrapper
passed raw observations to the frozen opponent policy although the active policy
was trained through `VecNormalize(norm_obs=True)`. Those results and the aborted
first full run are excluded. The corrected wrapper applies the same frozen
`obs_rms` transformation before every frozen-opponent prediction in training,
own-adversary evaluation, and common-bank evaluation. A numerical unit check
verifies the transformation.

The corrected gate is Slurm job `1629087` (18/18 exit zero). Its paired robust
last-window results are:

| Schedule | Mean QP+G-noG gain | Seed wins |
|---|---:|---:|
| Alternating 5:1, no warm-up | +115.70 | 2/3 |
| Synchronized-lagged 1:1, no warm-up | +64.46 | 3/3 |
| Synchronized-lagged 1:1, 50k clean warm-up | -7.37 | 1/3 |

Accordingly, the pre-specified expansion candidate is synchronized-lagged 1:1
without warm-up. The final one-million-step full-suite job is `1629251`, and the
dependent standard RARL evaluation job is `1629276`. Earlier full jobs were
cancelled before use because they either predated the observation fix or did not
save checkpoint-specific normalization states.

The final main figure uses four distinct channels:

1. protagonist training return from the `PRO` learner's TensorBoard
   `rollout/ep_rew_mean`, against protagonist environment steps; the mixed shared
   Monitor stream is excluded from the main figure;
2. frozen co-trained-pair learned-adversary evaluation;
3. frozen clean evaluation; and
4. frozen common-adversary-bank evaluation at every protagonist checkpoint.

Every protagonist checkpoint is saved as `pro_model_<steps>_steps.zip` together
with `pro_model_vecnormalize_<steps>_steps.pkl`. Adversary checkpoints use an
independent `adv_model` prefix. In cross-play, each protagonist uses its own
checkpoint observation statistics and each fixed attacker uses the statistics
from the attacker's source run. This prevents either policy from receiving
observations in another run's coordinate system.

No paper result will be updated from either active job until all array tasks exit
successfully.

## 2026-07-20 superseding exact-composite amendment

This section supersedes the earlier recommendation to use the PPO `+153.69`
endpoint as the final VI-C result.  A subsequent corrected one-million-step,
five-paired-seed replication has mean robust endpoint gain `-19.42`, only `1/5`
endpoint wins, and bootstrap 95% interval `[-169.58, 184.36]`.  The positive
short-run PPO schedule gates and the earlier `+153.69` run remain historical
diagnostics, not evidence for the final main-paper claim.

The exact actor-field experiment is now required to implement the paper's
composite merit rather than the field-energy-only special case:

`V = lambda_F E_F/kappa_F + lambda_P P_tau/kappa_P`,

where `P_tau` is a smooth, deterministic finite-inner-step proximal
protagonist-plus-adversary saddle-gap residual and the normalizers are fixed at
the initial point.  The critic remains outside the saddle variable but is now a
pair of independently minibatched smooth critics.  The actor objective uses a
smooth conservative soft-min, and use of `G=J_F F` is gated by both Monte-Carlo
critic correlation and twin-critic disagreement.

The two-dimensional box QP explicitly contains the independently solved noG
edge `(beta_noG, 0)`.  Every update records and asserts
`q_QP <= q_noG`; a 10,000-case random positive-definite numerical test passes
10,000/10,000.  On the same frozen replay batch, a safeguard backtracks the QP
candidate and selects the noG edge whenever its realized composite merit is
lower.  This local frozen-batch property does not by itself guarantee a larger
future episodic return because the two training trajectories induce different
replay distributions.

Training and evaluation continue to use the unmodified, undiscounted native
Gymnasium MuJoCo task return.  The adversary is a state-dependent radial policy
with `||f_t||_2 <= Fmax`; its output can choose any magnitude from zero to the
budget.  Each paired optimizer run shares the same clean-pretrained protagonist
checkpoint, and the force budget is increased by a fixed curriculum from 10%
to 100% of `Fmax` over the first 10k joint environment steps.

Force/environment selection is blind to QP+G return.  A noG trajectory records
counterfactual two-dimensional-QP reduced-gradient gain and must satisfy all of:

- final clean return at least 1000 and at least 70% of pretrained clean return;
- `corr_Q_MC >= 0.60` at at least 60% of checkpoints and curvature reliability
  enabled at at least 60% of updates;
- 100% predicted noG-inclusion and realized frozen-batch safeguard checks, with
  zero-update fraction at most 30%; and
- counterfactual nonzero-G and positive reduced-gradient-gain fractions each at
  least 10%, with positive mean normalized predicted gain.

For HalfCheetah-v4, Slurm job `1630205` evaluates radial budgets
`0.5/1.0/2.5 N` under this blind gate.  It is incomplete at the time of this
amendment and must not be used to update the paper.  Independently, clean
pretraining job `1630441` and 50-episode checkpoint evaluation job `1630456`
screen Hopper-v4, Walker2d-v4, and Ant-v4 without running QP+G.  Only Ant-v4's
final checkpoint passes the clean locomotion threshold (`1331.9` over 50
episodes).  Its force calibration job `1630476` locks `Fmax=0.5 N` using the
pre-registered smallest-budget rule: worst fixed-direction degradation is
19.84% while mean episode length retains 99.7% of clean.  Ant's noG geometry
gate is Slurm job `1630491` and is also incomplete at this amendment.

No environment advances to a paired noG/QP+G performance gate unless all
learning, critic, numerical, and reduced-gradient gates pass.  No paired gate
advances to the five-method, five-seed suite unless native clean, common-bank
robust AUC, late-window behavior, and paired endpoint criteria are all reported.

### Exact-composite candidate selected by the blind gate

The full-parameter MLP-critic gates did not pass the reduced-gradient rule in
HalfCheetah-v4 at `0.5/1.0/2.5 N` or in Ant-v4 at calibrated `0.5/1.0 N`.
Freezing the actor trunks and optimizing only the two policy heads increased G
activation only to 2.25%, still below the 10% threshold.  These failed gates are
retained and are not paired-performance experiments.

A state-conditioned game critic was then screened without running QP+G.  It
models the native-return action dependence as state value, protagonist/adversary
linear terms, an explicit bilinear `u--w` interaction, and optional diagonal
own-action quadratic terms.  At own-quadratic scales `0, 0.1, 1.0`, only scale
zero passed all pre-registered gates: final clean/robust return `1224.4/1369.9`,
critic-correlation gate at 80% of checkpoints, mean `WF/SF=1.68`, skew dominance
at 80% of checkpoints, and counterfactual nonzero-G/positive-gain fractions of
54.5%.  The critic is an inductive bias, not a reward modification; critic
correlation is mandatory so high rotation from a poor value model cannot pass.

The eligible configuration is therefore Ant-v4, native Gymnasium reward,
state-dependent radial force with `Fmax=0.5 N`, full `(64,64)` deterministic
actors, independently minibatched twin bilinear game critics with zero explicit
own-action quadratic term, smooth conservative critic soft-min, and the complete
normalized field-energy plus proximal-gap composite merit.

Its 100k-step single-joint-seed paired pilot (`1630608`, 2/2 exit zero)
initially passed the endpoint/AUC and mechanism checks:

- own-pair robust endpoint QP+G-minus-noG `+35.35`;
- robust AUC gain `+4.507e6` and 60% positive late checkpoints;
- clean endpoint gain `+151.26`;
- gamma activation 24.9%, critic-reliable updates 70%, and 100% predicted
  noG-inclusion and realized frozen-batch noG safeguard checks; and
- frozen two-adversary common-bank mean/worst gains `+115.90/+98.35` over 50
  episodes per pairing.

Visual and numerical re-audit shows that this pilot does **not** establish
convergence.  In particular, QP+G clean and co-trained-pair robust evaluation
drop sharply around 75--90k steps, the paired robust gain changes sign, and only
60% of the final five checkpoints are positive.  Under the corrected gate, each
training, clean-evaluation, and robust-evaluation curve must have a stable final
five-checkpoint plateau (coefficient of variation at most 0.15, minimum at least
0.8 times the median, relative range at most 0.40, and absolute normalized
linear drift at most 0.20), while at least 80% of late paired-gain checkpoints
must be positive.  The pilot is therefore retrospectively classified
`DO_NOT_EXPAND`; its positive endpoint must not be quoted as a positive result.

The already launched extended stability diagnostic is Slurm job `1631111`:
GDA, EGM, three-inner-step PPM, noG, and QP+G,
five joint-training seeds each, 200k joint steps.  All five seeds deliberately
share the same pre-locked 150k-step clean Ant protagonist checkpoint; only the
joint critic initialization, replay/environment sampling, exploration, and
training randomness vary.  Consequently, any final claim concerns robustness
to joint-training randomness conditional on one shared pretrained policy and
must not claim coverage of pretraining-seed variance.

### Final strict audit of the 200k Ant suite

Jobs `1631111` and `1631124` completed 25/25 tasks each with exit code zero.
The five-seed result is not a valid positive VI-C result.  QP+G has a large
mean own-pair robust endpoint gain over noG (`+2769.96`, four of five seeds),
a positive robust AUC mean, and 84% positive late paired checkpoints.  The same
qualitative gain appears against the frozen bank of five final noG adversaries
(`+2751.71`, four of five seeds).  However, the gain is primarily caused by
baseline collapse: noG loses locomotion in four of five seeds, GDA and QP+G in
two of five, and EGM and PPM in one of five.  Multiple training, clean, own-pair
robust, and common-bank curves fail the preregistered final-plateau test.
Curvature reliability is only 36%, below the 60% mechanism threshold.  The
strict paired, mechanism, and common-bank decisions are therefore all
`NOT_POSITIVE`; these curves must not be used in the paper as evidence of
convergence or superior robust performance.

A separate true single-agent clean check (`1631558`, 15/15 exit zero) removes
the adversary parameter from the optimization vector and uses identically zero
force in collection, critic input, and evaluation.  Frozen clean evaluation is
stable for GDA, noG, and QP+G, while noG and QP+G also pass the noisy training
return plateau.  QP inclusion and the realized noG safeguard pass 100% of
updates.  QP+G curvature activates in only 0.43% of updates and its final mean
clean return is 68.02 below noG; this is a sanity check, not a positive result.
It indicates that the principal instability is introduced by the joint
bilinear-critic RARL dynamics rather than by ordinary clean locomotion alone.

### Return to standard alternating PPO-RARL baselines

Because neither the exact-joint suite nor its baseline trajectories exhibit
paper-quality convergence, the next gate returns to the conventional RARL
training protocol before any further QP+G comparison.  Slurm job `1631879`
uses HalfCheetah-v4 with unmodified Gymnasium reward, PPO for both protagonist
and adversary, `N_mu:N_nu=5:1` alternating phases, ten outer iterations of
zero-force protagonist warm-up, and a state-dependent two-dimensional torso
force with explicitly resolved scale `0.5`.  SGD/GDA, EGM, and three-step PPM
use the same learning rate `2e-5`, max-gradient norm `0.8`, and value coefficient
`0.58096` for both players.  Training return includes only episodes wholly
contained in protagonist collection phases.  Frozen clean and co-trained-policy
adversarial evaluation are recorded every 25k protagonist environment steps.

QP+G, noG, common-adversary banks, and parameter sweeps remain excluded until
all three ordinary baselines rise materially from initialization, avoid late
collapse, and form stable late clean and adversarial-evaluation plateaus.  The
two-iteration smoke test `1631867` passed the schedule, phase-isolated logging,
optimizer/LR, explicit force-scale, and zero-force warm-up invariants.

### Paper-exact composite Lyapunov requirement for the next proposed run

The PPO-surrogate `proposed_*_perfLyap` optimizer is not the paper-exact
composite method. It combines a gradient-norm term with a PPO minibatch loss,
whereas the paper requires

`V = lambda_F E_F/kappa_F + lambda_P P_tau/kappa_P + lambda_C C/kappa_C`,

with weighted joint-field energy `E_F`, the protagonist-plus-adversary smooth
finite-inner-step proximal saddle gap `P_tau`, and an optional nonnegative
critic residual `C`. The normalizers are fixed at the common initial point;
for a warm-started actor-only experiment this point is the shared post-warm-up
checkpoint and `lambda_C=0`.

The standard detached PPO runner does not expose a differentiable joint
objective `J(theta, psi)` and therefore cannot implement `P_tau` exactly. The
next paper-eligible implementation must fit a smooth joint-action critic
`Q_omega(s,u,w)` on RARL transitions, freeze the same diagnostic batch and
critic state while constructing each update, unroll a fixed differentiable
number of proximal actor-improvement steps for each player, apply the smooth
softplus envelope, and evaluate `E_F` and `P_tau` with normalizers shared by the
paired noG and QP+G runs. Critic fitting remains a separate auxiliary update
for the actor-only experiment.

The two-dimensional QP must explicitly include the independently optimized
`gamma=0` noG edge and assert predicted QP objective no larger than that edge.
The realized frozen-batch safeguard must select noG whenever it gives lower
paper-exact composite merit. Required diagnostics include both raw gaps,
`E_F`, `P_tau`, all fixed normalizers and weights, predicted and realized
composite changes, gamma activation, G contribution, fallback, and numerical
finite/gradient checks. Until these checks pass, the formal method names
`proposed_noG` and `proposed_qp` are blocked in the PPO runner; the older
surrogate implementation is accessible only under explicit `surrogate_*`
diagnostic names and is ineligible for the paper.

### Locked search order after the exact-composite audit

The primary performance protocol is conventional `5:1` alternating PPO-RARL.
This is the protocol used to establish ordinary learning curves and standard
RARL evaluation. A synchronized or exact-joint actor recursion is retained only
as a supplemental saddle-field and rotation diagnostic; it cannot replace a
stable standard-RARL result.

Environment, common actor/critic learning rates, clean protagonist warm-up, and
the maximum adversarial force may be selected only in baseline-only runs. A
setting is eligible when standard Adam-PPO learns normally, reaches a stable
late clean plateau, and retains locomotion under a nontrivial learned attack.
Force is state dependent and policy selected within the common radial/component
bound; native Gymnasium reward is unchanged. No QP+G outcome may be inspected
while choosing these settings.

The current baseline checks are HalfCheetah-v4 Adam-PPO job `1632805` and
InvertedPendulum-v4 force-calibration job `1632813`. The latter explicitly
applies the MuJoCo external force to `pole`; job `1632804` used the invalid
HalfCheetah body name `torso`, failed before training, and produced no result.
Job `1632811` was a one-iteration body-mapping smoke: it completed the PPO
phases but intentionally did not reach an evaluation checkpoint.

After one setting is locked, paired noG and QP+G runs must share the warm-start
checkpoint, seeds, frozen diagnostic batches, critic state at each comparison,
and fixed composite normalizers. Expansion to GDA, EGM, three-step PPM, noG,
and QP+G requires stable training, clean, co-trained-adversary, and common-bank
curves plus the critic, rotation, composite-drift, QP-inclusion, and realized
noG-safeguard gates. Positive endpoints from collapsing baselines, selected
seeds/checkpoints, modified rewards, or the old field-energy/PPO-surrogate
implementation remain ineligible.
