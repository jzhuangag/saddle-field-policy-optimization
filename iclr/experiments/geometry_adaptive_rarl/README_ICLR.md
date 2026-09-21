# Geometry-adaptive RARL experiments for ICLR 2027

This directory is the active ICLR experiment implementation.
It was initialized from the archived July 2026 HPC4 PPO-RARL code, but it is developed and evaluated under a new protocol.
The archived source remains unchanged under `../provenance/hpc4_20260719`.

The first new component is `models/heldout_safeguard.py`.
It compares the field-plus-curvature and matched field-only candidates on a frozen trajectory batch that was not used to construct the local quadratic model.
Curvature is accepted only when its held-out composite merit is lower by more than a prespecified tolerance.

The next implementation stage will connect this rule to the synchronized minimax PPO update, log both candidate merits, and verify that all optimizers share the same policies, critic, rollouts, training budget, and seeds.
