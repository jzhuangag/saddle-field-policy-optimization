#!/bin/bash
#SBATCH --job-name=vic-exact-smoke
#SBATCH --partition=amd
#SBATCH --account=vincentlau
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --output=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/exact_smoke_%j.out
#SBATCH --error=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/exact_smoke_%j.err

set -euo pipefail
ROOT=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719
PYTHON=/project/vincentlau/jzhuangag/rarl_vic_20260718/env/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MUJOCO_GL=egl
"$PYTHON" "$ROOT/torch-RARL/scripts/phaseVIC_exact_joint_actor_game.py" \
  --env HalfCheetah-v4 --method qpg --seed 0 --warmup-steps 300 \
  --total-steps 250 --rollout-steps 250 --critic-updates-per-step 0.05 \
  --actor-update-interval 250 --eval-interval 250 --eval-episodes 1 \
  --geometry-probes 2 --output "$ROOT/exact_joint_smoke"
