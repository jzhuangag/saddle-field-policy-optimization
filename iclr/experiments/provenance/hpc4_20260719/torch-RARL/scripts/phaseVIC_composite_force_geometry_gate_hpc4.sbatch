#!/bin/bash
#SBATCH --job-name=vic-comp-force
#SBATCH --partition=amd
#SBATCH --account=vincentlau
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/composite_force_%A_%a.out
#SBATCH --error=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/composite_force_%A_%a.err

set -euo pipefail
ROOT=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719
PYTHON=/project/vincentlau/jzhuangag/rarl_vic_20260718/env/bin/python
FORCES=(0.5 1.0 2.5)
FORCE=${FORCES[$SLURM_ARRAY_TASK_ID]}
PRETRAIN="$ROOT/exact_clean_pretrain_gate/HalfCheetah-v4/seed_0/pretrain_checkpoint_00100000.pt"
OUT="$ROOT/composite_force_geometry_gate/HalfCheetah-v4/force_${FORCE}/seed_0/nog"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MUJOCO_GL=egl
rm -rf "$OUT"
mkdir -p "$OUT" "$ROOT/slurm"
"$PYTHON" "$ROOT/torch-RARL/scripts/phaseVIC_exact_joint_actor_game.py" \
  --mode joint --env HalfCheetah-v4 --method nog --seed 0 \
  --pretrained-checkpoint "$PRETRAIN" --force-scale "$FORCE" \
  --warmup-steps 5000 --total-steps 20000 --rollout-steps 250 \
  --critic-updates-per-step 1.0 --actor-update-interval 50 --actor-lr 3e-5 \
  --exploration-std 0.1 --eval-interval 5000 --eval-episodes 5 \
  --geometry-probes 4 --merit composite --force-ramp-steps 10000 \
  --critic-disagreement-max 0.25 --output "$OUT"
