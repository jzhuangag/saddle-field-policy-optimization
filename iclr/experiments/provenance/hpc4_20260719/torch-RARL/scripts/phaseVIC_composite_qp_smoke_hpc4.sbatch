#!/bin/bash
#SBATCH --job-name=vic-comp-smoke
#SBATCH --partition=amd
#SBATCH --account=vincentlau
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/composite_smoke_%A_%a.out
#SBATCH --error=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/composite_smoke_%A_%a.err

set -euo pipefail
ROOT=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719
PYTHON=/project/vincentlau/jzhuangag/rarl_vic_20260718/env/bin/python
METHODS=(nog qpg)
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}
PRETRAIN="$ROOT/exact_clean_pretrain_gate/HalfCheetah-v4/seed_0/pretrain_checkpoint_00100000.pt"
OUT="$ROOT/composite_qp_smoke/$METHOD"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MUJOCO_GL=egl
rm -rf "$OUT"
mkdir -p "$OUT" "$ROOT/slurm"
"$PYTHON" "$ROOT/torch-RARL/scripts/phaseVIC_exact_joint_actor_game.py" \
  --mode joint --env HalfCheetah-v4 --method "$METHOD" --seed 0 \
  --pretrained-checkpoint "$PRETRAIN" --force-scale 0.5 \
  --warmup-steps 300 --total-steps 500 --rollout-steps 250 \
  --critic-updates-per-step 0.05 --actor-update-interval 250 --actor-lr 3e-5 \
  --exploration-std 0.1 --eval-interval 500 --eval-episodes 1 \
  --geometry-probes 2 --merit composite --force-ramp-steps 500 \
  --output "$OUT"
