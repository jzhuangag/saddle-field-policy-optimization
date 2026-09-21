#!/bin/bash
#SBATCH --job-name=vic-ppm3
#SBATCH --partition=amd
#SBATCH --account=vincentlau
#SBATCH --array=0-14
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/ppm3_%A_%a.out
#SBATCH --error=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/ppm3_%A_%a.err

set -euo pipefail

ROOT=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719
REPO="$ROOT/torch-RARL"
PYTHON=/project/vincentlau/jzhuangag/rarl_vic_20260718/env/bin/python
STAGE5=/project/vincentlau/jzhuangag/rarl_vic_20260718/ppo_full_policy_optimizer_tuning
STAGE7D=/project/vincentlau/jzhuangag/rarl_vic_20260718/ppo_full_policy_proposed_qp
ENVS=(HalfCheetah-v4 Hopper-v4 Walker2d-v4)

TASK=${SLURM_ARRAY_TASK_ID}
SEED=$((TASK % 5))
ENV_INDEX=$((TASK / 5))
ENV_ID=${ENVS[$ENV_INDEX]}
OUT="$ROOT/results_ppm3/$ENV_ID/seed_$SEED/ppm"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export XDG_CACHE_HOME="$ROOT/xdg-cache"
export MPLCONFIGDIR="$ROOT/xdg-cache/matplotlib"
export MUJOCO_GL=egl

mkdir -p "$OUT" "$ROOT/xdg-cache/matplotlib"
echo "task=$TASK env=$ENV_ID seed=$SEED method=ppm inner_steps=3 output=$OUT"
"$PYTHON" "$REPO/scripts/run_stage7e_short_comparison.py" \
  --repo-dir "$REPO" \
  --python-path "$PYTHON" \
  --stage5-root "$STAGE5" \
  --stage7d-root "$STAGE7D" \
  --output-root "$OUT" \
  --env "$ENV_ID" \
  --seed "$SEED" \
  --device cpu \
  --iterations 100 \
  --eval-freq 25000 \
  --n-eval-episodes 20 \
  --ppm-inner-steps 3 \
  --methods ppm

