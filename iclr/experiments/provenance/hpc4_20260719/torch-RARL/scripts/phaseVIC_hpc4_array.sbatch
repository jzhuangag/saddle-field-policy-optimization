#!/bin/bash
#SBATCH --job-name=vic-hc
#SBATCH --account=vincentlau
#SBATCH --partition=amd
#SBATCH --array=0-4%5
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/project/vincentlau/jzhuangag/rarl_vic_20260718/slurm/vic_%A_%a.out
#SBATCH --error=/project/vincentlau/jzhuangag/rarl_vic_20260718/slurm/vic_%A_%a.err

set -euo pipefail

ROOT=/project/vincentlau/jzhuangag/rarl_vic_20260718
REPO="$ROOT/torch-RARL"
PYTHON="$ROOT/env/bin/python"
OUT="$ROOT/results/seed_${SLURM_ARRAY_TASK_ID}"

export MPLBACKEND=Agg
export MUJOCO_GL=egl
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TORCH_HOME="$ROOT/torch-cache"
export XDG_CACHE_HOME="$ROOT/xdg-cache"

mkdir -p "$ROOT/slurm" "$OUT"
cd "$REPO"

"$PYTHON" scripts/run_stage7e_short_comparison.py \
  --repo-dir "$REPO" \
  --python-path "$PYTHON" \
  --stage5-root "$ROOT/ppo_full_policy_optimizer_tuning" \
  --stage7d-root "$ROOT/ppo_full_policy_proposed_qp" \
  --output-root "$OUT" \
  --env HalfCheetah-v4 \
  --seed "$SLURM_ARRAY_TASK_ID" \
  --device cpu \
  --iterations 5 \
  --eval-freq 2500 \
  --n-eval-episodes 20 \
  --methods egm proposed_noG proposed_qp
