#!/bin/bash
#SBATCH --job-name=vic-jeval
#SBATCH --partition=amd
#SBATCH --account=vincentlau
#SBATCH --array=0-24%24
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --output=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/jeval_%A_%a.out
#SBATCH --error=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/jeval_%A_%a.err

set -euo pipefail

ROOT=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719
REPO="$ROOT/torch-RARL"
PYTHON=/project/vincentlau/jzhuangag/rarl_vic_20260718/env/bin/python
SUITE="$ROOT/joint_full_final3/synchronized/HalfCheetah-v4"
METHODS=(sgd egm ppm proposed_noG proposed_qp)

TASK=${SLURM_ARRAY_TASK_ID}
SEED=$((TASK % 5))
METHOD_INDEX=$((TASK / 5))
METHOD=${METHODS[$METHOD_INDEX]}
OUTPUT="$ROOT/joint_full_final3/standard_eval/seed_${SEED}_${METHOD}.csv"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export XDG_CACHE_HOME="$ROOT/xdg-cache"
export MPLCONFIGDIR="$ROOT/xdg-cache/matplotlib"
export MUJOCO_GL=egl

mkdir -p "$ROOT/joint_full_final3/standard_eval" "$ROOT/slurm"
cd "$REPO"
"$PYTHON" scripts/phaseVIC_standard_rarl_eval.py \
  --suite-root "$SUITE" \
  --target-seed "$SEED" \
  --target-method "$METHOD" \
  --output "$OUTPUT" \
  --episodes 20 \
  --device cpu
