#!/bin/bash
#SBATCH --job-name=vic-pretrain-eval
#SBATCH --partition=amd
#SBATCH --account=vincentlau
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=03:00:00
#SBATCH --output=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/pretrain_eval_%A_%a.out
#SBATCH --error=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/pretrain_eval_%A_%a.err

set -euo pipefail
ROOT=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719
PYTHON=/project/vincentlau/jzhuangag/rarl_vic_20260718/env/bin/python
ENVS=(Hopper-v4 Walker2d-v4 Ant-v4)
ENV_ID=${ENVS[$SLURM_ARRAY_TASK_ID]}
RUN="$ROOT/composite_clean_pretrain/$ENV_ID/seed_0"
OUT="$ROOT/composite_clean_pretrain_eval/$ENV_ID/seed_0"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MUJOCO_GL=egl
rm -rf "$OUT"
mkdir -p "$OUT" "$ROOT/slurm"
"$PYTHON" "$ROOT/torch-RARL/scripts/phaseVIC_pretrain_curve_eval.py" \
  --env "$ENV_ID" --root "$RUN" --output "$OUT" --episodes 50 --seed 0
