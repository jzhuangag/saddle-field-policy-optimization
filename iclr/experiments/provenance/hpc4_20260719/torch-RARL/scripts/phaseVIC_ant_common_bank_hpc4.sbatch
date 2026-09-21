#!/bin/bash
#SBATCH --job-name=vic-ant-bank
#SBATCH --partition=amd
#SBATCH --account=vincentlau
#SBATCH --array=0-24%10
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=06:00:00
#SBATCH --output=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/ant_bank_%A_%a.out
#SBATCH --error=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/ant_bank_%A_%a.err

set -euo pipefail
ROOT=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719
PYTHON=/project/vincentlau/jzhuangag/rarl_vic_20260718/env/bin/python
RUN="$ROOT/composite_bilinear_full_suite/Ant-v4/force_0.5"
OUT="$ROOT/composite_bilinear_common_bank/Ant-v4/force_0.5"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MUJOCO_GL=egl
mkdir -p "$OUT" "$ROOT/slurm"
"$PYTHON" "$ROOT/torch-RARL/scripts/phaseVIC_ant_common_bank_worker.py" \
  --root "$RUN" --output "$OUT" --task-id "$SLURM_ARRAY_TASK_ID" \
  --episodes 5 --force-scale 0.5
