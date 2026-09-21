#!/bin/bash
#SBATCH --job-name=vic-ant-crossplay
#SBATCH --partition=amd
#SBATCH --account=vincentlau
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=03:00:00
#SBATCH --output=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/ant_crossplay_%j.out
#SBATCH --error=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/ant_crossplay_%j.err

set -euo pipefail
ROOT=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719
PYTHON=/project/vincentlau/jzhuangag/rarl_vic_20260718/env/bin/python
RUN="$ROOT/composite_bilinear_paired_gate/Ant-v4/force_0.5/seed_0"
OUT="$ROOT/composite_bilinear_paired_crossplay/Ant-v4/force_0.5/seed_0"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MUJOCO_GL=egl
rm -rf "$OUT"
mkdir -p "$OUT" "$ROOT/slurm"
"$PYTHON" "$ROOT/torch-RARL/scripts/phaseVIC_exact_crossplay_eval.py" \
  --root "$RUN" --output "$OUT" --env Ant-v4 --methods nog qpg \
  --checkpoint-step 100000 --episodes 50 --force-scale 0.5 --seed 0
