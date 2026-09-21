#!/bin/bash
#SBATCH --job-name=vic-force-cal
#SBATCH --partition=amd
#SBATCH --account=vincentlau
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --output=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/force_cal_%j.out
#SBATCH --error=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/force_cal_%j.err

set -euo pipefail
ROOT=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719
PYTHON=/project/vincentlau/jzhuangag/rarl_vic_20260718/env/bin/python
OUT="$ROOT/exact_force_calibration"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MUJOCO_GL=egl
rm -rf "$OUT"
"$PYTHON" "$ROOT/torch-RARL/scripts/phaseVIC_force_calibration.py" \
  --checkpoint "$ROOT/exact_clean_pretrain_gate/HalfCheetah-v4/seed_0/pretrain_checkpoint_00100000.pt" \
  --env HalfCheetah-v4 --seed 0 --episodes 20 --strengths 0.5 1.0 2.5 --output "$OUT"
