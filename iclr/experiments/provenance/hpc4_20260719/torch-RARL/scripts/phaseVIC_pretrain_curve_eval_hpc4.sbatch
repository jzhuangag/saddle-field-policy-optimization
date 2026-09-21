#!/bin/bash
#SBATCH --job-name=vic-precurve
#SBATCH --partition=amd
#SBATCH --account=vincentlau
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --output=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/precurve_%j.out
#SBATCH --error=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/precurve_%j.err

set -euo pipefail
ROOT=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719
PYTHON=/project/vincentlau/jzhuangag/rarl_vic_20260718/env/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MUJOCO_GL=egl
rm -rf "$ROOT/exact_pretrain_curve_eval"
"$PYTHON" "$ROOT/torch-RARL/scripts/phaseVIC_pretrain_curve_eval.py" \
  --root "$ROOT/exact_clean_pretrain_gate/HalfCheetah-v4/seed_0" \
  --output "$ROOT/exact_pretrain_curve_eval" --episodes 20 --seed 0
