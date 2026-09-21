#!/bin/bash
#SBATCH --job-name=vic-crossplay
#SBATCH --partition=amd
#SBATCH --account=vincentlau
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --output=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/radial_crossplay_%j.out
#SBATCH --error=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/radial_crossplay_%j.err

set -euo pipefail
ROOT=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719
PYTHON=/project/vincentlau/jzhuangag/rarl_vic_20260718/env/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MUJOCO_GL=egl
rm -rf "$ROOT/radial_warm_joint_crossplay"
"$PYTHON" "$ROOT/torch-RARL/scripts/phaseVIC_radial_crossplay_eval.py" \
  --root "$ROOT/radial_warm_joint_gate/HalfCheetah-v4/seed_0" \
  --output "$ROOT/radial_warm_joint_crossplay" --episodes 20 --force-scale 0.5 --seed 0
