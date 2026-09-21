#!/bin/bash
#SBATCH --job-name=vic-ant1-geometry
#SBATCH --partition=amd
#SBATCH --account=vincentlau
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=06:00:00
#SBATCH --output=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/ant1_geometry_%j.out
#SBATCH --error=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/ant1_geometry_%j.err

set -euo pipefail
ROOT=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719
PYTHON=/project/vincentlau/jzhuangag/rarl_vic_20260718/env/bin/python
PRETRAIN="$ROOT/composite_clean_pretrain/Ant-v4/seed_0/pretrain_checkpoint_00150000.pt"
OUT="$ROOT/composite_geometry_gate/Ant-v4/force_1.0/seed_0/nog"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MUJOCO_GL=egl
rm -rf "$OUT"
mkdir -p "$OUT" "$ROOT/slurm"
"$PYTHON" "$ROOT/torch-RARL/scripts/phaseVIC_exact_joint_actor_game.py" \
  --mode joint --env Ant-v4 --method nog --seed 0 \
  --pretrained-checkpoint "$PRETRAIN" --force-scale 1.0 \
  --warmup-steps 5000 --total-steps 20000 --rollout-steps 250 \
  --critic-updates-per-step 1.0 --actor-update-interval 50 --actor-lr 3e-5 \
  --exploration-std 0.1 --eval-interval 5000 --eval-episodes 10 \
  --geometry-probes 4 --merit composite --force-ramp-steps 10000 \
  --critic-disagreement-max 0.25 --critic-corr-min 0.60 --output "$OUT"
