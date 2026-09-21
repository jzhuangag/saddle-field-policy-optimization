#!/bin/bash
#SBATCH --job-name=vic-radial-joint
#SBATCH --partition=amd
#SBATCH --account=vincentlau
#SBATCH --array=0-4%5
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/radial_joint_%A_%a.out
#SBATCH --error=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/radial_joint_%A_%a.err

set -euo pipefail
ROOT=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719
PYTHON=/project/vincentlau/jzhuangag/rarl_vic_20260718/env/bin/python
METHODS=(gda egm ppm nog qpg)
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}
PRETRAIN="$ROOT/exact_clean_pretrain_gate/HalfCheetah-v4/seed_0/pretrain_checkpoint_00100000.pt"
OUT="$ROOT/radial_warm_joint_gate/HalfCheetah-v4/seed_0/$METHOD"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MUJOCO_GL=egl
rm -rf "$OUT"
mkdir -p "$OUT" "$ROOT/slurm"
"$PYTHON" "$ROOT/torch-RARL/scripts/phaseVIC_exact_joint_actor_game.py" \
  --mode joint --env HalfCheetah-v4 --method "$METHOD" --seed 0 \
  --pretrained-checkpoint "$PRETRAIN" --force-scale 0.5 \
  --warmup-steps 10000 --total-steps 100000 --rollout-steps 250 \
  --critic-updates-per-step 1.0 --actor-update-interval 20 --actor-lr 3e-5 \
  --exploration-std 0.1 --eval-interval 10000 --eval-episodes 20 \
  --geometry-probes 8 --ppm-inner 3 --output "$OUT"
