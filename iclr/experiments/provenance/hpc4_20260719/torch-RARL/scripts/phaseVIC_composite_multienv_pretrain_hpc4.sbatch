#!/bin/bash
#SBATCH --job-name=vic-comp-pretrain
#SBATCH --partition=amd
#SBATCH --account=vincentlau
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --output=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/composite_pretrain_%A_%a.out
#SBATCH --error=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/composite_pretrain_%A_%a.err

set -euo pipefail
ROOT=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719
PYTHON=/project/vincentlau/jzhuangag/rarl_vic_20260718/env/bin/python
ENVS=(Hopper-v4 Walker2d-v4 Ant-v4)
ENV_ID=${ENVS[$SLURM_ARRAY_TASK_ID]}
OUT="$ROOT/composite_clean_pretrain/$ENV_ID/seed_0"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MUJOCO_GL=egl
rm -rf "$OUT"
mkdir -p "$OUT" "$ROOT/slurm"
"$PYTHON" "$ROOT/torch-RARL/scripts/phaseVIC_exact_joint_actor_game.py" \
  --mode pretrain --env "$ENV_ID" --seed 0 --force-scale 0.5 \
  --warmup-steps 10000 --total-steps 150000 --rollout-steps 250 \
  --critic-updates-per-step 1.0 --actor-update-interval 2 \
  --pretrain-actor-lr 1e-4 --exploration-std 0.1 \
  --eval-interval 10000 --eval-episodes 10 --output "$OUT"
