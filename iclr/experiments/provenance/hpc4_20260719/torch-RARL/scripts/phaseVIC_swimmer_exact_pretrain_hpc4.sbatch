#!/bin/bash
#SBATCH --job-name=vic-swimmer-pre
#SBATCH --partition=amd
#SBATCH --account=vincentlau
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=04:00:00
#SBATCH --output=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/swimmer_exact_pre_%j.out
#SBATCH --error=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/swimmer_exact_pre_%j.err

set -euo pipefail
ROOT=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719
REPO="$ROOT/torch-RARL"
PYTHON=/project/vincentlau/jzhuangag/rarl_vic_20260718/env/bin/python
OUT="$ROOT/exact_swimmer_clean_pretrain/seed_0"

rm -rf "$OUT"
mkdir -p "$OUT" "$ROOT/slurm"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

"$PYTHON" "$REPO/scripts/phaseVIC_exact_joint_actor_game.py" \
  --env Swimmer-v4 --mode pretrain --seed 0 --force-scale 2.0 --force-body back \
  --warmup-steps 10000 --total-steps 75000 --rollout-steps 250 \
  --critic-updates-per-step 1.0 --twin-critic --critic-arch mlp \
  --pretrain-actor-lr 1e-4 --actor-update-interval 50 \
  --exploration-std 0.1 --eval-interval 5000 --eval-episodes 20 \
  --output "$OUT"
