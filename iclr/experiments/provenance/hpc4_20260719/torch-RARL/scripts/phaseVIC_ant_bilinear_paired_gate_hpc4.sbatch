#!/bin/bash
#SBATCH --job-name=vic-ant-paired
#SBATCH --partition=amd
#SBATCH --account=vincentlau
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=12:00:00
#SBATCH --output=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/ant_paired_%A_%a.out
#SBATCH --error=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/ant_paired_%A_%a.err

set -euo pipefail
ROOT=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719
PYTHON=/project/vincentlau/jzhuangag/rarl_vic_20260718/env/bin/python
METHODS=(nog qpg)
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}
PRETRAIN="$ROOT/composite_clean_pretrain/Ant-v4/seed_0/pretrain_checkpoint_00150000.pt"
OUT="$ROOT/composite_bilinear_paired_gate/Ant-v4/force_0.5/seed_0/$METHOD"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MUJOCO_GL=egl
rm -rf "$OUT"
mkdir -p "$OUT" "$ROOT/slurm"
"$PYTHON" "$ROOT/torch-RARL/scripts/phaseVIC_exact_joint_actor_game.py" \
  --mode joint --env Ant-v4 --method "$METHOD" --actor-scope full --seed 0 \
  --critic-arch game_bilinear --critic-own-curvature-scale 0.0 \
  --no-load-pretrained-critic --pretrained-checkpoint "$PRETRAIN" --force-scale 0.5 \
  --warmup-steps 5000 --total-steps 100000 --rollout-steps 250 \
  --critic-updates-per-step 1.0 --actor-update-interval 50 --actor-lr 3e-5 \
  --exploration-std 0.1 --eval-interval 5000 --eval-episodes 20 \
  --geometry-probes 8 --merit composite --force-ramp-steps 10000 \
  --critic-disagreement-max 0.25 --critic-corr-min 0.60 --output "$OUT"
