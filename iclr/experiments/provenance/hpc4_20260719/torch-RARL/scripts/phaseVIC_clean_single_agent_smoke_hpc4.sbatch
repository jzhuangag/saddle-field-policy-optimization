#!/bin/bash
#SBATCH --job-name=vic-clean-smoke
#SBATCH --partition=amd
#SBATCH --account=vincentlau
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=01:00:00
#SBATCH --output=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/clean_smoke_%j.out
#SBATCH --error=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/clean_smoke_%j.err

set -euo pipefail
ROOT=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719
PYTHON=/project/vincentlau/jzhuangag/rarl_vic_20260718/env/bin/python
PRETRAIN="$ROOT/composite_clean_pretrain/Ant-v4/seed_0/pretrain_checkpoint_00150000.pt"
OUT="$ROOT/clean_single_agent_smoke"
rm -rf "$OUT"
mkdir -p "$OUT" "$ROOT/slurm"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MUJOCO_GL=egl
"$PYTHON" "$ROOT/torch-RARL/scripts/phaseVIC_exact_joint_actor_game.py" \
  --mode clean_optimizer --env Ant-v4 --method qpg --seed 0 \
  --critic-arch mlp --load-pretrained-critic --pretrained-checkpoint "$PRETRAIN" \
  --force-scale 0.0 --warmup-steps 500 --total-steps 1000 --rollout-steps 250 \
  --critic-updates-per-step 0.25 --actor-update-interval 50 --actor-lr 3e-5 \
  --exploration-std 0.1 --eval-interval 500 --eval-episodes 2 \
  --geometry-probes 1 --merit composite --output "$OUT"
