#!/bin/bash
#SBATCH --job-name=vic-exact-comp-hc
#SBATCH --partition=amd
#SBATCH --account=vincentlau
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/exact_comp_hc_%A_%a.out
#SBATCH --error=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/exact_comp_hc_%A_%a.err

set -euo pipefail
ROOT=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719
REPO="$ROOT/torch-RARL"
PYTHON=/project/vincentlau/jzhuangag/rarl_vic_20260718/env/bin/python
METHODS=(nog qpg)
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}
WARMUP_STEPS=${WARMUP_STEPS:-20000}
TOTAL_STEPS=${TOTAL_STEPS:-100000}
OUT_SUFFIX=${OUT_SUFFIX:-}
PRETRAIN="$ROOT/exact_clean_pretrain_gate/HalfCheetah-v4/seed_0/pretrain_checkpoint_00100000.pt"
OUT="$ROOT/exact_composite_halfcheetah_gate/seed_0/${METHOD}${OUT_SUFFIX}"

rm -rf "$OUT"
mkdir -p "$OUT" "$ROOT/slurm"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

"$PYTHON" "$REPO/scripts/phaseVIC_exact_joint_actor_game.py" \
  --env HalfCheetah-v4 --mode joint --method "$METHOD" --seed 0 \
  --pretrained-checkpoint "$PRETRAIN" --load-pretrained-critic \
  --force-scale 1.0 --force-start-fraction 0.1 --force-ramp-steps 25000 \
  --warmup-steps "$WARMUP_STEPS" --total-steps "$TOTAL_STEPS" --rollout-steps 250 \
  --critic-updates-per-step 1.0 --twin-critic --critic-arch mlp \
  --critic-disagreement-max 0.25 --critic-corr-min 0.60 \
  --actor-scope full --actor-lr 3e-6 --actor-update-interval 100 \
  --exploration-std 0.1 --eval-interval 5000 --eval-episodes 20 \
  --merit composite --lambda-f 0.01 --lambda-p 1.0 \
  --gap-tau 0.03 --gap-inner-steps 2 --gap-inner-lr 3e-6 \
  --gap-radius 0.1 --gap-softplus-eps 1e-3 \
  --qp-radius-mult 2.0 --qp-ridge 1e-6 \
  --qp-backtrack-factor 0.5 --qp-backtrack-steps 6 --nog-safe-realized-merit \
  --output "$OUT"
