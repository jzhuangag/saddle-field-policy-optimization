#!/bin/bash
#SBATCH --job-name=vic-swim-bridge-ms
#SBATCH --partition=amd
#SBATCH --account=vincentlau
#SBATCH --array=0-8%9
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=04:00:00
#SBATCH --output=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/swimmer_bridge_ms_%A_%a.out
#SBATCH --error=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/swimmer_bridge_ms_%A_%a.err

set -euo pipefail
ROOT=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719
REPO="$ROOT/torch-RARL"
PYTHON=/project/vincentlau/jzhuangag/rarl_vic_20260718/env/bin/python
MODES=(none diagnostic gda)
MODE_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
SEED=$((SLURM_ARRAY_TASK_ID % 3))
MODE=${MODES[$MODE_INDEX]}
OUT="$ROOT/ppo_swimmer_outer_bridge_multiseed/$MODE/seed_$SEED"

rm -rf "$OUT"
mkdir -p "$OUT" "$ROOT/slurm"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

"$PYTHON" "$REPO/scripts/run_stage7e_short_comparison.py" \
  --repo-dir "$REPO" --python-path "$PYTHON" \
  --stage5-root /project/vincentlau/jzhuangag/rarl_vic_20260718/ppo_full_policy_optimizer_tuning \
  --stage7d-root /project/vincentlau/jzhuangag/rarl_vic_20260718/ppo_full_policy_proposed_qp \
  --output-root "$OUT" --env Swimmer-v4 --seed "$SEED" --device cpu --iterations 100 \
  --eval-freq 12800 --n-eval-episodes 20 --N-mu 5 --N-nu 1 --adv-delay 10 \
  --rarl-update-mode alternating --clean-warmup --adv-impact force --no-control-proxy-eval \
  --adv-fraction 2.0 --adv-index-list back \
  --shared-lr 3e-4 --shared-max-grad-norm 0.5 --shared-vf-coef 0.5 \
  --composite-outer-mode "$MODE" --composite-outer-warmup 10 \
  --composite-critic-updates 100 --composite-batch-size 256 \
  --composite-state-batch-size 256 --composite-actor-lr 1e-7 \
  --methods adam
