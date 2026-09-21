#!/bin/bash
#SBATCH --job-name=vic-swim-bridge
#SBATCH --partition=amd
#SBATCH --account=vincentlau
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=01:00:00
#SBATCH --output=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/swimmer_bridge_%A_%a.out
#SBATCH --error=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/swimmer_bridge_%A_%a.err

set -euo pipefail
ROOT=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719
REPO="$ROOT/torch-RARL"
PYTHON=/project/vincentlau/jzhuangag/rarl_vic_20260718/env/bin/python
MODES=(none diagnostic gda)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
OUT="$ROOT/ppo_swimmer_outer_bridge_smoke/$MODE"

rm -rf "$OUT"
mkdir -p "$OUT" "$ROOT/slurm"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

"$PYTHON" "$REPO/scripts/run_stage7e_short_comparison.py" \
  --repo-dir "$REPO" --python-path "$PYTHON" \
  --stage5-root /project/vincentlau/jzhuangag/rarl_vic_20260718/ppo_full_policy_optimizer_tuning \
  --stage7d-root /project/vincentlau/jzhuangag/rarl_vic_20260718/ppo_full_policy_proposed_qp \
  --output-root "$OUT" --env Swimmer-v4 --seed 0 --device cpu --iterations 20 \
  --eval-freq 12800 --n-eval-episodes 5 --N-mu 5 --N-nu 1 --adv-delay 5 \
  --rarl-update-mode alternating --clean-warmup --adv-impact force --no-control-proxy-eval \
  --adv-fraction 2.0 --adv-index-list back \
  --shared-lr 3e-4 --shared-max-grad-norm 0.5 --shared-vf-coef 0.5 \
  --composite-outer-mode "$MODE" --composite-outer-warmup 5 \
  --composite-critic-updates 20 --composite-batch-size 128 \
  --composite-state-batch-size 64 --composite-actor-lr 1e-7 \
  --methods adam
