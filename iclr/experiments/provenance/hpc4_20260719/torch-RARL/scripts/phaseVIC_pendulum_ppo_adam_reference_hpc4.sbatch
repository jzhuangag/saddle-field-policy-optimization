#!/bin/bash
#SBATCH --job-name=vic-pend-adam-ref
#SBATCH --partition=amd
#SBATCH --account=vincentlau
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/vic_pend_adam_ref_%j.out
#SBATCH --error=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/vic_pend_adam_ref_%j.err

set -euo pipefail

ROOT=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719
REPO="$ROOT/torch-RARL"
PYTHON=/project/vincentlau/jzhuangag/rarl_vic_20260718/env/bin/python
OUT="$ROOT/ppo_alt_pendulum_adam_reference"

rm -rf "$OUT"
mkdir -p "$OUT" "$ROOT/slurm"
cd "$REPO"

"$PYTHON" scripts/run_stage7e_short_comparison.py \
  --repo-dir "$REPO" --python-path "$PYTHON" \
  --stage5-root /project/vincentlau/jzhuangag/rarl_vic_20260718/ppo_full_policy_optimizer_tuning \
  --stage7d-root /project/vincentlau/jzhuangag/rarl_vic_20260718/ppo_full_policy_proposed_qp \
  --output-root "$OUT" --env Pendulum-v1 --seed 0 --device cpu \
  --iterations 40 --eval-freq 10240 --n-eval-episodes 20 \
  --N-mu 5 --N-nu 1 --adv-delay 5 --rarl-update-mode alternating --clean-warmup \
  --shared-lr 3e-4 --shared-max-grad-norm 0.5 --shared-vf-coef 0.5 \
  --adv-impact control --adv-fraction 0.2 --methods adam \
  --no-control-proxy-eval
