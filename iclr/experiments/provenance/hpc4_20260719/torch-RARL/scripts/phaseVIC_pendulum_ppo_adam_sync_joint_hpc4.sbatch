#!/bin/bash
#SBATCH --job-name=vic-pend-adam-sync
#SBATCH --partition=amd
#SBATCH --account=vincentlau
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/pend_adam_sync_%j.out
#SBATCH --error=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/pend_adam_sync_%j.err

set -euo pipefail
ROOT=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719
REPO="$ROOT/torch-RARL"
PYTHON=/project/vincentlau/jzhuangag/rarl_vic_20260718/env/bin/python
OUT="$ROOT/ppo_pendulum_adam_synchronized_lagged"

rm -rf "$OUT"
mkdir -p "$OUT" "$ROOT/slurm"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

"$PYTHON" "$REPO/scripts/run_stage7e_short_comparison.py" \
  --repo-dir "$REPO" --python-path "$PYTHON" \
  --stage5-root /project/vincentlau/jzhuangag/rarl_vic_20260718/ppo_full_policy_optimizer_tuning \
  --stage7d-root /project/vincentlau/jzhuangag/rarl_vic_20260718/ppo_full_policy_proposed_qp \
  --output-root "$OUT" --env Pendulum-v1 --seed 0 --device cpu --iterations 200 \
  --eval-freq 10240 --n-eval-episodes 20 --N-mu 1 --N-nu 1 --adv-delay 25 \
  --rarl-update-mode synchronized_lagged --clean-warmup \
  --adv-impact control --no-control-proxy-eval --adv-fraction 0.2 \
  --shared-lr 3e-4 --shared-max-grad-norm 0.5 --shared-vf-coef 0.5 \
  --methods adam
