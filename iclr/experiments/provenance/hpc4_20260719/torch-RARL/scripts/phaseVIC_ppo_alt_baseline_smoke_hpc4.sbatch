#!/bin/bash
#SBATCH --job-name=vic-ppo-smoke
#SBATCH --partition=amd
#SBATCH --account=vincentlau
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/ppo_smoke_%j.out
#SBATCH --error=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/ppo_smoke_%j.err

set -euo pipefail
ROOT=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719
REPO="$ROOT/torch-RARL"
PYTHON=/project/vincentlau/jzhuangag/rarl_vic_20260718/env/bin/python
OUT="$ROOT/ppo_alt_baseline_smoke"
rm -rf "$OUT"
mkdir -p "$OUT" "$ROOT/slurm"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MUJOCO_GL=egl
"$PYTHON" "$REPO/scripts/run_stage7e_short_comparison.py" \
  --repo-dir "$REPO" --python-path "$PYTHON" \
  --stage5-root /project/vincentlau/jzhuangag/rarl_vic_20260718/ppo_full_policy_optimizer_tuning \
  --stage7d-root /project/vincentlau/jzhuangag/rarl_vic_20260718/ppo_full_policy_proposed_qp \
  --output-root "$OUT" --env HalfCheetah-v4 --seed 0 --device cpu --iterations 2 \
  --eval-freq 2048 --n-eval-episodes 2 --N-mu 5 --N-nu 1 --adv-delay 1 \
  --rarl-update-mode alternating --clean-warmup --adv-fraction 0.5 \
  --shared-lr 2e-5 --shared-max-grad-norm 0.8 --shared-vf-coef 0.58096 --methods sgd
