#!/bin/bash
#SBATCH --job-name=vic-pend-lr1e2
#SBATCH --partition=amd
#SBATCH --account=vincentlau
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/pend_lr1e2_%A_%a.out
#SBATCH --error=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/pend_lr1e2_%A_%a.err

set -euo pipefail
ROOT=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719
REPO="$ROOT/torch-RARL"
PYTHON=/project/vincentlau/jzhuangag/rarl_vic_20260718/env/bin/python
METHODS=(sgd egm ppm)
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}
OUT="$ROOT/ppo_alt_pendulum_lr_gate/lr_1e-2/$METHOD"

rm -rf "$OUT"
mkdir -p "$OUT" "$ROOT/slurm"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

"$PYTHON" "$REPO/scripts/run_stage7e_short_comparison.py" \
  --repo-dir "$REPO" --python-path "$PYTHON" \
  --stage5-root /project/vincentlau/jzhuangag/rarl_vic_20260718/ppo_full_policy_optimizer_tuning \
  --stage7d-root /project/vincentlau/jzhuangag/rarl_vic_20260718/ppo_full_policy_proposed_qp \
  --output-root "$OUT" --env Pendulum-v1 --seed 0 --device cpu --iterations 40 \
  --eval-freq 10240 --n-eval-episodes 20 --N-mu 5 --N-nu 1 --adv-delay 5 \
  --rarl-update-mode alternating --clean-warmup --adv-impact control --no-control-proxy-eval \
  --adv-fraction 0.2 --shared-lr 1e-2 --shared-max-grad-norm 0.5 --shared-vf-coef 0.5 \
  --ppm-inner-steps 3 --methods "$METHOD"
