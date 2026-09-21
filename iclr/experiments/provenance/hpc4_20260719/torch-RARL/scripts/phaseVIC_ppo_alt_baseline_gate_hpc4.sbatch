#!/bin/bash
#SBATCH --job-name=vic-ppo-alt
#SBATCH --partition=amd
#SBATCH --account=vincentlau
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/ppo_alt_%A_%a.out
#SBATCH --error=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/ppo_alt_%A_%a.err

set -euo pipefail
ROOT=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719
REPO="$ROOT/torch-RARL"
PYTHON=/project/vincentlau/jzhuangag/rarl_vic_20260718/env/bin/python
STAGE5=/project/vincentlau/jzhuangag/rarl_vic_20260718/ppo_full_policy_optimizer_tuning
STAGE7D=/project/vincentlau/jzhuangag/rarl_vic_20260718/ppo_full_policy_proposed_qp
METHODS=(sgd egm ppm)
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}
OUT="$ROOT/ppo_alt_baseline_gate/HalfCheetah-v4/seed_0/$METHOD"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export XDG_CACHE_HOME="$ROOT/xdg-cache"
export MPLCONFIGDIR="$ROOT/xdg-cache/matplotlib"
export MUJOCO_GL=egl
rm -rf "$OUT"
mkdir -p "$OUT" "$ROOT/slurm" "$ROOT/xdg-cache/matplotlib"

"$PYTHON" "$REPO/scripts/run_stage7e_short_comparison.py" \
  --repo-dir "$REPO" --python-path "$PYTHON" \
  --stage5-root "$STAGE5" --stage7d-root "$STAGE7D" --output-root "$OUT" \
  --env HalfCheetah-v4 --seed 0 --device cpu --iterations 100 \
  --eval-freq 25000 --n-eval-episodes 20 \
  --N-mu 5 --N-nu 1 --adv-delay 10 --rarl-update-mode alternating --clean-warmup \
  --adv-fraction 0.5 --shared-lr 2e-5 --shared-max-grad-norm 0.8 --shared-vf-coef 0.58096 \
  --ppm-inner-steps 3 --methods "$METHOD"
