#!/bin/bash
#SBATCH --job-name=vic-w20
#SBATCH --partition=amd
#SBATCH --account=vincentlau
#SBATCH --array=0-5%6
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/w20_%A_%a.out
#SBATCH --error=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/w20_%A_%a.err

set -euo pipefail
ROOT=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719
REPO="$ROOT/torch-RARL"
PYTHON=/project/vincentlau/jzhuangag/rarl_vic_20260718/env/bin/python
STAGE5=/project/vincentlau/jzhuangag/rarl_vic_20260718/ppo_full_policy_optimizer_tuning
STAGE7D=/project/vincentlau/jzhuangag/rarl_vic_20260718/ppo_full_policy_proposed_qp
METHODS=(proposed_noG proposed_qp)
TASK=${SLURM_ARRAY_TASK_ID}
SEED=$((TASK % 3))
METHOD=${METHODS[$((TASK / 3))]}
OUT="$ROOT/joint_gate_normfixed/synchronized_warmup20/seed_$SEED/$METHOD"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export XDG_CACHE_HOME="$ROOT/xdg-cache"
export MPLCONFIGDIR="$ROOT/xdg-cache/matplotlib"
export MUJOCO_GL=egl
mkdir -p "$OUT" "$ROOT/slurm"

"$PYTHON" "$REPO/scripts/run_stage7e_short_comparison.py" \
  --repo-dir "$REPO" --python-path "$PYTHON" \
  --stage5-root "$STAGE5" --stage7d-root "$STAGE7D" \
  --output-root "$OUT" --env HalfCheetah-v4 --seed "$SEED" --device cpu \
  --iterations 100 --eval-freq 25000 --n-eval-episodes 10 \
  --N-mu 1 --N-nu 1 --adv-delay 10 --clean-warmup \
  --rarl-update-mode synchronized_lagged --methods "$METHOD"
