#!/bin/bash
#SBATCH --job-name=vic-comp-smoke
#SBATCH --partition=amd
#SBATCH --account=vincentlau
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/comp_smoke_%A_%a.out
#SBATCH --error=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/comp_smoke_%A_%a.err

set -euo pipefail
ROOT=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719
REPO="$ROOT/torch-RARL"
PYTHON=/project/vincentlau/jzhuangag/rarl_vic_20260718/env/bin/python
METHODS=(surrogate_noG surrogate_qp)
METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}
OUT="$ROOT/ppo_pendulum_composite_perflyap_smoke/$METHOD"

rm -rf "$OUT"
mkdir -p "$OUT" "$ROOT/slurm"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

"$PYTHON" "$REPO/scripts/run_stage7e_short_comparison.py" \
  --repo-dir "$REPO" --python-path "$PYTHON" \
  --stage5-root /project/vincentlau/jzhuangag/rarl_vic_20260718/ppo_full_policy_optimizer_tuning \
  --stage7d-root /project/vincentlau/jzhuangag/rarl_vic_20260718/ppo_full_policy_proposed_qp \
  --output-root "$OUT" --env Pendulum-v1 --seed 0 --device cpu --iterations 1 \
  --eval-freq 1024 --n-eval-episodes 2 --N-mu 1 --N-nu 1 --adv-delay 0 \
  --rarl-update-mode alternating --adv-impact control --no-control-proxy-eval \
  --adv-fraction 0.2 --shared-lr 3e-4 --shared-max-grad-norm 0.5 --shared-vf-coef 0.5 \
  --perflyap-scope full_policy_actor_weighted \
  --perflyap-lambda-n 0.1 --perflyap-lambda-p 1.0 --perflyap-lambda-critic 0.1 \
  --perflyap-beta-max 1e-2 --perflyap-gamma-max 3e-5 --perflyap-update-cap 3e-3 \
  --perflyap-cost-mode unclipped_actor_surrogate_cost \
  --perflyap-selector-mode current_q_pred --perflyap-direction-mode egm_plus_JF_F \
  --methods "$METHOD"
