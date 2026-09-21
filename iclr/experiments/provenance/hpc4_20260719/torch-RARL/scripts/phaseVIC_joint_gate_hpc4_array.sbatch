#!/bin/bash
#SBATCH --job-name=vic-nfgate
#SBATCH --partition=amd
#SBATCH --account=vincentlau
#SBATCH --array=0-17%18
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/nfgate_%A_%a.out
#SBATCH --error=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719/slurm/nfgate_%A_%a.err

set -euo pipefail

ROOT=/project/vincentlau/jzhuangag/rarl_vic_full_1m_20260719
REPO="$ROOT/torch-RARL"
PYTHON=/project/vincentlau/jzhuangag/rarl_vic_20260718/env/bin/python
STAGE5=/project/vincentlau/jzhuangag/rarl_vic_20260718/ppo_full_policy_optimizer_tuning
STAGE7D=/project/vincentlau/jzhuangag/rarl_vic_20260718/ppo_full_policy_proposed_qp
METHODS=(proposed_noG proposed_qp)
SCHEDULES=(alternating synchronized synchronized_warmup)

TASK=${SLURM_ARRAY_TASK_ID}
SEED=$((TASK % 3))
METHOD_INDEX=$(((TASK / 3) % 2))
SCHEDULE_INDEX=$((TASK / 6))
METHOD=${METHODS[$METHOD_INDEX]}
SCHEDULE=${SCHEDULES[$SCHEDULE_INDEX]}

MODE=alternating
N_MU=5
N_NU=1
ITERATIONS=20
ADV_DELAY=-1
WARMUP=()
if [[ "$SCHEDULE" == "synchronized" ]]; then
  MODE=synchronized_lagged
  N_MU=1
  N_NU=1
  ITERATIONS=100
elif [[ "$SCHEDULE" == "synchronized_warmup" ]]; then
  MODE=synchronized_lagged
  N_MU=1
  N_NU=1
  ITERATIONS=100
  ADV_DELAY=25
  WARMUP=(--clean-warmup)
fi

# All schedules collect approximately 205k protagonist steps. Alternating uses
# 20 rounds x 5 PPO rollouts, while synchronized schedules use 100 x 1.
OUT="$ROOT/joint_gate_normfixed/$SCHEDULE/seed_$SEED/$METHOD"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export XDG_CACHE_HOME="$ROOT/xdg-cache"
export MPLCONFIGDIR="$ROOT/xdg-cache/matplotlib"
export MUJOCO_GL=egl

mkdir -p "$OUT" "$ROOT/slurm" "$ROOT/xdg-cache/matplotlib"
echo "task=$TASK schedule=$SCHEDULE seed=$SEED method=$METHOD mode=$MODE n_mu=$N_MU n_nu=$N_NU iterations=$ITERATIONS"

"$PYTHON" "$REPO/scripts/run_stage7e_short_comparison.py" \
  --repo-dir "$REPO" \
  --python-path "$PYTHON" \
  --stage5-root "$STAGE5" \
  --stage7d-root "$STAGE7D" \
  --output-root "$OUT" \
  --env HalfCheetah-v4 \
  --seed "$SEED" \
  --device cpu \
  --iterations "$ITERATIONS" \
  --eval-freq 25000 \
  --n-eval-episodes 10 \
  --N-mu "$N_MU" \
  --N-nu "$N_NU" \
  --adv-delay "$ADV_DELAY" \
  --rarl-update-mode "$MODE" \
  --methods "$METHOD" \
  "${WARMUP[@]}"
