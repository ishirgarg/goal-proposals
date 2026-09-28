#!/bin/bash
#SBATCH --job-name=mega_succ_range_high
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --array=0-29

# CRL baseline vs MEGA with different intrinsic-success ranges [success_lo, success_hi]
# for the adaptive value cutoff. Launch from the repo root, either:
#   bash scripts/rail_slurm_mega_success_range_sweep_high.sh
#       splits the 30 tasks across QoS levels: 0-15 high, 16-23 normal, 24-29 low
#   sbatch scripts/rail_slurm_mega_success_range_sweep_high.sh
#       submits all 30 tasks on rail_gpu4_high
#
# Sweep: 2 envs x 5 configs x 3 seeds = 30 runs.
#   ENV_IDX  = SLURM_ARRAY_TASK_ID / 15      (0..1)
#   CFG_IDX  = (SLURM_ARRAY_TASK_ID % 15) / 3 (0..4)
#   SEED_IDX = SLURM_ARRAY_TASK_ID % 3       (0..2)
# Configs:
#   0: baseline CRL (env goals)
#   1: MEGA, success range [0.3, 0.7] (default)
#   2: MEGA, success range [0, 0.9]
#   3: MEGA, success range [0.5, 0.9]
#   4: MEGA, success range [0.75, 0.9]
# 30M env steps per run. All other settings are the repo defaults (512 envs, episode_length 1001,
# MEGA cutoff starting at initial_cutoff=-6 with ceiling max_cutoff=0).

# Run with bash (outside a Slurm job): submit this script three times, one QoS per slice of the
# array. Command-line --qos/--array override the #SBATCH lines above.
if [ -z "$SLURM_JOB_ID" ]; then
  set -e
  sbatch --qos=rail_gpu4_high --array=0-15 "$0"
  sbatch --qos=rail_gpu4_normal --array=16-23 "$0"
  sbatch --qos=rail_gpu4_low --array=24-29 "$0"
  exit 0
fi

# Local wandb run data goes to BRC scratch (home quota is small).
export WANDB_DIR=/global/scratch/users/ishirgarg/goal-proposals
mkdir -p "$WANDB_DIR"

ENVS=("ant_u_maze" "ant_ball")
SEEDS=(0 1 2)
SUCCESS_LO=("" 0.3 0 0.5 0.75)
SUCCESS_HI=("" 0.7 0.9 0.9 0.9)

ENV_IDX=$((SLURM_ARRAY_TASK_ID / 15))
CFG_IDX=$(((SLURM_ARRAY_TASK_ID % 15) / 3))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 3))

ENV=${ENVS[$ENV_IDX]}
SEED=${SEEDS[$SEED_IDX]}

if [ "$CFG_IDX" -eq 0 ]; then
  PROPOSER_ARGS="env-goals"
  EXP_NAME="${ENV}__crl__s${SEED}"
else
  LO=${SUCCESS_LO[$CFG_IDX]}
  HI=${SUCCESS_HI[$CFG_IDX]}
  PROPOSER_ARGS="mega --success_lo $LO --success_hi $HI"
  EXP_NAME="${ENV}__mega_lo${LO}_hi${HI}__s${SEED}"
fi

echo "TASK=$SLURM_ARRAY_TASK_ID  ENV=$ENV  SEED=$SEED  CFG_IDX=$CFG_IDX  EXP=$EXP_NAME"

python run.py crl $PROPOSER_ARGS \
        --env $ENV \
        --seed $SEED \
        --total_env_steps 30000000 \
        --exp_name $EXP_NAME \
        --wandb_project_name goal-proposals \
        --wandb_group mega_success_range_sweep \
        --log_wandb
