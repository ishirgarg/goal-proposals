#!/bin/bash
#SBATCH --job-name=mega_low_succ
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --array=0-17

# MEGA with low intrinsic-success ranges [success_lo, success_hi] for the adaptive value cutoff.
# Submit from the repo root:
#   sbatch scripts/rail_slurm_mega_low_success_range_sweep.sh
#
# Sweep: 2 envs x 3 ranges x 3 seeds = 18 runs, all on rail_gpu4_high.
#   ENV_IDX  = SLURM_ARRAY_TASK_ID / 9       (0..1)
#   CFG_IDX  = (SLURM_ARRAY_TASK_ID % 9) / 3 (0..2)
#   SEED_IDX = SLURM_ARRAY_TASK_ID % 3       (0..2)
# Envs: ant_u_maze, ant_ball.
# Ranges:
#   0: [0.1, 0.3]
#   1: [0.3, 0.5]
#   2: [0.1, 0.5]
# 80M env steps per run. All other settings are the repo defaults (512 envs, episode_length 1001,
# MEGA cutoff starting at initial_cutoff=-6 with ceiling max_cutoff=0).

# Local wandb run data goes to BRC scratch (home quota is small).
export WANDB_DIR=/global/scratch/users/ishirgarg/goal-proposals
mkdir -p "$WANDB_DIR"

ENVS=("ant_u_maze" "ant_ball")
SEEDS=(0 1 2)
SUCCESS_LO=(0.1 0.3 0.1)
SUCCESS_HI=(0.3 0.5 0.5)

ENV_IDX=$((SLURM_ARRAY_TASK_ID / 9))
CFG_IDX=$(((SLURM_ARRAY_TASK_ID % 9) / 3))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 3))

ENV=${ENVS[$ENV_IDX]}
SEED=${SEEDS[$SEED_IDX]}
LO=${SUCCESS_LO[$CFG_IDX]}
HI=${SUCCESS_HI[$CFG_IDX]}
EXP_NAME="${ENV}__mega_lo${LO}_hi${HI}__s${SEED}"

echo "TASK=$SLURM_ARRAY_TASK_ID  ENV=$ENV  SEED=$SEED  CFG_IDX=$CFG_IDX  EXP=$EXP_NAME"

python run.py crl mega --success_lo $LO --success_hi $HI \
        --env $ENV \
        --seed $SEED \
        --total_env_steps 80000000 \
        --exp_name $EXP_NAME \
        --wandb_project_name goal-proposals \
        --wandb_group mega_low_success_range_sweep \
        --log_wandb
