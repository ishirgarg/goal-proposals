#!/bin/bash
#SBATCH --job-name=ucritic_succ_range_high
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --array=0-23

# U-critic goal proposer without the adaptive value cutoff vs with different
# intrinsic-success ranges [success_lo, success_hi] for the cutoff. Launch from the repo root, either:
#   bash scripts/rail_slurm_ucritic_success_range_sweep_high.sh
#       splits the 24 tasks across QoS levels: 0-15 high, 16-23 normal
#   sbatch scripts/rail_slurm_ucritic_success_range_sweep_high.sh
#       submits all 24 tasks on rail_gpu4_high
#
# Sweep: 2 envs x 4 configs x 3 seeds = 24 runs.
#   ENV_IDX  = SLURM_ARRAY_TASK_ID / 12      (0..1)
#   CFG_IDX  = (SLURM_ARRAY_TASK_ID % 12) / 3 (0..3)
#   SEED_IDX = SLURM_ARRAY_TASK_ID % 3       (0..2)
# Envs: ant_u_maze, ant_ball.
# Configs:
#   0: U-critic, no cutoff (initial_cutoff=-inf)
#   1: U-critic, success range [0.1, 0.3]
#   2: U-critic, success range [0.1, 0.5]
#   3: U-critic, success range [0.3, 0.5]
# 80M env steps; all other settings are the repo defaults (512 envs, episode_length 1001,
# cutoff starting at initial_cutoff=-6 with ceiling max_cutoff=0 when enabled).

# Run with bash (outside a Slurm job): submit this script twice, one QoS per slice of the
# array. Command-line --qos/--array override the #SBATCH lines above.
if [ -z "$SLURM_JOB_ID" ]; then
  set -e
  sbatch --qos=rail_gpu4_high --array=0-15 "$0"
  sbatch --qos=rail_gpu4_normal --array=16-23 "$0"
  exit 0
fi

# Local wandb run data goes to BRC scratch (home quota is small).
export WANDB_DIR=/global/scratch/users/ishirgarg/goal-proposals
mkdir -p "$WANDB_DIR"

ENVS=("ant_u_maze" "ant_ball")
SEEDS=(0 1 2)
SUCCESS_LO=("" 0.1 0.1 0.3)
SUCCESS_HI=("" 0.3 0.5 0.5)

ENV_IDX=$((SLURM_ARRAY_TASK_ID / 12))
CFG_IDX=$(((SLURM_ARRAY_TASK_ID % 12) / 3))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 3))

ENV=${ENVS[$ENV_IDX]}
SEED=${SEEDS[$SEED_IDX]}

if [ "$CFG_IDX" -eq 0 ]; then
  # the "=" is required: argparse would read a separate "-inf" as a flag
  PROPOSER_ARGS="u-critic --initial_cutoff=-inf"
  EXP_NAME="${ENV}__ucritic_nocutoff__s${SEED}"
else
  LO=${SUCCESS_LO[$CFG_IDX]}
  HI=${SUCCESS_HI[$CFG_IDX]}
  PROPOSER_ARGS="u-critic --success_lo $LO --success_hi $HI"
  EXP_NAME="${ENV}__ucritic_lo${LO}_hi${HI}__s${SEED}"
fi

echo "TASK=$SLURM_ARRAY_TASK_ID  ENV=$ENV  SEED=$SEED  CFG_IDX=$CFG_IDX  EXP=$EXP_NAME"

python run.py crl $PROPOSER_ARGS \
        --env $ENV \
        --seed $SEED \
        --total_env_steps 80000000 \
        --exp_name $EXP_NAME \
        --wandb_project_name goal-proposals \
        --wandb_group ucritic_success_range_sweep \
        --log_wandb
