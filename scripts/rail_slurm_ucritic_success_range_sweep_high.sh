#!/bin/bash
#SBATCH --job-name=ucritic_succ_range_high
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --array=0-47

# U-critic goal proposer: U learned by 1-step TD vs by regression onto episode returns ("mc"),
# with non-positive ensemble pessimism (0: rank by mean U; -0.5: optimistic, a bonus for ensemble
# disagreement) and low intrinsic-success ranges [success_lo, success_hi] for the adaptive value
# cutoff. U discount: the 0.99 default; ensemble size: the default 5.
# Launch from the repo root, either:
#   bash scripts/rail_slurm_ucritic_success_range_sweep_high.sh
#       splits the 48 tasks across QoS levels: 0-15 high, 16-23 normal, 24-31 low, 32-47 lowest
#       (tasks 0-15 are seed 0 of every (env, target, pessimism, range), so one seed of each is on high)
#   sbatch scripts/rail_slurm_ucritic_success_range_sweep_high.sh
#       submits all 48 tasks on rail_gpu4_high
#
# Sweep: 2 envs x 2 U targets x 2 pessimisms x 2 ranges x 3 seeds = 48 runs. The seed is the
# outermost index.
#   SEED_IDX   = SLURM_ARRAY_TASK_ID / 16       (0..2)
#   ENV_IDX    = (SLURM_ARRAY_TASK_ID % 16) / 8 (0..1)
#   TARGET_IDX = (SLURM_ARRAY_TASK_ID % 8) / 4  (0..1)
#   PESS_IDX   = (SLURM_ARRAY_TASK_ID % 4) / 2  (0..1)
#   RANGE_IDX  = SLURM_ARRAY_TASK_ID % 2        (0..1)
# Envs: ant_u_maze, ant_ball.
# U targets (--u_target): td, mc.
# Pessimisms (--pessimism): 0, -0.5.
# Success ranges: [0.1, 0.3], [0.1, 0.5].
# 80M env steps; all other settings are the repo defaults (512 envs, episode_length 1001,
# cutoff starting at initial_cutoff=-6 with ceiling max_cutoff=0).

# Run with bash (outside a Slurm job): submit this script four times, one QoS per slice of the
# array. Command-line --qos/--array override the #SBATCH lines above.
if [ -z "$SLURM_JOB_ID" ]; then
  set -e
  sbatch --qos=rail_gpu4_high --array=0-15 "$0"
  sbatch --qos=rail_gpu4_normal --array=16-23 "$0"
  sbatch --qos=rail_gpu4_low --array=24-31 "$0"
  sbatch --qos=rail_gpu4_lowest --array=32-47 "$0"
  exit 0
fi

# Local wandb run data goes to BRC scratch (home quota is small).
export WANDB_DIR=/global/scratch/users/ishirgarg/goal-proposals
mkdir -p "$WANDB_DIR"

ENVS=("ant_u_maze" "ant_ball")
SEEDS=(0 1 2)
TARGETS=(td mc)
PESSIMISMS=(0 -0.5)
SUCCESS_LO=(0.1 0.1)
SUCCESS_HI=(0.3 0.5)

SEED_IDX=$((SLURM_ARRAY_TASK_ID / 16))
ENV_IDX=$(((SLURM_ARRAY_TASK_ID % 16) / 8))
TARGET_IDX=$(((SLURM_ARRAY_TASK_ID % 8) / 4))
PESS_IDX=$(((SLURM_ARRAY_TASK_ID % 4) / 2))
RANGE_IDX=$((SLURM_ARRAY_TASK_ID % 2))

ENV=${ENVS[$ENV_IDX]}
SEED=${SEEDS[$SEED_IDX]}
TARGET=${TARGETS[$TARGET_IDX]}
PESS=${PESSIMISMS[$PESS_IDX]}
LO=${SUCCESS_LO[$RANGE_IDX]}
HI=${SUCCESS_HI[$RANGE_IDX]}
EXP_NAME="${ENV}__ucritic_${TARGET}_p${PESS}_lo${LO}_hi${HI}__s${SEED}"

echo "TASK=$SLURM_ARRAY_TASK_ID  ENV=$ENV  SEED=$SEED  TARGET=$TARGET  PESS=$PESS  RANGE=[$LO, $HI]  EXP=$EXP_NAME"

# "--pessimism=" keeps argparse from reading a negative value as a flag
python run.py crl u-critic --u_target $TARGET --pessimism=$PESS --success_lo $LO --success_hi $HI \
        --env $ENV \
        --seed $SEED \
        --total_env_steps 80000000 \
        --exp_name $EXP_NAME \
        --wandb_project_name goal-proposals \
        --wandb_group ucritic_pessimism_target_sweep \
        --log_wandb
