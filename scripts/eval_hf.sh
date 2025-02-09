#!/bin/bash
#SBATCH --time=3-0:00
#SBATCH --job-name=sbatch
#SBATCH --output=slurm_out/out_%j.txt
#SBATCH --gres="gpu:a6000:1"
#SBATCH --ntasks=16
#SBATCH --exclude=lime-mint,ink-mia

ROOT_DIR=.
CONFIG_DIR=configs
SRC_DIR=src

#SBATCH --nodelist=dill-sage

#This exits the script if any command fails
set -e

export PYTHONPATH=${ROOT_DIR}
export TOKENIZERS_PARALLELISM=false


### START EDITING HERE ###
mode="eval_tofu_configs"
config_file="${CONFIG_DIR}/${mode}.yaml"
WANDB_PROJECT=decouple

CUDA_LAUNCH_BLOCKING=1 python ${SRC_DIR}/run_eval_hf.py\
    --mode=${mode}\
    --config_file=${config_file}\
