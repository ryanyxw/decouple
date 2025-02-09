#!/bin/bash
#SBATCH --time=3-0:00
#SBATCH --job-name=sbatch
#SBATCH --output=slurm_out/out_%j.txt
#SBATCH --gres="gpu:a6000:1"
#SBATCH --ntasks=16


NEOX_DIR=gpt-neox
CONFIG_DIR=configs
SRC_DIR=src
#SBATCH --nodelist=ink-noah
#SBATCH --exclude=glamor-ruby


#This exits the script if any command fails
set -e

export PYTHONPATH=.

### START EDITING HERE ###
mode="train_tofu_configs"
config_file=${CONFIG_DIR}/${mode}.yaml

WANDB_PROJECT=decouple

CUDA_LAUNCH_BLOCKING=1 python ${SRC_DIR}/run_train_hf.py\
    --mode=${mode}\
    --config_file=${config_file}\
