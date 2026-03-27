#!/bin/bash

#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --job-name=g2e_train
#SBATCH --output=/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/G2E/logs/%x_%j.out
#SBATCH --error=/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/G2E/logs/%x_%j.err

set -euo pipefail

module load Anaconda3/2022.10
module load CUDA/12.5.1

eval "$(conda shell.bash hook)"
conda deactivate || true
conda activate xuyue

# 按要求：激活环境后设置 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/ximutian/miniconda3/envs/xuyue/lib:$LD_LIBRARY_PATH

CODE_DIR=/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/G2E/code/G2E
mkdir -p /cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/MutianXi/G2E/logs
cd "$CODE_DIR"

# 单机多卡分布式训练（mainfsdp.py 需要 torchrun 启动）
NPROC_PER_NODE=${SLURM_GPUS_ON_NODE:-2}

torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC_PER_NODE} mainfsdp.py
