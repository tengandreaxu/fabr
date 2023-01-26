#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --time=60:0:0
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=120G
#SBATCH --partition=gpu
#SBATCH --output=hpc/logs/resnet34_expanded.log

ulimit -c 0
module load gcc/8.4.0-cuda python/3.7.7 cuda/11.0.2
source ~/infinite_random_features/venv/bin/activate
srun python3 benchmarks/resnet_cifar_expanded.py
