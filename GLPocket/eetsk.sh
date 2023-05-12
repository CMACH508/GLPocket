#!/bin/bash
#SBATCH -J trans
#SBATCH -p a100
#SBATCH -o GLPocket_adam.out
#SBATCH -e GLPocket_adam.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:3
#SBATCH -x vol08

module load miniconda3
source activate deeppocket

batch_size=12
output="GLPocket_adam/seg0"
gpu="0,1,2"
epoch=150
base_lr=1e-3
optim='Adam'
python -u train.py --solver $optim --gpu $gpu -b $batch_size -o $output -e $epoch --base_lr $base_lr

