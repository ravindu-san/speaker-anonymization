#!/bin/bash -l
#
##SBATCH --gres=gpu:1
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=./slurm-output/exec.sh.o%J

unset SLURM_EXPORT_ENV

module load python/3.12-conda
conda activate speaker-anonymization

#python3 train.py --epochs=4 --batch_size=32 --spectrogram_time_dim=200 --timesteps=300 --save_model

python3 train.py

python3 generator.py