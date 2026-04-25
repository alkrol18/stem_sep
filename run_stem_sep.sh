#!/bin/bash
#SBATCH --job-name=stem_sep
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/usr/project/xtmp/ak724/stem_sep_logs/slurm_%j.out
#SBATCH --error=/usr/project/xtmp/ak724/stem_sep_logs/slurm_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=ak724@duke.edu

source ~/.bashrc
source /usr/pkg/miniconda-23.9.0/etc/profile.d/conda.sh
conda activate stem_sep

cd /home/users/ak724/stem_sep

mkdir -p /usr/project/xtmp/ak724/stem_sep_logs
mkdir -p /usr/project/xtmp/ak724/stem_sep_checkpoints
mkdir -p /usr/project/xtmp/ak724/stem_sep_outputs

export PYTHONUNBUFFERED=1
python -u -m training.train --config configs/dcc.yaml --mode b
