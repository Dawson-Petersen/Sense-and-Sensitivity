#!/bin/bash
#SBATCH --job-name="legal_interp_test"
#SBATCH --output="slurm_logs/%x_%j.o"
#SBATCH --gres=gpu:7g.94gb:1
#SBATCH --mincpus=4
#SBATCH --mem=32gb
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=dp1147@georgetown.edu

cd /home/dp1147/llms-legal-interp || return

conda activate /scratch/dp1147/conda-envs/llms-legal-interp

python3 -m pip install -r requirements.txt
python3 --version

python3 main.py
