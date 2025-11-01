#!/bin/bash
#SBATCH --job-name=Muscl
#SBATCH --output=Muscl.txt
#SBATCH --partition=astro2_gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=00:10:00

module load astro
module load cuda
module load python/anaconda3/2021.05

srun python All-steps.py