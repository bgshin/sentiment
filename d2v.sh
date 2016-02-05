#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=40000
#SBATCH --nodes=1

# python run.py
# python senti.py
# python sent2.py
python d2v.py
