#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=7
#SBATCH --mem=5G
#SBATCH -t 8:00:00
#SBATCH--gres=gpu:1
#SBATCH--array=1-30

# sends mail when process begins, and
# when it ends. Make sure you define your email
# address.
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=zdulberg@princeton.edu

echo "Running Array Job ${SLURM_ARRAY_TASK_ID}"
pip install gym

module load anaconda3

conda activate polygon

#python World_location.py $1
python World.py $1
#python World_stoptrain.py $1
