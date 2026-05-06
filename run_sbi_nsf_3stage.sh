#!/bin/bash
#SBATCH --job-name=sbi_nsf_3s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=24:00:00
#SBATCH --output=/home/poehlers/cosmo_thesis/den_pie/output/logs/%x-%j-%N_slurm.out
#SBATCH --error=/home/poehlers/cosmo_thesis/den_pie/output/logs/R-%x.%j.err
## Activate right env

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load OpenMPI/5.0.7-GCC-14.2.0
module load FFTW/3.3.10-GCC-14.2.0

echo "Modules loaded"

source /home/poehlers/sbivenv/bin/activate

pip list | grep 21cmFAST
pip list | grep FrEIA
pip list | grep sbi

python -c 'import py21cmfast as p21c; print("py21cmfast imported!")'
python -c 'import FrEIA as Ff; print("FrEIA imported!")'
python -c 'import sbi; print("sbi imported!")'

echo "Everything imported"

cd /home/poehlers/cosmo_thesis/den_pie

# Stage 1: Pretrain encoder with MSE
python -m den_pie density-train params/sbi_nsf_train_stage1_encoder.yaml --verbose

# Stage 2: Train flow with frozen encoder
python -m den_pie density-train params/sbi_nsf_train_stage2_flow.yaml --verbose

# Stage 3: Fine-tune encoder + flow end-to-end
python -m den_pie density-train params/sbi_nsf_train_stage3_finetune.yaml --verbose

# Evaluate and plot
python -m den_pie density-plot params/sbi_nsf_plot_3stage.yaml --verbose
