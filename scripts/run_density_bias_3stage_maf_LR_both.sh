#!/bin/bash
#SBATCH --job-name=21cm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=08:00:00
#SBATCH --output=/home/poehlers/cosmo_thesis/den_pie/output/logs/%x-%j-%N_slurm.out
#SBATCH --error=/home/poehlers/cosmo_thesis/den_pie/output/logs/R-%x.%j.err
## Activate right env

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load OpenMPI/5.0.7-GCC-14.2.0
module load FFTW/3.3.10-GCC-14.2.0

echo "Modules loaded"

source /home/poehlers/21cmvenv/bin/activate

pip list | grep 21cmFAST
pip list | grep FrEIA

python -c 'import py21cmfast as p21c; print("py21cmfast imported!")'
python -c 'import FrEIA as Ff; print("FrEIA imported!")'

echo "Everything imported"

cd /home/poehlers/cosmo_thesis/den_pie

python -m den_pie density-train params/density_bias_train_stage1_encoder_maf_both.yaml --verbose

python -m den_pie density-train params/density_bias_train_stage2_flow_maf_both.yaml --verbose

python -m den_pie density-train params/density_bias_train_stage3_finetune_maf_both.yaml --verbose

python -m den_pie density-plot params/density_bias_plot_3stage_maf_both.yaml --verbose
