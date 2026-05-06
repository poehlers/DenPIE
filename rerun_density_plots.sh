#!/bin/bash
#SBATCH --job-name=density_replot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=08:00:00
#SBATCH --output=/home/poehlers/cosmo_thesis/den_pie/output/logs/%x-%j-%N_slurm.out
#SBATCH --error=/home/poehlers/cosmo_thesis/den_pie/output/logs/R-%x.%j.err

module purge
module load 2025
module load Python/3.13.1-GCCcore-14.2.0
module load OpenMPI/5.0.7-GCC-14.2.0
module load FFTW/3.3.10-GCC-14.2.0

source /home/poehlers/21cmvenv/bin/activate

cd /home/poehlers/cosmo_thesis/den_pie

YAMLS=(
    params/density_plot.yaml                # base_test (FrEIA, 1-stage)
    params/density_plot_1stage.yaml         # NSF (1-stage)
    params/density_plot_1stage_MAF.yaml     # MAF (1-stage)
    params/density_plot_multi_stage.yaml    # stage3_finetune (FrEIA, 3-stage)
    params/density_plot_3stage.yaml         # stage3_finetune_NSF (3-stage)
    params/density_plot_3stage_MAF.yaml     # stage3_finetune_MAF (3-stage)
)

for yaml in "${YAMLS[@]}"; do
    echo "============================================"
    echo "Running density-plot with ${yaml}"
    echo "============================================"
    python -m den_pie density-plot "${yaml}" --verbose
    echo "Finished ${yaml}"
    echo ""
done

echo "All density plots complete."
