#!/bin/bash
Data_list=( roman-empire amazon-ratings minesweeper tolokers questions )

for ds in "${Data_list[@]}"; do
    job_name=hgnn_${ds}
    sbatch << HERE
#!/bin/env bash
#SBATCH --output=experiments/logs/%A_%a_${job_name}.out
#SBATCH --job-name=${job_name}
#SBATCH --array=1-5
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --gpus=1

source activate hgnn
n_layer=\${SLURM_ARRAY_TASK_ID}
python train.py --name SAGE_l${n_layer} --dataset ${ds} --model SAGE --num_layers ${n_layer} --device cuda:0 --verbose
HERE
done