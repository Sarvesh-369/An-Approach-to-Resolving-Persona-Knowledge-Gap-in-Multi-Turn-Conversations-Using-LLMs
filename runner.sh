Shell script:
#!/bin/bash
#SBATCH --job-name=ESConv
#SBATCH --gres=gpu:2
#SBATCH --mem=64GB
#SBATCH --output=ESConv.out
#SBATCH --error=ESConv.err
#SBATCH --time=3-00:00:00

source /usr/ebuild/software/Anaconda3/2020.07/etc/profile.d/conda.sh
conda activate llm
python main.py --dataset data.ccpe.json --dataset_type ccpe --temperature 0.7 --max_tokens 400
