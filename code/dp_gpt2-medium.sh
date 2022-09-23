#!/bin/bash
#
#SBATCH --job-name=df_gpt2
#SBATCH --output=/storage/ukp/work/matzken/fplm/dp-transformers/slurm/res_dfgptlora1.txt
#SBATCH --error=/storage/ukp/work/matzken/fplm/dp-transformers/slurm/error_dfgptlora1.txt
#SBATCH --mail-user=cleomatzken@gmail.com
#SBATCH --account=ukp-student
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --chdir=/storage/ukp/work/matzken/fplm

module load cuda/11.0
module purge
source /storage/ukp/work/matzken/fplm/miniconda3/bin/activate
conda activate /storage/ukp/work/matzken/fplm/miniconda3/env
python -m torch.distributed.run --nproc_per_node 1 /storage/ukp/work/matzken/fplm/dp-transformers/fine-tune-dp.py --output_dir=/storage/ukp/work/matzken/fplm/dp-transformers/dp_gpt2-medium --model_name gpt2-medium --sequence_len 128 --per_device_train_batch_size 8 --gradient_accumulation_steps 128 --evaluation_strategy steps --eval_steps 100000 --save_strategy steps --log_level info --per_device_eval_batch_size 64 --eval_accumulation_steps 1 --seed 42 --target_epsilon 8 --per_sample_max_grad_norm 1.0 --prediction_loss_only --weight_decay 0.01 --remove_unused_columns False --num_train_epochs 3 --logging_steps 5 --lora_dim 4 --lora_alpha 32 --lora_dropout 0.0 --max_grad_norm 0 --lr_scheduler_type constant --learning_rate 3e-5 --disable_tqdm True --dataloader_num_workers 2
