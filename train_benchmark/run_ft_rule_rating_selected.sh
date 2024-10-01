#!/bin/bash

percentages=(1 2 5 10 20)
folder_dirs=("ft_all50rules/" "ft_10rules_dpp/" "ft_10rules_randomA/" "ft_10rules_randomB/" "ft_10rules_randomC/" "ft_no_rule/")/

for folder_dir in "${folder_dirs[@]}"; do
    for percent in "${percentages[@]}"; do
        data_path="selected_data/SlimPajama6B_short_1M_${percent}percent.jsonl"
        save_model_path="finetuned_models/finetuned_Llama3-8B_${percent}percent"
        python /DataSelection/experiments_Math/ft_rule_rating_selected.py --folder_path $folder_dir --dataset_path $data_path --save_model_path $save_model_path --wandb_proj_name "experiment_Code"
    done
done
