#!/bin/sh
seed=$1 # 3 per
seq_len=$2 # 128, 512
amb_fn=$3 # none, csi, wn

output_dir=/content/gdrive/MyDrive/Humor-Detection/custom_"$amb_fn"_"$seq_len"_"$seed"_clean
echo "$output_dir"
rm -rf "$output_dir"
mkdir "$output_dir"
echo Beginning training. Writing to "$output_dir"

# clean vs unclean

export BERT_BASE_DIR=./uncased_L-24_H-1024_A-16
export CUDA_VISIBLE_DEVICES=0
python3 run_classifier.py \
--task_name=cola \
--do_train \
--do_eval \
--use_clean_data \
--data_dir=./data \
--max_seq_length="$seq_len" \
--train_batch_size=256 \
--learning_rate=2e-5 \
--num_train_epochs=1.0 \
--do_lower_case \
--gradient_accumulation_steps 8 \
--overwrite_cache \
--bert_model=bert-base-uncased \
--output_dir="$output_dir" \
--ambiguity_fn="$amb_fn" \
--seed "$seed" \
