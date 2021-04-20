#!/bin/sh

export BERT_BASE_DIR=./uncased_L-24_H-1024_A-16
export CUDA_VISIBLE_DEVICES=0
python3 eval.py \
--data_dir=./data \
--do_lower_case \
--overwrite_cache  \
--bert_model=bert-base-uncased \
--max_seq_length 128 \
--old_load \
--bert_base \
--model_weights /content/gdrive/MyDrive/Humor-Detection/baseline_new_clean_128_0/state_dict.pt,/content/gdrive/MyDrive/Humor-Detection/baseline_new_clean_128_1/state_dict.pt,/content/gdrive/MyDrive/Humor-Detection/baseline_new_clean_128_2/state_dict.pt
