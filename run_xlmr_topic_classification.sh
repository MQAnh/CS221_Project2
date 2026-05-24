#!/bin/bash

run(){
  mode=$1
  output_dir=$2

  python run_topic_classification.py \
    --model_name_or_path xlm-roberta-base \
    --train_file ./data/UIT-ViON_train.csv \
    --validation_file ./data/UIT-ViON_dev.csv \
    --test_file ./data/UIT-ViON_test.csv \
    --text_column title \
    --label_column label \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 128 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 15 \
    --output_dir "${output_dir}" \
    --overwrite_output_dir \
    --mode "${mode}" \
    --l_num 4 \
    --rank 8 \
    --lora_alpha 16 \
    --target_modules query value
}

run base ./topic_xlmr_lora
run me ./topic_xlmr_melora
