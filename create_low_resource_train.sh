#!/bin/bash

mkdir -p ./data_low_resource

python create_low_resource_train.py \
  --train_file ./data/UIT-ViON_train.csv \
  --output_file ./data_low_resource/UIT-ViON_train_2000_seed42.csv \
  --label_column label \
  --num_samples 5000 \
  --seed 42
