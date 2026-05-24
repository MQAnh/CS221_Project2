#!/bin/bash

python plot_confusion_matrices.py \
  --test_file ./data/UIT-ViON_test.csv \
  --lora_pred_file ./topic_phobert_lora/predict_results_topic.txt \
  --melora_pred_file ./topic_phobert_melora/predict_results_topic.txt \
  --label_column label \
  --output_dir ./confusion_matrices
