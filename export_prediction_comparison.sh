#!/bin/bash

python export_prediction_comparison.py \
  --test_file ./data/UIT-ViON_test.csv \
  --lora_pred_file ./topic_phobert_lora/predict_results_topic.txt \
  --melora_pred_file ./topic_phobert_melora/predict_results_topic.txt \
  --text_column title \
  --label_column label \
  --output_file ./prediction_comparison.json
