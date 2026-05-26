# CS221 Project 2 - Vietnamese News Topic Classification

Repo này dùng các mô hình Transformer kết hợp PEFT để phân loại chủ đề tin tức tiếng Việt từ cột `title` của bộ dữ liệu UIT-ViON. Mục tiêu chính là so sánh LoRA và MELoRA trên bài toán topic classification, đồng thời cung cấp các script để huấn luyện, đánh giá, vẽ confusion matrix và chạy demo nhập tiêu đề bằng Gradio.

## Bài Toán

Input là một câu tiêu đề tin tức, ví dụ:

```text
vieri ronaldo gioi hon cristiano_ronaldo
```

Model dự đoán tiêu đề thuộc một trong 13 lớp:

```text
0  TECHNOLOGY
1  TRAVEL
2  EDUCATION
3  ENTERTAINMENT
4  SCIENCE
5  BUSINESS
6  LAW
7  HEATH
8  WORLD
9  SPORT
10 NEWS
11 VEHICLE
12 LIFE
```

## Cấu Trúc Chính

```text
.
├── run_topic_classification.py       # Train/eval/predict cho topic classification
├── run_topic_classification.sh       # Chạy PhoBERT + LoRA
├── run_xlmr_topic_classification.sh  # Chạy XLM-RoBERTa + LoRA/MELoRA
├── run_topic_low_resource.sh         # Thí nghiệm low-resource
├── create_low_resource_train.py      # Tạo tập train nhỏ hơn từ train CSV
├── evaluate_predictions.py           # Tính Accuracy, F1, classification report
├── plot_confusion_matrices.py        # Vẽ confusion matrix cho LoRA và MELoRA
├── export_prediction_comparison.py   # Xuất JSON so sánh prediction từng mẫu
├── demo_topic_gradio.py              # Demo Gradio nhập title và dự đoán class
├── peft-0.5.0/                       # PEFT đã chỉnh để hỗ trợ MELoRA
├── utils/, templates/, figs/          # Code/phụ trợ từ MELoRA
└── requirements.txt
```

## Cài Đặt

Tạo môi trường Python rồi cài dependencies:

```bash
conda create -n cs221-topic python=3.10
conda activate cs221-topic
pip install -r requirements.txt
```

Cài bản PEFT local trong repo:

```bash
cd peft-0.5.0
pip install -e .
cd ..
```

## Dữ Liệu

Dataset mong đợi có 3 file CSV:

```text
data/UIT-ViON_train.csv
data/UIT-ViON_dev.csv
data/UIT-ViON_test.csv
```

Các file CSV cần có ít nhất các cột:

```text
title, link, label
```

Có thể tải dữ liệu bằng:

```bash
bash load_topic_dataset.sh
```

## Huấn Luyện

Chạy PhoBERT + LoRA:

```bash
bash run_topic_classification.sh
```

Chạy XLM-RoBERTa với LoRA và MELoRA:

```bash
bash run_xlmr_topic_classification.sh
```

Chạy thí nghiệm low-resource:

```bash
bash run_topic_low_resource.sh
```

Các checkpoint và kết quả prediction sẽ được lưu trong thư mục `output_dir` tương ứng, ví dụ `topic_phobert_lora/`, `topic_xlmr_lora/`, `topic_xlmr_melora/`.

## Đánh Giá

Sau khi có file prediction dạng `predict_results_topic.txt`, chạy:

```bash
bash run_evalutate_prediction.sh
```

Script này tạo report gồm Accuracy, Macro-F1, Micro-F1, Weighted-F1, classification report và confusion matrix dạng text.

Vẽ confusion matrix:

```bash
bash plot_confusion_matrices.sh
```

Xuất file JSON so sánh kết quả LoRA và MELoRA theo từng mẫu:

```bash
bash export_prediction_comparison.sh
```

## Demo Gradio

File `demo_topic_gradio.py` cho phép nhập một `title` và nhận class dự đoán cùng top probability.

Chạy demo:

```bash
python demo_topic_gradio.py
```

Mặc định demo tìm adapter ở:

```text
./topic_phobert_lora
../hihi/topic_phobert_lora
```

Nếu model nằm ở đường dẫn khác, truyền trực tiếp:

```bash
python demo_topic_gradio.py --adapter_path "D:\2026\HK4\cs221\project2\hihi\topic_phobert_lora"
```

Sau đó mở:

```text
http://127.0.0.1:7860
```

## Ghi Chú

- `run_topic_classification.py` tự suy ra số class từ cột `label` của train set.
- Với PhoBERT, tokenizer thường dùng `use_fast=False`.
- `target_modules` mặc định cho LoRA/MELoRA là `query value`.
- Cần đảm bảo thư mục adapter có đủ `adapter_config.json` và `adapter_model.bin` hoặc `adapter_model.safetensors` hợp lệ.

## Nguồn Tham Khảo

Code được phát triển dựa trên MELoRA, HuggingFace PEFT và Transformers.

```bibtex
@article{melora,
  title={Mini-Ensemble Low-Rank Adapters for Parameter-Efficient Fine-Tuning},
  author={Ren, Pengjie and Shi, Chengshun and Wu, Shiguang and Zhang, Mengqi and Ren, Zhaochun and de Rijke, Maarten and Chen, Zhumin and Pei, Jiahuan},
  journal={arXiv preprint arXiv:2402.17263},
  year={2024}
}
```

Link github: https://github.com/MQAnh/CS221_Project2
