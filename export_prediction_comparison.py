import argparse
import json

import pandas as pd


CLASS_NAMES = [
    "TECHNOLOGY",
    "TRAVEL",
    "EDUCATION",
    "ENTERTAINMENT",
    "SCIENCE",
    "BUSINESS",
    "LAW",
    "HEATH",
    "WORLD",
    "SPORT",
    "NEWS",
    "VEHICLE",
    "LIFE",
]


def label_to_class_name(label):
    if label < 0 or label >= len(CLASS_NAMES):
        raise ValueError(f"Unknown label id {label}. Expected 0-{len(CLASS_NAMES) - 1}.")
    return CLASS_NAMES[label]


def read_predictions(pred_file):
    df = pd.read_csv(pred_file, sep="\t")
    if "prediction" not in df.columns:
        raise ValueError(f"Prediction file must contain a 'prediction' column: {pred_file}")
    return df["prediction"].astype(int).tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Export per-sample LoRA and MeLORA predictions with ground truth to JSON."
    )
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--lora_pred_file", type=str, required=True)
    parser.add_argument("--melora_pred_file", type=str, required=True)
    parser.add_argument("--text_column", type=str, default="title")
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument("--output_file", type=str, default="prediction_comparison.json")
    args = parser.parse_args()

    test_df = pd.read_csv(args.test_file)
    if args.text_column not in test_df.columns:
        raise ValueError(f"Test file does not contain text column '{args.text_column}'")
    if args.label_column not in test_df.columns:
        raise ValueError(f"Test file does not contain label column '{args.label_column}'")

    lora_predictions = read_predictions(args.lora_pred_file)
    melora_predictions = read_predictions(args.melora_pred_file)
    titles = test_df[args.text_column].astype(str).tolist()
    ground_truths = test_df[args.label_column].astype(int).tolist()

    if len(lora_predictions) != len(ground_truths):
        raise ValueError(
            f"LoRA length mismatch: predictions={len(lora_predictions)}, "
            f"ground_truth={len(ground_truths)}"
        )
    if len(melora_predictions) != len(ground_truths):
        raise ValueError(
            f"MeLORA length mismatch: predictions={len(melora_predictions)}, "
            f"ground_truth={len(ground_truths)}"
        )

    records = []
    for index, (title, ground_truth, lora_pred, melora_pred) in enumerate(
        zip(titles, ground_truths, lora_predictions, melora_predictions)
    ):
        lora_correct = lora_pred == ground_truth
        melora_correct = melora_pred == ground_truth
        records.append(
            {
                "index": index,
                "title": title,
                "lora": {
                    "prediction": label_to_class_name(lora_pred),
                    "correct": lora_correct,
                },
                "melora": {
                    "prediction": label_to_class_name(melora_pred),
                    "correct": melora_correct,
                },
                "ground_truth": label_to_class_name(ground_truth),
            }
        )

    with open(args.output_file, "w", encoding="utf-8") as writer:
        json.dump(records, writer, ensure_ascii=False, indent=2)

    print(f"Saved comparison JSON to: {args.output_file}")


if __name__ == "__main__":
    main()
