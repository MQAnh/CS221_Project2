import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


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


def read_predictions(pred_file):
    df = pd.read_csv(pred_file, sep="\t")
    if "prediction" not in df.columns:
        raise ValueError(f"Prediction file must contain a 'prediction' column: {pred_file}")
    return df["prediction"].astype(int).tolist()


def save_confusion_matrix(y_true, y_pred, title, output_prefix, normalize=False):
    labels = list(range(len(CLASS_NAMES)))
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize="true" if normalize else None,
    )

    csv_file = f"{output_prefix}.csv"
    png_file = f"{output_prefix}.png"

    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_df.to_csv(csv_file, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(14, 12))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    display.plot(
        ax=ax,
        cmap="Blues",
        colorbar=True,
        values_format=".2f" if normalize else "d",
        xticks_rotation=45,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Ground Truth")
    plt.tight_layout()
    plt.savefig(png_file, dpi=300)
    plt.close(fig)

    print(f"Saved: {png_file}")
    print(f"Saved: {csv_file}")


def main():
    parser = argparse.ArgumentParser(description="Plot confusion matrices for LoRA and MeLORA.")
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--lora_pred_file", type=str, required=True)
    parser.add_argument("--melora_pred_file", type=str, required=True)
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument("--output_dir", type=str, default="./confusion_matrices")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize each row by ground-truth class count.",
    )
    args = parser.parse_args()

    test_df = pd.read_csv(args.test_file)
    if args.label_column not in test_df.columns:
        raise ValueError(f"Test file does not contain label column '{args.label_column}'")

    y_true = test_df[args.label_column].astype(int).tolist()
    lora_predictions = read_predictions(args.lora_pred_file)
    melora_predictions = read_predictions(args.melora_pred_file)

    if len(lora_predictions) != len(y_true):
        raise ValueError(
            f"LoRA length mismatch: predictions={len(lora_predictions)}, ground_truth={len(y_true)}"
        )
    if len(melora_predictions) != len(y_true):
        raise ValueError(
            f"MeLORA length mismatch: predictions={len(melora_predictions)}, ground_truth={len(y_true)}"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = "_normalized" if args.normalize else ""

    save_confusion_matrix(
        y_true,
        lora_predictions,
        "LoRA Confusion Matrix",
        os.path.join(args.output_dir, f"lora_confusion_matrix{suffix}"),
        normalize=args.normalize,
    )
    save_confusion_matrix(
        y_true,
        melora_predictions,
        "MeLORA Confusion Matrix",
        os.path.join(args.output_dir, f"melora_confusion_matrix{suffix}"),
        normalize=args.normalize,
    )


if __name__ == "__main__":
    main()
