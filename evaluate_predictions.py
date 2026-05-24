import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def read_predictions(pred_file):
    """
    Đọc file predict_results_xxx.txt của HuggingFace Trainer.
    Format thường là:
    index   prediction
    0       3
    1       9
    """
    df = pd.read_csv(pred_file, sep="\t")
    df["prediction"] = df["prediction"].astype(int)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument("--output_report", type=str, default="classification_report.txt")
    args = parser.parse_args()

    pred_df = read_predictions(args.pred_file)
    test_df = pd.read_csv(args.test_file)

    y_pred = pred_df["prediction"].values
    y_true = test_df[args.label_column].astype(int).values

    assert len(y_pred) == len(y_true), \
        f"Length mismatch: predictions={len(y_pred)}, groundtruth={len(y_true)}"

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    print("===== Evaluation Result =====")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Macro-F1    : {macro_f1:.4f}")
    print(f"Micro-F1    : {micro_f1:.4f}")
    print(f"Weighted-F1 : {weighted_f1:.4f}")

    report = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write("===== Evaluation Result =====\n")
        f.write(f"Accuracy    : {acc:.4f}\n")
        f.write(f"Macro-F1    : {macro_f1:.4f}\n")
        f.write(f"Micro-F1    : {micro_f1:.4f}\n")
        f.write(f"Weighted-F1 : {weighted_f1:.4f}\n\n")
        f.write("===== Classification Report =====\n")
        f.write(report)
        f.write("\n\n===== Confusion Matrix =====\n")
        f.write(str(cm))

    print(f"\nSaved report to: {args.output_report}")


if __name__ == "__main__":
    main()