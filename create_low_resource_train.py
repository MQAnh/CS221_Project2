import argparse

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description="Create a low-resource train CSV split.")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no_stratify",
        action="store_true",
        help="Disable stratified sampling by label.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.train_file)
    if args.label_column not in df.columns:
        raise ValueError(f"Train file does not contain label column '{args.label_column}'")
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be positive")
    if args.num_samples > len(df):
        raise ValueError(
            f"--num_samples={args.num_samples} is larger than train size={len(df)}"
        )

    stratify = None if args.no_stratify else df[args.label_column]
    low_resource_df, _ = train_test_split(
        df,
        train_size=args.num_samples,
        random_state=args.seed,
        shuffle=True,
        stratify=stratify,
    )
    low_resource_df = low_resource_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    low_resource_df.to_csv(args.output_file, index=False, encoding="utf-8-sig")

    print(f"Original train size: {len(df)}")
    print(f"Low-resource train size: {len(low_resource_df)}")
    print(f"Saved to: {args.output_file}")
    print("\nLabel distribution:")
    print(low_resource_df[args.label_column].value_counts().sort_index())


if __name__ == "__main__":
    main()
