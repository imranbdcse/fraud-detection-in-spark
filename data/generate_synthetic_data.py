"""
Generate synthetic credit card transaction dataset for fraud detection.

Creates a realistic imbalanced dataset with 100,000 samples and 2% fraud ratio,
matching the format of the Kaggle Credit Card Fraud Detection dataset.
"""

import numpy as np
import pandas as pd
import os


def generate_synthetic_data(
    n_samples: int = 100000,
    fraud_ratio: float = 0.02,
    random_state: int = 42,
    output_path: str = None,
) -> pd.DataFrame:
    """
    Generate synthetic credit card transaction data.

    Args:
        n_samples: Total number of transactions to generate.
        fraud_ratio: Fraction of transactions that are fraudulent.
        random_state: Random seed for reproducibility.
        output_path: Path to save the CSV file. Defaults to data/creditcard.csv.

    Returns:
        DataFrame containing the synthetic dataset.
    """
    rng = np.random.RandomState(random_state)

    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud

    print(f"Generating {n_samples:,} transactions ({n_normal:,} normal, {n_fraud:,} fraudulent)...")

    # ------------------------------------------------------------------ #
    # Time feature (seconds elapsed over ~2 days)
    # ------------------------------------------------------------------ #
    time_normal = np.sort(rng.uniform(0, 172800, n_normal))
    time_fraud = rng.uniform(0, 172800, n_fraud)
    time_all = np.concatenate([time_normal, time_fraud])

    # ------------------------------------------------------------------ #
    # V1-V28: PCA-like anonymous features
    # Normal transactions cluster around zero; fraud has shifted means.
    # ------------------------------------------------------------------ #
    normal_features = rng.randn(n_normal, 28)

    # Fraud transactions have different statistical properties
    fraud_shift = rng.uniform(-3, 3, 28)
    fraud_features = rng.randn(n_fraud, 28) * 1.5 + fraud_shift

    features_all = np.vstack([normal_features, fraud_features])

    # ------------------------------------------------------------------ #
    # Amount feature
    # Normal: log-normal with lower amounts; Fraud: slightly higher variance
    # ------------------------------------------------------------------ #
    amount_normal = np.exp(rng.normal(3.5, 1.5, n_normal)).clip(0.5, 5000)
    amount_fraud = np.exp(rng.normal(4.0, 2.0, n_fraud)).clip(0.5, 10000)
    amount_all = np.concatenate([amount_normal, amount_fraud])

    # ------------------------------------------------------------------ #
    # Class labels
    # ------------------------------------------------------------------ #
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])

    # ------------------------------------------------------------------ #
    # Assemble DataFrame and shuffle
    # ------------------------------------------------------------------ #
    columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    data = np.column_stack([time_all, features_all, amount_all, labels])
    df = pd.DataFrame(data, columns=columns)

    # Shuffle rows
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df["Class"] = df["Class"].astype(int)

    return df


def display_statistics(df: pd.DataFrame) -> None:
    """Print summary statistics for the generated dataset."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total samples : {len(df):,}")
    print(f"Features      : {df.shape[1] - 1}")
    print(f"Normal (0)    : {(df['Class'] == 0).sum():,}  ({(df['Class'] == 0).mean() * 100:.1f}%)")
    print(f"Fraud  (1)    : {(df['Class'] == 1).sum():,}  ({(df['Class'] == 1).mean() * 100:.1f}%)")

    print("\nAmount statistics by class:")
    print(df.groupby("Class")["Amount"].describe().round(2))

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes.value_counts())


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "creditcard.csv")

    df = generate_synthetic_data(
        n_samples=100000,
        fraud_ratio=0.02,
        random_state=42,
        output_path=output_path,
    )

    display_statistics(df)

    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / (1024 * 1024):.1f} MB")


if __name__ == "__main__":
    main()
