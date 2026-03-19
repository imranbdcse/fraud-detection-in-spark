"""
Generate synthetic credit card transaction dataset for fraud detection.

Generates 100,000 samples with 30 features (V1-V28, Amount, Time)
with a realistic 2% fraud ratio.
"""

import numpy as np
import pandas as pd
import os


def generate_synthetic_data(n_samples: int = 100000, fraud_ratio: float = 0.02, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic credit card transaction data.

    Args:
        n_samples: Total number of transactions to generate.
        fraud_ratio: Fraction of transactions that are fraudulent.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with columns V1-V28, Amount, Time, and Class.
    """
    np.random.seed(random_state)

    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud

    # --- Normal transactions ---
    normal_features = np.random.randn(n_normal, 28)
    normal_amount = np.abs(np.random.exponential(scale=88, size=n_normal))
    normal_time = np.sort(np.random.uniform(0, 172800, n_normal))  # 2-day window
    normal_labels = np.zeros(n_normal, dtype=int)

    # --- Fraudulent transactions ---
    fraud_features = np.random.randn(n_fraud, 28) * 1.5 + 0.5
    # Fraudulent amounts tend to be either very small or very large
    fraud_amount = np.abs(np.concatenate([
        np.random.exponential(scale=10, size=n_fraud // 2),
        np.random.exponential(scale=500, size=n_fraud - n_fraud // 2),
    ]))
    fraud_time = np.random.uniform(0, 172800, n_fraud)
    fraud_labels = np.ones(n_fraud, dtype=int)

    # --- Combine and shuffle ---
    features = np.vstack([normal_features, fraud_features])
    amounts = np.concatenate([normal_amount, fraud_amount])
    times = np.concatenate([normal_time, fraud_time])
    labels = np.concatenate([normal_labels, fraud_labels])

    shuffle_idx = np.random.permutation(n_samples)
    features = features[shuffle_idx]
    amounts = amounts[shuffle_idx]
    times = times[shuffle_idx]
    labels = labels[shuffle_idx]

    columns = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
    df = pd.DataFrame(
        np.column_stack([features, amounts, times]),
        columns=columns,
    )
    df["Class"] = labels
    df["Amount"] = df["Amount"].round(2)
    df["Time"] = df["Time"].round(0).astype(int)

    return df


def main():
    output_dir = os.path.join(os.path.dirname(__file__))
    output_path = os.path.join(output_dir, "creditcard.csv")

    print("Generating synthetic credit card transaction dataset...")
    df = generate_synthetic_data(n_samples=100000, fraud_ratio=0.02)

    # --- Statistics ---
    total = len(df)
    n_fraud = df["Class"].sum()
    n_normal = total - n_fraud
    print(f"\nDataset Statistics:")
    print(f"  Total samples  : {total:,}")
    print(f"  Normal (0)     : {n_normal:,} ({n_normal / total * 100:.2f}%)")
    print(f"  Fraud  (1)     : {n_fraud:,}  ({n_fraud / total * 100:.2f}%)")
    print(f"\nFeature Summary:")
    print(df.describe())

    print(f"\nSample Data (first 5 rows):")
    print(df.head())

    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")


if __name__ == "__main__":
    main()
