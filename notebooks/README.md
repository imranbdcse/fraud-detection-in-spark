# Notebooks

This directory contains Jupyter notebooks for exploratory data analysis (EDA) and machine learning model training.

## Notebooks

| Notebook | Description |
|---|---|
| `01_exploratory_data_analysis.ipynb` | Load and inspect the dataset, visualize class distribution, amount distributions, time-based patterns, and feature correlations. |
| `02_model_training.ipynb` | Preprocess data, train a Logistic Regression model, evaluate performance (ROC-AUC, F1, Precision-Recall), and save the model. |

## Running the Notebooks

### Prerequisites

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Generate the synthetic dataset if it does not yet exist:

```bash
python data/generate_synthetic_data.py
```

### Launch Jupyter

```bash
jupyter notebook notebooks/
```

Open the notebooks in order:
1. `01_exploratory_data_analysis.ipynb`
2. `02_model_training.ipynb`

## Training Script (Alternative to Notebook)

You can also train the model using the command-line script:

```bash
python src/batch/train_model.py \
    --data_path data/creditcard.csv \
    --model_dir models/ \
    --test_size 0.2
```

This saves the trained model and scaler to the `models/` directory.

## Usage Examples

### Generate predictions on new data

```python
from src.batch.predict import FraudPredictor

predictor = FraudPredictor(model_dir="models/")

# Single transaction
pred, prob = predictor.predict_single([0.1, -0.5, ..., 150.0, 3600])
print(f"Fraud: {pred}  Probability: {prob:.4f}")

# Batch predictions from DataFrame
import pandas as pd
df = pd.read_csv("data/creditcard.csv").drop(columns=["Class"])
results = predictor.predict_batch(df)
print(results[["prediction", "fraud_probability"]].head())
```
