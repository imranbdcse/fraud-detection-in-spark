# Notebooks

Jupyter notebooks for exploratory data analysis and model training.

## Contents

| Notebook | Description |
|----------|-------------|
| `01_exploratory_data_analysis.ipynb` | Load and inspect the dataset, visualise class distribution, amounts, correlation matrix |
| `02_model_training.ipynb` | Preprocess data, train Logistic Regression, evaluate with ROC-AUC / F1 / confusion matrix, save model |

## Running the Notebooks

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate the dataset (if not already done)

```bash
python data/generate_synthetic_data.py
```

This creates `data/creditcard.csv` with 100 000 transactions (2 % fraud).

### 3. Launch Jupyter

```bash
jupyter notebook notebooks/
```

Open the notebooks in order:
1. `01_exploratory_data_analysis.ipynb`
2. `02_model_training.ipynb`

## Usage Examples

### Exploratory Analysis

```python
import pandas as pd
df = pd.read_csv('../data/creditcard.csv')
df['Class'].value_counts()
```

### Model Training (script equivalent)

```bash
python src/batch/train_model.py \
    --data-path data/creditcard.csv \
    --model-dir models/ \
    --test-size 0.2
```

After training, `models/fraud_model.pkl` and `models/scaler.pkl` are ready
for use by the prediction and streaming components.
