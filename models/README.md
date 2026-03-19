# Models Directory

This directory stores pretrained machine learning models used for real-time fraud inference.

## Saving a Model

After training in the Jupyter notebook, save the model:
```python
model.save("models/logistic_regression_fraud_model")
```

## Loading a Model

Load the saved model for inference in the streaming pipeline:
```python
from pyspark.ml.classification import LogisticRegressionModel
model = LogisticRegressionModel.load("models/logistic_regression_fraud_model")
```

## Model Files

- `logistic_regression_fraud_model/` — Trained Logistic Regression model (Spark MLlib format)
