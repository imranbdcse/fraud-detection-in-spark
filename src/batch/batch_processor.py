"""
Batch Processor: Offline Fraud Analysis with Spark
===================================================
Reads historical credit card transaction data, applies feature engineering,
trains or evaluates a Logistic Regression model, and writes the results
(predictions and performance metrics) to an output directory.

Usage:
    spark-submit src/batch/batch_processor.py \\
        --input-path data/creditcard.csv \\
        --model-path models/logistic_regression_fraud_model \\
        --output-path output/batch_predictions \\
        --mode train          # or 'predict'
"""

import argparse
import logging

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Feature columns present in the Kaggle Credit Card Fraud dataset
# (V1–V28 are PCA-transformed, plus Amount and Time)
FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
LABEL_COL = "Class"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(spark: SparkSession, input_path: str) -> DataFrame:
    """
    Load the transaction dataset from a CSV file.

    Args:
        spark:      Active SparkSession.
        input_path: Path to the CSV file.

    Returns:
        A Spark DataFrame with all columns inferred.
    """
    logger.info("Loading data from %s", input_path)
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(input_path)
    )
    logger.info("Loaded %d rows with %d columns.", df.count(), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess(df: DataFrame) -> DataFrame:
    """
    Drop nulls, rename label column, and return a cleaned DataFrame.

    Args:
        df: Raw input DataFrame.

    Returns:
        Cleaned DataFrame with a 'label' column (double) for MLlib.
    """
    df = df.dropna(subset=FEATURE_COLS + [LABEL_COL])
    df = df.withColumn("label", F.col(LABEL_COL).cast("double"))
    logger.info("After preprocessing: %d rows.", df.count())
    return df


# ---------------------------------------------------------------------------
# Build ML pipeline
# ---------------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    """
    Construct a Spark ML Pipeline:
      1. VectorAssembler  — combine feature columns into a single vector.
      2. StandardScaler   — normalize features (zero mean, unit variance).
      3. LogisticRegression — binary classifier.

    Returns:
        An unfitted Spark ML Pipeline.
    """
    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="raw_features")
    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withMean=True,
        withStd=True,
    )
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=100,
        regParam=0.01,
        elasticNetParam=0.0,
    )
    return Pipeline(stages=[assembler, scaler, lr])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(df: DataFrame, model_path: str) -> None:
    """
    Train a Logistic Regression pipeline with cross-validation and save
    the best model.

    Args:
        df:         Preprocessed DataFrame with features and 'label' column.
        model_path: Directory to save the trained model.
    """
    logger.info("Splitting data: 80%% train / 20%% test.")
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    pipeline = build_pipeline()

    # Parameter grid for cross-validation
    param_grid = (
        ParamGridBuilder()
        .addGrid(pipeline.getStages()[-1].regParam, [0.01, 0.1])
        .addGrid(pipeline.getStages()[-1].maxIter, [50, 100])
        .build()
    )

    evaluator = BinaryClassificationEvaluator(
        labelCol="label", metricName="areaUnderROC"
    )

    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=3,
    )

    logger.info("Training model with cross-validation...")
    cv_model = cv.fit(train_df)
    best_model = cv_model.bestModel

    # Evaluate on test set
    predictions = best_model.transform(test_df)
    roc_auc = evaluator.evaluate(predictions)

    mc_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction"
    )
    f1 = mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "f1"})
    precision = mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "weightedPrecision"})
    recall = mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "weightedRecall"})

    logger.info("=== Model Evaluation (Test Set) ===")
    logger.info("  AUC-ROC   : %.4f", roc_auc)
    logger.info("  F1 Score  : %.4f", f1)
    logger.info("  Precision : %.4f", precision)
    logger.info("  Recall    : %.4f", recall)

    logger.info("Saving best model to %s", model_path)
    best_model.write().overwrite().save(model_path)
    logger.info("Model saved successfully.")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def run_predictions(
    spark: SparkSession,
    df: DataFrame,
    model_path: str,
    output_path: str,
) -> None:
    """
    Load a saved pipeline model and run batch predictions on the input data.

    Args:
        spark:       Active SparkSession.
        df:          Input DataFrame to score.
        model_path:  Path to the saved Spark ML model.
        output_path: Directory to write prediction results as Parquet.
    """
    from pyspark.ml import PipelineModel  # noqa: PLC0415

    logger.info("Loading model from %s", model_path)
    model = PipelineModel.load(model_path)

    logger.info("Running predictions...")
    predictions = model.transform(df)

    result = predictions.select(
        *[F.col(c) for c in df.columns if c in ["TransactionID", "UserID", "label"]],
        "prediction",
        "probability",
    )

    logger.info("Writing predictions to %s", output_path)
    result.write.mode("overwrite").parquet(output_path)
    logger.info("Batch predictions complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spark Batch Fraud Processor")
    parser.add_argument("--input-path",  default="data/creditcard.csv",
                        help="Path to input CSV data")
    parser.add_argument("--model-path",  default="models/logistic_regression_fraud_model",
                        help="Path to save or load Spark ML model")
    parser.add_argument("--output-path", default="output/batch_predictions",
                        help="Directory to write predictions")
    parser.add_argument("--mode",        choices=["train", "predict"], default="train",
                        help="Run mode: 'train' (default) or 'predict'")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    spark = (
        SparkSession.builder
        .appName("FraudDetectionBatch")
        .config("spark.sql.shuffle.partitions", "50")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    df = load_data(spark, args.input_path)
    df = preprocess(df)

    if args.mode == "train":
        train_model(df, args.model_path)
    else:
        run_predictions(spark, df, args.model_path, args.output_path)

    spark.stop()


if __name__ == "__main__":
    main()
