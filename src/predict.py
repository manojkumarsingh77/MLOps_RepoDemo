from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col, when
import argparse
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_spark(app_name: str = "LoanDefaultRiskScoring-Predict") -> SparkSession:
    """Initialize Spark session"""
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    return spark


def load_model(spark: SparkSession, model_path: str) -> PipelineModel:
    """Load trained model"""
    model = PipelineModel.load(model_path)
    logger.info(f"Model loaded from {model_path}")
    return model


def load_data(spark: SparkSession, data_path: str):
    """Load prediction dataset"""
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    logger.info(f"Loaded data: {df.count()} rows, {len(df.columns)} columns")
    return df


def preprocess_data(df):
    """Preprocess prediction data"""
    df = df.na.fill({"age": df.agg({"age": "avg"}).collect()[0][0]})
    df = df.withColumn(
        "income_bucket",
        when(col("annual_income") < 30000, "low")
        .when(col("annual_income") < 70000, "medium")
        .otherwise("high")
    )
    logger.info("Data preprocessing completed")
    return df


def predict(model: PipelineModel, df):
    """Generate predictions"""
    predictions = model.transform(df)
    return predictions.select("*", col("prediction"), col("probability"))


def save_predictions(predictions, output_path: str):
    """Save predictions to CSV"""
    predictions.coalesce(1).write.csv(output_path, header=True, mode="overwrite")
    logger.info(f"Predictions saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Loan Default Risk Scoring - Prediction")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--data_path", required=True, help="Path to prediction data CSV")
    parser.add_argument("--output_path", required=True, help="Path to save predictions")
    
    args = parser.parse_args()
    
    spark = initialize_spark()
    model = load_model(spark, args.model_path)
    df = load_data(spark, args.data_path)
    df = preprocess_data(df)
    predictions = predict(model, df)
    save_predictions(predictions, args.output_path)
    
    spark.stop()
    logger.info("Prediction completed successfully")


if __name__ == "__main__":
    main()