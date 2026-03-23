from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col, when
import argparse
import logging
from datetime import datetime
import json

"""
Loan Default Risk Scoring Pipeline - PySpark Model Training
Banking Use Case: Enterprise-grade ML model for retail loan default prediction
"""

from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_spark(app_name: str = "LoanDefaultRiskScoring") -> SparkSession:
    """Initialize Spark session with banking ML configs"""
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    return spark


def load_data(spark: SparkSession, data_path: str) -> any:
    """Load training dataset from CSV"""
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    logger.info(f"Loaded data shape: {df.count()} rows, {len(df.columns)} columns")
    return df


def preprocess_data(df: any) -> any:
    """Data preprocessing and feature engineering"""
    # Handle missing values
    df = df.na.fill({"age": df.agg({"age": "avg"}).collect()[0][0]})
    
    # Define target variable (1: default, 0: no default)
    df = df.withColumn("default", col("default").cast("integer"))
    
    # Feature engineering
    df = df.withColumn(
        "income_bucket",
        when(col("annual_income") < 30000, "low")
        .when(col("annual_income") < 70000, "medium")
        .otherwise("high")
    )
    
    logger.info("Data preprocessing completed")
    return df


def build_pipeline(categorical_features: list, numerical_features: list) -> Pipeline:
    """Build ML pipeline with feature engineering and Random Forest"""
    
    stages = []
    
    # Categorical feature processing
    for cat_col in categorical_features:
        indexer = StringIndexer(inputCol=cat_col, outputCol=f"{cat_col}_indexed")
        encoder = OneHotEncoder(inputCol=f"{cat_col}_indexed", outputCol=f"{cat_col}_encoded")
        stages.extend([indexer, encoder])
    
    # Vector assembly
    encoded_cats = [f"{col}_encoded" for col in categorical_features]
    assembler = VectorAssembler(
        inputCols=encoded_cats + numerical_features,
        outputCol="features_raw"
    )
    stages.append(assembler)
    
    # Feature scaling
    scaler = StandardScaler(inputCol="features_raw", outputCol="features")
    stages.append(scaler)
    
    # Random Forest Classifier
    rf = RandomForestClassifier(
        labelCol="default",
        featuresCol="features",
        numTrees=100,
        maxDepth=10,
        minInstancesPerNode=5,
        seed=42,
        numPartitions=10
    )
    stages.append(rf)
    
    pipeline = Pipeline(stages=stages)
    logger.info(f"Pipeline created with {len(stages)} stages")
    return pipeline


def train_model(spark: SparkSession, df: any, pipeline: Pipeline) -> any:
    """Train Random Forest model"""
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
    
    logger.info(f"Training set: {train_data.count()}, Test set: {test_data.count()}")
    
    model = pipeline.fit(train_data)
    logger.info("Model training completed")
    
    return model, train_data, test_data


def evaluate_model(model: any, test_data: any) -> dict:
    """Evaluate model performance"""
    predictions = model.transform(test_data)
    
    # Binary classification evaluator
    auc_eval = BinaryClassificationEvaluator(labelCol="default", metricName="areaUnderROC")
    auc_score = auc_eval.evaluate(predictions)
    
    # Multiclass metrics
    mc_eval = MulticlassClassificationEvaluator(labelCol="default", metricName="accuracy")
    accuracy = mc_eval.evaluate(predictions)
    
    metrics = {
        "auc_roc": round(auc_score, 4),
        "accuracy": round(accuracy, 4),
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"Model Metrics: {json.dumps(metrics, indent=2)}")
    return metrics


def save_model(model: any, model_path: str):
    """Save trained model"""
    model.write().overwrite().save(model_path)
    logger.info(f"Model saved to {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Loan Default Risk Scoring - Model Training")
    parser.add_argument("--data_path", required=True, help="Path to training data CSV")
    parser.add_argument("--model_path", required=True, help="Path to save trained model")
    
    args = parser.parse_args()
    
    # Initialize Spark
    spark = initialize_spark()
    
    # Load and preprocess data
    df = load_data(spark, args.data_path)
    df = preprocess_data(df)
    
    # Define features
    categorical_features = ["employment_status", "income_bucket", "credit_category"]
    numerical_features = ["age", "annual_income", "existing_credit_score", "loan_amount"]
    
    # Build and train pipeline
    pipeline = build_pipeline(categorical_features, numerical_features)
    model, train_data, test_data = train_model(spark, df, pipeline)
    
    # Evaluate model
    metrics = evaluate_model(model, test_data)
    
    # Save model
    save_model(model, args.model_path)
    
    spark.stop()
    logger.info("Training pipeline completed successfully")


if __name__ == "__main__":
    main()