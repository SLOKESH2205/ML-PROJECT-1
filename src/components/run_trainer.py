import os
import joblib
import logging
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.feature_engineering import build_features
from src.components.model_trainer import ModelTrainer

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":

    logging.info("RUNNING FULL ML PIPELINE")

    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion(
        file_path="data/raw/online_retail_II.xlsx"
    )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df, kmeans_model = build_features(train_df)
    test_df, _ = build_features(test_df, kmeans_model)

    trainer = ModelTrainer()

    best_model, best_score, report, best_threshold = trainer.initiate_model_training(
        train_df, test_df
    )

    os.makedirs("artifacts", exist_ok=True)

    # Save kmeans model (encoded_features.pkl is saved by model_trainer.py)
    joblib.dump(kmeans_model, "artifacts/kmeans.pkl")

    print("Training Complete")