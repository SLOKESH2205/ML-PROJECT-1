import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.feature_engineering import build_features
from src.components.model_trainer import ModelTrainer


if __name__ == "__main__":

    print("RUNNING FULL ML PIPELINE")

    # INGESTION
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion(
        file_path="data/raw/online_retail_II.xlsx"
    )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # FEATURE ENGINEERING (IMPORTANT PART)
    train_df, kmeans_model = build_features(train_df)
    test_df, _ = build_features(test_df, kmeans_model)

    print("Feature Engineering + Clustering Done")

    # MODEL TRAINING
    trainer = ModelTrainer()

    best_model, best_score, report = trainer.initiate_model_training(
        train_df,
        test_df
    )

    print("\nBEST MODEL:", best_model)
    print("BEST F1 SCORE:", best_score)

    print("\nMODEL COMPARISON:")
    for model, scores in report.items():
        print(model, ":", scores)