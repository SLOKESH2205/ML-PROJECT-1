import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys
from src.logger import logging
from src.exception import CustomException


class DataTransformation:

    def __init__(self):
        self.model_features = [
            "recency_log",
            "frequency_log",
            "monetary_log",
            "tenure",
            "avg_order_value",
            "unique_items_purchased"
        ]
        self.scaler_path = "artifacts/scaler.pkl"

    def transform(self, df: pd.DataFrame):

        try:
            logging.info("Starting data transformation")

            X = df[self.model_features].fillna(0)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            os.makedirs("artifacts", exist_ok=True)
            joblib.dump(scaler, self.scaler_path)

            logging.info("Data transformation completed")

            return X_scaled

        except Exception as e:
            raise CustomException(e, sys)