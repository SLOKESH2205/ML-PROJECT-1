import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from src.logger import logging
from src.exception import CustomException


def build_features(df: pd.DataFrame):

    try:
        logging.info("Starting feature engineering")

        # RFM
        reference_date = df["invoicedate"].max() + pd.Timedelta(days=1)

        rfm = df.groupby("customer_id").agg({
            "invoicedate": lambda x: (reference_date - x.max()).days,
            "invoice": "nunique",
            "total_price": "sum"
        }).reset_index()

        rfm.columns = ["customer_id", "recency", "frequency", "monetary"]

        # avg gap
        df_sorted = df.sort_values(["customer_id", "invoicedate"])
        df_sorted["prev_date"] = df_sorted.groupby("customer_id")["invoicedate"].shift()
        df_sorted["gap"] = (df_sorted["invoicedate"] - df_sorted["prev_date"]).dt.days

        avg_gap = df_sorted.groupby("customer_id")["gap"].mean().reset_index()
        rfm = rfm.merge(avg_gap, on="customer_id", how="left")
        rfm.rename(columns={"gap": "avg_gap"}, inplace=True)

        # tenure
        tenure = df.groupby("customer_id")["invoicedate"].agg(min_date="min", max_date="max").reset_index()
        tenure["tenure"] = (tenure["max_date"] - tenure["min_date"]).dt.days
        rfm = rfm.merge(tenure[["customer_id", "tenure"]], on="customer_id", how="left")

        # avg order value
        rfm["avg_order_value"] = rfm["monetary"] / rfm["frequency"]

        # unique items
        items = df.groupby("customer_id")["stockcode"].nunique().reset_index()
        items.rename(columns={"stockcode": "unique_items_purchased"}, inplace=True)
        rfm = rfm.merge(items, on="customer_id", how="left")

        # log transforms
        for col in ["recency", "frequency", "monetary"]:
            rfm[f"{col}_log"] = np.log1p(rfm[col])

        # FINAL MODEL FEATURES (discipline)
        model_features = [
            "recency_log",
            "frequency_log",
            "monetary_log",
            "tenure",
            "avg_order_value",
            "unique_items_purchased"
        ]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(rfm[model_features])

        X_scaled = pd.DataFrame(X_scaled, columns=model_features)

        logging.info("Feature engineering completed")

        return rfm, X_scaled, scaler

    except Exception as e:
        raise CustomException(e, sys)
