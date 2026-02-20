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

        rfm["avg_gap"] = rfm["avg_gap"].fillna(rfm["avg_gap"].max())
        rfm["avg_gap_log"] = np.log1p(rfm["avg_gap"])

        # avg order value
        rfm["avg_order_value"] = rfm["monetary"] / rfm["frequency"].replace(0, 1)
        # unique items
        items = df.groupby("customer_id")["stockcode"].nunique().reset_index()
        items.rename(columns={"stockcode": "unique_items_purchased"}, inplace=True)
        rfm = rfm.merge(items, on="customer_id", how="left")
        # log transforms
        for col in ["recency", "frequency", "monetary"]:
            rfm[f"{col}_log"] = np.log1p(rfm[col])


        logging.info("Feature engineering completed")

        return rfm

    except Exception as e:
        raise CustomException(e, sys)
