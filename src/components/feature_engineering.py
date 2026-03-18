import pandas as pd
import numpy as np
import sys
from sklearn.cluster import KMeans

from src.logger import logging
from src.exception import CustomException


def build_features(df: pd.DataFrame, kmeans_model=None):

    try:
        logging.info("Starting feature engineering")

        df = df.copy()

        df["invoicedate"] = pd.to_datetime(df["invoicedate"], errors="coerce")
        df = df.dropna(subset=["invoicedate"])

        df = df.sort_values(["customer_id", "invoicedate"])

        # ================= RFM ================= #
        reference_date = df["invoicedate"].max() + pd.Timedelta(days=1)

        rfm = df.groupby("customer_id").agg({
            "invoicedate": lambda x: (reference_date - x.max()).days,
            "invoice": "nunique",
            "total_price": "sum"
        }).reset_index()

        rfm.columns = ["customer_id", "recency", "frequency", "monetary"]

        # ================= TENURE ================= #
        tenure = df.groupby("customer_id")["invoicedate"].agg(
            min_date="min",
            max_date="max"
        ).reset_index()

        tenure["tenure"] = (tenure["max_date"] - tenure["min_date"]).dt.days
        rfm = rfm.merge(tenure[["customer_id", "tenure"]], on="customer_id", how="left")

        # ================= DERIVED FEATURES ================= #
        rfm["avg_order_value"] = rfm["monetary"] / rfm["frequency"].replace(0, 1)

        items = df.groupby("customer_id")["stockcode"].nunique().reset_index()
        items.rename(columns={"stockcode": "unique_items_purchased"}, inplace=True)
        rfm = rfm.merge(items, on="customer_id", how="left")

        # ================= NEW FEATURES ================= #
        rfm["purchase_rate"] = rfm["frequency"] / (rfm["tenure"] + 1)
        rfm["monetary_per_day"] = rfm["monetary"] / (rfm["tenure"] + 1)

        # ================= LOG FEATURES ================= #
        for col in ["recency", "frequency", "monetary"]:
            rfm[f"{col}_log"] = np.log1p(rfm[col])

        # ================= TARGET ================= #
        rfm["retention_status"] = (rfm["recency"] <= 90).astype(int)

        # ================= CLUSTERING (NO RECENCY) ================= #
        cluster_features = rfm[["frequency_log", "monetary_log"]]

        if kmeans_model is None:
            kmeans_model = KMeans(n_clusters=2, random_state=42)
            rfm["final_kmeans_cluster"] = kmeans_model.fit_predict(cluster_features)
        else:
            rfm["final_kmeans_cluster"] = kmeans_model.predict(cluster_features)

        # 🔥 REMOVE LEAKAGE FEATURES COMPLETELY
        rfm.drop(columns=["recency", "recency_log"], inplace=True)

        logging.info("Feature engineering + clustering completed")

        return rfm, kmeans_model

    except Exception as e:
        raise CustomException(e, sys)