import pandas as pd
import numpy as np
import sys
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.logger import logging
from src.exception import CustomException


def build_features(df: pd.DataFrame, kmeans_model=None):

    try:
        logging.info("Starting feature engineering")

        df = df.copy()

        # ================= COLUMN CLEANING ================= #
        df.columns = df.columns.str.strip().str.lower()

        # Rename important columns
        df.rename(columns={
            "customer id": "customer_id",
            "invoicedate": "invoicedate",
            "stockcode": "stockcode",
            "invoice": "invoice",
            "price": "price",
            "quantity": "quantity"
        }, inplace=True)

        # ================= REQUIRED COLUMN CHECK ================= #
        required_cols = ["customer_id", "invoicedate", "invoice", "price", "quantity"]

        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # ================= CREATE TOTAL PRICE ================= #
        df["total_price"] = df["price"] * df["quantity"]

        # ================= DATE PROCESSING ================= #
        df["invoicedate"] = pd.to_datetime(df["invoicedate"], errors="coerce")
        df = df.dropna(subset=["invoicedate"])

        # ================= SORT ================= #
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

        rfm = rfm.merge(
            tenure[["customer_id", "tenure"]],
            on="customer_id",
            how="left"
        )

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

        # ================= CLUSTERING ================= #
        cluster_features = rfm[["frequency_log", "monetary_log", "tenure", "purchase_rate"]]
        
        # ✅ SCALE before clustering
        scaler = StandardScaler()
        cluster_features_scaled = scaler.fit_transform(cluster_features)

        if kmeans_model is None:
            kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=20)
            rfm["final_kmeans_cluster"] = kmeans_model.fit_predict(cluster_features_scaled)
        else:
            rfm["final_kmeans_cluster"] = kmeans_model.predict(cluster_features_scaled)

        # ================= REMOVE LEAKAGE ================= #
        rfm.drop(columns=["recency", "recency_log"], inplace=True)

        logging.info("Feature engineering + clustering completed")

        return rfm, kmeans_model

    except Exception as e:
        raise CustomException(e, sys)