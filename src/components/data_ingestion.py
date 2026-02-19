import pandas as pd
import sys
from src.logger import logging
from src.exception import CustomException


def ingest_data(file_path: str) -> pd.DataFrame:
    """
    Loads raw dataset and performs basic cleaning.
    Returns transaction-level dataframe.
    """

    try:
        logging.info("Starting data ingestion")

        df = pd.read_csv(file_path)

        # Standardize column names
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        # Convert invoice date
        df["invoicedate"] = pd.to_datetime(df["invoicedate"], errors="coerce")
        df = df.dropna(subset=["invoicedate"])

        # Drop missing customer_id
        df = df.dropna(subset=["customer_id"])
        df["customer_id"] = df["customer_id"].astype(int)


        # Remove cancelled invoices
        df = df[~df["invoice"].astype(str).str.startswith("C")]

        # Remove invalid rows
        df = df[(df["quantity"] > 0) & (df["price"] > 0)]

        # Create total price
        df["total_price"] = df["quantity"] * df["price"]

        logging.info("Data ingestion completed")

        return df

    except Exception as e:
        logging.error("Error in data ingestion")
        raise CustomException(e, sys)
