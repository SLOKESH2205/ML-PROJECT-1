import joblib
import pandas as pd
from src.components.feature_engineering import build_features


class PredictPipeline:

    def __init__(self):

        loaded_obj = joblib.load("artifacts/model.pkl")

        # Handle both cases
        if isinstance(loaded_obj, dict):
            self.model = loaded_obj["model"]
            self.threshold = loaded_obj["threshold"]
        else:
            self.model = loaded_obj
            self.threshold = 0.5

        self.kmeans = joblib.load("artifacts/kmeans.pkl")
        self.scaler = joblib.load("artifacts/model_scaler.pkl")
        
        # Load the encoded feature columns (try new format first, then fallback to old)
        try:
            self.train_columns = joblib.load("artifacts/encoded_features.pkl")
        except FileNotFoundError:
            print("⚠️  encoded_features.pkl not found. Using features.pkl (old format)")
            self.train_columns = joblib.load("artifacts/features.pkl")

    def preprocess(self, df):

        # ✅ STEP 0: Store customer IDs before any transformations
        if "customer_id" in df.columns:
            self.last_customer_ids = df["customer_id"].values
        else:
            self.last_customer_ids = None
        
        # ✅ STEP 1: feature engineering
        df, _ = build_features(df, self.kmeans)

        # ✅ STEP 2: SELECT ONLY TRAINING FEATURES (BEFORE ENCODING)
        df = df[
            [
                "frequency_log",
                "monetary_log",
                "tenure",
                "avg_order_value",
                "unique_items_purchased",
                "purchase_rate",
                "monetary_per_day",
                "final_kmeans_cluster",
            ]
        ]

        # 🔥 STORE RAW FEATURES BEFORE SCALING (FOR CLUSTER PROFILING)
        self.raw_feature_df = df[[
            "frequency_log",
            "monetary_log",
            "tenure",
            "avg_order_value",
            "purchase_rate",
            "final_kmeans_cluster"
        ]].copy()

        # 🔥 STORE CLUSTER BEFORE ENCODING (FOR RESULT DF)
        self.last_cluster = df["final_kmeans_cluster"].values

        # ✅ STEP 3: ONE HOT ENCODING (CRITICAL)
        df = pd.get_dummies(
            df,
            columns=["final_kmeans_cluster"],
            prefix="cluster",
            drop_first=True
        )
        
        # ✅ STEP 4: FORCE SAME COLUMNS AS TRAINING
        df = df.reindex(columns=self.train_columns, fill_value=0)
        
        # 🔥 STORE PROCESSED FEATURES FOR SHAP
        self.last_processed_df = df.copy()
        self.features = df.columns.tolist()

        # 🚨 DEBUG PRINT (remove later)
        print("PREDICT SHAPE:", df.shape)

        # ✅ STEP 5: SCALE
        df = self.scaler.transform(df)

        return df

    def predict(self, df):

        df_scaled = self.preprocess(df)

        probs = self.model.predict_proba(df_scaled)[:, 1]
        preds = (probs >= self.threshold).astype(int)
        
        # Return complete result dataframe
        if self.last_customer_ids is not None:
            result_df = pd.DataFrame({
                "customer_id": self.last_customer_ids,
                "Churn_Label": preds,
                "Churn Probability": probs
            })
        else:
            result_df = pd.DataFrame({
                "Churn_Label": preds,
                "Churn Probability": probs
            })
        
        # 🔥 ATTACH CLUSTER INFO
        if hasattr(self, "last_cluster"):
            result_df["cluster"] = self.last_cluster
        
        # 🔥 ATTACH RAW FEATURES FOR CLUSTER PROFILING
        if hasattr(self, "raw_feature_df"):
            result_df = result_df.merge(
                self.raw_feature_df,
                left_index=True,
                right_index=True,
                how="left"
            )
        
        return result_df