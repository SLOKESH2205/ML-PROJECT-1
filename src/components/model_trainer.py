import os
import sys
import joblib
import logging
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

from src.exception import CustomException


class ModelTrainer:

    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.scaler_path = os.path.join("artifacts", "model_scaler.pkl")

    def evaluate_model(self, y_true, y_pred, y_prob):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_prob)
        }

    def initiate_model_training(self, train_df, test_df):
        try:
            logging.info("Starting model training")

            # ================= FEATURES ================= #
            feature_cols = [
                "frequency_log",
                "monetary_log",
                "tenure",
                "avg_order_value",
                "unique_items_purchased",
                "purchase_rate",
                "monetary_per_day",
                "final_kmeans_cluster"
            ]

            target_col = "retention_status"

            # ================= TRAIN DATA ================= #
            X_train = train_df[feature_cols]
            y_train = train_df[target_col]

            # ================= TEST DATA ================= #
            X_test = test_df[feature_cols]
            y_test = test_df[target_col]

            # ================= ONE HOT ENCODING ================= #
            X_train = pd.get_dummies(
                X_train,
                columns=["final_kmeans_cluster"],
                prefix="cluster",
                drop_first=True
            )

            X_test = pd.get_dummies(
                X_test,
                columns=["final_kmeans_cluster"],
                prefix="cluster",
                drop_first=True
            )

            # Align columns
            X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

            # ================= SCALING ================= #
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            os.makedirs("artifacts", exist_ok=True)
            joblib.dump(scaler, self.scaler_path)

            # ================= CLASS IMBALANCE ================= #
            scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

            # ================= MODELS ================= #
            models = {
                "Logistic Regression": (
                    LogisticRegression(solver='liblinear', class_weight='balanced'),
                    {"C": [0.01, 0.1, 1, 10]}   # ✅ removed penalty (fix warning)
                ),
                "Random Forest": (
                    RandomForestClassifier(class_weight='balanced', random_state=42),
                    {"n_estimators": [100, 200], "max_depth": [10, None], "min_samples_split": [2, 5]}
                ),
                "XGBoost": (
                    XGBClassifier(
                        eval_metric='logloss',
                        random_state=42,
                        scale_pos_weight=scale_pos_weight
                    ),
                    {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1], "max_depth": [3, 5]}
                )
            }

            best_model = None
            best_score = 0
            best_model_name = None
            report = {}

            # ================= TRAIN LOOP ================= #
            for name, (model, params) in models.items():

                logging.info(f"Training {name}")

                grid = GridSearchCV(
                    model,
                    params,
                    cv=3,
                    scoring='f1',
                    n_jobs=-1
                )

                grid.fit(X_train, y_train)
                best_estimator = grid.best_estimator_

                # ================= THRESHOLD TUNING ================= #
                y_prob = best_estimator.predict_proba(X_test)[:, 1]
                y_pred = (y_prob > 0.4).astype(int)

                scores = self.evaluate_model(y_test, y_pred, y_prob)
                report[name] = scores

                logging.info(f"{name}: {scores}")

                if scores["f1_score"] > best_score:
                    best_score = scores["f1_score"]
                    best_model = best_estimator
                    best_model_name = name

            # ================= SAVE BEST MODEL ================= #
            joblib.dump(best_model, self.model_path)

            logging.info(f"Best Model: {best_model_name}")

            return best_model_name, best_score, report

        except Exception as e:
            raise CustomException(e, sys)