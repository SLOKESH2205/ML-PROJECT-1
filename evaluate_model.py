import pandas as pd

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split

from src.preprocessing import build_customer_features


def main():
    # Load the existing transaction-level dataset, then convert it to customer-level features.
    raw_df = pd.read_csv("artifacts/test.csv")
    customer_df, _ = build_customer_features(raw_df)
    customer_df = customer_df.fillna(0)

    target_col = "retention_status"

    X = customer_df.drop(columns=[target_col, "customer_id"])
    y = customer_df[target_col]

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(X)

    print("\n===== CLUSTERING METRICS =====")
    sil_score = silhouette_score(X, cluster_labels)
    db_index = davies_bouldin_score(X, cluster_labels)
    print(f"Silhouette Score: {sil_score:.4f}")
    print(f"Davies-Bouldin Index: {db_index:.4f}")
    print(f"Inertia: {kmeans.inertia_:.4f}")

    X = X.copy()
    X["cluster"] = cluster_labels

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n===== CLASSIFICATION METRICS =====")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
