from src.components.data_transformation import DataTransformation

if __name__ == "__main__":
    print("RUNNING DATA TRANSFORMATION")

    obj = DataTransformation()

    X_train, X_test, y_train, y_test, scaler_path = obj.initiate_data_transformation(
        train_path="artifacts/train.csv",
        test_path="artifacts/test.csv"
    )

    print("Transformation completed")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("Scaler saved at:", scaler_path)