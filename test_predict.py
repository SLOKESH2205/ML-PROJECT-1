import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline

if __name__ == "__main__":

    df = pd.read_csv("artifacts/test.csv")

    pipe = PredictPipeline()

    preds, probs = pipe.predict(df)

    print("Predictions:", preds[:10])
    print("Probabilities:", probs[:10])