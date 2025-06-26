from data_collection import StockDataCollector
from data_cleaning import StockDataCleaner
from technical_indicators import TechnicalIndicators
from feature_engineering import FeatureEngineer
from train_xgboost import StockPredicor
import joblib


def main():
    # 1. Load and preprocess data
    collector = StockDataCollector()
    collector.collect_data()
    cleaned_data = StockDataCleaner.clean_all(collector)
    aapl = cleaned_data["AAPL"].copy()

    # 2. Features Engineering
    df = TechnicalIndicators.calculate_all_indicators(aapl)
    df = FeatureEngineer().lagged_features(df)
    df = FeatureEngineer().rolling_stats(df)
    df = FeatureEngineer().prepare_target(df)

    # 3. Prepare train/test data
    X = df.drop(["Target", "Data"], axis=1, errors="ignore")
    y = df["Target"]

    # 4. Train Model
    predictor = StockPredicor()
    predictor.walk_forward_train(X, y)
    predictor.evaluate(X, y)

    # 5. Save model and latest features
    # joblib.dump(predictor.model, "xgboost_stock_model.pkl")
    # X.iloc[-1:].to_pickle("latest_features.pkl")

    # 6. Make prediction
    latest = X.iloc[-1:].copy()
    numeric_pred = predictor.model.predict(latest)[0]
    text_pred = predictor.class_map[numeric_pred]
    print(f"\nNext 5-day prediction: {text_pred}")


if __name__ == "__main__":
    main()
