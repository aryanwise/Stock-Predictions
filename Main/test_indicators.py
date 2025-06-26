from technical_indicators import TechnicalIndicators
from feature_engineering import FeatureEngineer
from data_cleaning import StockDataCleaner
from data_collection import StockDataCollector


def quick_check(df):
    """Minimalist verification of indicators and features"""
    # Calculate indicators
    df = TechnicalIndicators.calculate_all_indicators(df)
    print("\n=== Technical Indicators ===")
    print(df[["Close", "SMA_50", "EMA_12", "RSI", "MACD", "BB_Upper", "ATR"]].tail(3))

    # Generate features
    df = FeatureEngineer.create_all_features(df)
    print("\n=== Generated Features ===")
    print(
        df[
            ["MA_Signal", "MACD_Cross", "RSI_Signal", "BB_Signal", "Composite_Signal"]
        ].head(100)
    )

    # Show latest signals
    latest = df.iloc[-1]
    print("\n=== Latest Signals ===")
    print(f"Composite: {latest['Composite_Signal']}")
    print(f"MA: {latest['MA_Signal']}, MACD: {latest['MACD_Cross']}")
    print(f"RSI: {latest['RSI_Signal']}, BB: {latest['BB_Signal']}")
    print(f"Stop Loss Long: {latest['Stop_Loss_Long']:.2f}")

    return df


collector = StockDataCollector()
collector.collect_data()
cleaned_data = StockDataCleaner.clean_all(collector)

# Check AAPL -> Change from your preferences
aapl = cleaned_data["AAPL"]
quick_check(aapl)
