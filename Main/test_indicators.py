"""
This code provides is a simple test of technical indicators and features for stock data.

1. It imports necessary classes from other modules: TechnicalIndicators, FeatureEngineer, StockDataCleaner, and StockDataCollector.

2. It defines a function quick_check that takes a DataFrame df as input, calculates technical indicators, generates features, and prints the latest signals.

3. It creates a StockDataCollector instance, collects data, cleans it using StockDataCleaner, and stores the cleaned data in `cleaned_data.

4. It extracts the data for a specific stock (AAPL in this case) from cleaned_data and passes it to the quick_check function for verification.
"""

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
