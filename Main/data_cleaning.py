import pandas as pd


class StockDataCleaner:
    """
    StockDataCleaner class
    Class Methods:
    1. clean_data(df, sort_descending=True):
        - cleans and standardizes a single stock data DataFrame by renaming columns, cleaning numeric columns, and converting/sorting dates.
        - Returns the cleaned DataFrame.
    2. clean_all(data_collector, sort_descending=True):
        - Cleans all stock data from a StockDataCollector instance by applying clean_data to each ticker's DataFrame.
        - Returns a dictionary of cleaned DataFrames, keyed by ticker symbol.
    """

    @staticmethod
    def clean_data(df, sort_descending=True):
        """
        Clean and standardize stock data DataFrame
        """
        df = df.copy()

        # 1. Rename columns
        df.rename(columns={"Close/Last": "Close"}, inplace=True)

        # 2. Clean numeric columns
        price_cols = ["Close", "Open", "High", "Low"]
        for col in price_cols:
            if col in df.columns and df[col].dtype == object:
                df[col] = df[col].str.replace(r"[^\d.]", "", regex=True).astype(float)

        # 3. Convert and sort dates
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df.sort_values("Date", ascending=not sort_descending, inplace=True)
            df.reset_index(drop=True, inplace=True)

        return df

    @staticmethod
    def clean_all(data_collector, sort_descending=True):
        """
        Clean all data from a StockDataCollector instance
        """
        cleaned_data = {}
        for ticker, df in data_collector.get_stock_data().items():
            cleaned_data[ticker] = StockDataCleaner.clean_data(df, sort_descending)
        return cleaned_data
