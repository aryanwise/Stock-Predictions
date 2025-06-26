import pandas as pd


class StockDataCleaner:
    @staticmethod
    def clean_data(df, sort_descending=True):
        """
        Clean and standardize stock data DataFrame
        Args:
            df: Raw stock data DataFrame
        Returns:
            Cleaned DataFrame
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
        Args:
            data_collector: StockDataCollector instance
        Returns:
            Dict of cleaned DataFrames
        """
        cleaned_data = {}
        for ticker, df in data_collector.get_stock_data().items():
            cleaned_data[ticker] = StockDataCleaner.clean_data(df, sort_descending)
        return cleaned_data
