import os
import pandas as pd


class StockDataCollector:
    # historical_data_path => Dataset containing historical data for different tickers
    def __init__(self, historical_data_path=None):
        try:
            self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        except NameError:
            self.base_dir = os.getcwd()

        self.historical_data_path = os.path.join(
            self.base_dir, "Datasets", "Historical Data"
        )

        if historical_data_path:
            self.historical_data_path = historical_data_path

        self.all_csv_data = {}
        self.stock_data = {}

    def collect_data(self):
        """Load all CSV files from historical data folder"""
        try:
            # Convert to absolute path and normalize
            abs_path = os.path.abspath(self.historical_data_path)
            print(f"Looking for data in: {abs_path}")

            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"Directory not found: {abs_path}")

            for filename in os.listdir(abs_path):
                if filename.endswith(".csv"):
                    file_path = os.path.join(abs_path, filename)
                    self.all_csv_data[filename] = pd.read_csv(file_path)

                    if filename.startswith("HistoricalData_"):
                        ticker = self._extract_ticker(filename)
                        self.stock_data[ticker] = self.all_csv_data[filename]

            print(f"Successfully loaded {len(self.stock_data)} tickers")

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def _extract_ticker(self, filename):
        # Helper method to extract ticker from filename
        return filename.split("_")[1].split(".")[0]

    def get_raw_data(self, filename=None):
        # Access raw data by filename
        if filename:
            return self.all_csv_data.get(filename)
        return self.all_csv_data

    def get_stock_data(self, ticker=None):
        # Access processed data by ticker
        if ticker:
            return self.stock_data.get(ticker)
        return self.stock_data
