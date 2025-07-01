import os
import pandas as pd


class StockDataCollector:
    """
    StockDataCollector class is created to collect and manage historical stock data from CSV files.
    Class Methods:
    1. __init__(historical_data_path=None): Initializes the class by setting the base directory and historical data path. If a custom path is provided, it overrides the default path.
    2. collect_data(): Loads all CSV files from the historical data folder, extracts the ticker symbol from each file, and stores the data in two dictionaries: all_csv_data and stock_data.
    3. extract_ticker(filename): A helper method that extracts the ticker symbol from a filename.
    4. get_raw_data(filename=None): Returns the raw data for a specific filename or all filenames if no filename is provided.
    5. get_stock_data(ticker=None): Returns the processed data for a specific ticker symbol or all ticker symbols if no ticker is provided.

    The _extract_ticker method is a created as a private method to be used internally by the class.
    """

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
        """
        This function defines a method to collect_data that loads all CSV files from a specified historical data folder, stores the data in two dictionaries (all_csv_data and stock_data), and handles potential errors.
        Specifically, it:
        1. Checks if the specified directory exists, raising a FileNotFoundError if it doesn't.
        2. Iterates through all files in the directory, loading CSV files into all_csv_data.
        3. If a CSV file starts with "HistoricalData_", it extracts the ticker symbol and stores the data in stock_data.
        4. Prints the number of successfully loaded tickers or an error message if an exception occurs.
        """
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
        """Helper method to extract ticker from filename"""
        return filename.split("_")[1].split(".")[0]

    def get_raw_data(self, filename=None):
        """Access raw data by filename"""
        if filename:
            return self.all_csv_data.get(filename)
        return self.all_csv_data

    def get_stock_data(self, ticker=None):
        """Access processed data by ticker"""
        if ticker:
            return self.stock_data.get(ticker)
        return self.stock_data
