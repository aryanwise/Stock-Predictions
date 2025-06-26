from data_collection import StockDataCollector
from data_cleaning import StockDataCleaner

# 1. Collect data
collector = StockDataCollector()
collector.collect_data()

# 2. Clean data
cleaner = StockDataCleaner()
cleaned_data = cleaner.clean_all(collector)  # Returns dict of cleaned DataFrames
print(f"Cleaned Data: \n{cleaned_data}\n")
# Access individual stocks
aapl_data = cleaned_data["AAPL"]
print(f"Apple Stock: \n{aapl_data}")
