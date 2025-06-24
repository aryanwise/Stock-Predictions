import os
import pandas as pd
import ta
import numpy as np

###########################
# DATA COLLECTION, LOAD AND MAINTAIN
###########################

# Enter the path/link of where u have stored your data
historical_data_path = "Datasets/Historical Data"
sentiment_data_path = "Datasets/stock_sentiment_data.csv"

# Dictionary to store all CSV data
all_csv_data = {}  # Dictionary with filenames as keys
stock_data = {}  # Dictionary with tickers as keys

for filename in os.listdir(historical_data_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(historical_data_path, filename)

        # Store with filename key
        all_csv_data[filename] = pd.read_csv(file_path)

        # Extract ticker and store with ticker key if pattern matches
        if filename.startswith("HistoricalData_"):
            ticker = filename.split("_")[1].split(".")[0]
            stock_data[ticker] = all_csv_data[filename]  # Reference same DataFrame
            globals()[ticker] = all_csv_data[filename]  # Create global variable
            # print(f"Processed: {filename} → Ticker: {ticker}") # uncomment
        else:
            print(f"Loaded: {filename}")

# print(AAPL.head())


#############################
# DATA CLEANING
#############################


"""
Cleaning Data:
1. Removing $ sign to add consistency in all other data 
2. Changing the data type in the data from object to float for precision calculations
If u see the table below:
Data columns (total 6 columns):
 #   Column      Non-Null Count  Dtype
---  ------      --------------  -----
 0   Date        2516 non-null   object
 1   Close/Last  2516 non-null   object
 2   Volume      2516 non-null   int64
 3   Open        2516 non-null   object
 4   High        2516 non-null   object
 5   Low         2516 non-null   object
[AAPL] can be checked by pasting this line -> print(AAPL.info())
- It shows Date, Close/Last, Open, High, Low => Object, we want to cast it to Float except Date
- Additional change: Changing the col name from Close/Last to Close
"""

for ticker in [
    t for t in globals() if t.isupper() and len(t) <= 5
]:  # Finds all ticker variables
    df = globals()[ticker]

    # 1. Rename column
    df.rename(columns={"Close/Last": "Close"}, inplace=True)

    # 2. Remove $ and convert to float
    for col in ["Close", "Open", "High", "Low"]:
        df[col] = df[col].str.replace("$", "").astype(float)

    # 3. Convert Date to datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # print(f"Cleaned {ticker}") # uncomment

# print(AMZN.head())
# print(AMZN.info())


#############################
# TECHNICAL INDICATORS
#############################

""" Uncomment this later
# Trend Indicators
def add_technical_indicators(df):
    # Moving Averages - SMA & EMA
    df["SMA_50"] = df["Close"].rolling(window=50, min_periods=1).mean()
    df["SMA_200"] = df["Close"].rolling(window=200, min_periods=1).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # MACD
    macd = ta.trend.MACD(
        df["Close"], window_slow=26, window_fast=12, window_sign=9, fillna=True
    )
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    # RSI
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14, fillna=True).rsi()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2, fillna=True)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()

    return df


# Apply to all ticker DataFrames
for ticker in stock_data.keys():
    try:
        globals()[ticker] = add_technical_indicators(globals()[ticker])
        stock_data[ticker] = add_technical_indicators(stock_data[ticker])
        print(f"Added indicators to {ticker}")
    except Exception as e:
        print(f"Error processing {ticker}: {str(e)}")

# Verification
if "AAPL" in globals():
    print("\nAAPL with Technical Indicators:")
    print(
        AAPL[["Date", "Close", "SMA_50", "SMA_200", "EMA_12", "MACD", "RSI"]].head(100)
    )

import numpy as np

# Signal Columns

for ticker in stock_data.keys():
    df = globals()[ticker]

    # Moving Average Signals (Golden/Death Cross)
    df["MA_Signal"] = np.where(
        df["SMA_50"] > df["SMA_200"],
        "Golden Cross",
        np.where(df["SMA_50"] < df["SMA_200"], "Death Cross", "Neutral"),
    )

    # MACD Signals
    df["MACD_Signal"] = np.where(df["MACD"] > df["MACD_Signal"], "Bullish", "Bearish")

    print(f"Added signals to {ticker}")

# View Recent Signals
print("Recent Signals")
# For any stock (using AAPL as example)
print(AAPL[["Date", "Close", "SMA_50", "SMA_200", "MA_Signal"]].head())
print(AAPL[["Date", "Close", "MACD", "MACD_Signal", "MACD_Hist"]].head())


# Filter for specific signals
print("Specific Signals")
# Get all Golden Cross occurrences
golden_crosses = AAPL[AAPL["MA_Signal"] == "Golden Cross"]
print("Golden Cross Dates:")
print(golden_crosses[["Date", "Close"]])

# Get recent bullish MACD signals
bullish_macd = AAPL[AAPL["MACD_Signal"] == "Bullish"].head(5)
print("\nRecent Bullish MACD Signals:")
print(bullish_macd[["Date", "Close"]])

# Visualise Signals
print("Signal Visualisation")
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(AAPL["Date"], AAPL["Close"], label="Price")
plt.plot(AAPL["Date"], AAPL["SMA_50"], label="50-day SMA")
plt.plot(AAPL["Date"], AAPL["SMA_200"], label="200-day SMA")

# Highlight Golden Crosses
golden_dates = AAPL[AAPL["MA_Signal"] == "Golden Cross"]["Date"]
for date in golden_dates:
    plt.axvline(date, color="green", alpha=0.3, linestyle="--")

# Highlight Death Crosses
death_dates = AAPL[AAPL["MA_Signal"] == "Death Cross"]["Date"]
for date in death_dates:
    plt.axvline(date, color="red", alpha=0.3, linestyle="--")

plt.legend()
plt.title("AAPL Price with Moving Average Signals")
# plt.show()


# Creating Trading Signals
# Combined strategy (MA + MACD)
AAPL["Trading_Signal"] = np.where(
    (AAPL["MA_Signal"] == "Golden Cross") & (AAPL["MACD_Signal"] == "Bullish"),
    "Strong Buy",
    np.where(
        (AAPL["MA_Signal"] == "Death Cross") & (AAPL["MACD_Signal"] == "Bearish"),
        "Strong Sell",
        "Hold",
    ),
)

# View signals
print(AAPL[["Date", "Close", "MA_Signal", "MACD_Signal", "Trading_Signal"]].tail(10))
"""

""" Calculate the Indicators for each mentioned below like above Trend Indicator- These are to be added in the dashboard later
# Momentum Indicators

# Volatility Indicators

# Volume Indicators

# Support & Resistance Indicators

# Combine Indicators for Stock Recommendation
"""
