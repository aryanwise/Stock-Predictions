import pandas as pd
import numpy as np
import ta


class TechnicalIndicators:
    @staticmethod
    def trend_indicators(df):
        """Calculate trend-following indicators"""
        # Moving Averages
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

        return df

    @staticmethod
    def momentum_indicators(df):
        """Calculate momentum indicators"""
        # RSI
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14, fillna=True).rsi()

        # Stochastic
        stochastic = ta.momentum.StochasticOscillator(
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            window=14,
            smooth_window=3,
            fillna=True,
        )
        df["Stoch_%K"] = stochastic.stoch()
        df["Stoch_%D"] = stochastic.stoch_signal()

        return df

    @staticmethod
    def volatility_indicators(df):
        """Calculate volatility indicators"""
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(
            df["Close"], window=20, window_dev=2, fillna=True
        )
        df["BB_Upper"] = bb.bollinger_hband()
        df["BB_Lower"] = bb.bollinger_lband()

        # ATR
        df["ATR"] = ta.volatility.AverageTrueRange(
            high=df["High"], low=df["Low"], close=df["Close"], window=14, fillna=True
        ).average_true_range()

        return df

    @classmethod
    def calculate_all_indicators(cls, df):
        """Calculate all technical indicators"""
        df = cls.trend_indicators(df)
        df = cls.momentum_indicators(df)
        df = cls.volatility_indicators(df)
        return df
