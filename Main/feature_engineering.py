import pandas as pd
import numpy as np


class FeatureEngineer:
    @staticmethod
    def generate_signals(df):
        # Trading singlas from Technical Indicators @technical_indicators.py
        # MA Signals
        df["MA_Signal"] = np.where(
            df["SMA_50"] > df["SMA_200"],
            "Golden Cross",
            np.where(df["SMA_50"] < df["SMA_200"], "Death Cross", "Neutral"),
        )

        # MACD Signals
        df["MACD_Cross"] = np.where(
            df["MACD"] > df["MACD_Signal"], "Bullish", "Bearish"
        )

        # RSI Signals
        df["RSI_Signal"] = np.select(
            [df["RSI"] < 30, df["RSI"] > 70],
            ["Oversold", "Overbought"],
            default="Neutral",
        )

        # Stochastic Signals
        df["Stoch_Signal"] = np.select(
            [
                (df["Stoch_%K"] < 20) & (df["Stoch_%K"] > df["Stoch_%D"]),
                (df["Stoch_%K"] > 80) & (df["Stoch_%K"] < df["Stoch_%D"]),
            ],
            ["Oversold", "Overbought"],
            default="Neutral",
        )

        # Bollinger Band Signals
        df["BB_Signal"] = np.select(
            [df["Close"] <= df["BB_Lower"], df["Close"] >= df["BB_Upper"]],
            ["Lower Band", "Upper Band"],
            default="Within Bands",
        )

        return df

    @staticmethod
    def composite_signal(df):
        df["Composite_Signal"] = np.where(
            (df["RSI_Signal"] == "Oversold")
            & (df["Stoch_Signal"] == "Oversold")
            & (df["BB_Signal"] == "Lower Band"),
            "Strong Buy",
            np.where(
                (df["RSI_Signal"] == "Overbought")
                & (df["Stoch_Signal"] == "Overbought")
                & (df["BB_Signal"] == "Upper Band"),
                "Strong Sell",
                "Neutral",
            ),
        )
        return df

    @staticmethod
    def risk_management(df):
        df["Stop_Loss_Long"] = df["Close"] - 2 * df["ATR"]
        df["Stop_Loss_Short"] = df["Close"] + 2 * df["ATR"]
        return df

    @classmethod
    def create_all_features(cls, df):
        df = cls.generate_signals(df)
        df = cls.composite_signal(df)
        df = cls.risk_management(df)
        return df
