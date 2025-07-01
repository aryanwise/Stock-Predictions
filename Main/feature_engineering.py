import numpy as np


class FeatureEngineer:
    """
    FeatureEngineer class:
    Class Methods:
    1. generate_signals(df): Generates trading signals based on technical indicators (MA, MACD, RSI, Stochastic, Bollinger Bands) and adds them as new columns to the input dataframe df.
    2. composite_signal(df): Creates a composite signal based on the RSI, Stochastic, and Bollinger Band signals, categorizing the signal as "Strong Buy", "Strong Sell", or "Neutral".
    3. risk_management(df): Calculates stop-loss levels for long and short positions based on the Average True Range (ATR) and adds them as new columns to the input dataframe df.
    4. create_all_features(cls, df): A class method that calls the above three methods in sequence to generate all features (signals and risk management) for the input dataframe df.
    """

    @staticmethod
    def generate_signals(df):
        """
        Generating trading signals based on various technical indicators and adds them as new columns to the dataframe.

        Parameters:
        df (pandas dataframe): DataFrame containing stock price data and calculated technical indicators.

        Returns:
        df: The input DataFrame with additional columns for trading signals.

        Signals generated:
        - MA_Signal: Indicates "Golden Cross" when SMA_50 > SMA_200, "Death Cross" when SMA_50 < SMA_200, and "Neutral" otherwise.
        - MACD_Cross: Indicates "Bullish" when MACD > MACD_Signal and "Bearish" otherwise.
        - RSI_Signal: Indicates "Oversold" when RSI < 30, "Overbought" when RSI > 70, and "Neutral" otherwise.
        - Stoch_Signal: Indicates "Oversold" when Stoch_%K < 20 and %K > %D, "Overbought" when Stoch_%K > 80 and %K < %D, and "Neutral" otherwise.
        - BB_Signal: Indicates "Lower Band" when Close price is less than or equal to BB_Lower, "Upper Band" when Close price is greater than or equal to BB_Upper, and "Within Bands" otherwise.
        """
        # Trading signals using Technical Indicators

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
        """
        This function takes a pandas dataframe df as input and adds a new column "Composite_Signal"
        based on the signals from the RSI, Stochastic Oscillator, and Bollinger Bands indicators.
        The composite signal is a strong buy signal when all three indicators are oversold and
        the price is at the lower band, a strong sell signal when all three are overbought and
        the price is at the upper band, and a neutral signal otherwise.

        Parameters:
        df: The dataframe containing the stock price data

        Returns:
        df: The dataframe with the added "Composite_Signal" column
        """
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
        """
        This function takes a pandas dataframe df as input and adds two new columns "Stop_Loss_Long"
        and "Stop_Loss_Short" based on the ATR (Average True Range) indicator. The stop loss levels are
        calculated as twice the ATR from the current close price.

        Parameters:
        df: The dataframe containing the stock price data

        Returns:
        df: The dataframe with the added "Stop_Loss_Long" and "Stop_Loss_Short" columns
        """
        df["Stop_Loss_Long"] = df["Close"] - 2 * df["ATR"]
        df["Stop_Loss_Short"] = df["Close"] + 2 * df["ATR"]
        return df

    @classmethod
    def create_all_features(cls, df):
        """
        This function takes a pandas dataframe df as input and generates all features used in the dashboard
        by calling the generate_signals, composite_signal, and risk_management methods.

        Parameters:
        df: The dataframe containing the stock price data

        Returns:
        df: The dataframe with all features added
        """
        df = cls.generate_signals(df)
        df = cls.composite_signal(df)
        df = cls.risk_management(df)
        return df
