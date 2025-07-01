import ta


class TechnicalIndicators:
    """
    The TechnicalIndicators class provides methods to calculate various technical indicators for a given stock price dataset. The class methods are:

    1. trend_indicators(df): Calculates trend-following indicators, including Simple Moving Averages (SMA) and Exponential Moving Averages (EMA), as well as the Moving Average Convergence Divergence (MACD) indicator.
    2. momentum_indicators(df): Calculates momentum indicators, including the Relative Strength Index (RSI) and the Stochastic Oscillator.
    3. volatility_indicators(df): Calculates volatility indicators, including Bollinger Bands and the Average True Range (ATR).
    4. calculate_all_indicators(cls, df): Calculates all technical indicators (trend, momentum, and volatility) and returns the resulting dataframe.

    These methods take a pandas dataframe df as input and return the modified dataframe with the calculated indicators added as new columns.
    """

    @staticmethod
    def trend_indicators(df):
        """
        This function calculates four trend-following indicators for a given stock price dataset (df):

        1. Simple Moving Averages (SMA): 50-day (SMA_50) and 200-day (SMA_200) averages of the closing price.
        2. Exponential Moving Averages (EMA): 12-day (EMA_12) and 26-day (EMA_26) averages of the closing price.
        3. Moving Average Convergence Divergence (MACD): a momentum indicator that calculates the difference between the 26-day and 12-day EMAs, along with a signal line and histogram.

        These indicators are added as new columns to the original dataframe (df).
        """
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
        """
        This function calculates two momentum indicators for a given stock price dataset (df):

        1. Relative Strength Index (RSI): measures overbought/oversold conditions using the closing price (df["Close"]) over a 14-day window.
        2. Stochastic Oscillator: compares the closing price (df["Close"]) to its price range over a 14-day window, and calculates two lines: %K (fast line) and %D (slow line).

        The results are added as new columns to the original dataframe (df): RSI, Stoch_%K, and Stoch_%D.
        """
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
        """
        This function calculates two volatility indicators for a given stock price dataset (df):

        1. Bollinger Bands: It calculates the upper and lower bands of Bollinger Bands, which are used to measure volatility and identify potential breakouts. The bands are calculated with a window size of 20 and a standard deviation of 2.
        2. Average True Range (ATR: It calculates the ATR, which measures the average range of price movements over a given period (in this case, 14 days). ATR is used to gauge market volatility.

        Both indicators are added as new columns to the original dataframe (df).
        """
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
        """
        This function defines calculates and returns all technical indicators for a given pandas dataframe df.
        The function calls the earlier three created methods: trend_indicators, momentum_indicators, and volatility_indicators, each of which adds new columns to the dataframe with the corresponding indicators.
        """
        df = cls.trend_indicators(df)
        df = cls.momentum_indicators(df)
        df = cls.volatility_indicators(df)
        return df
