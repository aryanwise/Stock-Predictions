"""
This code provides with a simple Streamlit dashboard for stock analysis and recommendation based on sentiment scores.

It allows users to select a stock ticker, view its historical price chart, technical indicators, and receive recommendations based on technical analysis and sentiment analysis.
The dashboard also provides a summary of all tickers, including their mean close price, standard deviation, minimum and maximum close price, mean volume, mean RSI, and mean ATR.

The dashboard uses various libraries, including Streamlit, Pandas, Plotly, and NumPy, to load and process historical stock data, calculate technical indicators, and generate visualizations.
It also uses a custom get_recommendation function to generate recommendations based on technical analysis and sentiment analysis.

The dashboard is divided into several tabs, including:

1. Price Chart: Displays the historical price chart of the selected stock.
2. Technical Analysis: Displays various technical indicators, such as moving averages, RSI, and Bollinger Bands.
3. Candlestick Chart: Displays a candlestick chart of the selected stock.
4. Recommendations: Provides a recommendation based on technical analysis and sentiment analysis.
5. All Tickers: Provides a summary of all tickers, including their mean close price, standard deviation, minimum and maximum close price, mean volume, mean RSI, and mean ATR.
6. Summary Statistics: Provides a summary of the selected stock's statistics, including its mean close price, standard deviation, minimum and maximum close price, mean volume, mean RSI, and mean ATR.

The main idea of this dashboard is to provide users with a range of tools and visualizations to help them make informed investment decisions.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from technical_indicators import TechnicalIndicators
from feature_engineering import FeatureEngineer
from data_collection import StockDataCollector
from data_cleaning import StockDataCleaner
import numpy as np
from datetime import datetime
import os
import io

st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
st.title("📈 Stock Analysis & Recommendation Dashboard")

base_dir = os.path.dirname(os.path.abspath(__file__))
historical_data_dir = os.path.join(base_dir, "..", "Datasets", "Historical Data")
sentiment_data_path = os.path.join(
    base_dir, "..", "Datasets", "Stock Suggestion", "top5_stock_recommendations.csv"
)


@st.cache_data
def load_data():
    data = {}
    collector = StockDataCollector(historical_data_path=historical_data_dir)

    try:
        collector.collect_data()
        data = StockDataCleaner.clean_all(collector)
        if not data:
            st.error("No valid ticker data loaded.")
            return data

        # Apply technical indicators and features to all tickers
        valid_data = {}
        required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        for ticker, df in data.items():
            if not all(col in df.columns for col in required_cols):
                st.warning(
                    f"Skipping {ticker}: Missing required columns {', '.join(required_cols)}"
                )
                continue
            try:
                df = TechnicalIndicators.calculate_all_indicators(df)
                df = FeatureEngineer.create_all_features(df)
                if df.empty:
                    st.warning(f"Skipping {ticker}: Empty DataFrame after processing.")
                    continue
                valid_data[ticker] = df
            except Exception as e:
                st.warning(f"Failed to process {ticker}: {str(e)}")
                continue

        data = valid_data
        if not data:
            st.error("No valid ticker data after processing.")
    except Exception as e:
        st.error(f"Error loading historical data: {str(e)}")
        return data

    return data


data = load_data()
if not data:
    st.stop()

tickers = sorted(list(data.keys()))
if not tickers:
    st.error(
        "No tickers loaded. Check the historical data directory and CSV column names."
    )
    st.stop()


# Generate random sentiment data for specified tickers
@st.cache_data
def load_sentiment_data(tickers):
    np.random.seed(42)
    sentiment_data = {}
    for ticker in tickers:
        score = np.random.uniform(-1, 1)
        if ticker in ["AAPL", "META", "NFLX", "GOOGL", "AMZN"]:
            if score > 0.5:
                sentiment_data[ticker] = (
                    score,
                    f"Due to rising hype in {ticker} stocks, they are likely to increase - Strong Buy",
                )
            elif score < -0.5:
                sentiment_data[ticker] = (
                    score,
                    f"Due to declining sentiment in {ticker} stocks, they may face downward pressure - Strong Sell",
                )
            else:
                sentiment_data[ticker] = (
                    score,
                    f"{ticker} stocks are showing neutral sentiment - Hold",
                )
        else:
            sentiment_data[ticker] = (
                score,
                f"{ticker} sentiment analysis not available - Hold",
            )
    return sentiment_data


sentiment_data = load_sentiment_data(tickers)

# Sidebar for stock selection
st.sidebar.header("Stock Selection")
selected_ticker = st.sidebar.selectbox("Select Ticker", tickers)
start_date = st.sidebar.date_input("Start Date", value=datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.now())

# Process selected ticker data
df = data[selected_ticker].copy()
df = df[
    (df["Date"] >= pd.to_datetime(start_date))
    & (df["Date"] <= pd.to_datetime(end_date))
]

if df.empty:
    st.warning("No data available for the selected date range.")
    st.stop()


# Recommendation logic
def get_recommendation(latest_data, sentiment_score, atr_threshold):
    tech_score = 0
    reasons = []
    signal_contributions = {
        "Composite": 0,
        "MA": 0,
        "MACD": 0,
        "RSI": 0,
        "Stochastic": 0,
        "Sentiment": sentiment_score * 0.4,
    }

    if latest_data["Composite_Signal"] == "Strong Buy":
        tech_score += 1
        signal_contributions["Composite"] = 1
        reasons.append("Strong buy signal from combined indicators")
    elif latest_data["Composite_Signal"] == "Strong Sell":
        tech_score -= 1
        signal_contributions["Composite"] = -1
        reasons.append("Strong sell signal from combined indicators")

    if latest_data["MA_Signal"] == "Golden Cross":
        tech_score += 0.5
        signal_contributions["MA"] = 0.5
        reasons.append("Golden Cross in moving averages")
    elif latest_data["MA_Signal"] == "Death Cross":
        tech_score -= 0.5
        signal_contributions["MA"] = -0.5
        reasons.append("Death Cross in moving averages")

    if latest_data["MACD_Cross"] == "Bullish":
        tech_score += 0.4
        signal_contributions["MACD"] = 0.4
        reasons.append("Bullish MACD crossover")
    elif latest_data["MACD_Cross"] == "Bearish":
        tech_score -= 0.4
        signal_contributions["MACD"] = -0.4
        reasons.append("Bearish MACD crossover")

    if latest_data["RSI"] < 30:
        tech_score += 0.3
        signal_contributions["RSI"] = 0.3
        reasons.append("RSI indicates oversold condition")
    elif latest_data["RSI"] > 70:
        tech_score -= 0.3
        signal_contributions["RSI"] = -0.3
        reasons.append("RSI indicates overbought condition")

    if (
        latest_data["Stoch_%K"] > latest_data["Stoch_%D"]
        and latest_data["Stoch_%K"] < 20
    ):
        tech_score += 0.2
        signal_contributions["Stochastic"] = 0.2
        reasons.append("Stochastic oscillator suggests buying opportunity")
    elif (
        latest_data["Stoch_%K"] < latest_data["Stoch_%D"]
        and latest_data["Stoch_%K"] > 80
    ):
        tech_score -= 0.2
        signal_contributions["Stochastic"] = -0.2
        reasons.append("Stochastic oscillator suggests selling pressure")

    final_score = tech_score * 0.6 + sentiment_score * 0.4
    if latest_data["ATR"] > atr_threshold:
        reasons.append(
            f"High volatility (ATR: {latest_data['ATR']:.2f} > {atr_threshold:.2f})"
        )

    if final_score > 0.8 and latest_data["ATR"] <= atr_threshold:
        recommendation = "Strong Buy"
    elif final_score > 0.3:
        recommendation = "Buy"
    elif final_score < -0.8 and latest_data["ATR"] <= atr_threshold:
        recommendation = "Strong Sell"
    elif final_score < -0.3:
        recommendation = "Sell"
    else:
        recommendation = "Hold"

    return recommendation, reasons, final_score, signal_contributions


# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Price Chart",
        "Technical Analysis",
        "Candlestick Chart",
        "Recommendations",
        "All Tickers",
        "Summary Statistics",
    ]
)

with tab1:
    st.subheader(f"{selected_ticker} Stock Price")
    fig_price = px.line(
        df, x="Date", y="Close", title=f"{selected_ticker} Closing Price"
    )
    fig_price.update_layout(template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig_price, use_container_width=True)

with tab2:
    st.subheader("Technical Indicators")
    indicators = [
        "SMA_50",
        "SMA_200",
        "EMA_12",
        "EMA_26",
        "MACD",
        "MACD_Signal",
        "MACD_Hist",
        "RSI",
        "Stoch_%K",
        "Stoch_%D",
        "BB_Upper",
        "BB_Lower",
        "ATR",
    ]
    selected_indicators = st.multiselect(
        "Select Indicators", indicators, default=["SMA_50", "EMA_12", "RSI"]
    )

    fig_tech = go.Figure()
    fig_tech.add_trace(
        go.Scatter(x=df["Date"], y=df["Close"], name="Close", line=dict(color="white"))
    )
    palette = {
        "SMA_50": "yellow",
        "SMA_200": "gold",
        "EMA_12": "orange",
        "EMA_26": "darkorange",
        "MACD": "cyan",
        "MACD_Signal": "purple",
        "MACD_Hist": "lightskyblue",
        "RSI": "lime",
        "Stoch_%K": "magenta",
        "Stoch_%D": "pink",
        "BB_Upper": "royalblue",
        "BB_Lower": "steelblue",
        "ATR": "red",
    }

    for ind in selected_indicators:
        if ind in df.columns:
            linestyle = "dash" if ind == "BB_Lower" else "solid"
            fig_tech.add_trace(
                go.Scatter(
                    x=df["Date"],
                    y=df[ind],
                    name=ind,
                    line=dict(color=palette[ind], dash=linestyle),
                )
            )
        else:
            st.warning(f"Indicator {ind} not found.")

    fig_tech.update_layout(
        title="Technical Indicators",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified",
    )
    st.plotly_chart(fig_tech, use_container_width=True)

with tab3:
    st.subheader("Candlestick Chart")
    fig_candle = go.Figure(
        data=[
            go.Candlestick(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="OHLC",
            ),
            go.Bar(
                x=df["Date"],
                y=df["Volume"],
                name="Volume",
                marker_color="rgba(128, 128, 128, 0.3)",
                yaxis="y2",
            ),
        ]
    )
    fig_candle.update_layout(
        title=f"{selected_ticker} Candlestick Chart",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Price",
        yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
        hovermode="x unified",
    )
    st.plotly_chart(fig_candle, use_container_width=True)

with tab4:
    st.subheader("Stock Recommendations")
    latest_data = df.iloc[-1]
    sentiment_score, sentiment_text = sentiment_data.get(
        selected_ticker, (0, "No sentiment data available - Hold")
    )
    atr_threshold = df["ATR"].quantile(0.75) if "ATR" in df else 1.0
    recommendation, reasons, final_score, signal_contributions = get_recommendation(
        latest_data, sentiment_score, atr_threshold
    )

    st.write(f"**Recommendation for {selected_ticker}: {recommendation}**")
    st.write(f"**Sentiment Analysis**: {sentiment_text}")
    st.write("**Technical Reasons:**")
    for reason in reasons:
        st.write(f"- {reason}")
    st.write(f"**Combined Score:** {final_score:.2f}")

    st.subheader("Signal Contributions")
    fig_signals = go.Figure(
        data=[
            go.Bar(
                x=list(signal_contributions.keys()),
                y=list(signal_contributions.values()),
                marker_color=[
                    "#00CC96" if v > 0 else "#EF553B"
                    for v in signal_contributions.values()
                ],
            )
        ]
    )
    fig_signals.update_layout(
        title="Signal Contributions to Recommendation",
        template="plotly_dark",
        xaxis_title="Signal",
        yaxis_title="Score Contribution",
    )
    st.plotly_chart(fig_signals, use_container_width=True)

    st.subheader("Top Picks")
    try:
        top_picks_df = pd.read_csv(sentiment_data_path)
        required_cols = [
            "Ticker",
            "Sentiment Score",
            "Confidence",
            "Positive",
            "Negative",
            "Count",
        ]
        if not all(col in top_picks_df.columns for col in required_cols):
            st.warning(
                f"top5_stock_recommendations.csv is missing required columns: {', '.join(required_cols)}"
            )
        else:
            top_picks_df = top_picks_df[required_cols].sort_values(
                by=["Sentiment Score", "Confidence"], ascending=False
            )
            top_picks_df = top_picks_df.round(2)

            def color_sentiment(val):
                if val >= 0.8:
                    return "color: #00CC96"
                elif val <= 0.2:
                    return "color: #EF553B"
                else:
                    return "color: #FFFFFF"

            st.dataframe(
                top_picks_df.style.applymap(
                    color_sentiment, subset=["Sentiment Score"]
                ),
                use_container_width=True,
            )
    except FileNotFoundError:
        st.warning(
            "top5_stock_recommendations.csv not found in Datasets/Stock Suggestion/."
        )
    except Exception as e:
        st.warning(f"Failed to load top5_stock_recommendations.csv: {e}")

with tab5:
    st.subheader("All Tickers Summary")
    st.write("Filter by:")
    filter_rec = st.multiselect(
        "Recommendation",
        ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"],
        default=["Strong Buy", "Buy"],
    )
    score_threshold = st.slider("Minimum Score", -2.0, 2.0, 0.0)

    all_tickers = []
    for ticker in tickers:
        ticker_df = data[ticker].copy()
        ticker_df = ticker_df[
            (ticker_df["Date"] >= pd.to_datetime(start_date))
            & (ticker_df["Date"] <= pd.to_datetime(end_date))
        ]
        if ticker_df.empty:
            st.warning(f"No data for {ticker} in the selected date range.")
            continue
        latest = ticker_df.iloc[-1]
        atr_thresh = ticker_df["ATR"].quantile(0.75) if "ATR" in ticker_df else 1.0
        sentiment_score, sentiment_text = sentiment_data.get(
            ticker, (0, "No sentiment data available - Hold")
        )
        rec, _, score, _ = get_recommendation(latest, sentiment_score, atr_thresh)
        all_tickers.append(
            {
                "Ticker": ticker,
                "Close": latest["Close"],
                "RSI": latest["RSI"],
                "ATR": latest["ATR"],
                "Sentiment": sentiment_score,
                "Sentiment Analysis": sentiment_text,
                "Score": score,
                "Recommendation": rec,
            }
        )

    all_tickers_df = pd.DataFrame(all_tickers)
    all_tickers_df["Close"] = all_tickers_df["Close"].round(2)
    all_tickers_df["RSI"] = all_tickers_df["RSI"].round(2)
    all_tickers_df["ATR"] = all_tickers_df["ATR"].round(2)
    all_tickers_df["Sentiment"] = all_tickers_df["Sentiment"].round(2)
    all_tickers_df["Score"] = all_tickers_df["Score"].round(2)

    filtered_df = all_tickers_df[
        all_tickers_df["Recommendation"].isin(filter_rec)
        & (all_tickers_df["Score"] >= score_threshold)
    ]

    def color_recommendation(val):
        color = {
            "Strong Buy": "#00CC96",
            "Buy": "#66CC66",
            "Hold": "#FFFFFF",
            "Sell": "#FF9999",
            "Strong Sell": "#EF553B",
        }
        return f'color: {color.get(val, "#FFFFFF")}'

    st.dataframe(
        filtered_df.style.applymap(color_recommendation, subset=["Recommendation"]),
        use_container_width=True,
    )

    # Export button
    csv_buffer = io.StringIO()
    filtered_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Export All Tickers to CSV",
        data=csv_buffer.getvalue(),
        file_name="all_tickers_summary.csv",
        mime="text/csv",
    )

with tab6:
    st.subheader("Summary Statistics")
    stats = []
    for ticker in tickers:
        ticker_df = data[ticker].copy()
        ticker_df = ticker_df[
            (ticker_df["Date"] >= pd.to_datetime(start_date))
            & (ticker_df["Date"] <= pd.to_datetime(end_date))
        ]
        if ticker_df.empty:
            st.warning(f"No data for {ticker} in the selected date range.")
            continue
        stats.append(
            {
                "Ticker": ticker,
                "Mean Close": ticker_df["Close"].mean(),
                "Std Close": ticker_df["Close"].std(),
                "Min Close": ticker_df["Close"].min(),
                "Max Close": ticker_df["Close"].max(),
                "Mean Volume": ticker_df["Volume"].mean(),
                "Mean RSI": ticker_df["RSI"].mean() if "RSI" in ticker_df else None,
                "Mean ATR": ticker_df["ATR"].mean() if "ATR" in ticker_df else None,
            }
        )

    stats_df = pd.DataFrame(stats)
    stats_df = stats_df.round(2)
    st.dataframe(stats_df, use_container_width=True)

st.subheader("Key Metrics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Latest Close", f"${latest_data['Close']:.2f}")
with col2:
    st.metric("RSI", f"{latest_data['RSI']:.2f}" if "RSI" in latest_data else "N/A")
with col3:
    st.metric("Sentiment Score", f"{sentiment_score:.2f}")
with col4:
    st.metric("ATR", f"{latest_data['ATR']:.2f}" if "ATR" in latest_data else "N/A")

# Footer
st.markdown("---")
st.markdown(
    """
**Developed by**: Aryan Mishra  
**Disclaimer**: This service provides stock analysis and recommendations for informational purposes only. It is not financial advice. Please conduct your own research and consult a qualified financial advisor before making investment decisions.
"""
)
