import pandas as pd
import pandas_ta as ta
import yfinance as yf


def analyze_stock(symbol):
    # Fetch historical data for the past 1 year
    stock_data = yf.download(symbol, period="1y", interval="1d")

    if stock_data.empty:
        print(f"No data found for {symbol}.")
        return None

    # Drop NaN values in 'Close' prices
    if stock_data["Close"].isnull().any():
        print(f"NaN values found in 'Close' prices for {symbol}. Dropping NaNs.")
        stock_data = stock_data.dropna(subset=["Close"])

    # Calculate technical indicators
    stock_data["RSI"] = ta.rsi(stock_data["Close"], length=14)
    stock_data["SMA50"] = ta.sma(stock_data["Close"], length=50)
    stock_data["SMA200"] = ta.sma(stock_data["Close"], length=200)

    # Manually calculate MACD
    exp12 = stock_data["Close"].ewm(span=12, adjust=False).mean()
    exp26 = stock_data["Close"].ewm(span=26, adjust=False).mean()
    stock_data["MACD"] = exp12 - exp26
    stock_data["Signal"] = stock_data["MACD"].ewm(span=9, adjust=False).mean()

    # Ensure that the required indicators are not NaN
    latest_data = stock_data.iloc[-1]
    required_columns = ["RSI", "SMA50", "SMA200", "MACD", "Signal", "Close"]
    if latest_data[required_columns].isnull().any():
        print(f"Latest data contains NaN values for {symbol}. Skipping.")
        return None

    # Initialize analysis result
    analysis = {
        "Symbol": symbol,
        "Recommendation": "Hold",
        "Reason": "",
        "Probability": "Low",
    }

    # Implement trading logic for a long position
    if latest_data["RSI"] < 30:
        if latest_data["MACD"] > latest_data["Signal"]:
            if latest_data["Close"] > latest_data["SMA50"]:
                analysis["Recommendation"] = "Buy"
                analysis["Reason"] = (
                    "RSI below 30 (oversold), MACD bullish crossover, price above SMA50."
                )
                analysis["Probability"] = "High"
            else:
                analysis["Recommendation"] = "Watch"
                analysis["Reason"] = (
                    "RSI below 30 (oversold), MACD bullish crossover, but price below SMA50."
                )
                analysis["Probability"] = "Medium"
        else:
            analysis["Recommendation"] = "Hold"
            analysis["Reason"] = (
                "RSI below 30 (oversold), but MACD not indicating bullish momentum."
            )
            analysis["Probability"] = "Low"
    else:
        analysis["Recommendation"] = "Hold"
        analysis["Reason"] = "RSI above 30, no oversold condition."
        analysis["Probability"] = "Low"

    return analysis


def main():
    # Read the CSV file
    df = pd.read_csv("stocks.csv")

    # Ensure 'Symbol' column exists
    if "Symbol" not in df.columns:
        print("The CSV file must contain a 'Symbol' column.")
        return

    # Loop over each symbol and analyze
    analysis_results = []
    for symbol in df["Symbol"]:
        result = analyze_stock(symbol)
        if result:
            analysis_results.append(result)
        else:
            print(f"Analysis could not be performed for {symbol}.")

    # Convert results to DataFrame
    results_df = pd.DataFrame(analysis_results)

    if not results_df.empty:
        # Output the results to a CSV file
        results_df.to_csv("recommend_output.csv", index=False)
        print("Analysis results have been saved to 'recommend_output.csv'.")
    else:
        print("No analysis results to save.")


if __name__ == "__main__":
    main()
