import csv
from datetime import datetime, timedelta

import numpy as np
import yfinance as yf


def is_bullish(candle):
    return candle["Close"] > candle["Open"]


def candle_size(candle):
    return candle["High"] - candle["Low"]


def meets_criteria(hist):
    if len(hist) < 2:
        return False

    # Calculate average candle size
    average_candle_size = np.mean(
        [candle_size(candle) for _, candle in hist.iterrows()]
    )

    # Define large candle as 3 times the average size
    large_candle_threshold = 3 * average_candle_size

    last_candle = hist.iloc[-1]
    second_last_candle = hist.iloc[-2]

    if not is_bullish(last_candle):
        if (
            not is_bullish(second_last_candle)
            or candle_size(second_last_candle) <= large_candle_threshold
        ):
            print("Not bullish")
            # return False

    # Calculate percentage of candles with size > large_candle_threshold
    large_candles = sum(candle_size(candle) > 0.20 for _, candle in hist.iterrows())
    percentage_large = large_candles / len(hist) * 100

    # Return True if more than 70% of candles are large
    return percentage_large > 70


def main():
    # Read input CSV
    input_stocks = []
    with open("completestocks.csv", "r") as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            input_stocks.append(
                {
                    "Symbol": row["Symbol"],
                    "Sector": row.get("Sector", ""),
                    "Analyst Rating": row.get("Analyst Rating", ""),
                }
            )

    # Get current date and date one month ago
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    qualifying_stocks = []

    # Loop through stocks
    for stock_info in input_stocks:
        symbol = stock_info["Symbol"]
        try:
            # Download stock data
            stock = yf.Ticker(symbol)
            hist = stock.history(start=start_date, end=end_date)

            # Check if the stock meets our criteria
            if meets_criteria(hist):
                last_candle = hist.iloc[-1]
                last_candle_size = candle_size(last_candle)
                stock_info["Last Candle Size"] = last_candle_size
                qualifying_stocks.append(stock_info)
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    # Write qualifying stocks to output CSV
    with open("stocks.csv", "w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(
            ["Symbol", "Sector", "Analyst Rating", "Last Candle Size"]
        )  # Header
        for stock_info in qualifying_stocks:
            csv_writer.writerow(
                [
                    stock_info["Symbol"],
                    stock_info["Sector"],
                    stock_info["Analyst Rating"],
                    stock_info["Last Candle Size"],
                ]
            )

    print(
        f"Analysis complete. {len(qualifying_stocks)} qualifying stocks written to stocks.csv"
    )


if __name__ == "__main__":
    main()
