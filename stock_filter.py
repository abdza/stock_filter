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
            return False

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
            input_stocks.append(row["Symbol"])

    # Get current date and date one month ago
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    qualifying_stocks = []

    # Loop through stocks
    for symbol in input_stocks:
        try:
            # Download stock data
            stock = yf.Ticker(symbol)
            hist = stock.history(start=start_date, end=end_date)

            # Check if the stock meets our criteria
            if meets_criteria(hist):
                qualifying_stocks.append(symbol)
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    # Write qualifying stocks to output CSV
    with open("stocks.csv", "w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["Symbol"])  # Header
        for symbol in qualifying_stocks:
            csv_writer.writerow([symbol])

    print(
        f"Analysis complete. {len(qualifying_stocks)} qualifying stocks written to stocks.csv"
    )


if __name__ == "__main__":
    main()
