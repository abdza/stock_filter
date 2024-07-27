import csv
import yfinance as yf
from datetime import datetime, timedelta

def is_bullish(candle):
    return candle['Close'] > candle['Open']

def main():
    # Read input CSV
    input_stocks = []
    with open('completestocks.csv', 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            input_stocks.append(row['Symbol'])

    # Get current date and date one month ago
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    bullish_stocks = []

    # Loop through stocks
    for symbol in input_stocks:
        try:
            # Download stock data
            stock = yf.Ticker(symbol)
            hist = stock.history(start=start_date, end=end_date)

            # Check if the latest candle is bullish
            if not hist.empty and is_bullish(hist.iloc[-1]):
                bullish_stocks.append(symbol)
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    # Write bullish stocks to output CSV
    with open('stocks.csv', 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Symbol'])  # Header
        for symbol in bullish_stocks:
            csv_writer.writerow([symbol])

    print(f"Analysis complete. {len(bullish_stocks)} bullish stocks written to stocks.csv")

if __name__ == "__main__":
    main()
