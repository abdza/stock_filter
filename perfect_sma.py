from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


def download_stock_data(symbol, start_date, end_date):
    stock = yf.Ticker(symbol)
    data = stock.history(start=start_date, end=end_date)
    return data


def calculate_sma(data, window):
    return data["Close"].rolling(window=window).mean()


def analyze_stocks(csv_file):
    # Read the CSV file
    stocks_df = pd.read_csv(csv_file)

    # Set date range for data download (e.g., last year)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    results = []

    for symbol in stocks_df["Symbol"]:
        try:
            # Download stock data
            stock_data = download_stock_data(symbol, start_date, end_date)

            # Calculate SMAs
            stock_data["SMA20"] = calculate_sma(stock_data, 20)
            stock_data["SMA50"] = calculate_sma(stock_data, 50)
            stock_data["SMA100"] = calculate_sma(stock_data, 100)
            stock_data["SMA200"] = calculate_sma(stock_data, 200)

            # Check conditions and count consecutive days
            consecutive_days = 0
            first_met = None
            for i in range(len(stock_data) - 1, -1, -1):  # Iterate backwards
                row = stock_data.iloc[i]
                if (
                    row["Close"]
                    > row["SMA20"]
                    > row["SMA50"]
                    > row["SMA100"]
                    > row["SMA200"]
                ):
                    first_met = stock_data.index[i]
                    consecutive_days += 1
                else:
                    break

            if consecutive_days > 0:
                results.append(
                    {
                        "symbol": symbol,
                        "consecutive_days": consecutive_days,
                        "last_met_date": stock_data.index[-1],
                        "first_met_date": first_met,
                        "latest_price": stock_data.iloc[-1]["Close"],
                    }
                )

        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")

    # Sort results by last_met_date (most recent first) and then by consecutive_days (highest first)
    sorted_results = sorted(
        results, key=lambda x: (x["last_met_date"], x["consecutive_days"]), reverse=True
    )

    return sorted_results


if __name__ == "__main__":
    csv_file = "stocks.csv"
    filtered_stocks = analyze_stocks(csv_file)

    print("Stocks meeting the criteria (sorted by recency and consecutive days):")
    for stock in filtered_stocks:
        print(
            f"{stock['symbol']}: {stock['consecutive_days']} days (First met: {stock['first_met_date'].date()}; Latest price: {stock['latest_price']})"
        )
