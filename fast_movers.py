import pandas as pd
import yfinance as yf
import datetime
import time
import concurrent.futures


def get_price_and_volume_change(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(
        period="2d", interval="1d", prepost=True
    )  # Get two days of daily data to compare

    if len(hist) < 2:
        print(f"Not enough historical data for {ticker}")
        return None

    # Extract necessary data
    prev_close = hist["Close"].iloc[-2]
    latest_data = hist.iloc[-1]

    price_change = latest_data["Close"] - prev_close
    price_change_percent = (price_change / prev_close) * 100
    volume_change = latest_data["Volume"] - hist["Volume"].iloc[-2]

    # Include pre-market and post-market data
    current_time = datetime.datetime.now().time()
    if current_time < datetime.time(9, 30) or current_time > datetime.time(16, 0):
        prepost_hist = stock.history(period="1d", interval="1m", prepost=True)
        if len(prepost_hist) > 0:
            latest_price = prepost_hist["Close"].iloc[-1]
            price_change = latest_price - prev_close
            price_change_percent = (price_change / prev_close) * 100

    return {
        "Symbol": ticker,
        "Price Change": price_change,
        "Price Change (%)": price_change_percent,
        "Volume Change": volume_change,
    }


def main():
    # Read ticker symbols from 'stocks.csv'
    stocks_df = pd.read_csv("stocks.csv")
    tickers = stocks_df["Symbol"].tolist()

    while True:
        data = []
        # Use ThreadPoolExecutor to fetch data in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_ticker = {
                executor.submit(get_price_and_volume_change, ticker): ticker
                for ticker in tickers
            }
            for future in concurrent.futures.as_completed(future_to_ticker):
                result = future.result()
                if result:
                    data.append(result)

        result_df = pd.DataFrame(data)

        # Display the results
        if not result_df.empty:
            print(result_df)
            # Alert if a stock is moving fast
            fast_movers = result_df[abs(result_df["Price Change (%)"]) > 2]
            if not fast_movers.empty:
                print("Alert: Fast moving stocks detected!")
                print(fast_movers)
        else:
            print("No valid data available.")

        # Wait for 5 minutes before the next update
        time.sleep(300)


if __name__ == "__main__":
    main()
