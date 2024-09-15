import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def get_latest_data(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if not data.empty:
            return data.iloc[-1]['Volume']
        else:
            return 0
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return 0

def main():
    # Read the CSV file
    df = pd.read_csv('stocks.csv')
    
    # Get the latest volume for each ticker
    df['Latest_Volume'] = df['Symbol'].apply(get_latest_data)
    
    # Sort by volume in descending order and get top 20
    top_20 = df.sort_values('Latest_Volume', ascending=False).head(20)
    
    # Display results
    print("Top 20 stocks by latest trading volume:")
    for index, row in top_20.iterrows():
        print(f"{row['Symbol']}: {row['Latest_Volume']:,}")

if __name__ == "__main__":
    main()
