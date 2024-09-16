import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np

def download_stock_data(symbol, period='3mo'):
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period)
    return data

def identify_breakout(data, window=20, threshold=1.5, min_volatility=0.01):
    data['Range'] = abs(data['Open'] - data['Close'])
    data['Avg_Range'] = data['Range'].rolling(window=window).mean()
    data['Upper_Band'] = data['Close'].rolling(window=window).max()
    data['Lower_Band'] = data['Close'].rolling(window=window).min()
    data['Bandwidth'] = (data['Upper_Band'] - data['Lower_Band']) / data['Close']
    
    tight_consolidation = data['Bandwidth'].iloc[-window:].mean() < 0.05
    
    breakout = (data['Close'].iloc[-1] > data['Upper_Band'].iloc[-2] and 
                data['Close'].iloc[-1] > data['Close'].iloc[-2] * (1 + threshold * data['Avg_Range'].iloc[-2] / data['Close'].iloc[-2]))
    
    breakout_distance = data['Close'].iloc[-1] - data['Upper_Band'].iloc[-2]
    breakout_distance_percent = (breakout_distance / data['Upper_Band'].iloc[-2]) * 100
    
    # Check if average range is at least 1% of the current price
    sufficient_volatility = data['Avg_Range'].iloc[-1] / data['Close'].iloc[-1] >= min_volatility
    
    return {
        'breakout': tight_consolidation and breakout and sufficient_volatility,
        'upper_band': data['Upper_Band'].iloc[-2],
        'lower_band': data['Lower_Band'].iloc[-2],
        'last_close': data['Close'].iloc[-1],
        'avg_bandwidth': data['Bandwidth'].iloc[-window:].mean(),
        'avg_range': data['Avg_Range'].iloc[-1],
        'breakout_distance': breakout_distance,
        'breakout_distance_percent': breakout_distance_percent,
        'avg_range_percent': (data['Avg_Range'].iloc[-1] / data['Close'].iloc[-1]) * 100
    }

def main():
    stocks_df = pd.read_csv('stocks.csv')
    symbols = stocks_df['Symbol'].tolist()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    breakout_stocks = []
    
    for symbol in symbols:
        try:
            data = download_stock_data(symbol)
            result = identify_breakout(data)
            if result['breakout']:
                breakout_stocks.append((symbol, result))
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    
    if breakout_stocks:
        # Sort breakout stocks by breakout distance percentage in descending order
        breakout_stocks.sort(key=lambda x: x[1]['breakout_distance_percent'], reverse=True)
        
        print("Stocks on bullish breakout after consolidation (sorted by breakout strength):")
        for stock, details in breakout_stocks:
            print(f"\nStock: {stock}")
            print(f"Breakout Distance: {details['breakout_distance_percent']:.2f}% (${details['breakout_distance']:.2f})")
            print(f"Last Close: ${details['last_close']:.2f}")
            print(f"Consolidation Range: ${details['lower_band']:.2f} - ${details['upper_band']:.2f}")
            print(f"Average Bandwidth: {details['avg_bandwidth']:.4f}")
            print(f"Average Range: ${details['avg_range']:.2f} ({details['avg_range_percent']:.2f}% of price)")
    else:
        print("No stocks found on bullish breakout after consolidation.")

if __name__ == "__main__":
    main()
