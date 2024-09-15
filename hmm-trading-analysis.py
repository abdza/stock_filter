import numpy as np
import yfinance as yf
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

def fetch_data(ticker, years=3):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*years)
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    if stock_data.empty:
        raise ValueError(f"No data available for {ticker} in the last {years} years")
    print(f"Data fetched from {start_date.date()} to {end_date.date()}")
    return stock_data

def prepare_data(stock_data):
    if len(stock_data) < 30:  # Require at least 30 data points
        raise ValueError(f"Insufficient data: only {len(stock_data)} data points available.")
    
    # Calculate daily returns
    stock_data['Returns'] = stock_data['Close'].pct_change()
    
    # Calculate volatility (20-day rolling standard deviation of returns)
    stock_data['Volatility'] = stock_data['Returns'].rolling(window=20).std()
    
    # Remove NaN values
    stock_data = stock_data.dropna()
    
    # Extract features for HMM
    X = stock_data[['Returns', 'Volatility']].values
    
    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, stock_data

def train_hmm(X, n_components=2, n_iter=2000):
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", n_iter=n_iter, random_state=42)
    model.fit(X)
    return model

def analyze_states(model, X, stock_data):
    hidden_states = model.predict(X)
    
    # Determine which state corresponds to "Buy" recommendation
    mean_returns = [np.mean(stock_data['Returns'][hidden_states == i]) for i in range(model.n_components)]
    buy_state = np.argmax(mean_returns)
    
    return hidden_states, buy_state

def plot_results(stock_data, hidden_states, buy_state, ticker):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Plot stock price
    ax1.plot(stock_data.index, stock_data['Close'])
    ax1.set_title(f'{ticker} Stock Price and Hidden States')
    ax1.set_ylabel('Price')
    
    # Plot hidden states
    ax2.set_ylabel('Hidden State')
    ax2.plot(stock_data.index, hidden_states)
    ax2.set_ylim(-1, 2)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['State 0', 'State 1'])
    
    # Highlight buy regions
    buy_regions = (hidden_states == buy_state).astype(int)
    ax2.fill_between(stock_data.index, -1, 2, where=buy_regions, color='green', alpha=0.3, label='Buy')
    
    ax2.legend()
    plt.tight_layout()
    
    # Save the plot as a PNG file
    filename = f"{ticker}_hmm_analysis.png"
    plt.savefig(filename)
    plt.close(fig)  # Close the figure to free up memory
    
    print(f"Plot saved as {filename}")

def main():
    ticker = input("Enter the stock ticker symbol: ")
    
    try:
        stock_data = fetch_data(ticker)
        X, stock_data = prepare_data(stock_data)
        
        model = train_hmm(X)
        hidden_states, buy_state = analyze_states(model, X, stock_data)
        
        plot_results(stock_data, hidden_states, buy_state, ticker)
        
        current_state = hidden_states[-1]
        if current_state == buy_state:
            print(f"Based on the HMM analysis, it may be advisable to buy long {ticker}.")
        else:
            print(f"Based on the HMM analysis, it may not be advisable to buy long {ticker} at this time.")
        
        print("\nNote: This analysis is based on historical data and should not be the sole basis for investment decisions.")
        print("Always conduct thorough research and consider consulting with a financial advisor before making investment choices.")
    
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
