#!/bin/env python 
import pandas as pd
import numpy as np
import yfinance as yf
import argparse
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def download_data(ticker, start_date, end_date):
    """Download stock data from Yahoo Finance."""
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    df.reset_index(inplace=True)
    return df

def calculate_candle_properties(df):
    """Calculate candle colors, sizes, gap information, price movement, and volume conditions."""
    df['Color'] = np.where(df['Close'] >= df['Open'], 'green', 'red')
    df['Body_Size'] = abs(df['Close'] - df['Open'])
    average_body_size = df['Body_Size'].mean()
    df['Is_Large'] = df['Body_Size'] > average_body_size
    df['Previous_Open'] = df['Open'].shift(1)
    df['Gap_Up'] = df['Open'] > df['Previous_Open']
    df['Price_Up'] = df['Close'] > df['Close'].shift(1)
    
    # Volume conditions
    df['Previous_Volume'] = df['Volume'].shift(1)
    df['Volume_Higher'] = df['Volume'] > df['Previous_Volume']
    df['Average_Volume'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Above_Average'] = df['Volume'] > df['Average_Volume']
    
    return df

def calculate_probabilities_for_scenario(args):
    """Calculate probabilities for a specific scenario."""
    df, scenario = args
    start_color, end_color, start_size, end_size, start_gap, end_gap, end_price_up, volume_higher, volume_above_avg = scenario
    results = {}
    
    for days in range(1, 11):  # max_days = 10
        total_occurrences = 0
        pattern_occurrences = 0
        
        for i in range(days, len(df) - 1):
            if all(df['Color'].iloc[i-days+1:i+1] == start_color) and \
               all(df['Is_Large'].iloc[i-days+1:i+1] == start_size) and \
               all(df['Gap_Up'].iloc[i-days+1:i+1] == start_gap) and \
               all(df['Volume_Higher'].iloc[i-days+1:i+1] == volume_higher) and \
               all(df['Volume_Above_Average'].iloc[i-days+1:i+1] == volume_above_avg):
                total_occurrences += 1
                if df['Color'].iloc[i+1] == end_color and \
                   df['Is_Large'].iloc[i+1] == end_size and \
                   df['Gap_Up'].iloc[i+1] == end_gap and \
                   df['Price_Up'].iloc[i+1] == end_price_up:
                    pattern_occurrences += 1
        
        if total_occurrences > 0:
            probability = pattern_occurrences / total_occurrences
            results[days] = (probability, total_occurrences)
        else:
            results[days] = (0, 0)
    
    return scenario, results

def print_probabilities(scenario, probabilities):
    """Print formatted probability results."""
    start_color, end_color, start_size, end_size, start_gap, end_gap, end_price_up, volume_higher, volume_above_avg = scenario
    start_size_str = "large" if start_size else "small"
    end_size_str = "large" if end_size else "small"
    start_gap_str = "gap up" if start_gap else "gap down"
    end_gap_str = "gap up" if end_gap else "gap down"
    end_price_str = "higher close" if end_price_up else "lower close"
    volume_higher_str = "higher volume" if volume_higher else "lower volume"
    volume_above_avg_str = "volume above average" if volume_above_avg else "volume below average"
    print(f"Probabilities of a {end_color} {end_size_str} {end_gap_str} candle with {end_price_str} after consecutive {start_color} {start_size_str} {start_gap_str} candles ({volume_higher_str}, {volume_above_avg_str}):")
    for days, (prob, occurrences) in probabilities.items():
        print(f"After {days:2d} {'day' if days == 1 else 'days'}: {prob:.2%} (based on {occurrences} occurrences)")
    print()

def get_current_streak(df):
    """Calculate the current streak of consecutive candles with the same properties."""
    last_row = df.iloc[-1]
    streak = 1
    
    for i in range(2, len(df) + 1):
        row = df.iloc[-i]
        if (row['Color'] == last_row['Color'] and
            row['Is_Large'] == last_row['Is_Large'] and
            row['Gap_Up'] == last_row['Gap_Up'] and
            row['Volume_Higher'] == last_row['Volume_Higher'] and
            row['Volume_Above_Average'] == last_row['Volume_Above_Average']):
            streak += 1
        else:
            break
    
    return streak

def predict_next_candle(df, all_probabilities):
    """Predict the color and price movement of the next candle based on recent data and streak."""
    last_candle = df.iloc[-1]
    color = last_candle['Color']
    size = last_candle['Is_Large']
    gap = last_candle['Gap_Up']
    volume_higher = last_candle['Volume_Higher']
    volume_above_avg = last_candle['Volume_Above_Average']
    
    streak = get_current_streak(df)
    streak = min(streak, 10)  # Cap the streak at 10 days
    
    prob_green = 0
    prob_red = 0
    prob_up = 0
    prob_down = 0

    # Check all possible combinations
    for end_color in ['green', 'red']:
        for end_size in [True, False]:
            for end_gap in [True, False]:
                for end_price_up in [True, False]:
                    key = (color, end_color, size, end_size, gap, end_gap, end_price_up, volume_higher, volume_above_avg)
                    
                    if key in all_probabilities and all_probabilities[key]:
                        prob = all_probabilities[key][streak][0]  # Use probability for the current streak
                        if end_color == 'green':
                            prob_green += prob
                        else:
                            prob_red += prob
                        if end_price_up:
                            prob_up += prob
                        else:
                            prob_down += prob

    total_prob_color = prob_green + prob_red
    total_prob_price = prob_up + prob_down

    if total_prob_color == 0 or total_prob_price == 0:
        return "Unable to predict", 0, "Unable to predict", 0

    norm_prob_green = prob_green / total_prob_color
    norm_prob_red = prob_red / total_prob_color
    norm_prob_up = prob_up / total_prob_price
    norm_prob_down = prob_down / total_prob_price

    color_prediction = "green" if norm_prob_green > norm_prob_red else "red"
    color_confidence = max(norm_prob_green, norm_prob_red)

    price_prediction = "higher" if norm_prob_up > norm_prob_down else "lower"
    price_confidence = max(norm_prob_up, norm_prob_down)

    return color_prediction, color_confidence, price_prediction, price_confidence, streak

def prepare_features(df):
    """Prepare features for machine learning model."""
    df['RSI'] = calculate_rsi(df['Close'])
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Create lagged features
    for i in range(1, 6):  # Create 5 days of lagged features
        df[f'Close_Lag_{i}'] = df['Close'].shift(i)
        df[f'Volume_Lag_{i}'] = df['Volume'].shift(i)
    
    # Target variable: 1 if price goes up, 0 if it goes down or stays the same
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    # Encode categorical variables
    le = LabelEncoder()
    df['Color'] = le.fit_transform(df['Color'])
    
    return df

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def train_model(df):
    """Train a Random Forest model."""
    features = [col for col in df.columns if col not in ['Date', 'Target']]
    X = df[features]
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model

def predict_next_day(model, df):
    """Predict the next day's price movement using the trained model."""
    last_row = df.iloc[-1:][model.feature_names_in_]
    prediction = model.predict(last_row)
    probability = model.predict_proba(last_row)
    
    return "up" if prediction[0] == 1 else "down", probability[0][prediction[0]]

def main(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*10)  # 10 years of data
    
    print(f"Downloading data for {ticker}...")
    df = download_data(ticker, start_date, end_date)
    
    if df.empty:
        print(f"No data available for ticker {ticker}")
        return
    
    df = calculate_candle_properties(df)
    
    scenarios = list(itertools.product(
        ['red', 'green'],  # start_color
        ['red', 'green'],  # end_color
        [True, False],     # start_size
        [True, False],     # end_size
        [True, False],     # start_gap
        [True, False],     # end_gap
        [True, False],     # end_price_up
        [True, False],     # volume_higher
        [True, False]      # volume_above_avg
    ))
    
    print(f"\nAnalyzing {len(scenarios)} scenarios for {ticker}...")
    all_probabilities = {}
    
    with ProcessPoolExecutor() as executor:
        future_to_scenario = {executor.submit(calculate_probabilities_for_scenario, (df, scenario)): scenario for scenario in scenarios}
        for future in as_completed(future_to_scenario):
            scenario, probabilities = future.result()
            all_probabilities[scenario] = probabilities
            print_probabilities(scenario, probabilities)

    # Additional statistics
    total_candles = len(df)
    volume_higher_count = df['Volume_Higher'].sum()
    volume_above_avg_count = df['Volume_Above_Average'].sum()
    
    print(f"\nAdditional Statistics:")
    print(f"Total trading days analyzed: {total_candles}")
    print(f"Days with higher volume than previous day: {volume_higher_count} ({volume_higher_count/total_candles:.2%})")
    print(f"Days with volume above 20-day average: {volume_above_avg_count} ({volume_above_avg_count/total_candles:.2%})")

    # Predict next candle using probability-based method
    color_prediction, color_confidence, price_prediction, price_confidence, streak = predict_next_candle(df, all_probabilities)
    
    print(f"\nProbability-based Prediction for next candle:")
    print(f"Current streak of similar candles: {streak} {'day' if streak == 1 else 'days'}")
    print(f"The next candle is predicted to be {color_prediction} with {color_confidence:.2%} confidence.")
    print(f"The next candle is predicted to close {price_prediction} than the current close with {price_confidence:.2%} confidence.")
    
    # Machine Learning approach
    print(f"\nPreparing data for machine learning model...")
    ml_df = prepare_features(df)
    
    print(f"\nTraining machine learning model for {ticker}...")
    model = train_model(ml_df)
    
    # Predict next day's movement using ML model
    movement, confidence = predict_next_day(model, ml_df)
    print(f"\nMachine Learning Model Prediction for next trading day:")
    print(f"The price is predicted to go {movement} with {confidence:.2%} confidence.")
    
    last_candle = df.iloc[-1]
    print(f"\nCurrent conditions:")
    print(f"Color: {'Green' if last_candle['Color'] == 'green' else 'Red'}")
    print(f"Size: {'Large' if last_candle['Is_Large'] else 'Small'}")
    print(f"Gap: {'Up' if last_candle['Gap_Up'] else 'Down'}")
    print(f"Volume compared to previous day: {'Higher' if last_candle['Volume_Higher'] else 'Lower'}")
    print(f"Volume compared to 20-day average: {'Above' if last_candle['Volume_Above_Average'] else 'Below'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze stock patterns using machine learning for a given stock ticker.")
    parser.add_argument("ticker", type=str, help="Stock ticker symbol (e.g., AAPL for Apple Inc.)")
    args = parser.parse_args()
    
    main(args.ticker)
