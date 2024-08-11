#!/bin/env python 
import pandas as pd
import numpy as np
import yfinance as yf
import argparse
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools

def download_data(ticker, start_date, end_date):
    """Download stock data from Yahoo Finance."""
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    df.reset_index(inplace=True)
    return df

def calculate_sma(df, window):
    """Calculate Simple Moving Average."""
    return df['Close'].rolling(window=window).mean()

def calculate_candle_properties(df):
    """Calculate candle colors, sizes, gap information, price movement, SMA conditions, and volume conditions."""
    df['Color'] = np.where(df['Close'] >= df['Open'], 'green', 'red')
    df['Body_Size'] = abs(df['Close'] - df['Open'])
    average_body_size = df['Body_Size'].mean()
    df['Is_Large'] = df['Body_Size'] > average_body_size
    df['Previous_Open'] = df['Open'].shift(1)
    df['Gap_Up'] = df['Open'] > df['Previous_Open']
    df['Price_Up'] = df['Close'] > df['Close'].shift(1)
    
    # Calculate SMAs
    df['SMA20'] = calculate_sma(df, 20)
    df['SMA50'] = calculate_sma(df, 50)
    df['SMA100'] = calculate_sma(df, 100)
    df['SMA200'] = calculate_sma(df, 200)
    
    # Check if price and SMAs are in the specified order
    df['SMA_Condition'] = (df['Close'] > df['SMA20']) & (df['SMA20'] > df['SMA50']) & \
                          (df['SMA50'] > df['SMA100']) & (df['SMA100'] > df['SMA200'])
    
    # Volume conditions
    df['Previous_Volume'] = df['Volume'].shift(1)
    df['Volume_Higher'] = df['Volume'] > df['Previous_Volume']
    df['Average_Volume'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Above_Average'] = df['Volume'] > df['Average_Volume']
    
    return df

def calculate_probabilities_for_scenario(args):
    """Calculate probabilities for a specific scenario."""
    df, scenario = args
    start_color, end_color, start_size, end_size, start_gap, end_gap, end_price_up, sma_condition, volume_higher, volume_above_avg = scenario
    results = {}
    
    for days in range(1, 11):  # max_days = 10
        total_occurrences = 0
        pattern_occurrences = 0
        
        for i in range(1, len(df) - days):
            if all(df['Color'].iloc[i:i+days] == start_color) and \
               all(df['Is_Large'].iloc[i:i+days] == start_size) and \
               all(df['Gap_Up'].iloc[i:i+days] == start_gap) and \
               all(df['SMA_Condition'].iloc[i:i+days] == sma_condition) and \
               all(df['Volume_Higher'].iloc[i:i+days] == volume_higher) and \
               all(df['Volume_Above_Average'].iloc[i:i+days] == volume_above_avg):
                total_occurrences += 1
                if df['Color'].iloc[i+days] == end_color and \
                   df['Is_Large'].iloc[i+days] == end_size and \
                   df['Gap_Up'].iloc[i+days] == end_gap and \
                   df['Price_Up'].iloc[i+days] == end_price_up:
                    pattern_occurrences += 1
        
        if total_occurrences > 0:
            probability = pattern_occurrences / total_occurrences
            results[days] = (probability, total_occurrences)
        else:
            results[days] = (0, 0)
    
    return scenario, results

def print_probabilities(scenario, probabilities):
    """Print formatted probability results."""
    start_color, end_color, start_size, end_size, start_gap, end_gap, end_price_up, sma_condition, volume_higher, volume_above_avg = scenario
    start_size_str = "large" if start_size else "small"
    end_size_str = "large" if end_size else "small"
    start_gap_str = "gap up" if start_gap else "gap down"
    end_gap_str = "gap up" if end_gap else "gap down"
    end_price_str = "higher close" if end_price_up else "lower close"
    sma_condition_str = "SMA condition met" if sma_condition else "SMA condition not met"
    volume_higher_str = "higher volume" if volume_higher else "lower volume"
    volume_above_avg_str = "volume above average" if volume_above_avg else "volume below average"
    print(f"Probabilities of a {end_color} {end_size_str} {end_gap_str} candle with {end_price_str} after consecutive {start_color} {start_size_str} {start_gap_str} candles ({sma_condition_str}, {volume_higher_str}, {volume_above_avg_str}):")
    for days, (prob, occurrences) in probabilities.items():
        print(f"After {days:2d} {'day' if days == 1 else 'days'}: {prob:.2%} (based on {occurrences} occurrences)")
    print()

def predict_next_candle(df, all_probabilities, sma_condition, volume_higher, volume_above_avg):
    """Predict the color and price movement of the next candle based on recent data."""
    last_candle = df.iloc[-1]
    color = last_candle['Color']
    size = last_candle['Is_Large']
    gap = last_candle['Gap_Up']

    prob_green = 0
    prob_red = 0
    prob_up = 0
    prob_down = 0

    # Check all possible combinations
    for end_color in ['green', 'red']:
        for end_size in [True, False]:
            for end_gap in [True, False]:
                for end_price_up in [True, False]:
                    key = (color, end_color, size, end_size, gap, end_gap, end_price_up, sma_condition, volume_higher, volume_above_avg)
                    
                    if key in all_probabilities and all_probabilities[key]:
                        prob = all_probabilities[key][1][0]  # Use probability for 1 day
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

    return color_prediction, color_confidence, price_prediction, price_confidence

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
        [True, False],     # sma_condition
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
    sma_condition_met = df['SMA_Condition'].sum()
    volume_higher_count = df['Volume_Higher'].sum()
    volume_above_avg_count = df['Volume_Above_Average'].sum()
    
    print(f"\nAdditional Statistics:")
    print(f"Total trading days analyzed: {total_candles}")
    print(f"Days with SMA condition met: {sma_condition_met} ({sma_condition_met/total_candles:.2%})")
    print(f"Days with higher volume than previous day: {volume_higher_count} ({volume_higher_count/total_candles:.2%})")
    print(f"Days with volume above 20-day average: {volume_above_avg_count} ({volume_above_avg_count/total_candles:.2%})")

    # Predict next candle for different scenarios
    last_candle = df.iloc[-1]
    last_candle_sma_condition = last_candle['SMA_Condition']
    last_candle_volume_higher = last_candle['Volume_Higher']
    last_candle_volume_above_avg = last_candle['Volume_Above_Average']
    
    print(f"\nPrediction for next candle:")
    for sma_condition in [True, False]:
        for volume_higher in [True, False]:
            for volume_above_avg in [True, False]:
                color_prediction, color_confidence, price_prediction, price_confidence = predict_next_candle(df, all_probabilities, sma_condition, volume_higher, volume_above_avg)
                sma_condition_str = "SMA condition met" if sma_condition else "SMA condition not met"
                volume_higher_str = "higher volume" if volume_higher else "lower volume"
                volume_above_avg_str = "volume above average" if volume_above_avg else "volume below average"
                print(f"\nScenario: {sma_condition_str}, {volume_higher_str}, {volume_above_avg_str}")
                print(f"The next candle is predicted to be {color_prediction} with {color_confidence:.2%} confidence.")
                print(f"The next candle is predicted to close {price_prediction} than the current close with {price_confidence:.2%} confidence.")
    
    print(f"\nCurrent conditions:")
    print(f"SMA condition: {'Met' if last_candle_sma_condition else 'Not met'}")
    print(f"Volume compared to previous day: {'Higher' if last_candle_volume_higher else 'Lower'}")
    print(f"Volume compared to 20-day average: {'Above' if last_candle_volume_above_avg else 'Below'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze candlestick patterns for a given stock ticker.")
    parser.add_argument("ticker", type=str, help="Stock ticker symbol (e.g., AAPL for Apple Inc.)")
    args = parser.parse_args()
    
    main(args.ticker)
