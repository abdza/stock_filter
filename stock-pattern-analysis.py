#!/usr/bin/env python3

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import csv
import sys
import os
import argparse
import asyncio
from telegram import Bot

def get_stock_data(ticker, timeframe='1d', end_date=datetime.now(), days=50):
    if timeframe == '5m':
        days = min(days, 7)
        interval = '5m'
    else:
        interval = '1d'

    start_date = end_date - timedelta(days=days)
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return data

def identify_doji(open_price, close_price, high, low, doji_threshold=0.1):
    body = abs(open_price - close_price)
    wick = high - low
    return body <= wick * doji_threshold

def identify_hammer(open_price, close_price, high, low, hammer_threshold=0.3):
    body = abs(open_price - close_price)
    wick = high - low
    lower_wick = min(open_price, close_price) - low
    return (body <= wick * hammer_threshold) and (lower_wick >= wick * 0.6)

def identify_reversal_pattern(data, today_only):
    if len(data) < 5:
        return None
    
    data['Returns'] = data['Close'].pct_change()
    
    for i in range(len(data) - 1, 3, -1):
        if sum(data['Returns'].iloc[i-4:i] < 0) < 3:
            continue
        
        day = data.iloc[i]
        is_doji = identify_doji(day['Open'], day['Close'], day['High'], day['Low'])
        is_hammer = identify_hammer(day['Open'], day['Close'], day['High'], day['Low'])
        
        if not (is_doji or is_hammer):
            continue
        
        avg_volume = data['Volume'].iloc[i-5:i].mean()
        if day['Volume'] <= avg_volume:
            continue
        
        if today_only and i != len(data) - 1:
            continue
        
        return data.iloc[i-4:i+1], "Doji" if is_doji else "Hammer"
    
    return None

def identify_ma_breakout_pattern(data, today_only):
    if len(data) < 25:
        return None
    
    data['MA20'] = data['Close'].rolling(window=20).mean()
    current_date = data.index[-1].date()
    
    for i in range(len(data) - 1, 20, -1):
        if data['Close'].iloc[i-1] <= data['MA20'].iloc[i-1] and data['Close'].iloc[i] > data['MA20'].iloc[i]:
            breakout_index = i
            
            for j in range(breakout_index + 1, len(data)):
                if data['Low'].iloc[j] <= data['MA20'].iloc[j]:
                    break
                if data['Low'].iloc[j] - data['MA20'].iloc[j] < data['Close'].iloc[j] * 0.01:
                    test_index = j
                    
                    # Check if the retest candle is at least a day old
                    if (current_date - data.index[test_index].date()).days < 1:
                        break
                    
                    for k in range(test_index + 1, len(data)):
                        if data['Close'].iloc[k] > data['Open'].iloc[k] and data['High'].iloc[k] > data['High'].iloc[test_index]:
                            if today_only and k != len(data) - 1:
                                break
                            return data.iloc[[breakout_index, test_index, k]], "MA Breakout and Retest"
                    
                    # If we reach here, we have a breakout and retest, but no confirmation yet
                    if not today_only or test_index == len(data) - 1:
                        return data.iloc[[breakout_index, test_index]], "MA Breakout and Retest (Pending Confirmation)"
                    
                    break
            break
    
    return None

def get_tradingview_link(ticker):
    return f"https://www.tradingview.com/chart/?symbol={ticker}"

async def send_telegram_message(bot_token, chat_id, patterns_found):
    bot = Bot(token=bot_token)
    message = "Patterns found:\n\n"
    for ticker, _, pattern_type in patterns_found:
        tv_link = get_tradingview_link(ticker)
        message += f"{ticker}: {pattern_type}\n{tv_link}\n\n"
    await bot.send_message(chat_id=chat_id, text=message, disable_web_page_preview=True)

def main(timeframe, today_only, telegram_token, telegram_chat_id):
    csv_file = 'stocks.csv'
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found in the current directory.")
        sys.exit(1)
    
    csv_tickers = []
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            csv_tickers.append(row['Symbol'])
    
    patterns_found = []
    for ticker in csv_tickers:
        print(f"Analyzing {ticker}...")
        try:
            data = get_stock_data(ticker, timeframe)
            
            reversal_result = identify_reversal_pattern(data, today_only)
            if reversal_result is not None:
                pattern, candle_type = reversal_result
                patterns_found.append((ticker, pattern, candle_type))
                print(f"{candle_type} reversal pattern found for {ticker}")
            
            ma_result = identify_ma_breakout_pattern(data, today_only)
            if ma_result is not None:
                pattern, pattern_type = ma_result
                patterns_found.append((ticker, pattern, pattern_type))
                print(f"{pattern_type} found for {ticker}")
            
            if reversal_result is None and ma_result is None:
                print(f"No pattern found for {ticker}")
        
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
    
    # Sort patterns_found, putting pending confirmation patterns last
    patterns_found.sort(key=lambda x: "Pending Confirmation" in x[2])
    
    filter_description = "today only" if today_only else "last month"
    print(f"\nPatterns found (Timeframe: {timeframe}, Filter: {filter_description}):")
    for ticker, pattern, pattern_type in patterns_found:
        print(f"\n{ticker} - {pattern_type}:")
        if "MA Breakout and Retest" in pattern_type:
            print("Breakout and Retest candles:")
            print(pattern[['Open', 'High', 'Low', 'Close', 'Volume', 'MA20']])
            if "Pending Confirmation" in pattern_type:
                print("Waiting for confirmation candle")
        else:
            print(pattern[['Open', 'High', 'Low', 'Close', 'Volume']])
        if pattern_type in ['Doji', 'Hammer']:
            print(f"Last period volume: {pattern['Volume'].iloc[-1]:.0f}")
            print(f"Average volume of previous 4 periods: {pattern['Volume'].iloc[:-1].mean():.0f}")
    
    if telegram_token and telegram_chat_id:
        asyncio.run(send_telegram_message(telegram_token, telegram_chat_id, patterns_found))
        print("Results sent to Telegram.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Pattern Analysis Program")
    parser.add_argument("-t", "--timeframe", choices=['1d', '5m'], default='1d', 
                        help="Timeframe for analysis: '1d' for daily (default), '5m' for 5-minute")
    parser.add_argument("-a", "--all", action="store_true", 
                        help="Show patterns from the last month (default is today only)")
    parser.add_argument("--telegram_token", help="Telegram Bot Token for sending results")
    parser.add_argument("--telegram_chat_id", help="Telegram Chat ID for sending results")
    args = parser.parse_args()

    main(args.timeframe, not args.all, args.telegram_token, args.telegram_chat_id)