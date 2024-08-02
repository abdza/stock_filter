#!/usr/bin/env python3

import argparse
import asyncio
import csv
import os
import sys
from datetime import datetime, timedelta

import yfinance as yf
from telegram import Bot


def get_stock_data(ticker, timeframe="1d", end_date=datetime.now(), days=365):
    if timeframe == "15m":
        days = min(days, 60)  # yfinance limitation for 15m data
        interval = "15m"
    elif timeframe == "5m":
        days = min(days, 7)
        interval = "5m"
    else:
        interval = "1d"

    start_date = end_date - timedelta(days=days)
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    if not data.empty and len(data) >= 200:
        data["MA20"] = data["Close"].rolling(window=20).mean()
        data["MA200"] = data["Close"].rolling(window=200).mean()
        data["body_range"] = abs(data["Close"] - data["Open"])
        data["candle_range"] = abs(data["High"] - data["Low"])
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

    data["Returns"] = data["Close"].pct_change()

    start_index = len(data) - 4 if today_only else 4
    for i in range(len(data) - 1, start_index - 1, -1):
        if sum(data["Returns"].iloc[i - 4 : i] < 0) < 3:
            continue

        day = data.iloc[i]
        is_doji = identify_doji(day["Open"], day["Close"], day["High"], day["Low"])
        is_hammer = identify_hammer(day["Open"], day["Close"], day["High"], day["Low"])

        if not (is_doji or is_hammer):
            continue

        avg_volume = data["Volume"].iloc[i - 5 : i].mean()
        if day["Volume"] <= 3 * avg_volume:
            continue

        return data.iloc[i - 4 : i + 1], "Doji" if is_doji else "Hammer"

    return None


def identify_ma_breakout_pattern(data, today_only):
    if len(data) < 200 or "MA20" not in data.columns or "MA200" not in data.columns:
        return None

    current_date = data.index[-1].date()

    data["CandleSize"] = data["High"] - data["Low"]
    avg_candle_size = data["CandleSize"].rolling(window=20).mean()

    for i in range(len(data) - 1, 20, -1):
        if (
            data["Close"].iloc[i - 1] <= data["MA20"].iloc[i - 1]
            and data["Close"].iloc[i] > data["MA20"].iloc[i]
        ):
            if data["CandleSize"].iloc[i] < 2 * avg_candle_size.iloc[i]:
                continue

            breakout_index = i

            for j in range(breakout_index + 1, len(data)):
                if data["Low"].iloc[j] <= data["MA20"].iloc[j]:
                    break
                if (
                    data["Low"].iloc[j] - data["MA20"].iloc[j]
                    < data["Close"].iloc[j] * 0.01
                ):
                    test_index = j

                    if (current_date - data.index[test_index].date()).days < 1:
                        break

                    for k in range(test_index + 1, len(data)):
                        if (
                            data["Close"].iloc[k] > data["Open"].iloc[k]
                            and data["High"].iloc[k] > data["High"].iloc[test_index]
                        ):
                            if today_only and k < len(data) - 4:
                                break

                            ma20_above_ma200 = (
                                data["MA20"].iloc[k] > data["MA200"].iloc[k]
                            )
                            ma_converging = abs(
                                data["MA20"].iloc[k] - data["MA200"].iloc[k]
                            ) < abs(
                                data["MA20"].iloc[k - 1] - data["MA200"].iloc[k - 1]
                            )

                            ma_info = f"MA20 {'above' if ma20_above_ma200 else 'below'} MA200, {'converging' if ma_converging else 'diverging'}"

                            return (
                                data.iloc[[breakout_index, test_index, k]],
                                f"MA Breakout and Retest ({ma_info})",
                            )

                    if not today_only or test_index >= len(data) - 4:
                        ma20_above_ma200 = (
                            data["MA20"].iloc[test_index]
                            > data["MA200"].iloc[test_index]
                        )
                        ma_converging = abs(
                            data["MA20"].iloc[test_index]
                            - data["MA200"].iloc[test_index]
                        ) < abs(
                            data["MA20"].iloc[test_index - 1]
                            - data["MA200"].iloc[test_index - 1]
                        )

                        ma_info = f"MA20 {'above' if ma20_above_ma200 else 'below'} MA200, {'converging' if ma_converging else 'diverging'}"

                        return (
                            data.iloc[[breakout_index, test_index]],
                            f"MA Breakout and Retest (Pending Confirmation, {ma_info})",
                        )

                    break
            break

    return None


def identify_morning_panic(data):
    if len(data) < 2 * 26:  # We need at least two trading days of 15-minute data
        return None

    data["CandleSize"] = data["High"] - data["Low"]
    avg_candle_size = (
        data["CandleSize"].rolling(window=26).mean()
    )  # Average over one trading day

    for i in range(len(data) - 26, 26, -26):  # Check each day
        yesterday = data.iloc[i - 26 : i]
        today_first_candle = data.iloc[i]

        if (
            yesterday["Close"].iloc[-1]
            < yesterday["Open"].iloc[0]  # Yesterday was bearish
            and today_first_candle["Close"]
            > today_first_candle["Open"]  # Today's first candle is bullish
            and today_first_candle["CandleSize"] > 5 * avg_candle_size.iloc[i]
        ):  # Today's first candle is large

            return data.iloc[i - 26 : i + 1], "Morning Panic"

    return None


def get_latest_price(ticker):
    stock = yf.Ticker(ticker)
    todays_data = stock.history(period="1d")
    if not todays_data.empty:
        return todays_data["Close"].iloc[-1]
    return None


def get_tradingview_link(ticker):
    return f"https://www.tradingview.com/chart/?symbol={ticker}"


def get_stock_float(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.info.get("floatShares")
    except:
        return None


async def send_telegram_message(bot_token, chat_id, patterns_found):
    bot = Bot(token=bot_token)
    message = f"Patterns found: {len(patterns_found)}\n\n"
    for i, (ticker, price, pattern_type, pattern, float_shares) in enumerate(
        patterns_found, 1
    ):
        tv_link = get_tradingview_link(ticker)
        price_str = f"${price:.2f}" if price is not None else "N/A"
        float_str = f"{float_shares:,}" if float_shares is not None else "N/A"

        if "Green After" in pattern_type:
            red_days = pattern_type.split(" ")[2]
            message += f"{i}. {ticker} ({price_str}): {pattern_type} Days\nFloat: {float_str}\n{tv_link}\n\n"
        else:
            message += f"{i}. {ticker} ({price_str}): {pattern_type}\nFloat: {float_str}\n{tv_link}\n\n"

        if not i % 20:
            await bot.send_message(
                chat_id=chat_id, text=message, disable_web_page_preview=True
            )
            message = ""
    if len(message) > 0:
        await bot.send_message(
            chat_id=chat_id, text=message, disable_web_page_preview=True
        )


def identify_green_after_long_red(data, today_only, min_red_candles=3):
    if len(data) < min_red_candles + 1:
        return None

    data["CandleSize"] = data["High"] - data["Low"]
    avg_candle_size = (
        data["CandleSize"].rolling(window=10).mean()
    )  # Average over one trading day

    for i in range(len(data) - 1, min_red_candles - 1, -1):
        # Check if the last candle is green
        if data["Close"].iloc[i] <= data["Open"].iloc[i]:
            continue

        if today_only and i < len(data) - 2:
            continue

        # Count the number of consecutive red candles
        red_candle_count = 0
        for j in range(i - 1, -1, -1):
            if data["Close"].iloc[j] >= data["Open"].iloc[j]:
                break
            red_candle_count += 1

        if red_candle_count >= min_red_candles:
            # Check if the green candle closes above the last red candle
            if (
                data["Close"].iloc[i] > data["Open"].iloc[i - 1]
                and data["CandleSize"].iloc[i] > avg_candle_size.iloc[i]
            ):
                return (
                    data.iloc[i - red_candle_count : i + 1],
                    f"Green After {red_candle_count} Red",
                )

    return None


def identify_extreme_volume_spike(data, today_only, threshold=5):
    if len(data) < 20:  # We need at least 20 days of data for a reliable average
        return None

    data["AvgVolume"] = data["Volume"].rolling(window=20).mean()
    data["VolumeRatio"] = data["Volume"] / data["AvgVolume"]

    start_index = len(data) - 3 if today_only else 0
    for i in range(len(data) - 1, start_index - 1, -1):
        if data["VolumeRatio"].iloc[i] >= threshold:
            # Check if all candles after the spike are bullish
            all_bullish = True
            for j in range(i + 1, len(data)):
                if data["Close"].iloc[j] <= data["Open"].iloc[j]:
                    all_bullish = False
                    break

            if all_bullish:
                return (
                    data.iloc[i : len(data)],  # Include all candles after the spike
                    f"Extreme Volume Spike ({data['VolumeRatio'].iloc[i]:.2f}x) with Bullish Follow-through",
                )

    return None


def identify_sustained_volume_increase(data, today_only, threshold=3, days=3):
    if (
        len(data) < 20 + days
    ):  # We need at least 20 days of data for a reliable average, plus the days for sustained increase
        return None

    data["AvgVolume"] = data["Volume"].rolling(window=20).mean()
    data["VolumeRatio"] = data["Volume"] / data["AvgVolume"]

    start_index = len(data) - 3 if today_only else days
    for i in range(len(data) - 1, start_index - 1, -1):
        if all(data["VolumeRatio"].iloc[i - days + 1 : i + 1] >= threshold) and all(
            data["VolumeRatio"].iloc[i - days + 1 : i + 1]
            > data["VolumeRatio"].iloc[i - days : i]
        ):
            return (
                data.iloc[i - days + 1 : i + 1],
                f"Sustained Volume Increase ({data['VolumeRatio'].iloc[i]:.2f}x)",
            )

    return None


def identify_waking_giant(data, days=3, volume_threshold=5):
    if (
        len(data) < 20 + days
    ):  # We need at least 20 days for a reliable average, plus the days for sustained increase
        return None

    data["AvgVolume"] = data["Volume"].rolling(window=20).mean()
    data["VolumeRatio"] = data["Volume"] / data["AvgVolume"]

    for i in range(len(data) - days, 20, -1):
        if all(data["VolumeRatio"].iloc[i : i + days] >= volume_threshold):
            if all(
                data["VolumeRatio"].iloc[:i] < volume_threshold
            ):  # Check if volume was low before
                return (
                    data.iloc[i : i + days],
                    f"Waking Giant (Volume {data['VolumeRatio'].iloc[i+days-1]:.2f}x average)",
                )

    return None


def identify_big_bullish_candles(data, today_only, threshold=3):
    if len(data) < 20:  # We need at least 20 days of data for a reliable average
        return None

    data["body_size"] = abs(data["Close"] - data["Open"])
    data["avg_body_size"] = data["body_size"].rolling(window=20).mean()
    data["body_size_ratio"] = data["body_size"] / data["avg_body_size"]

    start_index = len(data) - 3 if today_only else 0
    for i in range(len(data) - 1, start_index - 1, -1):
        if (
            data["Close"].iloc[i] > data["Open"].iloc[i]
            and data["body_size_ratio"].iloc[i] >= threshold
        ):
            return (
                data.iloc[i : i + 1],  # Return only the big bullish candle
                f"Big Bullish Candle ({data['body_size_ratio'].iloc[i]:.2f}x average body size)",
            )

    return None


def identify_diverging_volume_spread(data, today_only, threshold=3):
    if len(data) < 20:  # We need at least 20 days of data for reliable averages
        return None

    data["CandleSize"] = abs(data["Close"] - data["Open"])
    data["AvgCandleSize"] = data["CandleSize"].rolling(window=20).mean()
    data["CandleSizeRatio"] = data["CandleSize"] / data["AvgCandleSize"]

    data["AvgVolume"] = data["Volume"].rolling(window=20).mean()
    data["VolumeRatio"] = data["Volume"] / data["AvgVolume"]

    start_index = len(data) - 3 if today_only else 0
    for i in range(len(data) - 1, start_index - 1, -1):
        candle_size_ratio = data["CandleSizeRatio"].iloc[i]
        volume_ratio = data["VolumeRatio"].iloc[i]

        if (candle_size_ratio < 1 and volume_ratio > threshold) or (
            candle_size_ratio > threshold and volume_ratio < 1
        ):
            pattern_type = (
                "Small Candle, High Volume"
                if candle_size_ratio < 1
                else "Large Candle, Low Volume"
            )
            return (
                data.iloc[i : i + 1],
                f"Diverging Volume Spread ({pattern_type}): Candle Size {candle_size_ratio:.2f}x, Volume {volume_ratio:.2f}x",
            )

    return None


def identify_fading_volume(data, today_only, lookback=5, threshold=0.8):
    if len(data) < lookback + 1:  # We need at least lookback + 1 days of data
        return None

    data["PriceChange"] = data["Close"] - data["Close"].shift(1)
    data["CandleSize"] = abs(data["Close"] - data["Open"])

    start_index = len(data) - 3 if today_only else lookback
    for i in range(len(data) - 1, start_index - 1, -1):
        price_trend = sum(data["PriceChange"].iloc[i - lookback + 1 : i + 1]) > 0
        candle_size_trend = (
            sum(
                data["CandleSize"].iloc[i - lookback + 1 : i + 1]
                > data["CandleSize"].iloc[i - lookback : i]
            )
            > lookback / 2
        )
        volume_trend = (
            data["Volume"].iloc[i] < data["Volume"].iloc[i - lookback] * threshold
        )

        if (price_trend and candle_size_trend and volume_trend) or (
            not price_trend and not candle_size_trend and volume_trend
        ):
            trend_type = "Bullish" if price_trend else "Bearish"
            return (
                data.iloc[i - lookback + 1 : i + 1],
                f"Fading Volume ({trend_type}): Price and Candle Size {'Increasing' if price_trend else 'Decreasing'}, Volume Fading",
            )

    return None


def main(timeframe, today_only, patterns_to_find, telegram_token, telegram_chat_id):
    csv_file = "stocks.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found in the current directory.")
        sys.exit(1)

    csv_tickers = []
    with open(csv_file, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            csv_tickers.append(row["Symbol"])

    patterns_found = []
    for ticker in csv_tickers:
        print(f"Analyzing {ticker}...")
        try:
            data = get_stock_data(ticker, timeframe)

            if "waking_giant" in patterns_to_find:
                waking_giant_result = identify_waking_giant(data)
                if waking_giant_result is not None:
                    pattern, pattern_type = waking_giant_result
                    latest_price = get_latest_price(ticker)
                    patterns_found.append(
                        (ticker, latest_price, pattern_type, pattern, float_shares)
                    )
                    print(f"{pattern_type} found for {ticker}")

            if "candle_range" in data:
                data_count = len(data)
                flat_count = len(data[data["candle_range"] < 0.1])
                if flat_count / data_count > 0.05:
                    print(
                        "Data count:",
                        data_count,
                        " Filtered len:",
                        len(data[data["candle_range"] < 0.1]),
                    )
                    continue

            float_shares = get_stock_float(ticker)

            if "reversal" in patterns_to_find:
                reversal_result = identify_reversal_pattern(data, today_only)
                if reversal_result is not None:
                    pattern, candle_type = reversal_result
                    latest_price = get_latest_price(ticker)
                    patterns_found.append(
                        (ticker, latest_price, candle_type, pattern, float_shares)
                    )
                    print(f"{candle_type} reversal pattern found for {ticker}")

            if "ma_breakout" in patterns_to_find:
                ma_result = identify_ma_breakout_pattern(data, today_only)
                if ma_result is not None:
                    pattern, pattern_type = ma_result
                    latest_price = get_latest_price(ticker)
                    patterns_found.append(
                        (ticker, latest_price, pattern_type, pattern, float_shares)
                    )
                    print(f"{pattern_type} found for {ticker}")

            if "diverging_volume_spread" in patterns_to_find:
                diverging_volume_result = identify_diverging_volume_spread(
                    data, today_only
                )
                if diverging_volume_result is not None:
                    pattern, pattern_type = diverging_volume_result
                    latest_price = get_latest_price(ticker)
                    float_shares = get_stock_float(ticker)
                    patterns_found.append(
                        (ticker, latest_price, pattern_type, pattern, float_shares)
                    )
                    print(f"{pattern_type} found for {ticker}")

            if "fading_volume" in patterns_to_find:
                fading_volume_result = identify_fading_volume(data, today_only)
                if fading_volume_result is not None:
                    pattern, pattern_type = fading_volume_result
                    latest_price = get_latest_price(ticker)
                    float_shares = get_stock_float(ticker)
                    patterns_found.append(
                        (ticker, latest_price, pattern_type, pattern, float_shares)
                    )
                    print(f"{pattern_type} found for {ticker}")

            if "morning_panic" in patterns_to_find:
                morning_panic_data = get_stock_data(ticker, "15m", days=5)
                morning_panic_result = identify_morning_panic(morning_panic_data)
                if morning_panic_result is not None:
                    pattern, pattern_type = morning_panic_result
                    latest_price = get_latest_price(ticker)
                    patterns_found.append(
                        (ticker, latest_price, pattern_type, pattern, float_shares)
                    )
                    print(f"{pattern_type} found for {ticker}")

            if "green_after_long_red" in patterns_to_find:
                green_after_red_result = identify_green_after_long_red(data, today_only)
                if green_after_red_result is not None:
                    pattern, pattern_type = green_after_red_result
                    latest_price = get_latest_price(ticker)
                    patterns_found.append(
                        (ticker, latest_price, pattern_type, pattern, float_shares)
                    )
                    print(f"{pattern_type} found for {ticker}")

            if "extreme_volume_spike" in patterns_to_find:
                extreme_volume_result = identify_extreme_volume_spike(data, today_only)
                if extreme_volume_result is not None:
                    pattern, pattern_type = extreme_volume_result
                    latest_price = get_latest_price(ticker)
                    patterns_found.append(
                        (ticker, latest_price, pattern_type, pattern, float_shares)
                    )
                    print(f"{pattern_type} found for {ticker}")

            if "sustained_volume_increase" in patterns_to_find:
                sustained_volume_result = identify_sustained_volume_increase(
                    data, today_only
                )
                if sustained_volume_result is not None:
                    pattern, pattern_type = sustained_volume_result
                    latest_price = get_latest_price(ticker)
                    patterns_found.append(
                        (ticker, latest_price, pattern_type, pattern, float_shares)
                    )
                    print(f"{pattern_type} found for {ticker}")

            if "big_bullish_candles" in patterns_to_find:
                big_bullish_result = identify_big_bullish_candles(data, today_only)
                if big_bullish_result is not None:
                    pattern, pattern_type = big_bullish_result
                    latest_price = get_latest_price(ticker)
                    float_shares = get_stock_float(ticker)
                    patterns_found.append(
                        (ticker, latest_price, pattern_type, pattern, float_shares)
                    )
                    print(f"{pattern_type} found for {ticker}")

            if not any(
                pattern in patterns_to_find
                for pattern in [
                    "reversal",
                    "ma_breakout",
                    "morning_panic",
                    "green_after_long_red",
                    "extreme_volume_spike",
                    "sustained_volume_increase",
                    "waking_giant",
                    "big_bullish_candles",
                    "diverging_volume_spread",
                    "fading_volume",
                ]
            ):
                print(f"No specified patterns found for {ticker}")

        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")

    patterns_found.sort(
        key=lambda x: (
            -x[1] if x[1] is not None else float("-inf"),
            "Pending Confirmation" in x[2],
        )
    )

    filter_description = "last 3 days" if today_only else "last month"
    print(
        f"\nPatterns found: {len(patterns_found)} (Timeframe: {timeframe}, Filter: {filter_description}):"
    )
    for i, (ticker, price, pattern_type, pattern, float_shares) in enumerate(
        patterns_found, 1
    ):
        price_str = f"${price:.2f}" if price is not None else "N/A"
        float_str = f"{float_shares:,}" if float_shares is not None else "N/A"
        print(f"\n{i}. {ticker} ({price_str}) - {pattern_type}:")
        print(f"Float: {float_str}")
        data = pattern
        if "MA Breakout and Retest" in pattern_type:
            print("Breakout and Retest candles:")
            if "MA20" in data.columns and "MA200" in data.columns:
                if "Pending Confirmation" in pattern_type:
                    print(
                        data.iloc[-2:][
                            ["Open", "High", "Low", "Close", "Volume", "MA20", "MA200"]
                        ]
                    )
                    print("Waiting for confirmation candle")
                else:
                    print(
                        data.iloc[-3:][
                            ["Open", "High", "Low", "Close", "Volume", "MA20", "MA200"]
                        ]
                    )
            else:
                print("MA data not available")
        elif pattern_type == "Morning Panic":
            print("Morning Panic candles:")
            print(data.iloc[-52:][["Open", "High", "Low", "Close", "Volume"]])
        elif pattern_type == "Green After Long Red":
            print("Green After Long Red candles:")
            print(data.iloc[-4:][["Open", "High", "Low", "Close", "Volume"]])
        elif (
            "Extreme Volume Spike" in pattern_type
            or "Sustained Volume Increase" in pattern_type
        ):
            print(f"{pattern_type} candles:")
            print(
                data[
                    [
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Volume",
                        "AvgVolume",
                        "VolumeRatio",
                    ]
                ]
            )
        else:
            print(data.iloc[-5:][["Open", "High", "Low", "Close", "Volume"]])
        if pattern_type in ["Doji", "Hammer"]:
            print(f"Last period volume: {data['Volume'].iloc[-1]:.0f}")
            print(
                f"Average volume of previous 4 periods: {data['Volume'].iloc[-5:-1].mean():.0f}"
            )

    if telegram_token and telegram_chat_id:
        asyncio.run(
            send_telegram_message(telegram_token, telegram_chat_id, patterns_found)
        )
        print("Results sent to Telegram.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Pattern Analysis Program")
    parser.add_argument(
        "-t",
        "--timeframe",
        choices=["1d", "5m"],
        default="1d",
        help="Timeframe for analysis: '1d' for daily (default), '5m' for 5-minute",
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Show patterns from the last month (default is last 3 days)",
    )
    parser.add_argument(
        "-p",
        "--patterns",
        nargs="+",
        choices=[
            "reversal",
            "ma_breakout",
            "morning_panic",
            "green_after_long_red",
            "extreme_volume_spike",
            "sustained_volume_increase",
            "waking_giant",
            "big_bullish_candles",
            "diverging_volume_spread",
            "fading_volume",
        ],
        default=[
            "reversal",
            "ma_breakout",
            "morning_panic",
            "green_after_long_red",
            "extreme_volume_spike",
            "sustained_volume_increase",
            "waking_giant",
            "big_bullish_candles",
            "diverging_volume_spread",
            "fading_volume",
        ],
        help="Specify patterns to look for (default: all patterns)",
    )
    parser.add_argument(
        "--telegram_token", help="Telegram Bot Token for sending results"
    )
    parser.add_argument(
        "--telegram_chat_id", help="Telegram Chat ID for sending results"
    )
    args = parser.parse_args()

    main(
        args.timeframe,
        not args.all,
        args.patterns,
        args.telegram_token,
        args.telegram_chat_id,
    )
