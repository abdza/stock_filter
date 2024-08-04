import asyncio
import argparse
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from telegram import Bot
from telegram.error import TelegramError

def download_stock_data(symbol, start_date, end_date):
    stock = yf.Ticker(symbol)
    data = stock.history(start=start_date, end=end_date)
    return data

def calculate_sma(data, window):
    return data["Close"].rolling(window=window).mean()

def get_tradingview_link(symbol):
    return f"https://www.tradingview.com/chart/?symbol={symbol}"

async def send_telegram_message(bot, chat_id, message):
    MAX_MESSAGE_LENGTH = 4096
    messages = []
    
    while len(message) > 0:
        if len(message) <= MAX_MESSAGE_LENGTH:
            messages.append(message)
            break
        else:
            split_index = message.rfind('\n', 0, MAX_MESSAGE_LENGTH)
            if split_index == -1:
                split_index = MAX_MESSAGE_LENGTH
            messages.append(message[:split_index])
            message = message[split_index:].lstrip()

    try:
        for msg in messages:
            await bot.send_message(chat_id=chat_id, text=msg, parse_mode='HTML', disable_web_page_preview=True)
        print(f"Results sent successfully to Telegram in {len(messages)} message(s).")
    except TelegramError as e:
        print(f"Error sending Telegram message: {e}")
        print(f"Error details: {e.__class__.__name__}")
        if hasattr(e, 'message'):
            print(f"Error message: {e.message}")

async def analyze_stocks(csv_file, telegram_token=None, telegram_chat_id=None):
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

    # Prepare message for output
    cli_message = "Stocks meeting the criteria (sorted by recency and consecutive days):\n\n"
    telegram_message = "Stocks meeting the criteria (sorted by recency and consecutive days):\n\n"
    for stock in sorted_results:
        stock_info = f"{stock['symbol']}: {stock['consecutive_days']} days (First met: {stock['first_met_date'].date()}; Latest price: {stock['latest_price']:.2f})"
        cli_message += stock_info + "\n"
        telegram_message += f"<a href='{get_tradingview_link(stock['symbol'])}'>{stock['symbol']}</a>: {stock['consecutive_days']} days (First met: {stock['first_met_date'].date()}; Latest price: {stock['latest_price']:.2f})\n"

    # Print results to CLI
    print(cli_message)

    # Send message through Telegram if token and chat_id are provided
    if telegram_token and telegram_chat_id:
        bot = Bot(token=telegram_token)
        await send_telegram_message(bot, telegram_chat_id, telegram_message)

    return sorted_results

async def main():
    parser = argparse.ArgumentParser(description="Stock analysis with CLI output and optional Telegram integration")
    parser.add_argument("--csv_file", default="stocks.csv", help="Path to the CSV file containing stock symbols")
    parser.add_argument("--telegram_token", help="Telegram bot token")
    parser.add_argument("--telegram_chat_id", help="Telegram chat ID")
    args = parser.parse_args()

    print("Running stock analysis...")
    await analyze_stocks(args.csv_file, args.telegram_token, args.telegram_chat_id)
    print("Analysis complete.")

if __name__ == "__main__":
    asyncio.run(main())
