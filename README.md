# Stock Pattern Analysis Program

## Overview

The Stock Pattern Analysis Program is a powerful tool designed to identify and analyze specific stock patterns across multiple tickers. It focuses on detecting reversal patterns (Doji and Hammer) and Moving Average (MA) breakout patterns. The program can analyze stocks on both daily and 5-minute timeframes, providing flexibility for different trading strategies.

## Features

- Analyzes multiple stock tickers from a CSV file
- Identifies Doji and Hammer reversal patterns
- Detects Moving Average (MA) breakout patterns
- Supports both daily and 5-minute timeframes
- Option to filter results for today only or include patterns from the last month
- Sends results via Telegram for easy mobile notifications
- Provides TradingView links for quick chart access

## Requirements

- Python 3.7+
- yfinance
- pandas
- numpy
- python-telegram-bot
- A `stocks.csv` file containing the list of stocks to analyze

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/abdza/stock_filter.git
   cd stock_filter
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install yfinance pandas numpy python-telegram-bot
   ```

4. Create a `stocks.csv` file in the same directory as the script. This file should contain at least a 'Symbol' column with the ticker symbols you want to analyze. For example:
   ```
   Symbol
   AAPL
   GOOGL
   MSFT
   AMZN
   TSLA
   ```

## Usage

Ensure your virtual environment is activated and your `stocks.csv` file is properly set up, then run the program using the following command:

```
python stock_pattern_analysis.py [options]
```

### Options:

- `-t, --timeframe {1d,5m}`: Specify the timeframe for analysis. Use '1d' for daily (default) or '5m' for 5-minute intervals.
- `-a, --all`: Show patterns from the last month. If not specified, only today's patterns will be shown.
- `--telegram_token TOKEN`: Your Telegram Bot Token for sending results.
- `--telegram_chat_id CHAT_ID`: Your Telegram Chat ID for receiving results.

### Examples:

1. Analyze daily patterns for today only:
   ```
   python stock_pattern_analysis.py
   ```

2. Analyze 5-minute patterns for the last month and send results to Telegram:
   ```
   python stock_pattern_analysis.py -t 5m -a --telegram_token YOUR_BOT_TOKEN --telegram_chat_id YOUR_CHAT_ID
   ```

## stocks.csv File

The `stocks.csv` file is a crucial component of this program. It should be a comma-separated values file with at least one column named 'Symbol'. Each row in this column should contain a valid stock ticker symbol. The program will analyze each of these symbols for the specified patterns.

Example `stocks.csv` content:
```
Symbol
AAPL
GOOGL
MSFT
AMZN
TSLA
```

You can add or remove symbols from this file to customize which stocks the program analyzes.

## Output

The program will display the identified patterns in the console and, if specified, send them via Telegram. The output includes:

- Ticker symbol
- Pattern type (Doji, Hammer, or MA Breakout)
- Relevant candle data
- TradingView link for each identified pattern

Patterns pending confirmation (MA Breakout patterns waiting for a confirmation candle) are listed at the end of the output.

## Contributing

Contributions to improve the Stock Pattern Analysis Program are welcome. Please feel free to submit pull requests or open issues to suggest improvements or report bugs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This program is for educational and informational purposes only. It is not intended to be used as financial advice. Always conduct your own research and consult with a qualified financial advisor before making investment decisions.
