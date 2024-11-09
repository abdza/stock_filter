import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
import argparse
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Analyze stock daily ranges for maximum long profit')
    parser.add_argument('--date', type=str, 
                      help='Analysis date in YYYY-MM-DD format. Defaults to previous trading day if not specified.')
    parser.add_argument('--input', type=str, default='stocks.csv',
                      help='Input CSV file name (default: stocks.csv)')
    parser.add_argument('--output', type=str, default='results',
                      help='Output CSV file name prefix (default: results)')
    
    args = parser.parse_args()
    
    # Process date argument
    if args.date:
        try:
            analysis_date = datetime.strptime(args.date, '%Y-%m-%d').date()
        except ValueError:
            parser.error("Date must be in YYYY-MM-DD format")
    else:
        analysis_date = datetime.now().date() - timedelta(days=1)
    
    # Create output filename with date
    output_filename = f"{args.output}_{analysis_date.strftime('%Y%m%d')}.csv"
    
    return args, analysis_date, output_filename

def get_stock_data(symbol, date):
    """
    Download stock data for a specific symbol and date including pre/post market
    Returns DataFrame with stock data or None if download fails
    """
    try:
        # Convert date string to datetime if necessary
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')
        
        # Set start to previous day to ensure we get pre-market
        start_date = date - timedelta(days=1)
        end_date = date + timedelta(days=1)
        
        # Download data with 1-minute intervals
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval='1m')
        
        if df.empty:
            logger.warning(f"No data found for {symbol} on {date}")
            return None
            
        # Filter for the specific date including pre/post market
        df = df.loc[date.strftime('%Y-%m-%d')]
        return df
        
    except Exception as e:
        logger.error(f"Error downloading data for {symbol}: {str(e)}")
        return None

def calculate_max_long_profit(df):
    """
    Calculate the maximum potential long profit by finding the highest point 
    that occurs after each low point
    """
    if df is None or df.empty:
        return 0, 0, 0, 0, None, None
        
    max_profit = 0
    best_low = 0
    best_high = 0
    best_low_time = None
    best_high_time = None
    
    # Iterate through each minute as a potential entry point
    for idx in range(len(df)):
        entry_price = df['Low'].iloc[idx]
        # Look for highest price after this point
        if idx < len(df) - 1:  # Make sure we're not at the last point
            future_prices = df['High'].iloc[idx+1:]
            max_future_price = future_prices.max()
            max_future_idx = future_prices.idxmax()
            
            profit = max_future_price - entry_price
            profit_percentage = (profit / entry_price) * 100
            
            if profit_percentage > max_profit:
                max_profit = profit_percentage
                best_low = entry_price
                best_high = max_future_price
                best_low_time = df.index[idx]
                best_high_time = max_future_idx
    
    return max_profit, best_low, best_high, best_high - best_low, best_low_time, best_high_time

def analyze_stocks(input_file, date, output_file):
    """
    Main function to analyze stock ranges
    """
    try:
        # Read input CSV
        stocks_df = pd.read_csv(input_file)
        
        # Verify required columns
        required_columns = ['Symbol', 'Exchange']
        missing_columns = [col for col in required_columns if col not in stocks_df.columns]
        if missing_columns:
            raise ValueError(f"Input CSV missing required columns: {', '.join(missing_columns)}")
            
        # Filter out OTC stocks
        non_otc_df = stocks_df[~stocks_df['Exchange'].str.contains('OTC', case=False, na=False)]
        logger.info(f"Filtered out {len(stocks_df) - len(non_otc_df)} OTC stocks")
        
        if non_otc_df.empty:
            raise ValueError("No non-OTC stocks found in input file")
            
        # Initialize results list
        results = []
        
        # Process each symbol
        total_symbols = len(non_otc_df)
        processed_count = 0
        
        for idx, row in non_otc_df.iterrows():
            symbol = row['Symbol']
            exchange = row['Exchange']
            processed_count += 1
            
            # Calculate percentage complete
            progress = (processed_count / total_symbols) * 100
            logger.info(f"Processing {symbol} ({exchange}) - {processed_count}/{total_symbols} ({progress:.1f}%)")
            
            # Get stock data
            stock_data = get_stock_data(symbol, date)
            
            if stock_data is not None:
                max_profit_pct, low_price, high_price, absolute_range, low_time, high_time = calculate_max_long_profit(stock_data)
                
                results.append({
                    'Symbol': symbol,
                    'Exchange': exchange,
                    'Date': date,
                    'Entry_Price': low_price,
                    'Exit_Price': high_price,
                    'Entry_Time': low_time,
                    'Exit_Time': high_time,
                    'Absolute_Range': absolute_range,
                    'Profit_Percentage': max_profit_pct
                })
        
        # Create results DataFrame and sort by profit percentage
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Profit_Percentage', ascending=False)
        
        # Save to CSV
        results_df.to_csv(output_file, index=False)
        logger.info(f"Analysis complete. Results saved to {output_file}")
        
        return results_df
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Parse command line arguments
        args, analysis_date, output_filename = parse_arguments()
        
        logger.info(f"Starting analysis for date: {analysis_date}")
        logger.info(f"Output will be saved to: {output_filename}")
        
        results = analyze_stocks(args.input, analysis_date, output_filename)
        
        # Print top 5 opportunities
        print(f"\nTop 5 long opportunities for {analysis_date}:")
        print(results.head().to_string())
        
        print(f"\nResults saved to: {output_filename}")
        
    except Exception as e:
        logger.error(f"Program failed: {str(e)}")
