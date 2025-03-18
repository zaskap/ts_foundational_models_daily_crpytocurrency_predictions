import yfinance as yf
import os
import pandas as pd
from config.config import DATA_DIR

data_dir = os.path.join(DATA_DIR, 'raw')

currencies_in_focus = {
    "BTC-USD" : "Bitcoin"
    ,"BNB-USD" :  "Binance"
    ,"ETH-USD": "Ethereum"
    ,"SOL-USD" : "Solana"
}

def get_data_for_currency_from_yfinance(currency_code, end_date, verbose = False):
    if (verbose):
        print(f"Loading data for {currency_code} ...")
    currency_ticker = yf.Ticker(currency_code)
    currency_df = currency_ticker.history(period='max')
    currency_df = currency_df.reset_index()

    if(verbose):
        print(f"Total period available: {min(currency_df['Date'])} - {max(currency_df['Date'])}")
        print(f"Pruning data upto {end_date}")
    currency_df = currency_df[currency_df["Date"]<=end_date]

    currency_df.to_csv(f"{os.path.join(data_dir ,currency_code)}.csv", index = False)

    if (verbose):
        print(f"""{currency_code}.csv saved to {data_dir} with data from 
                        {min(currency_df['Date'])} - {max(currency_df['Date'])}""")

    return True

def get_data():
    for currency_code in currencies_in_focus:
        get_data_for_currency_from_yfinance(currency_code, end_date = "2024-12-31", verbose = True)


def load_crypto_data(file_path):
    """Loads cryptocurrency data from a CSV file.

    Parameters
    ---------
        file_path (str): Path to the CSV file.

    Returns
    -------
        pd.DataFrame: DataFrame with the loaded data, indexed by date.
    """
    try:
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

def preprocess_data(df):
    """Selects relevant columns (Open, Close) and sorts the DataFrame by date.

    Parameters
    ---------
        df (pd.DataFrame): Input DataFrame.

    Returns
    -------
        pd.DataFrame: Processed DataFrame.
    """
    if df is None:
        return None
    if 'Open' not in df.columns or 'Close' not in df.columns:
        print("Error: 'Open' or 'Close' columns not found in DataFrame.")
        return None
    return df[['Open', 'Close']].sort_index()

def split_data(df, train_end_date):
    """Splits the data into training and testing sets.

    Parameters
    ---------
        df (pd.DataFrame): Input DataFrame.
        train_end_date (str): Date string indicating the end of the training period (inclusive).

    Returns
    -------
        tuple: Two DataFrames, train_df and test_df.
    """
    if df is None:
        return None, None
    train_df = df[df.index <= train_end_date].copy()
    test_df = df[df.index > train_end_date].copy()
    return train_df, test_df

if __name__ == '__main__':
    get_data()