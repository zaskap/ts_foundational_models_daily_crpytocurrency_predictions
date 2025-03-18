import yfinance as yf
import os
import pandas as pd

data_dir = os.path.join("../data")

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

    currency_df.to_csv(f"{os.path.join(data_dir, currency_code)}.csv", index = False)

    if (verbose):
        print(f"""{currency_code}.csv saved to {data_dir} with data from 
                        {min(currency_df['Date'])} - {max(currency_df['Date'])}""")

    return True



def load_data():
    for currency_code in currencies_in_focus:
        get_data_for_currency_from_yfinance(currency_code, end_date = "2024-12-31", verbose = True)


if __name__ == '__main__':
    load_data()