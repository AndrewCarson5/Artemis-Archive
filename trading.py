import ccxt
import pandas as pd

# Define your trading strategy
def calculate_moving_averages(df, short_window, long_window):
    df['short_mavg'] = df['close'].rolling(window=short_window).mean()
    df['long_mavg'] = df['close'].rolling(window=long_window).mean()
    return df

def execute_orders(df):
    in_position = False
    for index, row in df.iterrows():
        if row['short_mavg'] > row['long_mavg'] and not in_position:
            print("Buy")
            # exchange.create_market_buy_order('BTC/USDT', amount)
            in_position = True
        elif row['short_mavg'] < row['long_mavg'] and in_position:
            print("Sell")
            # exchange.create_market_sell_order('BTC/USDT', amount)
            in_position = False

# Set up your development environment
exchange = ccxt.binance({
    'apiKey': 'YOUR_API_KEY',
    'secret': 'YOUR_SECRET_KEY',
    'enableRateLimit': True,
})

# Fetch market data
def fetch_ohlcv(symbol, timeframe):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Run the trading bot
symbol = 'BTC/USDT'
timeframe = '1h'
short_window = 50
long_window = 100

df = fetch_ohlcv(symbol, timeframe)
df = calculate_moving_averages(df, short_window, long_window)
execute_orders(df)