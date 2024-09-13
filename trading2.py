import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime
import logging
from typing import List, Tuple
import talib
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AdvancedTradingBot:
    def __init__(self, exchange: ccxt.Exchange, symbol: str, timeframe: str):
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.in_position = False
        self.balance = None
        self.df = pd.DataFrame()
        self.logger = self.setup_logger()
        
        # Strategy parameters (can be adjusted dynamically)
        self.short_window = 50
        self.long_window = 100
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.atr_period = 14
        self.risk_per_trade = 0.02  # 2% risk per trade

    def setup_logger(self):
        logger = logging.getLogger('TradingBot')
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler('trading_bot.log')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    def fetch_ohlcv(self):
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=1000)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            self.df = df
            self.logger.info(f"Fetched {len(df)} candles of data")
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV data: {e}")

    def calculate_indicators(self):
        try:
            # Moving Averages
            self.df['short_ma'] = talib.SMA(self.df['close'], timeperiod=self.short_window)
            self.df['long_ma'] = talib.SMA(self.df['close'], timeperiod=self.long_window)
            
            # RSI
            self.df['rsi'] = talib.RSI(self.df['close'], timeperiod=self.rsi_period)
            
            # Bollinger Bands
            self.df['upper_bb'], self.df['middle_bb'], self.df['lower_bb'] = talib.BBANDS(
                self.df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            
            # ATR for position sizing
            self.df['atr'] = talib.ATR(self.df['high'], self.df['low'], self.df['close'], timeperiod=self.atr_period)
            
            self.logger.info("Calculated technical indicators")
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")

    def update_balance(self):
        try:
            self.balance = self.exchange.fetch_balance()
            self.logger.info(f"Updated balance: {self.balance['total']}")
        except Exception as e:
            self.logger.error(f"Error updating balance: {e}")

    def get_signal(self) -> Tuple[str, float]:
        last_row = self.df.iloc[-1]
        signal = "HOLD"
        confidence = 0.0

        # Moving Average Crossover
        if last_row['short_ma'] > last_row['long_ma']:
            signal = "BUY"
            confidence += 0.3
        elif last_row['short_ma'] < last_row['long_ma']:
            signal = "SELL"
            confidence += 0.3

        # RSI
        if last_row['rsi'] < self.rsi_oversold:
            signal = "BUY"
            confidence += 0.3
        elif last_row['rsi'] > self.rsi_overbought:
            signal = "SELL"
            confidence += 0.3

        # Bollinger Bands
        if last_row['close'] < last_row['lower_bb']:
            signal = "BUY"
            confidence += 0.2
        elif last_row['close'] > last_row['upper_bb']:
            signal = "SELL"
            confidence += 0.2

        return signal, min(confidence, 1.0)

    def calculate_position_size(self, side: str) -> float:
        account_balance = self.balance['total']['USDT']
        risk_amount = account_balance * self.risk_per_trade
        last_atr = self.df['atr'].iloc[-1]
        
        if side == 'buy':
            entry_price = self.df['close'].iloc[-1]
            stop_loss = entry_price - (2 * last_atr)
            position_size = risk_amount / (entry_price - stop_loss)
        else:
            entry_price = self.df['close'].iloc[-1]
            stop_loss = entry_price + (2 * last_atr)
            position_size = risk_amount / (stop_loss - entry_price)
        
        return position_size

    def execute_trade(self, side: str, amount: float):
        try:
            if side == 'buy':
                order = self.exchange.create_market_buy_order(self.symbol, amount)
            else:
                order = self.exchange.create_market_sell_order(self.symbol, amount)
            self.logger.info(f"Executed {side} order: {order}")
        except Exception as e:
            self.logger.error(f"Error executing {side} order: {e}")

    def backtest(self, start_date: str, end_date: str) -> pd.DataFrame:
        backtest_df = self.df.loc[start_date:end_date].copy()
        backtest_df['signal'], backtest_df['confidence'] = zip(*backtest_df.apply(lambda row: self.get_signal(), axis=1))
        backtest_df['position'] = 0
        backtest_df.loc[backtest_df['signal'] == 'BUY', 'position'] = 1
        backtest_df.loc[backtest_df['signal'] == 'SELL', 'position'] = -1
        backtest_df['returns'] = backtest_df['close'].pct_change()
        backtest_df['strategy_returns'] = backtest_df['position'].shift() * backtest_df['returns']
        backtest_df['cumulative_returns'] = (1 + backtest_df['strategy_returns']).cumprod()
        return backtest_df

    def adjust_parameters(self):
        # Simple example of adjusting parameters based on recent volatility
        recent_volatility = self.df['close'].pct_change().rolling(window=30).std().iloc[-1]
        if recent_volatility > 0.02:  # High volatility
            self.short_window = max(10, self.short_window - 5)
            self.long_window = max(20, self.long_window - 10)
        else:  # Low volatility
            self.short_window = min(100, self.short_window + 5)
            self.long_window = min(200, self.long_window + 10)
        self.logger.info(f"Adjusted parameters: short_window={self.short_window}, long_window={self.long_window}")

    def run(self):
        while True:
            try:
                self.fetch_ohlcv()
                self.calculate_indicators()
                self.update_balance()
                self.adjust_parameters()
                
                signal, confidence = self.get_signal()
                self.logger.info(f"Signal: {signal}, Confidence: {confidence}")
                
                if signal == "BUY" and not self.in_position and confidence > 0.7:
                    amount = self.calculate_position_size('buy')
                    self.execute_trade('buy', amount)
                    self.in_position = True
                elif signal == "SELL" and self.in_position and confidence > 0.7:
                    amount = self.calculate_position_size('sell')
                    self.execute_trade('sell', amount)
                    self.in_position = False
                
                time.sleep(self.exchange.rateLimit / 1000)  # Respect rate limits
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(60)  # Wait a minute before retrying

class TradingBotGUI:
    def __init__(self, bot: AdvancedTradingBot):
        self.bot = bot
        self.root = tk.Tk()
        self.root.title("Forex Trading Bot")
        self.setup_ui()

    def setup_ui(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.start_button = ttk.Button(self.control_frame, text="Start", command=self.start_bot)
        self.start_button.pack(side=tk.LEFT)

        self.stop_button = ttk.Button(self.control_frame, text="Stop", command=self.stop_bot)
        self.stop_button.pack(side=tk.LEFT)

        self.update_button = ttk.Button(self.control_frame, text="Update Chart", command=self.update_chart)
        self.update_button.pack(side=tk.LEFT)

    def start_bot(self):
        # Start the bot in a separate thread
        import threading
        self.bot_thread = threading.Thread(target=self.bot.run)
        self.bot_thread.start()

    def stop_bot(self):
        # Implement a method to safely stop the bot
        pass

    def update_chart(self):
        self.ax.clear()
        self.bot.df['close'].plot(ax=self.ax)
        self.ax.set_title("Price Chart")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Price")
        self.canvas.draw()

    def run(self):
        self.root.mainloop()

# Usage
exchange = ccxt.binance({
    'apiKey': 'YOUR_API_KEY',
    'secret': 'YOUR_SECRET_KEY',
    'enableRateLimit': True,
})

bot = AdvancedTradingBot(exchange, 'EUR/USD', '1h')
gui = TradingBotGUI(bot)
gui.run()