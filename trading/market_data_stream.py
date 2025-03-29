import websocket
import json
import threading
import time
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
from sqlalchemy import create_engine

class MarketDataStream:
    """
    Handles streaming market data from Binance WebSocket API 
    and stores it in the database for trading engines to use.
    """
    
    def __init__(self, config_path, symbols=None, timeframes=None):
        """
        Initialize the market data stream
        
        Args:
            config_path: Path to configuration file
            symbols: List of symbols to stream (defaults to config list)
            timeframes: List of timeframes to stream (defaults to config)
        """
        self.config_path = Path(config_path)
        self.load_config()
        
        # Use provided symbols/timeframes or defaults from config
        self.symbols = symbols or self.config['trading']['symbols']
        self.timeframes = timeframes or [self.config['trading']['timeframe']]
        
        # Set up logging
        self.logger = self.setup_logging()
        
        # Database connection
        self.db_engine = create_engine(self.config['database']['connection_string'])
        
        # State storage
        self.latest_data = {}
        self.klines = {}
        self.ws = None
        self.running = False
        self.last_db_update = {}
        self.update_interval = self.config.get('market_data', {}).get('db_update_interval', 600)  # Default 10 min
        
        # Initialize data structures
        for symbol in self.symbols:
            self.latest_data[symbol] = {}
            self.klines[symbol] = {}
            self.last_db_update[symbol] = {}
            
            for timeframe in self.timeframes:
                self.klines[symbol][timeframe] = pd.DataFrame()
                self.last_db_update[symbol][timeframe] = time.time()
    
    def load_config(self):
        """Load configuration file"""
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
    
    def setup_logging(self):
        """Set up logging for the market data stream"""
        log_dir = Path(self.config.get('paths', {}).get('log_dir', 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"market_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logger = logging.getLogger('market_data_stream')
        logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            if 'e' in data and data['e'] == 'kline':
                self.process_kline(data)
            elif 'e' in data and data['e'] == 'error':
                self.logger.error(f"WebSocket error: {data}")
            else:
                # Other message types can be handled here
                pass
                
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
    
    def process_kline(self, data):
        """Process a kline/candlestick message"""
        kline = data['k']
        
        symbol = data['s']
        interval = kline['i']  # Timeframe
        
        # Store latest data
        self.latest_data[symbol][interval] = {
            'open': float(kline['o']),
            'high': float(kline['h']),
            'low': float(kline['l']),
            'close': float(kline['c']),
            'volume': float(kline['v']),
            'closeTime': int(kline['T']),
            'isFinal': kline['x']  # True if candle closed
        }
        
        # Add to dataframe if candle is closed
        if kline['x']:
            new_row = pd.DataFrame([{
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v']),
                'closeTime': int(kline['T']),
                'code': symbol
            }])
            
            # Append to klines dataframe
            if self.klines[symbol][interval].empty:
                self.klines[symbol][interval] = new_row
            else:
                self.klines[symbol][interval] = pd.concat([self.klines[symbol][interval], new_row])
                
            # Remove duplicates and sort
            self.klines[symbol][interval] = (
                self.klines[symbol][interval]
                .drop_duplicates(subset='closeTime')
                .sort_values('closeTime')
                .reset_index(drop=True)
            )
            
            # Check if we should update the database
            current_time = time.time()
            if current_time - self.last_db_update[symbol][interval] > self.update_interval:
                self.store_klines_to_db(symbol, interval)
                self.last_db_update[symbol][interval] = current_time
    
    def store_klines_to_db(self, symbol, interval):
        """Store klines to database"""
        if self.klines[symbol][interval].empty:
            return
        
        try:
            # Find the chart ID for this symbol and interval
            chart_id_query = f"""
                SELECT id FROM chart 
                WHERE code = '{symbol}' AND basis = '{interval}'
                LIMIT 1
            """
            
            with self.db_engine.connect() as conn:
                result = conn.execute(chart_id_query)
                chart_id_row = result.fetchone()
                
                if not chart_id_row:
                    # Create a new chart entry
                    insert_chart_query = f"""
                        INSERT INTO chart (code, basis, source)
                        VALUES ('{symbol}', '{interval}', 'binance_ws')
                    """
                    conn.execute(insert_chart_query)
                    
                    # Get the newly created chart ID
                    result = conn.execute(chart_id_query)
                    chart_id_row = result.fetchone()
                
                chart_id = chart_id_row[0]
                
                # Get most recent bar from database
                last_bar_query = f"""
                    SELECT MAX(closeTime) as max_time
                    FROM bar
                    WHERE chart_id = {chart_id}
                """
                result = conn.execute(last_bar_query)
                last_bar_row = result.fetchone()
                
                last_time = 0
                if last_bar_row and last_bar_row[0]:
                    last_time = last_bar_row[0]
                
                # Filter klines to only include new data
                new_klines = self.klines[symbol][interval][
                    self.klines[symbol][interval]['closeTime'] > last_time
                ]
                
                if not new_klines.empty:
                    # Insert new klines
                    for _, row in new_klines.iterrows():
                        insert_query = f"""
                            INSERT INTO bar (chart_id, open, high, low, close, volume, closeTime)
                            VALUES (
                                {chart_id},
                                {row['open']},
                                {row['high']},
                                {row['low']},
                                {row['close']},
                                {row['volume']},
                                {row['closeTime']}
                            )
                        """
                        conn.execute(insert_query)
                    
                    self.logger.info(f"Stored {len(new_klines)} new klines for {symbol} {interval} to database")
        
        except Exception as e:
            self.logger.error(f"Error storing klines to database: {e}")
    
    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        self.logger.error(f"WebSocket error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close"""
        self.logger.info(f"WebSocket connection closed: {close_msg} (code: {close_status_code})")
        
        # Attempt to reconnect after a delay if still running
        if self.running:
            self.logger.info("Attempting to reconnect in 5 seconds...")
            time.sleep(5)
            self.connect()
    
    def on_open(self, ws):
        """Handle WebSocket connection open"""
        self.logger.info("WebSocket connection established")
        
        # Subscribe to kline streams for all symbol/timeframe combinations
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                # Convert timeframe to Binance format if needed (e.g., '4h' to '4h')
                stream_name = f"{symbol.lower()}@kline_{timeframe}"
                
                subscribe_msg = {
                    "method": "SUBSCRIBE",
                    "params": [stream_name],
                    "id": int(time.time())
                }
                
                ws.send(json.dumps(subscribe_msg))
                self.logger.info(f"Subscribed to {stream_name}")
    
    def connect(self):
        """Connect to Binance WebSocket API"""
        websocket.enableTrace(False)
        
        # Binance WebSocket endpoint
        endpoint = "wss://stream.binance.com:9443/ws"
        
        # Create WebSocket connection
        self.ws = websocket.WebSocketApp(endpoint,
                                         on_open=self.on_open,
                                         on_message=self.on_message,
                                         on_error=self.on_error,
                                         on_close=self.on_close)
        
        # Start WebSocket connection in a separate thread
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()
        
        self.running = True
        self.logger.info("Market data stream started")
    
    def disconnect(self):
        """Disconnect from WebSocket API"""
        if self.ws:
            self.running = False
            self.ws.close()
            self.logger.info("Market data stream stopped")
    
    def get_latest_klines(self, symbol, timeframe, limit=100):
        """Get latest klines for a symbol and timeframe"""
        if symbol in self.klines and timeframe in self.klines[symbol]:
            df = self.klines[symbol][timeframe]
            if not df.empty:
                return df.tail(limit)
        
        return pd.DataFrame()
    
    def save_all_klines_to_db(self):
        """Save all klines to database"""
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                self.store_klines_to_db(symbol, timeframe)
                self.logger.info(f"Forced save of {symbol} {timeframe} klines to database")

def main():
    """Run as standalone script for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Market Data Stream")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--symbols", type=str, nargs="+", help="Symbols to stream")
    parser.add_argument("--timeframes", type=str, nargs="+", help="Timeframes to stream")
    
    args = parser.parse_args()
    
    # Create market data stream
    stream = MarketDataStream(args.config, args.symbols, args.timeframes)
    
    try:
        # Connect to WebSocket
        stream.connect()
        
        # Run until interrupted
        while True:
            time.sleep(60)  # Check every minute
            
            # Print some stats
            for symbol in stream.symbols:
                for timeframe in stream.timeframes:
                    df = stream.get_latest_klines(symbol, timeframe, 1)
                    if not df.empty:
                        last_candle = df.iloc[-1]
                        print(f"{symbol} {timeframe}: Close: {last_candle['close']}, "
                              f"Time: {datetime.fromtimestamp(last_candle['closeTime']/1000)}")
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Disconnect and save data
        stream.save_all_klines_to_db()
        stream.disconnect()

if __name__ == "__main__":
    main()