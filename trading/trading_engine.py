import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
import time
from datetime import datetime
from sqlalchemy import create_engine

# Import your existing components
from feature_generator import generate_features
from normalisation import calculate_normalization_params, normalize_features
from model import create_model

class TradingEngine:
    """Base class for trading engines that defines common functionality"""
    
    def __init__(self, config_path):
        """Initialize the trading engine with a configuration file"""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Database connection
        self.db_engine = create_engine(self.config['database']['connection_string'])
        
        # Initialize storage for data
        self.data = {}  # Will hold DataFrames for each symbol
        
        # Model
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Performance tracking
        self.positions = {
            'open': [],
            'closed': []
        }
        self.portfolio_value = self.config.get('paper_trading', {}).get('initial_capital', 10000)
        self.cash = self.portfolio_value
        self.equity_curve = [(datetime.now(), self.portfolio_value)]
        
        # Trading parameters
        self.trading_fee = self.config.get('paper_trading', {}).get('trading_fee', 0.001)
        self.slippage = self.config.get('paper_trading', {}).get('slippage', 0.0005)
        
        # Initialize tables if needed
        self._initialize_db_tables()
    
    def _setup_logging(self):
        """Set up logging configuration"""
        log_dir = Path(self.config.get('paths', {}).get('log_dir', 'logs'))
        log_dir.mkdir(exist_ok=True, parents=True)
        
        log_file = log_dir / f"trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logger = logging.getLogger('trading_engine')
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
    
    def _load_config(self):
        """Load configuration from JSON file"""
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def _initialize_db_tables(self):
        """Initialize database tables if they don't exist"""
        # Create trades table
        trades_table_query = """
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INT AUTO_INCREMENT PRIMARY KEY,
            position_id VARCHAR(36) UNIQUE,
            symbol VARCHAR(20) NOT NULL,
            direction VARCHAR(5) NOT NULL,
            amount FLOAT NOT NULL,
            entry_price FLOAT NOT NULL,
            stop_loss FLOAT,
            take_profit FLOAT,
            entry_time DATETIME NOT NULL,
            exit_time DATETIME,
            exit_price FLOAT,
            status VARCHAR(10) NOT NULL,
            pnl FLOAT DEFAULT 0,
            pnl_percent FLOAT DEFAULT 0,
            confidence FLOAT,
            exit_reason VARCHAR(20),
            additional_data TEXT
        )
        """
        
        # Create metrics table
        metrics_table_query = """
        CREATE TABLE IF NOT EXISTS trading_metrics (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME NOT NULL,
            win_rate FLOAT,
            precision_score FLOAT,
            economic_gain FLOAT,
            mean_holding_time FLOAT,
            portfolio_value FLOAT,
            total_trades INT,
            profit_factor FLOAT,
            expectancy FLOAT,
            additional_data TEXT
        )
        """
        
        try:
            with self.db_engine.connect() as conn:
                conn.execute(trades_table_query)
                conn.execute(metrics_table_query)
            self.logger.info("Database tables initialized")
        except Exception as e:
            self.logger.error(f"Error initializing database tables: {e}")
    
    def load_model(self):
        """Load the trading model"""
        model_path = Path(self.config['model']['model_path'])
        
        if not model_path.exists():
            self.logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Create model using your existing create_model function
            self.model, _ = create_model(checkpoint['config'], self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def fetch_historical_data(self, symbol, timeframe, limit=1000, start_time=None):
        """Fetch historical data from the database"""
        if start_time:
            time_condition = f"AND b.closeTime >= {int(start_time.timestamp())}"
        else:
            time_condition = ""
            
        query = f"""
            SELECT 
                b.open, b.high, b.low, b.close, b.volume,
                b.closeTime,
                '{symbol}' as code
            FROM bar b
            JOIN chart c ON b.chart_id = c.id
            WHERE c.code = '{symbol}' 
            AND c.basis = '{timeframe}'
            {time_condition}
            ORDER BY b.closeTime ASC
            LIMIT {limit}
        """
        
        try:
            df = pd.read_sql_query(query, self.db_engine)
            self.logger.info(f"Fetched {len(df)} historical bars for {symbol}")
            return df
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def update_data(self):
        """Update data for all symbols"""
        symbols = self.config['trading']['symbols']
        timeframe = self.config['trading']['timeframe']
        
        for symbol in symbols:
            # Check if we already have data for this symbol
            if symbol in self.data and not self.data[symbol].empty:
                # Update with new data since last record
                last_time = self.data[symbol]['closeTime'].max()
                start_time = pd.to_datetime(last_time, unit='s')
                new_data = self.fetch_historical_data(symbol, timeframe, start_time=start_time)
                
                if not new_data.empty:
                    # Combine and remove duplicates
                    combined = pd.concat([self.data[symbol], new_data])
                    self.data[symbol] = combined.drop_duplicates(subset='closeTime').sort_values('closeTime')
                    self.logger.info(f"Updated {symbol} data, now have {len(self.data[symbol])} bars")
            else:
                # First fetch for this symbol
                self.data[symbol] = self.fetch_historical_data(symbol, timeframe)
                self.logger.info(f"Initial fetch for {symbol}, got {len(self.data[symbol])} bars")
        
        return True
    
    def generate_features(self, symbol):
        """Generate features for a symbol using existing feature generation code"""
        if symbol not in self.data or self.data[symbol].empty:
            self.logger.warning(f"No data available for {symbol}")
            return None
            
        # Get required window size
        source_width = self.config['feature_generation']['source_width']
        
        # Check if we have enough data
        if len(self.data[symbol]) < source_width:
            self.logger.warning(f"Not enough data for feature generation for {symbol}. Need {source_width}, have {len(self.data[symbol])}")
            return None
        
        # Extract the window
        window_data = self.data[symbol].tail(source_width).copy()
        
        try:
            # Calculate normalization parameters
            norm_params = calculate_normalization_params(window_data, self.config)
            
            # Generate features using your existing function
            unified_features = generate_features(window_data, self.config)
            
            # Normalize features
            normalized_features = normalize_features(unified_features, norm_params)
            
            return normalized_features
        except Exception as e:
            self.logger.error(f"Error generating features for {symbol}: {e}")
            return None
    
    def generate_trading_signals(self):
        """Generate trading signals for all symbols"""
        if self.model is None:
            self.load_model()
        
        signals = []
        
        for symbol in self.config['trading']['symbols']:
            # Generate features
            features = self.generate_features(symbol)
            
            if features is None:
                continue
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                probs, confidence, predictions = self.model(features_tensor)
                
                # Extract values
                prob = float(probs.cpu().numpy()[0])
                conf = float(confidence.cpu().numpy()[0])
                
                # Get latest price
                latest_bar = self.data[symbol].iloc[-1]
                
                # Create signal
                signal = {
                    'symbol': symbol,
                    'timestamp': pd.to_datetime(latest_bar['closeTime'], unit='s'),
                    'price': float(latest_bar['close']),
                    'probability': prob,
                    'confidence': conf,
                    'action': 'buy' if prob > 0.5 else 'sell',
                    'strength': abs(prob - 0.5) * 2  # Scale to 0-1
                }
                
                signals.append(signal)
                self.logger.info(f"Generated signal for {symbol}: prob={prob:.4f}, conf={conf:.4f}, action={signal['action']}")
        
        return signals
    
    def execute_trades(self, signals):
        """Execute trades based on signals - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement execute_trades")
    
    def update_positions(self):
        """Update open positions - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement update_positions")
    
    def calculate_metrics(self):
        """Calculate performance metrics - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement calculate_metrics")
    
    def log_metrics(self, metrics):
        """Log metrics to the database"""
        timestamp = datetime.now()
        
        # Extract standard fields
        standard_fields = {
            'timestamp': timestamp,
            'win_rate': metrics.get('win_rate', None),
            'precision_score': metrics.get('precision', None),
            'economic_gain': metrics.get('economic_gain', None),
            'mean_holding_time': metrics.get('mean_holding_time', None),
            'portfolio_value': metrics.get('portfolio_value', None),
            'total_trades': metrics.get('total_trades', None),
            'profit_factor': metrics.get('profit_factor', None),
            'expectancy': metrics.get('expectancy', None)
        }
        
        # Add any additional fields as JSON
        additional_data = {}
        for key, value in metrics.items():
            if key not in standard_fields:
                additional_data[key] = value
        
        # Create SQL insert
        fields = []
        values = []
        
        for key, value in standard_fields.items():
            if value is not None:
                fields.append(key)
                
                if isinstance(value, datetime):
                    values.append(f"'{value.strftime('%Y-%m-%d %H:%M:%S')}'")
                elif isinstance(value, (int, float)):
                    values.append(str(value))
                else:
                    values.append(f"'{value}'")
        
        # Add additional data as JSON
        if additional_data:
            fields.append('additional_data')
            values.append(f"'{json.dumps(additional_data)}'")
        
        # Build query
        query = f"""
            INSERT INTO trading_metrics ({', '.join(fields)})
            VALUES ({', '.join(values)})
        """
        
        try:
            with self.db_engine.connect() as conn:
                conn.execute(query)
            self.logger.info(f"Logged metrics to database: {metrics}")
            return True
        except Exception as e:
            self.logger.error(f"Error logging metrics: {e}")
            return False
    
    def run_iteration(self):
        """Run a single trading iteration"""
        self.logger.info("Starting trading iteration")
        
        # Update market data
        self.update_data()
        
        # Generate trading signals
        signals = self.generate_trading_signals()
        
        # Execute trades based on signals
        self.execute_trades(signals)
        
        # Update open positions
        self.update_positions()
        
        # Calculate and log metrics
        metrics = self.calculate_metrics()
        self.log_metrics(metrics)
        
        self.logger.info("Trading iteration completed")
        return metrics
    
    def run(self, iterations=None, interval=None):
        """Run the trading engine for a specified number of iterations or indefinitely"""
        self.logger.info("Starting trading engine")
        
        if iterations is None:
            self.logger.info("Running indefinitely...")
        else:
            self.logger.info(f"Running for {iterations} iterations")
        
        iteration = 0
        running = True
        
        while running:
            try:
                metrics = self.run_iteration()
                
                iteration += 1
                self.logger.info(f"Completed iteration {iteration}")
                
                # Check if we should stop
                if iterations is not None and iteration >= iterations:
                    running = False
                    self.logger.info(f"Completed all {iterations} iterations")
                
                # Sleep if interval specified
                if interval and running:
                    self.logger.info(f"Sleeping for {interval} seconds")
                    time.sleep(interval)
                    
            except KeyboardInterrupt:
                self.logger.info("Trading engine stopped by user")
                running = False
            except Exception as e:
                self.logger.error(f"Error in trading iteration: {e}")
                if iterations is None:
                    # If running indefinitely, wait a bit and continue
                    time.sleep(60)
                else:
                    # If running a specific number of iterations, stop
                    running = False
        
        self.logger.info("Trading engine stopped")