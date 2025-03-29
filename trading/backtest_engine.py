import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import json
import logging
from datetime import datetime
from sqlalchemy import create_engine
from tqdm import tqdm

# Import your existing components
from feature_generator import generate_features
from normalisation import calculate_normalization_params, normalize_features
from model import create_model
from build_dataset import fetch_multi_asset_data

class BacktestEngine:
    """
    Engine for backtesting trading strategies on historical data.
    Uses the same model and feature generation as the trading engine.
    """
    
    def __init__(self, config_path):
        """Initialize the backtest engine with a configuration file"""
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
        
        # Backtest parameters
        self.backtest_config = self.config.get('backtest', {})
        self.initial_capital = self.backtest_config.get('initial_capital', 10000)
        self.position_size_pct = self.backtest_config.get('position_size_pct', 0.02)
        self.max_positions = self.backtest_config.get('max_positions', 5)
        self.stop_loss_pct = self.backtest_config.get('stop_loss_pct', 0.02)
        self.take_profit_pct = self.backtest_config.get('take_profit_pct', 0.03)
        self.trading_fee = self.backtest_config.get('trading_fee', 0.001)
        self.slippage = self.backtest_config.get('slippage', 0.0005)
        self.confidence_threshold = self.backtest_config.get('confidence_threshold', 0.7)
        
        # Backtest results
        self.positions = []
        self.equity_curve = []
        self.signals = []
        
        self.logger.info(f"Backtest engine initialized with {self.initial_capital} initial capital")
    
    def _setup_logging(self):
        """Set up logging configuration"""
        log_dir = Path(self.config.get('paths', {}).get('log_dir', 'logs'))
        log_dir.mkdir(exist_ok=True, parents=True)
        
        log_file = log_dir / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logger = logging.getLogger('backtest_engine')
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
    
    def fetch_historical_data(self, start_date=None, end_date=None):
        """Fetch historical data for all symbols"""
        symbols = self.config['trading']['symbols']
        timeframe = self.config['trading']['timeframe']
        
        if start_date:
            self.start_date = pd.to_datetime(start_date)
        else:
            self.start_date = pd.to_datetime(self.backtest_config.get('start_date', '2023-01-01'))
            
        if end_date:
            self.end_date = pd.to_datetime(end_date)
        else:
            self.end_date = pd.to_datetime(self.backtest_config.get('end_date', '2023-12-31'))
            
        self.logger.info(f"Fetching historical data from {self.start_date} to {self.end_date}")
        
        # Create config for fetch_multi_asset_data
        fetch_config = {
            'dataset_params': {
                'basis': timeframe,
                'codes': symbols
            }
        }
        
        # Use existing function to fetch data
        df = fetch_multi_asset_data(self.db_engine, fetch_config)
        
        # Filter by date
        df['datetime'] = pd.to_datetime(df['closeTime'], unit='s')
        df = df[(df['datetime'] >= self.start_date) & (df['datetime'] <= self.end_date)]
        
        # Store data by symbol
        for symbol in symbols:
            symbol_df = df[df['code'] == symbol].copy().reset_index(drop=True)
            
            if not symbol_df.empty:
                self.data[symbol] = symbol_df
                self.logger.info(f"Fetched {len(symbol_df)} bars for {symbol}")
            else:
                self.logger.warning(f"No data available for {symbol}")
        
        self.logger.info(f"Fetched historical data for {len(self.data)} symbols")
        return self.data
    
    def run_backtest(self):
        """Run the backtest"""
        if self.model is None:
            self.load_model()
        
        if not self.data:
            self.fetch_historical_data()
        
        # Initialize backtest state
        portfolio_value = self.initial_capital
        cash = self.initial_capital
        open_positions = []
        
        # Track equity curve and performance
        equity_curve = []
        trades = []
        
        # Get max data length to iterate through
        # Align data timesteps across all symbols
        all_timestamps = set()
        for symbol, df in self.data.items():
            all_timestamps.update(df['closeTime'].unique())
        
        timestamp_list = sorted(all_timestamps)
        
        # Minimum data needed for feature generation
        source_width = self.config['feature_generation']['source_width']
        
        # Run backtest step by step
        self.logger.info("Starting backtest...")
        
        for t in tqdm(range(len(timestamp_list)), desc="Backtest Progress"):
            current_ts = timestamp_list[t]
            current_dt = pd.to_datetime(current_ts, unit='s')
            
            # Skip until we have enough data for feature generation
            if t < source_width:
                continue
            
            # Update positions with current prices
            for pos in open_positions:
                symbol = pos['symbol']
                if symbol in self.data:
                    # Get latest price for this timestamp
                    symbol_df = self.data[symbol]
                    current_price_row = symbol_df[symbol_df['closeTime'] == current_ts]
                    
                    if not current_price_row.empty:
                        current_price = float(current_price_row['close'].iloc[0])
                        
                        # Update position
                        pos['current_price'] = current_price
                        pos['current_value'] = pos['quantity'] * current_price
                        pos['unrealized_pnl'] = (current_price - pos['entry_price']) * pos['quantity']
                        pos['unrealized_pnl_pct'] = (current_price / pos['entry_price'] - 1) * 100
            
            # Check for stop loss / take profit hits
            positions_to_close = []
            for pos in open_positions:
                # Check if we have a current price
                if 'current_price' in pos:
                    # Check stop loss
                    if pos['current_price'] <= pos['stop_loss']:
                        pos['exit_price'] = pos['stop_loss']
                        pos['exit_time'] = current_dt
                        pos['exit_reason'] = 'stop_loss'
                        positions_to_close.append(pos)
                    
                    # Check take profit
                    elif pos['current_price'] >= pos['take_profit']:
                        pos['exit_price'] = pos['take_profit']
                        pos['exit_time'] = current_dt
                        pos['exit_reason'] = 'take_profit'
                        positions_to_close.append(pos)
            
            # Close positions
            for pos in positions_to_close:
                # Apply slippage and fees
                exit_price_with_slippage = pos['exit_price'] * (1 - self.slippage)
                exit_fee = pos['quantity'] * exit_price_with_slippage * self.trading_fee
                
                # Calculate realized P&L
                pos['realized_pnl'] = (exit_price_with_slippage - pos['entry_price']) * pos['quantity'] - exit_fee - pos['entry_fee']
                pos['realized_pnl_pct'] = (exit_price_with_slippage / pos['entry_price'] - 1) * 100 - (self.trading_fee * 100 * 2)
                
                # Add to cash
                cash += (pos['quantity'] * exit_price_with_slippage - exit_fee)
                
                # Move to closed trades
                trades.append(pos)
                
                # Remove from open positions
                open_positions = [p for p in open_positions if p['position_id'] != pos['position_id']]
                
                self.logger.info(f"Closed position for {pos['symbol']} at {pos['exit_price']}: "
                               f"P&L={pos['realized_pnl']:.2f} ({pos['realized_pnl_pct']:.2f}%), "
                               f"reason={pos['exit_reason']}")
            
            # Generate signals for this timestamp
            signals = {}
            for symbol, df in self.data.items():
                # Skip if we already have a position for this symbol
                if any(p['symbol'] == symbol for p in open_positions):
                    continue
                
                # Check if we have data for this timestamp
                symbol_data = df[df['closeTime'] <= current_ts].tail(source_width)
                
                # Skip if we don't have enough data
                if len(symbol_data) < source_width:
                    continue
                
                # Generate features
                try:
                    # Calculate normalization parameters
                    norm_params = calculate_normalization_params(symbol_data, self.config)
                    
                    # Generate features
                    unified_features = generate_features(symbol_data, self.config)
                    
                    # Normalize features
                    normalized_features = normalize_features(unified_features, norm_params)
                    
                    # Convert to tensor
                    features_tensor = torch.FloatTensor(normalized_features).unsqueeze(0).to(self.device)
                    
                    # Get predictions
                    with torch.no_grad():
                        probs, confidence, predictions = self.model(features_tensor)
                        
                        # Extract values
                        prob = float(probs.cpu().numpy()[0])
                        conf = float(confidence.cpu().numpy()[0])
                        
                        # Create signal
                        signal = {
                            'symbol': symbol,
                            'timestamp': current_dt,
                            'price': float(symbol_data['close'].iloc[-1]),
                            'probability': prob,
                            'confidence': conf,
                            'action': 'buy' if prob > 0.5 else 'sell',
                            'strength': abs(prob - 0.5) * 2  # Scale to 0-1
                        }
                        
                        signals[symbol] = signal
                        
                except Exception as e:
                    self.logger.error(f"Error generating signal for {symbol}: {e}")
            
            # Execute trades based on signals
            for symbol, signal in signals.items():
                # Check if we have enough open positions
                if len(open_positions) >= self.max_positions:
                    break
                
                # Only execute trades for buy signals that meet our confidence threshold
                if (signal['action'] == 'buy' and 
                    signal['confidence'] >= self.confidence_threshold and
                    signal['probability'] >= 0.55):
                    
                    # Calculate position size
                    position_size_usd = portfolio_value * self.position_size_pct
                    
                    # Calculate quantity
                    symbol_price = signal['price']
                    quantity = position_size_usd / symbol_price
                    
                    # Apply slippage to simulate real-world execution
                    execution_price = symbol_price * (1 + self.slippage)
                    
                    # Calculate fees
                    entry_fee = quantity * execution_price * self.trading_fee
                    
                    # Check if we have enough cash
                    if cash < (quantity * execution_price + entry_fee):
                        continue
                    
                    # Calculate stop loss and take profit
                    stop_loss = execution_price * (1 - self.stop_loss_pct)
                    tp_pct = self.take_profit_pct * (1 + signal['confidence'])
                    take_profit = execution_price * (1 + tp_pct)
                    
                    # Create position
                    position = {
                        'position_id': len(trades) + len(open_positions),
                        'symbol': symbol,
                        'entry_time': current_dt,
                        'entry_price': execution_price,
                        'quantity': quantity,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'confidence': signal['confidence'],
                        'probability': signal['probability'],
                        'entry_fee': entry_fee,
                        'current_price': execution_price,
                        'current_value': quantity * execution_price,
                        'unrealized_pnl': 0,
                        'unrealized_pnl_pct': 0
                    }
                    
                    # Update cash
                    cash -= (quantity * execution_price + entry_fee)
                    
                    # Add to open positions
                    open_positions.append(position)
                    
                    self.logger.info(f"Opened position for {symbol} at {execution_price}: amount={quantity:.6f}, "
                                   f"stop={stop_loss:.2f}, target={take_profit:.2f}, confidence={signal['confidence']:.4f}")
            
            # Calculate portfolio value
            positions_value = sum(pos.get('current_value', 0) for pos in open_positions)
            portfolio_value = cash + positions_value
            
            # Add to equity curve
            equity_curve.append({
                'timestamp': current_dt,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'positions_value': positions_value,
                'open_positions': len(open_positions)
            })
        
        # Close any remaining open positions at the end
        for pos in open_positions:
            pos['exit_price'] = pos['current_price']
            pos['exit_time'] = pd.to_datetime(timestamp_list[-1], unit='s')
            pos['exit_reason'] = 'end_of_backtest'
            
            # Apply fees
            exit_fee = pos['quantity'] * pos['exit_price'] * self.trading_fee
            
            # Calculate realized P&L
            pos['realized_pnl'] = (pos['exit_price'] - pos['entry_price']) * pos['quantity'] - exit_fee - pos['entry_fee']
            pos['realized_pnl_pct'] = (pos['exit_price'] / pos['entry_price'] - 1) * 100 - (self.trading_fee * 100 * 2)
            
            # Add to closed trades
            trades.append(pos)
        
        # Store results
        self.positions = trades
        self.equity_curve = equity_curve
        
        # Calculate overall metrics
        metrics = self.calculate_metrics()
        
        self.logger.info(f"Backtest completed with final portfolio value: ${portfolio_value:.2f}")
        self.logger.info(f"Total trades: {len(trades)}, Win rate: {metrics['win_rate']:.2%}")
        
        return {
            'positions': trades,
            'equity_curve': equity_curve,
            'metrics': metrics
        }
    
    def calculate_metrics(self):
        """Calculate trading performance metrics"""
        if not self.positions:
            return {}
        
        # Basic trade metrics
        total_trades = len(self.positions)
        winning_trades = [t for t in self.positions if t.get('realized_pnl', 0) > 0]
        losing_trades = [t for t in self.positions if t.get('realized_pnl', 0) <= 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        # Win rate
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = sum(t.get('realized_pnl', 0) for t in self.positions)
        total_pnl_pct = sum(t.get('realized_pnl_pct', 0) for t in self.positions)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        avg_pnl_pct = total_pnl_pct / total_trades if total_trades > 0 else 0
        
        # Win/loss metrics
        if win_count > 0:
            avg_win = sum(t.get('realized_pnl', 0) for t in winning_trades) / win_count
            avg_win_pct = sum(t.get('realized_pnl_pct', 0) for t in winning_trades) / win_count
        else:
            avg_win = 0
            avg_win_pct = 0
            
        if loss_count > 0:
            avg_loss = sum(t.get('realized_pnl', 0) for t in losing_trades) / loss_count
            avg_loss_pct = sum(t.get('realized_pnl_pct', 0) for t in losing_trades) / loss_count
        else:
            avg_loss = 0
            avg_loss_pct = 0
        
        # Profit factor (ratio of gross profits to gross losses)
        gross_profit = sum(max(0, t.get('realized_pnl', 0)) for t in self.positions)
        gross_loss = sum(min(0, t.get('realized_pnl', 0)) for t in self.positions)
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
        
        # Expectancy (average amount you can expect to win or lose per trade)
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        # Equity curve metrics
        if self.equity_curve:
            initial_value = self.equity_curve[0]['portfolio_value']
            final_value = self.equity_curve[-1]['portfolio_value']
            
            # Calculate returns
            total_return = final_value / initial_value - 1
            
            # Calculate annualized return
            start_date = self.equity_curve[0]['timestamp']
            end_date = self.equity_curve[-1]['timestamp']
            days = (end_date - start_date).days
            
            if days > 0:
                annual_return = (1 + total_return) ** (365 / days) - 1
            else:
                annual_return = 0
            
            # Calculate drawdown
            equity_values = [e['portfolio_value'] for e in self.equity_curve]
            max_equity = 0
            max_drawdown = 0
            
            for equity in equity_values:
                if equity > max_equity:
                    max_equity = equity
                drawdown = (max_equity - equity) / max_equity
                max_drawdown = max(max_drawdown, drawdown)
            
            # Calculate Sharpe ratio (simplified)
            returns = []
            for i in range(1, len(equity_values)):
                daily_return = equity_values[i] / equity_values[i-1] - 1
                returns.append(daily_return)
            
            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = avg_return / std_return * np.sqrt(252) if std_return > 0 else 0
            else:
                avg_return = 0
                std_return = 0
                sharpe_ratio = 0
        else:
            total_return = 0
            annual_return = 0
            max_drawdown = 0
            sharpe_ratio = 0
        
        # Compile metrics
        metrics = {
            'total_trades': total_trades,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'avg_pnl': avg_pnl,
            'avg_pnl_pct': avg_pnl_pct,
            'avg_win': avg_win,
            'avg_win_pct': avg_win_pct,
            'avg_loss': avg_loss,
            'avg_loss_pct': avg_loss_pct,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
        
        return metrics
    
    def generate_report(self, save_dir=None):
        """Generate a comprehensive backtest report with visualizations"""
        if not self.positions or not self.equity_curve:
            self.logger.error("No backtest results to generate report from")
            return
        
        if save_dir is None:
            save_dir = Path(self.config.get('paths', {}).get('report_dir', 'backtest_reports'))
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # 1. Equity curve
        plt.figure(figsize=(12, 6))
        equity_df = pd.DataFrame(self.equity_curve)
        plt.plot(equity_df['timestamp'], equity_df['portfolio_value'], label='Portfolio Value')
        plt.title('Backtest Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(save_dir / 'equity_curve.png')
        plt.close()
        
        # 2. Drawdown chart
        plt.figure(figsize=(12, 6))
        equity_values = equity_df['portfolio_value'].values
        cummax = np.maximum.accumulate(equity_values)
        drawdown = (cummax - equity_values) / cummax
        plt.plot(equity_df['timestamp'], drawdown * 100)
        plt.title('Drawdown (%)')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        plt.savefig(save_dir / 'drawdown.png')
        plt.close()
        
        # 3. Trade P&L distribution
        plt.figure(figsize=(10, 6))
        pnl_values = [t.get('realized_pnl_pct', 0) for t in self.positions]
        plt.hist(pnl_values, bins=20)
        plt.axvline(x=0, color='red', linestyle='--')
        plt.title('Trade P&L Distribution (%)')
        plt.xlabel('P&L (%)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(save_dir / 'pnl_distribution.png')
        plt.close()
        
        # 4. Win rate by symbol
        symbol_results = {}
        for pos in self.positions:
            symbol = pos['symbol']
            if symbol not in symbol_results:
                symbol_results[symbol] = {'wins': 0, 'losses': 0, 'total_pnl': 0}
            
            if pos.get('realized_pnl', 0) > 0:
                symbol_results[symbol]['wins'] += 1
            else:
                symbol_results[symbol]['losses'] += 1
                
            symbol_results[symbol]['total_pnl'] += pos.get('realized_pnl', 0)
        
        # Calculate win rates
        symbols = []
        win_rates = []
        pnls = []
        
        for symbol, results in symbol_results.items():
            total = results['wins'] + results['losses']
            win_rate = results['wins'] / total if total > 0 else 0
            symbols.append(symbol)
            win_rates.append(win_rate * 100)
            pnls.append(results['total_pnl'])
        
        # Plot win rates
        plt.figure(figsize=(12, 6))
        plt.bar(symbols, win_rates)
        plt.title('Win Rate by Symbol')
        plt.xlabel('Symbol')
        plt.ylabel('Win Rate (%)')
        plt.grid(True, alpha=0.3, axis='y')
        plt.savefig(save_dir / 'win_rate_by_symbol.png')
        plt.close()
        
        # 5. Total P&L by symbol
        plt.figure(figsize=(12, 6))
        plt.bar(symbols, pnls)
        plt.title('Total P&L by Symbol')
        plt.xlabel('Symbol')
        plt.ylabel('P&L ($)')
        plt.grid(True, alpha=0.3, axis='y')
        plt.savefig(save_dir / 'pnl_by_symbol.png')
        plt.close()
        
        # 6. Trade duration analysis
        durations = []
        for pos in self.positions:
            if 'entry_time' in pos and 'exit_time' in pos:
                duration = (pos['exit_time'] - pos['entry_time']).total_seconds() / 3600  # Hours
                durations.append(duration)
        
        if durations:
            plt.figure(figsize=(10, 6))
            plt.hist(durations, bins=20)
            plt.title('Trade Duration Distribution (Hours)')
            plt.xlabel('Duration (Hours)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.savefig(save_dir / 'duration_distribution.png')
            plt.close()
        
        # 7. Exit reason distribution
        exit_reasons = {}
        for pos in self.positions:
            reason = pos.get('exit_reason', 'unknown')
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        plt.figure(figsize=(10, 6))
        plt.bar(exit_reasons.keys(), exit_reasons.values())
        plt.title('Exit Reason Distribution')
        plt.xlabel('Exit Reason')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3, axis='y')
        plt.savefig(save_dir / 'exit_reasons.png')
        plt.close()
        
        # 8. Confidence vs P&L scatter plot
        plt.figure(figsize=(10, 6))
        confidences = [pos['confidence'] for pos in self.positions]
        pnls = [pos.get('realized_pnl_pct', 0) for pos in self.positions]
        
        plt.scatter(confidences, pnls, alpha=0.6)
        plt.title('Confidence vs P&L %')
        plt.xlabel('Confidence')
        plt.ylabel('P&L %')
        plt.grid(True, alpha=0.3)
        plt.savefig(save_dir / 'confidence_vs_pnl.png')
        plt.close()
        
        # 9. Save positions to CSV
        positions_df = pd.DataFrame(self.positions)
        positions_df.to_csv(save_dir / 'trades.csv', index=False)
        
        # 10. Save equity curve to CSV
        equity_df.to_csv(save_dir / 'equity_curve.csv', index=False)
        
        # 11. Generate summary text file
        with open(save_dir / 'summary.txt', 'w') as f:
            f.write("BACKTEST PERFORMANCE SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Backtest parameters
            f.write("BACKTEST PARAMETERS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Start Date: {self.equity_curve[0]['timestamp']}\n")
            f.write(f"End Date: {self.equity_curve[-1]['timestamp']}\n")
            f.write(f"Initial Capital: ${self.initial_capital:,.2f}\n")
            f.write(f"Final Portfolio Value: ${self.equity_curve[-1]['portfolio_value']:,.2f}\n")
            f.write(f"Symbols: {', '.join(self.config['trading']['symbols'])}\n")
            f.write(f"Timeframe: {self.config['trading']['timeframe']}\n")
            f.write(f"Position Size: {self.position_size_pct:.1%} of portfolio\n")
            f.write(f"Max Positions: {self.max_positions}\n")
            f.write(f"Stop Loss: {self.stop_loss_pct:.1%}\n")
            f.write(f"Take Profit: {self.take_profit_pct:.1%}\n")
            f.write(f"Trading Fee: {self.trading_fee:.3%}\n")
            f.write(f"Slippage: {self.slippage:.3%}\n")
            f.write(f"Confidence Threshold: {self.confidence_threshold:.1f}\n\n")
            
            # Performance metrics
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Return: {metrics['total_return']:.2%}\n")
            f.write(f"Annual Return: {metrics['annual_return']:.2%}\n")
            f.write(f"Max Drawdown: {metrics['max_drawdown']:.2%}\n")
            f.write(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n\n")
            
            # Trade statistics
            f.write("TRADE STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Trades: {metrics['total_trades']}\n")
            f.write(f"Win Rate: {metrics['win_rate']:.2%} ({metrics['win_count']} wins, {metrics['loss_count']} losses)\n")
            f.write(f"Profit Factor: {metrics['profit_factor']:.2f}\n")
            f.write(f"Expectancy: ${metrics['expectancy']:.2f}\n")
            f.write(f"Average P&L: ${metrics['avg_pnl']:.2f} ({metrics['avg_pnl_pct']:.2f}%)\n")
            f.write(f"Average Win: ${metrics['avg_win']:.2f} ({metrics['avg_win_pct']:.2f}%)\n")
            f.write(f"Average Loss: ${metrics['avg_loss']:.2f} ({metrics['avg_loss_pct']:.2f}%)\n\n")
            
            # Symbol statistics
            f.write("SYMBOL STATISTICS\n")
            f.write("-" * 30 + "\n")
            for symbol, results in symbol_results.items():
                total = results['wins'] + results['losses']
                win_rate = results['wins'] / total if total > 0 else 0
                
                f.write(f"{symbol}:\n")
                f.write(f"  Trades: {total}\n")
                f.write(f"  Win Rate: {win_rate:.2%}\n")
                f.write(f"  Total P&L: ${results['total_pnl']:.2f}\n\n")
        
        self.logger.info(f"Backtest report generated and saved to {save_dir}")
        return save_dir

def main():
    """Run as standalone script for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Backtest Engine")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--report-dir", type=str, help="Directory to save report")
    
    args = parser.parse_args()
    
    # Create backtest engine
    engine = BacktestEngine(args.config)
    
    # Run backtest
    engine.run_backtest()
    
    # Generate report
    engine.generate_report(args.report_dir)

if __name__ == "__main__":
    main()