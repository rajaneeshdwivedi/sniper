import uuid
import pandas as pd
import numpy as np
from datetime import datetime
import json
from trading_engine import TradingEngine

class PaperTradingEngine(TradingEngine):
    """Paper trading implementation of the trading engine"""
    
    def __init__(self, config_path):
        """Initialize the paper trading engine"""
        super().__init__(config_path)
        
        # Initialize paper trading specific parameters
        paper_config = self.config.get('paper_trading', {})
        self.initial_capital = paper_config.get('initial_capital', 10000)
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        
        # Risk management
        self.base_risk_pct = paper_config.get('base_risk_percent', 0.01)
        self.max_risk_pct = paper_config.get('max_risk_percent', 0.05)
        self.stop_loss_pct = paper_config.get('stop_loss_percent', 0.02)
        self.take_profit_pct = paper_config.get('base_take_profit_percent', 0.03)
        
        # Implementation parameters
        self.slippage = paper_config.get('slippage', 0.0005)
        self.trading_fee = paper_config.get('trading_fee', 0.001)
        
        # Confidence threshold
        self.confidence_threshold = self.config['trading'].get('confidence_threshold', 0.7)
        
        # Load open positions from database
        self.load_open_positions()
        
        self.logger.info(f"Paper trading engine initialized with {self.initial_capital} capital")
    
    def load_open_positions(self):
        """Load open positions from the database"""
        query = """
            SELECT * FROM paper_trades 
            WHERE status = 'open'
        """
        
        try:
            positions_df = pd.read_sql_query(query, self.db_engine)
            
            if not positions_df.empty:
                for _, row in positions_df.iterrows():
                    position = row.to_dict()
                    
                    # Convert string timestamp to datetime if needed
                    if isinstance(position['entry_time'], str):
                        position['entry_time'] = pd.to_datetime(position['entry_time'])
                    
                    # Add additional data if available
                    if position.get('additional_data'):
                        try:
                            additional_data = json.loads(position['additional_data'])
                            position.update(additional_data)
                        except:
                            pass
                    
                    self.positions['open'].append(position)
                
                self.logger.info(f"Loaded {len(self.positions['open'])} open positions from database")
            else:
                self.logger.info("No open positions found in database")
        
        except Exception as e:
            self.logger.error(f"Error loading open positions: {e}")
    
    def execute_trades(self, signals):
        """Execute paper trades based on generated signals"""
        trades_executed = 0
        
        for signal in signals:
            # Only execute trades for buy signals that meet our confidence threshold
            if (signal['action'] == 'buy' and 
                signal['confidence'] >= self.confidence_threshold and
                signal['probability'] >= 0.55):  # Additional probability threshold
                
                # Skip if we already have an open position for this symbol
                if any(p['symbol'] == signal['symbol'] for p in self.positions['open']):
                    self.logger.info(f"Skipping trade for {signal['symbol']} - already have an open position")
                    continue
                
                # Execute the trade
                position = self.execute_paper_trade(signal)
                
                if position:
                    trades_executed += 1
        
        self.logger.info(f"Executed {trades_executed} paper trades")
        return trades_executed
    
    def execute_paper_trade(self, signal):
        """Execute a single paper trade"""
        symbol = signal['symbol']
        price = signal['price']
        confidence = signal['confidence']
        
        # Apply slippage to simulate real-world execution
        execution_price = self.apply_slippage(price)
        
        # Calculate position size based on portfolio value, confidence and risk
        position_size = self.calculate_position_size(confidence)
        
        # Calculate fees
        quantity = position_size / execution_price
        fee = self.calculate_fee(quantity, execution_price)
        
        # Check if we have enough cash
        if self.cash < position_size + fee:
            self.logger.warning(f"Not enough cash for trade. Need {position_size + fee}, have {self.cash}")
            return None
        
        # Calculate stop loss and take profit
        stop_loss = self.calculate_stop_loss(execution_price)
        take_profit = self.calculate_take_profit(execution_price, confidence)
        
        # Generate a unique position ID
        position_id = str(uuid.uuid4())
        
        # Create position object
        position = {
            'position_id': position_id,
            'symbol': symbol,
            'direction': 'long',  # We're only implementing long positions for now
            'amount': quantity,
            'entry_price': execution_price,
            'current_price': execution_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': datetime.now(),
            'exit_time': None,
            'exit_price': None,
            'status': 'open',
            'pnl': 0.0,
            'pnl_percent': 0.0,
            'confidence': confidence,
            'probability': signal['probability'],
            'exit_reason': None
        }
        
        # Update cash
        self.cash -= (position_size + fee)
        
        # Add to open positions
        self.positions['open'].append(position)
        
        # Log the trade to database
        self.log_trade(position)
        
        self.logger.info(f"Opened position for {symbol} at {execution_price}: amount={quantity:.6f}, "
                         f"stop={stop_loss:.2f}, target={take_profit:.2f}, confidence={confidence:.4f}")
        
        return position
    
    def apply_slippage(self, price):
        """Apply slippage to simulate real-world execution"""
        # For longs, slippage increases the price
        slippage_amount = price * self.slippage
        return price + slippage_amount
    
    def calculate_fee(self, quantity, price):
        """Calculate trading fee"""
        return quantity * price * self.trading_fee
    
    def calculate_position_size(self, confidence):
        """Calculate position size based on confidence and risk parameters"""
        # Adjust risk based on confidence
        risk_pct = self.base_risk_pct * (1 + confidence)
        
        # Cap to maximum risk
        risk_pct = min(risk_pct, self.max_risk_pct)
        
        # Calculate size
        position_size = self.portfolio_value * risk_pct
        
        return position_size
    
    def calculate_stop_loss(self, price):
        """Calculate stop loss price"""
        return price * (1 - self.stop_loss_pct)
    
    def calculate_take_profit(self, price, confidence):
        """Calculate take profit based on confidence"""
        # Higher confidence = higher take profit target
        tp_pct = self.take_profit_pct * (1 + confidence)
        return price * (1 + tp_pct)
    
    def update_positions(self):
        """Update open positions and check for stop/target hits"""
        if not self.positions['open']:
            return
        
        # Get latest prices for all symbols with open positions
        symbols = {p['symbol'] for p in self.positions['open']}
        latest_prices = {}
        
        for symbol in symbols:
            if symbol in self.data and not self.data[symbol].empty:
                latest_prices[symbol] = float(self.data[symbol]['close'].iloc[-1])
        
        positions_to_close = []
        
        # Check each position
        for position in self.positions['open']:
            symbol = position['symbol']
            
            if symbol not in latest_prices:
                self.logger.warning(f"No price data for {symbol}, skipping position update")
                continue
            
            current_price = latest_prices[symbol]
            position['current_price'] = current_price
            
            # Calculate unrealized P&L
            position['pnl'] = (current_price - position['entry_price']) * position['amount']
            position['pnl_percent'] = (current_price / position['entry_price'] - 1) * 100
            
            # Check for stop loss hit
            if current_price <= position['stop_loss']:
                position['exit_reason'] = 'stop_loss'
                positions_to_close.append(position)
                self.logger.info(f"Stop loss hit for {symbol} at {current_price}")
            
            # Check for take profit hit
            elif current_price >= position['take_profit']:
                position['exit_reason'] = 'take_profit'
                positions_to_close.append(position)
                self.logger.info(f"Take profit hit for {symbol} at {current_price}")
        
        # Close positions that hit stop or target
        for position in positions_to_close:
            self.close_position(position)
        
        # Update portfolio value
        self.update_portfolio_value()
    
    def close_position(self, position):
        """Close a position and update the database"""
        symbol = position['symbol']
        position_id = position['position_id']
        
        # Set exit details
        position['exit_time'] = datetime.now()
        position['exit_price'] = position['current_price']
        position['status'] = 'closed'
        
        # Calculate final P&L (including fees)
        exit_quantity = position['amount']
        exit_fee = self.calculate_fee(exit_quantity, position['exit_price'])
        
        # Update P&L to include fees
        position['pnl'] = (position['exit_price'] - position['entry_price']) * position['amount'] - exit_fee
        position['pnl_percent'] = (position['exit_price'] / position['entry_price'] - 1) * 100
        
        # Add to cash
        self.cash += (position['exit_price'] * position['amount'] - exit_fee)
        
        # Move to closed positions
        self.positions['closed'].append(position)
        
        # Remove from open positions
        self.positions['open'] = [p for p in self.positions['open'] if p['position_id'] != position_id]
        
        # Update database
        self.update_trade(position)
        
        self.logger.info(f"Closed position for {symbol} at {position['exit_price']}: "
                         f"P&L={position['pnl']:.2f} ({position['pnl_percent']:.2f}%), "
                         f"reason={position['exit_reason']}")
        
        return position
    
    def update_portfolio_value(self):
        """Update portfolio value based on open positions and cash"""
        position_value = sum(p['amount'] * p['current_price'] for p in self.positions['open'])
        self.portfolio_value = self.cash + position_value
        
        # Add to equity curve
        self.equity_curve.append((datetime.now(), self.portfolio_value))
        
        self.logger.info(f"Updated portfolio value: {self.portfolio_value:.2f} "
                         f"(Cash: {self.cash:.2f}, Positions: {position_value:.2f})")
        
        return self.portfolio_value
    
    def log_trade(self, position):
        """Log a trade to the database"""
        fields = [
            'position_id', 'symbol', 'direction', 'amount', 'entry_price',
            'stop_loss', 'take_profit', 'entry_time', 'status', 'confidence'
        ]
        
        values = []
        for field in fields:
            value = position[field]
            
            if isinstance(value, datetime):
                value = f"'{value.strftime('%Y-%m-%d %H:%M:%S')}'"
            elif isinstance(value, (int, float, bool)):
                value = str(value)
            elif value is None:
                value = 'NULL'
            else:
                value = f"'{value}'"
            
            values.append(value)
        
        # Additional data field for all other attributes
        additional_data = {k: v for k, v in position.items() if k not in fields}
        if additional_data:
            fields.append('additional_data')
            values.append(f"'{json.dumps(additional_data)}'")
        
        # Build query
        query = f"""
            INSERT INTO paper_trades ({', '.join(fields)})
            VALUES ({', '.join(values)})
        """
        
        try:
            with self.db_engine.connect() as conn:
                conn.execute(query)
            self.logger.info(f"Logged trade to database: {position['symbol']}")
            return True
        except Exception as e:
            self.logger.error(f"Error logging trade: {e}")
            return False
    
    def update_trade(self, position):
        """Update a trade in the database"""
        fields = []
        
        # Fields that need to be updated for position close
        update_fields = [
            'exit_price', 'exit_time', 'status', 'pnl', 'pnl_percent', 'exit_reason'
        ]
        
        for field in update_fields:
            value = position.get(field)
            
            if isinstance(value, datetime):
                value_str = f"{field} = '{value.strftime('%Y-%m-%d %H:%M:%S')}'"
            elif isinstance(value, (int, float, bool)):
                value_str = f"{field} = {value}"
            elif value is None:
                value_str = f"{field} = NULL"
            else:
                value_str = f"{field} = '{value}'"
            
            fields.append(value_str)
        
        # Additional data
        additional_data = {k: v for k, v in position.items() 
                           if k not in update_fields + ['position_id']}
        
        if additional_data:
            fields.append(f"additional_data = '{json.dumps(additional_data)}'")
        
        # Build query
        query = f"""
            UPDATE paper_trades
            SET {', '.join(fields)}
            WHERE position_id = '{position['position_id']}'
        """
        
        try:
            with self.db_engine.connect() as conn:
                conn.execute(query)
            self.logger.info(f"Updated trade in database: {position['symbol']}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating trade: {e}")
            return False
    
    def calculate_metrics(self):
        """Calculate trading performance metrics"""
        # Get all closed positions
        closed_positions = self.positions['closed']
        
        # Basic counts
        total_positions = len(closed_positions)
        open_positions = len(self.positions['open'])
        
        if total_positions == 0:
            win_rate = 0
            precision = 0
            economic_gain = 0
            mean_holding_time = 0
            profit_factor = 0
            expectancy = 0
        else:
            # Win/loss stats
            winning_trades = [p for p in closed_positions if p['pnl'] > 0]
            win_count = len(winning_trades)
            win_rate = win_count / total_positions
            
            # Profit metrics
            total_profit = sum(p['pnl'] for p in winning_trades) if winning_trades else 0
            losing_trades = [p for p in closed_positions if p['pnl'] <= 0]
            total_loss = abs(sum(p['pnl'] for p in losing_trades)) if losing_trades else 0
            
            # Precision (similar to win rate but weighted by confidence)
            if sum(p['confidence'] for p in closed_positions) > 0:
                precision = sum(p['confidence'] if p['pnl'] > 0 else 0 for p in closed_positions) / \
                            sum(p['confidence'] for p in closed_positions)
            else:
                precision = 0
            
            # Economic gain (average P&L as percentage of position size)
            economic_gain = sum(p['pnl_percent'] for p in closed_positions) / total_positions
            
            # Holding time in hours
            holding_times = []
            for p in closed_positions:
                if p['exit_time'] and p['entry_time']:
                    delta = (p['exit_time'] - p['entry_time']).total_seconds() / 3600
                    holding_times.append(delta)
            
            mean_holding_time = sum(holding_times) / len(holding_times) if holding_times else 0
            
            # Profit factor & expectancy
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            avg_win = total_profit / win_count if win_count > 0 else 0
            avg_loss = total_loss / len(losing_trades) if losing_trades else 0
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # Compile metrics
        metrics = {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'total_trades': total_positions,
            'open_positions': open_positions,
            'win_rate': win_rate,
            'precision': precision,
            'economic_gain': economic_gain,
            'mean_holding_time': mean_holding_time,
            'profit_factor': profit_factor,
            'expectancy': expectancy
        }
        
        self.logger.info(f"Metrics calculated: Win Rate={win_rate:.2f}, "
                         f"Precision={precision:.2f}, Gain={economic_gain:.2f}%")
        
        return metrics