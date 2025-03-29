from trading_engine import TradingEngine
import pandas as pd
import numpy as np
from datetime import datetime
import json
import uuid
import requests
import hmac
import hashlib
import time
from urllib.parse import urlencode

class LiveTradingEngine(TradingEngine):
    """Live trading implementation that interfaces with Binance API"""
    
    def __init__(self, config_path):
        """Initialize live trading engine"""
        super().__init__(config_path)
        
        # Initialize Binance API credentials
        self.api_key = self.config.get('binance', {}).get('api_key', '')
        self.api_secret = self.config.get('binance', {}).get('api_secret', '')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Binance API key and secret are required for live trading")
        
        # Base URLs
        self.base_url = 'https://api.binance.com'
        
        # Initialize trading parameters
        self.initial_capital = self.config.get('live_trading', {}).get('initial_capital', 10000)
        self.base_risk_pct = self.config.get('live_trading', {}).get('base_risk_percent', 0.01)
        self.max_risk_pct = self.config.get('live_trading', {}).get('max_risk_percent', 0.05)
        self.stop_loss_pct = self.config.get('live_trading', {}).get('stop_loss_percent', 0.02)
        self.take_profit_pct = self.config.get('live_trading', {}).get('base_take_profit_percent', 0.03)
        
        # Confidence threshold
        self.confidence_threshold = self.config['trading'].get('confidence_threshold', 0.7)
        
        # Load open positions
        self.load_open_positions()
        
        # Initialize account state
        self.update_account_info()
        
        self.logger.info("Live trading engine initialized")
    
    def update_account_info(self):
        """Update account information"""
        try:
            account_info = self._send_binance_request('/api/v3/account', 'GET')
            
            # Extract balances
            balances = {}
            for asset in account_info.get('balances', []):
                asset_name = asset['asset']
                free_amount = float(asset['free'])
                locked_amount = float(asset['locked'])
                
                if free_amount > 0 or locked_amount > 0:
                    balances[asset_name] = {
                        'free': free_amount,
                        'locked': locked_amount,
                        'total': free_amount + locked_amount
                    }
            
            self.balances = balances
            
            # Calculate portfolio value in USD
            self.portfolio_value = self._calculate_portfolio_value(balances)
            
            self.logger.info(f"Account info updated: Portfolio value = ${self.portfolio_value:.2f}")
            
            return balances
        except Exception as e:
            self.logger.error(f"Error updating account info: {e}")
            return {}
    
    def _calculate_portfolio_value(self, balances):
        """Calculate portfolio value in USD"""
        # Implementation would include getting current prices and calculating value
        # This is a placeholder for now
        return 10000
    
    def _send_binance_request(self, endpoint, method, params=None):
        """Send a request to the Binance API with authentication"""
        url = self.base_url + endpoint
        
        # Add timestamp to params
        if params is None:
            params = {}
        
        # Add timestamp for signed endpoints
        if endpoint != '/api/v3/ticker/price':
            params['timestamp'] = int(time.time() * 1000)
        
        # Create query string
        query_string = urlencode(params)
        
        # Sign the request
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        query_string = query_string + '&signature=' + signature
        
        # Add API key to headers
        headers = {
            'X-MBX-APIKEY': self.api_key
        }
        
        # Send request
        if method == 'GET':
            response = requests.get(f"{url}?{query_string}", headers=headers)
        elif method == 'POST':
            response = requests.post(f"{url}?{query_string}", headers=headers)
        elif method == 'DELETE':
            response = requests.delete(f"{url}?{query_string}", headers=headers)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Check response
        if response.status_code != 200:
            raise Exception(f"Binance API error: {response.text}")
        
        return response.json()
    
    def get_market_price(self, symbol):
        """Get current market price for a symbol"""
        try:
            response = self._send_binance_request('/api/v3/ticker/price', 'GET', {'symbol': symbol})
            return float(response['price'])
        except Exception as e:
            self.logger.error(f"Error getting market price for {symbol}: {e}")
            return None
    
    def execute_trades(self, signals):
        """Execute live trades based on generated signals"""
        trades_executed = 0
        
        for signal in signals:
            # Only execute trades for buy signals that meet our confidence threshold
            if (signal['action'] == 'buy' and 
                signal['confidence'] >= self.confidence_threshold and 
                signal['probability'] >= 0.55):
                
                # Skip if we already have an open position for this symbol
                if any(p['symbol'] == signal['symbol'] for p in self.positions['open']):
                    self.logger.info(f"Skipping trade for {signal['symbol']} - already have an open position")
                    continue
                
                # Execute the trade
                position = self.execute_live_trade(signal)
                
                if position:
                    trades_executed += 1
        
        self.logger.info(f"Executed {trades_executed} live trades")
        return trades_executed
    
    def execute_live_trade(self, signal):
        """Execute a single live trade on Binance"""
        symbol = signal['symbol']
        confidence = signal['confidence']
        
        # Calculate position size
        position_size_usd = self.calculate_position_size(confidence)
        
        # Get current price
        current_price = self.get_market_price(symbol)
        
        if current_price is None:
            self.logger.error(f"Could not get price for {symbol}")
            return None
        
        # Calculate quantity based on price and position size
        quantity = position_size_usd / current_price
        
        # Round quantity to appropriate precision
        quantity = self.round_step_size(symbol, quantity)
        
        # Calculate stop loss and take profit levels
        stop_loss = current_price * (1 - self.stop_loss_pct)
        take_profit = current_price * (1 + self.take_profit_pct * (1 + confidence))
        
        # Round stop loss and take profit to appropriate precision
        stop_loss = self.round_price(symbol, stop_loss)
        take_profit = self.round_price(symbol, take_profit)
        
        # Place market order
        try:
            order = self._send_binance_request('/api/v3/order', 'POST', {
                'symbol': symbol,
                'side': 'BUY',
                'type': 'MARKET',
                'quantity': quantity
            })
            
            # Generate position ID
            position_id = str(uuid.uuid4())
            
            # Create position record
            position = {
                'position_id': position_id,
                'symbol': symbol,
                'direction': 'long',
                'amount': float(order['executedQty']),
                'entry_price': float(order['fills'][0]['price']) if 'fills' in order else current_price,
                'current_price': current_price,
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
                'exit_reason': None,
                'order_id': order['orderId']
            }
            
            # Add to open positions
            self.positions['open'].append(position)
            
            # Log trade to database
            self.log_trade(position)
            
            # Place stop loss order
            stop_order = self._send_binance_request('/api/v3/order', 'POST', {
                'symbol': symbol,
                'side': 'SELL',
                'type': 'STOP_LOSS_LIMIT',
                'timeInForce': 'GTC',
                'quantity': float(order['executedQty']),
                'price': stop_loss,
                'stopPrice': stop_loss * 1.01  # Trigger slightly above stop price
            })
            
            # Save stop loss order ID
            position['stop_order_id'] = stop_order['orderId']
            self.update_trade(position)
            
            self.logger.info(f"Executed live trade for {symbol}: quantity={quantity}, "
                           f"price={position['entry_price']}, stop={stop_loss}, "
                           f"target={take_profit}")
            
            return position
            
        except Exception as e:
            self.logger.error(f"Error executing live trade for {symbol}: {e}")
            return None
    
    def round_step_size(self, symbol, quantity):
        """Round quantity to appropriate step size for the symbol"""
        # This is a placeholder - you would implement proper rounding
        # based on exchange info for the symbol
        return float(f"{quantity:.6f}")
    
    def round_price(self, symbol, price):
        """Round price to appropriate precision for the symbol"""
        # This is a placeholder - you would implement proper rounding
        # based on exchange info for the symbol
        return float(f"{price:.2f}")
    
    def calculate_position_size(self, confidence):
        """Calculate position size based on confidence and risk parameters"""
        # Adjust risk based on confidence
        risk_pct = self.base_risk_pct * (1 + confidence)
        
        # Cap to maximum risk
        risk_pct = min(risk_pct, self.max_risk_pct)
        
        # Calculate size
        position_size = self.portfolio_value * risk_pct
        
        return position_size
    
    def update_positions(self):
        """Update open positions and check for stop/target hits"""
        if not self.positions['open']:
            return
        
        # Get latest prices for all symbols with open positions
        symbols = {p['symbol'] for p in self.positions['open']}
        latest_prices = {}
        
        for symbol in symbols:
            price = self.get_market_price(symbol)
            if price:
                latest_prices[symbol] = price
        
        # Check each position
        for position in self.positions['open']:
            symbol = position['symbol']
            
            if symbol not in latest_prices:
                continue
            
            current_price = latest_prices[symbol]
            position['current_price'] = current_price
            
            # Calculate unrealized P&L
            position['pnl'] = (current_price - position['entry_price']) * position['amount']
            position['pnl_percent'] = (current_price / position['entry_price'] - 1) * 100
            
            # Check order status - have any of our stop orders been triggered?
            try:
                # Check if position has a stop order ID
                if 'stop_order_id' in position:
                    order_status = self._send_binance_request('/api/v3/order', 'GET', {
                        'symbol': symbol,
                        'orderId': position['stop_order_id']
                    })
                    
                    # If order executed, close position
                    if order_status['status'] == 'FILLED':
                        position['exit_reason'] = 'stop_loss'
                        position['exit_price'] = float(order_status['price'])
                        position['exit_time'] = datetime.fromtimestamp(order_status['updateTime'] / 1000)
                        self.close_position(position)
                        continue
                
                # Check for take profit manually
                if current_price >= position['take_profit']:
                    # Execute market sell
                    self.close_position(position, 'take_profit')
            
            except Exception as e:
                self.logger.error(f"Error checking order status for {symbol}: {e}")
        
        # Update portfolio value
        self.update_account_info()
    
    def close_position(self, position, reason=None):
        """Close a live position by selling on the exchange"""
        symbol = position['symbol']
        position_id = position['position_id']
        
        # If reason provided, use it
        if reason:
            position['exit_reason'] = reason
        
        try:
            # Execute market sell order
            order = self._send_binance_request('/api/v3/order', 'POST', {
                'symbol': symbol,
                'side': 'SELL',
                'type': 'MARKET',
                'quantity': position['amount']
            })
            
            # Update position
            position['exit_time'] = datetime.now()
            position['exit_price'] = float(order['fills'][0]['price']) if 'fills' in order else position['current_price']
            position['status'] = 'closed'
            
            # Calculate final P&L
            position['pnl'] = (position['exit_price'] - position['entry_price']) * position['amount']
            position['pnl_percent'] = (position['exit_price'] / position['entry_price'] - 1) * 100
            
            # Move to closed positions
            self.positions['closed'].append(position)
            
            # Remove from open positions
            self.positions['open'] = [p for p in self.positions['open'] if p['position_id'] != position_id]
            
            # Update database
            self.update_trade(position)
            
            # Cancel any remaining orders for this position
            if 'stop_order_id' in position:
                try:
                    self._send_binance_request('/api/v3/order', 'DELETE', {
                        'symbol': symbol,
                        'orderId': position['stop_order_id']
                    })
                except:
                    # Ignore errors when canceling orders that might already be filled or canceled
                    pass
            
            self.logger.info(f"Closed position for {symbol} at {position['exit_price']}: "
                           f"P&L={position['pnl']:.2f} ({position['pnl_percent']:.2f}%), "
                           f"reason={position['exit_reason']}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error closing position for {symbol}: {e}")
            return False
    
    def calculate_metrics(self):
        """Calculate trading performance metrics"""
        # Similar to paper trading implementation
        # This is a placeholder - you would implement performance metrics calculation
        # based on closed positions
        
        metrics = {
            'portfolio_value': self.portfolio_value,
            'total_trades': len(self.positions['closed']),
            'open_positions': len(self.positions['open']),
            'win_rate': 0.5,  # Placeholder
            'precision': 0.5,  # Placeholder
            'economic_gain': 0.0,  # Placeholder
            'mean_holding_time': 0.0,  # Placeholder
            'profit_factor': 1.0,  # Placeholder
            'expectancy': 0.0  # Placeholder
        }
        
        return metrics