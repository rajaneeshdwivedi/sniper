import argparse
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from paper_trading_engine import PaperTradingEngine

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run paper trading engine')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--iterations', type=int, default=None, help='Number of iterations to run (default: run indefinitely)')
    parser.add_argument('--interval', type=int, default=3600, help='Interval between iterations in seconds (default: 3600 [1 hour])')
    parser.add_argument('--report', action='store_true', help='Generate performance report after running')
    return parser.parse_args()

def generate_performance_report(engine, save_dir=None):
    """Generate a performance report with charts"""
    # Create directory for reports
    if save_dir is None:
        save_dir = Path(engine.config.get('paths', {}).get('report_dir', 'reports'))
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Get data
    closed_positions = engine.positions['closed']
    
    if not closed_positions:
        print("No closed positions available for report")
        return
    
    # Create positions DataFrame
    positions_df = pd.DataFrame(closed_positions)
    
    # Calculate metrics
    metrics = engine.calculate_metrics()
    
    # 1. Create equity curve
    plt.figure(figsize=(12, 6))
    equity_data = pd.DataFrame(engine.equity_curve, columns=['timestamp', 'value'])
    plt.plot(equity_data['timestamp'], equity_data['value'], 'b-')
    plt.title('Paper Trading Equity Curve')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / 'equity_curve.png')
    plt.close()
    
    # 2. Create P&L distribution
    plt.figure(figsize=(10, 6))
    positions_df['pnl'].hist(bins=20)
    plt.title('P&L Distribution')
    plt.xlabel('P&L')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / 'pnl_distribution.png')
    plt.close()
    
    # 3. Confidence vs P&L scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(positions_df['confidence'], positions_df['pnl_percent'], alpha=0.6)
    plt.title('Confidence vs P&L %')
    plt.xlabel('Confidence')
    plt.ylabel('P&L %')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / 'confidence_vs_pnl.png')
    plt.close()
    
    # 4. Exit reason distribution
    plt.figure(figsize=(8, 6))
    positions_df['exit_reason'].value_counts().plot(kind='bar')
    plt.title('Exit Reason Distribution')
    plt.xlabel('Exit Reason')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(save_dir / 'exit_reasons.png')
    plt.close()
    
    # 5. Generate summary text file
    with open(save_dir / 'summary.txt', 'w') as f:
        f.write("PAPER TRADING PERFORMANCE SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total Trades: {metrics['total_trades']}\n")
        f.write(f"Win Rate: {metrics['win_rate']:.2%}\n")
        f.write(f"Precision: {metrics['precision']:.2%}\n")
        f.write(f"Economic Gain: {metrics['economic_gain']:.2f}%\n")
        f.write(f"Mean Holding Time: {metrics['mean_holding_time']:.2f} hours\n")
        f.write(f"Profit Factor: {metrics['profit_factor']:.2f}\n")
        f.write(f"Expectancy: {metrics['expectancy']:.2f}\n\n")
        
        f.write(f"Portfolio Value: ${metrics['portfolio_value']:.2f}\n")
        f.write(f"Cash: ${metrics['cash']:.2f}\n")
        f.write(f"Open Positions: {metrics['open_positions']}\n\n")
        
        f.write("SYMBOL PERFORMANCE\n")
        f.write("-" * 30 + "\n")
        
        # Calculate per-symbol metrics
        symbol_metrics = {}
        for pos in closed_positions:
            symbol = pos['symbol']
            if symbol not in symbol_metrics:
                symbol_metrics[symbol] = {
                    'count': 0,
                    'wins': 0,
                    'total_pnl': 0,
                    'pnl_percent': 0
                }
            
            symbol_metrics[symbol]['count'] += 1
            if pos['pnl'] > 0:
                symbol_metrics[symbol]['wins'] += 1
            symbol_metrics[symbol]['total_pnl'] += pos['pnl']
            symbol_metrics[symbol]['pnl_percent'] += pos['pnl_percent']
        
        # Write symbol metrics
        for symbol, metrics in symbol_metrics.items():
            win_rate = metrics['wins'] / metrics['count'] if metrics['count'] > 0 else 0
            avg_pnl_pct = metrics['pnl_percent'] / metrics['count'] if metrics['count'] > 0 else 0
            
            f.write(f"{symbol}:\n")
            f.write(f"  Trades: {metrics['count']}\n")
            f.write(f"  Win Rate: {win_rate:.2%}\n")
            f.write(f"  Total P&L: ${metrics['total_pnl']:.2f}\n")
            f.write(f"  Avg P&L %: {avg_pnl_pct:.2f}%\n\n")
    
    print(f"Performance report generated and saved to {save_dir}")
    return save_dir

def main():
    args = parse_args()
    
    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Create and run the paper trading engine
    engine = PaperTradingEngine(config_path)
    
    try:
        engine.run(iterations=args.iterations, interval=args.interval)
    except KeyboardInterrupt:
        print("\nPaper trading stopped by user")
    
    # Generate report if requested
    if args.report:
        generate_performance_report(engine)

if __name__ == "__main__":
    main()