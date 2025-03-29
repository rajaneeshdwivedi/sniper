import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from pathlib import Path
from sqlalchemy import create_engine

# Import directly from feature_generator module
from feature_generator import find_latest_swing_points, calculate_adaptive_vwaps
from build_dataset import fetch_multi_asset_data, connect_to_db

def connect_to_db():
	"""Establish database connection"""
	# Update these credentials to match your configuration
	return create_engine('mysql+pymysql://ctUser:-023poqw-023@127.0.0.1/ct')

def focused_vwap_chart(df, window_size=100, offset=0, left_pad=15, right_pad=10, output_path="focused_vwap.png"):
	"""
	Create a focused chart showing windowed data with swing points and VWAPs
	
	Args:
		df: DataFrame with OHLCV data
		window_size: Size of the window to extract
		offset: Offset from the beginning of the data
		left_pad: Swing point left padding
		right_pad: Swing point right padding
		output_path: Path to save the chart
	"""
	# Ensure we have enough data
	if len(df) < offset + window_size:
		print(f"Warning: Data has only {len(df)} bars, but requested offset {offset} + window {window_size} = {offset + window_size}")
		window_size = min(window_size, len(df) - offset)
		print(f"Adjusted window size to {window_size}")
	
	# Extract the window at the specified offset
	start_idx = offset
	end_idx = offset + window_size
	window_df = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
	print(f"Using windowed data: bars {start_idx} to {end_idx-1} (window size: {len(window_df)} bars)")
	
	# Calculate swing points using feature_generator function
	print("Finding swing points...")
	swing_point_low, swing_point_high = find_latest_swing_points(
		window_df, left_pad=left_pad, right_pad=right_pad
	)
	
	print(f"Found swing lows: {swing_point_low}")
	print(f"Found swing highs: {swing_point_high}")
	
	# Calculate VWAPs using feature_generator function
	print("Calculating VWAPs...")
	vwap_support, vwap_resistance, vwap_low, vwap_high = calculate_adaptive_vwaps(
		window_df, swing_point_low, swing_point_high, left_pad
	)
	
	# Scale VWAPs back to price
	reference_close = window_df['close'].iloc[-1]
	vwap_support_price = vwap_support * reference_close
	vwap_resistance_price = vwap_resistance * reference_close
	
	print(f"VWAP low shape: {vwap_support.shape}, NaN count: {vwap_support.isna().sum()}")
	print(f"VWAP high shape: {vwap_resistance.shape}, NaN count: {vwap_resistance.isna().sum()}")
	
	
	# Create figure
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
	
	# Plot candlesticks for windowed data
	for i in range(len(window_df)):
		# Determine colors
		color = 'green' if window_df['close'].iloc[i] >= window_df['open'].iloc[i] else 'red'
		
		# Plot candle body
		body_bottom = min(window_df['open'].iloc[i], window_df['close'].iloc[i])
		body_height = abs(window_df['close'].iloc[i] - window_df['open'].iloc[i])
		rect = plt.Rectangle((i-0.3, body_bottom), 0.6, body_height, 
						   linewidth=1, edgecolor=color, facecolor=color, alpha=0.7)
		ax1.add_patch(rect)
		
		# Plot wicks
		ax1.plot([i, i], [window_df['low'].iloc[i], body_bottom], color=color, linewidth=1)
		ax1.plot([i, i], [body_bottom+body_height, window_df['high'].iloc[i]], color=color, linewidth=1)
	
	# Plot swing points on windowed data - directly using numeric indices
	ax1.scatter(swing_point_low, window_df['low'].iloc[swing_point_low], color='blue', s=120, marker='^')
	ax1.text(swing_point_low, window_df['low'].iloc[swing_point_low]*0.995, f"Low #{i+1}", fontsize=10, fontweight='bold')
	print(f"Added swing low marker at position {swing_point_low}, price {window_df['low'].iloc[swing_point_low]}")
		
	ax1.scatter(swing_point_high, window_df['high'].iloc[swing_point_high], color='purple', s=120, marker='v')
	ax1.text(swing_point_high, window_df['high'].iloc[swing_point_high]*1.005, f"High #{i+1}", fontsize=10, fontweight='bold')
	print(f"Added swing high marker at position {swing_point_high}, price {window_df['high'].iloc[swing_point_high]}")
	
	# Plot VWAPs
	if not vwap_support_price.isna().all():
		ax1.plot(range(len(window_df)), vwap_support_price.values, 'b--', linewidth=1.5, label='VWAP Low')
	if not vwap_resistance_price.isna().all():
		ax1.plot(range(len(window_df)), vwap_resistance_price.values, 'm--', linewidth=1.5, label='VWAP High')
	
	# Plot volume
	ax2.bar(range(len(window_df)), window_df['volume'], color='gray', alpha=0.7)
	
	# Set up x-axis dates
	date_labels = [pd.to_datetime(ts, unit='s').strftime('%Y-%m-%d') 
				 for ts in window_df['closeTime']]
	
	# Add date labels
	step = max(1, len(window_df) // 10)  # Show at most 10 dates
	ax1.set_xticks(range(0, len(window_df), step))
	ax1.set_xticklabels([date_labels[i] for i in range(0, len(window_df), step)], rotation=45)
	ax2.set_xticks(range(0, len(window_df), step))
	ax2.set_xticklabels([date_labels[i] for i in range(0, len(window_df), step)], rotation=45)
	
	# Set title and labels
	symbol = window_df['code'].iloc[0] if 'code' in window_df.columns else "Asset"
	start_date = pd.to_datetime(window_df['closeTime'].iloc[0], unit='s').strftime('%Y-%m-%d')
	end_date = pd.to_datetime(window_df['closeTime'].iloc[-1], unit='s').strftime('%Y-%m-%d')
	
	# Include offset info in the title
	offset_info = f" (offset: {offset})" if offset > 0 else ""
	fig.suptitle(f"{symbol} Swing Points & VWAP ({start_date} to {end_date}){offset_info}", fontsize=16)
	
	ax1.set_ylabel('Price')
	ax1.grid(True, alpha=0.3)
	ax1.legend(loc='upper left')
	
	ax2.set_ylabel('Volume')
	ax2.grid(True, alpha=0.3)
	
	# Save figure
	plt.tight_layout()
	plt.subplots_adjust(top=0.95)
	plt.savefig(output_path, dpi=150)
	plt.close()
	
	print(f"Chart saved to {output_path}")
	
	return {
		'swing_point_low': swing_point_low,
		'swing_point_high': swing_point_high,
		'vwap_support': vwap_support,
		'vwap_resistance': vwap_resistance
	}

def fetch_full_data(conn, symbol, basis):
	"""Fetch the entire dataset without limit for a symbol"""
	query = f"""
		SELECT 
			b.open, b.high, b.low, b.close, b.volume,
			b.closeTime,
			'{symbol}' as code
		FROM bar b
		JOIN chart c ON b.chart_id = c.id
		WHERE c.code = '{symbol}' 
		AND c.basis = '{basis}'
		ORDER BY b.closeTime ASC
	"""
	df = pd.read_sql_query(query, conn)
	return df

def parse_args():
	parser = argparse.ArgumentParser(description='Create focused VWAP visualization')
	parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Symbol to visualize')
	parser.add_argument('--basis', type=str, default='4h', help='Timeframe basis')
	parser.add_argument('--window', type=int, default=100, help='Size of the window to extract')
	parser.add_argument('--offset', type=int, default=0, help='Offset from the beginning of the data')
	parser.add_argument('--left-pad', type=int, default=15, help='Swing point left padding')
	parser.add_argument('--right-pad', type=int, default=10, help='Swing point right padding')
	parser.add_argument('--output', type=str, default='vwap_chart.png', help='Output file path')
	return parser.parse_args()

def main():
	args = parse_args()
	
	# Connect to database
	conn = connect_to_db()
	
	# Fetch data
	print(f"Fetching all data for {args.symbol} {args.basis}...")
	config = {
		'dataset_params': {
			'basis': args.basis,
			'codes': [args.symbol]
		}
	}

	df = fetch_multi_asset_data(conn, config)
	
	if len(df) == 0:
		print(f"No data found for {args.symbol}")
		return
		
	print(f"Fetched {len(df)} bars for {args.symbol}")
	
	# Create focused chart with offset control
	focused_vwap_chart(
		df, 
		window_size=args.window,
		offset=args.offset,
		left_pad=args.left_pad,
		right_pad=args.right_pad,
		output_path=args.output
	)

if __name__ == "__main__":
	main()