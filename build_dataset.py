from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import json
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from feature_generator import generate_features
from normalisation import calculate_normalization_params, normalize_features
from analytics import analyze_position_outcomes
from visualisation import plot_feature_distributions, plot_position_analysis, plot_feature_group_comparison


class DatasetMetricsProcessor:
	"""Process and visualize dataset metrics during creation"""
	
	def __init__(self, config):
		self.config = config
		self.plots_dir = Path(config['paths']['dataset_dir'])
		self.plots_dir.mkdir(parents=True, exist_ok=True)

	def process_splits(self, splits):
		"""Process metrics for all dataset splits"""
		split_stats = {}
		for split_name, split_data in splits.items():
			# Basic dataset statistics
			split_stats[split_name] = self._calculate_split_stats(split_data)
			
			# Analyze position outcomes using binary labels for compatibility
			position_stats = analyze_position_outcomes(
				split_data['y_binary'] if 'y_binary' in split_data else split_data['y'],
				split_data['metadata']
			)
			
			# Plot improved position analysis (using binary labels)
			plot_position_analysis(
				analysis_results=position_stats,
				metadata_df=split_data['metadata'],
				labels=split_data['y_binary'] if 'y_binary' in split_data else split_data['y'],
				save_dir=self.plots_dir,
				prefix=f'{split_name}_'
			)
			
			# Plot feature distributions
			plot_feature_distributions(
				features=split_data['unified_features'],
				config=self.config,
				save_dir=self.plots_dir,
				prefix=f'{split_name}_'
			)
			
			plot_feature_group_comparison(split_data['unified_features'], self.config, self.plots_dir)

			# Plot outcome and duration distributions
			plt.figure(figsize=(15, 6))
			
			# 1. Plot derived binary label distribution (based on sign of composite target)
			plt.subplot(1, 3, 1)
			composite_target = split_data['y']
			binary_labels = (composite_target > 0).astype(int)  # Derive binary from composite
			plt.hist(binary_labels, bins=[-0.5, 0.5, 1.5], rwidth=0.8)
			plt.title(f'{split_name} - Binary Outcome Distribution')
			plt.xlabel('Outcome (0=Negative, 1=Positive)')
			plt.ylabel('Count')
			plt.xticks([0, 1])
			
			# 2. Plot original duration distribution
			plt.subplot(1, 3, 2)
			durations = split_data['metadata']['bars_to_exit']
			plt.hist(durations, bins=30)
			plt.title(f'{split_name} - Duration Distribution')
			plt.xlabel('Bars to Exit')
			plt.ylabel('Count')
			
			# 3. Plot composite target distribution
			plt.subplot(1, 3, 3)
			plt.hist(composite_target, bins=30)
			plt.title(f'{split_name} - Composite Target Distribution')
			plt.xlabel('Composite Target [-1 to 1]')
			plt.ylabel('Count')
			
			# Add vertical lines at important values
			plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)  # Decision boundary
			plt.axvline(x=1, color='g', linestyle='--', alpha=0.5)  # Max positive
			plt.axvline(x=-1, color='g', linestyle='--', alpha=0.5)  # Max negative
			
			plt.tight_layout()
			plt.savefig(self.plots_dir / f'{split_name}_outcome_distribution.png')
			plt.close()
				
			# Save statistics
			self._save_split_stats(
				split_name,
				{
					'dataset_stats': split_stats[split_name],
					'position_stats': position_stats
				}
			)
		
		return split_stats

	def _calculate_split_stats(self, split_data):
		"""Calculate statistics for a dataset split with composite target"""
		# Get the composite target
		composite_target = split_data['y']
		
		# Calculate composite target statistics
		positive_count = np.sum(composite_target > 0)
		negative_count = np.sum(composite_target <= 0)
		total_count = len(composite_target)
		
		stats = {
			'n_samples': total_count,
			'n_features': split_data['unified_features'].shape[-1],
			
			# Store composite target statistics
			'composite_target_stats': {
				'min': float(np.min(composite_target)),
				'max': float(np.max(composite_target)),
				'mean': float(np.mean(composite_target)),
				'std': float(np.std(composite_target)),
				'positive_count': int(positive_count),
				'negative_count': int(negative_count),
				'zero_count': int(np.sum(composite_target == 0)),
				'positive_ratio': float(positive_count / total_count),
				'negative_ratio': float(negative_count / total_count)
			},
			
			# For compatibility with existing code
			'class_distribution': {
				'positive': int(positive_count),
				'negative': int(negative_count)
			},
			
			'temporal_range': {
				'start': pd.to_datetime(split_data['metadata']['closeTime'].min(), unit='s'),
				'end': pd.to_datetime(split_data['metadata']['closeTime'].max(), unit='s')
			},
			'assets': {
				'count': split_data['metadata']['code'].nunique(),
				'symbols': split_data['metadata']['code'].unique().tolist()
			}
		}
		
		return stats	

	def _save_split_stats(self, split_name, stats):
		"""Save split statistics to JSON file"""
		stats_file = self.plots_dir / f'{split_name}_analysis.json'
		
		# Convert numpy/pandas types to native Python types
		def convert_types(obj):
			if isinstance(obj, (np.integer, np.floating)):
				return float(obj)
			elif isinstance(obj, np.ndarray):
				return obj.tolist()
			elif isinstance(obj, pd.Timestamp):
				return obj.isoformat()
			elif isinstance(obj, pd.Series):
				return obj.to_list()
			elif isinstance(obj, dict):
				return {k: convert_types(v) for k, v in obj.items()}
			elif isinstance(obj, (list, tuple)):
				return [convert_types(i) for i in obj]
			return obj
		
		converted_stats = convert_types(stats)
		
		with open(stats_file, 'w') as f:
			json.dump(converted_stats, f, indent=4)
			
		print(f"Saved {split_name} analysis to {stats_file}")		
		

def print_split_summary(split_stats):
	"""Print summary of dataset splits with composite target statistics"""
	print("\nDataset Split Summary:")
	print("=" * 50)
	
	for split_name, stats in split_stats.items():
		print(f"\n{split_name.upper()} SPLIT:")
		print("-" * 20)
		print(f"Samples: {stats['n_samples']:,}")
		print(f"Features: {stats['n_features']}")
		
		# For composite target stats
		if 'composite_target_stats' in stats:
			ct_stats = stats['composite_target_stats']
			print("\nComposite Target Statistics:")
			print(f"  Min: {ct_stats['min']:.4f}")
			print(f"  Max: {ct_stats['max']:.4f}")
			print(f"  Mean: {ct_stats['mean']:.4f}")
			print(f"  Std: {ct_stats['std']:.4f}")
			print(f"  Positive values: {ct_stats['positive_count']:,} ({ct_stats['positive_ratio']:.1%})")
			print(f"  Negative values: {ct_stats['negative_count']:,} ({ct_stats['negative_ratio']:.1%})")
		else:
			# Derived binary distribution (for backward compatibility)
			binary_distribution = stats.get('class_distribution', {})
			pos = binary_distribution.get('positive', 0)
			neg = binary_distribution.get('negative', 0)
			total = pos + neg
			
			if total > 0:
				print("\nDerived Binary Distribution:")
				print(f"  Positive: {pos:,} ({pos/total:.1%})")
				print(f"  Negative: {neg:,} ({neg/total:.1%})")
		
		print("\nTemporal Range:")
		print(f"Start: {stats['temporal_range']['start']}")
		print(f"End: {stats['temporal_range']['end']}")
		
		print("\nAssets:")
		print(f"Count: {stats['assets']['count']}")
		print(f"Symbols: {', '.join(stats['assets']['symbols'])}")
		print("-" * 50)		
			
		
def connect_to_db():
	"""Establish database connection"""
	# Update these credentials
	return create_engine('mysql+pymysql://ctUser:-023poqw-023@127.0.0.1/ct')


def calculate_atr(df, period=14):
	"""Calculate Average True Range
	
	Args:
		df: DataFrame with 'high', 'low', 'close' columns
		period: ATR period (default 14)
	"""
	# Calculate True Range
	df['tr'] = np.maximum(
		df['high'] - df['low'],
		np.maximum(
			abs(df['high'] - df['close'].shift(1)),
			abs(df['low'] - df['close'].shift(1))
		)
	)
	
	# Calculate ATR
	df['atr'] = df['tr'].rolling(window=period).mean()
	return df

def generate_labels(df, current_idx, target_pct, stop_pct):
	"""
	Generate labels for a single sample by looking forward until either target or stop is hit.
	
	Args:
		df (pd.DataFrame): DataFrame with OHLC data
		current_idx (int): Index of the current bar
		target_pct (float): Target percentage for profit
		stop_pct (float): Stop percentage for loss
		
	Returns:
		tuple: (label, metadata)
			- label: 1 if target hit, 0 if stop hit, None if neither hit
			- metadata: dict containing essential trade information
	"""
	entry_price = df['close'].iloc[current_idx]
	target_price = entry_price * (1 + target_pct)
	stop_price = entry_price * (1 - stop_pct)
	
	# Look at future bars
	future_data = df.iloc[current_idx + 1:]
	future_highs = future_data['high']
	future_lows = future_data['low']
	
	# Track maximum gains/losses
	max_gain = (future_highs.max() - entry_price) / entry_price
	max_loss = (future_lows.min() - entry_price) / entry_price
	
	# Check each future bar for target/stop hit
	for i, (high, low) in enumerate(zip(future_highs, future_lows), 1):
		if high >= target_price:
			return 1, {
				'bars_to_exit': i,
				'max_gain': max_gain,
				'max_loss': max_loss,
				'exit_type': 1,  # 1 = target hit
				'target_pct': target_pct,
				'stop_pct': stop_pct
				# Removed exit_price
			}
		
		if low <= stop_price:
			return 0, {
				'bars_to_exit': i,
				'max_gain': max_gain,
				'max_loss': max_loss,
				'exit_type': 2,  # 2 = stop hit
				'target_pct': target_pct,
				'stop_pct': stop_pct
				# Removed exit_price
			}
	
	# If we reach here, neither target nor stop was hit
	return None, None

def precompute_labels(df, config, label_params):
	"""
	Precompute labels for all valid indices in the dataframe.
	
	Args:
		df: DataFrame with OHLCV data
		config: Configuration dictionary
		label_params: Label generation parameters
		
	Returns:
		pd.DataFrame: DataFrame with original data and added label information
	"""
	source_width = config['feature_generation']['source_width']
	valid_indices = range(source_width, len(df)-1)
	
	# Lists to store label data
	label_data = []
	
	for idx in tqdm(valid_indices, desc="Generating labels"):
		# Calculate dynamic target and stop
		current_atr = df['atr'].iloc[idx]
		entry_price = df['close'].iloc[idx]
		
		if current_atr <= 0 or entry_price <= 0:
			continue
			
		target_pct = (current_atr * label_params['atr_target_mult']) / entry_price
		stop_pct = (current_atr * label_params['atr_stop_mult']) / entry_price
		
		# Generate label
		label, metadata = generate_labels(df, idx, target_pct, stop_pct)
		
		if label is None or metadata is None:
			continue
			
		# Add index and basic info
		metadata['df_index'] = idx  # Keep this temporarily for mapping
		metadata['label'] = label   # Keep this temporarily for processing
		metadata['code'] = df['code'].iloc[idx]
		metadata['closeTime'] = df['closeTime'].iloc[idx]
		
		# Ensure all required fields exist
		required_fields = {
			'bars_to_exit': 0,
			'max_gain': 0.0,
			'max_loss': 0.0,
			'exit_type': 0,
			'target_pct': target_pct,
			'stop_pct': stop_pct
		}
		
		# Add any missing fields with default values
		for field, default_value in required_fields.items():
			if field not in metadata:
				metadata[field] = default_value
		
		label_data.append(metadata)
	
	# Convert to DataFrame
	label_df = pd.DataFrame(label_data)
	
	return label_df

def process_window_with_precomputed(window_data, config, label_info=None):
	"""
	Process a single window of data using precomputed labels.
	
	Args:
		window_data: DataFrame slice containing the window data
		config: Configuration dictionary
		label_info: Dictionary with precomputed label information
		
	Returns:
		dict: Contains processed data or None if processing failed
	"""
	# Skip if this index doesn't have precomputed label
	if label_info is None:
		return None
	
	# Calculate normalization parameters from the window
	norm_params = calculate_normalization_params(window_data, config)
	
	# Generate unified features
	unified_features = generate_features(window_data, config)
	
	# Safety check for NaN/Inf values before normalization
	if np.isnan(unified_features).any() or np.isinf(unified_features).any():
		print(f"Warning: NaN/Inf values found in features")
		return None
	
	# Normalize features
	normalized_features = normalize_features(unified_features, norm_params)
	
	# Safety check for NaN/Inf values after normalization
	if np.isnan(normalized_features).any() or np.isinf(normalized_features).any():
		print(f"Warning: NaN/Inf values found after normalization")
		return None
	
	# Create metadata with only essential fields
	metadata = {
		'bars_to_exit': label_info['bars_to_exit'],
		'max_gain': label_info['max_gain'],
		'max_loss': label_info['max_loss'],
		'exit_type': label_info['exit_type'],
		'target_pct': label_info['target_pct'],
		'stop_pct': label_info['stop_pct'],
		'code': label_info['code'],
		'closeTime': label_info['closeTime']
		# Removed: df_index, exit_price, label, and norm_* fields
	}
	
	return {
		'unified_features': normalized_features,
		'label': label_info['label'],  # Still needed during processing
		'metadata': metadata
	}

def process_single_window_wrapper(args):
	"""
	Wrapper function for process_window_with_precomputed to work with multiprocessing.
	This needs to be at module level (not nested) for pickling to work.
	"""
	window_data, config, label_info = args
	return process_window_with_precomputed(window_data, config, label_info)

def parallel_feature_generation_with_precomputed(df, label_df, config, n_workers=None):
	"""
	Parallel implementation of feature generation using precomputed labels
	
	Args:
		df: Input DataFrame with OHLCV data
		label_df: DataFrame with precomputed labels
		config: Configuration dictionary
		n_workers: Number of worker processes (defaults to CPU count)
		
	Returns:
		dict: Contains processed data
	"""
	source_width = config['feature_generation']['source_width']
	
	# Create a mapping from df_index to label information
	label_mapping = {row['df_index']: row.to_dict() for _, row in label_df.iterrows()}
	
	# Create list of valid indices that have labels
	valid_indices = [idx for idx in label_mapping.keys()]
	
	# Pre-extract window data for each valid index
	window_data_list = []
	label_infos = []
	
	for idx in valid_indices:
		start_idx = idx - source_width
		end_idx = idx + 1
		
		# Skip if window boundaries are invalid
		if start_idx < 0 or end_idx > len(df):
			continue
			
		# Extract the window data
		window_data = df.iloc[start_idx:end_idx].copy()
		window_data_list.append(window_data)
		label_infos.append(label_mapping[idx])
	
	# Create a list of arguments for the parallel processing
	process_args = [(window_data, config, label_info) 
					for window_data, label_info in zip(window_data_list, label_infos)]
	
	# Initialize lists to store results
	all_features = []
	all_labels = []
	metadata_list = []
	skipped_count = 0
	
	# Process windows in parallel
	with ProcessPoolExecutor(max_workers=n_workers) as executor:
		# Use tqdm to show progress
		results = list(tqdm(
			executor.map(process_single_window_wrapper, process_args),
			total=len(process_args),
			desc="Processing windows with unified features"
		))
		
		# Process results
		for result in results:
			if result is not None:
				all_features.append(result['unified_features'])
				all_labels.append(result['label'])
				metadata_list.append(result['metadata'])
			else:
				skipped_count += 1
	
	# Convert to arrays
	X = np.array(all_features)
	y = np.array(all_labels)
	
	return {
		'unified_features': np.array(all_features),
		'labels': np.array(all_labels),
		'metadata': pd.DataFrame(metadata_list),
		'skipped_count': skipped_count
	}

def convert_numpy_types(obj):
	"""Convert numpy types to native Python types for JSON serialization"""
	if isinstance(obj, np.integer):
		return int(obj)
	elif isinstance(obj, np.floating):
		return float(obj)
	elif isinstance(obj, np.ndarray):
		return obj.tolist()
	elif isinstance(obj, dict):
		return {key: convert_numpy_types(value) for key, value in obj.items()}
	elif isinstance(obj, list):
		return [convert_numpy_types(item) for item in obj]
	return obj

def save_json_stats(stats, filepath):
	"""Save statistics to JSON file with proper type conversion"""
	converted_stats = convert_numpy_types(stats)
	with open(filepath, 'w') as f:
		json.dump(converted_stats, f, indent=4)

def date_to_epoch(date_str):
	"""Convert YYYY-MM-DD string to Unix epoch timestamp"""
	return int(pd.to_datetime(date_str).timestamp())

def fetch_multi_asset_data(engine, config):
	"""Fetch raw data for multiple trading pairs from database"""
	codes = config['dataset_params']['codes']
	basis = config['dataset_params']['basis']
	limit = config['dataset_params'].get('limit', 0)
	limit_clause = f"LIMIT {limit}" if limit > 0 else ''
	
	# Check if we need to fetch all available assets
	if codes == "*":
		# First fetch all available asset codes for the given basis
		code_query = f"""
			SELECT DISTINCT code 
			FROM chart 
			WHERE basis = '{basis}'
		"""
		codes_df = pd.read_sql_query(code_query, engine)
		codes = codes_df['code'].tolist()
		
		print(f"Found {len(codes)} available assets for basis '{basis}'")
	elif isinstance(codes, str):
		codes = [codes]  # Convert single code to list
	
	# Fetch data for each code
	all_data = []
	for code in codes:
		query = f"""
			SELECT 
				b.open, b.high, b.low, b.close, b.volume,
				b.closeTime,
				'{code}' as code  -- Add code identifier
			FROM bar b
			JOIN chart c ON b.chart_id = c.id
			WHERE c.code = '{code}' 
			AND c.basis = '{basis}'
			ORDER BY b.closeTime ASC
			{limit_clause}
		"""
		df = pd.read_sql_query(query, engine)
		
		# Skip if no data
		if len(df) == 0:
			print(f"No data found for {code}, skipping...")
			continue
			
		print(f"Retrieved {len(df)} rows for {code}")
		all_data.append(df)
	
	if not all_data:
		raise ValueError(f"No data retrieved for any of the specified assets with basis '{basis}'")
		
	# Combine all datasets
	combined_df = pd.concat(all_data, axis=0)
	combined_df = combined_df.sort_values('closeTime')
	
	print(f"Combined dataset: {len(combined_df)} rows across {combined_df['code'].nunique()} assets")
	return combined_df

def split_by_dates(data, metadata_df, config):
	"""Split dataset by dates with improved algorithm to handle non-linear data density
	
	This improved algorithm accounts for increasing sample density over time
	(e.g., when more assets are added throughout the dataset timespan)
	
	Args:
		data: Dictionary containing unified_features and labels
		metadata_df: DataFrame containing metadata including closeTime
		config: Configuration dictionary containing split parameters
		
	Returns:
		dict: Dictionary containing train, val, test splits
	"""
	if 'closeTime' not in metadata_df.columns:
		raise ValueError("No closeTime column found in metadata")
		
	# Get split parameters from config
	dataset_params = config.get('dataset_params', {})
	
	# Check if max sizes are specified
	max_val_size = dataset_params.get('max_val_size', None)
	max_test_size = dataset_params.get('max_test_size', None)
	
	# Fall back to proportions if max sizes not specified
	train_prop = dataset_params.get('train_proportion', 0.7)
	val_prop = dataset_params.get('val_proportion', 0.15)
	
	# Sort timestamps and count total samples
	metadata_df = metadata_df.sort_values('closeTime')
	timestamps = metadata_df['closeTime'].unique()
	total_timestamps = len(timestamps)
	total_samples = len(metadata_df)
	
	print(f"\nTotal dataset: {total_samples} samples across {total_timestamps} unique timestamps")
	
	# Create a cumulative sample count over time
	timestamp_sample_counts = metadata_df.groupby('closeTime').size()
	cumulative_samples = timestamp_sample_counts.cumsum()
	
	# Calculate target samples for each split
	if max_val_size and max_test_size:
		# Use max sizes for validation and test, remainder for training
		val_size = min(max_val_size, int(total_samples * val_prop))
		test_size = min(max_test_size, int(total_samples * (1 - train_prop - val_prop)))
	else:
		# Use proportions
		val_size = int(total_samples * val_prop)
		test_size = int(total_samples * (1 - train_prop - val_prop))

	train_size = total_samples - val_size - test_size
	
	print(f"Target split sizes: Train={train_size}, Val={val_size}, Test={test_size}")
	
	# Binary search to find split points based on cumulative sample counts
	def binary_search_split(cumulative_counts, target_count):
		"""Find the timestamp index where cumulative count is closest to target"""
		left, right = 0, len(cumulative_counts) - 1
		
		while left <= right:
			mid = (left + right) // 2
			mid_count = cumulative_counts.iloc[mid]
			
			if mid_count == target_count:
				return mid
			elif mid_count < target_count:
				left = mid + 1
			else:
				right = mid - 1
		
		# Handle edge cases and return best match
		if left >= len(cumulative_counts):
			return len(cumulative_counts) - 1
		if right < 0:
			return 0
			
		# Return the index that is closest to target count
		left_diff = abs(cumulative_counts.iloc[left] - target_count) if left < len(cumulative_counts) else float('inf')
		right_diff = abs(cumulative_counts.iloc[right] - target_count) if right >= 0 else float('inf')
		
		return left if left_diff < right_diff else right
	
	# Find split timestamps using binary search
	val_start_idx = binary_search_split(cumulative_samples, train_size)
	test_start_idx = binary_search_split(cumulative_samples, train_size + val_size)
	
	# Get actual timestamp values
	val_start_epoch = cumulative_samples.index[val_start_idx]
	test_start_epoch = cumulative_samples.index[test_start_idx]
	
	# Print the split dates for visibility
	val_date = pd.to_datetime(val_start_epoch, unit='s')
	test_date = pd.to_datetime(test_start_epoch, unit='s')
	print(f"\nImproved split dates calculated:")
	print(f"Validation start: {val_date.strftime('%Y-%m-%d')} (epoch: {val_start_epoch})")
	print(f"Test start: {test_date.strftime('%Y-%m-%d')} (epoch: {test_start_epoch})")
	
	# Create masks for splits
	train_mask = metadata_df['closeTime'] < val_start_epoch
	val_mask = (metadata_df['closeTime'] >= val_start_epoch) & (metadata_df['closeTime'] < test_start_epoch)
	test_mask = metadata_df['closeTime'] >= test_start_epoch
	
	# Calculate actual sizes and proportions achieved
	split_sizes = {
		'train': train_mask.sum(),
		'val': val_mask.sum(),
		'test': test_mask.sum()
	}
	
	actual_props = {
		'train': split_sizes['train'] / total_samples,
		'val': split_sizes['val'] / total_samples,
		'test': split_sizes['test'] / total_samples
	}
	
	print("\nActual split sizes and proportions achieved:")
	print(f"Train: {split_sizes['train']} samples ({actual_props['train']:.1%})")
	print(f"Validation: {split_sizes['val']} samples ({actual_props['val']:.1%})")
	print(f"Test: {split_sizes['test']} samples ({actual_props['test']:.1%})")
	
	# Visualize the timestamp distribution and split points
	try:
		import matplotlib.pyplot as plt
		from pathlib import Path
		
		# Create a figure to visualize sample distribution
		plt.figure(figsize=(12, 6))
		timestamps_series = pd.Series(metadata_df['closeTime'])
		timestamp_counts = timestamps_series.value_counts().sort_index()
		cumulative_counts = timestamp_counts.cumsum()
		
		# Convert timestamps to dates for better visualization
		date_index = pd.to_datetime(timestamp_counts.index, unit='s')
		plt.plot(date_index, cumulative_counts, 'b-', label='Cumulative Sample Count')
		
		# Mark split points
		val_date_index = pd.to_datetime(val_start_epoch, unit='s')
		test_date_index = pd.to_datetime(test_start_epoch, unit='s')
		
		plt.axvline(x=val_date_index, color='g', linestyle='--', 
				   label=f'Val Split ({split_sizes["val"]} samples)')
		plt.axvline(x=test_date_index, color='r', linestyle='--', 
				   label=f'Test Split ({split_sizes["test"]} samples)')
		
		plt.title('Dataset Split Points with Cumulative Sample Distribution')
		plt.xlabel('Date')
		plt.ylabel('Cumulative Samples')
		plt.legend()
		plt.grid(True, alpha=0.3)
		
		# Save the visualization
		save_dir = Path(config['paths'].get('dataset_dir', '.'))
		plt.savefig(save_dir / 'split_distribution.png', dpi=100, bbox_inches='tight')
		plt.close()
		print(f"Split visualization saved to: {save_dir / 'split_distribution.png'}")
	except Exception as e:
		print(f"Could not create visualization: {e}")
	
	# Create splits dictionary
	splits = {}
	for name, mask in [('train', train_mask), ('val', val_mask), ('test', test_mask)]:
		splits[name] = {
			'unified_features': data['unified_features'][mask],
			'y': data['labels'][mask],
			'metadata': metadata_df[mask].copy()
		}
		
	return splits

def _transform_outcome(outcome, duration, k=0.05, shift=-1.0, max_duration=None):
	"""
	Transform financial outcome (1 or 0) and duration into a single value in [-1, 1].
	
	Parameters:
	-----------
	outcome : int or array-like
		Binary outcome (1 for positive, 0 for negative)
	duration : int or array-like
		Number of bars until outcome is reached
	k : float, default=0.02
		Controls the rate at which values approach zero as duration increases
	shift : float, default=5.0
		Shift applied to durations to avoid most values clustering at extremes
	max_duration : int, default=200
		Maximum duration to consider for normalization
		
	Returns:
	--------
	float or array-like
		Transformed value(s) in range [-1, 1]
	"""
	# Convert inputs to numpy arrays
	outcome = np.array(outcome, dtype=float)
	duration = np.array(duration, dtype=float)
	
	# Cap duration at max_duration if specified
	if max_duration is not None:
		duration = np.minimum(duration, max_duration)
	
	# Map outcome from {0, 1} to {-1, 1}
	sign = 2 * outcome - 1
	
	# Apply shift to durations to better distribute values
	shifted_duration = duration + shift
	
	# Use inverse duration to create a hyperbolic decay
	# This spreads out the values better than exponential for short durations
	duration_factor = 1.0 / (1.0 + k * shifted_duration)
	
	# Apply the sign to get final transformed outcome
	transformed = sign * duration_factor
	
	return transformed

def precompute_multi_asset_labels(df, config, label_params):
	"""Precompute labels for multiple assets separately"""
	all_labels = []
	
	for code in df['code'].unique():
		asset_df = df[df['code'] == code].copy()
		asset_df = asset_df.reset_index(drop=True)
		
		# Skip if insufficient data
		if len(asset_df) <= config['feature_generation']['source_width']:
			print(f"Warning: Insufficient data for {code}, skipping...")
			continue
			
		# Precompute labels for this asset
		asset_labels = precompute_labels(asset_df, config, label_params)
		
		# Add to combined results
		if not asset_labels.empty:
			all_labels.append(asset_labels)
	
	# Combine all asset labels
	combined_labels = pd.concat(all_labels, axis=0) if all_labels else pd.DataFrame()
	
	return combined_labels

def parallel_multi_asset_feature_generation_with_precomputed(df, label_df, config, n_workers=None):
	"""Parallel feature generation for multiple assets using precomputed labels"""
	# Group labels by code
	all_features = []
	all_labels = []
	all_metadata = []
	total_skipped = 0
	
	for code in df['code'].unique():
		# Skip assets that have no labels
		if code not in label_df['code'].unique():
			print(f"No labels for {code}, skipping...")
			continue
			
		asset_df = df[df['code'] == code].copy()
		asset_df = asset_df.reset_index(drop=True)
		
		# Get labels for this asset
		asset_labels = label_df[label_df['code'] == code].copy()
		
		# Create mapping from original df indices to reset indices
		orig_to_reset = {idx: i for i, idx in enumerate(asset_df.index)}
		
		# Update df_index in labels to match reset indices
		asset_labels['df_index'] = asset_labels['df_index'].map(
			lambda x: orig_to_reset.get(x, -1))
		
		# Filter out invalid indices
		asset_labels = asset_labels[asset_labels['df_index'] >= 0]
		
		if asset_labels.empty:
			print(f"No valid labels after index mapping for {code}, skipping...")
			continue
			
		# Process this asset
		results = parallel_feature_generation_with_precomputed(
			asset_df, asset_labels, config, n_workers)
		
		if results['unified_features'].size > 0:
			all_features.append(results['unified_features'])
			all_labels.append(results['labels'])
			all_metadata.append(results['metadata'])
		
		total_skipped += results['skipped_count']
	
	if not all_features:
		raise ValueError("No valid samples were generated during feature generation")
		
	return {
		'unified_features': np.vstack(all_features),
		'labels': np.concatenate(all_labels),
		'metadata': pd.concat(all_metadata, axis=0),
		'skipped_count': total_skipped
	}

def save_datasets(splits, dataset_dir):
	"""
	Save split datasets to disk with binary classification labels.
	
	Uses the original binary outcome rather than the composite target that combined
	outcome and duration information.
	"""
	for name, data in splits.items():
		# Extract binary labels - positive (1) when outcome is successful, negative (0) otherwise
		binary_labels = (data['y'] > 0).astype(np.float32)
		
		# Save with binary labels as the y value
		torch.save({
			'unified_features': torch.FloatTensor(data['unified_features']),
			'y': torch.FloatTensor(binary_labels),  # Binary targets (0 or 1)
		}, dataset_dir / f'{name}.pt')
		
		# Update the in-memory data structure
		data['y'] = binary_labels  # Replace with binary labels
		
		# Ensure metadata only contains necessary fields
		essential_columns = [
			'bars_to_exit', 'max_gain', 'max_loss', 'exit_type',
			'target_pct', 'stop_pct', 'code', 'closeTime'
		]
		
		# Only keep columns that actually exist in the metadata
		existing_essential_columns = [col for col in essential_columns if col in data['metadata'].columns]
		data['metadata'] = data['metadata'][existing_essential_columns]
		
		# Save metadata
		data['metadata'].to_csv(dataset_dir / f'{name}_metadata.csv', index=False)
		
		# Print statistics about the binary targets
		print(f"{name.upper()} split binary target  :  Positive: {np.sum(binary_labels == 1)} ({np.mean(binary_labels == 1):.1%})  -  Negative: {np.sum(binary_labels == 0)} ({np.mean(binary_labels == 0):.1%})")		
				
def calculate_atr_for_assets(df, period):
	"""Calculate ATR for each asset separately"""
	dfs = []
	for code in df['code'].unique():
		code_df = df[df['code'] == code].copy()
		code_df = calculate_atr(code_df, period=period)
		dfs.append(code_df)
	return pd.concat(dfs, axis=0)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str, required=True,
					  help='Path to configuration JSON file')
	return parser.parse_args()

def main():
	args = parse_args()
	
	# Load configuration
	with open(args.config) as f:
		config = json.load(f)
	
	# Extract paths and parameters
	save_dir = Path(config['paths']['run_dir'])
	dataset_dir = Path(config['paths']['dataset_dir'])
	label_params = config['feature_generation']['label_params']
	print(f"Dataset directory: {dataset_dir}")
	
	# Ensure directories exist
	save_dir.mkdir(parents=True, exist_ok=True)
	dataset_dir.mkdir(parents=True, exist_ok=True)
	
	# Connect and fetch data
	print("Fetching data from database...")
	conn = connect_to_db()
	df = fetch_multi_asset_data(conn, config)
	print(f"Retrieved {len(df)} total bars across {df['code'].nunique()} assets")
	
	# Calculate ATR
	print("\nCalculating ATR...")
	df = calculate_atr_for_assets(df, label_params['atr_period'])
	
	# Precompute all labels first
	all_labels_df = precompute_multi_asset_labels(df, config, label_params)
	print(f"Generated {len(all_labels_df)} valid labels across all assets")
	
	# Generate features using precomputed labels
	print("\nGenerating unified features with precomputed labels...")
	results = parallel_multi_asset_feature_generation_with_precomputed(df, all_labels_df, config)
	
	# Split datasets
	print("\nSplitting datasets by date...")
	splits = split_by_dates(results, results['metadata'], config)
	
	# Save datasets
	print("\nSaving datasets...")
	save_datasets(splits, dataset_dir)
	
	# Process and visualize dataset metrics
	print("\nAnalyzing datasets...")
	metrics_processor = DatasetMetricsProcessor(config)
	split_stats = metrics_processor.process_splits(splits)
	
	# # Print summary
	# print_split_summary(split_stats)
	
	print("\nDataset generation complete!")

if __name__ == "__main__":
	main()