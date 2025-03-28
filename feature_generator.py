import numpy as np
import pandas as pd



def find_latest_swing_points(df, left_pad, right_pad):
    """
    Find the latest swing high and swing low points in the data.
    Uses early termination for efficiency.
    
    Args:
        df: DataFrame with OHLC data
        left_pad: Number of bars to look back for each potential swing point
        right_pad: Number of bars to look ahead for each potential swing point
    
    Returns:
        tuple: (latest_swing_low, latest_swing_high) as numeric indices
    """
    latest_swing_low = None
    latest_swing_high = None
    
    # Start from the most recent bar that has enough right_pad data
    for i in range(len(df) - right_pad - 1, left_pad, -1):
        # Stop when we've found both swing points
        if latest_swing_low is not None and latest_swing_high is not None:
            break
            
        current_bar = df.iloc[i]
        
        # Check for swing low if we haven't found one yet
        if latest_swing_low is None:
            is_swing_low = True
            
            # Check left (past bars)
            for j in range(1, left_pad + 1):
                if df.iloc[i - j]['low'] <= current_bar['low']:
                    is_swing_low = False
                    break
            
            # If still a candidate, check right (future bars)
            if is_swing_low:
                for j in range(1, right_pad + 1):
                    if df.iloc[i + j]['low'] <= current_bar['low']:
                        is_swing_low = False
                        break
            
            # If passed all tests, it's a swing low
            if is_swing_low:
                latest_swing_low = i
        
        # Check for swing high if we haven't found one yet
        if latest_swing_high is None:
            is_swing_high = True
            
            # Check left (past bars)
            for j in range(1, left_pad + 1):
                if df.iloc[i - j]['high'] >= current_bar['high']:
                    is_swing_high = False
                    break
            
            # If still a candidate, check right (future bars)
            if is_swing_high:
                for j in range(1, right_pad + 1):
                    if df.iloc[i + j]['high'] >= current_bar['high']:
                        is_swing_high = False
                        break
            
            # If passed all tests, it's a swing high
            if is_swing_high:
                latest_swing_high = i
    
    return latest_swing_low, latest_swing_high



def calculate_enhanced_vwap_features(df, swing_point_low, swing_point_high, vwap_left_pad):
	# First, calculate the original VWAP lines using existing function
	vwap_support, vwap_resistance = calculate_adaptive_vwaps(
		df, swing_point_low, swing_point_high, vwap_left_pad
	)
	
	# Create reference series for positions within channel
	low_rel_channel = pd.Series(np.nan, index=df.index)
	high_rel_channel = pd.Series(np.nan, index=df.index)
	
	# Reference close for scaling
	reference_close = df['close'].iloc[-1]
	
	# Calculate channel positions
	for i in range(len(df)):
		if pd.notna(vwap_support[i]) and pd.notna(vwap_resistance[i]):
			# Convert VWAP ratios back to price space
			channel_bottom = vwap_support[i] * reference_close
			channel_top = vwap_resistance[i] * reference_close
			
			# Calculate channel width
			channel_width = max(0.00001 * reference_close, channel_top - channel_bottom)
			
			# Calculate relative positions
			current_low = df['low'].iloc[i]
			current_high = df['high'].iloc[i]
			
			# Calculate position as proportion of channel width
			low_rel_channel[i] = (current_low - channel_bottom) / channel_width
			high_rel_channel[i] = (current_high - channel_bottom) / channel_width
	
	# Forward fill any remaining NaN values
	low_rel_channel = low_rel_channel.ffill()
	high_rel_channel = high_rel_channel.ffill()
	
	# Create resulting DataFrame with all four features
	vwap_df = pd.DataFrame({
		'vwap_support': vwap_support,			  # Original VWAP low relative to close
		'vwap_resistance': vwap_resistance,			# Original VWAP high relative to close
		'low_rel_channel': low_rel_channel, # Low price position relative to channel
		'high_rel_channel': high_rel_channel # High price position relative to channel
	})
	
	return vwap_df



def calculate_adaptive_vwaps(df, swing_point_low, swing_point_high, vwap_left_pad):
	"""
	Calculate VWAPs that reset at multiple swing points to ensure full lookback coverage.
	Fixed to properly handle numeric indices returned by find_latest_swing_points.
	
	Args:
		df: DataFrame with OHLC and volume data
		swing_point_low: List of indices of swing lows sorted by recency
		swing_point_high: List of indices of swing highs sorted by recency
		vwap_left_pad: Required lookback period for VWAP calculations
		
	Returns:
		tuple: (vwap_support, vwap_resistance, vwap_low, vwap_high) Series with VWAP values
	"""
	import pandas as pd
	import numpy as np
	
	# Initialize with NaN values
	vwap_support = pd.Series(np.nan, index=df.index)
	vwap_resistance = pd.Series(np.nan, index=df.index)
	
	# Add new series for raw high and low relative to close
	vwap_low = pd.Series(np.nan, index=df.index)
	vwap_high = pd.Series(np.nan, index=df.index)
	
	# Helper function to calculate VWAP from a specific starting point
	def calc_vwap_from_point(idx_position, price_col):
		# Convert numeric position to actual DataFrame position
		if idx_position < 0 or idx_position >= len(df):
			return None
			
		# Get slice from this swing point to the end
		swing_slice = df.iloc[idx_position:].copy()
		
		# Only calculate if we have enough data after this point
		if len(swing_slice) < 2:
			return None
			
		price = swing_slice[price_col]
			
		# Calculate as percentage of close price for better scaling
		reference_close = df['close'].iloc[-1]  # Use last close as reference
		
		cumulative_tpv = (price * swing_slice['volume']).cumsum()
		cumulative_vol = swing_slice['volume'].cumsum()
		
		# Avoid division by zero
		with np.errstate(divide='ignore', invalid='ignore'):
			vwap_values = np.where(cumulative_vol > 0, 
								  cumulative_tpv / cumulative_vol, 
								  price)  # Use price as fallback
			
			# Express as ratio to reference close for better normalization
			vwap_values = vwap_values / reference_close

		return pd.Series(vwap_values, index=swing_slice.index)
	
	# Try to calculate VWAP from swing lows until we have full coverage
	if swing_point_low:
		vwap_from_low = calc_vwap_from_point(swing_point_low, 'low')
		if vwap_from_low is not None:
			# Update the vwap_support series with values from this calculation
			common_indices = vwap_support.index.intersection(vwap_from_low.index)
			if len(common_indices) > 0:
				# Only fill NaN values to ensure more recent swing points take precedence
				mask = vwap_support.loc[common_indices].isna()
				if mask.any():
					vwap_support.loc[common_indices[mask]] = vwap_from_low.loc[common_indices[mask]]
	
	# Try to calculate VWAP from swing highs until we have full coverage
	if swing_point_high:
		vwap_from_high = calc_vwap_from_point(swing_point_high, 'high')
		if vwap_from_high is not None:
			# Update the vwap_resistance series with values from this calculation
			common_indices = vwap_resistance.index.intersection(vwap_from_high.index)
			if len(common_indices) > 0:
				# Only fill NaN values to ensure more recent swing points take precedence
				mask = vwap_resistance.loc[common_indices].isna()
				if mask.any():
					vwap_resistance.loc[common_indices[mask]] = vwap_from_high.loc[common_indices[mask]]
	
	# Forward fill any remaining NaN values
	vwap_support = vwap_support.ffill()
	vwap_resistance = vwap_resistance.ffill()
	
	# Calculate high and low prices as a ratio to reference close
	reference_close = df['close'].iloc[-1]
	vwap_low = df['low'] / reference_close
	vwap_high = df['high'] / reference_close
	
	# Check if we have coverage for the required left_pad
	if vwap_support.iloc[-vwap_left_pad:].isna().any() or vwap_resistance.iloc[-vwap_left_pad:].isna().any():
		# If we still have NaNs, use the average price for those periods
		avg_price = (df['high'] + df['low']) / 2
		
		vwap_support.fillna(avg_price / df['close'].iloc[-1], inplace=True)
		vwap_resistance.fillna(avg_price / df['close'].iloc[-1], inplace=True)

	return vwap_support, vwap_resistance, vwap_low, vwap_high


def calculate_returns(df, columns):
	"""Calculate returns for specified columns"""
	features = pd.DataFrame(index=df.index)
	for col in columns:
		if col == 'volume':
			# For volume, use log difference instead of percentage change
			# This is more stable for highly variable crypto volumes
			log_volume = np.log1p(df[col])
			features[f'{col}_ret'] = log_volume.diff()
		else:
			features[f'{col}_ret'] = df[col].pct_change()
	return features

def calculate_true_range(df):
	"""Calculate true range feature with more aggressive scaling"""
	tr = np.maximum(
		df['high'] - df['low'],
		np.maximum(
			abs(df['high'] - df['close'].shift(1)),
			abs(df['low'] - df['close'].shift(1))
		)
	)
	# Use a power transformation to spread out the values more
	tr_pct = tr / df['close'] * 100  # TR as percentage
	return pd.Series(
		np.power(tr_pct + 1, 0.3) - 1,  # Power transform spreads small values better than log
		index=df.index,
		name='true_range'
	)
	
	
def calculate_volatility_features(df, atr_period=14):
	"""Calculate volatility features with better scaling"""
	features = pd.DataFrame(index=df.index)
	
	# Calculate ATR
	tr = np.maximum(
		df['high'] - df['low'],
		np.maximum(
			abs(df['high'] - df['close'].shift(1)),
			abs(df['low'] - df['close'].shift(1))
		)
	)
	atr = pd.Series(tr).rolling(window=atr_period).mean()
	
	# 1. ATR as percentage with power transform
	atr_pct = atr / df['close'] * 100
	features['vol_atr_pct'] = np.power(atr_pct + 1, 0.3) - 1
	
	# 2. ATR acceleration - scale to similar range as other features
	raw_acc = features['vol_atr_pct'].diff(3)
	# Scale acceleration to range similar to other features
	features['vol_atr_acc'] = raw_acc * 5  # Amplify small changes
	
	# 3. BB width - normalize to similar range
	bb_period = 20
	bb_std = 2
	rolling_mean = df['close'].rolling(window=bb_period).mean()
	rolling_std = df['close'].rolling(window=bb_period).std()
	bb_width = (rolling_std * bb_std * 2) / rolling_mean
	
	# Scale BB width to match other features
	features['vol_bb_width'] = np.power(bb_width * 100 + 1, 0.3) - 1
	
	# 4. Volatility regime - ensure similar range
	vol_rolling_mean = features['vol_atr_pct'].rolling(window=100).mean().fillna(features['vol_atr_pct'].mean())
	vol_ratio = features['vol_atr_pct'] / vol_rolling_mean
	# Apply transformation to ensure similar range to other features
	features['vol_regime'] = np.sign(vol_ratio - 1) * np.power(np.abs(vol_ratio - 1) * 10 + 1, 0.3) - 1
	
	return features


def calculate_swing_features(df, swing_point_low, swing_point_high, required_features):
	"""Calculate swing features with improved scaling"""
	features = pd.DataFrame(index=df.index)
	if 'bars_since_swing_low' in required_features:
		features['bars_since_swing_low'] = np.sqrt(len(df)-swing_point_low)
	if 'bars_since_swing_high' in required_features:
		features['bars_since_swing_high'] = np.sqrt(len(df)-swing_point_high)
	return features



def analyze_feature_importance(config, model, dataloader, device, n_repeats=10):
	"""Analyze feature importance using permutation method with feature groups (updated for simplified model)"""
	test_features = []
	test_labels = []
	for unified_features, labels, durations, metadata in dataloader:
		test_features.append(unified_features.cpu().numpy())
		test_labels.append(labels.cpu().numpy())

	X = np.concatenate(test_features, axis=0)
	y = np.concatenate(test_labels, axis=0)
	
	model.eval()
	X_tensor = torch.FloatTensor(X).to(device)
	y_tensor = torch.LongTensor(y).to(device)
	
	with torch.no_grad():
		class_logits, _, _ = model(X_tensor)  # Note: we ignore duration and predictions
		baseline_loss = F.binary_cross_entropy_with_logits(class_logits, y_tensor.float())
	
	feature_importance = []
	feature_names = []
	
	# Get ordered feature list from feature groups
	for group_name, features in config['feature_generation']['feature_groups'].items():
		feature_names.extend(features)
	
	for feature_idx in range(X.shape[2]):
		feature_losses = []
		for _ in range(n_repeats):
			X_permuted = X_tensor.clone()
			X_permuted[:, :, feature_idx] = X_permuted[:, torch.randperm(X_permuted.shape[1]), feature_idx]
			
			with torch.no_grad():
				perm_class_logits, _, _ = model(X_permuted)  # Note: we ignore duration and predictions
				permuted_loss = F.binary_cross_entropy_with_logits(perm_class_logits, y_tensor.float())
			
			feature_losses.append((permuted_loss - baseline_loss).item())
		
		feature_importance.append({
			'name': feature_names[feature_idx],
			'group': next(group for group, features in config['feature_generation']['feature_groups'].items() 
						 if feature_names[feature_idx] in features),
			'importance': np.mean(feature_losses)
		})

	return {
		'importance_scores': [item['importance'] for item in feature_importance],
		'feature_details': feature_importance,
		'baseline_loss': baseline_loss.item()
	}

def analyze_prediction_distribution(model, dataloader, config, device):
	"""Analyze prediction distribution with uncertainty estimates"""
	model.eval()
	all_target_means = []
	all_target_vars = []
	all_stop_means = []
	all_stop_vars = []
	all_targets = []
	
	feature_groups = config['feature_generation']['feature_groups']
	feature_set = [f for group in feature_groups.values() for f in group]
	
	with torch.no_grad():
		for unified_features, y, durations, metadata in dataloader:
			unified_features = unified_features.to(device)
			
			target_mean, target_var, stop_mean, stop_var, _ = model(unified_features)
			
			target_probs = torch.sigmoid(target_mean)
			stop_probs = torch.sigmoid(stop_mean)
			
			all_target_means.extend(target_probs.cpu().numpy())
			all_target_vars.extend(target_var.cpu().numpy())
			all_stop_means.extend(stop_probs.cpu().numpy())
			all_stop_vars.extend(stop_var.cpu().numpy())
			all_targets.extend(y.cpu().numpy())
	
	# Rest of the function remains the same...
	all_target_means = np.array(all_target_means)
	all_target_vars = np.array(all_target_vars)
	all_stop_means = np.array(all_stop_means)
	all_stop_vars = np.array(all_stop_vars)
	all_targets = np.array(all_targets)
	
	confidence_weights = np.exp(-all_target_vars) * np.exp(-all_stop_vars)
	prediction_scores = all_target_means * confidence_weights
	
	confidence_bins = np.linspace(0, 1, 11)
	hist, _ = np.histogram(prediction_scores, bins=confidence_bins)
	
	return {
		'target_means': all_target_means,
		'target_vars': all_target_vars,
		'stop_means': all_stop_means,
		'stop_vars': all_stop_vars,
		'targets': all_targets,
		'confidence_histogram': hist,
		'confidence_bins': confidence_bins,
		'feature_set': feature_set,
		'prediction_scores': prediction_scores,
		'group_summary': {
			group: len(features) for group, features in feature_groups.items()
		},
		'summary_stats': {
			'target_mean': all_target_means.mean(),
			'target_std': all_target_means.std(),
			'target_range': [all_target_means.min(), all_target_means.max()],
			'stop_mean': all_stop_means.mean(),
			'stop_std': all_stop_means.std(),
			'stop_range': [all_stop_means.min(), all_stop_means.max()],
			'target_var_mean': all_target_vars.mean(),
			'stop_var_mean': all_stop_vars.mean()
		}
	}
	

def analyze_feature_requirements(config):
	"""
	Analyze config to determine which base calculations are needed.
	
	Args:
		config: Configuration dictionary with feature parameters
	
	Returns:
		dict: Requirements for different feature types
	"""
	feature_groups = config['feature_generation']['feature_groups']
	all_features = set(feature for group in feature_groups.values() for feature in group)
	
	requirements = {
		'returns': set(),  # Which return calculations are needed
		'need_true_range': False,
		'need_swing_points': False,
		'need_volatility': False,
		'need_volume_relative': False,  # New flag for volume relative features
		'swing_features': set()  # Which swing-based features are needed
	}
	
	# Check return-based features
	for col in ['open', 'high', 'low', 'close', 'volume']:
		if f'{col}_ret' in all_features:
			requirements['returns'].add(col)
			
	# Check true range and volatility features
	if 'true_range' in all_features:
		requirements['need_true_range'] = True
	if any(f for f in all_features if f.startswith('vol_')):
		requirements['need_volatility'] = True
		
	# Check for volume relative features individually
	vol_rel_features = {
		'vol_rel_short': False,
		'vol_rel_long': False,
		'vol_rel_vlong': False, 
		'vol_trend': False
	}
	
	for feature in vol_rel_features:
		if feature in all_features:
			vol_rel_features[feature] = True
			requirements['need_volume_relative'] = True
			
	# Store which specific volume relative features are needed
	requirements['volume_relative_features'] = {k: v for k, v in vol_rel_features.items() if v}
				
	# Check swing-based features
	swing_features = {'bars_since_swing_low', 'bars_since_swing_high'}
	requirements['swing_features'] = all_features.intersection(swing_features)
	
	# Always need swing points for VWAP calculation
	requirements['need_swing_points'] = any(f for f in all_features if f.startswith('vwap_'))
	
	return requirements

def calculate_volume_relative_features(df, required_features, short_window=20, long_window=50, very_long_window=100):
	"""Calculate volume relative to its rolling averages.
	
	Args:
		df: DataFrame with 'volume' column
		required_features: Dictionary of which features to calculate
		short_window: Short-term rolling window period
		long_window: Long-term rolling window period
		very_long_window: Very long-term rolling window period
		
	Returns:
		DataFrame with volume relative features
	"""
	import pandas as pd
	import numpy as np
	
	features = pd.DataFrame(index=df.index)
	
	# Only calculate the moving averages we need
	need_short = required_features.get('vol_rel_short', False) or required_features.get('vol_trend', False)
	need_long = required_features.get('vol_rel_long', False) or required_features.get('vol_trend', False)
	need_vlong = required_features.get('vol_rel_vlong', False)
	
	# Calculate required moving averages
	if need_short:
		vol_short_mavg = df['volume'].rolling(window=short_window).median()
	
	if need_long:
		vol_long_mavg = df['volume'].rolling(window=long_window).median()
	
	if need_vlong:
		vol_vlong_mavg = df['volume'].rolling(window=very_long_window).median()
	
	# Calculate only the required features
	# 1. Volume relative to short-term average
	if required_features.get('vol_rel_short', False):
		features['vol_rel_short'] = df['volume'] / (vol_short_mavg + 1e-8)  # Avoid division by zero
	
	# 2. Volume relative to long-term average
	if required_features.get('vol_rel_long', False):
		features['vol_rel_long'] = df['volume'] / (vol_long_mavg + 1e-8)
	
	# 3. Volume relative to very long-term average
	if required_features.get('vol_rel_vlong', False):
		features['vol_rel_vlong'] = df['volume'] / (vol_vlong_mavg + 1e-8)
	
	# 4. Short-term vs long-term volume trend
	if required_features.get('vol_trend', False):
		features['vol_trend'] = vol_short_mavg / (vol_long_mavg + 1e-8) - 1.0
	
	# Apply log transformation for better scaling (preserves sign)
	for col in features.columns:
		# Apply signed log transformation for better scaling
		features[col] = np.sign(features[col]) * np.log1p(np.abs(features[col]))
	
	# Replace infinities and NaNs with 0
	features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
	
	return features

def generate_features(window_data, config):
	"""
	Generate unified features according to feature groups.
	
	Args:
		window_data: DataFrame containing historical OHLC data
		config: Configuration dictionary containing feature parameters
	
	Returns:
		numpy.ndarray: Features array of shape (feature_width, n_features)
	"""
	# Verify input data
	if window_data.isnull().any().any():
		window_data = window_data.ffill().bfill()
	
	# Get feature_width parameter
	feature_width = config['feature_generation']['feature_width']
	
	# Analyze feature requirements
	requirements = analyze_feature_requirements(config)
	feature_dfs = []
	
	# Calculate all required features
	if requirements['returns']:
		returns_df = calculate_returns(window_data, requirements['returns'])
		feature_dfs.append(returns_df)
	
	if requirements['need_true_range']:
		true_range = calculate_true_range(window_data)
		feature_dfs.append(pd.DataFrame(true_range))
			
	if requirements['need_volatility']:
		volatility_df = calculate_volatility_features(window_data)
		feature_dfs.append(volatility_df)
	
	# Add calculation for volume relative features
	if requirements['need_volume_relative']:
		# Get parameters from config if available, otherwise use defaults
		vol_params = config.get('volume_relative_params', {})
		short_window = vol_params.get('short_window', 20)
		long_window = vol_params.get('long_window', 50)
		very_long_window = vol_params.get('very_long_window', 100)
		
		# Pass only the required features
		volume_rel_df = calculate_volume_relative_features(
			window_data,
			required_features=requirements['volume_relative_features'],
			short_window=short_window,
			long_window=long_window,
			very_long_window=very_long_window
		)
		feature_dfs.append(volume_rel_df)
	
	# Calculate swing points and VWAPs if needed
	if requirements['need_swing_points']:
		swing_point_low, swing_point_high = find_latest_swing_points(
			window_data,
			left_pad=config['vwap_params']['left_pad'],
			right_pad=config['vwap_params']['right_pad'],
		)
		
		# Handle missing swing points
		if not swing_point_low:
			min_idx = window_data['low'].iloc[-feature_width*2:].idxmin()
			swing_point_low = [min_idx]
		if not swing_point_high:
			max_idx = window_data['high'].iloc[-feature_width*2:].idxmax()
			swing_point_high = [max_idx]
			
		# Calculate swing features if needed
		if requirements['swing_features']:
			swing_df = calculate_swing_features(
				window_data,
				swing_point_low,
				swing_point_high,
				requirements['swing_features']
			)
			feature_dfs.append(swing_df)
			
		# Calculate VWAPs if needed
		feature_groups = config['feature_generation']['feature_groups']
		if any(f.startswith('vwap_') for f in feature_groups.get('vwap', [])):
			vwap_support, vwap_resistance, vwap_low, vwap_high = calculate_adaptive_vwaps(
				window_data, 
				swing_point_low,
				swing_point_high,
				feature_width
			)
			vwap_df = pd.DataFrame({
				'vwap_support': vwap_support,
				'vwap_resistance': vwap_resistance,
				'vwap_low': vwap_low,
				'vwap_high': vwap_high
			})
			feature_dfs.append(vwap_df)
    	
	# Combine all calculated features
	if feature_dfs:
		all_features = pd.concat(feature_dfs, axis=1)
		all_features = all_features.ffill().fillna(0)
		all_features = all_features.replace([np.inf, -np.inf], np.nan).fillna(0)
	else:
		raise ValueError("No features specified in config")
	
	# Ensure correct feature length
	if len(window_data) < feature_width:
		raise ValueError(f"Window data length {len(window_data)} is less than required feature_width {feature_width}")
	
	# Take the last 'feature_width' bars and extract features in order of feature groups
	recent_features = all_features.iloc[-feature_width:]
	ordered_features = []
	
	# Get features in the order specified by feature groups
	for group_features in config['feature_generation']['feature_groups'].values():
		group_features = [f for f in group_features if f in recent_features.columns]
		if group_features:
			ordered_features.extend(group_features)
	
	unified_features = recent_features[ordered_features].values
	
	# Final NaN check
	if np.isnan(unified_features).any():
		unified_features = np.nan_to_num(unified_features, 0)
	
	return unified_features	