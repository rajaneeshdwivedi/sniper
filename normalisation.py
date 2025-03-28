import numpy as np
from scipy import stats
import warnings

def calculate_normalization_params(window_data, config):
	"""
	Calculate normalization parameters for unified features with improved techniques
	
	Args:
		window_data: DataFrame containing raw price/volume data
		config: Configuration dictionary with feature group information
		
	Returns:
		dict: Normalization parameters for different feature groups
	"""
	norm_params = {}
	epsilon = 1e-8
	
	# Use first close price as reference for price normalization
	norm_params['price'] = {
		'reference': window_data['close'].iloc[-1],
		'epsilon': epsilon
	}
	
	# Volume uses robust scaling with more detailed distribution statistics
	volume_data = window_data['volume']
	# Calculate percentiles for better characterization of distribution
	percentiles = [1, 5, 25, 50, 75, 95, 99]
	vol_percentiles = {f'p{p}': volume_data.quantile(p/100) for p in percentiles}
	
	q1 = vol_percentiles['p25']
	q3 = vol_percentiles['p75']
	iqr = max(q3 - q1, epsilon)  # Ensure non-zero IQR
	
	# Calculate additional statistics for distribution validation
	volume_std = volume_data.std()
	volume_skew = stats.skew(volume_data.dropna())
	volume_kurt = stats.kurtosis(volume_data.dropna())
	
	norm_params['volume'] = {
		'median': volume_data.median(),
		'iqr': iqr,
		'min': volume_data.min(),
		'max': volume_data.max(),
		'mean': volume_data.mean(),
		'std': volume_std if volume_std > epsilon else epsilon,
		'skew': volume_skew,
		'kurtosis': volume_kurt,
		'percentiles': vol_percentiles,
		'epsilon': epsilon,
		# Adaptive clipping thresholds based on distribution
		'lower_clip': max(-5, -3 * (iqr / volume_std if volume_std > epsilon else 1)),
		'upper_clip': min(5, 3 * (iqr / volume_std if volume_std > epsilon else 1))
	}
	
	# Calculate enhanced statistics for return features
	return_columns = [col for col in window_data.columns if col.endswith('_ret')]
	if return_columns:
		return_data = window_data[return_columns].values.flatten()
		return_data = return_data[~np.isnan(return_data)]  # Remove NaNs
		
		if len(return_data) > 10:  # Ensure we have enough data points
			# Get percentiles for adaptive clipping
			ret_percentiles = {f'p{p}': np.percentile(return_data, p) for p in percentiles}
			p1, p99 = ret_percentiles['p1'], ret_percentiles['p99']
			
			# Calculate robust statistics
			ret_median = np.median(return_data)
			ret_iqr = np.subtract(*np.percentile(return_data, [75, 25]))
			ret_iqr = max(ret_iqr, epsilon)
			
			# Check for heavy tails using kurtosis
			ret_kurtosis = stats.kurtosis(return_data)
			ret_skew = stats.skew(return_data)
			
			# Determine appropriate scaling factors based on distribution characteristics
			if ret_kurtosis > 3:  # Heavy-tailed distribution
				# Use wider scaling for heavy tails
				scale_factor = min(4.0, max(2.0, 1 + ret_kurtosis / 6))
				# For asymmetric distributions, adjust clipping thresholds
				lower_scale = scale_factor * (1 + max(0, -ret_skew / 2))
				upper_scale = scale_factor * (1 + max(0, ret_skew / 2))
			else:
				lower_scale = upper_scale = 3.0
				
			# Standard stats
			ret_mean = np.mean(return_data)
			ret_std = max(np.std(return_data), epsilon)
		else:
			# Default values if insufficient data
			p1, p99 = -0.1, 0.1
			ret_median = 0
			ret_iqr = 0.01
			ret_mean = 0
			ret_std = 0.01
			ret_kurtosis = 0
			ret_skew = 0
			lower_scale = upper_scale = 3.0
			ret_percentiles = {f'p{p}': 0 for p in percentiles}
			
		norm_params['returns'] = {
			'p1': p1,
			'p99': p99,
			'median': ret_median,
			'iqr': ret_iqr,
			'mean': ret_mean,
			'std': ret_std,
			'kurtosis': ret_kurtosis,
			'skew': ret_skew,
			'percentiles': ret_percentiles,
			'epsilon': epsilon,
			'lower_clip': -lower_scale,
			'upper_clip': upper_scale,
			'distribution_type': 'heavy_tailed' if ret_kurtosis > 3 else 'normal'
		}
	else:
		# Default returns params if no return columns found
		norm_params['returns'] = {
			'p1': -0.1, 'p99': 0.1, 'mean': 0, 'std': 0.01, 'epsilon': epsilon,
			'lower_clip': -3, 'upper_clip': 3, 'distribution_type': 'normal'
		}
	
	# Technical features with enhanced normalization parameters
	tech_features = config['feature_generation']['feature_groups'].get('technical', [])
	tech_params = {}
	
	for feature in tech_features:
		if feature in window_data.columns:
			feature_data = window_data[feature].dropna().values
			
			if len(feature_data) > 0:
				# Calculate distribution statistics
				feat_percentiles = {f'p{p}': np.percentile(feature_data, p) if len(feature_data) > 0 else 0 
								   for p in percentiles}
				feat_median = np.median(feature_data)
				feat_mean = np.mean(feature_data)
				feat_std = max(np.std(feature_data), epsilon)
				
				# Try to calculate skew/kurtosis (may fail for constant data)
				try:
					feat_skew = stats.skew(feature_data)
					feat_kurt = stats.kurtosis(feature_data)
				except:
					feat_skew = 0
					feat_kurt = 0
				
				# Different strategies based on feature type
				if feature.startswith('vol_') or feature == 'true_range':
					# Check for zeros or negatives
					has_zeros = np.any(feature_data == 0)
					has_negatives = np.any(feature_data < 0)
					
					if has_negatives:
						# For features with negative values, use signed log
						transform = 'signed_log'
						# Find appropriate threshold based on data
						threshold = max(1e-4, np.percentile(np.abs(feature_data), 5))
					elif has_zeros:
						# For features with zeros but no negatives, use log1p
						transform = 'log1p'
						# Calculate log of non-zero values for percentiles
						log_data = np.log1p(feature_data[feature_data > 0])
						log_p1 = np.percentile(log_data, 1) if len(log_data) > 0 else 0
						log_p99 = np.percentile(log_data, 99) if len(log_data) > 0 else 5
					else:
						# For strictly positive data, use log
						transform = 'log'
						log_data = np.log(feature_data)
						log_p1 = np.percentile(log_data, 1) if len(log_data) > 0 else 0
						log_p99 = np.percentile(log_data, 99) if len(log_data) > 0 else 5
					
					# Determine adaptive clipping bounds
					if transform in ['log', 'log1p']:
						tech_params[feature] = {
							'p1': log_p1,
							'p99': log_p99,
							'mean': np.mean(log_data) if 'log_data' in locals() and len(log_data) > 0 else 0,
							'std': max(np.std(log_data), epsilon) if 'log_data' in locals() and len(log_data) > 0 else 1,
							'transform': transform,
							'raw_stats': {
								'median': feat_median,
								'mean': feat_mean,
								'std': feat_std,
								'skew': feat_skew,
								'kurtosis': feat_kurt,
								'percentiles': feat_percentiles
							},
							'lower_clip': -3,  # For log-transformed data
							'upper_clip': 3	# For log-transformed data
						}
					else:  # signed_log
						tech_params[feature] = {
							'threshold': threshold,
							'transform': transform,
							'raw_stats': {
								'median': feat_median,
								'mean': feat_mean,
								'std': feat_std,
								'skew': feat_skew,
								'kurtosis': feat_kurt,
								'percentiles': feat_percentiles
							},
							'lower_clip': -3,
							'upper_clip': 3
						}
				
				elif 'bars_since_swing' in feature:
					# For swing indicators: adaptive log(1+x) transform
					max_val = np.max(feature_data)
					log_max = np.log1p(max_val)
					
					tech_params[feature] = {
						'max': max_val,
						'log_max': log_max,
						'transform': 'log1p',
						'raw_stats': {
							'median': feat_median,
							'mean': feat_mean,
							'std': feat_std,
							'skew': feat_skew,
							'kurtosis': feat_kurt,
							'percentiles': feat_percentiles
						},
						'lower_clip': 0,	# Cannot be negative
						'upper_clip': 3	 # After log1p and scaling
					}
				
				else:
					# Determine appropriate transform based on distribution
					if feat_kurt > 5:  # Very heavy tailed
						transform = 'robust_scale'
					elif feat_skew > 2 or feat_skew < -2:  # Highly skewed
						transform = 'quantile'
					else:
						transform = 'zscore'
					
					tech_params[feature] = {
						'median': feat_median,
						'iqr': max(feat_percentiles['p75'] - feat_percentiles['p25'], epsilon),
						'mean': feat_mean,
						'std': feat_std,
						'transform': transform,
						'raw_stats': {
							'skew': feat_skew,
							'kurtosis': feat_kurt,
							'percentiles': feat_percentiles
						},
						'lower_clip': -3,
						'upper_clip': 3
					}
			else:
				# Default params for empty data
				tech_params[feature] = {
					'mean': 0, 'std': 1, 'transform': 'zscore',
					'raw_stats': {'skew': 0, 'kurtosis': 0},
					'lower_clip': -3, 'upper_clip': 3
				}
	
	norm_params['technical'] = tech_params
	
	# VWAP features need additional analysis for better thresholds
	vwap_features = config['feature_generation']['feature_groups'].get('vwap', [])
	vwap_params = {}
	
	if vwap_features:
		vwap_data = []
		for feature in vwap_features:
			if feature in window_data.columns:
				# Calculate relative difference from close price
				close_price = window_data['close'].iloc[-1]
				vwap_values = window_data[feature].dropna().values
				if len(vwap_values) > 0:
					rel_diff = (vwap_values - close_price) / (close_price + epsilon)
					vwap_data.extend(rel_diff)
		
		if vwap_data:
			# Find appropriate threshold based on distribution
			abs_values = np.abs(vwap_data)
			vwap_threshold = max(1e-4, np.percentile(abs_values[abs_values > 0], 5))
			
			# Get adaptive clipping thresholds for transformed data
			vwap_percentiles = {f'p{p}': np.percentile(vwap_data, p) for p in percentiles}
			
			# Calculate statistics for transformed values
			transformed_values = signed_log(np.array(vwap_data), vwap_threshold)
			trans_percentiles = {f'p{p}': np.percentile(transformed_values, p) for p in percentiles}
			
			# Set appropriate clipping thresholds
			lower_clip = max(-5, trans_percentiles['p1'])
			upper_clip = min(5, trans_percentiles['p99'])
			
			norm_params['vwap'] = {
				'threshold': vwap_threshold,
				'percentiles': vwap_percentiles,
				'transformed_percentiles': trans_percentiles,
				'lower_clip': lower_clip,
				'upper_clip': upper_clip
			}
		else:
			# Default VWAP params
			norm_params['vwap'] = {
				'threshold': 1e-4,
				'lower_clip': -5,
				'upper_clip': 5
			}
	
	# Store feature group information
	feature_groups = config['feature_generation']['feature_groups']
	
	# Calculate feature indices for each group
	feature_indices = {}
	current_idx = 0
	
	for group_name, features in feature_groups.items():
		start_idx = current_idx
		current_idx += len(features)
		feature_indices[group_name] = (start_idx, current_idx)
	
	# Store all feature names in order for reference
	all_features = [f for group in feature_groups.values() for f in group]
	
	# Create feature info with more detailed indexing
	norm_params['feature_info'] = {
		'group_indices': feature_indices,
		'n_features': current_idx,
		'feature_names': all_features,
		'return_feature_indices': [i for i, name in enumerate(all_features) if name.endswith('_ret')],
		'volatility_feature_indices': [i for i, name in enumerate(all_features) if name.startswith('vol_') or name == 'true_range'],
		'swing_feature_indices': [i for i, name in enumerate(all_features) if 'bars_since_swing' in name],
		'vwap_feature_indices': [i for i, name in enumerate(all_features) if name.startswith('vwap_')],
	}
	
	# Add global scale normalization parameter
	norm_params['global_scale'] = {
		'target_range': (-1, 1),  # Common target range for all features
		'unified_std': 0.5,	   # Target standard deviation
	}
			
	return norm_params

def signed_log(x, threshold=1e-4):
	"""
	Enhanced symmetric log transformation with adaptive thresholding
	
	Args:
		x: Input value or array
		threshold: Small value threshold below which to use linear scaling
		
	Returns:
		Transformed value maintaining sign but with log scaling
	"""
	# Treat small values linearly to avoid discontinuity at zero
	sign = np.sign(x)
	abs_x = np.abs(x)
	
	result = np.where(
		abs_x > threshold,
		sign * np.log1p(abs_x / threshold),
		x / threshold
	)
	
	return result

def inverse_signed_log(y, threshold=1e-4):
	"""
	Inverse of the signed log transformation
	
	Args:
		y: Transformed value
		threshold: The threshold used in the forward transformation
		
	Returns:
		Original value before transformation
	"""
	sign = np.sign(y)
	abs_y = np.abs(y)
	
	# For values that were log-transformed (abs_y > 1)
	log_mask = abs_y > 1.0
	linear_mask = ~log_mask
	
	result = np.zeros_like(y, dtype=float)
	result[linear_mask] = y[linear_mask] * threshold
	result[log_mask] = sign[log_mask] * (np.exp(abs_y[log_mask]) - 1.0) * threshold
	
	return result

def quantile_transform(x, n_quantiles=100, output_distribution='normal'):
	"""
	Apply quantile transformation to ensure uniform or normal distribution
	
	Args:
		x: Input array
		n_quantiles: Number of quantiles to use
		output_distribution: 'uniform' or 'normal'
		
	Returns:
		Transformed array with desired distribution
	"""
	# Handle NaN values
	not_nan = ~np.isnan(x)
	if np.sum(not_nan) <= 1:
		return np.zeros_like(x)
	
	x_valid = x[not_nan]
	
	# Create output array
	result = np.full_like(x, np.nan)
	
	# Calculate ranks (0 to 1)
	ranks = stats.rankdata(x_valid) / len(x_valid)
	
	# Transform to desired distribution
	if output_distribution == 'normal':
		# Transform to standard normal using inverse CDF
		transformed = stats.norm.ppf(ranks)
		# Clip to avoid extreme values from floating point precision
		transformed = np.clip(transformed, -4, 4)
	else:  # uniform
		transformed = ranks * 2 - 1  # Scale to [-1, 1]
	
	# Put back in original array
	result[not_nan] = transformed
	
	return result

def robust_scale(x, median=None, iqr=None):
	"""
	Apply robust scaling using median and IQR
	
	Args:
		x: Input array
		median: Pre-computed median (optional)
		iqr: Pre-computed IQR (optional)
		
	Returns:
		Robustly scaled array
	"""
	if median is None:
		median = np.median(x)
	
	if iqr is None:
		q75, q25 = np.percentile(x, [75, 25])
		iqr = q75 - q25
	
	# Avoid division by zero
	if iqr == 0:
		iqr = 1e-8
	
	return (x - median) / iqr

def normalize_features(features, norm_params):
	# Create a copy to avoid modifying the original array
	norm_features = features.copy()
	
	# Extract parameters
	feature_info = norm_params['feature_info']
	epsilon = norm_params.get('price', {}).get('epsilon', 1e-8)
	group_indices = feature_info['group_indices']
	
	# 1. Process each feature group separately with robust normalization
	for group_name, (start_idx, end_idx) in group_indices.items():
		group_features = norm_features[:, start_idx:end_idx]
		
		# Skip empty groups
		if group_features.size == 0:
			continue
			
		# Calculate robust statistics for this group
		non_nan = ~np.isnan(group_features)
		if np.sum(non_nan) <= 1:
			continue
			
		# Use percentiles for robust scaling
		q1 = np.percentile(group_features[non_nan], 25)
		q2 = np.percentile(group_features[non_nan], 50)  # median
		q3 = np.percentile(group_features[non_nan], 75)
		iqr = max(q3 - q1, epsilon)
		
		# Center around median and scale by IQR
		norm_features[:, start_idx:end_idx] = (group_features - q2) / iqr
		
		# Apply reasonable clipping
		norm_features[:, start_idx:end_idx] = np.clip(
			norm_features[:, start_idx:end_idx], -3, 3
		)
	
	# 2. Check for any remaining NaN/Inf values
	if np.isnan(norm_features).any() or np.isinf(norm_features).any():
		norm_features = np.nan_to_num(norm_features, nan=0.0, posinf=3.0, neginf=-3.0)
	
	return norm_features