import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score

def calculate_precision_gain_at_rg1(prediction_scores, targets):
	"""
	Calculate the precision gain at recall gain = 1 from prediction scores and targets.
	
	Args:
			prediction_scores: Array of model prediction scores/probabilities
			targets: Array of true labels (0/1)
			
	Returns:
			float: Precision gain at recall gain = 1
	"""
	from sklearn.metrics import precision_recall_curve
	import numpy as np
	
	# Calculate Precision-Recall curve
	precision, recall, _ = precision_recall_curve(targets, prediction_scores)
	
	# Calculate baseline (random classifier)
	baseline = np.mean(targets)
	
	# Calculate Precision Gain and Recall Gain
	precision_gain = (precision - baseline) / (1 - baseline + 1e-10)
	recall_gain = recall / (baseline + 1e-10)
	
	# Find the precision gain at recall gain closest to 1.0
	sorted_indices = np.argsort(np.abs(recall_gain - 1.0))
	y_intersection_idx = sorted_indices[0]  # Index of point closest to recall gain = 1.0
	return float(precision_gain[y_intersection_idx])


def analyze_position_outcomes(labels, metadata_df):
	"""Analyze position outcomes to understand trade characteristics"""
	if len(labels) == 0 or len(metadata_df) == 0:
		raise ValueError("Empty labels or metadata provided")
		
	labels = np.array(labels)
	
	total_positions = len(labels)
	profitable_positions = np.sum(labels == 1)
	win_rate = profitable_positions / total_positions if total_positions > 0 else 0
	
	required_columns = ['bars_to_exit', 'max_gain', 'max_loss', 'exit_type']
	if not all(col in metadata_df.columns for col in required_columns):
		raise KeyError(f"Missing required columns in metadata: {[col for col in required_columns if col not in metadata_df.columns]}")
	
	return {
		'win_rate': win_rate,
		'avg_bars_to_exit': metadata_df['bars_to_exit'].mean(),
		'avg_bars_to_exit_wins': metadata_df.loc[labels == 1, 'bars_to_exit'].mean(),
		'avg_bars_to_exit_losses': metadata_df.loc[labels == 0, 'bars_to_exit'].mean(),
		'max_gain_avg': metadata_df['max_gain'].mean(),
		'max_loss_avg': metadata_df['max_loss'].mean(),
		'exit_types': metadata_df['exit_type'].value_counts().to_dict(),
		'total_positions': total_positions,
		'profitable_positions': profitable_positions,
		'rolling_metrics': {
			'wins': pd.Series(labels).rolling(min(50, len(labels) // 10)).mean(),
			'duration': pd.Series(metadata_df['bars_to_exit']).rolling(min(50, len(labels) // 10)).mean()
		}
	}

def calculate_composite_score(confidence_scores, targets, metadata_df, trade_pcts=[0.1, 0.25, 0.5], 
										 weights=[0.0, 0.0, 1.0], min_trades_per_pct=30):
		"""
		Calculate model evaluation metrics including AUC and threshold-based metrics
		
		Updated to work with targets that may be continuous values or binary.
		When targets are continuous, we convert to binary by checking if > 0
		"""
		if len(trade_pcts) != len(weights) or abs(sum(weights) - 1.0) > 1e-6:
				raise ValueError("Trade percentages and weights must have same length and weights must sum to 1")
				
		total_samples = len(confidence_scores)
		
		# Convert targets to binary for metrics calculation if they're not already
		binary_targets = targets
		if not np.array_equal(targets, targets.astype(bool).astype(targets.dtype)):
				# If targets contain values other than 0 and 1, convert to binary
				binary_targets = (targets > 0.5).astype(int)
		
		# Calculate base rate from binary targets
		base_rate = float(np.mean(binary_targets))
		
		# Convert to numpy arrays to ensure consistent types
		confidence_scores = np.array(confidence_scores)
		binary_targets = np.array(binary_targets)
		
		# Calculate AUC using binary targets
		try:
				auc = roc_auc_score(binary_targets, confidence_scores)
		except Exception as e:
				print(f"Warning: Could not calculate AUC: {e}")
				auc = 0.5
		
		sorted_indices = np.argsort(confidence_scores)[::-1]
		sorted_scores = confidence_scores[sorted_indices]
		sorted_targets = binary_targets[sorted_indices]
		sorted_metadata = metadata_df.iloc[sorted_indices].reset_index(drop=True)
		
		# Create metrics dictionary with AUC as the main score
		metrics = {
				'auc': auc,
				'threshold_metrics': {},
				'total_samples': total_samples,
				'base_rate': base_rate
		}
		
		# Calculate metrics at specific percentages
		best_score = -float('inf')
		best_metrics = None
		best_threshold = 0.5
		
		# For each percentage threshold, calculate metrics
		for idx, (pct, weight) in enumerate(zip(trade_pcts, weights)):
				n_trades = max(min_trades_per_pct, int(total_samples * pct))
				if n_trades > total_samples:
						continue
						
				curr_metadata = sorted_metadata.iloc[:n_trades]
				curr_targets = sorted_targets[:n_trades]
				curr_threshold = sorted_scores[n_trades - 1]
				
				# Use metadata values if available, otherwise use defaults
				if 'target_pct' in curr_metadata.columns and 'stop_pct' in curr_metadata.columns:
						target_pct = float(curr_metadata['target_pct'].mean())
						stop_pct = float(curr_metadata['stop_pct'].mean())
				else:
						target_pct = 0.015
						stop_pct = 0.01
				
				threshold_metrics = calculate_threshold_metrics(
						curr_targets,
						target_pct,
						stop_pct,
						curr_threshold,
						base_rate
				)
				
				metrics['threshold_metrics'][pct] = threshold_metrics
				
				# Update best metrics
				if idx == 1 and threshold_metrics['balanced_precision'] > best_score:
						best_score = threshold_metrics['balanced_precision']
						best_metrics = threshold_metrics
						best_threshold = curr_threshold
		
		# If we couldn't calculate any threshold metrics, create a default
		if not metrics['threshold_metrics']:
				pct = 0.05  # Use the middle value
				metrics['threshold_metrics'][pct] = {
						'threshold': 0.5,
						'precision': base_rate,
						'balanced_precision': 0.5,
						'expected_gain': 0.0,
						'precision_lift': 0.0,
				}
				best_metrics = metrics['threshold_metrics'][pct]
				best_threshold = 0.5
		
		metrics['optimal'] = best_metrics
		
		# Add confusion matrix calculation using best threshold
		metrics['confusion_matrix'] = calculate_confusion_matrix(
				confidence_scores >= best_threshold,
				binary_targets
		)
		
		return metrics

def calculate_confusion_matrix(predictions, targets):
	"""Calculate confusion matrix and derived metrics"""
	cm = confusion_matrix(targets, predictions)
	return {
		'matrix': cm,
		'metrics': {
			'true_negatives': int(cm[0, 0]),
			'false_positives': int(cm[0, 1]),
			'false_negatives': int(cm[1, 0]),
			'true_positives': int(cm[1, 1]),
			'accuracy': float((cm[0, 0] + cm[1, 1]) / np.sum(cm))
		}
	}
		
def calculate_expected_gain(targets, target_pct, stop_pct):
	"""Calculate expected gain using actual target/stop percentages"""
	precision = float(np.mean(targets))
	return (precision * target_pct) - ((1 - precision) * stop_pct)


def calculate_threshold_metrics(targets, target_pct, stop_pct, threshold, base_rate):
	"""Calculate comprehensive metrics for a specific threshold"""
	precision = float(np.mean(targets))
	expected_gain = calculate_expected_gain(targets, target_pct, stop_pct)

	if precision >= base_rate:
		precision_lift = (precision - base_rate) / (1 - base_rate)
	else:
		precision_lift = (precision - base_rate) / base_rate
	
	# Calculate balanced_precision (rescaled to 0 to 1)
	balanced_precision = (precision_lift + 1) / 2
	
			
	return {
		'threshold': float(threshold),
		'precision': precision,
		'balanced_precision': float(balanced_precision),  # Added new metric
		'expected_gain': float(expected_gain),
		'precision_lift': float(precision_lift),
	}
	
	
def analyze_feature_importance(config, model, dataloader, device, n_repeats=25, use_grouped=False):
	"""
	Analyze feature importance using improved permutation method with more stability
	
	Args:
		config: Model configuration
		model: Trained model
		dataloader: Data loader for test data
		device: Computation device
		n_repeats: Number of permutation repeats (increased for stability)
		use_grouped: Whether to permute features by group (more stable, but less granular)
	"""
	device = torch.device('cpu')
	model.to(device)
	test_features = []
	test_labels = []
	
	# Handle both the old and new dataloader format
	for batch in dataloader:
		if len(batch) == 3:  # New format: features, targets, metadata
			unified_features, targets, metadata = batch
		elif len(batch) == 4:  # Old format: features, targets, durations, metadata
			unified_features, targets, durations, metadata = batch
		else:
			raise ValueError(f"Unexpected batch format with {len(batch)} elements")
			
		test_features.append(unified_features.cpu().numpy())
		test_labels.append(targets.cpu().numpy())

	X = np.concatenate(test_features, axis=0)
	y = np.concatenate(test_labels, axis=0)
	
	model.eval()
	X_tensor = torch.FloatTensor(X).to(device)
	y_tensor = torch.FloatTensor(y).to(device)
	
	with torch.no_grad():
		# Get unified output from model
		unified_output, _, _ = model(X_tensor)
		
		# Use MSE loss for the continuous target
		baseline_loss = F.mse_loss(unified_output, y_tensor)
	
	feature_importance = []
	feature_groups = config['feature_generation']['feature_groups']
	
	# Calculate feature indices for each group
	group_indices = {}
	current_idx = 0
	for group_name, features in feature_groups.items():
		start_idx = current_idx
		current_idx += len(features)
		group_indices[group_name] = (start_idx, current_idx, features)
	
	# Decide whether to analyze by group or by individual feature
	if use_grouped:
		# Analyze importance by group
		for group_name, (start_idx, end_idx, features) in group_indices.items():
			if end_idx <= start_idx:
				continue  # Skip empty groups
				
			group_losses = []
			for _ in range(n_repeats):
				X_permuted = X_tensor.clone()
				
				# Permute all features in this group together
				perm_indices = torch.randperm(X_permuted.shape[1])
				for i in range(start_idx, end_idx):
					X_permuted[:, :, i] = X_permuted[:, perm_indices, i]
				
				with torch.no_grad():
					perm_unified_output, _, _ = model(X_permuted)
					permuted_loss = F.mse_loss(perm_unified_output, y_tensor)
				
				group_losses.append((permuted_loss - baseline_loss).item())
			
			# Calculate group importance
			group_importance = np.mean(group_losses)
			
			# Assign the group importance to each feature in the group
			for feature_name in features:
				feature_importance.append({
					'name': feature_name,
					'group': group_name,
					'importance': group_importance / len(features)  # Distribute group importance
				})
	else:
		# Analyze importance per individual feature
		feature_names = []
		for _, features in feature_groups.items():
			feature_names.extend(features)
		
		for feature_idx in range(X.shape[2]):
			feature_losses = []
			for _ in range(n_repeats):
				X_permuted = X_tensor.clone()
				X_permuted[:, :, feature_idx] = X_permuted[:, torch.randperm(X_permuted.shape[1]), feature_idx]
				
				with torch.no_grad():
					perm_unified_output, _, _ = model(X_permuted)
					permuted_loss = F.mse_loss(perm_unified_output, y_tensor)
				
				feature_losses.append((permuted_loss - baseline_loss).item())
			
			# Find which group this feature belongs to
			feature_name = feature_names[feature_idx]
			feature_group = next(group for group, features in feature_groups.items() 
								if feature_name in features)
			
			feature_importance.append({
				'name': feature_name,
				'group': feature_group,
				'importance': np.mean(feature_losses)
			})

	return {
		'importance_scores': [item['importance'] for item in feature_importance],
		'feature_details': feature_importance,
		'baseline_loss': baseline_loss.item(),
		'method': 'group_permutation' if use_grouped else 'individual_permutation'
	}
