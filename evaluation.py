import torch
from pathlib import Path
import numpy as np
import pandas as pd

from analytics import (
	analyze_feature_importance,
	calculate_composite_score,
	calculate_precision_gain_at_rg1 
)
from visualisation import (
	plot_prediction_analysis,
	plot_feature_importance,
	plot_training_history,
	plot_classification_performance,
	plot_confidence_analysis,
	log_test_metrics
)






class ModelEvaluator:
	"""Unified interface for model evaluation and visualization"""

	def __init__(self, model, config, device):
		self.model = model
		self.config = config
		self.device = device
		
		# Initialize metrics_history
		self.metrics_history = {
			'train_loss': [],
			'val_loss': [],
			'val_metrics_by_trades': [],
			'auc_values': [],		   # Renamed from composite_scores
			'precision_gain_at_rg1': [],  # New metric for tracking precision gain at RG=1
			'network_mean': [],
			'network_std': [],
			'confidence_mean': [],   # New metrics for confidence
			'confidence_std': [],	# New metrics for confidence
			'confidence_calibration_error': []  # New metrics for confidence
		}
		
		self.test_metrics = None
		self.network_statistics = {
			'mean': [],
			'std': []
		}


	def evaluate_epoch(self, train_loss, val_loss, val_dataloader):
		"""Evaluate model on validation data and compute metrics"""
		from sklearn.metrics import precision_recall_curve, roc_auc_score
		import numpy as np
		
		self.model.eval()
		all_unified_outputs = []
		all_probs = []
		all_confidence = []
		all_predictions = []
		all_targets = []
		all_metadata = []
		
		with torch.no_grad():
			for features, targets, metadata in val_dataloader:
				features = features.to(self.device)
				targets = targets.to(self.device)
				
				# Forward pass with dual-output model
				probs, confidence, predictions = self.model(features)
				
				# Flatten any multi-dimensional outputs to ensure consistent shapes
				all_unified_outputs.append(probs.cpu().numpy().flatten())
				all_probs.append(probs.cpu().numpy().flatten())
				all_confidence.append(confidence.cpu().numpy().flatten())
				all_predictions.append(predictions.cpu().numpy().flatten())
				all_targets.append(targets.cpu().numpy().flatten())
				all_metadata.append(metadata)
		
		all_unified_outputs = np.concatenate(all_unified_outputs)
		all_probs = np.concatenate(all_probs)
		all_confidence = np.concatenate(all_confidence)
		all_predictions = np.concatenate(all_predictions)
		all_targets = np.concatenate(all_targets)
		metadata_df = pd.concat(all_metadata, ignore_index=True)
		
		# Calculate metrics
		metrics = calculate_composite_score(all_probs, all_targets, metadata_df)
		
		# Calculate precision gain at RG=1
		precision_gain_at_rg1 = calculate_precision_gain_at_rg1(all_probs, all_targets)
		
		# Calculate precision @ k metrics for top 1%, 5%, and 10% with expected returns
		# Pass the trading fees parameter from config if available
		trading_fees = self.config.get('trading_params', {}).get('trading_fees', 0.0005)
		precision_k_metrics = self.calculate_precision_at_k(
			all_probs, all_targets, metadata_df, 
			k_percentages=[0.01, 0.05, 0.1],
			trading_fees=trading_fees
		)
		
		# Calculate confidence calibration error
		prediction_error = np.abs(all_probs - all_targets)
		expected_confidence = 1.0 - prediction_error
		calibration_error = np.mean(np.abs(all_confidence - expected_confidence))
		
		# Ensure train_loss and val_loss are properly extracted as numeric values
		if torch.is_tensor(train_loss):
			train_loss = train_loss.item()
		if torch.is_tensor(val_loss):
			val_loss = val_loss.item()
			
		# Handle dictionary values
		if isinstance(train_loss, dict):
			if 'value' in train_loss:
				train_loss = train_loss['value']
			elif 'total_loss' in train_loss:
				train_loss = train_loss['total_loss']
			else:
				# Find any numeric value in the dictionary
				for k, v in train_loss.items():
					if isinstance(v, (int, float)) or (hasattr(v, 'item') and callable(getattr(v, 'item'))):
						train_loss = v
						break
						
		if isinstance(val_loss, dict):
			if 'value' in val_loss:
				val_loss = val_loss['value']
			elif 'total_loss' in val_loss:
				val_loss = val_loss['total_loss']
			else:
				# Find any numeric value in the dictionary
				for k, v in val_loss.items():
					if isinstance(v, (int, float)) or (hasattr(v, 'item') and callable(getattr(v, 'item'))):
						val_loss = v
						break
						
		# Convert to float for consistency
		try:
			train_loss = float(train_loss)
		except (TypeError, ValueError):
			print(f"Warning: Could not convert train_loss to float: {train_loss}")
			train_loss = 0.0
			
		try:
			val_loss = float(val_loss)
		except (TypeError, ValueError):
			print(f"Warning: Could not convert val_loss to float: {val_loss}")
			val_loss = 0.0
		
		# Check if losses are very small/zero and print warning
		if abs(train_loss) < 1e-10:
			print(f"Warning: train_loss is very close to zero: {train_loss}")
		if abs(val_loss) < 1e-10:
			print(f"Warning: val_loss is very close to zero: {val_loss}")
		
		# Add confidence metrics
		metrics.update({
			'train_loss': train_loss,
			'val_loss': val_loss,
			'precision_gain_at_rg1': precision_gain_at_rg1,
			'avg_confidence': float(np.mean(all_confidence)),
			'confidence_calibration_error': float(calibration_error)
		})
		
		# Add precision @ k metrics
		metrics.update(precision_k_metrics)

		# After collecting all outputs, compute network statistics
		output_mean = float(np.mean(all_unified_outputs))
		output_std = float(np.std(all_unified_outputs))
		confidence_mean = float(np.mean(all_confidence))
		confidence_std = float(np.std(all_confidence))
		
		# Store statistics in history
		self.network_statistics['mean'].append(output_mean)
		self.network_statistics['std'].append(output_std)
		
		# Add statistics to metrics
		metrics.update({
			'network_mean': output_mean,
			'network_std': output_std,
			'confidence_mean': confidence_mean,
			'confidence_std': confidence_std
		})

		return metrics

	def evaluate_test(self, test_dataloader, test_epoch, save_dir=None):
		"""Evaluate model on test data and generate visualizations"""
		from pathlib import Path
		import numpy as np
		import pandas as pd
		from sklearn.metrics import precision_recall_curve
		
		save_dir = Path(save_dir) if save_dir else None
		
		# Analyze predictions and collect metadata
		self.model.eval()
		all_unified_outputs = []
		all_probs = []
		all_confidence = []
		all_predictions = []
		all_targets = []
		all_metadata = []
		
		with torch.no_grad():
			for features, targets, metadata in test_dataloader:
				features = features.to(self.device)
				targets = targets.to(self.device)
				
				# Forward pass with dual-output model
				probs, confidence, predictions = self.model(features)
				
				# Flatten any multi-dimensional outputs to ensure consistent shapes
				all_unified_outputs.append(probs.cpu().numpy().flatten())
				all_probs.append(probs.cpu().numpy().flatten())
				all_confidence.append(confidence.cpu().numpy().flatten())
				all_predictions.append(predictions.cpu().numpy().flatten())
				all_targets.append(targets.cpu().numpy().flatten())
				all_metadata.append(metadata)
		
		all_unified_outputs = np.concatenate(all_unified_outputs)
		all_probs = np.concatenate(all_probs)
		all_confidence = np.concatenate(all_confidence)
		all_predictions = np.concatenate(all_predictions)
		all_targets = np.concatenate(all_targets)
		metadata_df = pd.concat(all_metadata, ignore_index=True)
		
		# Enhanced prediction analysis with precision vs trades data and confidence
		pred_analysis = {
			'prediction_scores': all_probs,
			'confidence_scores': all_confidence,
			'unified_outputs': all_unified_outputs,
			'predictions': all_predictions,
			'targets': all_targets,
			'summary_stats': {
				'target_mean': np.mean(all_unified_outputs),
				'target_std': np.std(all_unified_outputs),
				'confidence_mean': np.mean(all_confidence),
				'confidence_std': np.std(all_confidence),
				'target_var_mean': 0.0,  # Placeholder for compatibility
				'stop_mean': 0.0,        # Placeholder for compatibility
				'stop_std': 0.0,         # Placeholder for compatibility
				'stop_var_mean': 0.0     # Placeholder for compatibility
			},
			'target_means': all_unified_outputs,
			'target_vars': np.zeros_like(all_unified_outputs),  # Placeholder
			'stop_means': np.array([]),  # Empty for compatibility
			'stop_vars': np.array([])    # Empty for compatibility
		}
		
		# Calculate precision gain at RG=1 for final metrics
		final_precision_gain_at_rg1 = calculate_precision_gain_at_rg1(all_probs, all_targets)
		
		# Calculate enhanced precision @ k metrics for top 1%, 5%, and 10%
		# Pass the trading fees parameter from config if available
		trading_fees = self.config.get('trading_params', {}).get('trading_fees', 0.0005)
		precision_k_metrics = self.calculate_precision_at_k(
			all_probs, all_targets, metadata_df,
			k_percentages=[0.01, 0.05, 0.1],
			trading_fees=trading_fees
		)
		
		# Add precision vs trades analysis
		# Sort prediction scores in descending order
		sorted_indices = np.argsort(all_probs)[::-1]
		sorted_scores = all_probs[sorted_indices]
		sorted_targets = all_targets[sorted_indices]
		sorted_confidence = all_confidence[sorted_indices]
		
		# Calculate precision at each position
		cumulative_sum = np.cumsum(sorted_targets)
		positions = np.arange(1, len(sorted_targets) + 1)
		precision_at_k = cumulative_sum / positions
		
		# Add to prediction analysis
		pred_analysis['precision_vs_trades'] = {
			'positions': positions,
			'precision_at_k': precision_at_k,
			'sorted_scores': sorted_scores,
			'sorted_targets': sorted_targets,
			'sorted_confidence': sorted_confidence
		}
		
		# Calculate confidence-correctness correlation
		prediction_error = np.abs(all_probs - all_targets)
		expected_confidence = 1.0 - prediction_error
		confidence_correlation = np.corrcoef(all_confidence, expected_confidence)[0, 1]
		
		# Calculate feature importance
		# try:
		# 	importance_analysis = analyze_feature_importance(
		# 		self.config, self.model, test_dataloader, self.device
		# 	)
		# except Exception as e:
		# 	print(f"Warning: Feature importance analysis failed: {e}")
		# 	importance_analysis = {
		# 		'feature_details': []  # Empty list for compatibility
		# 	}
		
		# Get composite metrics
		test_metrics = calculate_composite_score(
			pred_analysis['prediction_scores'],
			pred_analysis['targets'],
			metadata_df
		)
		
		# Add additional information and confidence metrics
		test_metrics.update({
			'prediction_analysis': pred_analysis,
			# 'feature_importance': importance_analysis,
			'test_epoch': test_epoch,
			'final_precision_gain_at_rg1': final_precision_gain_at_rg1,
			'avg_confidence': float(np.mean(all_confidence)),
			'confidence_std': float(np.std(all_confidence)),
			'confidence_correlation': float(confidence_correlation),
			'confidence_calibration_error': float(np.mean(np.abs(all_confidence - expected_confidence)))
		})
		
		# Add precision @ k metrics to test metrics
		test_metrics.update(precision_k_metrics)
		
		self.test_metrics = test_metrics
		
		# Create visualizations if save_dir provided
		if save_dir:
			save_dir.mkdir(parents=True, exist_ok=True)
			try:
				plot_prediction_analysis(pred_analysis, save_dir)
			except Exception as e:
				print(f"Warning: Failed to create prediction analysis plot: {e}")
				
			# try:
			# 	plot_feature_importance(importance_analysis, save_dir)
			# except Exception as e:
			# 	print(f"Warning: Failed to create feature importance plot: {e}")
			
			if self.metrics_history:
				try:
					plot_training_history(self.metrics_history, test_metrics, save_dir)
				except Exception as e:
					print(f"Warning: Failed to create training history plot: {e}")

			plot_classification_performance(
				pred_analysis['prediction_scores'],
				pred_analysis['targets'],
				save_dir
			)
			
			# Add new confidence visualization
			try:
				plot_confidence_analysis(
					pred_analysis['prediction_scores'],
					pred_analysis['confidence_scores'],
					pred_analysis['targets'],
					save_dir
				)
			except Exception as e:
				print(f"Warning: Failed to create confidence analysis plot: {e}")

		return test_metrics

	def calculate_precision_at_k(self, prediction_scores, targets, metadata_df=None, k_percentages=[0.01, 0.05, 0.1], trading_fees=0.0075):
		"""
		Calculate precision and expected returns at different k values (top k% of predictions)
		
		Args:
			prediction_scores: Array of model prediction scores/probabilities
			targets: Array of true labels (0/1)
			metadata_df: DataFrame containing metadata for each sample (optional)
			k_percentages: List of percentages (0-1) to evaluate
			trading_fees: Trading fees as a decimal (e.g., 0.0075 for 0.75% per transaction)
			
		Returns:
			dict: Precision values and expected returns at each k percentage
		"""
		# Sort by prediction score (descending)
		sorted_indices = np.argsort(prediction_scores)[::-1]
		sorted_targets = targets[sorted_indices]
		
		# Prepare metadata if available
		has_returns_data = False
		if metadata_df is not None:
			# Sort metadata to match sorted predictions
			sorted_metadata = metadata_df.iloc[sorted_indices].reset_index(drop=True)
			
			# Check if we have the necessary columns for return calculations
			has_returns_data = all(col in sorted_metadata.columns for col in ['target_pct', 'stop_pct'])
			
			if has_returns_data:
				# Calculate fee-adjusted returns for each trade
				# Entry fee: 0.75% of entry price
				# Exit fee: 0.75% of exit price
				# For winning trades: target_pct - entry_fee - exit_fee_adjusted
				# For losing trades: -stop_pct - entry_fee - exit_fee_adjusted
				
				entry_fee = trading_fees  # 0.75% of entry price
				
				# For winning trades (target hit)
				# If price rises by target_pct, exit fee is trading_fees * (1 + target_pct)
				win_exit_fee_adjusted = sorted_metadata['target_pct'].apply(
					lambda x: trading_fees * (1 + x)
				)
				win_returns = sorted_metadata['target_pct'] - entry_fee - win_exit_fee_adjusted
				
				# For losing trades (stop hit)
				# If price falls by stop_pct, exit fee is trading_fees * (1 - stop_pct)
				loss_exit_fee_adjusted = sorted_metadata['stop_pct'].apply(
					lambda x: trading_fees * (1 - x)
				)
				loss_returns = -sorted_metadata['stop_pct'] - entry_fee - loss_exit_fee_adjusted
				
				# Create an array of expected returns based on the actual outcome
				# This approach assumes binary outcomes (win/loss) and fixed target/stop values
				expected_returns = np.where(sorted_targets == 1, win_returns, loss_returns)
		
		result = {}
		for k_pct in k_percentages:
			# Calculate k as number of samples
			k = max(1, int(len(targets) * k_pct))
			
			# Calculate precision at k
			precision_at_k = np.mean(sorted_targets[:k])
			
			# Get threshold value at this k
			threshold_at_k = prediction_scores[sorted_indices[k-1]] if k < len(sorted_indices) else 0
			
			# Store basic precision metrics
			result[f'precision_at_{k_pct}'] = float(precision_at_k)
			result[f'threshold_at_{k_pct}'] = float(threshold_at_k)
			result[f'count_at_{k_pct}'] = int(k)
			
			# Calculate expected return if we have the data
			if has_returns_data:
				mean_return_at_k = float(np.mean(expected_returns[:k]))
				result[f'mean_return_at_{k_pct}'] = mean_return_at_k
				
				# Calculate risk metrics
				std_return_at_k = float(np.std(expected_returns[:k]))
				sharpe_at_k = float(mean_return_at_k / std_return_at_k) if std_return_at_k > 0 else 0
				
				result[f'std_return_at_{k_pct}'] = std_return_at_k
				result[f'sharpe_at_{k_pct}'] = sharpe_at_k
				
				# Calculate win/loss statistics
				win_count = int(np.sum(sorted_targets[:k]))
				loss_count = k - win_count
				
				if win_count > 0:
					avg_win = float(np.mean(win_returns[:k][sorted_targets[:k] == 1]))
					result[f'avg_win_at_{k_pct}'] = avg_win
				
				if loss_count > 0:
					avg_loss = float(np.mean(loss_returns[:k][sorted_targets[:k] == 0]))
					result[f'avg_loss_at_{k_pct}'] = avg_loss
					
					# Calculate profit factor (sum of wins / sum of losses)
					sum_wins = float(np.sum(win_returns[:k][sorted_targets[:k] == 1]))
					sum_losses = float(np.sum(-loss_returns[:k][sorted_targets[:k] == 0]))
					
					if sum_losses > 0:
						profit_factor = sum_wins / sum_losses
						result[f'profit_factor_at_{k_pct}'] = float(profit_factor)
		
		return result

	def update_training_history(self, epoch_metrics):
		"""Update training history with metrics from current epoch"""
		# Extract values safely, handling both simple values and dictionaries
		train_loss = epoch_metrics.get('train_loss', 0.0)
		val_loss = epoch_metrics.get('val_loss', 0.0)
		auc = epoch_metrics.get('auc', 0.5)  # Using AUC instead of composite_score
		network_mean = epoch_metrics.get('network_mean', 0.0)
		network_std = epoch_metrics.get('network_std', 0.0)
		precision_gain_at_rg1 = epoch_metrics.get('precision_gain_at_rg1', 0.0)
		confidence_mean = epoch_metrics.get('confidence_mean', 0.5)
		confidence_std = epoch_metrics.get('confidence_std', 0.0)
		confidence_calibration_error = epoch_metrics.get('confidence_calibration_error', 0.0)
		
		# Extract precision @ k metrics and returns data
		precision_at_001 = epoch_metrics.get('precision_at_0.01', 0.0)
		precision_at_005 = epoch_metrics.get('precision_at_0.05', 0.0)
		precision_at_01 = epoch_metrics.get('precision_at_0.1', 0.0)
		
		# Extract return metrics if available
		mean_return_at_001 = epoch_metrics.get('mean_return_at_0.01', 0.0)
		mean_return_at_005 = epoch_metrics.get('mean_return_at_0.05', 0.0)
		mean_return_at_01 = epoch_metrics.get('mean_return_at_0.1', 0.0)
		
		# Extract Sharpe ratio metrics if available
		sharpe_at_001 = epoch_metrics.get('sharpe_at_0.01', 0.0)
		sharpe_at_005 = epoch_metrics.get('sharpe_at_0.05', 0.0)
		sharpe_at_01 = epoch_metrics.get('sharpe_at_0.1', 0.0)
		
		# More robust handling of various input types
		# Handle case where values might be dictionaries or other complex structures
		if isinstance(train_loss, dict) and 'value' in train_loss:
			train_loss = train_loss['value']
		if isinstance(val_loss, dict) and 'value' in val_loss:
			val_loss = val_loss['value']
		if isinstance(auc, dict) and 'value' in auc:
			auc = auc['value']
		
		# Convert to Python float if it's a tensor or any other numeric type
		try:
			train_loss = float(train_loss)
		except (TypeError, ValueError):
			train_loss = 0.0  # Default if conversion fails
			
		try:
			val_loss = float(val_loss)
		except (TypeError, ValueError):
			val_loss = 0.0  # Default if conversion fails
			
		try:
			auc = float(auc)
		except (TypeError, ValueError):
			auc = 0.5  # Default to random classifier performance if conversion fails
		
		# Similar conversion for other metrics...
		try:
			network_mean = float(network_mean)
		except (TypeError, ValueError):
			network_mean = 0.0
			
		try:
			network_std = float(network_std)
		except (TypeError, ValueError):
			network_std = 0.0
			
		try:
			precision_gain_at_rg1 = float(precision_gain_at_rg1)
		except (TypeError, ValueError):
			precision_gain_at_rg1 = 0.0
			
		try:
			confidence_mean = float(confidence_mean)
		except (TypeError, ValueError):
			confidence_mean = 0.5  # Default to middle value for confidence
			
		try:
			confidence_std = float(confidence_std)
		except (TypeError, ValueError):
			confidence_std = 0.0
			
		try:
			confidence_calibration_error = float(confidence_calibration_error)
		except (TypeError, ValueError):
			confidence_calibration_error = 0.0
			
		# Convert precision @ k metrics to float
		try:
			precision_at_001 = float(precision_at_001)
		except (TypeError, ValueError):
			precision_at_001 = 0.0
			
		try:
			precision_at_005 = float(precision_at_005)
		except (TypeError, ValueError):
			precision_at_005 = 0.0
			
		try:
			precision_at_01 = float(precision_at_01)
		except (TypeError, ValueError):
			precision_at_01 = 0.0
			
		# Convert return metrics to float
		try:
			mean_return_at_001 = float(mean_return_at_001)
		except (TypeError, ValueError):
			mean_return_at_001 = 0.0
			
		try:
			mean_return_at_005 = float(mean_return_at_005)
		except (TypeError, ValueError):
			mean_return_at_005 = 0.0
			
		try:
			mean_return_at_01 = float(mean_return_at_01)
		except (TypeError, ValueError):
			mean_return_at_01 = 0.0
			
		# Convert Sharpe metrics to float
		try:
			sharpe_at_001 = float(sharpe_at_001)
		except (TypeError, ValueError):
			sharpe_at_001 = 0.0
			
		try:
			sharpe_at_005 = float(sharpe_at_005)
		except (TypeError, ValueError):
			sharpe_at_005 = 0.0
			
		try:
			sharpe_at_01 = float(sharpe_at_01)
		except (TypeError, ValueError):
			sharpe_at_01 = 0.0
		
		# Initialize history lists if they don't exist yet
		for key in [
			'precision_at_001', 'precision_at_005', 'precision_at_01',
			'mean_return_at_001', 'mean_return_at_005', 'mean_return_at_01',
			'sharpe_at_001', 'sharpe_at_005', 'sharpe_at_01'
		]:
			if key not in self.metrics_history:
				self.metrics_history[key] = []
		
		# Store the float values - ensuring we have valid values even if conversions failed
		self.metrics_history['train_loss'].append(train_loss)
		self.metrics_history['val_loss'].append(val_loss)
		self.metrics_history['auc_values'].append(auc)
		self.metrics_history['precision_gain_at_rg1'].append(precision_gain_at_rg1)
		self.metrics_history['network_mean'].append(network_mean)
		self.metrics_history['network_std'].append(network_std)
		self.metrics_history['confidence_mean'].append(confidence_mean)
		self.metrics_history['confidence_std'].append(confidence_std)
		self.metrics_history['confidence_calibration_error'].append(confidence_calibration_error)
		
		# Store precision @ k metrics
		self.metrics_history['precision_at_001'].append(precision_at_001)
		self.metrics_history['precision_at_005'].append(precision_at_005)
		self.metrics_history['precision_at_01'].append(precision_at_01)
		
		# Store return metrics
		self.metrics_history['mean_return_at_001'].append(mean_return_at_001)
		self.metrics_history['mean_return_at_005'].append(mean_return_at_005)
		self.metrics_history['mean_return_at_01'].append(mean_return_at_01)
		
		# Store Sharpe metrics
		self.metrics_history['sharpe_at_001'].append(sharpe_at_001)
		self.metrics_history['sharpe_at_005'].append(sharpe_at_005)
		self.metrics_history['sharpe_at_01'].append(sharpe_at_01)
		
		# Store the dictionary separately - add proper default if not available
		threshold_metrics = epoch_metrics.get('threshold_metrics', {})
		if not threshold_metrics:
			# Create a minimal default if not present
			threshold_metrics = {0.1: {'precision': auc, 'expected_gain': 0.0, 'balanced_precision': 0.5}}
			
		self.metrics_history['val_metrics_by_trades'].append({
			'threshold_metrics': threshold_metrics,
			'auc': auc
		})
		
		return epoch_metrics  # Return metrics for easy access by TrainingManager
