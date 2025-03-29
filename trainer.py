import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import argparse
import json
from model import create_model
from evaluation import ModelEvaluator
from datetime import datetime
import pandas as pd
import math


class TrainingManager:
	"""Centralized manager for model training, early stopping, and checkpointing"""
	
	def __init__(self, model, optimizer, scheduler, criterion, config, device, evaluator):
		self.model = model
		self.optimizer = optimizer
		self.scheduler = scheduler
		self.criterion = criterion
		self.config = config
		self.device = device
		self.evaluator = evaluator
		
		# Early stopping configuration
		self.patience = config['training_params'].get('patience', 10)
		self.min_delta = config['training_params'].get('min_delta', 0.001)
		
		# Initialize early stopping state
		self.best_score = float('-inf')
		self.best_state = None
		self.best_epoch = None
		self.patience_counter = 0
		self.stopped_early = False
		
		# Checkpoint directory
		self.models_dir = Path(config['paths']['models_dir'])
		self.models_dir.mkdir(parents=True, exist_ok=True)
		
		# SWA configuration
		self.use_swa = config['training_params'].get('use_swa', False)
		if self.use_swa:
			self.swa_start = config['training_params'].get('swa_start', 10)
			self.swa_model = AveragedModel(model)
			
			# Create SWA scheduler if specified
			if config['training_params'].get('use_swa_scheduler', False):
				swa_lr = config['training_params'].get('swa_lr', config['training_params']['learning_rate'] / 10)
				anneal_epochs = config['training_params'].get('swa_anneal_epochs', 5)
				self.swa_scheduler = SWALR(
					optimizer, 
					swa_lr=swa_lr,
					anneal_epochs=anneal_epochs,
					anneal_strategy='cos'
				)
				print(f"SWA scheduler initialized: LR={swa_lr}, annealing over {anneal_epochs} epochs")
			else:
				self.swa_scheduler = None
			
			print(f"SWA initialized: starting at epoch {self.swa_start}")
		else:
			self.swa_model = None
			self.swa_scheduler = None
			self.swa_start = None
	
	def train_epoch(self, train_loader, epoch):
		"""Train for a single epoch"""
		self.model.train()
		epoch_losses = []
		
		for batch_idx, batch_data in enumerate(train_loader):
			# Handle different dataloader formats
			if len(batch_data) == 3:  # Features, targets, metadata
				unified_features, targets, metadata = batch_data
			elif len(batch_data) == 2:  # Just features and targets
				unified_features, targets = batch_data
				metadata = None
			else:
				raise ValueError(f"Unexpected batch format with {len(batch_data)} elements")
			
			# Move data to device
			unified_features = unified_features.to(self.device)
			targets = targets.to(self.device)
			
			# Zero gradients before forward pass
			self.optimizer.zero_grad()
			
			# Forward pass
			probs, confidence, predictions = self.model(unified_features)
			
			# Calculate loss
			loss = self.criterion(probs, confidence, targets)
			
			# Backward and optimize
			loss.backward()
			
			# Gradient clipping if configured
			if hasattr(self.model, 'config') and self.model.config['training_params'].get('gradient_clip', 0) > 0:
				nn.utils.clip_grad_norm_(
					self.model.parameters(), 
					self.model.config['training_params']['gradient_clip']
				)
			
			self.optimizer.step()
			
			# Update learning rate if using OneCycleLR scheduler (needs per-batch updates)
			if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
				self.scheduler.step()
			
			# Update SWA model if it's enabled and we've reached the SWA start epoch
			if self.use_swa and epoch >= self.swa_start:
				self.swa_model.update_parameters(self.model)
				
				# If using SWA-specific scheduler, step it here
				if self.swa_scheduler is not None:
					self.swa_scheduler.step()
					
			# Track metrics
			epoch_losses.append(float(loss.item()))
		
		# Compute average metrics
		train_metrics = {
			'total_loss': float(np.mean(epoch_losses))
		}
		
		return train_metrics
	
	def evaluate(self, val_loader, train_loss):
		"""Evaluate model on validation data"""
		# Collect validation metrics using the evaluator
		val_metrics = collect_validation_metrics(self.model, self.criterion, val_loader, self.device)
		
		# Get comprehensive evaluation metrics
		epoch_metrics = self.evaluator.evaluate_epoch(
			train_loss, 
			val_metrics['total_loss'], 
			val_loader
		)
		
		# Update training history
		self.evaluator.update_training_history(epoch_metrics)
		
		return epoch_metrics
	
	def step_scheduler(self, val_loss):
		"""Step the learning rate scheduler if applicable"""
		if not self.scheduler:
			return
			
		# Skip if it's OneCycleLR (handled per batch) or SWA scheduler (handled in train_epoch)
		if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
			return
			
		if self.use_swa and isinstance(self.scheduler, SWALR):
			return
			
		# Step ReduceLROnPlateau with validation loss
		self.scheduler.step()
	

	def check_early_stopping(self, metrics, epoch):
		"""
		Check if training should stop early based on top percentile precision metrics.
		
		Modified to use precision at top 1% as the primary metric for early stopping.
		
		Args:
			metrics: Dictionary of validation metrics
			epoch: Current epoch number
			
		Returns:
			bool: True if training should stop, False otherwise
		"""
		# Extract the score used for early stopping
		current_score = metrics['precision_gain_at_rg1']
		
		if current_score > self.best_score + self.min_delta:
			# Improvement found
			self.best_score = current_score
			self.best_epoch = epoch
			self.patience_counter = 0
			
			# Save best model state
			self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
			
			# Save best checkpoint
			self.save_checkpoint(metrics, epoch=epoch, is_best=True)
			return False
			
		else:
			# No improvement
			self.patience_counter += 1
			
			if self.patience_counter >= self.patience:
				self.stopped_early = True
				return True
				
			return False
	
	def save_checkpoint(self, metrics, epoch=None, is_best=False, is_swa=False):
		"""Save model checkpoint"""
		# Create checkpoint
		checkpoint = {
			'epoch': epoch,
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'score': metrics['precision_gain_at_rg1'],
			'config': self.config,
			'metrics': metrics
		}
		
		if self.scheduler is not None:
			try:
				checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
			except:
				pass  # Some schedulers might not have state_dict
		
		# Save latest checkpoint
		latest_path = self.models_dir / 'latest_checkpoint.pt'
		torch.save(checkpoint, latest_path)
		
		# Save best checkpoint if this is the best model
		if is_best:
			best_path = self.models_dir / 'best_checkpoint.pt'
			torch.save(checkpoint, best_path)
			print(f"\n*******************************************")
			print(f"*** New best model saved. Score: {metrics['precision_gain_at_rg1']:.4f} ***")
			print(f"*******************************************")
		
		# Save SWA checkpoint if it's an SWA model
		if is_swa:
			swa_path = self.models_dir / 'swa_best_checkpoint.pt'
			torch.save(checkpoint, swa_path)
		
		# Save regular checkpoint if epoch is specified and divisible by 10
		if epoch is not None and epoch % 10 == 0:
			epoch_path = self.models_dir / f'model_epoch_{epoch}.pt'
			torch.save(checkpoint, epoch_path)
	
	def restore_best_model(self):
		"""Restore the model to its best state"""
		if self.best_state is not None:
			self.model.load_state_dict(self.best_state)
			print("Restored model to best checkpoint")
		else:
			try:
				checkpoint = torch.load(self.models_dir / 'best_checkpoint.pt')
				self.model.load_state_dict(checkpoint['model_state_dict'])
				print("Restored model from best checkpoint file")
			except:
				print("No best state available to restore")
	
def collect_validation_metrics(model, criterion, val_loader, device):
	"""Collect basic validation metrics"""
	model.eval()
	val_losses = []
	all_probs = []
	all_confidence = []
	all_preds = []
	all_targets = []
	
	with torch.no_grad():
		for batch_data in val_loader:
			# Handle different dataloader formats
			if len(batch_data) == 3:  # Features, targets, metadata
				unified_features, targets, metadata = batch_data
			elif len(batch_data) == 2:  # Just features and targets
				unified_features, targets = batch_data
				metadata = None
			else:
				raise ValueError(f"Unexpected batch format with {len(batch_data)} elements")
			
			# Move data to device
			unified_features = unified_features.to(device)
			targets = targets.to(device)
			
			# Forward pass
			probs, confidence, predictions = model(unified_features)
			
			# Calculate loss
			loss = criterion(probs, confidence, targets)
			
			# Track metrics
			val_losses.append(float(loss.item()))
			
			# Save outputs for metric calculation
			all_probs.append(probs.detach().cpu())
			all_confidence.append(confidence.detach().cpu())
			all_preds.append(predictions.detach().cpu())
			all_targets.append(targets.detach().cpu())
	
	# Concatenate predictions and targets
	all_probs = torch.cat(all_probs).numpy()
	all_confidence = torch.cat(all_confidence).numpy()
	all_preds = torch.cat(all_preds).numpy()
	all_targets = torch.cat(all_targets).numpy()
	
	# Compute ROC AUC if applicable
	roc_auc = 0.5
	try:
		# Convert targets to binary if they're not already
		binary_targets = (all_targets > 0.5).astype(np.int32)
		roc_auc = roc_auc_score(binary_targets, all_probs)
	except Exception as e:
		print(f"Warning: Could not calculate ROC AUC: {str(e)}")
	
	# Calculate calibration error (how well confidence matches correctness)
	prediction_error = np.abs(all_probs - all_targets)
	expected_confidence = 1.0 - prediction_error
	calibration_error = np.mean(np.abs(all_confidence - expected_confidence))
	
	# Calculate confidence-aware metrics
	confidence_weighted_accuracy = np.mean(
		(1 - np.abs(all_targets - all_preds)) * all_confidence
	)
	
	# Compute average metrics
	val_metrics = {
		'total_loss': float(np.mean(val_losses)),
		'auc': float(roc_auc),
		'accuracy': float(np.mean(all_preds == all_targets)),
		'calibration_error': float(calibration_error),
		'avg_confidence': float(np.mean(all_confidence)),
		'confidence_weighted_accuracy': float(confidence_weighted_accuracy)
	}
	
	return val_metrics


def create_scheduler(optimizer, config, train_loader=None):
	"""Create a learning rate scheduler based on configuration"""
	scheduler_type = config['training_params'].get('scheduler_type', 'one_cycle')
	
	if not config['training_params'].get('use_lr_scheduler', False):
		return None
	
	# Retrieve scheduler-specific parameters
	num_epochs = config['training_params']['num_epochs']
	max_lr = config['training_params']['learning_rate']
	min_lr = config['training_params'].get('min_learning_rate', max_lr / 100)
	warmup_epochs = config['training_params'].get('warmup_epochs', 5)
	
	if scheduler_type == 'one_cycle':
		# One Cycle LR
		steps_per_epoch = len(train_loader) if train_loader else 1000
		total_steps = num_epochs * steps_per_epoch
		return torch.optim.lr_scheduler.OneCycleLR(
			optimizer,
			max_lr=max_lr,
			total_steps=total_steps,
			pct_start=0.3,
			anneal_strategy='cos',
			div_factor=25.0,
			final_div_factor=1000.0
		)
	elif scheduler_type == 'cosine':
		# Cosine Annealing
		return torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer,
			T_max=num_epochs,
			eta_min=min_lr
		)
	elif scheduler_type == 'cosine_warmup':
		# Cosine annealing with warmup
		# Create warmup scheduler first
		warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
			optimizer, 
			start_factor=0.1,
			end_factor=1.0,
			total_iters=warmup_epochs
		)
		
		# Create cosine annealing scheduler for after warmup
		cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer,
			T_max=num_epochs - warmup_epochs,
			eta_min=min_lr
		)
		
		# Combine both schedulers
		return torch.optim.lr_scheduler.SequentialLR(
			optimizer,
			schedulers=[warmup_scheduler, cosine_scheduler],
			milestones=[warmup_epochs]
		)
	elif scheduler_type == 'cosine_warmup_restarts':
		# Custom implementation of cosine annealing with warmup and restarts
		class CosineWarmupRestartsScheduler(torch.optim.lr_scheduler._LRScheduler):
			"""
			Scheduler that implements warmup and cosine annealing with restarts.
			"""
			def __init__(self, optimizer, warmup_epochs, cycle_length, num_epochs, 
						min_lr, max_lr, restart_multiplier=1.0, 
						warmup_start_factor=0.1, last_epoch=-1):
				self.warmup_epochs = warmup_epochs
				self.cycle_length = cycle_length  # Length of each cycle before restart
				self.num_epochs = num_epochs
				self.min_lr = min_lr
				self.max_lr = max_lr
				self.restart_multiplier = restart_multiplier  # Multiplier for cycle length after each restart
				self.warmup_start_factor = warmup_start_factor
				super(CosineWarmupRestartsScheduler, self).__init__(optimizer, last_epoch)
				
			def get_lr(self):
				if self.last_epoch < self.warmup_epochs:
					# Linear warmup phase
					alpha = self.last_epoch / self.warmup_epochs
					factor = self.warmup_start_factor + (1 - self.warmup_start_factor) * alpha
					return [base_lr * factor for base_lr in self.base_lrs]
				
				# After warmup - calculate where we are in the cycle
				epoch_since_warmup = self.last_epoch - self.warmup_epochs
				
				# Determine which cycle we're in
				cycle_epoch = epoch_since_warmup
				cycle_idx = 0
				cycle_start = 0
				current_cycle_length = self.cycle_length
				
				# Find the current cycle and the epoch within that cycle
				while cycle_start + current_cycle_length <= epoch_since_warmup:
					cycle_start += current_cycle_length
					cycle_idx += 1
					current_cycle_length = int(self.cycle_length * (self.restart_multiplier ** cycle_idx))
				
				cycle_epoch = epoch_since_warmup - cycle_start
				
				# Apply cosine annealing within the current cycle
				cosine_factor = 0.5 * (1 + math.cos(math.pi * cycle_epoch / current_cycle_length))
				factor = self.min_lr / self.max_lr + cosine_factor * (1 - self.min_lr / self.max_lr)
				
				return [base_lr * factor for base_lr in self.base_lrs]
		
		# Parameters for cosine with restarts
		cycle_length = config['training_params'].get('restart_cycle_length', num_epochs // 3)
		restart_multiplier = config['training_params'].get('restart_multiplier', 1.5)
		
		return CosineWarmupRestartsScheduler(
			optimizer,
			warmup_epochs=warmup_epochs,
			cycle_length=cycle_length,
			num_epochs=num_epochs,
			min_lr=min_lr,
			max_lr=max_lr,
			restart_multiplier=restart_multiplier,
			warmup_start_factor=0.1
		)
	elif scheduler_type == 'reduce_on_plateau':
		# Reduce on Plateau
		return torch.optim.lr_scheduler.ReduceLROnPlateau(
			optimizer,
			mode='max',
			factor=0.5,
			patience=5,
			verbose=True
		)
	elif scheduler_type == 'swa':
		# SWA learning rate scheduler
		swa_lr = config['training_params'].get('swa_lr', min_lr)
		return SWALR(
			optimizer,
			swa_lr=swa_lr,
			anneal_epochs=config['training_params'].get('swa_anneal_epochs', 5),
			anneal_strategy='cos'
		)
	else:
		print(f"Warning: Unknown scheduler type '{scheduler_type}'. No scheduler will be used.")
		return None


def create_optimizer(model, config):
	"""Create optimizer with parameter groups"""
	# Parameters that don't contain 'confidence' in their name
	pred_params = [p for n, p in model.named_parameters() 
				  if 'confidence' not in n and p.requires_grad]
	
	# Parameters that contain 'confidence' in their name
	conf_params = [p for n, p in model.named_parameters() 
				  if 'confidence' in n and p.requires_grad]
	
	# Create parameter groups with different learning rates
	param_groups = [
		{'params': pred_params},
		{'params': conf_params, 
		 'lr': config['training_params']['learning_rate'] * 
			   config['training_params']['confidence_lr_multiplier']}
	]
	
	return torch.optim.AdamW(
		param_groups,
		lr=config['training_params']['learning_rate'],
		weight_decay=config['training_params']['weight_decay'],
		betas=(config['training_params']['AdamW_beta1'], 
			   config['training_params']['AdamW_beta2'])
	)


def load_data(data_dir='data', batch_size=32, use_sampler=True):
	"""Load datasets and create dataloaders with the single composite target"""
	data_dir = Path(data_dir)
	dataloaders = {}

	# Custom collate function to handle structured data
	def custom_collate(batch):
		features = torch.stack([item[0] for item in batch])
		targets = torch.stack([item[1] for item in batch])
		metadata = pd.concat([item[2] for item in batch], axis=0)
		return features, targets, metadata

	for split in ['train', 'val', 'test']:
		# Load data
		data = torch.load(data_dir / f'{split}.pt', weights_only=True)
		X_unified = data['unified_features']
		
		# Get the composite target (continuous in range [-1, 1])
		y_composite = data['y']
		
		# Load metadata
		metadata_df = pd.read_csv(data_dir / f'{split}_metadata.csv')
		
		# Create sampler for training data - derived from the sign of the composite target
		sampler = None
		if split == 'train' and use_sampler:
			binary_indicator = (y_composite > 0).long()
			class_counts = torch.bincount(binary_indicator)
			pos_count = class_counts[1].item() if len(class_counts) > 1 else 0
			neg_count = class_counts[0].item()
			
			# More aggressive class balancing
			pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0
			
			sampler = WeightedRandomSampler(
				weights=[pos_weight if val > 0 else 1.0 for val in y_composite],
				num_samples=len(y_composite),
				replacement=True
			)
			shuffle = False   
		else:
			shuffle = (split == 'train')
		
		# Create custom dataset with single target
		class TradeDataset(torch.utils.data.Dataset):
			def __init__(self, features, labels, metadata_df):
				self.features = features
				self.labels = labels
				self.metadata = metadata_df

			def __len__(self):
				return len(self.labels)

			def __getitem__(self, idx):
				return (
					self.features[idx], 
					self.labels[idx],
					self.metadata.iloc[idx:idx+1]
				)
		
		dataset = TradeDataset(
			X_unified,
			y_composite,
			metadata_df
		)
		
		dataloaders[split] = DataLoader(
			dataset,
			batch_size=batch_size,
			shuffle=shuffle,
			sampler=sampler,
			collate_fn=custom_collate
		)
	
	return dataloaders


def train_model(model, criterion, optimizer, train_loader, val_loader, config, evaluator, device, scheduler=None):
	"""Main training loop using the centralized TrainingManager"""
	# Create training manager
	training_manager = TrainingManager(
		model=model,
		optimizer=optimizer,
		scheduler=scheduler,
		criterion=criterion,
		config=config,
		device=device,
		evaluator=evaluator
	)
	
	# Training parameters
	num_epochs = config['training_params']['num_epochs']
	
	# Training loop
	for epoch in range(1, num_epochs + 1):
		print(f"\nEpoch {epoch}/{num_epochs}", end='')
		
		# Train for one epoch
		train_metrics = training_manager.train_epoch(train_loader, epoch)
		
		# Evaluate on validation set
		epoch_metrics = training_manager.evaluate(val_loader, train_metrics['total_loss'])
		
		# Step scheduler if applicable
		training_manager.step_scheduler(epoch_metrics['val_loss'])
		
		# Print metrics
		print()
		print(f"Train Loss: {train_metrics['total_loss']:.4f} | "
			f"Val Loss: {epoch_metrics['val_loss']:.4f} | "
			f"ROC AUC: {epoch_metrics['auc']:.4f} | "
			f"Precision @ RG1: {epoch_metrics['precision_gain_at_rg1']:.4f}")

		print(f"Pred mean: {epoch_metrics['network_mean']:.4f} | "
			f"Pred std: {epoch_metrics['network_std']:.4f} | "
			f"Conf mean: {epoch_metrics['confidence_mean']:.4f} | "
			f"Conf std: {epoch_metrics['confidence_std']:.4f}")

		# Add precision @ k metrics to output
		if all(key in epoch_metrics for key in ['precision_at_0.01', 'precision_at_0.05', 'precision_at_0.1']):
			print(f"Precision @ 1%: {epoch_metrics['precision_at_0.01']:.4f} | "
				f"Precision @ 5%: {epoch_metrics['precision_at_0.05']:.4f} | "
				f"Precision @ 10%: {epoch_metrics['precision_at_0.1']:.4f}")

		# Add expected return metrics if available
		if all(key in epoch_metrics for key in ['mean_return_at_0.01', 'mean_return_at_0.05', 'mean_return_at_0.1']):
			print(f"Return @ 1%: {epoch_metrics['mean_return_at_0.01']*100:.2f}% | "
				f"Return @ 5%: {epoch_metrics['mean_return_at_0.05']*100:.2f}% | "
				f"Return @ 10%: {epoch_metrics['mean_return_at_0.1']*100:.2f}%")

		# Add Sharpe ratio metrics if available
		if all(key in epoch_metrics for key in ['sharpe_at_0.01', 'sharpe_at_0.05', 'sharpe_at_0.1']):
			print(f"Sharpe @ 1%: {epoch_metrics['sharpe_at_0.01']:.2f} | "
				f"Sharpe @ 5%: {epoch_metrics['sharpe_at_0.05']:.2f} | "
				f"Sharpe @ 10%: {epoch_metrics['sharpe_at_0.1']:.2f}")
			
		
		# Show current learning rate if available
		if scheduler:
			try:
				if isinstance(scheduler, SWALR) and epoch < training_manager.swa_start:
					print(f"Waiting for SWA to start at epoch {training_manager.swa_start}")
				else:
					print(f"Scheduler LR: {scheduler.get_last_lr()[0]:.6f}")
			except:
				print(f"Scheduler LR: (not available)")
		
		# Save checkpoint for this epoch
		training_manager.save_checkpoint(epoch_metrics, epoch=epoch)
		
		# Check for early stopping
		if training_manager.check_early_stopping(epoch_metrics, epoch):
			print(f"Early stopping triggered after {epoch} epochs")
			break
	
	print("\nTraining completed!")
	
	# If using SWA, evaluate the SWA model
	final_metrics = epoch_metrics
	best_epoch = training_manager.best_epoch  # Use the tracked best epoch
	best_score = training_manager.best_score
	
	# Restore the best model
	training_manager.restore_best_model()
	
	return model, best_epoch, best_score, final_metrics


def parse_args():
	"""Parse command line arguments"""
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str, required=True, help='Path to config file')
	return parser.parse_args()


def main():
	"""Main entry point"""
	# Parse arguments
	args = parse_args()
	
	# Load configuration
	with open(args.config, 'r') as f:
		config = json.load(f)
	
	# Set device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}")
	
	# Load data
	dataset_dir = Path(config['paths']['dataset_dir'])
	dataloaders = load_data(
		data_dir=dataset_dir,
		batch_size=config['training_params']['batch_size'],
		use_sampler=True
	)
	
	# Create model and criterion
	model, criterion = create_model(config, device)
	
	# Create optimizer
	optimizer = create_optimizer(model, config)
	
	# Create scheduler
	scheduler = create_scheduler(optimizer, config, dataloaders['train'])
	
	# Create evaluator
	evaluator = ModelEvaluator(model, config, device)
	
	# Train model
	model, best_epoch, best_score, final_metrics = train_model(
		model, criterion, optimizer, dataloaders['train'], dataloaders['val'],
		config, evaluator, device, scheduler=scheduler
	)
	
	print(f"Best model from epoch {best_epoch} with score {best_score:.4f}")
	
	# Final evaluation on test set
	print("\nPerforming final evaluation on test set...")
	test_metrics = evaluator.evaluate_test(
		dataloaders['test'],
		save_dir=Path(config['paths']['run_dir']) / 'evaluation',
		test_epoch=best_epoch
	)
	
	print("\nTraining and evaluation complete!")


if __name__ == "__main__":
	main()