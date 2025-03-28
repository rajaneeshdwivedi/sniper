import subprocess
import logging
from pathlib import Path
import json
from datetime import datetime
import os

def setup_logging(base_path):
	"""Setup logging configuration"""
	log_file = base_path / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
	
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(levelname)s - %(message)s',
		handlers=[
			logging.FileHandler(log_file),
			logging.StreamHandler()
		]
	)
	
	return logging.getLogger(__name__)

def run_feature_generation(config_path, logger):
	"""Run feature generation script with config"""
	cmd = ['python', 'build_dataset.py', '--config', str(config_path)]
	logger.info(f"Running feature generation: {' '.join(cmd)}")
	
	try:
		subprocess.run(cmd, check=True)
		logger.info("Feature generation completed successfully")
		return True
	except subprocess.CalledProcessError as e:
		logger.error(f"Feature generation failed: {e}")
		return False

def run_model_training(config_path, logger):
	"""Run model training script with config"""
	cmd = ['python', 'trainer.py', '--config', str(config_path)]
	logger.info(f"Running model training: {' '.join(cmd)}")
	
	try:
		# Create environment with MKL settings
		env = os.environ.copy()
		env['MKL_THREADING_LAYER'] = 'GNU'
		
		subprocess.run(cmd, check=True, env=env)
		logger.info("Model training completed successfully")
		return True
	except subprocess.CalledProcessError as e:
		logger.error(f"Model training failed: {e}")
		return False

def parse_metrics_file(metrics_text):
	"""Parse metrics from the standardized metrics file format"""
	# Parse threshold analysis section
	threshold_section = metrics_text.split('PRECISION VS TRADES ANALYSIS')[1]
	lines = threshold_section.split('\n')
	
	# Find the line with 0.5 threshold metrics
	for line in lines:
		if line.strip().startswith('3.0'):
			# Split and clean the line
			parts = [p.strip() for p in line.split() if p.strip()]
			if len(parts) >= 5:
				trades = int(parts[1])
				precision = float(parts[3])
				exp_value = float(parts[4].split('%')[0])
				break
	return precision, trades, exp_value

def collect_results(base_path, logger, override):
	"""Collect and summarize results from all runs"""
	results = []
	# logger.info(f"Searching for results in {base_path}")
	
	run_dirs = list(base_path.glob('run_*'))
	# logger.info(f"Found {len(run_dirs)} run directories")

	for run_dir in run_dirs:
		config_path = run_dir / 'config.json'
		metrics_files = list(run_dir.glob('test_metrics_*.txt'))
		
		# logger.info(f"\nChecking run directory: {run_dir}")
		# logger.info(f"Config exists: {config_path.exists()}")
		# logger.info(f"Number of metrics files found: {len(metrics_files)}")
		
		if config_path.exists() and metrics_files:
			try:
				with open(config_path) as f:
					config = json.load(f)
				
				with open(metrics_files[0]) as f:
					metrics_text = f.read()
				
				result = get_original_values(config, override)
				try:
					# Parse metrics using the new function
					precision, trades, exp_value = parse_metrics_file(metrics_text)
					result.update({
						'run_id': config['run_id'],
						'precision': precision,
						'trades': trades,
						'exp_value': exp_value
					})
					
					results.append(result)
					# logger.info(f"Successfully parsed results for {run_dir}")
					
				except Exception as e:
					logger.error(f"Error parsing metrics file: {e}")
					# logger.error(f"Metrics file content: {metrics_text[:50]}...")
					
			except Exception as e:
				logger.error(f"Error processing run directory {run_dir}: {e}")
		else:
			logger.warning(f"Skipping {run_dir} - Missing config or metrics files")
	
	if not results:
		logger.error("No valid results found!")
		return []
	
	# Save summary
	results_path = base_path / 'results_summary.json'
	with open(results_path, 'w') as f:
		json.dump(results, f, indent=4)
	
	# logger.info(f"Collected {len(results)} valid results")
	return results




def apply_override(config, override=None):
	if override is not None:		
		for key, value in override.items():
			if isinstance(value, dict) and key in config:
				apply_override(config[key], value)
			else:
				config[key] = value

		
def get_leaf_elements(d):
	leaves = {}
	for key, value in d.items():
		if isinstance(value, dict):
			leaves.update(get_leaf_elements(value))
		else:
			leaves[key] = value
	return leaves


def get_original_values(config, override):
	result = {}
	
	def extract_original_values(config_subset, override_subset, path=[]):
		for key, value in override_subset.items():
			current_path = path + [key]
			
			if isinstance(value, dict) and key in config_subset:
				# Continue recursion for nested dictionaries
				extract_original_values(config_subset[key], value, current_path)
			elif key in config_subset:
				# Found a leaf value to override, add the original to result
				result[key] = config_subset[key]
	
	# Start recursion from the root
	extract_original_values(config, override)
	return result