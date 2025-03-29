import subprocess
import logging
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import shutil
import os
import re
from config_generator import generate_config
from pipeline_utils import setup_logging, run_feature_generation, run_model_training
import argparse
import feature_signal_analysis as fsa


def parse_metrics_file(metrics_file_path):
	"""Parse metrics from the precision metrics file"""
	try:
		with open(metrics_file_path, 'r') as f:
			content = f.read()
			
		# Extract AUC (primary metric)
		auc_match = re.search(r'AUC: ([0-9.]+)', content)
		auc = float(auc_match.group(1)) if auc_match else 0.5
		
		# Extract low volume metrics (for trades and expected gain)
		metrics_section = content.split('LOW VOLUME PRECISION METRICS')[1].split('LOW VOLUME SCORE')[0]
		lines = metrics_section.strip().split('\n')
		
		# Extract the 5% volume line (index 3 after the header lines)
		if len(lines) >= 5:  # Header + 3 data lines
			five_pct_line = lines[4].strip()  # 1% is line 3, 5% is line 4
			parts = re.split(r'\s+', five_pct_line)
			
			if len(parts) >= 5:
				precision = float(parts[1])
				gain = float(parts[2])
				trades = int(parts[3])
				
				return {
					'auc': auc, 
					'precision': precision,
					'gain': gain,
					'trades': trades
				}
		
		# Default values if parsing fails
		return {'auc': auc, 'precision': 0.5, 'gain': 0.0, 'trades': 0}
		
	except Exception as e:
		logging.error(f"Error parsing metrics file: {e}")
		return {'auc': 0.5, 'precision': 0.5, 'gain': 0.0, 'trades': 0}


def run_pipeline(dataset_basename, mode='single', rebuild=False):
	"""
	Unified pipeline runner that handles single, parameter search, and analysis modes.
	
	Args:
		dataset_basename (str): Base name for the dataset directory
		mode (str): 'single', 'search', or 'analyse' mode
		rebuild (bool): If True, forces feature regeneration
	"""
	# Setup base directory for all runs
	base_path = Path('../pipeline_runs') / f"{dataset_basename}_{datetime.now():%Y%m%d_%H%M%S}"
	base_path.mkdir(parents=True)
	
	# Create source subfolder and copy all Python files
	source_dir = base_path / 'source'
	source_dir.mkdir(exist_ok=True)
	
	# Copy all Python files from current directory to source folder
	for py_file in Path('.').glob('*.py'):
		shutil.copy2(py_file, source_dir / py_file.name)
	
	# Setup logging
	logger = setup_logging(base_path)
	logger.info(f"Starting pipeline in {mode} mode at {base_path}")
	logger.info(f"Copied source files to {source_dir}")
	
	if mode == 'search':
		# Parameter search mode
		results = []
		
		for left_pad in [2, 9, 15, 21, 30]:		  
			for right_pad in [2, 9, 15, 21, 30]:		  
				override = {
					'vwap_params': {
						'left_pad': left_pad,
						'right_pad': right_pad
					},
				}
				
				# Generate config for this run
				run_dir = generate_config(base_path, override)
				config_path = run_dir / 'config.json'
				
				if not process_single_run(config_path, logger, rebuild):
					logger.error(f"Run failed")
					continue
				
				# Find the precision metrics file
				metrics_file = list(run_dir.glob('**/precision_metrics.txt'))
				if not metrics_file:
					logger.error(f"No precision metrics file found")
					continue
					
				metrics_data = parse_metrics_file(metrics_file[0])
				
				# Store the leaf parameters for easier comparison
				run_summary = {
					'run_id': str(run_dir.name),
					'auc': metrics_data['auc'],
					'precision': metrics_data['precision'],
					'gain': metrics_data['gain'],
					'trades': metrics_data['trades']
				}
				
				results.append(run_summary)
				logger.info(f"Run completed successfully with metrics: AUC={metrics_data['auc']:.4f}, "
							f"Precision={metrics_data['precision']:.4f}, Gain={metrics_data['gain']:.4f}")
		
		# Analyze results from parameter search
		if results:
			analyze_search_results(base_path, results, logger)
		else:
			logger.error("No valid results to analyze. Check the logs for errors in the pipeline execution.")
	
	elif mode == 'analyse':
		# Analysis mode - generate config but only run feature analysis
		run_dir = generate_config(base_path)
		config_path = run_dir / 'config.json'
		
		# Create analysis directory
		analysis_dir = run_dir / 'analysis'
		analysis_dir.mkdir(exist_ok=True)
		
		# Run feature generation if rebuild flag is set or features don't exist
		if rebuild:
			if not run_feature_generation(config_path, logger):
				logger.error("Feature generation failed, cannot run analysis")
				return
		
		# Read the config for dataset path
		with open(config_path, 'r') as f:
			config = json.load(f)
		
		dataset_dir = Path(config['paths']['dataset_dir'])
		logger.info(f"Running feature signal analysis on dataset at {dataset_dir}")
		
		# Run the feature signal analysis
		run_feature_analysis(config_path, dataset_dir, analysis_dir, logger)
	
	else:
		# Single run mode
		run_dir = generate_config(base_path)
		config_path = run_dir / 'config.json'
		process_single_run(config_path, logger, rebuild)


def process_single_run(config_path, logger, rebuild):
	"""
	Process a single pipeline run with the given configuration.
	
	Returns:
		bool: True if the run completed successfully, False otherwise
	"""
	# Run feature generation if rebuild flag is set or features don't exist
	if rebuild:
		if not run_feature_generation(config_path, logger):
			logger.error("Feature generation failed")
			return False

	# Run model training
	if not run_model_training(config_path, logger):
		logger.error("Model training failed")
		return False
		
	return True


def run_feature_analysis(config_path, dataset_dir, output_dir, logger):
	"""Feature analysis function implementation - kept as is"""
	try:
		logger.info("Importing feature_signal_analysis module")
		# Import the feature analysis script
		import sys
		sys.path.append(str(Path(config_path).parent / 'source'))
				
		# Load config
		with open(config_path, 'r') as f:
			config = json.load(f)
			
		# Run the analysis
		logger.info(f"Running feature signal analysis on dataset at {dataset_dir}")
		results = fsa.run_feature_analysis(dataset_dir, output_dir, config)
		
		# Save summary results
		summary_path = output_dir / 'feature_analysis_summary.json'
		with open(summary_path, 'w') as f:
			json.dump({
				'feature_correlations': results['feature_correlations'].head(20).to_dict(),
				'baseline_results': results['baseline_results'],
				'asset_results': results['asset_results'].to_dict(),
			}, f, indent=2)
			
		logger.info(f"Analysis complete. Results saved to {summary_path} and PNG files in {output_dir}")
		
		# Log key findings
		logger.info("\nKey findings from feature analysis:")
		logger.info(f"- Best baseline model: Random Forest with ROC AUC of {results['baseline_results']['Random Forest']['all_features']['mean_roc_auc']:.4f}")
		
		# Log top features by correlation
		logger.info("\nTop features by correlation with target:")
		for _, row in results['feature_correlations'].head(5).iterrows():
			logger.info(f"- {row['feature']}: {row['correlation']:.4f}")
			
		# Log asset-specific performance
		logger.info("\nAsset-specific performance:")
		for asset, data in results['asset_results'].iterrows():
			if data['roc_auc'] is not None:
				logger.info(f"- {asset}: ROC AUC = {data['roc_auc']:.4f}")
			
	except Exception as e:
		logger.error(f"Error running feature analysis: {str(e)}")
		import traceback
		logger.error(traceback.format_exc())


def analyze_search_results(run_dir, results, logger):
	"""
	Analyze and log the results from parameter search runs.
	
	Now using AUC as the primary metric for determining the best run.
	"""
	# Find best run based on AUC (primary metric)
	best_run = max(results, key=lambda x: x['auc'])
	
	logger.info("\nParameter search complete.")
	logger.info("\nBest configuration:")
	
	logger.info(f"AUC: {best_run['auc']:.4f}")
	logger.info(f"Precision: {best_run['precision']:.4f}")
	logger.info(f"Gain: {best_run['gain']:.4f}")
	logger.info(f"Trades: {best_run['trades']}")
	
	# Save overall results
	with open(run_dir / 'parameter_search_results.json', 'w') as f:
		json.dump({
			'all_runs': results,
			'best_run': best_run
		}, f, indent=2)
	
	# Create a comparison table for all runs
	sorted_results = sorted(results, key=lambda x: x['auc'], reverse=True)
	
	logger.info("\nAll runs sorted by AUC:")
	logger.info(f"{'AUC':<10} {'Precision':<10} {'Gain':<10} {'Trades':<10}")
	logger.info("-" * 55)
	
	for run in sorted_results:
		logger.info(f"{run['auc']:<10.4f} {run['precision']:<10.4f} {run['gain']:<10.4f} {run['trades']:<10}")
	
	# Print summary
	logger.info("\nAUC Statistics:")
	auc_values = [run['auc'] for run in results]
	logger.info(f"Min: {min(auc_values):.4f}")
	logger.info(f"Max: {max(auc_values):.4f}")
	logger.info(f"Mean: {np.mean(auc_values):.4f}")
	logger.info(f"Std: {np.std(auc_values):.4f}")


def parse_args():
	parser = argparse.ArgumentParser(description='Run the ML pipeline in various modes')
	parser.add_argument('dataset_basename', type=str, help='Base name for the dataset directory')
	
	# Mode options - mutually exclusive
	mode_group = parser.add_mutually_exclusive_group()
	mode_group.add_argument('--mode', type=str, choices=['single', 'search', 'analyse'], 
						   default='single', help='Pipeline execution mode')
	
	# Backwards compatibility
	mode_group.add_argument('--multimode', action='store_true', help='Run in parameter search mode (deprecated)')
	mode_group.add_argument('--analyse', action='store_true', help='Run feature signal analysis (deprecated)')
	
	parser.add_argument('--rebuild', action='store_true', help='Force rebuild of feature datasets')
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()
	
	# Handle deprecated flags for backwards compatibility
	if args.multimode:
		mode = 'search'
	elif args.analyse:
		mode = 'analyse'
	else:
		mode = args.mode
		
	run_pipeline(dataset_basename=args.dataset_basename, mode=mode, rebuild=args.rebuild)