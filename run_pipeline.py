from pathlib import Path
from datetime import datetime
import numpy as np
import shutil
import os
import json
from config_generator import generate_config
from pipeline_utils import setup_logging, run_feature_generation, run_model_training, collect_results, get_leaf_elements
import argparse
import feature_signal_analysis as fsa


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
        feature_widths = [12, 15, 18, 21, 24, 27, 30]
        results = []
        
        for feature_width in feature_widths:
            override = {
                'feature_generation': {
                    'feature_width': feature_width
                },
            }
            
            # Generate config for this run
            run_dir = generate_config(base_path, override)
            config_path = run_dir / 'config.json'
            
            if not process_single_run(config_path, logger, rebuild):
                continue
                
            results.append(collect_results(base_path, logger, override))
        
        # Analyze results from parameter search
        if results:
            analyze_search_results(results, override, logger)
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
    """
    Run comprehensive feature signal analysis on the dataset.
    
    Args:
        config_path (Path): Path to the configuration file
        dataset_dir (Path): Path to the dataset directory
        output_dir (Path): Directory to save analysis outputs
        logger: Logger instance
    """
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
        results = fsa.main(dataset_dir, config)
        
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


def create_feature_analysis_script(target_dir):
    """
    Create the feature_signal_analysis.py script if it doesn't exist
    """
    script_path = target_dir / 'feature_signal_analysis.py'
    
    # If the script exists, don't overwrite it
    if script_path.exists():
        return
        

    
    with open(script_path, 'w') as f:
        f.write(script_content)


def analyze_search_results(results, override, logger):
    """
    Analyze and log the results from parameter search runs.
    """
    leaves = get_leaf_elements(override)
    best_run = max(results, key=lambda x: x['exp_value'])
    
    logger.info("\nBest configuration:")
    for key in leaves.keys():
        logger.info(f"{key}: {best_run[key]}")
    
    logger.info(f"Precision: {best_run['precision']:.4f}")
    logger.info(f"Trades: {best_run['trades']}")
    logger.info(f"Expected Value: {best_run['exp_value']}%")


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