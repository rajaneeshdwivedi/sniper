import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from datetime import datetime
import os
import shutil
import warnings
import logging
warnings.filterwarnings('ignore')

# Import utility functions from pipeline_utils
from pipeline_utils import setup_logging, apply_override, get_leaf_elements


def load_dataset(dataset_dir, split='train'):
    """Load dataset from saved PT files and CSV metadata"""
    dataset_dir = Path(dataset_dir)
    
    # Load tensor data
    data = torch.load(dataset_dir / f'{split}.pt')
    features = data['unified_features'].numpy()
    targets = data['y'].numpy()
    
    # Load metadata
    metadata = pd.read_csv(dataset_dir / f'{split}_metadata.csv')
    
    return features, targets, metadata


def feature_target_correlation(features, targets, feature_names=None):
    """Calculate correlation between each feature and the target variable"""
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(features.shape[2])]
    
    correlations = []
    for i in range(features.shape[2]):
        # Flatten feature across all sequences and timesteps
        flat_feature = features[:, :, i].flatten()
        
        # Repeat target for each timestep
        expanded_targets = np.repeat(targets, features.shape[1])
        
        # Calculate correlation
        corr = np.corrcoef(flat_feature, expanded_targets)[0, 1]
        correlations.append((feature_names[i], corr))
    
    return pd.DataFrame(correlations, columns=['feature', 'correlation']).sort_values('correlation', key=abs, ascending=False)


def train_baseline_models(features, targets, feature_group_indices=None, group_names=None):
    """Train simple baseline models to evaluate predictive power"""
    # Reshape features to 2D by using only the last timestep for each feature
    X = features[:, -1, :]
    y = targets
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, C=0.1),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    }
    
    results = {}
    
    # Evaluate overall performance
    print("\nBaseline model performance on all features:")
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
        results[name] = {
            'all_features': {
                'mean_roc_auc': cv_scores.mean(),
                'std_roc_auc': cv_scores.std()
            }
        }
        print(f"{name}: ROC AUC = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # If feature groups are provided, evaluate each group separately
    if feature_group_indices is not None and group_names is not None:
        print("\nBaseline model performance by feature group:")
        for group_name, (start_idx, end_idx) in zip(group_names, feature_group_indices):
            group_features = X_scaled[:, start_idx:end_idx]
            
            # Skip if no features in this group
            if group_features.shape[1] == 0:
                continue
                
            for name, model in models.items():
                cv_scores = cross_val_score(model, group_features, y, cv=5, scoring='roc_auc')
                results[name][group_name] = {
                    'mean_roc_auc': cv_scores.mean(),
                    'std_roc_auc': cv_scores.std()
                }
                print(f"{name} - {group_name}: ROC AUC = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return results


def calculate_feature_importance(features, targets, feature_names=None):
    """Calculate feature importance using a Random Forest model"""
    # Use only the last timestep for each feature
    X = features[:, -1, :]
    y = targets
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    rf.fit(X, y)
    
    # Get feature importance
    importances = rf.feature_importances_
    
    # Calculate permutation importance
    perm_importance = permutation_importance(rf, X, y, n_repeats=10, random_state=42)
    perm_importances = perm_importance.importances_mean
    
    # Combine results
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'gini_importance': importances,
        'permutation_importance': perm_importances
    })
    
    return importance_df.sort_values('permutation_importance', ascending=False)


def analyze_feature_temporal_patterns(features, targets, n_features=5, feature_names=None):
    """Analyze how feature patterns over time relate to the target"""
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(features.shape[2])]
    
    # Get indices of most important features based on correlation with target
    corr_df = feature_target_correlation(features, targets, feature_names)
    top_feature_indices = [feature_names.index(fname) for fname in corr_df['feature'].head(n_features)]
    
    # Separate positive and negative examples
    positive_mask = targets == 1
    negative_mask = targets == 0
    
    # Create plots
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 4*n_features))
    if n_features == 1:
        axes = [axes]
    
    for i, feature_idx in enumerate(top_feature_indices):
        feature_name = feature_names[feature_idx]
        
        # Calculate mean and std for each timestep
        pos_mean = features[positive_mask, :, feature_idx].mean(axis=0)
        pos_std = features[positive_mask, :, feature_idx].std(axis=0)
        neg_mean = features[negative_mask, :, feature_idx].mean(axis=0)
        neg_std = features[negative_mask, :, feature_idx].std(axis=0)
        
        timesteps = np.arange(features.shape[1])
        
        # Plot with confidence intervals
        axes[i].plot(timesteps, pos_mean, 'g-', label='Positive Outcome')
        axes[i].fill_between(timesteps, pos_mean - pos_std, pos_mean + pos_std, color='g', alpha=0.2)
        axes[i].plot(timesteps, neg_mean, 'r-', label='Negative Outcome')
        axes[i].fill_between(timesteps, neg_mean - neg_std, neg_mean + neg_std, color='r', alpha=0.2)
        
        axes[i].set_title(f'Feature: {feature_name}')
        axes[i].set_xlabel('Timestep')
        axes[i].set_ylabel('Normalized Value')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def analyze_feature_distributions_by_outcome(features, targets, n_features=5, feature_names=None):
    """Analyze how feature distributions differ between positive and negative outcomes"""
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(features.shape[2])]
    
    # Get indices of most important features based on correlation with target
    corr_df = feature_target_correlation(features, targets, feature_names)
    top_feature_indices = [feature_names.index(fname) for fname in corr_df['feature'].head(n_features)]
    
    # Create plots
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 4*n_features))
    if n_features == 1:
        axes = [axes]
    
    for i, feature_idx in enumerate(top_feature_indices):
        feature_name = feature_names[feature_idx]
        
        # Use the last timestep for this feature
        feature_values = features[:, -1, feature_idx]
        
        # Separate by outcome
        pos_values = feature_values[targets == 1]
        neg_values = feature_values[targets == 0]
        
        # Plot distributions
        sns.kdeplot(pos_values, ax=axes[i], label='Positive Outcome', color='g')
        sns.kdeplot(neg_values, ax=axes[i], label='Negative Outcome', color='r')
        
        # Calculate KS statistic to quantify distribution difference
        from scipy.stats import ks_2samp
        ks_stat, p_value = ks_2samp(pos_values, neg_values)
        
        axes[i].set_title(f'Feature: {feature_name} (KS stat: {ks_stat:.4f}, p-value: {p_value:.4f})')
        axes[i].set_xlabel('Normalized Value')
        axes[i].set_ylabel('Density')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def analyze_group_separability(features, targets, feature_group_indices=None, group_names=None):
    """Analyze how well each feature group separates positive and negative examples"""
    
    # Use only the last timestep for each feature
    X = features[:, -1, :]
    
    results = []
    
    if feature_group_indices is None or group_names is None:
        # If no groups provided, treat all features as one group
        feature_group_indices = [(0, X.shape[1])]
        group_names = ['all_features']
    
    for group_name, (start_idx, end_idx) in zip(group_names, feature_group_indices):
        # Extract features for this group
        group_features = X[:, start_idx:end_idx]
        
        # Skip if no features in this group
        if group_features.shape[1] == 0:
            continue
        
        # Train a logistic regression model on this group
        model = LogisticRegression(max_iter=1000, C=0.1)
        model.fit(group_features, targets)
        
        # Get probabilities
        probs = model.predict_proba(group_features)[:, 1]
        
        # Calculate ROC AUC
        roc_auc = roc_auc_score(targets, probs)
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(targets, probs)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # ROC curve
        fpr, tpr, _ = roc_curve(targets, probs)
        ax1.plot(fpr, tpr)
        ax1.plot([0, 1], [0, 1], 'k--')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title(f'ROC Curve for {group_name} (AUC = {roc_auc:.4f})')
        
        # Precision-Recall curve
        ax2.plot(recall, precision)
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title(f'Precision-Recall Curve for {group_name}')
        
        plt.tight_layout()
        
        results.append({
            'group_name': group_name,
            'roc_auc': roc_auc,
            'plot': fig
        })
    
    return results


def analyze_asset_specific_patterns(features, targets, metadata, feature_names=None):
    """Analyze predictive power across different assets"""
    # Extract assets
    assets = metadata['code'].unique()
    
    # Use only the last timestep for each feature
    X = features[:, -1, :]
    
    results = {}
    
    for asset in assets:
        # Extract data for this asset
        asset_mask = metadata['code'] == asset
        asset_X = X[asset_mask]
        asset_y = targets[asset_mask]
        
        # Skip if too few examples
        if len(asset_y) < 100 or np.unique(asset_y).size <= 1:
            results[asset] = {'roc_auc': None, 'n_samples': len(asset_y)}
            continue
        
        # Train a model
        model = LogisticRegression(max_iter=1000, C=0.1)
        
        try:
            # Calculate cross-validation score
            cv_scores = cross_val_score(model, asset_X, asset_y, cv=min(5, len(asset_y) // 20), scoring='roc_auc')
            
            results[asset] = {
                'roc_auc': cv_scores.mean(),
                'std': cv_scores.std(),
                'n_samples': len(asset_y),
                'class_balance': asset_y.mean()
            }
        except Exception as e:
            results[asset] = {
                'roc_auc': None, 
                'error': str(e),
                'n_samples': len(asset_y),
                'class_balance': asset_y.mean() if len(asset_y) > 0 else None
            }
    
    # Convert to DataFrame for easier viewing
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df = results_df.sort_values('roc_auc', ascending=False)
    
    return results_df


def run_feature_analysis(dataset_dir, output_dir=None, config=None, n_features=None, rebuild=False):
    """Run the feature analysis pipeline with the given parameters"""
    # If rebuild flag is set, regenerate the dataset first
    if rebuild:
        from pipeline_utils import run_feature_generation
        logger = logging.getLogger(__name__)
        
        # Find the config path in the output_dir
        if output_dir:
            config_path = Path(output_dir) / 'config.json'
            if config_path.exists():
                logger.info(f"Rebuilding dataset using config at {config_path}")
                if not run_feature_generation(config_path, logger):
                    logger.error("Feature generation failed")
                    return None
    
    # Load datasets
    train_features, train_targets, train_metadata = load_dataset(dataset_dir, 'train')
    val_features, val_targets, val_metadata = load_dataset(dataset_dir, 'val')
    
    print(f"Train features shape: {train_features.shape}")
    print(f"Train targets shape: {train_targets.shape}")
    print(f"Class balance: {train_targets.mean():.4f} positive, {1-train_targets.mean():.4f} negative")
    
    # Extract feature names from config if available
    feature_names = None
    feature_group_indices = None
    group_names = None
    
    if config is not None:
        feature_groups = config.get('feature_generation', {}).get('feature_groups', {})
        feature_names = []
        group_names = []
        feature_group_indices = []
        
        start_idx = 0
        for group_name, features in feature_groups.items():
            group_names.append(group_name)
            feature_names.extend(features)
            
            end_idx = start_idx + len(features)
            feature_group_indices.append((start_idx, end_idx))
            start_idx = end_idx
    
    # If n_features is not specified, use the number of feature names or default to 5
    if n_features is None:
        if feature_names:
            n_features = min(5, len(feature_names))  # Default to 5 or fewer if there are fewer features
        else:
            n_features = 5
    
    # 1. Calculate feature-target correlations
    print("\nFeature-Target Correlations:")
    correlations = feature_target_correlation(train_features, train_targets, feature_names)
    print(correlations.head(10))
    
    # 2. Train and evaluate baseline models
    baseline_results = train_baseline_models(train_features, train_targets, 
                                           feature_group_indices, group_names)
    
    # 3. Calculate feature importance
    print("\nFeature Importance Analysis:")
    importance_df = calculate_feature_importance(train_features, train_targets, feature_names)
    print(importance_df.head(10))
    
    # 4. Analyze temporal patterns
    temporal_fig = analyze_feature_temporal_patterns(train_features, train_targets, 
                                                   n_features=n_features, feature_names=feature_names)
    
    # 5. Analyze feature distributions
    dist_fig = analyze_feature_distributions_by_outcome(train_features, train_targets, 
                                                     n_features=n_features, feature_names=feature_names)
    
    # 6. Analyze group separability
    separability_results = analyze_group_separability(train_features, train_targets, 
                                                   feature_group_indices, group_names)
    
    # 7. Analyze asset-specific patterns
    asset_results = analyze_asset_specific_patterns(train_features, train_targets, 
                                                 train_metadata, feature_names)
    print("\nAsset-Specific Performance:")
    print(asset_results)
    
    # Save figures and results if output_dir is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save figures
        temporal_fig.savefig(output_dir / 'temporal_patterns.png')
        dist_fig.savefig(output_dir / 'feature_distributions.png')
        
        for i, result in enumerate(separability_results):
            result['plot'].savefig(output_dir / f"group_separability_{result['group_name']}.png")
            # Remove plot object before serializing to JSON
            result.pop('plot')
        
        # Save summary results
        summary_results = {
            'feature_correlations': correlations.head(20).to_dict(),
            'baseline_results': baseline_results,
            'asset_results': asset_results.to_dict(),
            'separability_results': separability_results,
            'best_features': importance_df.head(10).to_dict(),
            'config': config,
            'n_features': n_features
        }
        
        with open(output_dir / 'feature_analysis_summary.json', 'w') as f:
            json.dump(summary_results, f, indent=2)
    
    # Return summary results
    return {
        'feature_correlations': correlations,
        'baseline_results': baseline_results,
        'feature_importance': importance_df,
        'asset_results': asset_results,
        'separability_results': separability_results
    }


# Note: This function is no longer used as we're using config_generator directly in main()
# Keeping it here for reference but can be removed
def run_parameter_search(dataset_dir, base_path, parameters_to_search=None, config=None):
    """Run feature analysis with parameter search - DEPRECATED"""
    # This functionality is now handled directly in main()
    pass


def main():
    parser = argparse.ArgumentParser(description='Feature Signal Analysis with Parameter Search')
    parser.add_argument('--multimode', action='store_true', help='Run in parameter search mode')
    parser.add_argument('--output', type=str, help='Custom output directory for results')
    parser.add_argument('--rebuild', action='store_true', help='Force rebuild of feature datasets')
    
    args = parser.parse_args()
    
    # Setup base directory for all runs
    base_path = Path(args.output) if args.output else Path('../feature_analysis') / f"{datetime.now():%Y%m%d_%H%M%S}"
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Import config_generator here to avoid circular imports
    from config_generator import generate_config
    
    if args.multimode:
        # Parameter search mode
        logger = setup_logging(base_path)
        logger.info(f"Starting feature analysis parameter search in {base_path}")
        
        # Create source subfolder and copy all Python files
        source_dir = base_path / 'source'
        source_dir.mkdir(exist_ok=True)
        
        # Copy all Python files from current directory to source folder
        for py_file in Path('.').glob('*.py'):
            shutil.copy2(py_file, source_dir / py_file.name)
        
        logger.info(f"Copied source files to {source_dir}")
        
        # Define parameters to search - using a nested structure matching the config
        parameter_search = {
			"feature_generation": {
				"label_params": {
					"atr_target_mult": [2.0, 3.0],
					"atr_stop_mult": [2.0, 3.0],
				},
				"source_width": [100, 300],
				"lookback": [10, 20],
			},
			"vwap_params": {
				"left_pad": [8, 16],
				"right_pad": [8, 16]
			},
        }
        
        # Generate combinations of parameters
        def generate_parameter_combinations(param_dict):
            """Generate all combinations of parameters from a nested dictionary."""
            # First, extract all leaf parameters and their values
            leaf_params = []
            
            def extract_leaf_params(d, prefix=''):
                for key, value in d.items():
                    path = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, dict):
                        extract_leaf_params(value, path)
                    elif isinstance(value, list):
                        leaf_params.append((path, value))
            
            extract_leaf_params(param_dict)
            
            # Generate all combinations
            param_names = [param[0] for param in leaf_params]
            param_values = [param[1] for param in leaf_params]
            
            from itertools import product
            combinations = []
            
            for value_combination in product(*param_values):
                combination = {}
                for i, name in enumerate(param_names):
                    combination[name] = value_combination[i]
                combinations.append(combination)
            
            return combinations
        
        # Get all parameter combinations
        param_combinations = generate_parameter_combinations(parameter_search)
        logger.info(f"Generated {len(param_combinations)} parameter combinations")
        
        results = []
        
        # Process each parameter combination
        for combination_idx, param_combination in enumerate(param_combinations):
            logger.info(f"\nProcessing combination {combination_idx+1}/{len(param_combinations)}")
            
            # Build override dictionary for this combination
            override = {}
            for param_path, param_value in param_combination.items():
                path_parts = param_path.split('.')
                
                # Build nested dictionary structure
                current = override
                for i, part in enumerate(path_parts[:-1]):
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                current[path_parts[-1]] = param_value
            
            # Log the override values in a readable format
            logger.info("Using parameters:")
            for param_path, param_value in param_combination.items():
                logger.info(f"  {param_path} = {param_value}")
            
            # Generate a config for this run
            run_dir = generate_config(base_path, override)
            config_path = run_dir / 'config.json'
            
            # Load the generated config
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            try:
                # Run analysis with these parameters
                analysis_results = run_feature_analysis(
                    dataset_dir=config['paths']['dataset_dir'],
                    output_dir=run_dir,
                    config=config,
                    rebuild=args.rebuild
                )
                
                if analysis_results is None:
                    logger.error("Analysis failed for this parameter combination")
                    continue
                
                # Extract key metrics for comparison
                if 'Random Forest' in analysis_results['baseline_results']:
                    model_performance = analysis_results['baseline_results']['Random Forest']['all_features']['mean_roc_auc']
                else:
                    model_performance = analysis_results['baseline_results']['Logistic Regression']['all_features']['mean_roc_auc']
                
                # Get top feature correlation
                top_feature_corr = analysis_results['feature_correlations']['correlation'].iloc[0]
                
                # Store the leaf parameters for easier comparison
                run_summary = {
                    'run_id': str(run_dir.name),
                    'params': param_combination,
                    'model_performance': model_performance,
                    'top_feature_correlation': top_feature_corr,
                }
                
                results.append(run_summary)
                logger.info("Run completed successfully")
                
            except Exception as e:
                logger.error(f"Error in run: {str(e)}")
        
        # Find best run based on model performance
        if results:
            best_run = max(results, key=lambda x: x['model_performance'])
            
            logger.info("\nParameter search complete.")
            logger.info("\nBest configuration:")
            for param_path, param_value in best_run['params'].items():
                logger.info(f"{param_path} = {param_value}")
            
            logger.info(f"Model Performance: {best_run['model_performance']:.4f}")
            logger.info(f"Top Feature Correlation: {best_run['top_feature_correlation']:.4f}")
            
            # Save overall results
            with open(base_path / 'parameter_search_results.json', 'w') as f:
                json.dump({
                    'all_runs': results,
                    'best_run': best_run
                }, f, indent=2)
            
            print("\nParameter search complete. Results saved to:", base_path)
        else:
            logger.error("No valid results to analyze. Check the logs for errors.")
    
    else:
        # Single run mode - generate a default config
        run_dir = generate_config(base_path)
        config_path = run_dir / 'config.json'
        
        # Load the generated config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Run single analysis
        results = run_feature_analysis(
            dataset_dir=config['paths']['dataset_dir'],
            output_dir=run_dir,
            config=config,
            rebuild=args.rebuild
        )
        
        print(f"\nAnalysis complete. Results saved to {run_dir}")


if __name__ == "__main__":
    main()