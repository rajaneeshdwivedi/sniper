#!/usr/bin/env python3
import os
import sys
import json
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from tqdm import tqdm
from sklearn.inspection import permutation_importance

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Calculate and visualize feature importance for a trained model')
    parser.add_argument('--run_dir', type=str, required=True, 
                       help='Path to the training run directory containing config.json and models folder')
    parser.add_argument('--test_split', type=str, default='test',
                       help='Data split to use for importance calculation (default: test)')
    parser.add_argument('--n_repeats', type=int, default=10,
                       help='Number of repeats for permutation importance (default: 10)')
    parser.add_argument('--min_importance', type=float, default=0.0,
                       help='Minimum absolute importance threshold to display (default: 0.0)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use for computation (default: auto-detect)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for processing (default: 64)')
    return parser.parse_args()

def load_config(run_dir):
    """Load configuration file"""
    config_path = Path(run_dir) / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)

def load_model(run_dir, device):
    """Load the trained model from checkpoint"""
    checkpoint_path = Path(run_dir) / 'models' / 'best_checkpoint.pt'
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Import model architecture
    sys.path.append(str(Path.cwd()))  # Add current directory to path
    try:
        from model import create_model, HierarchicalGRUModel
    except ImportError:
        raise ImportError("Cannot import model module. Make sure model.py is in the current directory.")
    
    # Create model and load weights
    model, _ = create_model(checkpoint['config'], device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['config']

def load_data(config, split, device):
    """Load dataset"""
    dataset_dir = Path(config['paths']['dataset_dir'])
    data_path = dataset_dir / f'{split}.pt'
    metadata_path = dataset_dir / f'{split}_metadata.csv'
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {data_path}")
    
    data = torch.load(data_path, map_location=device, weights_only=True)
    
    # Load metadata if available
    metadata = None
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
    
    return data, metadata

def calculate_feature_importance(model, dataset, config, device, n_repeats=10, batch_size=64):
    """
    Calculate feature importance using permutation method
    
    Args:
        model: Trained model
        dataset: Dictionary containing features and targets
        config: Configuration dictionary
        device: Computation device
        n_repeats: Number of permutation repeats
        batch_size: Batch size for processing
        
    Returns:
        Dictionary with importance scores and feature details
    """
    print("Calculating feature importance using permutation method...")
    
    # Extract features and labels
    features = dataset['unified_features']  # Shape: [n_samples, seq_len, n_features]
    targets = dataset['y']  # Binary targets
    
    # Move data to device
    features = features.to(device)
    targets = targets.to(device)
    
    # Extract feature names
    feature_groups = config['feature_generation']['feature_groups']
    feature_names = []
    group_names = []
    
    for group_name, group_features in feature_groups.items():
        for feature in group_features:
            feature_names.append(feature)
            group_names.append(group_name)
    
    # Calculate baseline loss
    n_samples = features.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    all_predictions = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            batch_features = features[start_idx:end_idx]
            batch_targets = targets[start_idx:end_idx]
            
            # Forward pass (ignoring confidence and predictions)
            probs, _, _ = model(batch_features)
            
            all_predictions.append(probs.cpu())
            all_targets.append(batch_targets.cpu())
    
    all_predictions = torch.cat(all_predictions).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    # Calculate baseline loss using binary cross-entropy
    eps = 1e-7
    predictions_clipped = np.clip(all_predictions, eps, 1 - eps)
    baseline_loss = -np.mean(
        all_targets * np.log(predictions_clipped) + 
        (1 - all_targets) * np.log(1 - predictions_clipped)
    )
    
    # Calculate importance for each feature
    feature_importance = []
    
    progress_bar = tqdm(total=len(feature_names), desc="Feature Importance")
    
    for feature_idx in range(len(feature_names)):
        feature_losses = []
        
        for _ in range(n_repeats):
            # Create a permuted version of the features
            permuted_features = features.clone()
            
            # Permute the selected feature
            permuted_indices = torch.randperm(n_samples)
            permuted_features[:, :, feature_idx] = permuted_features[permuted_indices, :, feature_idx]
            
            # Calculate loss with permuted feature
            all_permuted_predictions = []
            
            with torch.no_grad():
                for i in range(n_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, n_samples)
                    
                    batch_features = permuted_features[start_idx:end_idx]
                    
                    # Forward pass
                    probs, _, _ = model(batch_features)
                    all_permuted_predictions.append(probs.cpu())
            
            all_permuted_predictions = torch.cat(all_permuted_predictions).numpy()
            permuted_predictions_clipped = np.clip(all_permuted_predictions, eps, 1 - eps)
            
            permuted_loss = -np.mean(
                all_targets * np.log(permuted_predictions_clipped) + 
                (1 - all_targets) * np.log(1 - permuted_predictions_clipped)
            )
            
            # Calculate importance as the difference in loss
            feature_losses.append(permuted_loss - baseline_loss)
            
        # Calculate mean and std of feature importance
        mean_importance = np.mean(feature_losses)
        std_importance = np.std(feature_losses)
        
        feature_importance.append({
            'name': feature_names[feature_idx],
            'group': group_names[feature_idx],
            'importance': mean_importance,
            'std': std_importance
        })
        
        progress_bar.update(1)
    
    progress_bar.close()
    
    return {
        'importance_scores': [item['importance'] for item in feature_importance],
        'feature_details': feature_importance,
        'baseline_loss': baseline_loss
    }

def plot_feature_importance(importance_results, config, output_dir, min_importance=0.0):
    """
    Plot feature importance visualizations
    
    Args:
        importance_results: Results from feature importance calculation
        config: Configuration dictionary
        output_dir: Directory to save plots
        min_importance: Minimum absolute importance threshold to display
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get feature details
    feature_details = importance_results['feature_details']
    
    # Normalize importance scores to make them more comparable
    max_importance = max(abs(detail['importance']) for detail in feature_details)
    for detail in feature_details:
        detail['normalized_importance'] = detail['importance'] / max_importance if max_importance > 0 else 0
    
    # Filter out features with importance below threshold
    if min_importance > 0:
        feature_details = [d for d in feature_details if abs(d['normalized_importance']) >= min_importance]
    
    # Sort features by absolute importance
    sorted_details = sorted(feature_details, key=lambda x: abs(x['normalized_importance']), reverse=True)
    
    # Create feature importance bar plot
    plt.figure(figsize=(15, max(8, len(sorted_details) * 0.25)))
    
    names = [detail['name'] for detail in sorted_details]
    groups = [detail['group'] for detail in sorted_details]
    scores = [detail['normalized_importance'] for detail in sorted_details]
    
    # Create colormap for different feature groups
    unique_groups = list(set(groups))
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(unique_groups)))
    group_colors = dict(zip(unique_groups, colors))
    bar_colors = [group_colors[group] for group in groups]
    
    # Create horizontal bar plot
    y_pos = np.arange(len(names))
    bars = plt.barh(y_pos, scores, color=bar_colors)
    
    # Add zero line for reference
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    
    # Customize plot
    plt.yticks(y_pos, names)
    plt.xlabel("Normalized Importance Score")
    plt.title("Feature Importance by Group (Sorted by Absolute Importance)")
    
    # Add legend for feature groups
    legend_elements = [plt.Rectangle((0,0), 1, 1, facecolor=group_colors[group]) 
                      for group in unique_groups]
    plt.legend(legend_elements, unique_groups, loc='best', title='Feature Groups')
    
    # Add gridlines
    plt.grid(True, axis='x', alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # Create a group-level plot
    # Calculate aggregate importance by group
    group_importance = {}
    for detail in feature_details:
        group = detail['group']
        if group not in group_importance:
            group_importance[group] = []
        group_importance[group].append(detail['normalized_importance'])
    
    # Calculate average importance for each group
    group_avg_importance = {group: np.mean(scores) for group, scores in group_importance.items()}
    group_std_importance = {group: np.std(scores) for group, scores in group_importance.items()}
    
    # Sort groups by absolute importance
    sorted_groups = sorted(group_avg_importance.keys(), 
                        key=lambda g: abs(group_avg_importance[g]), 
                        reverse=True)
    
    # Create group importance plot
    plt.figure(figsize=(12, 8))
    
    # Plot means with error bars
    y_pos = np.arange(len(sorted_groups))
    means = [group_avg_importance[g] for g in sorted_groups]
    stds = [group_std_importance[g] for g in sorted_groups]
    colors = [group_colors[g] for g in sorted_groups]
    
    plt.barh(y_pos, means, xerr=stds, color=colors, alpha=0.7)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    
    # Add feature count for each group
    for i, group in enumerate(sorted_groups):
        feature_count = len(group_importance[group])
        plt.text(
            0.01, 
            y_pos[i] + 0.1, 
            f"({feature_count} features)",
            va='center',
            size=9,
            alpha=0.7
        )
    
    plt.yticks(y_pos, sorted_groups)
    plt.xlabel("Average Normalized Importance")
    plt.title("Feature Group Importance (Aggregated)")
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Save group importance plot
    plt.savefig(output_dir / 'feature_group_importance.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # Create a detailed report of top features in each group
    report_path = output_dir / 'feature_importance_report.txt'
    with open(report_path, 'w') as f:
        f.write("FEATURE IMPORTANCE REPORT\n")
        f.write("=======================\n\n")
        
        # Overall statistics
        f.write(f"Total features analyzed: {len(feature_details)}\n")
        f.write(f"Features with positive importance: {sum(1 for d in feature_details if d['normalized_importance'] > 0)}\n")
        f.write(f"Features with negative importance: {sum(1 for d in feature_details if d['normalized_importance'] < 0)}\n\n")
        
        # Top 10 features overall
        f.write("TOP 10 FEATURES OVERALL\n")
        f.write("-" * 22 + "\n")
        for i, detail in enumerate(sorted_details[:10]):
            f.write(f"{i+1}. {detail['name']} ({detail['group']}): {detail['normalized_importance']:.4f}\n")
        f.write("\n")
        
        # Group by group analysis
        for group in sorted_groups:
            f.write(f"\n{group.upper()} GROUP\n")
            f.write("-" * (len(group) + 7) + "\n")
            f.write(f"Average importance: {group_avg_importance[group]:.4f} ± {group_std_importance[group]:.4f}\n")
            
            # Sort features within group by absolute importance
            group_features = [d for d in feature_details if d['group'] == group]
            sorted_group_features = sorted(group_features, key=lambda x: abs(x['normalized_importance']), reverse=True)
            
            # Print top features (max 5)
            f.write("Top features:\n")
            for i, feature in enumerate(sorted_group_features[:5]):
                f.write(f"  {i+1}. {feature['name']}: {feature['normalized_importance']:.4f}\n")
            
            if len(sorted_group_features) > 5:
                f.write(f"  ... and {len(sorted_group_features)-5} more features\n")
            
            f.write("\n")
    
    print(f"Feature importance visualizations saved to {output_dir}")
    
    # Also save raw importance data as CSV
    csv_path = output_dir / 'feature_importance.csv'
    importance_df = pd.DataFrame([{
        'feature': detail['name'],
        'group': detail['group'],
        'importance': detail['importance'],
        'normalized_importance': detail['normalized_importance'],
        'std': detail['std']
    } for detail in feature_details])
    
    importance_df.sort_values('importance', key=abs, ascending=False, inplace=True)
    importance_df.to_csv(csv_path, index=False)
    
    return importance_df

def main():
    args = parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    try:
        # Load configuration
        run_dir = Path(args.run_dir)
        config = load_config(run_dir)
        
        # Setup output directory
        eval_dir = run_dir / 'evaluation'
        eval_dir.mkdir(exist_ok=True)
        feature_importance_dir = eval_dir / 'feature_importance'
        
        # Load model
        model, updated_config = load_model(run_dir, device)
        
        # Merge configs (checkpoint config might be more up-to-date)
        for key in updated_config:
            config[key] = updated_config[key]
        
        # Load data
        data, metadata = load_data(config, args.test_split, device)
        
        # Calculate feature importance
        importance_results = calculate_feature_importance(
            model, data, config, device, 
            n_repeats=args.n_repeats,
            batch_size=args.batch_size
        )
        
        # Plot and save visualizations
        importance_df = plot_feature_importance(
            importance_results, 
            config, 
            feature_importance_dir,
            min_importance=args.min_importance
        )
        
        print("Feature importance analysis completed successfully.")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())