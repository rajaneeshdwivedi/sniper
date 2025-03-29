#!/usr/bin/env python3
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
import json
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate combinations of 1d and 4h models')
    parser.add_argument('--daily_model', type=str, required=True, 
                        help='Path to daily model checkpoint (best_checkpoint.pt)')
    parser.add_argument('--hourly_model', type=str, required=True, 
                        help='Path to 4h model checkpoint (best_checkpoint.pt)')
    parser.add_argument('--daily_data', type=str, required=True, 
                        help='Path to daily test data (.pt file)')
    parser.add_argument('--hourly_data', type=str, required=True, 
                        help='Path to 4h test data (.pt file)')
    parser.add_argument('--output_dir', type=str, default='model_combination_results',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cpu/cuda, default: auto-detect)')
    return parser.parse_args()

def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    # Try to add the parent directory to the path to find model.py
    parent_dir = str(Path(checkpoint_path).parent.parent)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    
    try:
        from model import create_model
    except ImportError:
        raise ImportError("Cannot import model module. Make sure model.py is accessible.")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model, _ = create_model(checkpoint['config'], device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['config']

def load_data(data_path):
    """Load test data"""
    data = torch.load(data_path)
    
    # Check if metadata file exists
    metadata_path = Path(data_path).parent / f"{Path(data_path).stem}_metadata.csv"
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
    else:
        # Create empty metadata with just indices
        metadata = pd.DataFrame({'index': range(len(data['y']))})
    
    return data, metadata

def get_model_predictions(model, data, device, batch_size=64):
    """Run model inference on data"""
    features = data['unified_features']
    targets = data['y']
    
    n_samples = features.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    all_probs = []
    all_confidence = []
    
    model.eval()
    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            batch_features = features[start_idx:end_idx].to(device)
            
            # Forward pass
            probs, confidence, _ = model(batch_features)
            
            all_probs.append(probs.cpu().numpy())
            all_confidence.append(confidence.cpu().numpy())
    
    return {
        'prediction_scores': np.concatenate(all_probs).flatten(),
        'confidence_scores': np.concatenate(all_confidence).flatten(),
        'targets': targets.numpy().flatten()
    }

def align_timeframes(daily_metadata, hourly_metadata):
    """
    Align 4h and daily predictions by mapping each 4h sample to its daily bar
    
    Returns:
        daily_to_hourly_map: Dict mapping daily indices to lists of hourly indices
        hourly_to_daily_map: Dict mapping hourly indices to daily indices
    """
    daily_to_hourly_map = {}
    hourly_to_daily_map = {}
    
    # Check if we have timestamps in both metadata
    if 'closeTime' not in daily_metadata.columns or 'closeTime' not in hourly_metadata.columns:
        print("Warning: closeTime not found in metadata, using sequential mapping")
        # Without timestamps, map 6 hourly bars to 1 daily bar (rough approximation)
        for i, daily_idx in enumerate(daily_metadata.index):
            start_idx = i * 6
            end_idx = (i + 1) * 6
            hourly_indices = list(range(start_idx, end_idx))
            hourly_indices = [idx for idx in hourly_indices if idx < len(hourly_metadata)]
            if hourly_indices:
                daily_to_hourly_map[daily_idx] = hourly_indices
                for h_idx in hourly_indices:
                    hourly_to_daily_map[h_idx] = daily_idx
        return daily_to_hourly_map, hourly_to_daily_map
    
    # Convert timestamps to datetime for easier comparison
    daily_metadata = daily_metadata.copy()
    hourly_metadata = hourly_metadata.copy()
    
    if pd.api.types.is_numeric_dtype(daily_metadata['closeTime']):
        daily_metadata['datetime'] = pd.to_datetime(daily_metadata['closeTime'], unit='s')
    else:
        daily_metadata['datetime'] = pd.to_datetime(daily_metadata['closeTime'])
        
    if pd.api.types.is_numeric_dtype(hourly_metadata['closeTime']):
        hourly_metadata['datetime'] = pd.to_datetime(hourly_metadata['closeTime'], unit='s')
    else:
        hourly_metadata['datetime'] = pd.to_datetime(hourly_metadata['closeTime'])
    
    # Extract just the date part for daily bars
    daily_metadata['date'] = daily_metadata['datetime'].dt.date
    hourly_metadata['date'] = hourly_metadata['datetime'].dt.date
    
    # Map each 4h bar to its corresponding daily bar
    for hourly_idx, hourly_row in hourly_metadata.iterrows():
        hourly_date = hourly_row['date']
        
        # Find matching daily bar
        matching_daily = daily_metadata[daily_metadata['date'] == hourly_date]
        
        if not matching_daily.empty:
            daily_idx = matching_daily.index[0]
            
            # Add to hourly-to-daily map
            hourly_to_daily_map[hourly_idx] = daily_idx
            
            # Add to daily-to-hourly map
            if daily_idx not in daily_to_hourly_map:
                daily_to_hourly_map[daily_idx] = []
            daily_to_hourly_map[daily_idx].append(hourly_idx)
    
    return daily_to_hourly_map, hourly_to_daily_map

def combine_predictions(daily_preds, hourly_preds, hourly_to_daily_map, method='hierarchical', daily_threshold=0.5):
    """
    Combine predictions using different strategies
    
    Args:
        daily_preds: Dictionary with daily prediction results
        hourly_preds: Dictionary with hourly prediction results
        hourly_to_daily_map: Mapping from hourly indices to daily indices
        method: Combination method ('hierarchical', 'weighted', 'max', 'min')
        daily_threshold: Threshold for daily predictions in hierarchical method
    
    Returns:
        Dict with combined prediction scores and targets
    """
    # Create arrays for combined predictions
    combined_scores = np.zeros_like(hourly_preds['prediction_scores'])
    combined_targets = hourly_preds['targets'].copy()
    
    # Apply combination method
    for hourly_idx in range(len(hourly_preds['prediction_scores'])):
        # Get corresponding daily prediction if available
        if hourly_idx in hourly_to_daily_map:
            daily_idx = hourly_to_daily_map[hourly_idx]
            daily_score = daily_preds['prediction_scores'][daily_idx]
            daily_conf = daily_preds['confidence_scores'][daily_idx]
        else:
            # No matching daily bar, use neutral values
            daily_score = 0.5
            daily_conf = 0.5
        
        hourly_score = hourly_preds['prediction_scores'][hourly_idx]
        hourly_conf = hourly_preds['confidence_scores'][hourly_idx]
        
        # Apply selected combination method
        if method == 'hierarchical':
            # Only consider hourly signals that align with daily bias
            if daily_score >= daily_threshold:
                combined_scores[hourly_idx] = hourly_score
            else:
                combined_scores[hourly_idx] = 0.0  # Or some low score
                
        elif method == 'weighted':
            # Confidence-weighted average
            total_conf = daily_conf + hourly_conf
            if total_conf > 0:
                w1 = daily_conf / total_conf
                w2 = hourly_conf / total_conf
                combined_scores[hourly_idx] = w1 * daily_score + w2 * hourly_score
            else:
                combined_scores[hourly_idx] = (daily_score + hourly_score) / 2
                
        elif method == 'max':
            # Take maximum of the two predictions
            combined_scores[hourly_idx] = max(daily_score, hourly_score)
            
        elif method == 'min':
            # Take minimum of the two predictions
            combined_scores[hourly_idx] = min(daily_score, hourly_score)
            
        elif method == 'product':
            # Multiply probabilities (more conservative)
            combined_scores[hourly_idx] = daily_score * hourly_score
            
        elif method == 'geometric':
            # Geometric mean
            combined_scores[hourly_idx] = np.sqrt(daily_score * hourly_score)
            
        else:
            # Default to simple average
            combined_scores[hourly_idx] = (daily_score + hourly_score) / 2
    
    return {
        'prediction_scores': combined_scores,
        'targets': combined_targets,
        'method': method
    }

def calculate_metrics(predictions):
    """Calculate ROC AUC and average precision"""
    try:
        auc = roc_auc_score(predictions['targets'], predictions['prediction_scores'])
    except:
        auc = 0.5
        
    try:
        avg_precision = average_precision_score(predictions['targets'], predictions['prediction_scores'])
    except:
        avg_precision = np.mean(predictions['targets'])
        
    return {
        'auc': auc,
        'average_precision': avg_precision
    }

def calculate_precision_recall_gain(prediction_scores, targets):
    """Calculate precision-recall gain curve"""
    # Get precision-recall curve
    precision, recall, thresholds = precision_recall_curve(targets, prediction_scores)
    
    # Calculate baseline (random classifier)
    baseline = np.mean(targets)
    
    # Calculate Precision Gain and Recall Gain
    precision_gain = (precision - baseline) / (1 - baseline + 1e-10)
    recall_gain = recall / (baseline + 1e-10)
    
    return precision_gain, recall_gain, thresholds

def plot_roc_curves(results, output_path):
    """Plot ROC curves for different models/combinations"""
    plt.figure(figsize=(12, 8))
    
    for name, preds in results.items():
        metrics = calculate_metrics(preds)
        fpr, tpr, _ = roc_curve(preds['targets'], preds['prediction_scores'])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {metrics['auc']:.4f})")
    
    # Add diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()

def plot_pr_gain_curves(results, output_path):
    """Plot precision-recall gain curves for different models/combinations"""
    plt.figure(figsize=(12, 8))
    
    for name, preds in results.items():
        precision_gain, recall_gain, _ = calculate_precision_recall_gain(
            preds['prediction_scores'], preds['targets']
        )
        
        # Find the point where recall gain = 1
        sorted_indices = np.argsort(np.abs(recall_gain - 1.0))
        rg1_idx = sorted_indices[0]  # Index of point closest to recall gain = 1.0
        rg1_precision_gain = precision_gain[rg1_idx]
        
        plt.plot(recall_gain, precision_gain, 
                 label=f"{name} (PG@RG1 = {rg1_precision_gain:.4f})")
        
        # Mark the RG=1 point
        plt.scatter([1], [rg1_precision_gain], marker='o')
    
    plt.xlabel('Recall Gain')
    plt.ylabel('Precision Gain')
    plt.title('Precision-Recall Gain Curves')
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()

def plot_threshold_distributions(results, output_path):
    """Plot distribution of prediction scores"""
    plt.figure(figsize=(12, 8))
    
    for name, preds in results.items():
        sns.kdeplot(preds['prediction_scores'], label=name)
    
    plt.xlabel('Prediction Score')
    plt.ylabel('Density')
    plt.title('Distribution of Prediction Scores')
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()

def main():
    args = parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    print("Loading daily model...")
    daily_model, daily_config = load_model(args.daily_model, device)
    
    print("Loading hourly model...")
    hourly_model, hourly_config = load_model(args.hourly_model, device)
    
    # Load data
    print("Loading test data...")
    daily_data, daily_metadata = load_data(args.daily_data)
    hourly_data, hourly_metadata = load_data(args.hourly_data)
    
    # Get model predictions
    print("Getting daily model predictions...")
    daily_preds = get_model_predictions(daily_model, daily_data, device)
    
    print("Getting hourly model predictions...")
    hourly_preds = get_model_predictions(hourly_model, hourly_data, device)
    
    # Align timeframes
    print("Aligning timeframes...")
    daily_to_hourly_map, hourly_to_daily_map = align_timeframes(daily_metadata, hourly_metadata)
    
    print(f"Mapped {len(hourly_to_daily_map)} hourly bars to {len(daily_to_hourly_map)} daily bars")
    
    # Calculate baseline metrics
    daily_metrics = calculate_metrics(daily_preds)
    hourly_metrics = calculate_metrics(hourly_preds)
    
    print(f"Daily model:   AUC = {daily_metrics['auc']:.4f}, AP = {daily_metrics['average_precision']:.4f}")
    print(f"Hourly model:  AUC = {hourly_metrics['auc']:.4f}, AP = {hourly_metrics['average_precision']:.4f}")
    
    # Combine predictions using different strategies
    combination_methods = [
        'hierarchical', 'weighted', 'max', 'min', 'product', 'geometric'
    ]
    
    # Store results for each method
    results = {
        'Daily': daily_preds,
        'Hourly': hourly_preds
    }
    
    for method in combination_methods:
        print(f"Combining predictions using {method} method...")
        combined_preds = combine_predictions(
            daily_preds, hourly_preds, hourly_to_daily_map, 
            method=method, daily_threshold=0.5
        )
        
        combined_metrics = calculate_metrics(combined_preds)
        print(f"{method.capitalize()} combination: AUC = {combined_metrics['auc']:.4f}, AP = {combined_metrics['average_precision']:.4f}")
        
        # Add to results
        results[method.capitalize()] = combined_preds
    
    # Create visualizations
    print("Creating visualizations...")
    plot_roc_curves(results, output_dir / 'roc_curves.png')
    plot_pr_gain_curves(results, output_dir / 'pr_gain_curves.png')
    plot_threshold_distributions(results, output_dir / 'threshold_distributions.png')
    
    # Save metrics to JSON
    metrics_summary = {}
    for name, preds in results.items():
        metrics = calculate_metrics(preds)
        
        # Calculate precision gain at recall gain = 1
        precision_gain, recall_gain, _ = calculate_precision_recall_gain(
            preds['prediction_scores'], preds['targets']
        )
        
        # Find the point where recall gain = 1
        sorted_indices = np.argsort(np.abs(recall_gain - 1.0))
        rg1_idx = sorted_indices[0]  # Index of point closest to recall gain = 1.0
        rg1_precision_gain = precision_gain[rg1_idx]
        
        metrics['precision_gain_at_rg1'] = float(rg1_precision_gain)
        metrics_summary[name] = metrics
    
    with open(output_dir / 'metrics_summary.json', 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
    
    
    
python model_combination_evaluator.py --daily_model /path/to/daily/model/models/best_checkpoint.pt --hourly_model /path/to/4h/model/models/best_checkpoint.pt --daily_data /path/to/daily_dataset/test.pt --hourly_data /path/to/4h_dataset/test.pt --output_dir model_combination_results