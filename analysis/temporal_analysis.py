#!/usr/bin/env python3
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import roc_auc_score, precision_score, average_precision_score, roc_curve, precision_recall_curve

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze model performance over time')
    parser.add_argument('--run_dir', type=str, required=True, 
                        help='Path to the training run directory containing config.json')
    parser.add_argument('--checkpoint', type=str, default='best_checkpoint.pt',
                        help='Model checkpoint to use (default: best_checkpoint.pt)')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'],
                        help='Data split to analyze (default: test)')
    parser.add_argument('--window_size', type=int, default=500,
                        help='Size of sliding window in number of samples (default: 500)')
    parser.add_argument('--window_step', type=int, default=100,
                        help='Step size for sliding window (default: 100)')
    parser.add_argument('--mode', type=str, default='sliding', choices=['sliding', 'cumulative'],
                        help='Analysis mode: sliding window or cumulative (default: sliding)')
    parser.add_argument('--precision_threshold', type=float, default=None,
                        help='Fixed probability threshold for precision calculation (default: None, uses optimized threshold)')
    parser.add_argument('--quantile_threshold', type=float, default=0.9,
                        help='Quantile threshold for high precision regime (default: 0.9)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for analysis results (default: <run_dir>/temporal_analysis)')
    
    return parser.parse_args()

def load_config(run_dir):
    """Load configuration file"""
    config_path = Path(run_dir) / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)

def load_model(run_dir, checkpoint_file, device):
    """Load the trained model from checkpoint"""
    models_dir = Path(run_dir) / 'models'
    checkpoint_path = models_dir / checkpoint_file
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    # Import necessary modules
    import sys
    sys.path.append(str(Path(run_dir).parent))  # Add parent directory to path
    try:
        from model import create_model, HierarchicalGRUModel
    except ImportError:
        try:
            # Try loading from current directory
            from model import create_model, HierarchicalGRUModel
        except ImportError:
            raise ImportError("Cannot import model module. Make sure model.py is accessible.")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model, _ = create_model(checkpoint['config'], device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['config']

def load_data(config, split):
    """Load dataset"""
    dataset_dir = Path(config['paths']['dataset_dir'])
    data_path = dataset_dir / f'{split}.pt'
    metadata_path = dataset_dir / f'{split}_metadata.csv'
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {data_path}")
    
    data = torch.load(data_path)
    
    # Load metadata if available
    metadata = None
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
    else:
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    
    return data, metadata

def get_predictions(model, data, device, batch_size=64):
    """Get model predictions"""
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

def get_optimized_threshold(scores, targets):
    """Find the optimal threshold that maximizes F1 score"""
    precision, recall, thresholds = precision_recall_curve(targets, scores)
    
    # Calculate F1 score at each threshold
    f1_scores = np.zeros_like(thresholds)
    for i, threshold in enumerate(thresholds):
        predictions = (scores >= threshold).astype(int)
        # Avoid division by zero
        if np.sum(predictions) == 0:
            f1_scores[i] = 0
        else:
            prec = precision[i]
            rec = recall[i+1]  # Precision and recall arrays have different lengths
            f1_scores[i] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    # Get threshold with maximum F1 score
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx]

def calculate_metrics_over_time(predictions, metadata, args):
    """Calculate metrics over time using sliding window or cumulative approach"""
    # Get timestamps in chronological order
    timestamps = metadata['closeTime'].values
    unique_timestamps = np.sort(np.unique(timestamps))
    
    # Create a mapping between sample index and its position in chronological order
    time_order = np.argsort(timestamps)
    
    # Sort predictions and targets by time
    sorted_scores = predictions['prediction_scores'][time_order]
    sorted_confidence = predictions['confidence_scores'][time_order]
    sorted_targets = predictions['targets'][time_order]
    sorted_metadata = metadata.iloc[time_order].reset_index(drop=True)
    
    # Get trading durations if available
    has_duration = 'bars_to_exit' in sorted_metadata.columns
    if has_duration:
        sorted_durations = sorted_metadata['bars_to_exit'].values
    
    # Determine threshold selection
    if args.precision_threshold is not None:
        # Use fixed threshold
        threshold = args.precision_threshold
        print(f"Using fixed threshold: {threshold}")
    else:
        # Use optimized threshold
        threshold = get_optimized_threshold(sorted_scores, sorted_targets)
        print(f"Optimized threshold: {threshold}")
    
    # Also calculate high-precision threshold (top X% of predictions)
    high_precision_threshold = np.quantile(sorted_scores, args.quantile_threshold)
    print(f"High precision threshold (quantile {args.quantile_threshold}): {high_precision_threshold}")
    
    # Initialize result containers
    result = {
        'timestamps': [],
        'auc': [],
        'precision': [],
        'high_precision': [],
        'avg_precision': [],
        'positive_rate': [],
        'count': []
    }
    
    if has_duration:
        result['avg_duration'] = []
        result['win_duration'] = []
        result['loss_duration'] = []
    
    # Add confidence metrics
    result['avg_confidence'] = []
    result['confidence_correlation'] = []
    
    # Sliding window analysis
    if args.mode == 'sliding':
        for start_idx in tqdm(range(0, len(sorted_scores), args.window_step), desc="Sliding window analysis"):
            end_idx = min(start_idx + args.window_size, len(sorted_scores))
            
            # Skip if window is too small
            if end_idx - start_idx < 50:  # Require at least 50 samples
                continue
            
            # Get data for this window
            window_scores = sorted_scores[start_idx:end_idx]
            window_targets = sorted_targets[start_idx:end_idx]
            window_confidence = sorted_confidence[start_idx:end_idx]
            
            # Get representative timestamp (median time in window)
            window_times = sorted_metadata['closeTime'].iloc[start_idx:end_idx].values
            median_time = np.median(window_times)
            result['timestamps'].append(median_time)
            
            # Calculate metrics
            try:
                # AUC
                auc = roc_auc_score(window_targets, window_scores)
                result['auc'].append(auc)
                
                # Average precision
                avg_prec = average_precision_score(window_targets, window_scores)
                result['avg_precision'].append(avg_prec)
                
                # Precision at threshold
                window_preds = (window_scores >= threshold).astype(int)
                if np.sum(window_preds) > 0:
                    prec = precision_score(window_targets, window_preds)
                else:
                    prec = np.nan
                result['precision'].append(prec)
                
                # High precision metrics
                high_prec_preds = (window_scores >= high_precision_threshold).astype(int)
                if np.sum(high_prec_preds) > 0:
                    high_prec = precision_score(window_targets, high_prec_preds)
                else:
                    high_prec = np.nan
                result['high_precision'].append(high_prec)
                
                # Positive rate
                result['positive_rate'].append(np.mean(window_targets))
                
                # Count
                result['count'].append(len(window_targets))
                
                # Confidence metrics
                result['avg_confidence'].append(np.mean(window_confidence))
                
                # Confidence-correctness correlation
                prediction_error = np.abs(window_scores - window_targets)
                expected_confidence = 1.0 - prediction_error
                conf_corr = np.corrcoef(window_confidence, expected_confidence)[0, 1]
                result['confidence_correlation'].append(conf_corr)
                
                # Duration metrics if available
                if has_duration:
                    window_durations = sorted_durations[start_idx:end_idx]
                    result['avg_duration'].append(np.mean(window_durations))
                    
                    # Separate durations by outcome
                    win_durations = window_durations[window_targets == 1]
                    loss_durations = window_durations[window_targets == 0]
                    
                    result['win_duration'].append(np.mean(win_durations) if len(win_durations) > 0 else np.nan)
                    result['loss_duration'].append(np.mean(loss_durations) if len(loss_durations) > 0 else np.nan)
                
            except Exception as e:
                print(f"Error calculating metrics for window {start_idx}:{end_idx}: {e}")
                # Add NaN values for this window
                for key in result:
                    if key != 'timestamps' and len(result[key]) < len(result['timestamps']):
                        result[key].append(np.nan)
    
    # Cumulative analysis
    else:  # args.mode == 'cumulative'
        for end_idx in tqdm(range(args.window_size, len(sorted_scores), args.window_step), desc="Cumulative analysis"):
            # Get data from beginning to this point
            cumulative_scores = sorted_scores[:end_idx]
            cumulative_targets = sorted_targets[:end_idx]
            cumulative_confidence = sorted_confidence[:end_idx]
            
            # Get representative timestamp (last time in cumulative window)
            current_time = sorted_metadata['closeTime'].iloc[end_idx-1]
            result['timestamps'].append(current_time)
            
            # Calculate metrics
            try:
                # AUC
                auc = roc_auc_score(cumulative_targets, cumulative_scores)
                result['auc'].append(auc)
                
                # Average precision
                avg_prec = average_precision_score(cumulative_targets, cumulative_scores)
                result['avg_precision'].append(avg_prec)
                
                # Precision at threshold
                cumulative_preds = (cumulative_scores >= threshold).astype(int)
                if np.sum(cumulative_preds) > 0:
                    prec = precision_score(cumulative_targets, cumulative_preds)
                else:
                    prec = np.nan
                result['precision'].append(prec)
                
                # High precision metrics
                high_prec_preds = (cumulative_scores >= high_precision_threshold).astype(int)
                if np.sum(high_prec_preds) > 0:
                    high_prec = precision_score(cumulative_targets, high_prec_preds)
                else:
                    high_prec = np.nan
                result['high_precision'].append(high_prec)
                
                # Positive rate
                result['positive_rate'].append(np.mean(cumulative_targets))
                
                # Count
                result['count'].append(len(cumulative_targets))
                
                # Confidence metrics
                result['avg_confidence'].append(np.mean(cumulative_confidence))
                
                # Confidence-correctness correlation
                prediction_error = np.abs(cumulative_scores - cumulative_targets)
                expected_confidence = 1.0 - prediction_error
                conf_corr = np.corrcoef(cumulative_confidence, expected_confidence)[0, 1]
                result['confidence_correlation'].append(conf_corr)
                
                # Duration metrics if available
                if has_duration:
                    cumulative_durations = sorted_durations[:end_idx]
                    result['avg_duration'].append(np.mean(cumulative_durations))
                    
                    # Separate durations by outcome
                    win_durations = cumulative_durations[cumulative_targets == 1]
                    loss_durations = cumulative_durations[cumulative_targets == 0]
                    
                    result['win_duration'].append(np.mean(win_durations) if len(win_durations) > 0 else np.nan)
                    result['loss_duration'].append(np.mean(loss_durations) if len(loss_durations) > 0 else np.nan)
                
            except Exception as e:
                print(f"Error calculating metrics for cumulative window to {end_idx}: {e}")
                # Add NaN values for this window
                for key in result:
                    if key != 'timestamps' and len(result[key]) < len(result['timestamps']):
                        result[key].append(np.nan)
    
    # Convert to DataFrame
    result_df = pd.DataFrame({
        'timestamp': pd.to_datetime(result['timestamps'], unit='s'),
        'AUC': result['auc'],
        'Precision': result['precision'],
        'High_Precision': result['high_precision'],
        'Avg_Precision': result['avg_precision'],
        'Positive_Rate': result['positive_rate'],
        'Count': result['count'],
        'Avg_Confidence': result['avg_confidence'],
        'Confidence_Correlation': result['confidence_correlation']
    })
    
    # Add duration metrics if available
    if has_duration:
        result_df['Avg_Duration'] = result['avg_duration']
        result_df['Win_Duration'] = result['win_duration']
        result_df['Loss_Duration'] = result['loss_duration']
    
    return result_df, threshold, high_precision_threshold

def plot_metrics_over_time(result_df, output_dir, args, has_duration=False):
    """Create visualizations of metrics over time"""
    # Convert timestamp to datetime if it's not already
    if not pd.api.types.is_datetime64_dtype(result_df['timestamp']):
        result_df['timestamp'] = pd.to_datetime(result_df['timestamp'], unit='s')
    
    # Set a consistent style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (14, 8)
    
    # 1. Plot AUC and Precision metrics
    fig, ax1 = plt.subplots()
    
    # Plot AUC
    ax1.plot(result_df['timestamp'], result_df['AUC'], 'b-', label='AUC', linewidth=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('AUC', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Plot precision metrics on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(result_df['timestamp'], result_df['Precision'], 'r-', label=f'Precision @ {args.precision_threshold}', linewidth=2)
    ax2.plot(result_df['timestamp'], result_df['High_Precision'], 'g-', 
             label=f'High Precision (top {int(100*(1-args.quantile_threshold))}%)', linewidth=2)
    ax2.set_ylabel('Precision', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add baseline (positive rate)
    ax2.plot(result_df['timestamp'], result_df['Positive_Rate'], 'k--', 
             label='Positive Rate (baseline)', alpha=0.7)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    title = f"{args.mode.capitalize()} Window Analysis - AUC and Precision"
    if args.mode == 'sliding':
        title += f" (window: {args.window_size}, step: {args.window_step})"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_dir / 'auc_precision_over_time.png', dpi=150)
    plt.close()
    
    # 2. Plot duration metrics if available
    if has_duration:
        plt.figure()
        
        plt.plot(result_df['timestamp'], result_df['Avg_Duration'], 'b-', 
                 label='Average Duration', linewidth=2)
        plt.plot(result_df['timestamp'], result_df['Win_Duration'], 'g-', 
                 label='Win Duration', linewidth=2)
        plt.plot(result_df['timestamp'], result_df['Loss_Duration'], 'r-', 
                 label='Loss Duration', linewidth=2)
        
        plt.xlabel('Time')
        plt.ylabel('Duration (bars)')
        plt.title(f"{args.mode.capitalize()} Window Analysis - Trade Duration")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'duration_over_time.png', dpi=150)
        plt.close()
    
    # 3. Plot sample count
    plt.figure()
    plt.plot(result_df['timestamp'], result_df['Count'], 'k-', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Sample Count')
    plt.title(f"{args.mode.capitalize()} Window Analysis - Sample Count")
    plt.tight_layout()
    plt.savefig(output_dir / 'sample_count_over_time.png', dpi=150)
    plt.close()
    
    # 4. Plot confidence metrics
    plt.figure()
    plt.plot(result_df['timestamp'], result_df['Avg_Confidence'], 'b-', 
             label='Average Confidence', linewidth=2)
    plt.plot(result_df['timestamp'], result_df['Confidence_Correlation'], 'g-', 
             label='Confidence-Correctness Correlation', linewidth=2)
    
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f"{args.mode.capitalize()} Window Analysis - Confidence Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_over_time.png', dpi=150)
    plt.close()
    
    # 5. Create a heatmap of correlations between metrics
    correlation_cols = ['AUC', 'Precision', 'High_Precision', 'Avg_Precision', 
                        'Positive_Rate', 'Avg_Confidence', 'Confidence_Correlation']
    
    if has_duration:
        correlation_cols.extend(['Avg_Duration', 'Win_Duration', 'Loss_Duration'])
    
    corr_matrix = result_df[correlation_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Between Metrics')
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_correlation.png', dpi=150)
    plt.close()

def analyze_trends(result_df, output_dir, args):
    """Analyze trends in metrics over time"""
    # Create a trends summary
    trends = {}
    
    # Convert timestamp to datetime if it's not already
    if not pd.api.types.is_datetime64_dtype(result_df['timestamp']):
        result_df['timestamp'] = pd.to_datetime(result_df['timestamp'], unit='s')
    
    # Calculate overall trends
    for col in ['AUC', 'Precision', 'High_Precision', 'Positive_Rate']:
        if col in result_df.columns:
            # Calculate slope using linear regression
            x = np.arange(len(result_df))
            y = result_df[col].values
            valid_mask = ~np.isnan(y)
            
            if np.sum(valid_mask) > 2:  # Need at least 3 points for regression
                x_valid = x[valid_mask]
                y_valid = y[valid_mask]
                
                slope, intercept = np.polyfit(x_valid, y_valid, 1)
                trends[f'{col}_slope'] = slope
                
                # Calculate percent change from first to last point
                first_valid = y_valid[0]
                last_valid = y_valid[-1]
                pct_change = (last_valid - first_valid) / first_valid if first_valid != 0 else np.nan
                trends[f'{col}_pct_change'] = pct_change
                
                # Calculate volatility (standard deviation)
                trends[f'{col}_volatility'] = np.std(y_valid)
    
    # Identify periodic patterns using autocorrelation
    for col in ['AUC', 'Precision', 'High_Precision']:
        if col in result_df.columns:
            data = result_df[col].values
            valid_mask = ~np.isnan(data)
            
            if np.sum(valid_mask) > 10:  # Need sufficient points for autocorrelation
                data_valid = data[valid_mask]
                
                # Calculate autocorrelation
                from statsmodels.tsa.stattools import acf
                try:
                    acf_values = acf(data_valid, nlags=min(20, len(data_valid)//2), fft=False)
                    
                    # Find significant peaks in autocorrelation
                    from scipy.signal import find_peaks
                    peaks, _ = find_peaks(acf_values, height=0.2)  # Peaks with correlation > 0.2
                    
                    if len(peaks) > 0:
                        trends[f'{col}_periodicity'] = peaks[0]  # First significant peak
                    else:
                        trends[f'{col}_periodicity'] = None
                except:
                    trends[f'{col}_periodicity'] = None
    
    # Plot trend analysis
    plt.figure(figsize=(14, 8))
    
    # Plot AUC with trend line
    plt.subplot(2, 2, 1)
    x = np.arange(len(result_df))
    y = result_df['AUC'].values
    valid_mask = ~np.isnan(y)
    
    plt.plot(result_df['timestamp'][valid_mask], y[valid_mask], 'b-', label='AUC')
    
    if 'AUC_slope' in trends:
        # Plot trend line
        x_valid = x[valid_mask]
        slope = trends['AUC_slope']
        intercept = np.polyfit(x_valid, y[valid_mask], 1)[1]
        plt.plot(result_df['timestamp'][valid_mask], 
                 slope * x_valid + intercept, 'r--', 
                 label=f'Trend (slope: {slope:.4f})')
    
    plt.xlabel('Time')
    plt.ylabel('AUC')
    plt.title('AUC Trend Analysis')
    plt.legend()
    
    # Plot Precision with trend line
    plt.subplot(2, 2, 2)
    y = result_df['Precision'].values
    valid_mask = ~np.isnan(y)
    
    plt.plot(result_df['timestamp'][valid_mask], y[valid_mask], 'g-', label='Precision')
    
    if 'Precision_slope' in trends:
        # Plot trend line
        x_valid = x[valid_mask]
        slope = trends['Precision_slope']
        intercept = np.polyfit(x_valid, y[valid_mask], 1)[1]
        plt.plot(result_df['timestamp'][valid_mask], 
                 slope * x_valid + intercept, 'r--', 
                 label=f'Trend (slope: {slope:.4f})')
    
    plt.xlabel('Time')
    plt.ylabel('Precision')
    plt.title('Precision Trend Analysis')
    plt.legend()
    
    # Plot High Precision with trend line
    plt.subplot(2, 2, 3)
    y = result_df['High_Precision'].values
    valid_mask = ~np.isnan(y)
    
    plt.plot(result_df['timestamp'][valid_mask], y[valid_mask], 'm-', label='High Precision')
    
    if 'High_Precision_slope' in trends:
        # Plot trend line
        x_valid = x[valid_mask]
        slope = trends['High_Precision_slope']
        intercept = np.polyfit(x_valid, y[valid_mask], 1)[1]
        plt.plot(result_df['timestamp'][valid_mask], 
                 slope * x_valid + intercept, 'r--', 
                 label=f'Trend (slope: {slope:.4f})')
    
    plt.xlabel('Time')
    plt.ylabel('High Precision')
    plt.title('High Precision Trend Analysis')
    plt.legend()
    
    # Plot Positive Rate with trend line
    plt.subplot(2, 2, 4)
    y = result_df['Positive_Rate'].values
    valid_mask = ~np.isnan(y)
    
    plt.plot(result_df['timestamp'][valid_mask], y[valid_mask], 'k-', label='Positive Rate')
    
    if 'Positive_Rate_slope' in trends:
        # Plot trend line
        x_valid = x[valid_mask]
        slope = trends['Positive_Rate_slope']
        intercept = np.polyfit(x_valid, y[valid_mask], 1)[1]
        plt.plot(result_df['timestamp'][valid_mask], 
                 slope * x_valid + intercept, 'r--', 
                 label=f'Trend (slope: {slope:.4f})')
    
    plt.xlabel('Time')
    plt.ylabel('Positive Rate')
    plt.title('Positive Rate Trend Analysis')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'trend_analysis.png', dpi=150)
    plt.close()
    
    # Convert NumPy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
            
    # Save trends to a JSON file
    with open(output_dir / 'trends_summary.json', 'w') as f:
        json.dump(convert_numpy_types(trends), f, indent=4)
    
    return trends

def analyze_asset_performance(result_df, metadata, predictions, output_dir, args):
    """Analyze performance metrics by asset if multiple assets are present"""
    # Check if we have asset information
    if 'code' not in metadata.columns:
        return None
    
    # Get unique assets
    assets = metadata['code'].unique()
    
    # If only one asset, skip this analysis
    if len(assets) <= 1:
        return None
    
    print(f"Analyzing performance across {len(assets)} assets...")
    
    # Get timestamps in chronological order
    timestamps = metadata['closeTime'].values
    time_order = np.argsort(timestamps)
    
    # Sort predictions and targets by time
    sorted_scores = predictions['prediction_scores'][time_order]
    sorted_targets = predictions['targets'][time_order]
    sorted_metadata = metadata.iloc[time_order].reset_index(drop=True)
    
    # Create per-asset performance data
    asset_performance = {}
    
    for asset in assets:
        asset_mask = sorted_metadata['code'] == asset
        
        # Skip assets with insufficient data
        if np.sum(asset_mask) < 50:
            continue
        
        asset_scores = sorted_scores[asset_mask]
        asset_targets = sorted_targets[asset_mask]
        
        # Calculate metrics
        try:
            # AUC
            auc = roc_auc_score(asset_targets, asset_scores)
            
            # Average precision
            avg_prec = average_precision_score(asset_targets, asset_scores)
            
            # Positive rate
            positive_rate = np.mean(asset_targets)
            
            # Calculate precision at optimal threshold
            opt_threshold = get_optimized_threshold(asset_scores, asset_targets)
            asset_preds = (asset_scores >= opt_threshold).astype(int)
            
            if np.sum(asset_preds) > 0:
                precision = precision_score(asset_targets, asset_preds)
            else:
                precision = np.nan
            
            # Calculate precision at high threshold (top 10%)
            high_threshold = np.quantile(asset_scores, 0.9)
            high_preds = (asset_scores >= high_threshold).astype(int)
            
            if np.sum(high_preds) > 0:
                high_precision = precision_score(asset_targets, high_preds)
            else:
                high_precision = np.nan
            
            # Store metrics
            asset_performance[asset] = {
                'count': np.sum(asset_mask),
                'auc': auc,
                'avg_precision': avg_prec,
                'precision': precision,
                'high_precision': high_precision,
                'positive_rate': positive_rate,
                'optimal_threshold': opt_threshold
            }
            
            # Add duration metrics if available
            if 'bars_to_exit' in sorted_metadata.columns:
                asset_durations = sorted_metadata.loc[asset_mask, 'bars_to_exit'].values
                asset_performance[asset]['avg_duration'] = np.mean(asset_durations)
                
                # Separate durations by outcome
                win_durations = asset_durations[asset_targets == 1]
                loss_durations = asset_durations[asset_targets == 0]
                
                asset_performance[asset]['win_duration'] = np.mean(win_durations) if len(win_durations) > 0 else np.nan
                asset_performance[asset]['loss_duration'] = np.mean(loss_durations) if len(loss_durations) > 0 else np.nan
            
        except Exception as e:
            print(f"Error calculating metrics for asset {asset}: {e}")
    
    # Convert to DataFrame for easier analysis
    asset_df = pd.DataFrame.from_dict(asset_performance, orient='index')
    asset_df.index.name = 'Asset'
    asset_df.reset_index(inplace=True)
    
    # Sort by count (most frequent assets first)
    asset_df.sort_values('count', ascending=False, inplace=True)
    
    # Save asset performance data
    asset_df.to_csv(output_dir / 'asset_performance.csv', index=False)
    
    # Create visualizations
    plt.figure(figsize=(14, 8))
    
    # Plot AUC by asset
    plt.subplot(2, 2, 1)
    sns.barplot(x='Asset', y='auc', data=asset_df)
    plt.title('AUC by Asset')
    plt.xticks(rotation=45)
    plt.ylim(0.5, 1.0)  # AUC range
    
    # Plot precision by asset
    plt.subplot(2, 2, 2)
    sns.barplot(x='Asset', y='precision', data=asset_df)
    plt.title('Precision by Asset')
    plt.xticks(rotation=45)
    plt.ylim(0, 1.0)
    
    # Plot high precision by asset
    plt.subplot(2, 2, 3)
    sns.barplot(x='Asset', y='high_precision', data=asset_df)
    plt.title('High Precision by Asset')
    plt.xticks(rotation=45)
    plt.ylim(0, 1.0)
    
    # Plot positive rate by asset
    plt.subplot(2, 2, 4)
    sns.barplot(x='Asset', y='positive_rate', data=asset_df)
    plt.title('Positive Rate by Asset')
    plt.xticks(rotation=45)
    plt.ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'asset_performance.png', dpi=150)
    plt.close()
    
    # If duration metrics are available, create a duration chart
    if 'avg_duration' in asset_df.columns:
        plt.figure(figsize=(14, 6))
        
        # Plot average duration by asset
        plt.subplot(1, 2, 1)
        sns.barplot(x='Asset', y='avg_duration', data=asset_df)
        plt.title('Average Duration by Asset')
        plt.xticks(rotation=45)
        
        # Plot win vs loss duration by asset
        plt.subplot(1, 2, 2)
        
        # Reshape data for grouped bar chart
        duration_data = []
        for _, row in asset_df.iterrows():
            duration_data.append({
                'Asset': row['Asset'],
                'Type': 'Win',
                'Duration': row['win_duration']
            })
            duration_data.append({
                'Asset': row['Asset'],
                'Type': 'Loss',
                'Duration': row['loss_duration']
            })
        
        duration_df = pd.DataFrame(duration_data)
        
        sns.barplot(x='Asset', y='Duration', hue='Type', data=duration_df)
        plt.title('Win vs Loss Duration by Asset')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'asset_duration.png', dpi=150)
        plt.close()
    
    return asset_df

def analyze_performance_vs_market(result_df, metadata, predictions, output_dir, args):
    """Analyze how model performance correlates with market indicators if available"""
    # Check if we have price information
    has_price = all(col in metadata.columns for col in ['close', 'closeTime'])
    if not has_price:
        return None
    
    print("Analyzing performance vs market indicators...")
    
    # Get timestamps in chronological order
    timestamps = metadata['closeTime'].values
    unique_timestamps = np.sort(np.unique(timestamps))
    
    # Create a time series of prices
    prices = []
    price_times = []
    
    # For each unique timestamp, take the median price across all assets
    for ts in unique_timestamps:
        mask = timestamps == ts
        median_price = np.median(metadata.loc[mask, 'close'].values)
        prices.append(median_price)
        price_times.append(ts)
    
    price_df = pd.DataFrame({
        'timestamp': pd.to_datetime(price_times, unit='s'),
        'price': prices
    })
    
    # Calculate returns
    price_df['return'] = price_df['price'].pct_change()
    
    # Calculate volatility (rolling standard deviation of returns)
    price_df['volatility'] = price_df['return'].rolling(window=10).std()
    
    # Calculate trend (direction of price movement)
    price_df['trend'] = price_df['price'].rolling(window=20).mean().pct_change()
    
    # Align market indicators with performance metrics
    merged_df = pd.merge_asof(result_df.sort_values('timestamp'), 
                             price_df.sort_values('timestamp'),
                             on='timestamp',
                             direction='nearest')
    
    # Calculate correlations
    market_corr = merged_df[['AUC', 'Precision', 'High_Precision', 'price', 
                             'return', 'volatility', 'trend']].corr()
    
    # Create visualizations
    plt.figure(figsize=(14, 8))
    
    # Plot price and AUC
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(merged_df['timestamp'], merged_df['price'], 'b-', label='Price')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax2 = ax1.twinx()
    ax2.plot(merged_df['timestamp'], merged_df['AUC'], 'r-', label='AUC')
    ax2.set_ylabel('AUC', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.title('Price vs AUC')
    
    # Plot volatility and precision
    ax1 = plt.subplot(2, 2, 2)
    ax1.plot(merged_df['timestamp'], merged_df['volatility'], 'g-', label='Volatility')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Volatility', color='g')
    ax1.tick_params(axis='y', labelcolor='g')
    
    ax2 = ax1.twinx()
    ax2.plot(merged_df['timestamp'], merged_df['Precision'], 'm-', label='Precision')
    ax2.set_ylabel('Precision', color='m')
    ax2.tick_params(axis='y', labelcolor='m')
    
    plt.title('Volatility vs Precision')
    
    # Plot correlation heatmap
    plt.subplot(2, 2, 3)
    sns.heatmap(market_corr.iloc[:3, 3:], annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation: Metrics vs Market')
    
    # Plot scatterplot of most significant correlation
    plt.subplot(2, 2, 4)
    
    # Find most correlated pair
    metric_cols = ['AUC', 'Precision', 'High_Precision']
    market_cols = ['price', 'return', 'volatility', 'trend']
    
    max_corr = 0
    max_pair = ('AUC', 'price')
    
    for metric in metric_cols:
        for market in market_cols:
            corr = abs(market_corr.loc[metric, market])
            if corr > max_corr:
                max_corr = corr
                max_pair = (metric, market)
    
    # Plot the most correlated pair
    sns.regplot(x=max_pair[1], y=max_pair[0], data=merged_df)
    plt.title(f'{max_pair[0]} vs {max_pair[1]} (corr: {market_corr.loc[max_pair[0], max_pair[1]]:.2f})')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'market_performance.png', dpi=150)
    plt.close()
    
    # Save correlation matrix
    market_corr.to_csv(output_dir / 'market_correlation.csv')
    
    return market_corr

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare output directory
    run_dir = Path(args.run_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = run_dir / 'temporal_analysis' / args.split
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    try:
        # Load configuration
        config = load_config(run_dir)
        
        # Load model
        model, config = load_model(run_dir, args.checkpoint, device)
        
        # Load data
        data, metadata = load_data(config, args.split)
        
        # Get predictions
        print("Getting model predictions...")
        predictions = get_predictions(model, data, device)
        
        # Calculate metrics over time
        print(f"Calculating metrics over time using {args.mode} windows...")
        result_df, threshold, high_precision_threshold = calculate_metrics_over_time(
            predictions, metadata, args)
        
        # Check if we have duration data
        has_duration = 'Avg_Duration' in result_df.columns
        
        # Save metrics data
        result_df.to_csv(output_dir / 'metrics_over_time.csv', index=False)
        
        # Plot metrics over time
        print("Creating visualizations...")
        plot_metrics_over_time(result_df, output_dir, args, has_duration)
        
        # Analyze trends
        print("Analyzing trends...")
        trends = analyze_trends(result_df, output_dir, args)
        
        # Analyze asset performance if multiple assets
        asset_df = analyze_asset_performance(result_df, metadata, predictions, output_dir, args)
        
        # Analyze performance vs market indicators
        market_corr = analyze_performance_vs_market(result_df, metadata, predictions, output_dir, args)
        
        # Save summary information
        summary = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'split': args.split,
            'mode': args.mode,
            'window_size': args.window_size,
            'window_step': args.window_step,
            'threshold': float(threshold),
            'high_precision_threshold': float(high_precision_threshold),
            'quantile_threshold': args.quantile_threshold,
            'total_samples': len(predictions['targets']),
            'positive_rate': float(np.mean(predictions['targets'])),
            'has_duration_data': has_duration,
            'has_asset_analysis': asset_df is not None,
            'has_market_analysis': market_corr is not None
        }
        
        # Add mean metrics
        for col in ['AUC', 'Precision', 'High_Precision', 'Avg_Precision']:
            if col in result_df:
                summary[f'mean_{col.lower()}'] = float(result_df[col].mean())
        
        # Add duration metrics if available
        if has_duration:
            for col in ['Avg_Duration', 'Win_Duration', 'Loss_Duration']:
                if col in result_df:
                    summary[f'mean_{col.lower()}'] = float(result_df[col].mean())
        
        # Convert NumPy types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
                
        # Save summary
        with open(output_dir / 'analysis_summary.json', 'w') as f:
            json.dump(convert_numpy_types(summary), f, indent=4)
        
        print(f"Analysis complete. Results saved to {output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    main()