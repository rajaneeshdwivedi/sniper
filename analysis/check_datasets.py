import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json
from collections import defaultdict

def load_datasets(dataset_dir):
    """Load all datasets and their metadata"""
    dataset_dir = Path(dataset_dir)
    
    # Dictionary to store datasets
    datasets = {}
    
    # Load train, val, test
    for split in ['train', 'val', 'test']:
        # Load pytorch dataset
        try:
            pt_path = dataset_dir / f'{split}.pt'
            datasets[split] = torch.load(pt_path)
            print(f"Loaded {split} dataset from {pt_path}")
        except Exception as e:
            print(f"Error loading {split} dataset: {e}")
            continue
        
        # Load metadata
        try:
            meta_path = dataset_dir / f'{split}_metadata.csv'
            datasets[f"{split}_metadata"] = pd.read_csv(meta_path)
            print(f"Loaded {split} metadata from {meta_path}")
        except Exception as e:
            print(f"Error loading {split} metadata: {e}")
    
    return datasets

def analyze_class_distributions(datasets):
    """Analyze class distributions across datasets"""
    print("\n=== Class Distribution Analysis ===")
    
    results = {}
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, split in enumerate(['train', 'val', 'test']):
        if split not in datasets:
            continue
            
        # Get labels
        if isinstance(datasets[split]['y'], torch.Tensor):
            y = datasets[split]['y'].numpy()
        else:
            y = datasets[split]['y']
            
        # Count class distribution
        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique, counts))
        
        # Calculate positive ratio
        total = sum(counts)
        pos_ratio = class_dist.get(1, 0) / total if total > 0 else 0
        
        # Store results
        results[split] = {
            'class_distribution': class_dist,
            'positive_ratio': pos_ratio,
            'total_samples': total
        }
        
        # Plot
        sns.barplot(x=unique, y=counts, ax=ax[i])
        ax[i].set_title(f"{split.capitalize()} - Class Distribution")
        ax[i].set_xlabel("Class")
        ax[i].set_ylabel("Count")
        ax[i].text(0.5, 0.9, f"Positive Ratio: {pos_ratio:.2%}", 
                  transform=ax[i].transAxes, ha='center')
    
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    
    # Print results
    for split, stats in results.items():
        print(f"\n{split.upper()}:")
        print(f"Total samples: {stats['total_samples']:,}")
        print(f"Class distribution: {stats['class_distribution']}")
        print(f"Positive ratio: {stats['positive_ratio']:.2%}")
    
    return results

def analyze_metadata_consistency(datasets):
    """Check metadata consistency across splits"""
    print("\n=== Metadata Consistency Analysis ===")
    
    splits = ['train_metadata', 'val_metadata', 'test_metadata']
    available_splits = [s for s in splits if s in datasets]
    
    if len(available_splits) < 2:
        print("Not enough metadata files available for comparison")
        return
    
    # Check columns
    print("\nColumn Comparison:")
    for split in available_splits:
        print(f"{split}: {list(datasets[split].columns)}")
    
    # Common columns across all splits
    common_columns = set(datasets[available_splits[0]].columns)
    for split in available_splits[1:]:
        common_columns = common_columns.intersection(set(datasets[split].columns))
    
    print(f"\nCommon columns across all splits: {list(common_columns)}")
    
    # Check for missing values
    print("\nMissing Values Analysis:")
    for split in available_splits:
        missing = datasets[split].isna().sum()
        print(f"\n{split}:")
        print(missing[missing > 0] if any(missing > 0) else "No missing values")
    
    # Check data types
    print("\nData Types Comparison:")
    for split in available_splits:
        print(f"\n{split}:")
        print(datasets[split].dtypes)
    
    # Check index alignment
    print("\nIndex Ranges:")
    for split in available_splits:
        print(f"{split}: {datasets[split].index.min()} to {datasets[split].index.max()}")

def analyze_asset_distribution(datasets):
    """Analyze distribution of assets across splits"""
    print("\n=== Asset Distribution Analysis ===")
    
    splits = ['train_metadata', 'val_metadata', 'test_metadata']
    available_splits = [s for s in splits if s in datasets]
    
    if len(available_splits) < 2:
        print("Not enough metadata files available for comparison")
        return
    
    # Check if 'code' column exists in all splits
    if not all('code' in datasets[split].columns for split in available_splits):
        print("'code' column not found in all metadata files")
        return
    
    # Get asset counts for each split
    asset_counts = {}
    all_assets = set()
    
    for split in available_splits:
        counts = datasets[split]['code'].value_counts().to_dict()
        asset_counts[split] = counts
        all_assets.update(counts.keys())
    
    # Create comparison table
    comparison = pd.DataFrame(index=sorted(all_assets))
    
    for split in available_splits:
        comparison[split.replace('_metadata', '')] = comparison.index.map(
            lambda x: asset_counts[split].get(x, 0))
    
    # Add percentage columns
    for split in available_splits:
        split_name = split.replace('_metadata', '')
        total = comparison[split_name].sum()
        comparison[f"{split_name}_pct"] = comparison[split_name] / total if total > 0 else 0
    
    print("\nAsset Distribution:")
    print(comparison)
    
    # Plot asset distribution
    plt.figure(figsize=(15, 8))
    
    # Convert to long format for seaborn
    long_df = pd.melt(
        comparison.reset_index(),
        id_vars='index',
        value_vars=[s.replace('_metadata', '') for s in available_splits],
        var_name='Split',
        value_name='Count'
    )
    long_df.rename(columns={'index': 'Asset'}, inplace=True)
    
    # Plot
    sns.barplot(x='Asset', y='Count', hue='Split', data=long_df)
    plt.title('Asset Distribution Across Splits')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('asset_distribution.png')
    
    return comparison

def analyze_exit_types(datasets):
    """Analyze exit type distribution (target hit, stop hit, timeout)"""
    print("\n=== Exit Type Analysis ===")
    
    splits = ['train_metadata', 'val_metadata', 'test_metadata']
    available_splits = [s for s in splits if s in datasets]
    
    if len(available_splits) < 2:
        print("Not enough metadata files available for comparison")
        return
    
    # Check if 'exit_type' column exists in all splits
    if not all('exit_type' in datasets[split].columns for split in available_splits):
        print("'exit_type' column not found in all metadata files")
        return
    
    # Generate exit type distribution for each split
    exit_counts = {}
    exit_pcts = {}
    
    # Exit type meanings: 1 = target hit, 2 = stop hit, 3 = timeout
    exit_names = {1: 'Target Hit', 2: 'Stop Hit', 3: 'Timeout', 0: 'Unknown'}
    
    for split in available_splits:
        counts = datasets[split]['exit_type'].value_counts().to_dict()
        total = sum(counts.values())
        
        # Ensure all exit types are represented
        for exit_type in exit_names.keys():
            if exit_type not in counts:
                counts[exit_type] = 0
        
        exit_counts[split] = counts
        exit_pcts[split] = {k: v/total for k, v in counts.items() if total > 0}
    
    # Print results
    for split in available_splits:
        split_name = split.replace('_metadata', '')
        print(f"\n{split_name.upper()} Exit Types:")
        for exit_type, count in sorted(exit_counts[split].items()):
            if count > 0:
                pct = exit_pcts[split][exit_type]
                print(f"  {exit_names.get(exit_type, f'Type {exit_type}')}: {count:,} ({pct:.2%})")
    
    # Plot exit type distribution
    plt.figure(figsize=(14, 7))
    
    # Create DataFrame for plotting
    plot_data = []
    for split in available_splits:
        split_name = split.replace('_metadata', '')
        for exit_type, count in exit_counts[split].items():
            if exit_type in exit_names:  # Only include known exit types
                plot_data.append({
                    'Split': split_name,
                    'Exit Type': exit_names[exit_type],
                    'Count': count,
                    'Percentage': exit_pcts[split][exit_type] * 100
                })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create two subplots - counts and percentages
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot counts
    sns.barplot(x='Exit Type', y='Count', hue='Split', data=plot_df, ax=ax1)
    ax1.set_title('Exit Type Counts')
    ax1.set_ylabel('Count')
    
    # Plot percentages
    sns.barplot(x='Exit Type', y='Percentage', hue='Split', data=plot_df, ax=ax2)
    ax2.set_title('Exit Type Percentages')
    ax2.set_ylabel('Percentage (%)')
    
    plt.tight_layout()
    plt.savefig('exit_types_distribution.png')
    
    return {'counts': exit_counts, 'percentages': exit_pcts}

def analyze_temporal_distribution(datasets):
    """Analyze temporal distribution of samples"""
    print("\n=== Temporal Distribution Analysis ===")
    
    splits = ['train_metadata', 'val_metadata', 'test_metadata']
    available_splits = [s for s in splits if s in datasets]
    
    if len(available_splits) < 2:
        print("Not enough metadata files available for comparison")
        return
    
    # Check if 'closeTime' column exists in all splits
    if not all('closeTime' in datasets[split].columns for split in available_splits):
        print("'closeTime' column not found in all metadata files")
        return
    
    # Convert closeTime to datetime if needed
    for split in available_splits:
        if pd.api.types.is_numeric_dtype(datasets[split]['closeTime']):
            # If closeTime is numeric (epoch timestamp), convert to datetime
            datasets[split]['datetime'] = pd.to_datetime(datasets[split]['closeTime'], unit='s')
        else:
            # Try to parse as datetime if it's already a string
            datasets[split]['datetime'] = pd.to_datetime(datasets[split]['closeTime'])
    
    # Get time ranges for each split
    time_ranges = {}
    for split in available_splits:
        split_name = split.replace('_metadata', '')
        time_ranges[split_name] = {
            'min': datasets[split]['datetime'].min(),
            'max': datasets[split]['datetime'].max(),
            'duration_days': (datasets[split]['datetime'].max() - datasets[split]['datetime'].min()).days
        }
    
    # Print results
    print("\nTime Ranges:")
    for split, range_info in time_ranges.items():
        print(f"\n{split.upper()}:")
        print(f"Start: {range_info['min']}")
        print(f"End: {range_info['max']}")
        print(f"Duration: {range_info['duration_days']} days")
    
    # Plot timeline
    plt.figure(figsize=(15, 6))
    
    # For each split, create a horizontal line on the timeline
    colors = {'train': 'blue', 'val': 'green', 'test': 'red'}
    y_positions = {'train': 3, 'val': 2, 'test': 1}
    
    for split, range_info in time_ranges.items():
        start = range_info['min']
        end = range_info['max']
        y_pos = y_positions.get(split, 0)
        
        plt.plot([start, end], [y_pos, y_pos], linewidth=10, 
                 solid_capstyle='butt', color=colors.get(split, 'gray'))
        
        # Add labels
        plt.text(start, y_pos + 0.1, start.strftime('%Y-%m-%d'), 
                 verticalalignment='bottom', horizontalalignment='left', rotation=45)
        plt.text(end, y_pos + 0.1, end.strftime('%Y-%m-%d'), 
                 verticalalignment='bottom', horizontalalignment='right', rotation=45)
        
    # Customize plot
    plt.yticks(list(y_positions.values()), list(y_positions.keys()))
    plt.xlabel('Date')
    plt.title('Dataset Time Ranges')
    plt.grid(True, axis='x', alpha=0.3)
    
    # Format x-axis as dates
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('temporal_distribution.png')
    
    return time_ranges

def analyze_feature_distributions(datasets):
    """Analyze feature distributions across splits"""
    print("\n=== Feature Distribution Analysis ===")
    
    splits = ['train', 'val', 'test']
    available_splits = [s for s in splits if s in datasets]
    
    if len(available_splits) < 2:
        print("Not enough dataset files available for comparison")
        return
    
    # Check feature shapes
    print("\nFeature Tensor Shapes:")
    for split in available_splits:
        features = datasets[split]['unified_features']
        if isinstance(features, torch.Tensor):
            shape = features.shape
        else:
            shape = features.shape if hasattr(features, 'shape') else "Unknown"
        print(f"{split}: {shape}")
    
    # Calculate feature statistics
    feature_stats = {}
    for split in available_splits:
        features = datasets[split]['unified_features']
        if isinstance(features, torch.Tensor):
            features = features.numpy()
        
        # For 3D tensors (batch, sequence, features), flatten the sequence dimension
        if len(features.shape) == 3:
            # Take the last timestep for each feature
            features = features[:, -1, :]
        
        # Calculate statistics
        feature_stats[split] = {
            'mean': np.mean(features, axis=0),
            'std': np.std(features, axis=0),
            'min': np.min(features, axis=0),
            'max': np.max(features, axis=0),
            'shape': features.shape
        }
    
    # Plot feature distributions
    # For simplicity, we'll plot histograms of a few random features
    n_features = min(5, feature_stats[available_splits[0]]['shape'][1])
    feature_indices = np.random.choice(feature_stats[available_splits[0]]['shape'][1], 
                                        size=n_features, replace=False)
    
    fig, axes = plt.subplots(n_features, 1, figsize=(15, 4*n_features))
    
    for i, feature_idx in enumerate(feature_indices):
        ax = axes[i] if n_features > 1 else axes
        
        for split in available_splits:
            features = datasets[split]['unified_features']
            if isinstance(features, torch.Tensor):
                features = features.numpy()
            
            # Extract feature values
            if len(features.shape) == 3:
                feature_values = features[:, -1, feature_idx]
            else:
                feature_values = features[:, feature_idx]
            
            # Plot histogram
            sns.histplot(feature_values, bins=50, alpha=0.5, label=split, ax=ax)
        
        ax.set_title(f'Feature {feature_idx} Distribution')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    
    # Compare means and stds across splits
    print("\nFeature Statistics Comparison:")
    
    # Find common reference split (usually train)
    ref_split = available_splits[0]
    
    for split in available_splits[1:]:
        mean_diff = np.mean(np.abs(feature_stats[split]['mean'] - feature_stats[ref_split]['mean']))
        std_diff = np.mean(np.abs(feature_stats[split]['std'] - feature_stats[ref_split]['std']))
        
        print(f"\n{split} vs {ref_split}:")
        print(f"Mean absolute difference in feature means: {mean_diff:.6f}")
        print(f"Mean absolute difference in feature stds: {std_diff:.6f}")
    
    return feature_stats

def analyze_feature_shapes(datasets):
    """Analyze and validate feature shapes across splits"""
    print("\n=== Feature Shape Analysis ===")
    
    splits = ['train', 'val', 'test']
    available_splits = [s for s in splits if s in datasets]
    
    if len(available_splits) < 2:
        print("Not enough dataset files available for comparison")
        return
    
    # Check feature shapes
    shapes = {}
    for split in available_splits:
        features = datasets[split]['unified_features']
        if isinstance(features, torch.Tensor):
            shapes[split] = tuple(features.shape)
        else:
            shapes[split] = tuple(features.shape) if hasattr(features, 'shape') else None
    
    print("Feature shapes:")
    for split, shape in shapes.items():
        print(f"{split}: {shape}")
    
    # Check if shapes are consistent
    is_consistent = len(set(str(shape) for shape in shapes.values())) == 1
    print(f"\nFeature shapes are {'consistent' if is_consistent else 'inconsistent'} across splits")
    
    # Check for NaN/Inf values
    print("\nChecking for NaN/Inf values:")
    for split in available_splits:
        features = datasets[split]['unified_features']
        if isinstance(features, torch.Tensor):
            features = features.numpy()
        
        n_nan = np.isnan(features).sum()
        n_inf = np.isinf(features).sum()
        
        print(f"{split}:")
        print(f"  NaN values: {n_nan}")
        print(f"  Inf values: {n_inf}")
    
    return shapes

def check_index_alignment(datasets):
    """Check index alignment between features and metadata"""
    print("\n=== Index Alignment Check ===")
    
    splits = ['train', 'val', 'test']
    available_splits = [split for split in splits if split in datasets and f"{split}_metadata" in datasets]
    
    if not available_splits:
        print("Metadata is missing for all splits")
        return
    
    for split in available_splits:
        features = datasets[split]['unified_features']
        metadata = datasets[f"{split}_metadata"]
        
        if isinstance(features, torch.Tensor):
            n_samples = features.shape[0]
        else:
            n_samples = features.shape[0] if hasattr(features, 'shape') else 0
        
        n_metadata = len(metadata)
        
        print(f"\n{split.upper()}:")
        print(f"Number of feature samples: {n_samples}")
        print(f"Number of metadata entries: {n_metadata}")
        
        if n_samples == n_metadata:
            print("✓ Features and metadata are aligned")
        else:
            print("✗ Features and metadata have different lengths!")
            
            # Check the sorting of metadata
            if 'closeTime' in metadata.columns:
                is_sorted = metadata['closeTime'].is_monotonic_increasing
                print(f"  Metadata is {'sorted' if is_sorted else 'not sorted'} by closeTime")
            
            # Simulate what might happen during index sorting
            print("\nSimulating potential index errors:")
            
            # Example: what happens if we sort by confidence scores
            if n_samples > 0 and n_metadata > 0:
                mock_confidence = np.random.random(n_samples)
                sorted_indices = np.argsort(mock_confidence)[::-1]
                
                max_idx = min(n_samples, n_metadata) - 1
                print(f"  Max valid index: {max_idx}")
                print(f"  Max sorted index: {sorted_indices.max()}")
                
                if sorted_indices.max() >= n_metadata:
                    print("  ✗ Index out of bounds would occur during sorting!")
                else:
                    print("  ✓ Indices would remain in bounds during sorting")

def analyze_output_distribution(datasets):
    """Analyze prediction output distribution"""
    print("\n=== Model Output Distribution Analysis ===")
    
    # This is a placeholder - in a real scenario, you would run the model on each dataset
    # and analyze the distributions of the outputs
    
    print("Note: This analysis requires running the model on each dataset")
    print("Implement this function based on your model inference code")

def main(dataset_dir):
    # Load datasets
    datasets = load_datasets(dataset_dir)
    
    if not datasets:
        print("No datasets found or could be loaded")
        return
    
    # Run analyses
    analyze_class_distributions(datasets)
    analyze_metadata_consistency(datasets)
    analyze_asset_distribution(datasets)
    analyze_exit_types(datasets)
    analyze_temporal_distribution(datasets)
    analyze_feature_shapes(datasets)
    analyze_feature_distributions(datasets)
    check_index_alignment(datasets)
    
    print("\nAnalysis complete. Check the generated plots and output for insights.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze ML datasets")
    parser.add_argument("--dataset_dir", type=str, required=True, 
                        help="Directory containing dataset files")
    args = parser.parse_args()
    
    main(args.dataset_dir)    