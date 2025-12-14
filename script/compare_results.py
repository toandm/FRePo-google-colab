"""
Compare results from different dataset distillation methods.

This script scans training logs and compares performance metrics across
different distillation methods (FRePo, MTT, KIP, DC, DM).

Usage:
    # Compare all methods for a specific configuration
    python -m script.compare_results --base_dir=train_log --dataset=cifar100 --config="step3K_num100"

    # Output to specific file
    python -m script.compare_results --base_dir=train_log --dataset=cifar100 --output=results.csv

    # Use TensorBoard parsing (more detailed)
    python -m script.compare_results --base_dir=train_log --use_tensorboard=True
"""

import os
import sys
import glob
import re
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import fire

try:
    from tensorboard.backend.event_processing import event_accumulator
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")


def find_experiment_dirs(base_dir: str, dataset: str, config_pattern: str = "*") -> Dict[str, List[str]]:
    """
    Find all experiment directories for each method.

    Args:
        base_dir: Base directory containing train_log
        dataset: Dataset name (e.g., 'cifar100')
        config_pattern: Pattern for configuration (e.g., 'step3K_num100')

    Returns:
        Dictionary mapping method names to list of experiment directories
    """
    methods = {}

    # Pattern: {base_dir}/{dataset}/{config}/{method}_*/seed*
    search_pattern = os.path.join(base_dir, dataset, config_pattern, "*", "seed*")

    for exp_dir in glob.glob(search_pattern):
        # Extract method name from path
        # Example: train_log/cifar100/step3K_num100/mtt_conv_w128_d3_batch_llTrue/seed0
        parts = exp_dir.split(os.sep)
        method_arch = parts[-2]  # e.g., "mtt_conv_w128_d3_batch_llTrue"

        # Extract method name (first part before _)
        method_match = re.match(r'^([a-zA-Z]+)_', method_arch)
        if method_match:
            method = method_match.group(1)
            if method not in methods:
                methods[method] = []
            methods[method].append(exp_dir)

    return methods


def parse_tensorboard_events(logdir: str) -> Optional[Dict[str, float]]:
    """
    Parse TensorBoard event files to extract metrics.

    Args:
        logdir: Directory containing TensorBoard event files

    Returns:
        Dictionary with metrics or None if parsing fails
    """
    if not HAS_TENSORBOARD:
        return None

    try:
        ea = event_accumulator.EventAccumulator(logdir)
        ea.Reload()

        results = {}

        # Extract final evaluation accuracy
        if 'eval/accuracy_mean' in ea.Tags()['scalars']:
            acc_mean_events = ea.Scalars('eval/accuracy_mean')
            if acc_mean_events:
                results['accuracy_mean'] = acc_mean_events[-1].value
                results['final_step'] = acc_mean_events[-1].step

        if 'eval/accuracy_std' in ea.Tags()['scalars']:
            acc_std_events = ea.Scalars('eval/accuracy_std')
            if acc_std_events:
                results['accuracy_std'] = acc_std_events[-1].value

        # Find best accuracy across all steps
        if 'eval/accuracy_mean' in ea.Tags()['scalars']:
            all_acc = ea.Scalars('eval/accuracy_mean')
            best_acc = max(all_acc, key=lambda x: x.value)
            results['best_accuracy'] = best_acc.value
            results['best_step'] = best_acc.step

        return results if results else None

    except Exception as e:
        print(f"Warning: Failed to parse TensorBoard events in {logdir}: {e}")
        return None


def parse_log_file(logdir: str) -> Optional[Dict[str, float]]:
    """
    Parse text log files to extract final accuracy.

    Args:
        logdir: Directory containing log files

    Returns:
        Dictionary with metrics or None if parsing fails
    """
    # Look for common log file patterns
    log_patterns = ['*.log', '*.txt', 'training.log', 'output.log']

    for pattern in log_patterns:
        log_files = glob.glob(os.path.join(logdir, pattern))
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    content = f.read()

                # Search for accuracy patterns
                # Example: "Best accuracy: 45.23" or "Accuracy: 45.23 ± 2.1"
                best_acc_match = re.search(r'[Bb]est accuracy:?\s*([\d.]+)', content)
                acc_pattern = re.search(r'[Aa]ccuracy:?\s*([\d.]+)\s*[±]\s*([\d.]+)', content)

                results = {}

                if best_acc_match:
                    results['best_accuracy'] = float(best_acc_match.group(1))

                if acc_pattern:
                    results['accuracy_mean'] = float(acc_pattern.group(1))
                    results['accuracy_std'] = float(acc_pattern.group(2))

                if results:
                    return results

            except Exception as e:
                continue

    return None


def extract_metrics(exp_dir: str, use_tensorboard: bool = True) -> Optional[Dict[str, float]]:
    """
    Extract metrics from an experiment directory.

    Args:
        exp_dir: Experiment directory
        use_tensorboard: Whether to try TensorBoard parsing first

    Returns:
        Dictionary with metrics or None
    """
    # Try TensorBoard first if available
    if use_tensorboard and HAS_TENSORBOARD:
        metrics = parse_tensorboard_events(exp_dir)
        if metrics:
            return metrics

    # Fallback to log file parsing
    metrics = parse_log_file(exp_dir)
    return metrics


def aggregate_results(exp_dirs: List[str], use_tensorboard: bool = True) -> Dict[str, float]:
    """
    Aggregate results from multiple experiment runs (different seeds).

    Args:
        exp_dirs: List of experiment directories
        use_tensorboard: Whether to use TensorBoard parsing

    Returns:
        Dictionary with aggregated metrics
    """
    import numpy as np

    all_metrics = []
    for exp_dir in exp_dirs:
        metrics = extract_metrics(exp_dir, use_tensorboard)
        if metrics:
            all_metrics.append(metrics)

    if not all_metrics:
        return {
            'accuracy_mean': 0.0,
            'accuracy_std': 0.0,
            'num_runs': 0
        }

    # Aggregate across seeds
    accuracies = [m.get('accuracy_mean', m.get('best_accuracy', 0.0)) for m in all_metrics]

    result = {
        'accuracy_mean': np.mean(accuracies),
        'accuracy_std': np.std(accuracies),
        'num_runs': len(all_metrics)
    }

    # Add best step if available
    if 'best_step' in all_metrics[0]:
        result['best_step'] = np.mean([m.get('best_step', 0) for m in all_metrics])

    return result


def compare_methods(
    base_dir: str = 'train_log',
    dataset: str = 'cifar100',
    config: str = '*',
    use_tensorboard: bool = True,
    output: Optional[str] = None,
    output_format: str = 'both'
) -> Dict[str, Dict[str, float]]:
    """
    Compare all distillation methods.

    Args:
        base_dir: Base directory containing training logs
        dataset: Dataset name
        config: Configuration pattern (e.g., 'step3K_num100' or '*')
        use_tensorboard: Use TensorBoard parsing if available
        output: Output file path (without extension)
        output_format: 'csv', 'markdown', or 'both'

    Returns:
        Dictionary mapping method names to aggregated metrics
    """
    print(f"Scanning experiments in {base_dir}/{dataset}/{config}...")

    # Find all experiment directories
    method_dirs = find_experiment_dirs(base_dir, dataset, config)

    if not method_dirs:
        print(f"No experiments found in {base_dir}/{dataset}/{config}")
        return {}

    print(f"Found methods: {list(method_dirs.keys())}")

    # Aggregate results for each method
    results = {}
    for method, exp_dirs in method_dirs.items():
        print(f"\nProcessing {method}: {len(exp_dirs)} runs")
        results[method] = aggregate_results(exp_dirs, use_tensorboard)

    # Print results
    print("\n" + "="*70)
    print(f"Results Comparison for {dataset.upper()} - {config}")
    print("="*70)
    print(f"\n{'Method':<10} {'Accuracy (%)':<20} {'Runs':<10}")
    print("-"*70)

    # Sort by accuracy
    sorted_methods = sorted(results.items(), key=lambda x: x[1]['accuracy_mean'], reverse=True)

    for method, metrics in sorted_methods:
        acc_mean = metrics['accuracy_mean']
        acc_std = metrics['accuracy_std']
        num_runs = metrics['num_runs']
        print(f"{method.upper():<10} {acc_mean:>6.2f} ± {acc_std:<6.2f}    {num_runs:<10}")

    # Save to file
    if output:
        if output_format in ['csv', 'both']:
            save_csv(results, f"{output}.csv", dataset, config)
        if output_format in ['markdown', 'both']:
            save_markdown(results, f"{output}.md", dataset, config)

    return results


def save_csv(results: Dict[str, Dict[str, float]], filepath: str, dataset: str, config: str):
    """Save results to CSV file."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'config', 'method', 'accuracy_mean', 'accuracy_std', 'num_runs', 'best_step'])

        for method, metrics in sorted(results.items(), key=lambda x: x[1]['accuracy_mean'], reverse=True):
            writer.writerow([
                dataset,
                config,
                method,
                f"{metrics['accuracy_mean']:.4f}",
                f"{metrics['accuracy_std']:.4f}",
                metrics['num_runs'],
                int(metrics.get('best_step', 0))
            ])

    print(f"\nResults saved to {filepath}")


def save_markdown(results: Dict[str, Dict[str, float]], filepath: str, dataset: str, config: str):
    """Save results to Markdown file."""
    with open(filepath, 'w') as f:
        f.write(f"# Results Comparison: {dataset.upper()} - {config}\n\n")
        f.write("| Method | Accuracy (%) | Runs | Best Step |\n")
        f.write("|--------|--------------|------|----------|\n")

        for method, metrics in sorted(results.items(), key=lambda x: x[1]['accuracy_mean'], reverse=True):
            acc_mean = metrics['accuracy_mean']
            acc_std = metrics['accuracy_std']
            num_runs = metrics['num_runs']
            best_step = int(metrics.get('best_step', 0))

            f.write(f"| {method.upper():<6} | {acc_mean:>6.2f} ± {acc_std:<5.2f} | {num_runs:<4} | {best_step:<9} |\n")

    print(f"Results saved to {filepath}")


def main(
    base_dir: str = 'train_log',
    dataset: str = 'cifar100',
    config: str = '*',
    use_tensorboard: bool = True,
    output: Optional[str] = None,
    output_format: str = 'both'
):
    """
    Main entry point for comparing distillation method results.

    Args:
        base_dir: Base directory containing training logs (default: 'train_log')
        dataset: Dataset name (default: 'cifar100')
        config: Configuration pattern (default: '*' for all)
        use_tensorboard: Use TensorBoard parsing if available (default: True)
        output: Output file path without extension (default: None, no file output)
        output_format: Output format - 'csv', 'markdown', or 'both' (default: 'both')
    """
    if not HAS_TENSORBOARD and use_tensorboard:
        print("TensorBoard not available. Falling back to log file parsing.")
        print("Install TensorBoard with: pip install tensorboard\n")

    results = compare_methods(
        base_dir=base_dir,
        dataset=dataset,
        config=config,
        use_tensorboard=use_tensorboard,
        output=output,
        output_format=output_format
    )

    if not results:
        print("\nNo results found. Make sure experiments have been run and logs exist.")
        return 1

    return 0


if __name__ == '__main__':
    fire.Fire(main)
