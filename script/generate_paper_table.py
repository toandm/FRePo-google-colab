"""
Generate comparison tables in paper format (LaTeX, Markdown, CSV).

This script scans training logs and generates publication-quality comparison
tables similar to Table 1 in the FRePo paper.

Usage:
    # Generate table for all datasets
    python -m script.generate_paper_table --base_dir=train_log --output_dir=results/tables

    # Generate table for specific datasets
    python -m script.generate_paper_table --base_dir=train_log --datasets=cifar10,cifar100

    # Generate only specific formats
    python -m script.generate_paper_table --base_dir=train_log --formats=latex,markdown
"""

import os
import glob
import re
from collections import defaultdict
from typing import Dict, List, Optional
import fire
import numpy as np

try:
    from tensorboard.backend.event_processing import event_accumulator
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")


def parse_experiment_path(path: str) -> Optional[Dict[str, any]]:
    """
    Extract dataset, ipc, method, seed from experiment path.

    Example paths:
        train_log/cifar10/step0K_num10/mtt_conv_w128_d3_batch_llTrue/seed0
        train_log/mnist/step3K_num1/frepo_conv_w128/seed0

    Returns:
        Dict with keys: dataset, ipc, method, seed, arch
        Or None if path doesn't match expected format
    """
    # Pattern: base_dir/dataset/step*_num{ipc}/method_arch_*/seed{seed}
    pattern = r'([^/]+)/step\d+K_num(\d+)/([a-zA-Z]+)_([^/]+)/seed(\d+)$'
    match = re.search(pattern, path)

    if match:
        dataset, ipc, method, arch, seed = match.groups()
        return {
            'dataset': dataset.lower(),
            'ipc': int(ipc),
            'method': method.lower(),
            'arch': arch,
            'seed': int(seed),
            'path': path
        }
    return None


def extract_final_accuracy_from_json(logdir: str) -> Optional[float]:
    """
    Extract final test accuracy from JSON metrics file.

    Args:
        logdir: Directory containing metrics.json

    Returns:
        Final accuracy value or None if not found
    """
    import json

    metrics_file = os.path.join(logdir, 'metrics.json')

    if not os.path.exists(metrics_file):
        return None

    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        # Find eval metrics
        eval_metrics = [m for m in metrics if 'eval/accuracy_mean' in m or 'eval/step_acc_mean' in m]

        if not eval_metrics:
            return None

        # Get final accuracy
        final_metric = eval_metrics[-1]
        acc_key = 'eval/accuracy_mean' if 'eval/accuracy_mean' in final_metric else 'eval/step_acc_mean'

        return final_metric[acc_key]
    except Exception as e:
        print(f"Warning: Failed to parse JSON metrics in {logdir}: {e}")
        return None


def scan_experiments(base_dir: str, datasets: Optional[List[str]] = None) -> Dict:
    """
    Scan all experiment directories and extract results.

    Args:
        base_dir: Base directory containing training logs
        datasets: Optional list of datasets to include (default: all)

    Returns:
        Nested dict: {dataset: {ipc: {method: {'runs': [acc1, acc2, ...], 'mean': X, 'std': Y}}}}
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'runs': []})))

    # Find all seed directories
    pattern = os.path.join(base_dir, '*', '*', '*', 'seed*')
    seed_dirs = glob.glob(pattern)

    print(f"Found {len(seed_dirs)} experiment directories")

    for seed_dir in seed_dirs:
        # Parse experiment metadata from path
        exp_info = parse_experiment_path(seed_dir)
        if not exp_info:
            continue

        # Filter by dataset if specified
        if datasets and exp_info['dataset'] not in datasets:
            continue

        # Extract accuracy from TensorBoard events
        accuracy = extract_final_accuracy_from_json(seed_dir)
        if accuracy is None:
            print(f"Warning: No accuracy found for {seed_dir}")
            continue

        # Store the run
        dataset = exp_info['dataset']
        ipc = exp_info['ipc']
        method = exp_info['method']

        results[dataset][ipc][method]['runs'].append(accuracy)

    # Compute mean and std for each method
    for dataset in results:
        for ipc in results[dataset]:
            for method in results[dataset][ipc]:
                runs = results[dataset][ipc][method]['runs']
                if runs:
                    results[dataset][ipc][method]['mean'] = np.mean(runs)
                    results[dataset][ipc][method]['std'] = np.std(runs)
                    results[dataset][ipc][method]['num_runs'] = len(runs)

    return dict(results)


def generate_latex_table(results: Dict, methods_order: List[str]) -> str:
    """
    Generate LaTeX table string.

    Args:
        results: Results dictionary from scan_experiments()
        methods_order: Ordered list of methods to display

    Returns:
        LaTeX table string
    """
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Test accuracies of models trained on distilled data}")
    lines.append("\\label{tab:results}")

    # Table header
    header = "l|c|" + "c" * len(methods_order)
    lines.append(f"\\begin{{tabular}}{{{header}}}")
    lines.append("\\toprule")

    method_names = [m.upper() for m in methods_order]
    lines.append(f"Dataset & IPC & {' & '.join(method_names)} \\\\")
    lines.append("\\midrule")

    # Sort datasets
    dataset_order = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'tiny_imagenet', 'cub200']
    datasets = [d for d in dataset_order if d in results]

    for dataset_idx, dataset in enumerate(datasets):
        # Sort IPCs
        ipcs = sorted(results[dataset].keys())

        for ipc_idx, ipc in enumerate(ipcs):
            # Dataset name (only for first IPC)
            if ipc_idx == 0:
                dataset_name = dataset.upper().replace('_', '-')
                lines.append(f"{dataset_name} & {ipc}", end='')
            else:
                lines.append(f" & {ipc}", end='')

            # Method results
            for method in methods_order:
                if method in results[dataset][ipc]:
                    mean = results[dataset][ipc][method]['mean']
                    std = results[dataset][ipc][method]['std']
                    lines.append(f" & ${mean:.1f} \\pm {std:.1f}$", end='')
                else:
                    lines.append(" & —", end='')

            lines.append(" \\\\")

        # Add line between datasets (except last)
        if dataset_idx < len(datasets) - 1:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def generate_markdown_table(results: Dict, methods_order: List[str]) -> str:
    """
    Generate Markdown table string.

    Args:
        results: Results dictionary from scan_experiments()
        methods_order: Ordered list of methods to display

    Returns:
        Markdown table string
    """
    lines = []
    lines.append("# Comparison Results")
    lines.append("")

    # Table header
    method_names = [m.upper() for m in methods_order]
    lines.append(f"| Dataset | IPC | {' | '.join(method_names)} |")
    lines.append(f"|---------|-----|{'----|' * len(methods_order)}")

    # Sort datasets
    dataset_order = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'tiny_imagenet', 'cub200']
    datasets = [d for d in dataset_order if d in results]

    for dataset in datasets:
        # Sort IPCs
        ipcs = sorted(results[dataset].keys())

        for ipc_idx, ipc in enumerate(ipcs):
            # Dataset name (only for first IPC)
            if ipc_idx == 0:
                dataset_name = dataset.upper().replace('_', '-')
                row = [f"**{dataset_name}**", str(ipc)]
            else:
                row = ["", str(ipc)]

            # Method results
            for method in methods_order:
                if method in results[dataset][ipc]:
                    mean = results[dataset][ipc][method]['mean']
                    std = results[dataset][ipc][method]['std']
                    row.append(f"{mean:.1f}±{std:.1f}")
                else:
                    row.append("—")

            lines.append(f"| {' | '.join(row)} |")

    return "\n".join(lines)


def generate_csv(results: Dict, methods_order: List[str]) -> str:
    """
    Generate CSV string.

    Args:
        results: Results dictionary from scan_experiments()
        methods_order: Ordered list of methods to display

    Returns:
        CSV string
    """
    lines = []

    # Header
    method_headers = []
    for method in methods_order:
        method_headers.append(f"{method}_mean")
        method_headers.append(f"{method}_std")
        method_headers.append(f"{method}_runs")
    lines.append(f"dataset,ipc,{','.join(method_headers)}")

    # Sort datasets
    dataset_order = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'tiny_imagenet', 'cub200']
    datasets = [d for d in dataset_order if d in results]

    for dataset in datasets:
        # Sort IPCs
        ipcs = sorted(results[dataset].keys())

        for ipc in ipcs:
            row = [dataset, str(ipc)]

            # Method results
            for method in methods_order:
                if method in results[dataset][ipc]:
                    mean = results[dataset][ipc][method]['mean']
                    std = results[dataset][ipc][method]['std']
                    num_runs = results[dataset][ipc][method]['num_runs']
                    row.extend([f"{mean:.4f}", f"{std:.4f}", str(num_runs)])
                else:
                    row.extend(["", "", "0"])

            lines.append(','.join(row))

    return "\n".join(lines)


def main(
    base_dir: str = 'train_log',
    output_dir: str = 'results/tables',
    datasets: Optional[str] = None,
    formats: str = 'all',
    methods: Optional[str] = None
):
    """
    Generate comparison tables from training logs.

    Args:
        base_dir: Base directory containing training logs (default: 'train_log')
        output_dir: Output directory for tables (default: 'results/tables')
        datasets: Comma-separated list of datasets to include (default: all)
        formats: Comma-separated list of formats: latex, markdown, csv, or 'all' (default: 'all')
        methods: Comma-separated list of methods in order (default: auto-detect)
    """
    # Parse datasets filter
    dataset_list = None if datasets is None else [d.strip().lower() for d in datasets.split(',')]

    # Parse formats
    if formats == 'all':
        format_list = ['latex', 'markdown', 'csv']
    elif isinstance(formats, (list, tuple)):
        # Fire sometimes passes comma-separated values as tuples
        format_list = [f.strip().lower() for f in formats]
    else:
        # String format
        format_list = [f.strip().lower() for f in formats.split(',')]

    # Scan experiments
    print(f"Scanning experiments in {base_dir}...")
    results = scan_experiments(base_dir, dataset_list)

    if not results:
        print("No results found!")
        return 1

    # Determine method order
    if methods:
        methods_order = [m.strip().lower() for m in methods.split(',')]
    else:
        # Auto-detect methods from results
        all_methods = set()
        for dataset in results.values():
            for ipc in dataset.values():
                all_methods.update(ipc.keys())

        # Default order
        preferred_order = ['dsa', 'dm', 'kip', 'mtt', 'dc', 'frepo']
        methods_order = [m for m in preferred_order if m in all_methods]
        # Add any remaining methods
        methods_order.extend(sorted(all_methods - set(methods_order)))

    print(f"Methods found: {methods_order}")
    print(f"Datasets found: {list(results.keys())}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate and save tables
    if 'latex' in format_list:
        latex_table = generate_latex_table(results, methods_order)
        latex_path = os.path.join(output_dir, 'comparison_table.tex')
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        print(f"\nLaTeX table saved to: {latex_path}")

    if 'markdown' in format_list:
        markdown_table = generate_markdown_table(results, methods_order)
        markdown_path = os.path.join(output_dir, 'comparison_table.md')
        with open(markdown_path, 'w') as f:
            f.write(markdown_table)
        print(f"Markdown table saved to: {markdown_path}")

        # Also print to console
        print("\n" + "="*70)
        print(markdown_table)
        print("="*70)

    if 'csv' in format_list:
        csv_data = generate_csv(results, methods_order)
        csv_path = os.path.join(output_dir, 'comparison_table.csv')
        with open(csv_path, 'w') as f:
            f.write(csv_data)
        print(f"CSV data saved to: {csv_path}")

    return 0


if __name__ == '__main__':
    fire.Fire(main)
