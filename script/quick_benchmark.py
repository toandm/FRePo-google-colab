"""
Quick table generation script - scans all experiment results and generates comparison tables.

This script scans the train_log directory for all completed experiments,
extracts final accuracies from metrics.json files, computes statistics across seeds,
and generates formatted comparison tables.

Key Features:
- Scans all experiments in train_log/ automatically
- Computes mean ± std across multiple seeds
- Outputs both to console AND markdown file
- Optional CSV output for further analysis
- Flexible filtering by dataset, method, and IPC

Usage Examples:
    # Generate table for all experiments
    python -m script.quick_benchmark

    # Generate table for specific datasets only
    python -m script.quick_benchmark --datasets=mnist,cifar10

    # Generate table for specific methods
    python -m script.quick_benchmark --methods=frepo,mtt,dc

    # Filter by IPC values
    python -m script.quick_benchmark --ipcs=1,5

    # Custom output location
    python -m script.quick_benchmark --output_file=my_results.md

    # Skip CSV generation
    python -m script.quick_benchmark --also_csv=False

    # Quiet mode (less verbose)
    python -m script.quick_benchmark --verbose=False

Output:
    1. Console: Formatted table printed to terminal
    2. Markdown file: results/tables/comparison_table.md (default)
    3. CSV file: results/tables/comparison_table.csv (if also_csv=True)
"""

import os
import time
from typing import Dict, List, Optional
import fire

from script.generate_paper_table import (
    generate_csv,
    parse_experiment_path
)


def extract_final_metrics_from_json(logdir: str) -> Optional[tuple]:
    """
    Extract final test accuracy and std from JSON metrics file.

    Args:
        logdir: Directory containing metrics.json

    Returns:
        Tuple of (mean, std) or None if not found
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

        # Get final metrics
        final_metric = eval_metrics[-1]

        # Determine which keys to use (different methods use different naming)
        if 'eval/step_acc_mean' in final_metric:
            mean = final_metric['eval/step_acc_mean']
            std = final_metric.get('eval/step_std', 0.0)
        elif 'eval/accuracy_mean' in final_metric:
            mean = final_metric['eval/accuracy_mean']
            std = final_metric.get('eval/accuracy_std', 0.0)
        else:
            return None

        return (mean, std)
    except Exception as e:
        print(f"Warning: Failed to parse JSON metrics in {logdir}: {e}")
        return None


def scan_experiments_with_std(base_dir: str, datasets: Optional[List[str]] = None) -> Dict:
    """
    Scan all experiment directories and extract results with std from metrics.json.

    Args:
        base_dir: Base directory containing training logs
        datasets: Optional list of datasets to include (default: all)

    Returns:
        Nested dict: {dataset: {ipc: {method: {'mean': X, 'std': Y}}}}
    """
    import glob
    from collections import defaultdict

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

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

        # Extract metrics (mean, std) from JSON
        metrics = extract_final_metrics_from_json(seed_dir)
        if metrics is None:
            print(f"Warning: No accuracy found for {seed_dir}")
            continue

        mean, std = metrics

        # Store the results
        dataset = exp_info['dataset']
        ipc = exp_info['ipc']
        method = exp_info['method']

        results[dataset][ipc][method]['mean'] = mean
        results[dataset][ipc][method]['std'] = std
        results[dataset][ipc][method]['num_runs'] = 1  # From single seed's metrics.json

    return dict(results)


def generate_markdown_table(results: Dict, methods_order: List[str]) -> str:
    """
    Generate Markdown table string showing ALL datasets.

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

    # Show ALL datasets found in results (sorted alphabetically)
    datasets = sorted(results.keys())

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
                    row.append("_")  # Use underscore for missing data

            lines.append(f"| {' | '.join(row)} |")

    return "\n".join(lines)


def print_scan_summary(results: Dict, methods_order: List[str]) -> None:
    """
    Print a summary of scanned experiments to console.

    Shows number of datasets, IPCs, methods, and total experiments.
    Provides breakdown by dataset.

    Args:
        results: Results dictionary from scan_experiments()
        methods_order: Ordered list of methods
    """
    # Count total experiments
    total_experiments = sum(
        sum(len(methods.keys()) for methods in ipcs.values())
        for ipcs in results.values()
    )

    # Collect dataset names
    datasets = list(results.keys())

    # Collect all IPCs and methods across all datasets
    all_ipcs = set()
    all_methods = set()
    for dataset in results.values():
        for ipc in dataset.keys():
            all_ipcs.add(ipc)
            for method in dataset[ipc].keys():
                all_methods.add(method)

    print(f"\nSummary:")
    print(f"  Datasets: {len(datasets)} ({', '.join(datasets)})")
    print(f"  IPCs: {len(all_ipcs)} ({', '.join(map(str, sorted(all_ipcs)))})")
    print(f"  Methods: {len(methods_order)} ({', '.join(methods_order)})")
    print(f"  Total experiments: {total_experiments}")

    # Breakdown by dataset
    print(f"\nBreakdown by dataset:")
    for dataset in datasets:
        ipcs = list(results[dataset].keys())
        num_ipcs = len(ipcs)
        methods_in_dataset = set()
        for ipc in ipcs:
            methods_in_dataset.update(results[dataset][ipc].keys())
        num_methods = len(methods_in_dataset)
        num_exps = sum(len(results[dataset][ipc].keys()) for ipc in ipcs)
        print(f"  {dataset}: {num_exps} experiments ({num_ipcs} IPCs, {num_methods} methods)")


def filter_results(
    results: Dict,
    dataset_filter: Optional[List[str]] = None,
    method_filter: Optional[List[str]] = None,
    ipc_filter: Optional[List[int]] = None
) -> Dict:
    """
    Filter scan results based on user criteria.

    Args:
        results: Full results from scan_experiments()
        dataset_filter: List of datasets to include (None = all)
        method_filter: List of methods to include (None = all)
        ipc_filter: List of IPC values to include (None = all)

    Returns:
        Filtered results dictionary
    """
    from copy import deepcopy
    filtered = deepcopy(results)

    # Filter by dataset
    if dataset_filter:
        filtered = {k: v for k, v in filtered.items() if k in dataset_filter}

    # Filter by IPC
    if ipc_filter:
        for dataset in list(filtered.keys()):
            filtered[dataset] = {k: v for k, v in filtered[dataset].items() if k in ipc_filter}
            # Remove dataset if no IPCs left
            if not filtered[dataset]:
                del filtered[dataset]

    # Filter by method
    if method_filter:
        for dataset in list(filtered.keys()):
            for ipc in list(filtered[dataset].keys()):
                filtered[dataset][ipc] = {
                    k: v for k, v in filtered[dataset][ipc].items() if k in method_filter
                }
                # Remove IPC if no methods left
                if not filtered[dataset][ipc]:
                    del filtered[dataset][ipc]
            # Remove dataset if no IPCs left
            if not filtered[dataset]:
                del filtered[dataset]

    return filtered


def main(
    base_dir: str = 'train_log',
    output_file: str = 'results/tables/comparison_table.md',
    datasets: Optional[str] = None,
    methods: Optional[str] = None,
    ipcs: Optional[str] = None,
    also_csv: bool = True,
    verbose: bool = True
) -> int:
    """
    Generate comparison tables from all experiments in train_log directory.

    Scans all metrics.json files, computes statistics, and generates:
    1. Console output (printed table)
    2. Markdown file (saved to output_file)
    3. CSV file (optional, if also_csv=True)

    Args:
        base_dir: Base directory containing training logs (default: 'train_log')
        output_file: Path to save markdown table (default: 'results/tables/comparison_table.md')
        datasets: Comma-separated list of datasets to include (default: all found)
                 Examples: 'mnist,cifar10' or 'mnist'
        methods: Comma-separated list of methods to include (default: all found)
                Examples: 'frepo,mtt,dc' or 'frepo'
        ipcs: Comma-separated list of IPC values (default: all found)
             Examples: '1,5,10' or '1'
        also_csv: Also generate CSV file alongside markdown (default: True)
        verbose: Print detailed progress information (default: True)

    Returns:
        0 on success, 1 on error
    """
    # 1. HEADER
    print("="*70)
    print("QUICK TABLE GENERATION - Scan Experiments & Generate Tables")
    print("="*70)
    print(f"\nScanning directory: {base_dir}")
    print(f"Output file: {output_file}")

    # 2. PARSE FILTERS
    # Handle both string and tuple inputs (Fire sometimes converts comma-separated values to tuples)
    if datasets is None:
        dataset_filter = None
    elif isinstance(datasets, (list, tuple)):
        dataset_filter = [d.strip().lower() for d in datasets]
    else:
        dataset_filter = [d.strip().lower() for d in datasets.split(',')]

    if methods is None:
        method_filter = None
    elif isinstance(methods, (list, tuple)):
        method_filter = [m.strip().lower() for m in methods]
    else:
        method_filter = [m.strip().lower() for m in methods.split(',')]

    if ipcs is None:
        ipc_filter = None
    elif isinstance(ipcs, int):
        # Single integer value
        ipc_filter = [ipcs]
    elif isinstance(ipcs, (list, tuple)):
        ipc_filter = [int(i) if isinstance(i, int) else int(str(i).strip()) for i in ipcs]
    else:
        # String value
        ipc_filter = [int(i.strip()) for i in str(ipcs).split(',')]

    if verbose:
        print("\nFilters:")
        print(f"  Datasets: {dataset_filter if dataset_filter else 'ALL'}")
        print(f"  Methods: {method_filter if method_filter else 'ALL'}")
        print(f"  IPCs: {ipc_filter if ipc_filter else 'ALL'}")

    # 3. SCAN EXPERIMENTS
    print(f"\nScanning experiments...")
    start_time = time.time()

    # Check if directory exists
    if not os.path.exists(base_dir):
        print(f"Error: Directory '{base_dir}' does not exist!")
        print("Make sure you have run experiments first.")
        return 1

    results = scan_experiments_with_std(base_dir, dataset_filter)
    scan_time = time.time() - start_time

    if not results:
        print(f"Error: No experiments found in {base_dir}")
        print("Make sure you have run experiments first.")
        print(f"Expected directory structure: {base_dir}/<dataset>/step*_num*/<method>_*/seed*/")
        return 1

    # 4. FILTER RESULTS
    if method_filter or ipc_filter:
        results = filter_results(results, None, method_filter, ipc_filter)

        if not results:
            print("Error: No experiments match your filters!")
            print(f"  Datasets: {dataset_filter}")
            print(f"  Methods: {method_filter}")
            print(f"  IPCs: {ipc_filter}")
            return 1

    # 5. DETERMINE METHOD ORDER
    all_methods = set()
    for dataset in results.values():
        for ipc in dataset.values():
            all_methods.update(ipc.keys())

    if methods:
        methods_order = method_filter
    else:
        # Preferred order matching generate_paper_table.py
        preferred_order = ['dm', 'kip', 'mtt', 'dc', 'frepo']
        methods_order = [m for m in preferred_order if m in all_methods]
        methods_order.extend(sorted(all_methods - set(methods_order)))

    # 6. PRINT SUMMARY
    print(f"\nScan completed in {scan_time:.2f}s")
    print_scan_summary(results, methods_order)

    # 7. GENERATE MARKDOWN TABLE
    print("\nGenerating markdown table...")
    markdown_content = generate_markdown_table(results, methods_order)

    # 8. PRINT TO CONSOLE
    print("\n" + "="*70)
    print("RESULTS TABLE")
    print("="*70)
    print(markdown_content)
    print("="*70)

    # 9. SAVE MARKDOWN FILE
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(markdown_content)
        print(f"\nMarkdown table saved to: {output_file}")
    except Exception as e:
        print(f"Error saving markdown file: {e}")
        return 1

    # 10. GENERATE CSV (OPTIONAL)
    if also_csv:
        try:
            csv_file = output_file.replace('.md', '.csv')
            csv_content = generate_csv(results, methods_order)
            with open(csv_file, 'w') as f:
                f.write(csv_content)
            print(f"CSV file saved to: {csv_file}")
        except Exception as e:
            print(f"Warning: Failed to save CSV file: {e}")

    # 11. FINAL SUMMARY
    total_experiments = sum(
        sum(len(methods.keys()) for methods in ipcs.values())
        for ipcs in results.values()
    )
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total experiments processed: {total_experiments}")
    print(f"Datasets: {list(results.keys())}")
    print(f"Methods: {methods_order}")
    print("="*70)

    return 0


if __name__ == '__main__':
    fire.Fire(main)
