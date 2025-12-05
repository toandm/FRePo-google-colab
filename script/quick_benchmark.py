"""
Quick benchmark script to run all experiments with minimal config for testing.

This script runs all distillation methods across multiple datasets and IPC settings
with minimal training time to verify the pipeline works correctly.

Usage:
    # Run all experiments with minimal config
    python -m script.quick_benchmark

    # Run only specific datasets
    python -m script.quick_benchmark --datasets=cifar10,mnist

    # Run only specific methods
    python -m script.quick_benchmark --methods=frepo,mtt

    # Skip methods that already have results
    python -m script.quick_benchmark --skip_existing=True
"""

import os
import subprocess
import time
from typing import List, Optional
import fire


# Minimal config for quick testing
QUICK_CONFIG = {
    'num_distill_steps': 100,        # Very few training steps (normally 3000+)
    'steps_per_eval': 50,            # Evaluate more frequently
    'steps_per_log': 10,             # Log more frequently
    'width': 64,                     # Smaller model (normally 128)
    'depth': 2,                      # Shallower model (normally 3)
    'num_eval': 3,                   # Fewer evaluation models (normally 5)
    'num_online_eval_updates': 500,  # Fewer training steps per eval (normally 1000-5000)
}


def get_experiment_configs():
    """
    Define all experiment configurations to run.

    Returns list of tuples: (dataset, ipc, method)
    """
    configs = []

    # MNIST - Fast dataset, good for testing
    for ipc in [1, 10, 50]:
        for method in ['frepo', 'mtt', 'kip', 'dc', 'dm']:
            configs.append(('mnist', ipc, method))

    # Fashion-MNIST - Also fast
    for ipc in [1, 10, 50]:
        for method in ['frepo', 'mtt', 'kip', 'dc', 'dm']:
            configs.append(('fashion_mnist', ipc, method))

    # CIFAR-10 - More realistic
    for ipc in [1, 10, 50]:
        for method in ['frepo', 'mtt', 'kip', 'dc', 'dm']:
            configs.append(('cifar10', ipc, method))

    # CIFAR-100 - Harder dataset
    for ipc in [1, 10, 50]:
        for method in ['frepo', 'mtt', 'kip', 'dc', 'dm']:
            configs.append(('cifar100', ipc, method))

    return configs


def experiment_exists(base_dir: str, dataset: str, ipc: int, method: str, seed: int = 0) -> bool:
    """
    Check if experiment results already exist.

    Args:
        base_dir: Base directory for training logs
        dataset: Dataset name
        ipc: Images per class
        method: Method name
        seed: Random seed

    Returns:
        True if experiment results exist
    """
    # Check for TensorBoard events in expected location
    exp_dir = os.path.join(
        base_dir,
        dataset,
        f'step{QUICK_CONFIG["num_distill_steps"]//1000}K_num{ipc}',
        f'{method}_width{QUICK_CONFIG["width"]}_depth{QUICK_CONFIG["depth"]}',
        f'seed{seed}'
    )

    # Look for event files
    if os.path.exists(exp_dir):
        import glob
        events = glob.glob(os.path.join(exp_dir, 'events.out.tfevents.*'))
        if events:
            return True

    return False


def run_experiment(
    dataset: str,
    ipc: int,
    method: str,
    seed: int = 0,
    base_log_dir: str = 'train_log',
    base_img_dir: str = 'train_img',
    dry_run: bool = False
) -> int:
    """
    Run a single experiment.

    Args:
        dataset: Dataset name
        ipc: Images per class
        method: Method name
        seed: Random seed
        base_log_dir: Base directory for logs
        base_img_dir: Base directory for images
        dry_run: If True, only print command without running

    Returns:
        Exit code (0 = success)
    """
    # Build command
    cmd = [
        'python3', '-m', 'script.distill_unified',
        f'--method={method}',
        f'--dataset_name={dataset}',
        f'--num_prototypes_per_class={ipc}',
        f'--num_distill_steps={QUICK_CONFIG["num_distill_steps"]}',
        f'--steps_per_eval={QUICK_CONFIG["steps_per_eval"]}',
        f'--steps_per_log={QUICK_CONFIG["steps_per_log"]}',
        f'--width={QUICK_CONFIG["width"]}',
        f'--depth={QUICK_CONFIG["depth"]}',
        f'--num_eval={QUICK_CONFIG["num_eval"]}',
        f'--num_online_eval_updates={QUICK_CONFIG["num_online_eval_updates"]}',
        f'--random_seed={seed}',
        f'--train_log={base_log_dir}',
        f'--train_img={base_img_dir}',
        '--save_image=False',  # Don't save images to save time/space
    ]

    print(f"\n{'='*70}")
    print(f"Running: {dataset.upper()} | IPC={ipc} | Method={method.upper()} | Seed={seed}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")

    if dry_run:
        print("[DRY RUN - Not executing]")
        return 0

    # Run experiment
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        print(f"\n✓ Completed in {elapsed:.1f}s")
        return result.returncode
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Failed after {elapsed:.1f}s with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
        raise


def main(
    datasets: Optional[str] = None,
    methods: Optional[str] = None,
    ipcs: Optional[str] = None,
    seeds: str = '0',
    base_log_dir: str = 'train_log',
    base_img_dir: str = 'train_img',
    skip_existing: bool = False,
    dry_run: bool = False,
    stop_on_error: bool = False,
    auto_confirm: bool = False
):
    """
    Run quick benchmark experiments for all combinations.

    Args:
        datasets: Comma-separated list of datasets (default: all)
                 Options: mnist, fashion_mnist, cifar10, cifar100
        methods: Comma-separated list of methods (default: all)
                Options: frepo, mtt, kip, dc, dm
        ipcs: Comma-separated list of IPC values (default: 1,10,50)
        seeds: Comma-separated list of seeds (default: 0)
        base_log_dir: Base directory for training logs (default: train_log)
        base_img_dir: Base directory for training images (default: train_img)
        skip_existing: Skip experiments that already have results (default: False)
        dry_run: Print commands without executing (default: False)
        stop_on_error: Stop if any experiment fails (default: False)
        auto_confirm: Skip confirmation prompt (useful for Colab) (default: False)
    """
    print("="*70)
    print("QUICK BENCHMARK - Minimal Config for Pipeline Testing")
    print("="*70)
    print("\nConfiguration:")
    for key, value in QUICK_CONFIG.items():
        print(f"  {key}: {value}")
    print()

    # Parse filters
    dataset_filter = None if datasets is None else [d.strip().lower() for d in datasets.split(',')]
    method_filter = None if methods is None else [m.strip().lower() for m in methods.split(',')]
    ipc_filter = [1, 10, 50] if ipcs is None else [int(i.strip()) for i in ipcs.split(',')]
    seed_list = [int(s.strip()) for s in seeds.split(',')]

    # Get all experiment configs
    all_configs = get_experiment_configs()

    # Filter configs
    filtered_configs = []
    for dataset, ipc, method in all_configs:
        if dataset_filter and dataset not in dataset_filter:
            continue
        if method_filter and method not in method_filter:
            continue
        if ipc not in ipc_filter:
            continue
        filtered_configs.append((dataset, ipc, method))

    # Expand with seeds
    experiments = []
    for dataset, ipc, method in filtered_configs:
        for seed in seed_list:
            # Check if already exists
            if skip_existing and experiment_exists(base_log_dir, dataset, ipc, method, seed):
                print(f"Skipping existing: {dataset} | IPC={ipc} | {method} | seed={seed}")
                continue
            experiments.append((dataset, ipc, method, seed))

    print(f"\nTotal experiments to run: {len(experiments)}")

    if not experiments:
        print("No experiments to run!")
        return 0

    # Confirm before starting (unless auto_confirm is True)
    if not dry_run and not auto_confirm:
        print("\nPress Enter to start, or Ctrl+C to cancel...")
        try:
            input()
        except KeyboardInterrupt:
            print("\nCancelled by user")
            return 1
    elif auto_confirm:
        print("\nAuto-confirm enabled, starting immediately...")

    # Run all experiments
    failed = []
    completed = 0
    start_time = time.time()

    for idx, (dataset, ipc, method, seed) in enumerate(experiments, 1):
        print(f"\n\n[{idx}/{len(experiments)}] Starting experiment...")

        try:
            exit_code = run_experiment(
                dataset=dataset,
                ipc=ipc,
                method=method,
                seed=seed,
                base_log_dir=base_log_dir,
                base_img_dir=base_img_dir,
                dry_run=dry_run
            )

            if exit_code == 0:
                completed += 1
            else:
                failed.append((dataset, ipc, method, seed, exit_code))
                if stop_on_error:
                    print("\nStopping due to error (--stop_on_error=True)")
                    break

        except KeyboardInterrupt:
            print("\n\nInterrupted by user!")
            break

    # Summary
    elapsed = time.time() - start_time
    print("\n\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"Completed: {completed}/{len(experiments)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed experiments:")
        for dataset, ipc, method, seed, code in failed:
            print(f"  - {dataset} | IPC={ipc} | {method} | seed={seed} (exit code: {code})")

    if not dry_run and completed > 0:
        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("\n1. Generate comparison table:")
        print(f"   python3 -m script.generate_paper_table --base_dir={base_log_dir}")
        print("\n2. View in TensorBoard:")
        print(f"   tensorboard --logdir={base_log_dir}")

    return 0 if len(failed) == 0 else 1


if __name__ == '__main__':
    fire.Fire(main)
