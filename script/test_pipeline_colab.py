"""
Grid-based pipeline test for Google Colab (non-interactive).

This script runs grid experiments to verify:
1. JSON metrics logging works correctly
2. Experiment pipeline is functional
3. Table generation produces correct output

Designed for Google Colab - no interactive prompts needed.

Usage in Colab:
    !python -m script.test_pipeline_colab

Or with custom grid:
    !python -m script.test_pipeline_colab --methods=mtt,dm --datasets=mnist --ipcs=1 --num_steps=100
"""

import os
import subprocess
import time
import json
from typing import Optional, Dict
import fire


def run_command(cmd: list, description: str) -> int:
    """
    Run a command and print output.

    Args:
        cmd: Command as list of strings
        description: Description of what command does

    Returns:
        Exit code (0 = success)
    """
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")

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


def verify_json(seed_dir: str) -> Optional[Dict]:
    """
    Verify JSON metrics file exists and has data.

    Args:
        seed_dir: Directory containing metrics.json

    Returns:
        Dict with accuracy info or None if not found
    """
    metrics_file = os.path.join(seed_dir, 'metrics.json')

    if not os.path.exists(metrics_file):
        print(f"  ✗ {seed_dir}")
        print(f"    No metrics.json found!")
        return None

    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        # Find eval metrics
        eval_metrics = [m for m in metrics if 'eval/accuracy_mean' in m or 'eval/step_acc_mean' in m]

        if not eval_metrics:
            print(f"  ✗ {seed_dir}")
            print(f"    No eval metrics in JSON!")
            return None

        # Get final accuracy
        final_metric = eval_metrics[-1]
        acc_key = 'eval/accuracy_mean' if 'eval/accuracy_mean' in final_metric else 'eval/step_acc_mean'
        final_acc = final_metric[acc_key]

        print(f"  ✓ {seed_dir}")
        print(f"    Found {len(metrics)} metric entries")
        print(f"    Final accuracy: {final_acc:.2f}%")

        return {
            'accuracy': final_acc,
            'num_metrics': len(metrics),
            'num_evals': len(eval_metrics)
        }

    except Exception as e:
        print(f"  ✗ {seed_dir}")
        print(f"    Error reading JSON: {e}")
        return None


def main(
    # Grid parameters
    methods: str = 'mtt,dm,dc,kip,frepo',
    datasets: str = 'mnist,fashion_mnist,cifar10',
    ipcs: str = '1,5',

    # Training parameters
    num_steps: int = 50,
    width: int = 32,
    depth: int = 2,
    num_eval: int = 2,
    eval_updates: int = 300,
    base_log: str = 'train_log',
    base_img: str = 'train_img'
):
    """
    Run grid-based pipeline test.

    Args:
        methods: Comma-separated list of methods (default: 'mtt,dm,dc,kip,frepo')
        datasets: Comma-separated list of datasets (default: 'mnist,fashion_mnist,cifar10')
        ipcs: Comma-separated list of IPC values (default: '1,5')
        num_steps: Number of distillation steps (default: 50)
        width: Model width (default: 32)
        depth: Model depth (default: 2)
        num_eval: Number of evaluation models (default: 2)
        eval_updates: Number of evaluation updates (default: 300)
        base_log: Base directory for logs (default: 'train_log')
        base_img: Base directory for images (default: 'train_img')
    """
    # Parse grid parameters
    # Handle both string and tuple/list (Fire parses comma-separated values as tuples)
    if isinstance(methods, (list, tuple)):
        methods_list = [m.strip() for m in methods]
    else:
        methods_list = [m.strip() for m in methods.split(',')]

    if isinstance(datasets, (list, tuple)):
        datasets_list = [d.strip() for d in datasets]
    else:
        datasets_list = [d.strip() for d in datasets.split(',')]

    if isinstance(ipcs, (list, tuple)):
        ipcs_list = [int(i) if isinstance(i, int) else int(i.strip()) for i in ipcs]
    else:
        ipcs_list = [int(i.strip()) for i in ipcs.split(',')]

    # Generate grid
    experiments = []
    exp_id = 1
    total_exp = len(methods_list) * len(datasets_list) * len(ipcs_list)

    for dataset in datasets_list:
        for ipc in ipcs_list:
            for method in methods_list:
                experiments.append({
                    'method': method,
                    'dataset': dataset,
                    'ipc': ipc,
                    'description': f'[{exp_id}/{total_exp}] {dataset.upper()} | IPC={ipc} | {method.upper()}'
                })
                exp_id += 1

    print("="*70)
    print("GRID-BASED PIPELINE TEST (Google Colab)")
    print("="*70)
    print(f"""
Grid Configuration:
  - Methods: {methods_list}
  - Datasets: {datasets_list}
  - IPCs: {ipcs_list}
  - Total experiments: {total_exp}

Training Config:
  - num_steps: {num_steps}
  - width: {width}
  - depth: {depth}
  - num_eval: {num_eval}
  - eval_updates: {eval_updates}

Estimated time: ~{total_exp * 2} minutes.
""")

    total_start = time.time()
    failed_experiments = []

    # Run experiments
    print("\n" + "="*70)
    print(f"Step 1/4: Running {len(experiments)} test experiments")
    print("="*70)

    for exp in experiments:
        cmd = [
            'python3', '-m', 'script.distill_unified',
            f'--method={exp["method"]}',
            f'--dataset_name={exp["dataset"]}',
            f'--num_prototypes_per_class={exp["ipc"]}',
            f'--num_distill_steps={num_steps}',
            f'--steps_per_eval={num_steps//2}',
            '--steps_per_log=10',
            f'--width={width}',
            f'--depth={depth}',
            f'--num_eval={num_eval}',
            f'--num_online_eval_updates={eval_updates}',
            '--random_seed=0',
            f'--train_log={base_log}',
            f'--train_img={base_img}',
            '--save_image=False'
        ]

        exit_code = run_command(cmd, exp['description'])
        if exit_code != 0:
            failed_experiments.append(exp['description'])

    # Verify JSON metrics
    print("\n" + "="*70)
    print("Step 2/4: Verifying JSON metrics")
    print("="*70)

    import glob
    seed_dirs = glob.glob(f"{base_log}/*/*/*/seed*")
    print(f"\nChecking {len(seed_dirs)} experiment directories...\n")

    success_count = 0
    results = {}
    for seed_dir in seed_dirs:
        result = verify_json(seed_dir)
        if result:
            success_count += 1
            results[seed_dir] = result

    print(f"\nResult: {success_count}/{len(seed_dirs)} experiments have valid JSON data")

    if success_count == 0:
        print("\n⚠️  WARNING: No experiments have JSON metrics!")
    elif success_count < len(seed_dirs):
        print("\n⚠️  WARNING: Some experiments are missing JSON metrics!")
    else:
        print("\n✓ All experiments have valid JSON metrics!")

    # Generate comparison table
    print("\n" + "="*70)
    print("Step 3/4: Generating comparison table")
    print("="*70)

    os.makedirs('results/tables', exist_ok=True)

    cmd = [
        'python3', '-m', 'script.generate_paper_table',
        f'--base_dir={base_log}',
        '--output_dir=results/tables',
        '--formats=markdown,csv'
    ]

    run_command(cmd, "Generating comparison table")

    # Summary
    total_elapsed = time.time() - total_start
    print("\n" + "="*70)
    print("Step 4/4: Summary")
    print("="*70)

    print(f"\n✓ Pipeline test completed in {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")

    if failed_experiments:
        print(f"\n⚠️  {len(failed_experiments)} experiments failed:")
        for exp in failed_experiments:
            print(f"  - {exp}")

    print("\nGenerated files:")
    if os.path.exists('results/tables/comparison_table.md'):
        print("  ✓ results/tables/comparison_table.md")
    if os.path.exists('results/tables/comparison_table.csv'):
        print("  ✓ results/tables/comparison_table.csv")

    print("\nNext steps:")
    print("  1. Review comparison table:")
    print("     !cat results/tables/comparison_table.md")
    print("  2. Run custom grid:")
    print(f"     !python -m script.test_pipeline_colab --methods=mtt,dm --datasets=mnist --ipcs=1")
    print("  3. View metrics:")
    print(f"     !cat {base_log}/mnist/step1K_num1/mtt_*/seed0/metrics.json")

    return 0 if len(failed_experiments) == 0 else 1


if __name__ == '__main__':
    fire.Fire(main)
