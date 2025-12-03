"""
Ultra-quick pipeline test for Google Colab (non-interactive).

This script runs minimal experiments to verify:
1. TensorBoard logging fix works correctly
2. Experiment pipeline is functional
3. Table generation produces correct output

Designed for Google Colab - no interactive prompts needed.

Usage in Colab:
    !python -m script.test_pipeline_colab

Or with custom config:
    !python -m script.test_pipeline_colab --num_steps=100 --width=64
"""

import os
import subprocess
import time
from typing import Optional
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


def verify_tensorboard(seed_dir: str) -> bool:
    """
    Verify TensorBoard has scalar data.

    Args:
        seed_dir: Directory containing TensorBoard events

    Returns:
        True if scalar data exists
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
        import glob

        # Find event files
        event_files = glob.glob(os.path.join(seed_dir, 'events.out.tfevents.*'))
        if not event_files:
            print(f"  ✗ No event files in {seed_dir}")
            return False

        ea = event_accumulator.EventAccumulator(seed_dir)
        ea.Reload()

        scalar_tags = ea.Tags().get('scalars', [])

        if scalar_tags:
            print(f"  ✓ {seed_dir}")
            print(f"    Found {len(scalar_tags)} scalar tags")

            # Check for eval accuracy
            if 'eval/accuracy_mean' in scalar_tags:
                events = ea.Scalars('eval/accuracy_mean')
                if events:
                    final_acc = events[-1].value
                    print(f"    Final accuracy: {final_acc:.2f}%")
                    return True
        else:
            print(f"  ✗ {seed_dir}")
            print(f"    No scalar tags found!")
            return False

    except Exception as e:
        print(f"  ✗ {seed_dir}")
        print(f"    Error: {e}")
        return False

    return False


def main(
    num_steps: int = 50,
    width: int = 32,
    depth: int = 2,
    num_eval: int = 2,
    eval_updates: int = 300,
    base_log: str = 'train_log',
    base_img: str = 'train_img'
):
    """
    Run ultra-quick pipeline test.

    Args:
        num_steps: Number of distillation steps (default: 50)
        width: Model width (default: 32)
        depth: Model depth (default: 2)
        num_eval: Number of evaluation models (default: 2)
        eval_updates: Number of evaluation updates (default: 300)
        base_log: Base directory for logs (default: 'train_log')
        base_img: Base directory for images (default: 'train_img')
    """
    print("="*70)
    print("ULTRA-QUICK PIPELINE TEST (Google Colab)")
    print("="*70)
    print(f"""
Config:
  - num_steps: {num_steps}
  - width: {width}
  - depth: {depth}
  - num_eval: {num_eval}
  - eval_updates: {eval_updates}

This will run 3 minimal experiments (~5-10 minutes total).
""")

    total_start = time.time()
    experiments = []
    failed_experiments = []

    # Test 1: MNIST, IPC=1, MTT
    exp1 = {
        'method': 'mtt',
        'dataset': 'mnist',
        'ipc': 1,
        'description': '[1/15] MNIST | IPC=1 | MTT'
    }
    experiments.append(exp1)

    # Test 2: MNIST, IPC=10, DM
    exp2 = {
        'method': 'dm',
        'dataset': 'mnist',
        'ipc': 1,
        'description': '[2/15] MNIST | IPC=10 | DM'
    }
    experiments.append(exp2)

    # Test 3: CIFAR-10, IPC=1, FRePo
    exp3 = {
        'method': 'frepo',
        'dataset': 'mnist',
        'ipc': 1,
        'description': '[3/15] MNIST | IPC=1 | FRePo'
    }
    experiments.append(exp3)

    exp4 = {
        'method': 'kip',
        'dataset': 'mnist',
        'ipc': 1,
        'description': '[4/15] MNIST | IPC=1 | FRePo'
    }
    experiments.append(exp4)

    exp5 = {
        'method': 'dc',
        'dataset': 'mnist',
        'ipc': 1,
        'description': '[5/15] MNIST | IPC=1 | FRePo'
    }
    experiments.append(exp5)

    # Test 1: MNIST, IPC=1, MTT
    exp6 = {
        'method': 'mtt',
        'dataset': 'cifar10',
        'ipc': 1,
        'description': '[6/15] cifar10 | IPC=1 | MTT'
    }
    experiments.append(exp6)

    # Test 2: MNIST, IPC=10, DM
    exp7 = {
        'method': 'dm',
        'dataset': 'cifar10',
        'ipc': 1,
        'description': '[7/15] cifar10 | IPC=10 | DM'
    }
    experiments.append(exp7)

    # Test 3: CIFAR-10, IPC=1, FRePo
    exp8 = {
        'method': 'frepo',
        'dataset': 'cifar10',
        'ipc': 1,
        'description': '[8/15] cifar10 | IPC=1 | FRePo'
    }
    experiments.append(exp8)

    exp9 = {
        'method': 'kip',
        'dataset': 'cifar10',
        'ipc': 1,
        'description': '[9/15] cifar10 | IPC=1 | FRePo'
    }
    experiments.append(exp9)

    exp10 = {
        'method': 'dc',
        'dataset': 'cifar10',
        'ipc': 1,
        'description': '[10/15] cifar10 | IPC=1 | FRePo'
    }
    experiments.append(exp10)

    # Run experiments
    print("\n" + "="*70)
    print("Step 1/4: Running test experiments")
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
            '--save_image=True'
        ]

        exit_code = run_command(cmd, exp['description'])
        if exit_code != 0:
            failed_experiments.append(exp['description'])

    # Verify TensorBoard
    print("\n" + "="*70)
    print("Step 2/4: Verifying TensorBoard logging")
    print("="*70)

    import glob
    seed_dirs = glob.glob(f"{base_log}/*/*/*/seed*")
    print(f"\nChecking {len(seed_dirs)} experiment directories...\n")

    success_count = 0
    for seed_dir in seed_dirs:
        if verify_tensorboard(seed_dir):
            success_count += 1

    print(f"\nResult: {success_count}/{len(seed_dirs)} experiments have valid TensorBoard data")

    if success_count == 0:
        print("\n⚠️  WARNING: No experiments have TensorBoard data!")
    elif success_count < len(seed_dirs):
        print("\n⚠️  WARNING: Some experiments are missing TensorBoard data!")
    else:
        print("\n✓ All experiments have valid TensorBoard data!")

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
    print("  2. Run full benchmark:")
    print("     !python -m script.quick_benchmark --auto_confirm=True")
    print("  3. View in TensorBoard:")
    print("     %load_ext tensorboard")
    print(f"     %tensorboard --logdir {base_log}")

    return 0 if len(failed_experiments) == 0 else 1


if __name__ == '__main__':
    fire.Fire(main)
