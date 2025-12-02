#!/bin/bash
# Ultra-quick pipeline test - Run minimal experiments to verify everything works

set -e  # Exit on error

echo "========================================================================"
echo "ULTRA-QUICK PIPELINE TEST"
echo "========================================================================"
echo ""
echo "This script will run 3 ultra-minimal experiments to test:"
echo "  1. TensorBoard logging fix"
echo "  2. Experiment pipeline"
echo "  3. Table generation"
echo ""
echo "Estimated time: ~5-10 minutes"
echo ""
echo "Press Enter to continue, or Ctrl+C to cancel..."
read

# Config for ultra-minimal testing
NUM_STEPS=50
WIDTH=32
DEPTH=2
NUM_EVAL=2
EVAL_UPDATES=300

BASE_LOG="train_log"
BASE_IMG="train_img"

echo ""
echo "========================================================================"
echo "Step 1/4: Running 3 test experiments"
echo "========================================================================"

# Test 1: MNIST, IPC=1, MTT (smallest dataset, simplest method)
echo ""
echo "[1/3] MNIST | IPC=1 | MTT"
python3 -m script.distill_unified \
  --method=mtt \
  --dataset_name=mnist \
  --num_prototypes_per_class=1 \
  --num_distill_steps=$NUM_STEPS \
  --steps_per_eval=25 \
  --steps_per_log=10 \
  --width=$WIDTH \
  --depth=$DEPTH \
  --num_eval=$NUM_EVAL \
  --num_online_eval_updates=$EVAL_UPDATES \
  --random_seed=0 \
  --train_log=$BASE_LOG \
  --train_img=$BASE_IMG \
  --save_image=False

# Test 2: MNIST, IPC=10, DM
echo ""
echo "[2/3] MNIST | IPC=10 | DM"
python3 -m script.distill_unified \
  --method=dm \
  --dataset_name=mnist \
  --num_prototypes_per_class=10 \
  --num_distill_steps=$NUM_STEPS \
  --steps_per_eval=25 \
  --steps_per_log=10 \
  --width=$WIDTH \
  --depth=$DEPTH \
  --num_eval=$NUM_EVAL \
  --num_online_eval_updates=$EVAL_UPDATES \
  --random_seed=0 \
  --train_log=$BASE_LOG \
  --train_img=$BASE_IMG \
  --save_image=False

# Test 3: CIFAR-10, IPC=1, FRePo
echo ""
echo "[3/3] CIFAR-10 | IPC=1 | FRePo"
python3 -m script.distill_unified \
  --method=frepo \
  --dataset_name=cifar10 \
  --num_prototypes_per_class=1 \
  --num_distill_steps=$NUM_STEPS \
  --steps_per_eval=25 \
  --steps_per_log=10 \
  --width=$WIDTH \
  --depth=$DEPTH \
  --num_eval=$NUM_EVAL \
  --num_online_eval_updates=$EVAL_UPDATES \
  --random_seed=0 \
  --train_log=$BASE_LOG \
  --train_img=$BASE_IMG \
  --save_image=False

echo ""
echo "========================================================================"
echo "Step 2/4: Verifying TensorBoard logging"
echo "========================================================================"

# Verify TensorBoard has scalar data
python3 << 'PYEOF'
from tensorboard.backend.event_processing import event_accumulator
import glob
import os

def check_tensorboard(logdir):
    """Check if TensorBoard has scalar data."""
    try:
        ea = event_accumulator.EventAccumulator(logdir)
        ea.Reload()

        scalar_tags = ea.Tags().get('scalars', [])

        if scalar_tags:
            print(f"  ✓ {logdir}")
            print(f"    Found {len(scalar_tags)} scalar tags: {', '.join(scalar_tags[:5])}")

            # Check for eval accuracy
            if 'eval/accuracy_mean' in scalar_tags:
                events = ea.Scalars('eval/accuracy_mean')
                if events:
                    final_acc = events[-1].value
                    print(f"    Final accuracy: {final_acc:.2f}%")
                    return True
        else:
            print(f"  ✗ {logdir}")
            print(f"    No scalar tags found!")
            return False

    except Exception as e:
        print(f"  ✗ {logdir}")
        print(f"    Error: {e}")
        return False

    return False

# Find all seed directories
seed_dirs = glob.glob("train_log/*/*/*/seed*")
print(f"\nChecking {len(seed_dirs)} experiment directories...\n")

success_count = 0
for seed_dir in seed_dirs:
    if check_tensorboard(seed_dir):
        success_count += 1
    print()

print(f"Result: {success_count}/{len(seed_dirs)} experiments have valid TensorBoard data")

if success_count == 0:
    print("\n⚠️  WARNING: No experiments have TensorBoard data!")
    exit(1)
elif success_count < len(seed_dirs):
    print("\n⚠️  WARNING: Some experiments are missing TensorBoard data!")
else:
    print("\n✓ All experiments have valid TensorBoard data!")
PYEOF

echo ""
echo "========================================================================"
echo "Step 3/4: Generating comparison table"
echo "========================================================================"

# Create results directory
mkdir -p results/tables

# Generate comparison table
python3 -m script.generate_paper_table \
  --base_dir=$BASE_LOG \
  --output_dir=results/tables \
  --formats=markdown,csv

echo ""
echo "========================================================================"
echo "Step 4/4: Summary"
echo "========================================================================"

echo ""
echo "✓ Pipeline test completed successfully!"
echo ""
echo "Generated files:"
echo "  - results/tables/comparison_table.md"
echo "  - results/tables/comparison_table.csv"
echo ""
echo "Next steps:"
echo "  1. Review the comparison table: cat results/tables/comparison_table.md"
echo "  2. View in TensorBoard: tensorboard --logdir=$BASE_LOG"
echo "  3. Run full benchmark: python3 -m script.quick_benchmark"
echo ""
