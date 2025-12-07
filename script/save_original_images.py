"""
Save original dataset images for comparison with distilled images.

This standalone script creates original image visualizations for each dataset
without needing to run full distillation experiments.

Usage:
    # Save original images for a single dataset
    python -m script.save_original_images --dataset=mnist

    # Save for multiple datasets
    python -m script.save_original_images --datasets=mnist,cifar10,cifar100

    # Custom output directory
    python -m script.save_original_images --dataset=mnist --output_dir=my_images

    # List available datasets
    python -m script.save_original_images --list_datasets
"""

import sys
sys.path.append("..")

import os
import fire
from typing import Optional
import ml_collections

from absl import logging
import tensorflow as tf

from lib.dataset.dataloader import get_dataset, configure_dataloader
from lib.datadistillation.utils import save_original_images


# Available datasets with their configurations
DATASET_CONFIGS = {
    'mnist': {
        'num_classes': 10,
        'is_grey': True,
    },
    'fashion_mnist': {
        'num_classes': 10,
        'is_grey': True,
    },
    'cifar10': {
        'num_classes': 10,
        'is_grey': False,
    },
    'cifar100': {
        'num_classes': 100,
        'is_grey': False,
    },
    'svhn_cropped': {
        'num_classes': 10,
        'is_grey': False,
    },
    'caltech101': {
        'num_classes': 101,
        'is_grey': False,
    },
}


def save_for_dataset(
    dataset_name: str,
    output_dir: str = 'train_img',
    data_path: Optional[str] = None,
    zca_path: Optional[str] = None,
    samples_per_class: Optional[int] = None,
    force: bool = False
):
    """
    Save original images for a single dataset.

    Args:
        dataset_name: Name of dataset (mnist, cifar10, etc.)
        output_dir: Base output directory (default: train_img)
        data_path: Path to dataset files (optional)
        zca_path: Path to ZCA whitening data (optional)
        samples_per_class: Number of samples per class (auto-calculated if None)
        force: Skip confirmation prompt if file exists (default: False)
    """
    if dataset_name not in DATASET_CONFIGS:
        print(f"Error: Unknown dataset '{dataset_name}'")
        print(f"Available datasets: {', '.join(DATASET_CONFIGS.keys())}")
        return False

    config = DATASET_CONFIGS[dataset_name]
    num_classes = config['num_classes']
    is_grey = config['is_grey']

    # Auto-calculate samples per class if not specified
    if samples_per_class is None:
        samples_per_class = 100 // num_classes if num_classes <= 100 else 1

    print(f"\n{'='*70}")
    print(f"Saving original images for {dataset_name.upper()}")
    print(f"{'='*70}")
    print(f"Number of classes: {num_classes}")
    print(f"Samples per class: {samples_per_class}")
    print(f"Greyscale: {is_grey}")

    # Create output directory
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Check if already exists
    original_png = os.path.join(dataset_dir, 'original.png')
    if os.path.exists(original_png):
        if not force:
            print(f"\nWarning: {original_png} already exists!")
            try:
                response = input("Overwrite? (y/n): ")
                if response.lower() != 'y':
                    print("Skipped.")
                    return False
            except EOFError:
                # Non-interactive environment (e.g., Colab)
                print("Non-interactive environment detected. Use --force to overwrite.")
                print("Skipped.")
                return False
        else:
            print(f"\nOverwriting existing file (--force): {original_png}")

    # Load dataset
    print(f"\nLoading dataset...")
    try:
        # Create config for get_dataset
        dataset_config = ml_collections.ConfigDict()
        dataset_config.name = dataset_name
        dataset_config.data_path = data_path if data_path else None
        dataset_config.zca_path = zca_path if zca_path else None
        dataset_config.zca_reg = 0.1

        # get_dataset returns: (ds_train, ds_test), preprocess_op, rev_preprocess_op, proto_scale
        (ds_train, ds_test), preprocess_op, rev_preprocess_op, proto_scale = get_dataset(dataset_config)

        # configure_dataloader expects: (ds, batch_size, x_transform, y_transform, train, shuffle, seed)
        train_ds = configure_dataloader(
            ds=ds_train,
            batch_size=128,
            x_transform=None,
            y_transform=None,
            train=False,
            shuffle=False,
            seed=0
        )

        # Save original images
        print(f"Saving original images...")
        # Note: class_names is set on dataset_config by get_dataset
        save_original_images(
            dataset=train_ds,
            num_classes=num_classes,
            class_names=getattr(dataset_config, 'class_names', None),
            rev_preprocess_op=rev_preprocess_op,
            save_dir=dataset_dir,
            is_grey=is_grey,
            samples_per_class=samples_per_class
        )

        print(f"\n{'='*70}")
        print(f"SUCCESS!")
        print(f"{'='*70}")
        print(f"Original images saved to: {original_png}")
        print(f"{'='*70}\n")

        return True

    except Exception as e:
        print(f"\nError: Failed to save original images: {e}")
        import traceback
        traceback.print_exc()
        return False


def main(
    dataset: Optional[str] = None,
    datasets: Optional[str] = None,
    output_dir: str = 'train_img',
    data_path: Optional[str] = None,
    zca_path: Optional[str] = None,
    samples_per_class: Optional[int] = None,
    force: bool = False,
    list_datasets: bool = False
):
    """
    Save original dataset images for comparison with distilled images.

    Args:
        dataset: Single dataset name (e.g., 'mnist')
        datasets: Multiple datasets separated by comma (e.g., 'mnist,cifar10')
        output_dir: Base output directory (default: 'train_img')
        data_path: Path to dataset files (optional)
        zca_path: Path to ZCA whitening data (optional)
        samples_per_class: Number of samples per class (auto-calculated if None)
        force: Skip confirmation prompt if file exists (default: False)
        list_datasets: List available datasets and exit

    Examples:
        # Save for single dataset
        python -m script.save_original_images --dataset=mnist

        # Save for multiple datasets
        python -m script.save_original_images --datasets=mnist,cifar10,cifar100

        # Custom samples per class
        python -m script.save_original_images --dataset=mnist --samples_per_class=5

        # Force overwrite existing files (useful for Colab/non-interactive)
        python -m script.save_original_images --dataset=mnist --force
    """
    # List datasets if requested
    if list_datasets:
        print("\nAvailable datasets:")
        for name, config in DATASET_CONFIGS.items():
            print(f"  - {name:15s} ({config['num_classes']:3d} classes, "
                  f"{'greyscale' if config['is_grey'] else 'RGB'})")
        return 0

    # Determine which datasets to process
    if dataset and datasets:
        print("Error: Specify either --dataset or --datasets, not both")
        return 1

    if dataset:
        dataset_list = [dataset]
    elif datasets:
        # Handle both string and tuple inputs (Fire sometimes converts comma-separated values to tuples)
        if isinstance(datasets, (list, tuple)):
            dataset_list = [d.strip() for d in datasets]
        else:
            dataset_list = [d.strip() for d in datasets.split(',')]
    else:
        print("Error: Must specify --dataset or --datasets")
        print("Use --list_datasets to see available datasets")
        return 1

    # Process each dataset
    success_count = 0
    failed_count = 0

    for dataset_name in dataset_list:
        success = save_for_dataset(
            dataset_name=dataset_name,
            output_dir=output_dir,
            data_path=data_path,
            zca_path=zca_path,
            samples_per_class=samples_per_class,
            force=force
        )

        if success:
            success_count += 1
        else:
            failed_count += 1

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total datasets: {len(dataset_list)}")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"{'='*70}\n")

    return 0 if failed_count == 0 else 1


if __name__ == '__main__':
    fire.Fire(main)
