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
    Save TRULY ORIGINAL images for a single dataset (no preprocessing at all).

    Args:
        dataset_name: Name of dataset (mnist, cifar10, etc.)
        output_dir: Base output directory (default: train_img)
        data_path: Path to dataset files (optional)
        zca_path: Path to ZCA whitening data (optional, unused but kept for API compatibility)
        samples_per_class: Number of samples per class (auto-calculated if None)
        force: Skip confirmation prompt if file exists (default: False)
    """
    import tensorflow_datasets as tfds
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict
    
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
    print(f"Saving ORIGINAL (raw) images for {dataset_name.upper()}")
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

    # Load RAW dataset directly from tfds (no preprocessing!)
    print(f"\nLoading RAW dataset (no preprocessing)...")
    try:
        data_dir = data_path if data_path else './data'
        
        # Determine split
        if dataset_name in ['imagenet_resized/64x64', 'imagenette', 'imagewoof']:
            split = 'train'
        elif dataset_name in ['deep_weeds']:
            split = 'train[:80%]'
        else:
            split = 'train'
        
        # Load raw dataset directly
        ds_raw = tfds.load(
            dataset_name, 
            split=split, 
            data_dir=data_dir, 
            as_supervised=True,
            shuffle_files=False
        )
        
        # Get class names
        ds_info = tfds.builder(dataset_name, data_dir=data_dir).info
        class_names = ds_info.features['label'].names

        # Collect RAW samples from dataset
        collected_images = defaultdict(list)
        seen_classes = set()

        print(f"Collecting raw images ({samples_per_class} per class)...")

        for image, label in ds_raw:
            # Convert to numpy
            img = image.numpy()  # Raw uint8 [0, 255]
            label_idx = int(label.numpy())
            
            seen_classes.add(label_idx)

            # Collect if we need more samples for this class
            if len(collected_images[label_idx]) < samples_per_class:
                # Normalize to [0, 1] for visualization only (still "original" appearance)
                img_normalized = img.astype(np.float32) / 255.0
                collected_images[label_idx].append((img_normalized, label_idx))

            # Check if we have enough samples
            if (len(seen_classes) >= num_classes and 
                all(len(collected_images[c]) >= samples_per_class for c in list(seen_classes)[:num_classes])):
                break

        # Get sorted list of class indices
        found_classes = sorted(seen_classes)[:num_classes]
        print(f"Found {len(found_classes)} classes")

        # Organize images into arrays
        all_images = []
        all_labels = []
        for class_idx in found_classes:
            for img, lbl in collected_images[class_idx][:samples_per_class]:
                all_images.append(img)
                all_labels.append(lbl)

        x_original = np.array(all_images)
        y_original = np.array(all_labels)

        print(f"Collected {len(x_original)} raw images ({len(x_original)//len(found_classes)} per class)")

        # Create visualization grid
        total_images = len(x_original)
        row, col = total_images // 10, 10
        fig = plt.figure(figsize=(33, 33))

        for i in range(min(row * col, total_images)):
            img = x_original[i]
            ax = plt.subplot(row, col, i + 1)

            class_idx = y_original[i]
            if class_names is not None and class_idx < len(class_names):
                ax.set_title('{}'.format(class_names[class_idx]), x=0.5, y=0.9,
                            backgroundcolor='silver')
            else:
                ax.set_title('class_{}'.format(class_idx), x=0.5, y=0.9, backgroundcolor='silver')

            if is_grey:
                plt.imshow(np.squeeze(img), cmap='gray')
            else:
                plt.imshow(img)

            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            plt.xticks([])
            plt.yticks([])

        fig.patch.set_facecolor('black')
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.02, hspace=0.02)

        # Save PNG
        fig.savefig(original_png, bbox_inches='tight')
        plt.close(fig)

        print(f"\n{'='*70}")
        print(f"SUCCESS!")
        print(f"{'='*70}")
        print(f"Original (RAW) images saved to: {original_png}")
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
