"""
Save original dataset images - simple version.

Usage:
    python -m script.save_original_images --dataset=mnist
    python -m script.save_original_images --dataset=caltech101 --force
"""

import os
import fire
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from collections import defaultdict
from PIL import Image


DATASET_CONFIGS = {
    'mnist': {'num_classes': 10, 'is_grey': True},
    'fashion_mnist': {'num_classes': 10, 'is_grey': True},
    'cifar10': {'num_classes': 10, 'is_grey': False},
    'cifar100': {'num_classes': 100, 'is_grey': False},
    'svhn_cropped': {'num_classes': 10, 'is_grey': False},
    'caltech101': {'num_classes': 101, 'is_grey': False},
}


def save_original_images(
    dataset: str,
    output_dir: str = 'train_img',
    data_path: str = './data',
    samples_per_class: int = None,
    target_size: int = 64,
    force: bool = False
):
    """
    Save original images from dataset as PNG.
    
    Args:
        dataset: Dataset name (mnist, cifar10, caltech101, etc.)
        output_dir: Output directory
        data_path: Path to dataset
        samples_per_class: Samples per class (auto if None)
        target_size: Resize images to this size
        force: Overwrite existing files
    """
    if dataset not in DATASET_CONFIGS:
        print(f"Unknown dataset: {dataset}")
        print(f"Available: {list(DATASET_CONFIGS.keys())}")
        return
    
    cfg = DATASET_CONFIGS[dataset]
    num_classes = cfg['num_classes']
    is_grey = cfg['is_grey']
    
    if samples_per_class is None:
        samples_per_class = max(1, 100 // num_classes)
    
    # Output path
    out_dir = os.path.join(output_dir, dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'original.png')
    
    if os.path.exists(out_file) and not force:
        print(f"{out_file} exists. Use --force to overwrite.")
        return
    
    print(f"Loading {dataset}...")
    ds = tfds.load(dataset, split='train', data_dir=data_path, as_supervised=True)
    info = tfds.builder(dataset, data_dir=data_path).info
    class_names = info.features['label'].names
    
    # Collect images
    collected = defaultdict(list)
    
    for img, label in ds:
        label_idx = int(label.numpy())
        if len(collected[label_idx]) < samples_per_class:
            # Resize if needed
            img_np = img.numpy()
            pil_img = Image.fromarray(img_np)
            pil_img = pil_img.resize((target_size, target_size), Image.LANCZOS)
            collected[label_idx].append(np.array(pil_img))
        
        # Check if done
        if all(len(collected[c]) >= samples_per_class for c in list(collected.keys())[:num_classes]):
            if len(collected) >= num_classes:
                break
    
    # Build grid
    classes = sorted(collected.keys())[:num_classes]
    images = []
    labels = []
    for c in classes:
        for img in collected[c][:samples_per_class]:
            images.append(img)
            labels.append(c)
    
    total = len(images)
    cols = 10
    rows = (total + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten() if rows > 1 else [axes] if rows == 1 and cols == 1 else axes
    
    for i, ax in enumerate(axes):
        if i < total:
            if is_grey:
                ax.imshow(images[i], cmap='gray')
            else:
                ax.imshow(images[i])
            
            lbl = labels[i]
            name = class_names[lbl] if lbl < len(class_names) else f"class_{lbl}"
            ax.set_title(name, fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {out_file}")
    print(f"Total images: {total} ({samples_per_class} per class x {len(classes)} classes)")


if __name__ == '__main__':
    fire.Fire(save_original_images)
