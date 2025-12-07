import os
import logging

import numpy as np

import flax

import matplotlib.pyplot as plt

from typing import (Any, Tuple, Iterable, Union)

PRNGKey = Any
Array = Any
Shape = Tuple[int]
Dtype = Any

Axes = Union[int, Iterable[int]]


def save_proto_np(proto_state, step, image_dir=None, use_pmap=False):
    if use_pmap:
        proto_state = flax.jax_utils.unreplicate(proto_state)

    x_proto, y_proto = proto_state.params['x_proto'], proto_state.params['y_proto']

    path = os.path.join(image_dir, 'np')
    if not os.path.exists(path):
        os.makedirs(path)

    save_path = os.path.join(path, 'step{}'.format(str(step).zfill(6)))
    np.savez('{}.npz'.format(save_path), image=x_proto, label=y_proto)
    logging.info('Save prototype to numpy! Path: {}'.format(save_path))


def load_proto_np(path):
    npzfile = np.load('{}.npz'.format(path))
    return npzfile['image'], npzfile['label']


def scale_for_vis(img, rev_preprocess_op=None):
    """Scale image for visualization."""
    if rev_preprocess_op:
        img = rev_preprocess_op(img)
    else:
        img = img / img.std() * 0.2 + 0.5
    img = np.clip(img, 0, 1)
    return img


def save_original_images(dataset, num_classes=10, class_names=None, rev_preprocess_op=None,
                        save_dir=None, is_grey=False, samples_per_class=10):
    """
    Save original dataset images for comparison with distilled images.

    Saves ONE PNG file per dataset (not per experiment) at the dataset level.

    Args:
        dataset: Training dataset (iterable of (image, label) batches)
        num_classes: Number of classes
        class_names: List of class names
        rev_preprocess_op: Reverse preprocessing operation
        save_dir: Directory to save files (dataset level, e.g., train_img/mnist/)
        is_grey: Whether dataset is greyscale
        samples_per_class: Number of samples to collect per class
    """
    if save_dir is None:
        logging.warning("No save_dir provided, skipping original image save")
        return

    # Collect samples from dataset
    collected_images = {i: [] for i in range(num_classes)}
    collected_labels = {i: [] for i in range(num_classes)}

    logging.info(f"Collecting original images ({samples_per_class} per class)...")

    # Iterate through dataset to collect samples
    for batch in dataset:
        # Dataset batches are tuples (images, labels) from configure_dataloader
        images, labels = batch

        # Convert to numpy if needed
        images = np.array(images)
        labels = np.array(labels)

        # Process each image in batch
        for img, label in zip(images, labels):
            # Get class index
            if len(label.shape) > 0 and label.shape[0] > 1:
                # One-hot encoded
                class_idx = label.argmax()
            else:
                # Integer label
                class_idx = int(label)

            # Collect if we need more samples for this class
            if len(collected_images[class_idx]) < samples_per_class:
                collected_images[class_idx].append(img)
                collected_labels[class_idx].append(label)

        # Check if we have enough samples
        if all(len(collected_images[i]) >= samples_per_class for i in range(num_classes)):
            break

    # Organize images into arrays
    all_images = []
    all_labels = []
    for class_idx in range(num_classes):
        all_images.extend(collected_images[class_idx][:samples_per_class])
        all_labels.extend(collected_labels[class_idx][:samples_per_class])

    x_original = np.array(all_images)
    y_original = np.array(all_labels)

    logging.info(f"Collected {len(x_original)} original images ({len(x_original)//num_classes} per class)")

    # Scale for visualization
    x_vis = np.array([scale_for_vis(img, rev_preprocess_op) for img in x_original])

    # Create visualization grid (same as save_frepo_image)
    total_images = len(x_vis)
    select_idx = list(range(total_images))

    row, col = len(select_idx) // 10, 10
    fig = plt.figure(figsize=(33, 33))

    for i, idx in enumerate(select_idx[: row * col]):
        img = x_vis[idx]
        ax = plt.subplot(row, col, i + 1)

        # Get class label
        label = y_original[idx]
        if len(label.shape) > 0 and label.shape[0] > 1:
            class_idx = label.argmax()
        else:
            class_idx = int(label)

        if class_names is not None:
            ax.set_title('{}'.format(class_names[class_idx]), x=0.5, y=0.9,
                        backgroundcolor='silver')
        else:
            ax.set_title('class_{}'.format(class_idx), x=0.5, y=0.9, backgroundcolor='silver')

        if is_grey:
            plt.imshow(np.squeeze(img), cmap='gray')
        else:
            plt.imshow(np.squeeze(img))

        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        plt.xticks([])
        plt.yticks([])

    fig.patch.set_facecolor('black')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.02)

    # Save PNG
    png_path = os.path.join(save_dir, 'original.png')
    fig.savefig(png_path, bbox_inches='tight')
    logging.info(f'Saved original images visualization: {png_path}')

    plt.close(fig)

    return fig


def save_frepo_image(proto_state, step, num_classes=10, class_names=None, rev_preprocess_op=None, image_dir=None,
                    use_pmap=False, is_grey=False, save_np=False, save_img=False):
    if use_pmap:
        proto_state = flax.jax_utils.unreplicate(proto_state)

    x_proto, y_proto = proto_state.apply_fn(variables={'params': proto_state.params})

    if save_np and image_dir:
        path = os.path.join(image_dir, 'np')
        if not os.path.exists(path):
            os.mkdir(path)
        save_path = os.path.join(path, 'step{}'.format(str(step).zfill(6)))
        logging.info('Save prototype to numpy! Path: {}'.format(save_path))
        np.savez('{}.npz'.format(save_path), image=x_proto, label=y_proto)

    x_proto = scale_for_vis(x_proto, rev_preprocess_op)

    total_images = y_proto.shape[0]
    total_index = list(range(total_images))
    total_img_per_class = total_images // num_classes
    img_per_class = 100 // num_classes

    if num_classes <= 100:
        select_idx = []
        # always select the top to make it consistent
        for i in range(num_classes):
            select = total_index[i * total_img_per_class: (i + 1) * total_img_per_class][:img_per_class]
            select_idx.extend(select)
    else:
        select_idx = []
        # always select the top to make it consistent
        for i in range(100):
            select = total_index[i * total_img_per_class: (i + 1) * total_img_per_class][0]
            select_idx.append(select)

    row, col = len(select_idx) // 10, 10
    fig = plt.figure(figsize=(33, 33))

    for i, idx in enumerate(select_idx[: row * col]):
        img = x_proto[idx]
        ax = plt.subplot(row, col, i + 1)
        if class_names is not None:
            ax.set_title('{}'.format(class_names[y_proto[idx].argmax(-1)], y_proto[idx].argmax(-1)), x=0.5, y=0.9,
                         backgroundcolor='silver')
        else:
            ax.set_title('class_{}'.format(y_proto[idx].argmax(-1)), x=0.5, y=0.9, backgroundcolor='silver')

        if is_grey:
            plt.imshow(np.squeeze(img), cmap='gray')
        else:
            plt.imshow(np.squeeze(img))

        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        plt.imshow(np.squeeze(img))
        plt.xticks([])
        plt.yticks([])

    fig.patch.set_facecolor('black')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.02)

    if save_img and image_dir:
        path = os.path.join(image_dir, 'png')
        if not os.path.exists(path):
            os.mkdir(path)
        save_path = os.path.join(path, 'step{}'.format(str(step).zfill(6)))
        logging.info('Save prototype to numpy! Path: {}'.format(save_path))
        fig.savefig('{}.png'.format(save_path), bbox_inches='tight')

    return fig
