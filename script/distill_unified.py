"""
Unified entry point for all dataset distillation methods.

This script provides a unified interface to run any registered distillation
method (FRePo, MTT, KIP, DC, DM) with consistent configuration and evaluation.

Usage:
    # Run FRePo
    python -m script.distill_unified --method=frepo --dataset_name=cifar100

    # Run MTT (when implemented)
    python -m script.distill_unified --method=mtt --dataset_name=cifar10

    # List available methods
    python -m script.distill_unified --list_methods
"""

import sys
sys.path.append("..")

import os
import fire
import ml_collections
from functools import partial

import jax
from absl import logging
import tensorflow as tf

from lib.dataset.dataloader import get_dataset, configure_dataloader
from lib.models.utils import create_model
from lib.datadistillation.utils import save_frepo_image, save_proto_np
from lib.datadistillation import DistillationMethodRegistry
from lib.training.utils import create_train_state
from lib.dataset.augmax import get_aug_by_name

from clu import metric_writers


def get_base_config():
    """Get base configuration shared across all methods."""
    config = ml_collections.ConfigDict()
    config.random_seed = 0
    config.train_log = 'train_log'
    config.train_img = 'train_img'
    config.resume = True

    config.img_size = None
    config.img_channels = None
    config.num_prototypes = None
    config.train_size = None

    config.dataset = ml_collections.ConfigDict()
    config.kernel = ml_collections.ConfigDict()
    config.online = ml_collections.ConfigDict()

    # Dataset
    config.dataset.name = 'cifar100'
    config.dataset.data_path = 'data/tensorflow_datasets'
    config.dataset.zca_path = 'data/zca'
    config.dataset.zca_reg = 0.1

    # Online model config
    config.online.img_size = None
    config.online.img_channels = None
    config.online.optimizer = 'adam'
    config.online.learning_rate = 0.0003
    config.online.arch = 'conv'
    config.online.output = 'feat_fc'
    config.online.width = 128
    config.online.normalization = 'identity'

    # Kernel/distillation config
    config.kernel.img_size = None
    config.kernel.img_channels = None
    config.kernel.num_prototypes = None
    config.kernel.train_size = None
    config.kernel.resume = config.resume
    config.kernel.optimizer = 'lamb'
    config.kernel.learning_rate = 0.0003
    config.kernel.batch_size = 1024
    config.kernel.eval_batch_size = 1000

    return config


def main(
    method='frepo',
    dataset_name='cifar100',
    data_path=None,
    zca_path=None,
    train_log=None,
    train_img=None,
    save_image=True,
    arch='conv',
    width=128,
    depth=3,
    normalization='identity',
    learn_label=True,
    num_prototypes_per_class=10,
    random_seed=0,
    num_train_steps=None,
    max_online_updates=100,
    num_nn_state=10,
    list_methods=False,
    **method_kwargs
):
    """
    Unified entry point for distillation.

    Args:
        method: Distillation method ('frepo', 'mtt', 'kip', 'dc', 'dm')
        dataset_name: Dataset to use ('cifar10', 'cifar100', 'mnist', etc.)
        data_path: Path to dataset directory
        zca_path: Path to ZCA whitening data
        train_log: Directory for training logs
        train_img: Directory for saving images
        save_image: Whether to save synthetic images
        arch: Model architecture ('conv', 'resnet18', 'vgg11', etc.)
        width: Model width
        depth: Model depth
        normalization: Normalization type ('identity', 'batch', etc.)
        learn_label: Whether to learn labels
        num_prototypes_per_class: Number of synthetic images per class
        random_seed: Random seed
        num_train_steps: Total training steps
        max_online_updates: Max updates for online models (FRePo specific)
        num_nn_state: Number of online models in pool (FRePo specific)
        list_methods: If True, list available methods and exit
        **method_kwargs: Additional method-specific keyword arguments
    """

    # List available methods if requested
    if list_methods:
        methods = DistillationMethodRegistry.list_methods()
        print(f"\nAvailable distillation methods ({len(methods)}):")
        print("=" * 60)
        for method_name in methods:
            info = DistillationMethodRegistry.get_method_info(method_name)
            print(f"\n  {method_name}")
            print(f"    Class: {info['class']}")
            doc = info['docstring'].strip().split('\n')[0]
            print(f"    Description: {doc}")
        print("\n" + "=" * 60)
        return

    # Check if method is registered
    if not DistillationMethodRegistry.is_registered(method):
        available = ', '.join(DistillationMethodRegistry.list_methods())
        raise ValueError(
            f"Unknown method: '{method}'. "
            f"Available methods: {available}"
        )

    # Setup
    config = get_base_config()
    config.random_seed = random_seed
    config.train_log = train_log if train_log else 'train_log'
    config.train_img = train_img if train_img else 'train_img'

    if not os.path.exists(config.train_log):
        os.makedirs(config.train_log)
    if not os.path.exists(config.train_img):
        os.makedirs(config.train_img)

    try:
        use_pmap = jax.device_count('gpu') > 1
    except RuntimeError:
        use_pmap = False
    if use_pmap:
        logging.info(f'Using Multi-GPU Training. Number of GPUs: {jax.device_count()}')

    # Dataset
    config.dataset.data_path = data_path if data_path else 'data/tensorflow_datasets'
    config.dataset.zca_path = zca_path if zca_path else 'data/zca'
    config.dataset.name = dataset_name

    (ds_train, ds_test), preprocess_op, rev_preprocess_op, proto_scale = get_dataset(config.dataset)

    num_prototypes = num_prototypes_per_class * config.dataset.num_classes
    config.kernel.num_prototypes = num_prototypes

    # Online model config
    config.online.arch = arch
    config.online.width = width
    config.online.depth = depth
    config.online.normalization = normalization
    config.online.img_size = config.dataset.img_shape[0]
    config.online.img_channels = config.dataset.img_shape[-1]

    # Determine training parameters based on dataset
    if dataset_name in ['mnist', 'fashion_mnist']:
        use_flip = False
        aug_strategy = 'color_crop_rotate_translate_cutout'
    else:
        use_flip = True
        aug_strategy = 'flip_color_crop_rotate_translate_cutout'

    if dataset_name == 'cifar100':
        if num_prototypes_per_class == 1:
            use_flip = True
            num_online_eval_updates = 1000
        elif num_prototypes_per_class == 10:
            use_flip = False
            num_online_eval_updates = 2000
        elif num_prototypes_per_class == 50:
            use_flip = False
            num_online_eval_updates = 5000
        else:
            num_online_eval_updates = 1000
    else:
        num_online_eval_updates = 1000

    # Batch normalization setup
    if normalization == 'batch':
        has_bn = True
        eval_normalization = 'identity'
    else:
        has_bn = False
        eval_normalization = normalization

    # Set default num_train_steps if not provided
    if num_train_steps is None:
        # Set defaults based on dataset
        if dataset_name in ['mnist', 'fashion_mnist']:
            num_train_steps = 1000  # Faster datasets
        elif dataset_name == 'cifar10':
            num_train_steps = 3000  # Standard
        elif dataset_name == 'cifar100':
            num_train_steps = 5000  # Harder dataset needs more steps
        else:
            num_train_steps = 3000  # Default fallback
        logging.info(f'num_train_steps not specified, using default: {num_train_steps}')

    # Logging
    steps_per_epoch = config.dataset.train_size // config.kernel.batch_size

    exp_name = os.path.join(
        f'{dataset_name}',
        f'step{num_train_steps // 1000}K_num{num_prototypes}',
        f'{method}_{config.online.arch}_w{config.online.width}_d{config.online.depth}_{config.online.normalization}_ll{learn_label}',
        f'seed{random_seed}'
    )

    image_dir = os.path.join(config.train_img, exp_name)
    workdir = os.path.join(config.train_log, exp_name)
    writer = metric_writers.create_default_writer(logdir=workdir)
    logging.info(f'Working directory: {workdir}')

    if save_image:
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        logging.info(f'Image directory: {image_dir}')

        is_grey = dataset_name in ['mnist', 'fashion_mnist']
        image_saver = partial(
            save_frepo_image,
            num_classes=config.dataset.num_classes,
            class_names=config.dataset.class_names,
            rev_preprocess_op=rev_preprocess_op,
            image_dir=image_dir,
            is_grey=is_grey,
            save_img=True,
            save_np=False
        )
    else:
        image_saver = None

    # Create models
    online_model = create_model(
        arch,
        config.dataset.num_classes,
        width=config.online.width,
        depth=config.online.depth,
        normalization=normalization,
        output=config.online.output
    )

    eval_model = create_model(
        arch,
        config.dataset.num_classes,
        width=config.online.width,
        depth=config.online.depth,
        normalization=eval_normalization,
        output=config.online.output
    )

    create_online_state = partial(
        create_train_state,
        config=config.online,
        model=online_model,
        learning_rate_fn=lambda x: config.online.learning_rate,
        has_bn=has_bn
    )
    create_eval_state = partial(
        create_train_state,
        config=config.online,
        model=eval_model,
        learning_rate_fn=lambda x: config.online.learning_rate,
        has_bn=False
    )

    # Data augmentation
    diff_aug = get_aug_by_name(aug_strategy, res=config.dataset.img_shape[0])

    # Keep untransformed train dataset for prototype initialization
    # init_proto needs integer labels to group by class
    ds_train_raw = ds_train

    # Configure datasets
    y_transform = lambda y: tf.one_hot(
        y,
        config.dataset.num_classes,
        on_value=1 - 1 / config.dataset.num_classes,
        off_value=-1 / config.dataset.num_classes
    )
    ds_train = configure_dataloader(
        ds_train,
        batch_size=config.kernel.batch_size,
        y_transform=y_transform,
        train=True,
        shuffle=True
    )
    ds_test = configure_dataloader(
        ds_test,
        batch_size=config.kernel.eval_batch_size,
        y_transform=y_transform,
        train=False,
        shuffle=False
    )
    dataset = (ds_train, ds_test)

    # Create method instance
    logging.info(f"Creating distillation method: {method}")

    # Method-specific parameters
    if method == 'frepo':
        method_instance = DistillationMethodRegistry.create(
            method,
            num_nn_state=num_nn_state,
            max_online_updates=max_online_updates,
            learn_label=learn_label,
            use_flip=use_flip,
            **method_kwargs
        )
    else:
        # Other methods (will be implemented later)
        method_instance = DistillationMethodRegistry.create(
            method,
            learn_label=learn_label,
            **method_kwargs
        )

    logging.info(f"Method instance created: {method_instance}")

    # Run training
    logging.info("Starting training...")
    final_state = method_instance.train_and_evaluate(
        config=config,
        dataset=dataset,
        ds_train_raw=ds_train_raw,
        workdir=workdir,
        seed=random_seed,
        create_online_state=create_online_state,
        create_eval_state=create_eval_state,
        diff_aug=diff_aug,
        num_train_steps=num_train_steps,
        num_online_eval_updates=num_online_eval_updates,
        steps_per_epoch=steps_per_epoch,
        steps_per_log=500,
        steps_per_eval=10000,
        steps_per_checkpoint=1000,
        save_ckpt=num_train_steps,
        steps_per_save_image=num_train_steps // 10 if num_train_steps else 10000,
        has_bn=has_bn,
        use_pmap=use_pmap,
        writer=writer,
        image_saver=image_saver,
        num_eval=5
    )

    # Save final prototypes
    save_proto_np(final_state, step=num_train_steps, image_dir=image_dir, use_pmap=False)

    logging.info('Training finished!')
    logging.info(f'Results saved to: {workdir}')
    logging.info(f'Images saved to: {image_dir}')


if __name__ == '__main__':
    tf.config.experimental.set_visible_devices([], 'GPU')
    logging.set_verbosity('info')
    fire.Fire(main)
