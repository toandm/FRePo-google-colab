"""
FRePo adapter to conform to BaseDistillationMethod interface.

This module wraps the existing FRePo implementation without modifying the
original code, allowing it to work with the new unified framework while
maintaining backward compatibility.
"""

from typing import Any, Tuple, Callable, Dict
from functools import partial

import jax
import jax.numpy as jnp
import optax

from .base import BaseDistillationMethod, DistillationState
from .registry import DistillationMethodRegistry
from .frepo import (
    init_proto,
    ProtoHolder,
    create_proto_state,
    proto_train_and_evaluate,
    proto_train_step,
    nfr,
)

Array = Any
PRNGKey = Any


@DistillationMethodRegistry.register('frepo')
class FRePoMethod(BaseDistillationMethod):
    """
    FRePo (Feature Regression with Pooling) method adapter.

    This class wraps the existing FRePo implementation to conform to the
    BaseDistillationMethod interface. It delegates most operations to the
    original FRePo functions in frepo.py, ensuring backward compatibility.

    Key features of FRePo:
    - Uses kernel ridge regression for efficient meta-gradient computation
    - Maintains a pool of online models to prevent overfitting
    - Optimizes synthetic images and labels jointly

    Configuration parameters:
        num_nn_state: Number of models in the online model pool (default: 10)
        max_online_updates: Max steps before resetting an online model (default: 100)
        learn_label: Whether to learn labels or keep them fixed (default: True)
        use_flip: Use horizontal flip augmentation (default: False)
        kernel_reg: Regularization for kernel ridge regression (default: 1e-6)
    """

    def __init__(
        self,
        num_nn_state: int = 10,
        max_online_updates: int = 100,
        learn_label: bool = True,
        use_flip: bool = False,
        kernel_reg: float = 1e-6,
        **kwargs
    ):
        """
        Initialize FRePo method.

        Args:
            num_nn_state: Number of online models in the pool
            max_online_updates: Maximum updates before model reset
            learn_label: Whether to optimize labels
            use_flip: Use horizontal flip for data augmentation
            kernel_reg: Kernel ridge regression regularization
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.num_nn_state = num_nn_state
        self.max_online_updates = max_online_updates
        self.learn_label = learn_label
        self.use_flip = use_flip
        self.kernel_reg = kernel_reg

    def initialize_synthetic_data(
        self,
        ds: Any,
        num_prototypes_per_class: int,
        num_classes: int,
        seed: int = 0,
        scale_y: bool = True,
        **kwargs
    ) -> Tuple[Array, Array]:
        """
        Initialize prototypes from real data.

        Delegates to frepo.init_proto() which:
        1. Samples real images from each class
        2. Converts labels to one-hot and centers them
        3. Optionally scales labels

        Args:
            ds: Training dataset (TensorFlow dataset)
            num_prototypes_per_class: Number of prototypes per class
            num_classes: Total number of classes
            seed: Random seed
            scale_y: Whether to scale labels (default: True)
            **kwargs: Additional parameters

        Returns:
            Tuple of (x_proto, y_proto)
        """
        return init_proto(
            ds=ds,
            num_prototypes_per_class=num_prototypes_per_class,
            num_classes=num_classes,
            seed=seed,
            scale_y=scale_y,
            **kwargs
        )

    def create_distillation_state(
        self,
        rng: PRNGKey,
        x_proto_init: Array,
        y_proto_init: Array,
        learning_rate_fn: Callable,
        optimizer: str = 'lamb',
        **kwargs
    ) -> Any:  # Returns ProtoState, not DistillationState
        """
        Create ProtoHolder and its training state.

        Delegates to frepo.create_proto_state() which creates a ProtoState
        containing the learnable ProtoHolder module.

        Args:
            rng: JAX random key
            x_proto_init: Initial prototype images
            y_proto_init: Initial prototype labels
            learning_rate_fn: Learning rate schedule
            optimizer: Optimizer name (default: 'lamb')
            **kwargs: Additional parameters

        Returns:
            ProtoState (from frepo.py, compatible with DistillationState)
        """
        # Create ProtoHolder module
        num_prototypes = x_proto_init.shape[0]
        ph = ProtoHolder(
            x_proto_init=x_proto_init,
            y_proto_init=y_proto_init,
            num_prototypes=num_prototypes,
            learn_label=self.learn_label
        )

        # Create training state
        state = create_proto_state(
            rng=rng,
            model=ph,
            learning_rate_fn=learning_rate_fn,
            optimizer=optimizer
        )

        return state

    def distillation_step(
        self,
        state: Any,  # ProtoState
        nn_state: Any,
        batch: Dict[str, Array],
        rng: PRNGKey,
        feat_fn: Callable = None,
        **kwargs
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Perform one FRePo distillation step.

        Delegates to frepo.proto_train_step() which:
        1. Extracts current prototypes from state
        2. Computes predictions using kernel ridge regression
        3. Updates prototypes via gradient descent

        Args:
            state: Current ProtoState
            nn_state: Online model state for feature extraction
            batch: Real data batch {'image': ..., 'label': ...}
            rng: JAX random key
            feat_fn: Feature extraction function
            **kwargs: Additional parameters

        Returns:
            Tuple of (new_state, metrics)
        """
        new_state, metrics = proto_train_step(
            state=state,
            nn_state=nn_state,
            batch=batch,
            use_flip=self.use_flip,
            feat_fn=feat_fn
        )

        return new_state, metrics

    def train_and_evaluate(
        self,
        config: Any,
        dataset: Tuple[Any, Any],
        workdir: str,
        seed: int = 0,
        create_online_state: Callable = None,
        create_eval_state: Callable = None,
        diff_aug: Callable = None,
        num_train_steps: int = None,
        num_online_eval_updates: int = 1000,
        steps_per_epoch: int = None,
        steps_per_log: int = 500,
        steps_per_eval: int = 10000,
        steps_per_checkpoint: int = 1000,
        save_ckpt: int = None,
        steps_per_save_image: int = None,
        has_bn: bool = False,
        use_pmap: bool = False,
        writer: Any = None,
        image_saver: Any = None,
        num_eval: int = 5,
        **kwargs
    ) -> Any:
        """
        Main FRePo training and evaluation loop.

        Delegates to frepo.proto_train_and_evaluate() which handles:
        - Model pool initialization and management
        - Periodic evaluation on test set
        - Checkpoint saving
        - Logging and visualization

        Args:
            config: Configuration object
            dataset: Tuple of (ds_train, ds_test)
            workdir: Directory for checkpoints and logs
            seed: Random seed
            create_online_state: Function to create online model
            create_eval_state: Function to create evaluation model
            diff_aug: Differentiable augmentation function
            num_train_steps: Total training steps
            num_online_eval_updates: Steps to train eval models
            steps_per_epoch: Steps per epoch (for logging)
            steps_per_log: Logging frequency
            steps_per_eval: Evaluation frequency
            steps_per_checkpoint: Checkpoint frequency
            save_ckpt: Save checkpoint at specific steps
            steps_per_save_image: Image saving frequency
            has_bn: Whether model has batch normalization
            use_pmap: Use multi-GPU training
            writer: TensorBoard writer
            image_saver: Image saving function
            num_eval: Number of models to evaluate
            **kwargs: Additional parameters

        Returns:
            Final ProtoState
        """
        # Initialize prototypes
        ds_train, ds_test = dataset
        num_classes = config.dataset.num_classes

        x_proto, y_proto = self.initialize_synthetic_data(
            ds=ds_train,
            num_prototypes_per_class=config.kernel.num_prototypes // num_classes,
            num_classes=num_classes,
            seed=seed,
            scale_y=True
        )

        # Create ProtoHolder
        ph = ProtoHolder(
            x_proto_init=x_proto,
            y_proto_init=y_proto,
            num_prototypes=config.kernel.num_prototypes,
            learn_label=self.learn_label
        )

        # Delegate to original FRePo training function
        state = proto_train_and_evaluate(
            config=config.kernel,
            ph=ph,
            create_online_state=create_online_state,
            create_eval_state=create_eval_state,
            dataset=dataset,
            workdir=workdir,
            seed=seed,
            num_nn_state=self.num_nn_state,
            num_train_steps=num_train_steps,
            use_flip=self.use_flip,
            num_online_eval_updates=num_online_eval_updates,
            diff_aug=diff_aug,
            max_online_updates=self.max_online_updates,
            steps_per_epoch=steps_per_epoch,
            steps_per_log=steps_per_log,
            steps_per_eval=steps_per_eval,
            steps_per_checkpoint=steps_per_checkpoint,
            save_ckpt=save_ckpt,
            steps_per_save_image=steps_per_save_image,
            has_bn=has_bn,
            use_pmap=use_pmap,
            writer=writer,
            image_saver=image_saver,
            num_eval=num_eval,
            **kwargs
        )

        return state

    def get_synthetic_data(
        self,
        state: Any,  # ProtoState
        use_pmap: bool = False,
        **kwargs
    ) -> Tuple[Array, Array]:
        """
        Extract synthetic prototypes from state.

        Args:
            state: ProtoState containing learned prototypes
            use_pmap: Whether pmap was used during training
            **kwargs: Additional parameters

        Returns:
            Tuple of (x_proto, y_proto)
        """
        from .frepo import get_proto

        return get_proto(state, use_pmap=use_pmap)

    def __repr__(self):
        """String representation."""
        return (
            f"FRePoMethod("
            f"num_nn_state={self.num_nn_state}, "
            f"max_online_updates={self.max_online_updates}, "
            f"learn_label={self.learn_label}, "
            f"use_flip={self.use_flip})"
        )
