"""
Distribution Matching (DM) method via Maximum Mean Discrepancy (MMD).

This module implements the Distribution Matching algorithm which learns synthetic
data by matching the feature distributions between real and synthetic data using MMD.

Reference:
    Dataset Distillation via Factorization
    Bohdal et al., NeurIPS 2020
    https://arxiv.org/abs/2006.06393

Key idea:
    Minimize Maximum Mean Discrepancy between feature distributions:
    MMD(P_real, P_synthetic) = ||E[φ(x_real)] - E[φ(x_synthetic)]||²
"""

from typing import Any, Tuple, Callable, Dict
from functools import partial
import dataclasses

import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import tqdm
from absl import logging

from .base import BaseDistillationMethod, DistillationState
from .registry import DistillationMethodRegistry
from .frepo import (
    init_proto,
    ProtoHolder,
    create_proto_state,
    nn_feat_fn,
    train_on_proto,
    eval_on_proto_nn,
    get_proto,
)
from ..training.utils import process_batch, create_train_state, EMA
from ..training.metrics import get_metrics

Array = Any
PRNGKey = Any


@DistillationMethodRegistry.register('dm')
class DMMethod(BaseDistillationMethod):
    """
    Distribution Matching via Maximum Mean Discrepancy (MMD).

    This method learns synthetic data by matching the feature distributions
    of real and synthetic data. Unlike gradient matching (DC), DM operates
    in the feature space and uses kernel methods to measure distribution similarity.

    Algorithm:
        1. Extract features from real data batch: φ(x_real)
        2. Extract features from synthetic data: φ(x_synthetic)
        3. Compute MMD between the two distributions
        4. Update synthetic data to minimize MMD

    Configuration parameters:
        matching_type: Type of matching ('mmd', 'moment')
        mmd_kernel: Kernel type for MMD ('rbf', 'linear', 'polynomial')
        kernel_bandwidth: Bandwidth for RBF kernel (default: 1.0)
        learn_label: Whether to learn labels (default: True)
        match_in_feature_space: Use deep features vs raw pixels (default: True)
    """

    def __init__(
        self,
        matching_type: str = 'mmd',
        mmd_kernel: str = 'rbf',
        kernel_bandwidth: float = 1.0,
        learn_label: bool = True,
        match_in_feature_space: bool = True,
        **kwargs
    ):
        """
        Initialize DM method.

        Args:
            matching_type: Matching type ('mmd', 'moment')
            mmd_kernel: Kernel for MMD ('rbf', 'linear', 'polynomial')
            kernel_bandwidth: RBF kernel bandwidth
            learn_label: Whether to optimize labels
            match_in_feature_space: Match features vs pixels
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.matching_type = matching_type
        self.mmd_kernel = mmd_kernel
        self.kernel_bandwidth = kernel_bandwidth
        self.learn_label = learn_label
        self.match_in_feature_space = match_in_feature_space

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
        Initialize synthetic data from real data samples.

        Args:
            ds: Training dataset
            num_prototypes_per_class: Number of synthetic images per class
            num_classes: Total number of classes
            seed: Random seed
            scale_y: Whether to scale labels
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
        optimizer: str = 'adam',
        **kwargs
    ) -> Any:
        """
        Create ProtoHolder and its training state.

        Args:
            rng: JAX random key
            x_proto_init: Initial synthetic images
            y_proto_init: Initial synthetic labels
            learning_rate_fn: Learning rate schedule
            optimizer: Optimizer name (default: 'adam')
            **kwargs: Additional parameters

        Returns:
            ProtoState containing synthetic data parameters
        """
        num_prototypes = x_proto_init.shape[0]
        ph = ProtoHolder(
            x_proto_init=x_proto_init,
            y_proto_init=y_proto_init,
            num_prototypes=num_prototypes,
            learn_label=self.learn_label
        )

        state = create_proto_state(
            rng=rng,
            model=ph,
            learning_rate_fn=learning_rate_fn,
            optimizer=optimizer
        )

        return state

    def rbf_kernel(
        self,
        x: Array,
        y: Array,
        bandwidth: float = 1.0
    ) -> Array:
        """
        Compute RBF (Gaussian) kernel between two sets of samples.

        K(x, y) = exp(-||x - y||² / (2 * bandwidth²))

        Args:
            x: First set of samples [N, D]
            y: Second set of samples [M, D]
            bandwidth: Kernel bandwidth (sigma)

        Returns:
            Kernel matrix [N, M]
        """
        # Compute pairwise squared distances
        # ||x - y||² = ||x||² + ||y||² - 2<x, y>
        x_norm = jnp.sum(x ** 2, axis=1, keepdims=True)  # [N, 1]
        y_norm = jnp.sum(y ** 2, axis=1, keepdims=True)  # [M, 1]

        distances = x_norm + y_norm.T - 2 * jnp.dot(x, y.T)  # [N, M]

        # Apply RBF kernel
        kernel_matrix = jnp.exp(-distances / (2 * bandwidth ** 2))

        return kernel_matrix

    def linear_kernel(
        self,
        x: Array,
        y: Array
    ) -> Array:
        """
        Compute linear kernel between two sets of samples.

        K(x, y) = <x, y>

        Args:
            x: First set of samples [N, D]
            y: Second set of samples [M, D]

        Returns:
            Kernel matrix [N, M]
        """
        return jnp.dot(x, y.T)

    def polynomial_kernel(
        self,
        x: Array,
        y: Array,
        degree: int = 3,
        coef0: float = 1.0
    ) -> Array:
        """
        Compute polynomial kernel between two sets of samples.

        K(x, y) = (<x, y> + coef0)^degree

        Args:
            x: First set of samples [N, D]
            y: Second set of samples [M, D]
            degree: Polynomial degree
            coef0: Coefficient

        Returns:
            Kernel matrix [N, M]
        """
        return (jnp.dot(x, y.T) + coef0) ** degree

    def compute_mmd(
        self,
        x: Array,
        y: Array,
        kernel_type: str = 'rbf',
        bandwidth: float = 1.0
    ) -> float:
        """
        Compute Maximum Mean Discrepancy between two distributions.

        MMD²(P, Q) = E[K(x, x')] - 2E[K(x, y)] + E[K(y, y')]
        where x, x' ~ P and y, y' ~ Q

        Args:
            x: Samples from first distribution [N, D]
            y: Samples from second distribution [M, D]
            kernel_type: Kernel type ('rbf', 'linear', 'polynomial')
            bandwidth: Kernel bandwidth (for RBF)

        Returns:
            MMD² value (scalar)
        """
        # Select kernel function
        if kernel_type == 'rbf':
            kernel_fn = partial(self.rbf_kernel, bandwidth=bandwidth)
        elif kernel_type == 'linear':
            kernel_fn = self.linear_kernel
        elif kernel_type == 'polynomial':
            kernel_fn = self.polynomial_kernel
        else:
            raise ValueError(f'Unknown kernel type: {kernel_type}')

        # Compute kernel matrices
        k_xx = kernel_fn(x, x)  # [N, N]
        k_yy = kernel_fn(y, y)  # [M, M]
        k_xy = kernel_fn(x, y)  # [N, M]

        # Compute MMD² (unbiased estimator)
        n = x.shape[0]
        m = y.shape[0]

        # E[K(x, x')] - diagonal is excluded for unbiased estimate
        term1 = (jnp.sum(k_xx) - jnp.trace(k_xx)) / (n * (n - 1))

        # E[K(y, y')] - diagonal is excluded
        term2 = (jnp.sum(k_yy) - jnp.trace(k_yy)) / (m * (m - 1))

        # E[K(x, y)]
        term3 = jnp.mean(k_xy)

        mmd_squared = term1 + term2 - 2 * term3

        # Ensure non-negative (numerical stability)
        mmd_squared = jnp.maximum(mmd_squared, 0.0)

        return mmd_squared

    def compute_moment_matching(
        self,
        x: Array,
        y: Array,
        order: int = 2
    ) -> float:
        """
        Compute moment matching loss between two distributions.

        Matches mean and variance (or higher moments).

        Args:
            x: Samples from first distribution [N, D]
            y: Samples from second distribution [M, D]
            order: Moment order (1=mean, 2=mean+var)

        Returns:
            Moment matching loss
        """
        # First moment (mean)
        mean_x = jnp.mean(x, axis=0)
        mean_y = jnp.mean(y, axis=0)
        loss = jnp.sum((mean_x - mean_y) ** 2)

        if order >= 2:
            # Second moment (variance)
            var_x = jnp.var(x, axis=0)
            var_y = jnp.var(y, axis=0)
            loss += jnp.sum((var_x - var_y) ** 2)

        return loss

    def distillation_step(
        self,
        state: Any,
        nn_state: Any,
        batch: Dict[str, Array],
        rng: PRNGKey,
        feat_fn: Callable = None,
        **kwargs
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Perform one DM distillation step via distribution matching.

        Algorithm:
            1. Extract current synthetic data from state
            2. Extract features from real data (if match_in_feature_space)
            3. Extract features from synthetic data
            4. Compute MMD between distributions
            5. Update synthetic data to minimize MMD

        Args:
            state: Current distillation state
            nn_state: Current model state for feature extraction
            batch: Real data batch
            rng: JAX random key
            feat_fn: Feature extraction function
            **kwargs: Additional parameters

        Returns:
            Tuple of (new_state, metrics)
        """
        def loss_fn(params):
            """Compute distribution matching loss."""
            # Get current synthetic data
            x_syn, y_syn = state.apply_fn({'params': params})

            # Extract features
            if self.match_in_feature_space and feat_fn is not None:
                # Match in feature space
                feat_real = feat_fn(batch['image'], nn_state)
                feat_syn = feat_fn(x_syn, nn_state)
            else:
                # Match in pixel space
                feat_real = batch['image'].reshape(batch['image'].shape[0], -1)
                feat_syn = x_syn.reshape(x_syn.shape[0], -1)

            # Compute distribution matching loss
            if self.matching_type == 'mmd':
                dm_loss = self.compute_mmd(
                    feat_real,
                    feat_syn,
                    kernel_type=self.mmd_kernel,
                    bandwidth=self.kernel_bandwidth
                )
            elif self.matching_type == 'moment':
                dm_loss = self.compute_moment_matching(feat_real, feat_syn)
            else:
                raise ValueError(f'Unknown matching type: {self.matching_type}')

            return dm_loss

        # Compute loss and gradients for synthetic data
        loss_value, grads = jax.value_and_grad(loss_fn)(state.params)

        # Update synthetic data
        new_state = state.apply_gradients(grads=grads)

        # Compute metrics
        metrics = {
            'dm_loss': loss_value,
        }

        return new_state, metrics

    def train_and_evaluate(
        self,
        config: Any,
        dataset: Tuple[Any, Any],
        workdir: str,
        seed: int = 0,
        ds_train_raw: Any = None,
        create_online_state: Callable = None,
        create_eval_state: Callable = None,
        diff_aug: Callable = None,
        num_train_steps: int = 10000,
        num_online_eval_updates: int = 1000,
        steps_per_epoch: int = None,
        steps_per_log: int = 500,
        steps_per_eval: int = 5000,
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
        Main DM training and evaluation loop.

        Args:
            config: Configuration object
            dataset: Tuple of (ds_train, ds_test)
            workdir: Directory for checkpoints and logs
            seed: Random seed
            create_online_state: Function to create model state
            create_eval_state: Function to create evaluation model
            diff_aug: Data augmentation function
            num_train_steps: Total training steps
            num_online_eval_updates: Steps to train eval models
            steps_per_epoch: Steps per epoch
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
            Final distillation state
        """
        rng = jax.random.PRNGKey(seed)
        ds_train, ds_test = dataset

        # Initialize synthetic data
        num_classes = config.dataset.num_classes
        num_prototypes_per_class = config.kernel.num_prototypes // num_classes

        # Use untransformed dataset if provided (for init_proto which needs integer labels)
        ds_for_init = ds_train_raw if ds_train_raw is not None else ds_train

        logging.info(f'Initializing {config.kernel.num_prototypes} synthetic samples...')
        x_proto, y_proto = self.initialize_synthetic_data(
            ds=ds_for_init,
            num_prototypes_per_class=num_prototypes_per_class,
            num_classes=num_classes,
            seed=seed,
            scale_y=True
        )

        # Create learning rate schedule
        learning_rate_fn = self.cosine_decay_schedule(
            init_value=config.kernel.learning_rate,
            decay_steps=num_train_steps,
            alpha=0.0
        )

        # Create distillation state
        rng, state_rng = jax.random.split(rng)
        state = self.create_distillation_state(
            rng=state_rng,
            x_proto_init=x_proto,
            y_proto_init=y_proto,
            learning_rate_fn=learning_rate_fn,
            optimizer=config.kernel.optimizer
        )

        # Create model for feature extraction
        rng, nn_rng = jax.random.split(rng)
        nn_state = create_online_state(nn_rng)

        # Feature extraction function
        feat_fn = partial(nn_feat_fn, has_bn=has_bn, use_ema=False)

        # JIT compile training step
        from ..training.utils import train_step as generic_train_step
        from ..training.metrics import soft_cross_entropy_loss
        # has_feat=True because online model uses output='feat_fc'
        jit_nn_train_step = jax.jit(
            partial(generic_train_step, loss_type=soft_cross_entropy_loss, has_bn=has_bn, has_feat=True)
        )
        jit_nn_eval_step = jax.jit(
            partial(generic_train_step, train=False, loss_type=soft_cross_entropy_loss, has_bn=has_bn, has_feat=True)
        )

        # Training loop
        logging.info(f'Starting DM training for {num_train_steps} steps...')

        best_acc = 0.0
        train_iter = ds_train.as_numpy_iterator()

        for step in tqdm.tqdm(range(num_train_steps), desc='DM Training'):
            # Get batch
            batch = next(train_iter)
            img, lb = process_batch(batch, use_pmap=False)

            # Distillation step
            rng, step_rng = jax.random.split(rng)
            state, metrics = self.distillation_step(
                state=state,
                nn_state=nn_state,
                batch={'image': img, 'label': lb},
                rng=step_rng,
                feat_fn=feat_fn
            )

            # Update feature extractor periodically
            if (step + 1) % 100 == 0:
                rng, nn_rng = jax.random.split(rng)
                nn_state = create_online_state(nn_rng)

            # Logging
            if (step + 1) % steps_per_log == 0:
                if writer is not None:
                    writer.write_scalars(step + 1, {f'train/{k}': v for k, v in metrics.items()})
                    writer.flush()
                logging.info(f'Step {step + 1}: {metrics}')

            # Evaluation
            if (step + 1) % steps_per_eval == 0 or step == num_train_steps - 1:
                logging.info(f'Evaluating at step {step + 1}...')
                x_syn, y_syn = self.get_synthetic_data(state)

                # Evaluate on multiple random models
                eval_results = self.evaluate_synthetic_data(
                    x_syn=x_syn,
                    y_syn=y_syn,
                    ds_test=ds_test,
                    create_eval_state=create_eval_state,
                    jit_nn_train_step=jit_nn_train_step,
                    jit_nn_eval_step=jit_nn_eval_step,
                    rng=rng,
                    num_online_eval_updates=num_online_eval_updates,
                    num_eval=num_eval,
                    diff_aug=diff_aug
                )

                acc_mean = eval_results['accuracy_mean']
                acc_std = eval_results['accuracy_std']

                logging.info(f'Step {step + 1} - Accuracy: {acc_mean:.2f} ± {acc_std:.2f}')

                if writer is not None:
                    writer.write_scalars(step + 1, {
                        'eval/accuracy_mean': acc_mean,
                        'eval/accuracy_std': acc_std
                    })
                    writer.flush()

                if acc_mean > best_acc:
                    best_acc = acc_mean
                    state = dataclasses.replace(state, best_val_acc=best_acc)

            # Save checkpoint
            if save_ckpt and (step + 1) % steps_per_checkpoint == 0:
                self.save_checkpoint(state, workdir, step + 1)

            # Save images
            if image_saver and steps_per_save_image and (step + 1) % steps_per_save_image == 0:
                x_syn, y_syn = self.get_synthetic_data(state)
                image_saver(proto_state=state, step=step + 1)

        logging.info(f'DM training finished! Best accuracy: {best_acc:.2f}')
        return state

    def get_synthetic_data(
        self,
        state: Any,
        use_pmap: bool = False,
        **kwargs
    ) -> Tuple[Array, Array]:
        """
        Extract synthetic data from state.

        Args:
            state: Distillation state
            use_pmap: Whether pmap was used
            **kwargs: Additional parameters

        Returns:
            Tuple of (x_syn, y_syn)
        """
        from .frepo import get_proto
        return get_proto(state, use_pmap=use_pmap)

    def __repr__(self):
        """String representation."""
        return (
            f"DMMethod("
            f"matching_type={self.matching_type}, "
            f"mmd_kernel={self.mmd_kernel}, "
            f"kernel_bandwidth={self.kernel_bandwidth}, "
            f"learn_label={self.learn_label})"
        )
