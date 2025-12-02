"""
Kernel Inducing Points (KIP) method via Neural Tangent Kernel.

This module implements the KIP algorithm which learns synthetic data by matching
the Neural Tangent Kernel (NTK) between real and synthetic data.

Reference:
    Dataset Meta-Learning from Kernel Ridge-Regression
    Nguyen et al., ICLR 2021
    https://arxiv.org/abs/2011.00050

Key idea:
    In the infinite-width limit, neural network training can be described by
    the Neural Tangent Kernel. By matching NTK properties, we can find synthetic
    data that produces similar training dynamics to the full dataset.
"""

from typing import Any, Tuple, Callable, Dict, Optional
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


# Try to import neural_tangents library
try:
    import neural_tangents as nt
    HAS_NEURAL_TANGENTS = True
except ImportError:
    HAS_NEURAL_TANGENTS = False
    logging.warning(
        "neural_tangents library not found. "
        "KIP will use empirical NTK computation (slower but functional). "
        "Install with: pip install neural-tangents"
    )


@DistillationMethodRegistry.register('kip')
class KIPMethod(BaseDistillationMethod):
    """
    Kernel Inducing Points (KIP) via Neural Tangent Kernel matching.

    This method learns synthetic data by matching the Neural Tangent Kernel
    between real and synthetic data. The NTK describes the training dynamics
    of infinitely-wide neural networks, and matching it ensures similar behavior.

    Algorithm:
        1. Compute NTK for real data: K_real = NTK(x_real, x_real)
        2. Compute NTK for synthetic data: K_syn = NTK(x_syn, x_syn)
        3. Compute cross-kernel: K_cross = NTK(x_real, x_syn)
        4. Optimize synthetic data to maximize kernel alignment

    Configuration parameters:
        use_ntk: Use proper NTK vs empirical kernel (default: True if neural_tangents available)
        kernel_reg: Regularization for kernel (default: 1e-6)
        learn_label: Whether to learn labels (default: True)
        kernel_alignment_weight: Weight for alignment loss (default: 1.0)
        label_alignment_weight: Weight for label alignment (default: 0.1)
    """

    def __init__(
        self,
        use_ntk: bool = None,
        kernel_reg: float = 1e-6,
        learn_label: bool = True,
        kernel_alignment_weight: float = 1.0,
        label_alignment_weight: float = 0.1,
        **kwargs
    ):
        """
        Initialize KIP method.

        Args:
            use_ntk: Use NTK (if None, auto-detect neural_tangents)
            kernel_reg: Kernel regularization
            learn_label: Whether to optimize labels
            kernel_alignment_weight: Weight for kernel alignment
            label_alignment_weight: Weight for label matching
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)

        # Auto-detect neural_tangents availability
        if use_ntk is None:
            use_ntk = HAS_NEURAL_TANGENTS

        if use_ntk and not HAS_NEURAL_TANGENTS:
            logging.warning(
                "use_ntk=True but neural_tangents not available. "
                "Falling back to empirical kernel."
            )
            use_ntk = False

        self.use_ntk = use_ntk
        self.kernel_reg = kernel_reg
        self.learn_label = learn_label
        self.kernel_alignment_weight = kernel_alignment_weight
        self.label_alignment_weight = label_alignment_weight

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

    def compute_empirical_ntk(
        self,
        nn_state: Any,
        x1: Array,
        x2: Optional[Array] = None
    ) -> Array:
        """
        Compute empirical Neural Tangent Kernel.

        NTK(x1, x2) = J(x1)^T @ J(x2)
        where J is the Jacobian of the network output w.r.t. parameters.

        Args:
            nn_state: Neural network state
            x1: First set of inputs [N, ...]
            x2: Second set of inputs [M, ...] (if None, use x1)

        Returns:
            Kernel matrix [N, M]
        """
        if x2 is None:
            x2 = x1

        # Function to compute network output
        def net_fn(params, x):
            # Build variables dict - include batch_stats if available
            variables = {'params': params}
            if hasattr(nn_state, 'batch_stats') and nn_state.batch_stats is not None:
                variables['batch_stats'] = nn_state.batch_stats

            # Don't use mutable parameter when we don't need mutability
            out = nn_state.apply_fn(variables, x, train=False)

            # Handle different output formats (e.g., when output='feat_fc')
            if isinstance(out, tuple):
                out = out[0]  # Take logits if tuple

            return out

        # Compute Jacobians
        # For each input, compute jacobian w.r.t all parameters
        jac_fn = jax.jacobian(net_fn, argnums=0)

        # Vectorize over batch dimension with chunking to avoid OOM
        def batched_jac(x, chunk_size=8):
            """Compute jacobian for a batch of inputs using chunked computation."""
            n = x.shape[0]
            jac_list = []

            # Process in small chunks to avoid OOM
            for i in range(0, n, chunk_size):
                chunk = x[i:i+chunk_size]
                # jac will have shape [chunk_size, output_dim, param_tree]
                # Remove batch dimension from each parameter's jacobian using tree_map
                jac_chunk = jax.vmap(lambda xi: jax.tree_util.tree_map(lambda j: j[0], jac_fn(nn_state.params, xi[None])))(chunk)
                # Flatten jacobian to [chunk_size, output_dim * num_params]
                jac_flat_chunk = jnp.concatenate([
                    j.reshape(j.shape[0], -1)
                    for j in jax.tree_util.tree_leaves(jac_chunk)
                ], axis=-1)
                jac_list.append(jac_flat_chunk)

            # Concatenate all chunks
            return jnp.concatenate(jac_list, axis=0)

        # Compute jacobians for both inputs with chunking
        jac1 = batched_jac(x1, chunk_size=8)  # [N, D]
        jac2 = batched_jac(x2, chunk_size=8)  # [M, D]

        # Compute kernel: K = J1 @ J2^T
        # For very large jacobians, compute in chunks to avoid OOM
        n1, n2 = jac1.shape[0], jac2.shape[0]
        if n1 * n2 > 10000:  # If kernel matrix is large, compute in chunks
            kernel_rows = []
            chunk_size = 16
            for i in range(0, n1, chunk_size):
                jac1_chunk = jac1[i:i+chunk_size]
                kernel_chunk = jnp.dot(jac1_chunk, jac2.T)
                kernel_rows.append(kernel_chunk)
            kernel = jnp.concatenate(kernel_rows, axis=0)
        else:
            kernel = jnp.dot(jac1, jac2.T)  # [N, M]

        return kernel

    def compute_ntk(
        self,
        nn_state: Any,
        x1: Array,
        x2: Optional[Array] = None,
        compute_mode: str = 'auto'
    ) -> Array:
        """
        Compute Neural Tangent Kernel using neural_tangents library or empirical method.

        Args:
            nn_state: Neural network state
            x1: First set of inputs [N, ...]
            x2: Second set of inputs [M, ...] (if None, use x1)
            compute_mode: 'ntk' (use neural_tangents), 'empirical', or 'auto'

        Returns:
            Kernel matrix [N, M]
        """
        if compute_mode == 'auto':
            compute_mode = 'ntk' if self.use_ntk else 'empirical'

        if compute_mode == 'ntk' and HAS_NEURAL_TANGENTS:
            # Use neural_tangents library for exact NTK
            # Note: This requires the model to be a pure function
            # For simplicity, we'll use empirical NTK in this implementation
            logging.info("Using empirical NTK (neural_tangents integration not fully implemented)")
            return self.compute_empirical_ntk(nn_state, x1, x2)
        else:
            # Use empirical NTK
            return self.compute_empirical_ntk(nn_state, x1, x2)

    def kernel_alignment_loss(
        self,
        K_real: Array,
        K_syn: Array,
        K_cross: Optional[Array] = None,
        reg: float = 1e-6
    ) -> float:
        """
        Compute kernel alignment loss.

        The goal is to find synthetic data such that:
        - K_syn approximates the structure of K_real
        - The cross-kernel K_cross has high alignment

        Loss = ||K_real^{-1} @ K_cross - I||²

        Args:
            K_real: NTK on real data [N, N]
            K_syn: NTK on synthetic data [M, M]
            K_cross: Cross NTK [N, M] (if None, computed internally)
            reg: Regularization

        Returns:
            Alignment loss (scalar)
        """
        n = K_real.shape[0]
        m = K_syn.shape[0]

        # Regularize kernels
        K_real_reg = K_real + reg * jnp.trace(K_real) * jnp.eye(n) / n
        K_syn_reg = K_syn + reg * jnp.trace(K_syn) * jnp.eye(m) / m

        if K_cross is not None:
            # Alignment loss: minimize ||K_real^{-1} @ K_cross - optimal_weights||²
            # For regression, optimal is: alpha = K_real^{-1} @ y_real
            # We want K_syn @ beta = K_cross^T @ alpha
            # Simplified: minimize reconstruction error

            # Solve K_real @ alpha = K_cross (find weights for real data)
            alpha = jax.scipy.linalg.solve(K_real_reg, K_cross, assume_a='pos')

            # Reconstruction: K_cross^T @ K_real^{-1} @ K_cross ≈ K_syn
            reconstructed = jnp.dot(K_cross.T, alpha)

            # Loss: ||reconstructed - K_syn||²
            loss = jnp.mean((reconstructed - K_syn) ** 2)
        else:
            # Simple kernel matching: ||K_real - K_syn||²
            # Need to handle different sizes - use Frobenius norm
            min_size = min(n, m)
            loss = jnp.mean((K_real[:min_size, :min_size] - K_syn[:min_size, :min_size]) ** 2)

        return loss

    def distillation_step(
        self,
        state: Any,
        nn_state: Any,
        batch: Dict[str, Array],
        rng: PRNGKey,
        **kwargs
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Perform one KIP distillation step via NTK matching.

        Algorithm:
            1. Extract current synthetic data
            2. Compute NTK on real data batch
            3. Compute NTK on synthetic data
            4. Compute kernel alignment loss
            5. Update synthetic data to minimize loss

        Args:
            state: Current distillation state
            nn_state: Neural network state
            batch: Real data batch
            rng: JAX random key
            **kwargs: Additional parameters

        Returns:
            Tuple of (new_state, metrics)
        """
        # FIX: Sample real data OUTSIDE loss_fn to avoid gradient issues
        # Split RNG for sampling
        rng, sample_rng = jax.random.split(rng)

        # Pre-sample real data to reduce memory (real dataset can be very large)
        max_samples = 64  # Reduced from 256 to avoid OOM during NTK computation
        x_real = batch['image']
        y_real = batch['label']

        if x_real.shape[0] > max_samples:
            # Randomly sample real data (before loss_fn)
            real_indices = jax.random.choice(sample_rng, x_real.shape[0], shape=(max_samples,), replace=False)
            x_real_sampled = x_real[real_indices]
            y_real_sampled = y_real[real_indices]
        else:
            x_real_sampled = x_real
            y_real_sampled = y_real

        def loss_fn(params):
            """Compute KIP loss."""
            # Get current synthetic data
            x_syn, y_syn = state.apply_fn({'params': params})

            # Use ALL synthetic data (no sampling needed)
            # For typical IPC values (1-50), synthetic dataset is small (<500 images)
            # No need to subsample - use full synthetic dataset for better gradient signal
            x_syn_sample = x_syn
            y_syn_sample = y_syn

            # Compute NTK matrices using pre-sampled real data (from closure)
            K_real = self.compute_ntk(nn_state, x_real_sampled, x_real_sampled)
            K_syn = self.compute_ntk(nn_state, x_syn_sample, x_syn_sample)
            K_cross = self.compute_ntk(nn_state, x_real_sampled, x_syn_sample)

            # Kernel alignment loss
            kernel_loss = self.kernel_alignment_loss(K_real, K_syn, K_cross, self.kernel_reg)

            # Optional: Add label alignment loss
            # Encourage synthetic labels to span the label space
            label_loss = 0.0
            if self.learn_label and self.label_alignment_weight > 0:
                # Mean of labels should be similar (using pre-sampled real data)
                label_mean_real = jnp.mean(y_real_sampled, axis=0)
                label_mean_syn = jnp.mean(y_syn, axis=0)
                label_loss = jnp.mean((label_mean_real - label_mean_syn) ** 2)

            # Total loss
            total_loss = (
                self.kernel_alignment_weight * kernel_loss +
                self.label_alignment_weight * label_loss
            )

            return total_loss, (kernel_loss, label_loss)

        # Compute loss and gradients
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (total_loss, (kernel_loss, label_loss)), grads = grad_fn(state.params)

        # Update synthetic data
        new_state = state.apply_gradients(grads=grads)

        # Metrics
        metrics = {
            'total_loss': total_loss,
            'kernel_loss': kernel_loss,
            'label_loss': label_loss,
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
        Main KIP training and evaluation loop.

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

        # Create model for NTK computation
        rng, nn_rng = jax.random.split(rng)
        nn_state = create_online_state(nn_rng)

        # JIT compile training step
        from ..training.utils import train_step as generic_train_step
        from ..training.utils import eval_step as generic_eval_step
        from ..training.metrics import soft_cross_entropy_loss
        # has_feat=True because online model uses output='feat_fc'
        jit_nn_train_step = jax.jit(
            partial(generic_train_step, loss_type=soft_cross_entropy_loss, has_bn=has_bn, has_feat=True)
        )
        jit_nn_eval_step = jax.jit(
            partial(generic_eval_step, loss_type=soft_cross_entropy_loss, has_bn=False, has_feat=True, use_ema=False)
        )

        # Training loop
        logging.info(f'Starting KIP training for {num_train_steps} steps...')
        if self.use_ntk:
            logging.info('Using Neural Tangent Kernel (empirical computation)')
        else:
            logging.info('Using empirical kernel')

        best_acc = 0.0
        train_iter = ds_train.as_numpy_iterator()

        for step in tqdm.tqdm(range(num_train_steps), desc='KIP Training'):
            # Get batch
            batch = next(train_iter)
            img, lb = process_batch(batch, use_pmap=False)

            # Distillation step
            rng, step_rng = jax.random.split(rng)
            state, metrics = self.distillation_step(
                state=state,
                nn_state=nn_state,
                batch={'image': img, 'label': lb},
                rng=step_rng
            )

            # Update model periodically
            if (step + 1) % 200 == 0:
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

        logging.info(f'KIP training finished! Best accuracy: {best_acc:.2f}')
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
            f"KIPMethod("
            f"use_ntk={self.use_ntk}, "
            f"kernel_reg={self.kernel_reg}, "
            f"learn_label={self.learn_label})"
        )
