"""
Dataset Condensation (DC) method via gradient matching.

This module implements the Dataset Condensation algorithm which learns synthetic
data by matching gradients between real and synthetic data batches.

Reference:
    Dataset Condensation with Gradient Matching
    Zhao et al., ICLR 2021
    https://arxiv.org/abs/2006.05929

Key idea:
    Minimize the distance between gradients computed on real data and synthetic data:
    Loss = ||∇θ_real - ∇θ_synthetic||²
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
from ..training.metrics import get_metrics, cross_entropy_loss, soft_cross_entropy_loss

Array = Any
PRNGKey = Any


@DistillationMethodRegistry.register('dc')
class DCMethod(BaseDistillationMethod):
    """
    Dataset Condensation via gradient matching.

    This method learns synthetic data by matching the gradients computed on
    real data with gradients computed on synthetic data. The key insight is
    that if synthetic data produces similar gradients to real data, a model
    trained on synthetic data should perform similarly to one trained on real data.

    Algorithm:
        1. Sample a batch of real data
        2. Compute gradients ∇θ_real on real data
        3. Compute gradients ∇θ_synthetic on current synthetic data
        4. Update synthetic data to minimize ||∇θ_real - ∇θ_synthetic||²

    Configuration parameters:
        num_gradient_steps: Number of inner loop gradient steps (default: 1)
        distance_metric: Metric for gradient distance ('mse', 'cosine', 'l1')
        gradient_matching_layers: Which layers to match ('all', 'last', 'first_last')
        learn_label: Whether to learn labels (default: True)
        model_lr: Learning rate for model in inner loop (default: 0.01)
    """

    def __init__(
        self,
        num_gradient_steps: int = 1,
        distance_metric: str = 'mse',
        gradient_matching_layers: str = 'all',
        learn_label: bool = True,
        model_lr: float = 0.01,
        **kwargs
    ):
        """
        Initialize DC method.

        Args:
            num_gradient_steps: Number of gradient steps for inner loop
            distance_metric: Distance metric ('mse', 'cosine', 'l1')
            gradient_matching_layers: Layers to match ('all', 'last', 'first_last')
            learn_label: Whether to optimize labels
            model_lr: Learning rate for model updates
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.num_gradient_steps = num_gradient_steps
        self.distance_metric = distance_metric
        self.gradient_matching_layers = gradient_matching_layers
        self.learn_label = learn_label
        self.model_lr = model_lr

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

        Uses the same initialization as FRePo: sample real images from each class.

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
            optimizer: Optimizer name (default: 'adam' for DC)
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

    def compute_gradient_real(
        self,
        nn_state: Any,
        batch: Dict[str, Array],
        loss_fn: Callable = None
    ) -> Dict[str, Array]:
        """
        Compute gradients on real data batch.

        Args:
            nn_state: Current model state
            batch: Real data batch {'image': ..., 'label': ...}
            loss_fn: Loss function (default: cross entropy)

        Returns:
            Dictionary of gradients for each parameter
        """
        if loss_fn is None:
            loss_fn = soft_cross_entropy_loss

        def forward_loss(params):
            """Forward pass and compute loss."""
            if hasattr(nn_state, 'batch_stats'):
                variables = {'params': params, 'batch_stats': nn_state.batch_stats}
                logits = nn_state.apply_fn(variables, batch['image'], train=True, mutable=False)[0]
            else:
                variables = {'params': params}
                logits = nn_state.apply_fn(variables, batch['image'], train=True, mutable=False)

            loss = loss_fn(logits, batch['label']).mean()
            return loss

        grads = jax.grad(forward_loss)(nn_state.params)
        return grads

    def compute_gradient_synthetic(
        self,
        nn_state: Any,
        x_syn: Array,
        y_syn: Array,
        loss_fn: Callable = None
    ) -> Dict[str, Array]:
        """
        Compute gradients on synthetic data.

        Args:
            nn_state: Current model state
            x_syn: Synthetic images
            y_syn: Synthetic labels
            loss_fn: Loss function (default: cross entropy)

        Returns:
            Dictionary of gradients for each parameter
        """
        if loss_fn is None:
            loss_fn = soft_cross_entropy_loss

        def forward_loss(params):
            """Forward pass and compute loss."""
            if hasattr(nn_state, 'batch_stats'):
                variables = {'params': params, 'batch_stats': nn_state.batch_stats}
                logits = nn_state.apply_fn(variables, x_syn, train=True, mutable=False)[0]
            else:
                variables = {'params': params}
                logits = nn_state.apply_fn(variables, x_syn, train=True, mutable=False)

            loss = loss_fn(logits, y_syn).mean()
            return loss

        grads = jax.grad(forward_loss)(nn_state.params)
        return grads

    def gradient_matching_loss(
        self,
        grad_real: Dict[str, Array],
        grad_syn: Dict[str, Array],
        metric: str = 'mse'
    ) -> float:
        """
        Compute distance between real and synthetic gradients.

        Args:
            grad_real: Gradients from real data
            grad_syn: Gradients from synthetic data
            metric: Distance metric ('mse', 'cosine', 'l1')

        Returns:
            Scalar loss value
        """
        # Flatten all gradients
        grad_real_flat = jnp.concatenate([g.flatten() for g in jax.tree_util.tree_leaves(grad_real)])
        grad_syn_flat = jnp.concatenate([g.flatten() for g in jax.tree_util.tree_leaves(grad_syn)])

        if metric == 'mse':
            # Mean squared error
            loss = jnp.mean((grad_real_flat - grad_syn_flat) ** 2)
        elif metric == 'cosine':
            # Cosine distance (1 - cosine similarity)
            dot_product = jnp.dot(grad_real_flat, grad_syn_flat)
            norm_real = jnp.linalg.norm(grad_real_flat)
            norm_syn = jnp.linalg.norm(grad_syn_flat)
            cosine_sim = dot_product / (norm_real * norm_syn + 1e-8)
            loss = 1.0 - cosine_sim
        elif metric == 'l1':
            # L1 distance
            loss = jnp.mean(jnp.abs(grad_real_flat - grad_syn_flat))
        else:
            raise ValueError(f'Unknown distance metric: {metric}')

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
        Perform one DC distillation step via gradient matching.

        Algorithm:
            1. Extract current synthetic data from state
            2. Compute gradients on real data batch
            3. Compute gradients on synthetic data
            4. Compute gradient matching loss
            5. Update synthetic data to minimize the loss

        Args:
            state: Current distillation state
            nn_state: Current model state
            batch: Real data batch
            rng: JAX random key
            **kwargs: Additional parameters

        Returns:
            Tuple of (new_state, metrics)
        """
        def loss_fn(params):
            """Compute gradient matching loss for synthetic data optimization."""
            # Get current synthetic data
            x_syn, y_syn = state.apply_fn({'params': params})

            # Compute gradients on real data
            grad_real = self.compute_gradient_real(nn_state, batch)

            # Compute gradients on synthetic data
            grad_syn = self.compute_gradient_synthetic(nn_state, x_syn, y_syn)

            # Compute gradient matching loss
            gm_loss = self.gradient_matching_loss(grad_real, grad_syn, self.distance_metric)

            return gm_loss, (grad_real, grad_syn)

        # Compute loss and gradients for synthetic data
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (gm_loss, (grad_real, grad_syn)), grads = grad_fn(state.params)

        # Update synthetic data
        new_state = state.apply_gradients(grads=grads)

        # Compute metrics
        grad_real_norm = jnp.linalg.norm(
            jnp.concatenate([g.flatten() for g in jax.tree_util.tree_leaves(grad_real)])
        )
        grad_syn_norm = jnp.linalg.norm(
            jnp.concatenate([g.flatten() for g in jax.tree_util.tree_leaves(grad_syn)])
        )

        metrics = {
            'gm_loss': gm_loss,
            'grad_real_norm': grad_real_norm,
            'grad_syn_norm': grad_syn_norm,
        }

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
        Main DC training and evaluation loop.

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

        logging.info(f'Initializing {config.kernel.num_prototypes} synthetic samples...')
        x_proto, y_proto = self.initialize_synthetic_data(
            ds=ds_train,
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

        # Create model for gradient computation
        rng, nn_rng = jax.random.split(rng)
        nn_state = create_online_state(nn_rng)

        # JIT compile training step
        from ..training.utils import train_step as generic_train_step
        jit_nn_train_step = jax.jit(generic_train_step)
        jit_nn_eval_step = jax.jit(
            partial(
                generic_train_step,
                train=False
            )
        )

        # Training loop
        logging.info(f'Starting DC training for {num_train_steps} steps...')

        best_acc = 0.0
        train_iter = ds_train.as_numpy_iterator()

        for step in tqdm.tqdm(range(num_train_steps), desc='DC Training'):
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

            # Update model periodically (to avoid overfitting to one model)
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
                image_saver(state=state, step=step + 1)

        logging.info(f'DC training finished! Best accuracy: {best_acc:.2f}')
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
            f"DCMethod("
            f"num_gradient_steps={self.num_gradient_steps}, "
            f"distance_metric={self.distance_metric}, "
            f"learn_label={self.learn_label})"
        )
