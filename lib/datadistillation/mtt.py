"""
Matching Training Trajectories (MTT) for dataset distillation.

This module implements MTT which learns synthetic data by matching the training
trajectories of models trained on real vs synthetic data.

Reference:
    Dataset Distillation by Matching Training Trajectories
    Cazenavette et al., CVPR 2022
    https://arxiv.org/abs/2203.11932

Key idea:
    Instead of matching gradients at a single step, MTT matches gradients
    across multiple timesteps during training. This captures the dynamics
    of how models evolve during training on real data.
"""

from typing import Any, Tuple, Callable, Dict, List
from functools import partial
import dataclasses
import os

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
from ..training.metrics import get_metrics, soft_cross_entropy_loss

Array = Any
PRNGKey = Any


@dataclasses.dataclass
class ExpertTrajectory:
    """
    Stores expert model parameters at different training timesteps.

    Attributes:
        params_list: List of model parameters at different timesteps
        timesteps: List of timesteps corresponding to each checkpoint
        final_step: Total training steps for the trajectory
    """
    params_list: List[Any]
    timesteps: List[int]
    final_step: int


@DistillationMethodRegistry.register('mtt')
class MTTMethod(BaseDistillationMethod):
    """
    Matching Training Trajectories (MTT) for dataset distillation.

    MTT learns synthetic data by matching the gradient trajectories of models
    trained on real vs synthetic data. Unlike DC which matches gradients at
    a single initialization, MTT matches gradients at multiple points along
    the training trajectory.

    Algorithm:
        1. Collect expert trajectories:
           - Train models on real data
           - Save checkpoints at different timesteps
        2. For each distillation step:
           - Sample an expert checkpoint
           - Compute gradient on synthetic data from that checkpoint
           - Match it to the expert gradient at the next timestep
        3. Update synthetic data to minimize trajectory matching loss

    Configuration parameters:
        num_expert_trajectories: Number of expert trajectories to collect (default: 5)
        expert_steps: Training steps for each expert (default: 1000)
        trajectory_sample_interval: Steps between trajectory checkpoints (default: 100)
        trajectory_matching_steps: Number of matching steps along trajectory (default: 5)
        learn_label: Whether to learn labels (default: True)
        model_lr: Learning rate for expert models (default: 0.01)
    """

    def __init__(
        self,
        num_expert_trajectories: int = 5,
        expert_steps: int = 1000,
        trajectory_sample_interval: int = 100,
        trajectory_matching_steps: int = 5,
        learn_label: bool = True,
        model_lr: float = 0.01,
        **kwargs
    ):
        """
        Initialize MTT method.

        Args:
            num_expert_trajectories: Number of expert trajectories
            expert_steps: Steps to train each expert
            trajectory_sample_interval: Interval for saving checkpoints
            trajectory_matching_steps: Number of steps to match
            learn_label: Whether to optimize labels
            model_lr: Expert model learning rate
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.num_expert_trajectories = num_expert_trajectories
        self.expert_steps = expert_steps
        self.trajectory_sample_interval = trajectory_sample_interval
        self.trajectory_matching_steps = trajectory_matching_steps
        self.learn_label = learn_label
        self.model_lr = model_lr

        # Trajectory buffer (will be populated during training)
        self.expert_trajectories: List[ExpertTrajectory] = []

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

    def collect_expert_trajectory(
        self,
        ds_train: Any,
        create_model_state: Callable,
        jit_train_step: Callable,
        rng: PRNGKey,
        num_steps: int = None,
        sample_interval: int = None
    ) -> ExpertTrajectory:
        """
        Collect expert trajectory by training a model on real data.

        Args:
            ds_train: Training dataset
            create_model_state: Function to create model state
            jit_train_step: JIT-compiled training step
            rng: JAX random key
            num_steps: Number of training steps (default: self.expert_steps)
            sample_interval: Checkpoint interval (default: self.trajectory_sample_interval)

        Returns:
            ExpertTrajectory containing checkpoints
        """
        if num_steps is None:
            num_steps = self.expert_steps
        if sample_interval is None:
            sample_interval = self.trajectory_sample_interval

        logging.info(f'Collecting expert trajectory for {num_steps} steps...')

        # Create fresh model
        expert_state = create_model_state(rng)

        # Storage for trajectory
        params_list = []
        timesteps = []

        # Training loop
        train_iter = ds_train.as_numpy_iterator()

        # Add progress bar
        for step in tqdm.tqdm(range(num_steps), desc='Expert Training', leave=False):
            # Get batch
            batch = next(train_iter)
            img, lb = process_batch(batch, use_pmap=False)

            # Training step (first call will trigger JIT compilation)
            if step == 0:
                logging.info('Compiling training step (this may take a moment)...')

            rng, step_rng = jax.random.split(rng)
            expert_state, _ = jit_train_step(expert_state, {'image': img, 'label': lb}, step_rng)

            if step == 0:
                logging.info('Compilation complete, continuing training...')

            # Save checkpoint at intervals
            if (step + 1) % sample_interval == 0:
                params_list.append(jax.tree_util.tree_map(lambda x: x.copy(), expert_state.params))
                timesteps.append(step + 1)

        # Save final state
        if num_steps not in timesteps:
            params_list.append(jax.tree_util.tree_map(lambda x: x.copy(), expert_state.params))
            timesteps.append(num_steps)

        logging.info(f'Collected {len(params_list)} checkpoints at timesteps: {timesteps}')

        return ExpertTrajectory(
            params_list=params_list,
            timesteps=timesteps,
            final_step=num_steps
        )

    def collect_expert_trajectories(
        self,
        ds_train: Any,
        create_model_state: Callable,
        jit_train_step: Callable,
        rng: PRNGKey,
        num_trajectories: int = None
    ) -> List[ExpertTrajectory]:
        """
        Collect multiple expert trajectories with different random initializations.

        Args:
            ds_train: Training dataset
            create_model_state: Function to create model state
            jit_train_step: JIT-compiled training step
            rng: JAX random key
            num_trajectories: Number of trajectories (default: self.num_expert_trajectories)

        Returns:
            List of ExpertTrajectory objects
        """
        if num_trajectories is None:
            num_trajectories = self.num_expert_trajectories

        logging.info(f'Collecting {num_trajectories} expert trajectories...')

        trajectories = []
        for i in range(num_trajectories):
            rng, traj_rng = jax.random.split(rng)
            trajectory = self.collect_expert_trajectory(
                ds_train=ds_train,
                create_model_state=create_model_state,
                jit_train_step=jit_train_step,
                rng=traj_rng
            )
            trajectories.append(trajectory)
            logging.info(f'Collected trajectory {i + 1}/{num_trajectories}')

        return trajectories

    def compute_trajectory_gradient(
        self,
        model_state: Any,
        batch: Dict[str, Array],
        loss_fn: Callable = None
    ) -> Dict[str, Array]:
        """
        Compute gradients for trajectory matching.

        Args:
            model_state: Model state with parameters
            batch: Data batch {'image': ..., 'label': ...}
            loss_fn: Loss function (default: cross entropy)

        Returns:
            Dictionary of gradients
        """
        if loss_fn is None:
            loss_fn = soft_cross_entropy_loss

        def forward_loss(params):
            """Forward pass and compute loss."""
            if hasattr(model_state, 'batch_stats') and model_state.batch_stats is not None:
                variables = {'params': params, 'batch_stats': model_state.batch_stats}
                # Need mutable=['batch_stats'] when train=True with BatchNorm
                out, _ = model_state.apply_fn(variables, batch['image'], train=True, mutable=['batch_stats'])
                # Handle tuple output
                logits = out[0] if isinstance(out, tuple) else out
            else:
                variables = {'params': params}
                out = model_state.apply_fn(variables, batch['image'], train=True, mutable=False)
                logits = out

            loss = loss_fn(logits, batch['label']).mean()
            return loss

        grads = jax.grad(forward_loss)(model_state.params)
        return grads

    def trajectory_matching_loss(
        self,
        grad_expert: Dict[str, Array],
        grad_synthetic: Dict[str, Array]
    ) -> float:
        """
        Compute trajectory matching loss between expert and synthetic gradients.

        Args:
            grad_expert: Expert gradients
            grad_synthetic: Synthetic gradients

        Returns:
            Matching loss (MSE)
        """
        # Flatten all gradients
        grad_expert_flat = jnp.concatenate([g.flatten() for g in jax.tree_util.tree_leaves(grad_expert)])
        grad_synthetic_flat = jnp.concatenate([g.flatten() for g in jax.tree_util.tree_leaves(grad_synthetic)])

        # MSE loss
        loss = jnp.mean((grad_expert_flat - grad_synthetic_flat) ** 2)

        return loss

    def distillation_step(
        self,
        state: Any,
        nn_state: Any,
        batch: Dict[str, Array],
        rng: PRNGKey,
        expert_trajectories: List[ExpertTrajectory] = None,
        **kwargs
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Perform one MTT distillation step.

        Algorithm:
            1. Sample a random expert trajectory
            2. Sample a random checkpoint from the trajectory
            3. Create model state with those expert parameters
            4. Compute gradients on synthetic data from that model
            5. Match to expert's next-step gradient
            6. Update synthetic data

        Args:
            state: Current distillation state
            nn_state: Current model state (not used, kept for interface compatibility)
            batch: Real data batch (used to compute expert gradient)
            rng: JAX random key
            expert_trajectories: List of expert trajectories (default: self.expert_trajectories)
            **kwargs: Additional parameters

        Returns:
            Tuple of (new_state, metrics)
        """
        if expert_trajectories is None:
            expert_trajectories = self.expert_trajectories

        if len(expert_trajectories) == 0:
            raise ValueError("No expert trajectories available. Run collect_expert_trajectories first.")

        # FIX: Sample trajectory and checkpoint OUTSIDE loss_fn to avoid gradient issues
        # Split RNG for trajectory and checkpoint sampling
        rng, traj_rng, ckpt_rng = jax.random.split(rng, 3)

        # Sample random trajectory
        traj_idx = int(jax.random.choice(traj_rng, len(expert_trajectories)))
        trajectory = expert_trajectories[traj_idx]

        # Sample random checkpoint (not the last one, so we can get next step)
        if len(trajectory.params_list) < 2:
            checkpoint_idx = 0
        else:
            checkpoint_idx = int(jax.random.choice(ckpt_rng, len(trajectory.params_list) - 1))

        # Get expert parameters at this checkpoint (before loss_fn)
        expert_params = trajectory.params_list[checkpoint_idx]

        # Create temporary model state with expert parameters
        temp_model_state = dataclasses.replace(nn_state, params=expert_params)

        def loss_fn(params):
            """Compute trajectory matching loss."""
            # Get current synthetic data
            x_syn, y_syn = state.apply_fn({'params': params})

            # Compute gradient on synthetic data from expert checkpoint
            # Uses temp_model_state and batch from closure (deterministic)
            grad_synthetic = self.compute_trajectory_gradient(
                temp_model_state,
                {'image': x_syn, 'label': y_syn}
            )

            # Get expert gradient (gradient that led from checkpoint to next)
            # Approximate: use gradient computed on real batch at this checkpoint
            grad_expert = self.compute_trajectory_gradient(
                temp_model_state,
                batch
            )

            # Trajectory matching loss
            tm_loss = self.trajectory_matching_loss(grad_expert, grad_synthetic)

            return tm_loss

        # Compute loss and gradients
        loss_value, grads = jax.value_and_grad(loss_fn)(state.params)

        # Update synthetic data
        new_state = state.apply_gradients(grads=grads)

        # Metrics
        metrics = {
            'tm_loss': loss_value,
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
        Main MTT training and evaluation loop.

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

        # Create model for trajectory matching
        rng, nn_rng = jax.random.split(rng)
        nn_state = create_online_state(nn_rng)

        # JIT compile training step
        from ..training.utils import train_step as generic_train_step
        from ..training.utils import eval_step as generic_eval_step
        # has_feat=True because online model uses output='feat_fc'
        jit_nn_train_step = jax.jit(
            partial(generic_train_step, loss_type=soft_cross_entropy_loss, has_bn=has_bn, has_feat=True)
        )
        jit_nn_eval_step = jax.jit(
            partial(generic_eval_step, loss_type=soft_cross_entropy_loss, has_bn=False, has_feat=True, use_ema=False)
        )

        # Create training step for eval model (without batch norm)
        # Eval model uses identity normalization, so has_bn=False
        jit_eval_train_step = jax.jit(
            partial(generic_train_step, loss_type=soft_cross_entropy_loss, has_bn=False, has_feat=True)
        )

        # Collect expert trajectories FIRST
        logging.info('='  * 60)
        logging.info('PHASE 1: Collecting expert trajectories')
        logging.info('=' * 60)

        rng, expert_rng = jax.random.split(rng)
        self.expert_trajectories = self.collect_expert_trajectories(
            ds_train=ds_train,
            create_model_state=create_online_state,
            jit_train_step=jit_nn_train_step,
            rng=expert_rng,
            num_trajectories=self.num_expert_trajectories
        )

        logging.info('='  * 60)
        logging.info('PHASE 2: Training on synthetic data via trajectory matching')
        logging.info('=' * 60)

        # Training loop
        best_acc = 0.0
        train_iter = ds_train.as_numpy_iterator()

        for step in tqdm.tqdm(range(num_train_steps), desc='MTT Training'):
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
                expert_trajectories=self.expert_trajectories
            )

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
                    jit_nn_train_step=jit_eval_train_step,  # Use eval-specific train step
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

        logging.info(f'MTT training finished! Best accuracy: {best_acc:.2f}')
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
            f"MTTMethod("
            f"num_expert_trajectories={self.num_expert_trajectories}, "
            f"expert_steps={self.expert_steps}, "
            f"learn_label={self.learn_label})"
        )
