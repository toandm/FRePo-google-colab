"""
Base classes and interfaces for dataset distillation methods.

This module provides the abstract base class that all distillation methods
must implement, ensuring a consistent interface across different algorithms.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple, Callable, Optional, Dict
import dataclasses

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

Array = Any
PRNGKey = Any


class DistillationState(train_state.TrainState):
    """
    Extended training state for distillation methods.

    Attributes:
        epoch: Current epoch number
        best_val_acc: Best validation accuracy achieved
        method_specific: Dictionary for method-specific state variables
    """
    epoch: int = 0
    best_val_acc: float = 0.0
    method_specific: Dict[str, Any] = dataclasses.field(default_factory=dict)


class BaseDistillationMethod(ABC):
    """
    Abstract base class for dataset distillation methods.

    All distillation methods (FRePo, MTT, KIP, DC, DM) must inherit from this
    class and implement the required abstract methods. This ensures a consistent
    interface for training, evaluation, and comparison.

    The typical workflow is:
        1. Initialize synthetic data using initialize_synthetic_data()
        2. Create training state using create_distillation_state()
        3. Run training loop using train_and_evaluate()
        4. Extract final synthetic data using get_synthetic_data()
    """

    def __init__(self, **kwargs):
        """
        Initialize the distillation method.

        Args:
            **kwargs: Method-specific configuration parameters
        """
        self.config = kwargs

    @abstractmethod
    def initialize_synthetic_data(
        self,
        ds: Any,
        num_prototypes_per_class: int,
        num_classes: int,
        seed: int = 0,
        **kwargs
    ) -> Tuple[Array, Array]:
        """
        Initialize synthetic dataset (images and labels).

        Args:
            ds: Training dataset (TensorFlow dataset)
            num_prototypes_per_class: Number of synthetic images per class
            num_classes: Total number of classes
            seed: Random seed for initialization
            **kwargs: Method-specific parameters

        Returns:
            Tuple of (x_proto, y_proto):
                - x_proto: Synthetic images, shape [num_prototypes, H, W, C]
                - y_proto: Synthetic labels, shape [num_prototypes, num_classes]
        """
        pass

    @abstractmethod
    def create_distillation_state(
        self,
        rng: PRNGKey,
        x_proto_init: Array,
        y_proto_init: Array,
        learning_rate_fn: Callable,
        optimizer: str = 'lamb',
        **kwargs
    ) -> DistillationState:
        """
        Create the training state for synthetic data optimization.

        Args:
            rng: JAX random key
            x_proto_init: Initial synthetic images
            y_proto_init: Initial synthetic labels
            learning_rate_fn: Learning rate schedule function
            optimizer: Optimizer name ('lamb', 'adam', 'sgd')
            **kwargs: Method-specific parameters

        Returns:
            DistillationState containing optimizable parameters
        """
        pass

    @abstractmethod
    def distillation_step(
        self,
        state: DistillationState,
        nn_state: train_state.TrainState,
        batch: Dict[str, Array],
        rng: PRNGKey,
        **kwargs
    ) -> Tuple[DistillationState, Dict[str, float]]:
        """
        Perform one distillation training step.

        This is the core method where the synthetic data is updated based on
        the method's specific algorithm (gradient matching, distribution matching,
        trajectory matching, etc.).

        Args:
            state: Current distillation state (contains synthetic data)
            nn_state: Current neural network state (for feature extraction/training)
            batch: Batch of real data {'image': ..., 'label': ...}
            rng: JAX random key
            **kwargs: Method-specific parameters

        Returns:
            Tuple of (new_state, metrics):
                - new_state: Updated distillation state
                - metrics: Dictionary of training metrics (loss, accuracy, etc.)
        """
        pass

    @abstractmethod
    def train_and_evaluate(
        self,
        config: Any,
        dataset: Tuple[Any, Any],
        workdir: str,
        seed: int = 0,
        **kwargs
    ) -> DistillationState:
        """
        Main training loop for distillation.

        This method orchestrates the entire distillation process:
        - Initialize synthetic data
        - Create model states
        - Run training loop with distillation_step()
        - Periodic evaluation
        - Save checkpoints

        Args:
            config: Configuration object with hyperparameters
            dataset: Tuple of (ds_train, ds_test) TensorFlow datasets
            workdir: Directory for saving checkpoints and logs
            seed: Random seed
            **kwargs: Method-specific parameters

        Returns:
            Final DistillationState after training
        """
        pass

    @abstractmethod
    def get_synthetic_data(
        self,
        state: DistillationState,
        **kwargs
    ) -> Tuple[Array, Array]:
        """
        Extract synthetic data from training state.

        Args:
            state: Distillation state containing learned parameters
            **kwargs: Method-specific parameters

        Returns:
            Tuple of (x_syn, y_syn):
                - x_syn: Synthetic images
                - y_syn: Synthetic labels
        """
        pass

    # -------------------------------------------------------------------------
    # Shared utility methods (can be overridden if needed)
    # -------------------------------------------------------------------------

    def create_optimizer(
        self,
        learning_rate_fn: Callable,
        optimizer_name: str = 'lamb',
        **optimizer_kwargs
    ) -> optax.GradientTransformation:
        """
        Create optimizer for synthetic data optimization.

        Args:
            learning_rate_fn: Learning rate schedule
            optimizer_name: Name of optimizer ('lamb', 'adam', 'sgd', 'adamw')
            **optimizer_kwargs: Additional optimizer parameters

        Returns:
            Optax optimizer
        """
        if optimizer_name == 'lamb':
            return optax.lamb(learning_rate=learning_rate_fn, **optimizer_kwargs)
        elif optimizer_name == 'adam':
            return optax.adam(learning_rate=learning_rate_fn, **optimizer_kwargs)
        elif optimizer_name == 'sgd':
            return optax.sgd(learning_rate=learning_rate_fn, **optimizer_kwargs)
        elif optimizer_name == 'adamw':
            return optax.adamw(learning_rate=learning_rate_fn, **optimizer_kwargs)
        else:
            raise ValueError(f'Unknown optimizer: {optimizer_name}')

    def cosine_decay_schedule(
        self,
        init_value: float,
        decay_steps: int,
        alpha: float = 0.0
    ) -> Callable:
        """
        Create cosine decay learning rate schedule.

        Args:
            init_value: Initial learning rate
            decay_steps: Number of steps for decay
            alpha: Minimum learning rate (as fraction of init_value)

        Returns:
            Learning rate schedule function
        """
        return optax.cosine_decay_schedule(
            init_value=init_value,
            decay_steps=decay_steps,
            alpha=alpha
        )

    def evaluate_synthetic_data(
        self,
        x_syn: Array,
        y_syn: Array,
        ds_test: Any,
        create_eval_state: Callable,
        jit_nn_train_step: Callable,
        jit_nn_eval_step: Callable,
        rng: PRNGKey,
        num_online_eval_updates: int = 1000,
        num_eval: int = 5,
        diff_aug: Optional[Callable] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate quality of synthetic data.

        This is a shared evaluation protocol used by all methods:
        1. Create fresh random models
        2. Train them on synthetic data
        3. Evaluate on real test set
        4. Report mean and std accuracy

        Args:
            x_syn: Synthetic images
            y_syn: Synthetic labels
            ds_test: Test dataset
            create_eval_state: Function to create fresh model state
            jit_nn_train_step: JIT compiled training step
            jit_nn_eval_step: JIT compiled evaluation step
            rng: JAX random key
            num_online_eval_updates: Steps to train on synthetic data
            num_eval: Number of random models to evaluate
            diff_aug: Augmentation function
            **kwargs: Additional parameters

        Returns:
            Dictionary with 'accuracy_mean' and 'accuracy_std'
        """
        import tensorflow as tf
        from ..training.utils import process_batch
        from ..training.metrics import get_metrics

        # Import evaluation functions from frepo
        from .frepo import train_on_proto, eval_on_proto_nn

        accuracies = []

        for i in range(num_eval):
            rng, eval_rng = jax.random.split(rng)

            # Create fresh model
            nn_state_eval = create_eval_state(eval_rng)

            # Create dataset from synthetic data
            ds_proto = tf.data.Dataset.from_tensor_slices((x_syn, y_syn))
            ds_proto = ds_proto.cache().repeat().shuffle(buffer_size=5000)
            ds_proto = ds_proto.batch(batch_size=min(y_syn.shape[0], 500))

            # Train on synthetic data
            nn_state_eval = train_on_proto(
                ds_proto,
                nn_state_eval,
                jit_nn_train_step,
                diff_aug if diff_aug else lambda r, x: x,
                eval_rng,
                num_updates=num_online_eval_updates,
                has_bn=False
            )

            # Evaluate on test set
            summary = eval_on_proto_nn(ds_test, nn_state_eval, jit_nn_eval_step)
            accuracies.append(summary['accuracy'] * 100)

        # Convert to Python float for TensorBoard compatibility
        # JAX arrays are logged as tensors instead of scalars in TensorBoard
        return {
            'accuracy_mean': float(jnp.mean(jnp.array(accuracies))),
            'accuracy_std': float(jnp.std(jnp.array(accuracies)))
        }

    def save_checkpoint(
        self,
        state: DistillationState,
        workdir: str,
        step: int,
        prefix: str = 'ckpt',
        keep: int = 3
    ):
        """
        Save checkpoint.

        Args:
            state: State to save
            workdir: Directory to save checkpoint
            step: Current step number
            prefix: Checkpoint prefix
            keep: Number of checkpoints to keep
        """
        from flax.training import checkpoints
        import os

        # Convert to absolute path for Orbax compatibility
        ckpt_dir = os.path.abspath(os.path.join(workdir, prefix))
        checkpoints.save_checkpoint(ckpt_dir, state, step, keep=keep)

    def restore_checkpoint(
        self,
        state: DistillationState,
        workdir: str,
        prefix: str = 'ckpt'
    ) -> DistillationState:
        """
        Restore checkpoint.

        Args:
            state: Template state
            workdir: Directory containing checkpoint
            prefix: Checkpoint prefix

        Returns:
            Restored state
        """
        from flax.training import checkpoints
        import os

        # Convert to absolute path for Orbax compatibility
        ckpt_dir = os.path.abspath(os.path.join(workdir, prefix))
        return checkpoints.restore_checkpoint(ckpt_dir, state)

    def __repr__(self):
        """String representation of the method."""
        return f"{self.__class__.__name__}(config={self.config})"
