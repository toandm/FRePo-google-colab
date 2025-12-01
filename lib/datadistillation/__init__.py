"""
Dataset distillation methods.

This package provides various dataset distillation algorithms including:
- FRePo: Feature Regression with Pooling
- MTT: Matching Training Trajectories (to be implemented)
- KIP: Kernel Inducing Points (to be implemented)
- DC: Dataset Condensation (to be implemented)
- DM: Distribution Matching (to be implemented)
"""

# Base classes and registry
from .base import BaseDistillationMethod, DistillationState
from .registry import DistillationMethodRegistry, register_method

# FRePo original exports
from .frepo import (
    ProtoHolder,
    ProtoState,
    init_proto,
    create_proto_state,
    load_proto_state,
    nfr,
    nn_feat_fn,
    proto_train_step,
    proto_eval_step,
    proto_train_and_evaluate,
    proto_evaluate,
    get_proto,
    get_sample_proto,
)

# FRePo adapter (new interface)
from .frepo_adapter import FRePoMethod

# DC method (Dataset Condensation)
from .dc import DCMethod

# DM method (Distribution Matching)
from .dm import DMMethod

# KIP method (Kernel Inducing Points)
from .kip import KIPMethod

# MTT method (Matching Training Trajectories)
from .mtt import MTTMethod

__all__ = [
    # Base classes
    'BaseDistillationMethod',
    'DistillationState',
    # Registry
    'DistillationMethodRegistry',
    'register_method',
    # FRePo original
    'ProtoHolder',
    'ProtoState',
    'init_proto',
    'create_proto_state',
    'load_proto_state',
    'nfr',
    'nn_feat_fn',
    'proto_train_step',
    'proto_eval_step',
    'proto_train_and_evaluate',
    'proto_evaluate',
    'get_proto',
    'get_sample_proto',
    # FRePo adapter
    'FRePoMethod',
    # DC method
    'DCMethod',
    # DM method
    'DMMethod',
    # KIP method
    'KIPMethod',
    # MTT method
    'MTTMethod',
]

# Print available methods when module is imported (optional, for debugging)
def list_available_methods():
    """List all registered distillation methods."""
    return DistillationMethodRegistry.list_methods()
