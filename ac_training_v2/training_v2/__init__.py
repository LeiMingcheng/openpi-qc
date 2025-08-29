"""
AC Training v2 Training Module  

优化的训练组件，基于OpenPI模式重构，保持完整RL功能
"""

from .train_state import ACRLPDTrainState, init_train_state, compute_losses_and_gradients, apply_gradients
from .training_loop import ACRLPDTrainer, ACRLPDTrainingConfig
from .optimizer import create_optimizer, create_lr_scheduler, OptimizerConfig, LRSchedulerConfig

__all__ = [
    "ACRLPDTrainState",
    "init_train_state", 
    "compute_losses_and_gradients",
    "apply_gradients",
    "ACRLPDTrainer",
    "ACRLPDTrainingConfig", 
    "create_optimizer",
    "create_lr_scheduler",
    "OptimizerConfig",
    "LRSchedulerConfig",
]