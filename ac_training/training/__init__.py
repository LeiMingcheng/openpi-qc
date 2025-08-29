"""
ACRLPD + π₀ Training System.

This module provides the complete training infrastructure for ACRLPD + π₀ agents,
including offline pretraining, online fine-tuning, evaluation, and monitoring.

Key components:
- ACRLPDTrainer: Complete training system
- TrainingConfig: Training configuration management
- TrainingMetrics: Metrics tracking and monitoring
- CheckpointManager: Model persistence and recovery

Usage:
    from ac_training.training import ACRLPDTrainer, TrainingConfig
    
    # Create trainer
    trainer = ACRLPDTrainer(agent, dataloader, config, eval_fn)
    
    # Run training
    trained_agent = trainer.train(resume_from=checkpoint_path)
"""

from .training_loop import (
    ACRLPDTrainer,
    ACRLPDTrainingConfig as TrainingConfig,
    TrainingMetrics,
    ACRLPDCheckpointManager as CheckpointManager,
)

__all__ = [
    "ACRLPDTrainer",
    "TrainingConfig", 
    "TrainingMetrics",
    "CheckpointManager",
]

# Version information
__version__ = "0.1.0"
__author__ = "OpenPI ACRLPD Integration Team"

__doc__ += f"""

Version: {__version__}
Author: {__author__}

Training Pipeline:
1. Offline Pretraining:
   - Load robotic demonstration data
   - Train with combined BC + Critic losses
   - Build strong initialization for online phase

2. Online Fine-tuning:
   - Environment interaction with Best-of-N sampling
   - Adaptive exploration with temperature control
   - Early stopping based on evaluation performance

3. Evaluation & Monitoring:
   - Periodic evaluation on hold-out tasks
   - WandB integration for experiment tracking
   - Comprehensive metrics and visualization

4. Checkpointing:
   - Automatic checkpoint saving
   - Top-K checkpoint retention
   - Resume from checkpoint functionality

Features:
- Multi-phase training (offline → online)
- Comprehensive evaluation system
- WandB experiment tracking
- Checkpoint management and recovery
- Performance profiling and optimization
- Early stopping and hyperparameter scheduling
"""