"""
ACRLPD + π₀ Agent Integration Module.

This module provides the complete ACRLPD (Action-Chunked Reinforcement Learning 
with Prior Data) integration with π₀ diffusion models for robotic manipulation.

Key components:
- ACRLPDPi0Agent: Main agent class integrating Q-chunking RL with π₀
- CriticNetworks: Ensemble of Q-networks for value estimation
- Joint loss functions: Combined training objectives
- Configuration management: Platform-specific configs

Usage:
    from ac_training.agents import create_acrlpd_pi0_agent_from_rl_config
    from ac_training.config import get_config
    
    # Create agent with unified configuration
    rl_config = get_config('rl_fold_box')  # or rl_aloha_fold, rl_libero, rl_droid
    agent = create_acrlpd_pi0_agent_from_rl_config(rl_config, rng, lazy_init=True)
    
    # Setup from FSDP state (if using lazy_init)
    agent.setup_from_fsdp_state(fsdp_train_state)
"""

from .acrlpd_pi0_agent import (
    ACRLPDPi0Agent,
    ACRLPDPi0Config,
    TrainingState,
    create_acrlpd_pi0_agent,
    create_acrlpd_pi0_agent_from_rl_config,
)

from .critic_networks import (
    CriticNetworks,
    CriticEnsemble,
    SingleCriticNetwork,
    CriticConfig,
    create_critic_networks,
    DEFAULT_CRITIC_CONFIG,
    LARGE_ENSEMBLE_CONFIG,
    FAST_INFERENCE_CONFIG,
)

from .loss_functions import (
    JointLossComputer,
    LossWeights,
    LossInfo,
    TemperatureModule,
    CriticLossComputer,
    BCLossComputer,
    ActorLossComputer,
    EntropyEstimator,
    create_loss_computer,
)

__all__ = [
    # Main agent classes
    "ACRLPDPi0Agent",
    "ACRLPDPi0Config", 
    "TrainingState",
    "create_acrlpd_pi0_agent",
    "create_acrlpd_pi0_agent_from_rl_config",
    
    # Critic networks
    "CriticNetworks",
    "CriticEnsemble",
    "SingleCriticNetwork",
    "CriticConfig",
    "create_critic_networks",
    "DEFAULT_CRITIC_CONFIG",
    "LARGE_ENSEMBLE_CONFIG", 
    "FAST_INFERENCE_CONFIG",
    
    # Loss functions
    "JointLossComputer",
    "LossWeights",
    "LossInfo",
    "TemperatureModule",
    "CriticLossComputer",
    "BCLossComputer",
    "ActorLossComputer",
    "EntropyEstimator",
    "create_loss_computer",
]

# Version information
__version__ = "0.1.0"
__author__ = "OpenPI ACRLPD Integration Team"
__description__ = "ACRLPD + π₀ agent integration for robotic manipulation"

# Module-level documentation
__doc__ += f"""

Version: {__version__}
Author: {__author__}

This implementation follows the Q-Chunking methodology for action-chunked 
reinforcement learning, integrating it with π₀'s diffusion-based policy 
generation for sample-efficient robotic manipulation learning.

Architecture Overview:
    π₀ Model (Actor) → Multi-modal observations → Action sequences
    ↓
    Critic Ensemble → Q-value estimation → Best-of-N selection
    ↓
    Joint Loss = Critic Loss + BC Loss + π₀ Loss + Temperature Loss

Key Features:
- Direct π₀ integration as policy network
- 10-network critic ensemble for robust Q-estimation
- Best-of-N sampling with Q-value selection
- Bootstrap handling for episode boundaries
- Multi-modal observation processing
- Adaptive temperature control
- Platform-specific optimizations

Supported Platforms:
- DROID: Large-scale robot manipulation dataset
- ALOHA: Bimanual manipulation tasks
- Libero: Simulation benchmark environments
"""