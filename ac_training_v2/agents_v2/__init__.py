"""
AC Training v2 Agents Module

高效的ACRLPD Agent实现，保持完整RL功能的同时优化性能
"""

from .acrlpd_pi0_agent import ACRLPDPi0Agent, create_acrlpd_pi0_agent_from_rl_config, ACRLPDConfig
from .loss_functions import UnifiedLossComputer, create_loss_computer, LossConfig
from .critic_networks import CriticNetwork, CriticEnsemble

__all__ = [
    "ACRLPDPi0Agent",
    "create_acrlpd_pi0_agent_from_rl_config", 
    "ACRLPDConfig",
    "UnifiedLossComputer",
    "create_loss_computer",
    "LossConfig", 
    "CriticNetwork",
    "CriticEnsemble",
]