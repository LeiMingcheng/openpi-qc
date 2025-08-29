"""
AC_Training_v2: High-Efficiency ACRLPD Training Framework

基于OpenPI模式重构的ACRLPD训练框架，保持完整RL算法功能：
- Actor-Critic Regularized Learning from Play Data  
- π₀ diffusion模型 + Critic ensemble
- Best-of-N sampling策略优化
- Q-learning with action chunking

核心优化：
- 统一JIT编译训练步骤
- 特征共享计算
- 简化训练循环
- 高效数据加载

模块组织：
- agents_v2: Agent核心实现
- training_v2: 训练循环和状态管理  
- data_v2: 数据加载组件
- scripts_v2: 训练脚本
- utils_v2: 支持工具
"""

__version__ = "2.0.0"

# 核心组件导入
from .agents_v2 import (
    ACRLPDPi0Agent, 
    create_acrlpd_pi0_agent_from_rl_config,
    UnifiedLossComputer,
    create_loss_computer,
    CriticEnsemble
)

from .training_v2 import (
    ACRLPDTrainer,
    ACRLPDTrainingConfig,
    ACRLPDTrainState,
    init_train_state,
    create_optimizer,
    create_lr_scheduler
)

from .utils_v2 import (
    MetricsLogger,
    CheckpointManager, 
    PerformanceMonitor
)

__all__ = [
    # Agent组件
    "ACRLPDPi0Agent",
    "create_acrlpd_pi0_agent_from_rl_config",
    "UnifiedLossComputer", 
    "create_loss_computer",
    "CriticEnsemble",
    
    # 训练组件
    "ACRLPDTrainer",
    "ACRLPDTrainingConfig",
    "ACRLPDTrainState",
    "init_train_state", 
    "create_optimizer",
    "create_lr_scheduler",
    
    # 工具组件
    "MetricsLogger",
    "CheckpointManager",
    "PerformanceMonitor",
]