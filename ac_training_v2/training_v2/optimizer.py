"""
Optimizer Configuration for ACRLPD Training

优化器和学习率调度器配置，支持：
- 多种优化器类型 (AdamW, SGD, RMSprop等)
- 学习率调度策略 (cosine, linear, exponential等)
- 分别配置π₀和Critic的优化器
- 梯度裁剪和正则化选项
"""

import logging
from dataclasses import dataclass
from typing import Optional, Union, Callable
import jax
import jax.numpy as jnp
import optax

logger = logging.getLogger(__name__)


@dataclass
class OptimizerConfig:
    """优化器配置"""
    # 基础配置
    optimizer_type: str = "adamw"           # "adamw", "adam", "sgd", "rmsprop"
    learning_rate: float = 1e-4
    
    # AdamW/Adam配置
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01
    
    # SGD配置  
    momentum: float = 0.9
    nesterov: bool = False
    
    # RMSprop配置
    decay: float = 0.9
    
    # 梯度裁剪
    gradient_clip_value: Optional[float] = 1.0
    gradient_clip_norm: Optional[float] = None
    
    # 正则化
    l2_regularization: float = 0.0
    
    def __post_init__(self):
        """配置验证"""
        valid_optimizers = ["adamw", "adam", "sgd", "rmsprop"]
        if self.optimizer_type not in valid_optimizers:
            raise ValueError(f"optimizer_type must be one of {valid_optimizers}")


@dataclass
class LRSchedulerConfig:
    """学习率调度器配置"""
    # 基础配置
    scheduler_type: str = "cosine"          # "constant", "linear", "cosine", "exponential", "warmup_cosine"
    
    # 总训练配置
    total_steps: int = 100000
    warmup_steps: int = 1000
    
    # Cosine衰减配置
    cosine_final_lr_ratio: float = 0.01     # 最终lr与初始lr的比例
    
    # Linear衰减配置  
    linear_final_lr_ratio: float = 0.01
    
    # Exponential衰减配置
    exponential_decay_rate: float = 0.96
    exponential_decay_steps: int = 1000
    
    # Polynomial衰减配置
    polynomial_power: float = 1.0
    
    def __post_init__(self):
        """配置验证"""
        valid_schedulers = ["constant", "linear", "cosine", "exponential", "warmup_cosine", "polynomial"]
        if self.scheduler_type not in valid_schedulers:
            raise ValueError(f"scheduler_type must be one of {valid_schedulers}")


def create_optimizer(config: OptimizerConfig, 
                    lr_schedule: Optional[optax.Schedule] = None) -> optax.GradientTransformation:
    """
    创建优化器
    
    Args:
        config: 优化器配置
        lr_schedule: 可选的学习率调度器，如果None则使用常数学习率
        
    Returns:
        optax.GradientTransformation: 优化器
    """
    
    logger.info(f"创建优化器: {config.optimizer_type}")
    logger.info(f"  学习率: {config.learning_rate}")
    if config.weight_decay > 0:
        logger.info(f"  权重衰减: {config.weight_decay}")
    if config.gradient_clip_value:
        logger.info(f"  梯度裁剪值: {config.gradient_clip_value}")
    if config.gradient_clip_norm:
        logger.info(f"  梯度裁剪范数: {config.gradient_clip_norm}")
    
    # 确定学习率
    learning_rate = lr_schedule if lr_schedule is not None else config.learning_rate
    
    # 创建基础优化器
    if config.optimizer_type == "adamw":
        optimizer = optax.adamw(
            learning_rate=learning_rate,
            b1=config.beta1,
            b2=config.beta2,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    
    elif config.optimizer_type == "adam":
        optimizer = optax.adam(
            learning_rate=learning_rate,
            b1=config.beta1,
            b2=config.beta2,
            eps=config.eps
        )
        
        # Adam需要手动添加weight decay
        if config.weight_decay > 0:
            optimizer = optax.chain(
                optimizer,
                optax.add_decayed_weights(config.weight_decay)
            )
    
    elif config.optimizer_type == "sgd":
        optimizer = optax.sgd(
            learning_rate=learning_rate,
            momentum=config.momentum,
            nesterov=config.nesterov
        )
        
        if config.weight_decay > 0:
            optimizer = optax.chain(
                optimizer,
                optax.add_decayed_weights(config.weight_decay)
            )
    
    elif config.optimizer_type == "rmsprop":
        optimizer = optax.rmsprop(
            learning_rate=learning_rate,
            decay=config.decay,
            eps=config.eps
        )
        
        if config.weight_decay > 0:
            optimizer = optax.chain(
                optimizer,
                optax.add_decayed_weights(config.weight_decay)
            )
    
    else:
        raise ValueError(f"Unsupported optimizer type: {config.optimizer_type}")
    
    # 添加梯度裁剪
    transforms = []
    
    if config.gradient_clip_value is not None:
        transforms.append(optax.clip(config.gradient_clip_value))
    
    if config.gradient_clip_norm is not None:
        transforms.append(optax.clip_by_global_norm(config.gradient_clip_norm))
    
    # 添加基础优化器
    transforms.append(optimizer)
    
    # 组合所有变换
    if len(transforms) > 1:
        final_optimizer = optax.chain(*transforms)
    else:
        final_optimizer = optimizer
    
    logger.info(f"✅ 优化器创建完成: {config.optimizer_type}")
    return final_optimizer


def create_lr_scheduler(config: LRSchedulerConfig, 
                       base_learning_rate: float) -> optax.Schedule:
    """
    创建学习率调度器
    
    Args:
        config: 学习率调度器配置
        base_learning_rate: 基础学习率
        
    Returns:
        optax.Schedule: 学习率调度器
    """
    
    logger.info(f"创建学习率调度器: {config.scheduler_type}")
    logger.info(f"  基础学习率: {base_learning_rate}")
    logger.info(f"  总步数: {config.total_steps}")
    logger.info(f"  warmup步数: {config.warmup_steps}")
    
    if config.scheduler_type == "constant":
        scheduler = optax.constant_schedule(base_learning_rate)
    
    elif config.scheduler_type == "linear":
        final_lr = base_learning_rate * config.linear_final_lr_ratio
        scheduler = optax.linear_schedule(
            init_value=base_learning_rate,
            end_value=final_lr,
            transition_steps=config.total_steps
        )
    
    elif config.scheduler_type == "cosine":
        final_lr = base_learning_rate * config.cosine_final_lr_ratio
        scheduler = optax.cosine_decay_schedule(
            init_value=base_learning_rate,
            decay_steps=config.total_steps,
            alpha=config.cosine_final_lr_ratio
        )
    
    elif config.scheduler_type == "warmup_cosine":
        # Warmup + Cosine衰减
        final_lr = base_learning_rate * config.cosine_final_lr_ratio
        
        warmup_schedule = optax.linear_schedule(
            init_value=0.0,
            end_value=base_learning_rate,
            transition_steps=config.warmup_steps
        )
        
        cosine_schedule = optax.cosine_decay_schedule(
            init_value=base_learning_rate,
            decay_steps=config.total_steps - config.warmup_steps,
            alpha=config.cosine_final_lr_ratio
        )
        
        scheduler = optax.join_schedules(
            schedules=[warmup_schedule, cosine_schedule],
            boundaries=[config.warmup_steps]
        )
    
    elif config.scheduler_type == "exponential":
        scheduler = optax.exponential_decay(
            init_value=base_learning_rate,
            transition_steps=config.exponential_decay_steps,
            decay_rate=config.exponential_decay_rate
        )
    
    elif config.scheduler_type == "polynomial":
        final_lr = base_learning_rate * 0.01  # 最终降到1%
        scheduler = optax.polynomial_schedule(
            init_value=base_learning_rate,
            end_value=final_lr,
            power=config.polynomial_power,
            transition_steps=config.total_steps
        )
    
    else:
        raise ValueError(f"Unsupported scheduler type: {config.scheduler_type}")
    
    logger.info(f"✅ 学习率调度器创建完成: {config.scheduler_type}")
    return scheduler


def create_pi0_optimizer(pi0_lr: float = 1e-5,
                        total_steps: int = 100000,
                        warmup_steps: int = 1000) -> optax.GradientTransformation:
    """
    创建π₀模型专用优化器
    
    π₀模型较大(3.2B参数)，使用较小学习率和更保守的配置
    
    Args:
        pi0_lr: π₀学习率
        total_steps: 总训练步数
        warmup_steps: warmup步数
        
    Returns:
        π₀专用优化器
    """
    
    # π₀专用配置：较小学习率，较大权重衰减
    optimizer_config = OptimizerConfig(
        optimizer_type="adamw",
        learning_rate=pi0_lr,
        weight_decay=0.05,  # 较大的权重衰减
        gradient_clip_norm=1.0,  # 梯度裁剪
        beta1=0.9,
        beta2=0.95  # 更大的beta2用于大模型
    )
    
    # 学习率调度：warmup + cosine
    lr_config = LRSchedulerConfig(
        scheduler_type="warmup_cosine",
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        cosine_final_lr_ratio=0.1
    )
    
    lr_schedule = create_lr_scheduler(lr_config, pi0_lr)
    optimizer = create_optimizer(optimizer_config, lr_schedule)
    
    logger.info(f"✅ π₀优化器创建完成: lr={pi0_lr}")
    return optimizer


def create_critic_optimizer(critic_lr: float = 1e-3,
                           total_steps: int = 100000,
                           warmup_steps: int = 1000) -> optax.GradientTransformation:
    """
    创建Critic专用优化器
    
    Critic网络较小(20M参数)，可以使用较大学习率
    
    Args:
        critic_lr: Critic学习率
        total_steps: 总训练步数
        warmup_steps: warmup步数
        
    Returns:
        Critic专用优化器
    """
    
    # Critic专用配置：较大学习率，标准配置
    optimizer_config = OptimizerConfig(
        optimizer_type="adamw",
        learning_rate=critic_lr,
        weight_decay=0.01,  # 标准权重衰减
        gradient_clip_value=1.0,  # 值裁剪
        beta1=0.9,
        beta2=0.999  # 标准beta2
    )
    
    # 学习率调度：warmup + cosine
    lr_config = LRSchedulerConfig(
        scheduler_type="warmup_cosine",
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        cosine_final_lr_ratio=0.01
    )
    
    lr_schedule = create_lr_scheduler(lr_config, critic_lr)
    optimizer = create_optimizer(optimizer_config, lr_schedule)
    
    logger.info(f"✅ Critic优化器创建完成: lr={critic_lr}")
    return optimizer


def create_optimizers_from_config(rl_config: any) -> tuple[optax.GradientTransformation, optax.GradientTransformation]:
    """
    从RLConfig创建π₀和Critic优化器
    
    Args:
        rl_config: AC Training的RLConfig
        
    Returns:
        (pi0_optimizer, critic_optimizer)
    """
    
    # 提取配置参数
    pi0_lr = getattr(rl_config, 'pi0_lr', 1e-5)
    critic_lr = getattr(rl_config, 'critic_lr', 1e-3)
    total_steps = getattr(rl_config, 'max_steps', 100000)
    warmup_steps = getattr(rl_config, 'warmup_steps', 1000)
    
    # 创建优化器
    pi0_optimizer = create_pi0_optimizer(pi0_lr, total_steps, warmup_steps)
    critic_optimizer = create_critic_optimizer(critic_lr, total_steps, warmup_steps)
    
    logger.info("✅ 从RLConfig创建优化器完成")
    return pi0_optimizer, critic_optimizer


def get_learning_rate(optimizer_state: optax.OptState, step: int) -> float:
    """
    获取当前学习率
    
    Args:
        optimizer_state: 优化器状态
        step: 当前步数
        
    Returns:
        当前学习率
    """
    # 从优化器状态中提取学习率
    # 这个函数依赖于optax的内部实现，可能需要根据版本调整
    try:
        if hasattr(optimizer_state, 'hyperparams') and 'learning_rate' in optimizer_state.hyperparams:
            return float(optimizer_state.hyperparams['learning_rate'])
        else:
            # Fallback：无法直接提取，返回默认值
            return 0.0
    except:
        return 0.0