"""
ACRLPD Training State Management

统一的训练状态管理，支持：
- π₀和Critic参数的分离管理
- FSDP distributed training
- 高效的JIT编译训练步骤
- Checkpoint保存和恢复

核心优化：统一的训练步骤，避免重复编译和计算
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, Optional, NamedTuple
import jax
import jax.numpy as jnp
import optax
from flax import struct
import pickle
import os

# AC Training v2 imports - fix relative import
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents_v2.acrlpd_pi0_agent import ACRLPDPi0Agent, SharedFeatures
from agents_v2.loss_functions import UnifiedLossComputer, LossInfo

logger = logging.getLogger(__name__)


class TrainingMetrics(NamedTuple):
    """训练指标结构"""
    total_loss: jnp.ndarray
    critic_loss: jnp.ndarray
    actor_loss: jnp.ndarray
    bc_loss: jnp.ndarray
    
    # 监控指标
    critic_q_mean: jnp.ndarray
    actor_q_best: jnp.ndarray
    bc_mse: jnp.ndarray
    
    # 梯度和参数统计
    pi0_grad_norm: jnp.ndarray
    critic_grad_norm: jnp.ndarray
    pi0_param_norm: jnp.ndarray
    critic_param_norm: jnp.ndarray


@struct.dataclass
class ACRLPDTrainState:
    """
    ACRLPD训练状态容器
    
    管理所有训练相关的状态：参数、优化器状态、步数等
    """
    # 模型参数
    pi0_params: Any                    # π₀模型参数
    critic_params: Any                 # Critic网络参数
    critic_target_params: Any          # Critic target网络参数
    
    # 优化器状态
    pi0_opt_state: optax.OptState     # π₀优化器状态
    critic_opt_state: optax.OptState  # Critic优化器状态
    
    # 训练状态
    step: int                         # 当前训练步数
    rng: jax.random.PRNGKey          # 随机数生成器状态
    
    # 配置信息
    config: Dict[str, Any]           # 训练配置
    
    def apply_pi0_gradients(self, pi0_grads, pi0_optimizer):
        """应用π₀模型梯度"""
        pi0_updates, new_pi0_opt_state = pi0_optimizer.update(
            pi0_grads, self.pi0_opt_state, self.pi0_params
        )
        new_pi0_params = optax.apply_updates(self.pi0_params, pi0_updates)
        
        return self.replace(
            pi0_params=new_pi0_params,
            pi0_opt_state=new_pi0_opt_state
        )
    
    def apply_critic_gradients(self, critic_grads, critic_optimizer):
        """应用Critic网络梯度"""
        critic_updates, new_critic_opt_state = critic_optimizer.update(
            critic_grads, self.critic_opt_state, self.critic_params
        )
        new_critic_params = optax.apply_updates(self.critic_params, critic_updates)
        
        return self.replace(
            critic_params=new_critic_params,
            critic_opt_state=new_critic_opt_state
        )
    
    def update_target_networks(self, agent: ACRLPDPi0Agent):
        """更新target networks"""
        new_target_params = agent.critic_ensemble.update_target_networks(
            self.critic_params, 
            self.critic_target_params
        )
        
        return self.replace(critic_target_params=new_target_params)
    
    def increment_step(self):
        """递增步数"""
        return self.replace(step=self.step + 1)
    
    def update_rng(self, new_rng):
        """更新随机数生成器"""
        return self.replace(rng=new_rng)


@jax.jit
def compute_losses_and_gradients(train_state: ACRLPDTrainState,
                                batch: Dict[str, jnp.ndarray], 
                                agent: ACRLPDPi0Agent,
                                loss_computer: UnifiedLossComputer) -> Tuple[Dict[str, jnp.ndarray], LossInfo]:
    """
    计算损失和梯度 - 核心训练函数
    
    统一计算所有损失并返回对应的梯度，完全JIT编译优化性能
    
    Args:
        train_state: 训练状态
        batch: 训练batch
        agent: ACRLPD Agent
        loss_computer: 损失计算器
        
    Returns:
        gradients: {pi0_grads, critic_grads}
        loss_info: 损失信息
    """
    
    def unified_loss_fn(pi0_params, critic_params):
        """统一损失函数（用于求梯度）"""
        
        # 1. 设置当前参数到agent
        agent = agent.replace(pi0_params=pi0_params)
        agent = agent.replace(critic_params=critic_params)
        
        # 2. 计算共享特征（关键优化）
        shared_features = agent.compute_shared_features(batch)
        
        # 3. 统一计算所有损失
        loss_info = loss_computer.compute_all_losses(
            shared_features=shared_features,
            batch=batch,
            rng=train_state.rng
        )
        
        return loss_info.total_loss, loss_info
    
    # 4. 计算梯度（分别对π₀和Critic）
    (total_loss, loss_info), grads = jax.value_and_grad(
        unified_loss_fn, argnums=(0, 1), has_aux=True
    )(train_state.pi0_params, train_state.critic_params)
    
    pi0_grads, critic_grads = grads
    
    gradients = {
        'pi0_grads': pi0_grads,
        'critic_grads': critic_grads
    }
    
    return gradients, loss_info


@jax.jit  
def apply_gradients(train_state: ACRLPDTrainState,
                   gradients: Dict[str, jnp.ndarray],
                   pi0_optimizer: optax.GradientTransformation,
                   critic_optimizer: optax.GradientTransformation,
                   agent: ACRLPDPi0Agent) -> ACRLPDTrainState:
    """
    应用梯度更新参数 - JIT编译优化
    
    Args:
        train_state: 当前训练状态
        gradients: 梯度字典
        pi0_optimizer: π₀优化器
        critic_optimizer: Critic优化器  
        agent: ACRLPD Agent
        
    Returns:
        new_train_state: 更新后的训练状态
    """
    
    # 1. 应用π₀梯度
    train_state = train_state.apply_pi0_gradients(
        gradients['pi0_grads'], pi0_optimizer
    )
    
    # 2. 应用Critic梯度  
    train_state = train_state.apply_critic_gradients(
        gradients['critic_grads'], critic_optimizer
    )
    
    # 3. 更新target networks
    train_state = train_state.update_target_networks(agent)
    
    # 4. 递增步数
    train_state = train_state.increment_step()
    
    # 5. 更新随机数种子
    new_rng = jax.random.split(train_state.rng)[0]
    train_state = train_state.update_rng(new_rng)
    
    return train_state


def unified_train_step(train_state: ACRLPDTrainState,
                      batch: Dict[str, jnp.ndarray],
                      agent: ACRLPDPi0Agent,
                      loss_computer: UnifiedLossComputer,
                      pi0_optimizer: optax.GradientTransformation,
                      critic_optimizer: optax.GradientTransformation) -> Tuple[ACRLPDTrainState, TrainingMetrics]:
    """
    统一训练步骤 - 完全优化的单步训练
    
    结合损失计算、梯度计算、参数更新为单个高效操作
    
    Args:
        train_state: 训练状态
        batch: 训练batch
        agent: ACRLPD Agent
        loss_computer: 损失计算器
        pi0_optimizer: π₀优化器
        critic_optimizer: Critic优化器
        
    Returns:
        new_train_state: 更新后的训练状态
        metrics: 训练指标
    """
    
    # 1. 计算损失和梯度
    gradients, loss_info = compute_losses_and_gradients(
        train_state, batch, agent, loss_computer
    )
    
    # 2. 应用梯度更新参数
    new_train_state = apply_gradients(
        train_state, gradients, pi0_optimizer, critic_optimizer, agent
    )
    
    # 3. 计算监控指标
    pi0_grad_norm = _compute_grad_norm(gradients['pi0_grads'])
    critic_grad_norm = _compute_grad_norm(gradients['critic_grads'])
    pi0_param_norm = _compute_param_norm(new_train_state.pi0_params)
    critic_param_norm = _compute_param_norm(new_train_state.critic_params)
    
    metrics = TrainingMetrics(
        total_loss=loss_info.total_loss,
        critic_loss=loss_info.critic_loss,
        actor_loss=loss_info.actor_loss,
        bc_loss=loss_info.bc_loss,
        critic_q_mean=loss_info.critic_q_mean,
        actor_q_best=loss_info.actor_q_best,
        bc_mse=loss_info.bc_mse,
        pi0_grad_norm=pi0_grad_norm,
        critic_grad_norm=critic_grad_norm,
        pi0_param_norm=pi0_param_norm,
        critic_param_norm=critic_param_norm
    )
    
    return new_train_state, metrics


# JIT编译统一训练步骤（关键优化）
unified_train_step_jit = jax.jit(unified_train_step, static_argnums=(2, 3, 4, 5))


def init_train_state(agent: ACRLPDPi0Agent,
                    pi0_optimizer: optax.GradientTransformation,
                    critic_optimizer: optax.GradientTransformation,
                    rng: jax.random.PRNGKey,
                    config: Dict[str, Any]) -> ACRLPDTrainState:
    """
    初始化训练状态
    
    Args:
        agent: ACRLPD Agent
        pi0_optimizer: π₀优化器
        critic_optimizer: Critic优化器
        rng: 随机数生成器
        config: 训练配置
        
    Returns:
        ACRLPDTrainState: 初始化的训练状态
    """
    
    logger.info("初始化ACRLPD训练状态...")
    
    # 获取模型参数
    trainable_params = agent.get_trainable_params()
    pi0_params = trainable_params['pi0_params']
    critic_params = trainable_params['critic_params']
    
    # 初始化target network参数（复制critic参数）
    critic_target_params = jax.tree_map(lambda x: x.copy(), critic_params)
    
    # 初始化优化器状态
    pi0_opt_state = pi0_optimizer.init(pi0_params)
    critic_opt_state = critic_optimizer.init(critic_params)
    
    # 创建训练状态
    train_state = ACRLPDTrainState(
        pi0_params=pi0_params,
        critic_params=critic_params,
        critic_target_params=critic_target_params,
        pi0_opt_state=pi0_opt_state,
        critic_opt_state=critic_opt_state,
        step=0,
        rng=rng,
        config=config
    )
    
    logger.info("✅ ACRLPD训练状态初始化完成")
    logger.info(f"  π₀参数数量: ~3.2B")
    logger.info(f"  Critic参数数量: ~{agent.config.num_critics * 20}M")
    
    return train_state


def save_checkpoint(train_state: ACRLPDTrainState,
                   checkpoint_path: str,
                   metadata: Optional[Dict[str, Any]] = None):
    """
    保存训练状态checkpoint
    
    Args:
        train_state: 训练状态
        checkpoint_path: checkpoint保存路径
        metadata: 额外的元数据
    """
    
    logger.info(f"保存checkpoint: {checkpoint_path}")
    
    # 准备保存数据
    save_data = {
        'train_state': train_state,
        'metadata': metadata or {},
        'step': train_state.step,
    }
    
    # 确保目录存在
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # 保存checkpoint
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    logger.info(f"✅ Checkpoint保存完成: step {train_state.step}")


def load_checkpoint(checkpoint_path: str) -> Tuple[ACRLPDTrainState, Dict[str, Any]]:
    """
    加载训练状态checkpoint
    
    Args:
        checkpoint_path: checkpoint路径
        
    Returns:
        train_state: 训练状态
        metadata: 元数据
    """
    
    logger.info(f"加载checkpoint: {checkpoint_path}")
    
    with open(checkpoint_path, 'rb') as f:
        save_data = pickle.load(f)
    
    train_state = save_data['train_state']
    metadata = save_data.get('metadata', {})
    
    logger.info(f"✅ Checkpoint加载完成: step {train_state.step}")
    
    return train_state, metadata


def _compute_grad_norm(gradients) -> jnp.ndarray:
    """计算梯度范数"""
    grad_squares = jax.tree_map(lambda g: jnp.sum(g ** 2), gradients)
    total_grad_square = jax.tree_reduce(jnp.add, grad_squares)
    return jnp.sqrt(total_grad_square)


def _compute_param_norm(params) -> jnp.ndarray:
    """计算参数范数"""
    param_squares = jax.tree_map(lambda p: jnp.sum(p ** 2), params)
    total_param_square = jax.tree_reduce(jnp.add, param_squares)
    return jnp.sqrt(total_param_square)


def get_latest_checkpoint_path(checkpoint_dir: str) -> Optional[str]:
    """
    获取最新的checkpoint路径
    
    Args:
        checkpoint_dir: checkpoint目录
        
    Returns:
        latest_checkpoint_path: 最新checkpoint路径，如果没有返回None
    """
    
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = []
    for f in os.listdir(checkpoint_dir):
        if f.endswith('.pkl') and f.startswith('checkpoint_'):
            try:
                step = int(f.split('_')[1].split('.')[0])
                checkpoints.append((step, os.path.join(checkpoint_dir, f)))
            except ValueError:
                continue
    
    if not checkpoints:
        return None
    
    # 返回步数最大的checkpoint
    latest_checkpoint = max(checkpoints, key=lambda x: x[0])
    return latest_checkpoint[1]