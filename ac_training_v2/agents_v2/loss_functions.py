"""
Unified Loss Computation for ACRLPD

统一的损失计算系统，实现特征共享优化，同时保持完整的RL算法功能：
- Critic Loss: Q-learning with action chunking
- Actor Loss: Best-of-N sampling for policy optimization  
- BC Loss: Behavior cloning regularization

核心优化：通过共享特征避免重复encoder计算，整体JIT编译提升性能
"""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, NamedTuple
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.model import Observation
# Fix circular import by importing at runtime in functions that need it

logger = logging.getLogger(__name__)


@dataclass 
class LossConfig:
    """损失计算配置"""
    # 损失权重
    critic_weight: float = 1.0
    actor_weight: float = 1.0
    bc_weight: float = 0.1
    
    # Critic损失配置
    discount_factor: float = 0.99
    target_tau: float = 0.005  # Target network软更新系数
    
    # Actor损失配置  
    best_of_n: int = 4  # Best-of-N采样数量
    temperature: float = 1.0  # 温度参数
    
    # BC损失配置
    bc_loss_type: str = "mse"  # "mse" or "huber"
    huber_delta: float = 1.0
    
    # 正则化
    l2_regularization: float = 1e-4


class LossInfo(NamedTuple):
    """损失信息结构"""
    total_loss: jnp.ndarray
    critic_loss: jnp.ndarray
    actor_loss: jnp.ndarray
    bc_loss: jnp.ndarray
    
    # 额外信息用于监控
    critic_q_mean: jnp.ndarray
    critic_target_q_mean: jnp.ndarray
    actor_q_best: jnp.ndarray
    actor_q_candidates: jnp.ndarray
    bc_mse: jnp.ndarray


class UnifiedLossComputer:
    """
    统一损失计算器 - 核心优化组件
    
    将Actor、Critic、BC损失计算整合到单个JIT编译的函数中，
    通过共享特征避免重复encoder计算，大幅提升训练效率
    """
    
    def __init__(self, agent, config: LossConfig):
        """
        初始化统一损失计算器
        
        Args:
            agent: ACRLPD Agent实例
            config: 损失计算配置
        """
        self.agent = agent
        self.config = config
        
        # 缓存常用配置
        self._action_horizon = agent._action_horizon
        self._action_dim = agent._action_dim
        self._best_of_n = config.best_of_n
        self._discount = config.discount_factor
        
        logger.info(f"=== UnifiedLossComputer 初始化 ===")
        logger.info(f"损失权重: Critic={config.critic_weight}, Actor={config.actor_weight}, BC={config.bc_weight}")
        logger.info(f"Best-of-N采样: {config.best_of_n}")
        logger.info(f"折扣因子: {config.discount_factor}")
        
    @jax.jit  # 核心优化：整体JIT编译
    def compute_all_losses(self, 
                          shared_features,
                          batch: Dict[str, jnp.ndarray], 
                          rng: jax.random.PRNGKey) -> LossInfo:
        """
        统一计算所有损失 - 核心函数
        
        Args:
            shared_features: 共享特征（避免重复计算）
            batch: 训练batch
            rng: 随机数生成器
            
        Returns:
            LossInfo: 包含所有损失和监控信息
        """
        
        # 分割随机数
        rng_critic, rng_actor, rng_bc = jax.random.split(rng, 3)
        
        # 1. 计算Critic损失（Q-learning）
        critic_loss_info = self._compute_critic_loss(
            shared_features, batch, rng_critic
        )
        
        # 2. 计算Actor损失（Best-of-N sampling）
        actor_loss_info = self._compute_actor_loss(
            shared_features, batch, rng_actor
        )
        
        # 3. 计算BC损失（行为克隆正则化）
        bc_loss_info = self._compute_bc_loss(
            shared_features, batch, rng_bc
        )
        
        # 4. 计算总损失（加权组合）
        total_loss = (
            self.config.critic_weight * critic_loss_info['loss'] +
            self.config.actor_weight * actor_loss_info['loss'] + 
            self.config.bc_weight * bc_loss_info['loss']
        )
        
        # 5. L2正则化
        if self.config.l2_regularization > 0:
            l2_reg = self._compute_l2_regularization()
            total_loss = total_loss + self.config.l2_regularization * l2_reg
        
        return LossInfo(
            total_loss=total_loss,
            critic_loss=critic_loss_info['loss'],
            actor_loss=actor_loss_info['loss'], 
            bc_loss=bc_loss_info['loss'],
            
            # 监控信息
            critic_q_mean=critic_loss_info['q_mean'],
            critic_target_q_mean=critic_loss_info['target_q_mean'],
            actor_q_best=actor_loss_info['q_best'],
            actor_q_candidates=actor_loss_info['q_candidates_mean'],
            bc_mse=bc_loss_info['mse']
        )
    
    def _compute_critic_loss(self, 
                           shared_features, 
                           batch: Dict[str, jnp.ndarray],
                           rng: jax.random.PRNGKey) -> Dict[str, jnp.ndarray]:
        """
        计算Critic损失 - Q-learning with action chunking
        
        实现标准的时序差分(TD)学习，支持动作序列的Q值估计
        """
        batch_size = shared_features.fused_features.shape[0]
        
        # 当前状态和动作
        current_obs = shared_features.fused_features  # [B, obs_feat_dim]
        actions = batch['actions']  # [B, action_horizon, action_dim]
        rewards = batch['rewards']  # [B, action_horizon]
        terminals = batch['terminals']  # [B, action_horizon] 
        
        # 下一状态特征（如果有的话）
        if 'next_state' in batch:
            next_state_features = self.agent.observation_encoder.encode_state(batch['next_state'])
            # 简化处理：假设只有状态变化，图像特征保持相同
            next_obs_features = jnp.concatenate([
                next_state_features, 
                shared_features.fused_features[:, next_state_features.shape[-1]:]  # 复用图像特征部分
            ], axis=-1)
        else:
            # 如果没有next_state，使用当前状态（序列内的下一步）
            next_obs_features = current_obs
        
        # 1. 当前Q值估计
        current_q_values = self.agent.critic_ensemble.compute_q_values(
            obs_features=current_obs,
            actions=actions  # [B, H, A]
        )  # [num_critics, B, H]
        
        # 2. Target Q值计算
        # 对于动作序列，使用序列中下一步的Q值作为target
        with jax.lax.stop_gradient():
            # 获取下一步动作（序列内移位）
            next_actions = jnp.roll(actions, shift=-1, axis=1)  # [B, H, A]
            
            # 计算target Q值
            target_q_values = self.agent.critic_ensemble.compute_target_q_values(
                obs_features=next_obs_features,
                actions=next_actions
            )  # [num_critics, B, H]
            
            # TD target计算
            td_targets = rewards + self._discount * target_q_values * (1 - terminals)  # [num_critics, B, H]
        
        # 3. TD error计算
        td_errors = current_q_values - td_targets  # [num_critics, B, H]
        
        # 4. Huber loss（减少outlier影响）
        huber_loss = jnp.where(
            jnp.abs(td_errors) <= 1.0,
            0.5 * td_errors ** 2,
            jnp.abs(td_errors) - 0.5
        )
        
        # 5. 在时间维度和ensemble维度上平均
        critic_loss = jnp.mean(huber_loss)
        
        # 监控信息
        q_mean = jnp.mean(current_q_values)
        target_q_mean = jnp.mean(td_targets)
        
        return {
            'loss': critic_loss,
            'q_mean': q_mean,
            'target_q_mean': target_q_mean
        }
    
    def _compute_actor_loss(self, 
                          shared_features,
                          batch: Dict[str, jnp.ndarray], 
                          rng: jax.random.PRNGKey) -> Dict[str, jnp.ndarray]:
        """
        计算Actor损失 - Best-of-N sampling
        
        ACRLPD算法的核心：通过Best-of-N采样优化策略
        """
        batch_size = shared_features.fused_features.shape[0]
        
        # 1. 构建观察结构
        observation = self._build_observation_from_batch(batch)
        
        # 2. Best-of-N采样：为每个batch样本生成N个动作候选
        def sample_candidates_for_single_obs(single_obs, single_rng):
            """为单个观察采样动作候选"""
            return self.agent.sample_action_candidates(
                observation=single_obs,
                rng=single_rng, 
                num_candidates=self._best_of_n
            )
        
        # 为每个batch样本分配随机数
        sample_rngs = jax.random.split(rng, batch_size)
        
        # 向量化采样：[B, best_of_n, action_horizon, action_dim]
        action_candidates = jax.vmap(sample_candidates_for_single_obs)(
            observation, sample_rngs
        )
        
        # 3. 使用Critic评估所有候选动作的Q值
        def evaluate_candidate_batch(candidates):
            """评估一个batch的候选动作"""
            # candidates: [B, H, A]
            q_values = self.agent.critic_ensemble.compute_min_q_values(
                obs_features=shared_features.fused_features,
                actions=candidates
            )  # [B, H] -> [B] (在时间维度求和/平均)
            return jnp.mean(q_values, axis=-1)  # [B]
        
        # 对所有候选评估：[best_of_n, B]
        all_q_values = jax.vmap(evaluate_candidate_batch, in_axes=1, out_axes=0)(
            action_candidates
        )
        
        # 4. 选择最佳候选（最大Q值）
        best_indices = jnp.argmax(all_q_values, axis=0)  # [B]
        best_q_values = jnp.max(all_q_values, axis=0)   # [B]
        
        # 5. Actor损失：最大化最佳候选的Q值
        actor_loss = -jnp.mean(best_q_values)
        
        # 监控信息
        q_candidates_mean = jnp.mean(all_q_values)
        
        return {
            'loss': actor_loss,
            'q_best': jnp.mean(best_q_values),
            'q_candidates_mean': q_candidates_mean
        }
    
    def _compute_bc_loss(self,
                        shared_features,
                        batch: Dict[str, jnp.ndarray],
                        rng: jax.random.PRNGKey) -> Dict[str, jnp.ndarray]:
        """
        计算行为克隆损失 - 正则化项
        
        防止策略偏离专家轨迹过远
        """
        # 专家动作
        expert_actions = batch['actions']  # [B, H, A]
        
        # 构建观察并采样策略动作
        observation = self._build_observation_from_batch(batch)
        
        def sample_policy_action(single_obs, single_rng):
            """采样策略动作"""
            return self.agent.get_action(single_obs, single_rng)
        
        # 为每个batch样本采样策略动作
        sample_rngs = jax.random.split(rng, expert_actions.shape[0])
        policy_actions = jax.vmap(sample_policy_action)(observation, sample_rngs)  # [B, H, A]
        
        # BC损失计算
        if self.config.bc_loss_type == "mse":
            action_diff = policy_actions - expert_actions  # [B, H, A]
            bc_loss = jnp.mean(action_diff ** 2)
        elif self.config.bc_loss_type == "huber":
            action_diff = policy_actions - expert_actions
            bc_loss = jnp.mean(jnp.where(
                jnp.abs(action_diff) <= self.config.huber_delta,
                0.5 * action_diff ** 2,
                self.config.huber_delta * jnp.abs(action_diff) - 0.5 * self.config.huber_delta ** 2
            ))
        
        # 监控信息
        mse = jnp.mean((policy_actions - expert_actions) ** 2)
        
        return {
            'loss': bc_loss,
            'mse': mse
        }
    
    def _build_observation_from_batch(self, batch: Dict[str, jnp.ndarray]) -> Observation:
        """
        从batch构建Observation结构
        
        Args:
            batch: 训练batch
            
        Returns:
            Observation: 用于π₀模型的观察结构
        """
        # 构建标准的Observation格式
        return Observation.from_dict({
            'image': batch['image'],
            'state': batch['state'],
            'prompt': batch.get('prompt', ''),  # 任务描述
        })
    
    def _compute_l2_regularization(self) -> jnp.ndarray:
        """计算L2正则化项"""
        # 对所有可训练参数计算L2正则化
        params = self.agent.get_trainable_params()
        
        l2_reg = 0.0
        for param_group in params.values():
            l2_reg += jnp.sum(jnp.array([
                jnp.sum(p ** 2) for p in jax.tree_leaves(param_group)
            ]))
        
        return l2_reg


def create_loss_computer(agent, 
                        config: Optional[LossConfig] = None) -> UnifiedLossComputer:
    """
    工厂函数：创建统一损失计算器
    
    Args:
        agent: ACRLPD Agent实例
        config: 损失计算配置，默认使用标准配置
        
    Returns:
        UnifiedLossComputer实例
    """
    if config is None:
        config = LossConfig()
    
    logger.info("创建UnifiedLossComputer...")
    loss_computer = UnifiedLossComputer(agent=agent, config=config)
    
    logger.info(f"✅ UnifiedLossComputer创建完成")
    return loss_computer


def create_loss_config_from_rl_config(rl_config: Any) -> LossConfig:
    """
    从RLConfig创建LossConfig
    
    Args:
        rl_config: AC Training的RLConfig
        
    Returns:
        LossConfig实例
    """
    return LossConfig(
        # 损失权重
        critic_weight=getattr(rl_config, 'critic_weight', 1.0),
        actor_weight=getattr(rl_config, 'actor_weight', 1.0), 
        bc_weight=getattr(rl_config, 'bc_weight', 0.1),
        
        # Critic配置
        discount_factor=getattr(rl_config, 'discount', 0.99),
        target_tau=getattr(rl_config, 'target_tau', 0.005),
        
        # Actor配置
        best_of_n=getattr(rl_config, 'best_of_n', 4),
        temperature=getattr(rl_config, 'temperature', 1.0),
        
        # 正则化
        l2_regularization=getattr(rl_config, 'l2_reg', 1e-4)
    )