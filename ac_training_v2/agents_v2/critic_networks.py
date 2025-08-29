"""
Critic Networks for ACRLPD

Critic网络实现，支持Q-learning with action chunking:
- CriticNetwork: 单个Q网络，支持动作序列的Q值估计
- CriticEnsemble: Critic集合，提供更稳定的Q值估计和conservative Q-learning
- 支持target network和软更新机制

核心功能：
- 处理观察特征 + 动作序列 -> Q值序列
- Conservative Q-learning (使用ensemble最小值)
- Target network稳定训练
"""

import logging
from typing import List, Dict, Tuple, Any, Optional
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import numpy as np

logger = logging.getLogger(__name__)


class CriticNetwork(nn.Module):
    """
    单个Critic网络
    
    输入：观察特征 + 动作序列
    输出：对应的Q值序列
    """
    
    hidden_dims: List[int] = (256, 256)
    activation: str = 'relu'
    dropout_rate: float = 0.0
    use_batch_norm: bool = False
    
    @nn.compact
    def __call__(self, 
                 obs_features: jnp.ndarray,    # [B, obs_feat_dim] 
                 actions: jnp.ndarray,         # [B, action_horizon, action_dim]
                 training: bool = True) -> jnp.ndarray:
        """
        前向传播计算Q值
        
        Args:
            obs_features: 观察特征 [B, obs_feat_dim]
            actions: 动作序列 [B, action_horizon, action_dim]  
            training: 是否训练模式
            
        Returns:
            q_values: Q值序列 [B, action_horizon]
        """
        batch_size, action_horizon, action_dim = actions.shape
        obs_feat_dim = obs_features.shape[-1]
        
        # 1. 展开时间维度：[B, H, A] -> [B*H, A]
        actions_flat = actions.reshape(-1, action_dim)
        
        # 2. 扩展观察特征以匹配时间维度：[B, feat] -> [B*H, feat]  
        obs_features_expanded = jnp.repeat(obs_features, action_horizon, axis=0)
        
        # 3. 连接观察特征和动作：[B*H, obs_feat_dim + action_dim]
        inputs = jnp.concatenate([obs_features_expanded, actions_flat], axis=-1)
        
        # 4. MLP网络
        x = inputs
        for i, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(hidden_dim, name=f'dense_{i}')(x)
            
            if self.use_batch_norm:
                x = nn.BatchNorm(use_running_average=not training, name=f'bn_{i}')(x)
            
            if self.activation == 'relu':
                x = nn.relu(x)
            elif self.activation == 'tanh':
                x = nn.tanh(x)
            elif self.activation == 'swish':
                x = nn.swish(x)
                
            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        # 5. 输出层：[B*H, 1] -> [B*H]
        q_values_flat = nn.Dense(1, name='q_output')(x).squeeze(-1)
        
        # 6. 恢复时间维度：[B*H] -> [B, H]
        q_values = q_values_flat.reshape(batch_size, action_horizon)
        
        return q_values


class CriticEnsemble(nn.Module):
    """
    Critic网络集合
    
    管理多个Critic网络，提供更稳定的Q值估计和conservative Q-learning
    """
    
    num_critics: int = 2
    obs_feat_dim: int = 768
    action_dim: int = 14 
    hidden_dims: List[int] = (256, 256)
    activation: str = 'relu'
    dropout_rate: float = 0.0
    use_batch_norm: bool = False
    target_tau: float = 0.005  # Target network软更新系数
    
    def setup(self):
        """初始化多个Critic网络"""
        self.critics = [
            CriticNetwork(
                hidden_dims=self.hidden_dims,
                activation=self.activation,
                dropout_rate=self.dropout_rate,
                use_batch_norm=self.use_batch_norm
            ) for _ in range(self.num_critics)
        ]
        
        # Target networks (用于稳定Q-learning)
        self.target_critics = [
            CriticNetwork(
                hidden_dims=self.hidden_dims,
                activation=self.activation,
                dropout_rate=0.0,  # Target network不使用dropout
                use_batch_norm=self.use_batch_norm
            ) for _ in range(self.num_critics)
        ]
    
    def __call__(self,
                 obs_features: jnp.ndarray,
                 actions: jnp.ndarray, 
                 training: bool = True) -> jnp.ndarray:
        """
        计算所有Critic的Q值
        
        Args:
            obs_features: 观察特征 [B, obs_feat_dim]
            actions: 动作序列 [B, action_horizon, action_dim]
            training: 是否训练模式
            
        Returns:
            q_values: 所有Critic的Q值 [num_critics, B, action_horizon]
        """
        q_values_list = []
        
        for i, critic in enumerate(self.critics):
            q_vals = critic(obs_features, actions, training=training)
            q_values_list.append(q_vals)
        
        return jnp.stack(q_values_list, axis=0)  # [num_critics, B, H]
    
    def compute_q_values(self, 
                        obs_features: jnp.ndarray,
                        actions: jnp.ndarray) -> jnp.ndarray:
        """计算所有Critic的Q值（训练模式）"""
        return self(obs_features, actions, training=True)
    
    def compute_target_q_values(self,
                               obs_features: jnp.ndarray,
                               actions: jnp.ndarray) -> jnp.ndarray:
        """
        使用target networks计算Q值
        
        Args:
            obs_features: 观察特征 [B, obs_feat_dim]
            actions: 动作序列 [B, action_horizon, action_dim]
            
        Returns:
            target_q_values: Target Q值 [num_critics, B, action_horizon]
        """
        target_q_values_list = []
        
        for i, target_critic in enumerate(self.target_critics):
            q_vals = target_critic(obs_features, actions, training=False)
            target_q_values_list.append(q_vals)
        
        return jnp.stack(target_q_values_list, axis=0)  # [num_critics, B, H]
    
    def compute_min_q_values(self,
                            obs_features: jnp.ndarray,
                            actions: jnp.ndarray) -> jnp.ndarray:
        """
        Conservative Q-learning: 计算ensemble中的最小Q值
        
        这是conservative Q-learning的核心，通过使用最小Q值来避免过估计
        
        Args:
            obs_features: 观察特征 [B, obs_feat_dim]
            actions: 动作序列 [B, action_horizon, action_dim]
            
        Returns:
            min_q_values: 最小Q值 [B, action_horizon]
        """
        all_q_values = self.compute_q_values(obs_features, actions)  # [num_critics, B, H]
        min_q_values = jnp.min(all_q_values, axis=0)  # [B, H]
        
        return min_q_values
    
    def compute_mean_q_values(self,
                             obs_features: jnp.ndarray,
                             actions: jnp.ndarray) -> jnp.ndarray:
        """
        计算ensemble的平均Q值
        
        Args:
            obs_features: 观察特征 [B, obs_feat_dim]
            actions: 动作序列 [B, action_horizon, action_dim]
            
        Returns:
            mean_q_values: 平均Q值 [B, action_horizon]
        """
        all_q_values = self.compute_q_values(obs_features, actions)  # [num_critics, B, H]
        mean_q_values = jnp.mean(all_q_values, axis=0)  # [B, H]
        
        return mean_q_values
    
    @jax.jit
    def update_target_networks(self, params, target_params):
        """
        软更新target networks
        
        使用指数移动平均更新target参数：
        target_params = tau * params + (1 - tau) * target_params
        
        Args:
            params: 当前网络参数
            target_params: Target网络参数
            
        Returns:
            updated_target_params: 更新后的target参数
        """
        def update_single_param(param, target_param):
            return self.target_tau * param + (1 - self.target_tau) * target_param
        
        updated_target_params = jax.tree_map(
            update_single_param,
            params, 
            target_params
        )
        
        return updated_target_params


def create_critic_ensemble(num_critics: int,
                          obs_feat_dim: int, 
                          action_dim: int,
                          hidden_dims: Optional[List[int]] = None,
                          **kwargs) -> CriticEnsemble:
    """
    工厂函数：创建Critic ensemble
    
    Args:
        num_critics: Critic网络数量
        obs_feat_dim: 观察特征维度
        action_dim: 动作维度
        hidden_dims: 隐藏层维度，默认[256, 256]
        **kwargs: 其他配置参数
        
    Returns:
        CriticEnsemble实例
    """
    if hidden_dims is None:
        hidden_dims = [256, 256]
    
    logger.info(f"创建CriticEnsemble: {num_critics} critics, obs_dim={obs_feat_dim}, action_dim={action_dim}")
    
    ensemble = CriticEnsemble(
        num_critics=num_critics,
        obs_feat_dim=obs_feat_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        **kwargs
    )
    
    logger.info(f"✅ CriticEnsemble创建完成")
    return ensemble


def init_critic_ensemble_params(ensemble: CriticEnsemble,
                               rng: jax.random.PRNGKey,
                               obs_feat_dim: int,
                               action_horizon: int,
                               action_dim: int) -> Dict[str, Any]:
    """
    初始化Critic ensemble参数
    
    Args:
        ensemble: CriticEnsemble实例  
        rng: 随机数生成器
        obs_feat_dim: 观察特征维度
        action_horizon: 动作序列长度
        action_dim: 动作维度
        
    Returns:
        初始化的参数字典
    """
    # 创建示例输入用于参数初始化
    dummy_obs = jnp.zeros((1, obs_feat_dim))
    dummy_actions = jnp.zeros((1, action_horizon, action_dim))
    
    # 初始化参数
    params = ensemble.init(rng, dummy_obs, dummy_actions)
    
    # 同时初始化target network参数（复制主网络参数）
    target_params = jax.tree_map(lambda x: x.copy(), params)
    
    logger.info(f"✅ Critic ensemble参数初始化完成")
    
    return {
        'params': params,
        'target_params': target_params
    }


class CriticLoss:
    """Critic损失计算的辅助类"""
    
    @staticmethod
    @jax.jit
    def compute_td_error(current_q: jnp.ndarray,
                        target_q: jnp.ndarray,
                        rewards: jnp.ndarray,
                        terminals: jnp.ndarray,
                        discount: float = 0.99) -> jnp.ndarray:
        """
        计算时序差分误差
        
        Args:
            current_q: 当前Q值 [num_critics, B, H]
            target_q: Target Q值 [num_critics, B, H] 
            rewards: 奖励 [B, H]
            terminals: 终止标志 [B, H]
            discount: 折扣因子
            
        Returns:
            td_error: TD误差 [num_critics, B, H]
        """
        # 计算TD target
        td_targets = rewards + discount * target_q * (1.0 - terminals)  # [num_critics, B, H]
        
        # 计算TD error
        td_error = current_q - jax.lax.stop_gradient(td_targets)
        
        return td_error
    
    @staticmethod
    @jax.jit  
    def huber_loss(td_error: jnp.ndarray, delta: float = 1.0) -> jnp.ndarray:
        """
        计算Huber损失（对outlier更robust）
        
        Args:
            td_error: TD误差
            delta: Huber损失的阈值
            
        Returns:
            huber_loss: Huber损失
        """
        abs_error = jnp.abs(td_error)
        return jnp.where(
            abs_error <= delta,
            0.5 * td_error ** 2,
            delta * abs_error - 0.5 * delta ** 2
        )