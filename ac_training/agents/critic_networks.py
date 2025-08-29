"""
Critic Networks for ACRLPD + π₀ Integration.

This module implements Q-value estimation networks specifically designed for 
ACRLPD (Action-Chunked Reinforcement Learning with Prior Data) integration 
with π₀ models. The key features include:

- Ensemble of Q-networks for robust value estimation
- Multi-modal observation processing compatible with π₀
- Action sequence handling for Q-chunking 
- Target network management with soft updates
- Flax NNX implementation for π₀ framework compatibility
"""

import logging
from typing import Dict, Any, Tuple, Optional, Callable
import dataclasses

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

import openpi.models.model as _model
from openpi.shared import array_typing as at

# 🔑 Global initialization functions 
_KERNEL_INIT = nn.initializers.variance_scaling(1.0, "fan_in", "uniform")
_BIAS_INIT = nn.initializers.constant(0.0)

logger = logging.getLogger(__name__)
# 确保logger能正确输出到console
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@dataclasses.dataclass(frozen=True)
class CriticConfig:
    """Configuration for Critic Networks."""
    
    # Network architecture
    num_critics: int = 10                    # Ensemble size
    hidden_dims: Tuple[int, ...] = (256, 256, 256)  # Hidden layer dimensions
    dropout_rate: float = 0.1                # Dropout for regularization
    activation: str = "relu"                 # Activation function
    use_layer_norm: bool = True              # Apply LayerNorm for training stability
    
    # π₀ integration
    use_pi0_features: bool = True            # Use π₀ encoded features
    feature_fusion_method: str = "concat"    # "concat", "add", "mlp"
    
    # Training parameters
    target_update_tau: float = 0.005         # Soft update rate
    q_aggregation: str = "min"              # "min", "mean", "weighted"
    gradient_clip: float = 1.0              # Gradient clipping
    
    # Initialization
    kernel_init_scale: float = 1.0          # Kernel initialization scale
    bias_init: float = 0.0                  # Bias initialization
    
    def __post_init__(self):
        assert self.q_aggregation in ["min", "mean", "weighted"]
        assert self.feature_fusion_method in ["concat", "add", "mlp"]
        assert self.activation in ["relu", "swish", "gelu", "tanh"]


class SingleCriticNetwork(nn.Module):
    """Single Q-network for state-action value estimation."""
    
    config: CriticConfig
    observation_dim: int
    action_dim: int
    name: str = "critic"
    
    @nn.compact
    def __call__(
        self, 
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        *,
        train: bool = False
    ) -> jnp.ndarray:
        """Forward pass of the critic network."""
        # Activation function mapping
        activation_map = {
            "relu": nn.relu,
            "swish": nn.swish, 
            "gelu": nn.gelu,
            "tanh": nn.tanh,
        }
        activation_fn = activation_map[self.config.activation]
        
        # Concatenate observations and actions
        x = jnp.concatenate([observations, actions], axis=-1)
        
        # Feature fusion layer for multi-modal observations
        if self.config.use_pi0_features and self.config.feature_fusion_method == "mlp":
            x = nn.Dense(
                self.observation_dim, 
                kernel_init=_KERNEL_INIT,
                bias_init=_BIAS_INIT
            )(x)
        
        # Forward through network layers
        for i, hidden_dim in enumerate(self.config.hidden_dims):
            # Linear layer
            x = nn.Dense(
                hidden_dim, 
                kernel_init=_KERNEL_INIT,
                bias_init=_BIAS_INIT
            )(x)
            
            # LayerNorm before activation if enabled
            if self.config.use_layer_norm:
                x = nn.LayerNorm()(x)
            
            # Apply activation
            x = activation_fn(x)
            
            # Apply dropout if enabled
            if self.config.dropout_rate > 0:
                x = nn.Dropout(rate=self.config.dropout_rate)(x, deterministic=not train)
        
        # Output layer (single Q-value)
        q_value = nn.Dense(
            1, 
            kernel_init=_KERNEL_INIT,
            bias_init=_BIAS_INIT
        )(x)
        
        return q_value.squeeze(-1)  # [batch_size]


class CriticEnsemble(nn.Module):
    """Ensemble of Q-networks for robust value estimation."""
    
    config: CriticConfig
    observation_dim: int
    action_dim: int
    
    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        *,
        train: bool = False,
        aggregate: bool = True
    ) -> jnp.ndarray:
        """Forward pass through ensemble of critics using vmap."""
        # Create ensemble using vmap
        ensemble = nn.vmap(
            SingleCriticNetwork,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.config.num_critics,
        )
        
        # Get Q-values from each critic [num_critics, batch_size]
        q_values = ensemble(
            config=self.config,
            observation_dim=self.observation_dim,
            action_dim=self.action_dim
        )(observations, actions, train=train)
        
        if not aggregate:
            return q_values
        
        # Aggregate Q-values based on configuration
        if self.config.q_aggregation == "min":
            return jnp.min(q_values, axis=0)
        elif self.config.q_aggregation == "mean":
            return jnp.mean(q_values, axis=0)
        elif self.config.q_aggregation == "weighted":
            # Learnable aggregation weights
            weights = self.param(
                "aggregation_weights", 
                nn.initializers.ones, 
                (self.config.num_critics,)
            )
            normalized_weights = jax.nn.softmax(weights)
            return jnp.sum(q_values * normalized_weights[:, None], axis=0)
        else:
            return jnp.min(q_values, axis=0)  # Default to min


class CriticNetworks:
    """Complete Critic system with online and target networks."""
    
    def __init__(
        self,
        config: CriticConfig,
        observation_dim: int,
        action_dim: int,
        rngs  # Can be raw key
    ):
        self.config = config
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
        # Handle different rngs types
        if hasattr(rngs, 'online'):  # Check if it's nnx.Rngs
            online_raw_rng = rngs.online()
            target_raw_rng = rngs.target()
        else:
            # Split raw key
            online_raw_rng, target_raw_rng = jax.random.split(rngs, 2)
        
        # Initialize online and target networks
        self.online_critics = CriticEnsemble(
            config=config,
            observation_dim=observation_dim,
            action_dim=action_dim
        )
        
        self.target_critics = CriticEnsemble(
            config=config,
            observation_dim=observation_dim,
            action_dim=action_dim
        )
        
        # Initialize parameters
        dummy_obs = jnp.zeros((1, observation_dim))
        dummy_actions = jnp.zeros((1, action_dim))
        
        self.online_params = self.online_critics.init(
            online_raw_rng, dummy_obs, dummy_actions
        )
        self.target_params = self.target_critics.init(
            target_raw_rng, dummy_obs, dummy_actions
        )
        
        # Initialize target with same weights as online
        self.sync_target_networks()
    
    def sync_target_networks(self):
        """Synchronize target networks with online networks."""
        # Copy parameters from online to target (hard update)
        self.target_params = jax.tree_util.tree_map(lambda x: x, self.online_params)
    
    def soft_update_target_networks(self, tau: Optional[float] = None):
        """Soft update of target networks."""
        if tau is None:
            tau = self.config.target_update_tau
        
        # Soft update: target = (1-tau) * target + tau * online
        def soft_update_param(target_param, online_param):
            return (1.0 - tau) * target_param + tau * online_param
        
        self.target_params = jax.tree_map(
            soft_update_param, self.target_params, self.online_params
        )
    
    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        *,
        use_target: bool = False,
        train: bool = False,
        aggregate: bool = True,
        rng: Optional[jax.Array] = None
    ) -> jnp.ndarray:
        """Forward pass through critic networks."""
        if use_target:
            return self.target_critics.apply(
                self.target_params, observations, actions, 
                train=train, aggregate=aggregate
            )
        else:
            return self.online_critics.apply(
                self.online_params, observations, actions, 
                train=train, aggregate=aggregate
            )




def create_critic_networks(
    config: CriticConfig,
    pi0_model: Any,
    action_horizon: int,
    action_dim: int,
    rngs: jnp.ndarray,
    pi0_config: Any = None
) -> CriticNetworks:
    """
    Factory function to create critic networks with static dimension calculation.
    
    This function creates Critic networks that accept pre-encoded features rather
    than raw observations. The π₀ feature extraction is handled in the training loop.
    
    Args:
        config: Critic configuration
        pi0_model: π₀ model (used only for dimension extraction)
        action_horizon: Length of action sequences
        action_dim: Dimension of individual actions
        rngs: Random number generator array
        pi0_config: π₀ configuration for dimension calculation
        
    Returns:
        CriticNetworks instance ready for FSDP training
    """
    
    # 🔧 静态维度计算：从π₀配置获取特征维度，避免运行时π₀调用
    # 观测特征 = π₀视觉特征 + 状态特征
    
    # 获取LLM维度（视觉特征维度）
    llm_dim = None
    
    # 方法1：从π₀模型配置获取（如果可用）
    if hasattr(pi0_model, 'config') and hasattr(pi0_model.config, 'llm_dim'):
        llm_dim = pi0_model.config.llm_dim
        logger.info(f"✅ 从π₀模型配置获取: llm_dim={llm_dim}")
    elif hasattr(pi0_model, 'config') and hasattr(pi0_model.config, 'width'):
        llm_dim = pi0_model.config.width  # Gemma模型使用width作为LLM维度
        logger.info(f"✅ 从π₀模型width获取: llm_dim={llm_dim}")
    
    # 方法2：从传入的π₀配置获取
    elif pi0_config is not None:
        # 根据Gemma变体确定width
        if pi0_config.paligemma_variant == "gemma_2b":
            llm_dim = 2048  # Gemma 2B的width
        elif pi0_config.paligemma_variant == "gemma_300m":
            llm_dim = 1024  # Gemma 300M的width
        elif pi0_config.paligemma_variant == "dummy":
            llm_dim = 64   # Dummy variant的width
        else:
            raise ValueError(f"未知的paligemma_variant: {pi0_config.paligemma_variant}")
        logger.info(f"✅ 从π₀配置variant获取: {pi0_config.paligemma_variant} -> llm_dim={llm_dim}")
    
    # 方法3：Fallback - 如果以上都失败
    else:
        raise ValueError(
            f"无法从π₀配置获取LLM维度。"
            f"pi0_model.config: {getattr(pi0_model, 'config', 'None')}, "
            f"pi0_config: {pi0_config}"
        )
    
    # 计算总的observation维度
    state_dim = action_dim  # 状态特征维度通常等于action_dim
    observation_dim = llm_dim + state_dim  # concat fusion
    logger.info(f"📊 维度计算: llm_dim={llm_dim}, state_dim={state_dim}, total={observation_dim}")
    
    # Flatten action dimension for critic input
    flattened_action_dim = action_horizon * action_dim
    logger.info(f"🔍 [create_critic_networks] 输入参数: action_horizon={action_horizon}, action_dim={action_dim}")
    logger.info(f"🔍 [create_critic_networks] 计算结果: flattened_action_dim={flattened_action_dim}")
    
    # Create critic networks using raw RNG key to avoid pytree issues
    critic_networks = CriticNetworks(
        config=config,
        observation_dim=observation_dim,
        action_dim=flattened_action_dim,
        rngs=rngs  # Pass raw key directly
    )
    
    logger.info(f"✅ [create_critic_networks] 创建完成: obs_dim={observation_dim}, flattened_action_dim={flattened_action_dim}")
    logger.info(f"🎯 [create_critic_networks] Critic期望输入总维度: {observation_dim + flattened_action_dim}")
    logger.info("ℹ️  Critic expects pre-encoded features: [vision_features + state_features]")
    
    return critic_networks


# Default configurations for different scenarios
DEFAULT_CRITIC_CONFIG = CriticConfig()

LARGE_ENSEMBLE_CONFIG = CriticConfig(
    num_critics=20,
    hidden_dims=(512, 512, 256),
    dropout_rate=0.15
)

FAST_INFERENCE_CONFIG = CriticConfig(
    num_critics=5,
    hidden_dims=(256, 256),
    dropout_rate=0.05
)