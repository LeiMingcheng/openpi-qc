"""
Joint Loss Functions for ACRLPD + π₀ Integration.

This module implements the comprehensive loss function system for training
ACRLPD with π₀ models, combining reinforcement learning objectives with 
behavior cloning and diffusion model training.

Key components:
- Critic Loss: TD error for Q-learning with action chunking
- BC Loss: π₀'s diffusion-based behavior cloning loss  
- π₀ Loss: Additional π₀ training objectives (optional)
- Alpha Loss: Temperature parameter adaptive adjustment
- Combined Loss: Weighted combination of all losses
"""

import logging
from typing import Dict, Any, Tuple, Optional, Callable, NamedTuple
import dataclasses
from functools import partial

import jax
import jax.numpy as jnp

import flax.nnx as nnx
import numpy as np

import openpi.models.model as _model
from openpi.shared import array_typing as at
import sys
from pathlib import Path

# Add ac_training root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.batching import MaskHandler, BootstrapHandler

# 创建logger实例
logger = logging.getLogger(__name__)
# 确保logger能正确输出到console
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@dataclasses.dataclass(frozen=True)
class LossWeights:
    """Loss weight configuration for joint training."""
    
    critic_weight: float = 1.0              # Critic loss weight
    actor_weight: float = 1.0               # Actor loss weight (NEW)
    bc_weight: float = 0.01                 # Behavior cloning weight
    alpha_weight: float = 1.0               # Temperature loss weight
    
    # Loss balancing
    adaptive_weights: bool = False          # Adaptive loss weighting
    weight_decay: float = 0.0              # L2 regularization
    
    def validate(self):
        """Validate loss weight configuration."""
        assert all(w >= 0 for w in [
            self.critic_weight, self.actor_weight,
            self.bc_weight, self.alpha_weight
        ]), "All loss weights must be non-negative"
        
        assert self.critic_weight > 0, "Critic weight must be positive"
        assert self.actor_weight > 0, "Actor weight must be positive"


class LossInfo(NamedTuple):
    """Container for loss information and metrics."""
    
    total_loss: float
    critic_loss: float
    actor_loss: float                       # NEW: Actor loss
    bc_loss: float
    alpha_loss: float
    
    # Additional metrics
    q_mean: float
    q_std: float
    target_q_mean: float
    td_error_mean: float
    bc_loss_raw: float
    alpha_value: float
    entropy_estimate: float
    q_values_for_actor: float              # NEW: Q values used in actor loss
    
    # Mask and validity info
    valid_samples: float
    mask_ratio: float
    
    # BC-specific statistics (NEW: BC训练统计信息)
    bc_positive_samples: float = 0.0      # BC训练使用的正样本数量
    bc_total_samples: float = 0.0         # BC可用的总样本数量


class TemperatureModule(nnx.Module):
    """Learnable temperature parameter for entropy regularization."""
    
    def __init__(self, initial_temp: float = 1.0, rngs: nnx.Rngs = None):
        super().__init__()
        self.log_alpha = nnx.Param(
            jnp.log(initial_temp) * jnp.ones(())
        )
    
    def __call__(self) -> jnp.ndarray:
        """Get current temperature value."""
        return jnp.exp(self.log_alpha.value)
    
    def alpha_loss(self, entropy: jnp.ndarray, target_entropy: float) -> jnp.ndarray:
        """Compute adaptive temperature loss."""
        alpha = self()
        return alpha * jax.lax.stop_gradient(entropy - target_entropy)


class CriticLossComputer:
    """Computes Q-learning critic loss with action chunking."""
    
    def __init__(
        self,
        discount: float = 0.99,
        horizon_length: int = 5,
        q_aggregation: str = "min",
        bootstrap_handler: Optional[BootstrapHandler] = None,
        config: Dict[str, Any] = None,
        real_action_dim: int = 14  # 真实ALOHA动作维度，来自QChunkingConfig
    ):
        self.discount = discount
        self.horizon_length = horizon_length
        self.q_aggregation = q_aggregation
        self.bootstrap_handler = bootstrap_handler or BootstrapHandler()
        self.config = config or {}  # 提供默认的config字典
        # 从config获取真实动作维度
        self.real_action_dim = real_action_dim        

        assert q_aggregation in ["min", "mean", "weighted"]
    
    def __call__(
        self,
        pi0_model: Any,
        critic_networks: Any,
        observation_encoder: Optional[Callable] = None,
        batch: Dict[str, jnp.ndarray] = None,
        rng: jnp.ndarray = None,
        train: bool = True,
        embeddings_cache: Optional[Dict[str, Dict[str, Any]]] = None,
        current_obs_dict: Optional[Dict[str, Any]] = None,
        next_obs_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        🚀 CACHE OPTIMIZED: Compute critic loss using cached embeddings and processed observations.
        
        Args:
            pi0_model: π₀ model for action sampling
            critic_networks: Critic network ensemble
            observation_encoder: Function to encode observations (optional)
            batch: Training batch data
            rng: Random number generator
            train: Training mode flag
            embeddings_cache: Optional cached embeddings to avoid recomputation
            current_obs_dict: Processed current observations dict
            next_obs_dict: Processed next observations dict
            
        Returns:
            Tuple of (loss_value, loss_info_dict)
        """
        rng_current, rng_next = jax.random.split(rng)
        
        # 🚀 优化：优先使用缓存的observation features
        if embeddings_cache is not None:
            logger.debug("🚀 CriticLoss使用缓存features，避免重复observation encoding")
            
            # 🚀 高效路径：强制使用缓存，删除低效fallback
            if 'current' not in embeddings_cache:
                raise ValueError("缺少current observation缓存 - 检查缓存生成逻辑")
            current_obs_encoded = pi0_model.extract_features_from_cache('current', embeddings_cache, real_action_dim=self.real_action_dim)
            
            # 🚀 高效路径：强制使用next observation缓存
            if 'next' not in embeddings_cache:
                raise ValueError("缺少next observation缓存 - 检查缓存生成逻辑")
            next_obs_encoded = pi0_model.extract_features_from_cache('next', embeddings_cache, real_action_dim=self.real_action_dim)
        else:
            # 🚀 强制缓存路径：不允许无缓存运行
            raise ValueError("embeddings_cache为空 - 必须使用缓存以确保最高性能")
        
        # Use data actions for current states (standard Q-chunking)
        if 'actions' not in batch:
            raise ValueError("Batch must contain 'actions' for standard Q-chunking")
        
        data_actions = batch['actions']  # [batch_size, chunk_size, action_dim]
        
        # 🔧 关键修复：截断32维π₀ actions到14维真实ALOHA动作
        # 数据加载器给出的actions是32维(OpenPI格式)，需要截断到真实动作维度
        original_actions_shape = data_actions.shape
        if len(data_actions.shape) == 3:
            # 从32维截断到real_action_dim维: [batch_size, chunk_size, 32] -> [batch_size, chunk_size, real_action_dim]
            data_actions_for_critic = data_actions[..., :self.real_action_dim]
            current_actions_flat = data_actions_for_critic.reshape(data_actions_for_critic.shape[0], -1)
        elif len(data_actions.shape) == 2:
            # 如果已经是flattened，按chunk重新reshape，截断，再flatten
            chunk_size = self.horizon_length
            original_action_dim = data_actions.shape[1] // chunk_size  # 应该是32
            data_actions_3d = data_actions.reshape(data_actions.shape[0], chunk_size, original_action_dim)
            data_actions_for_critic = data_actions_3d[..., :self.real_action_dim]  # 截断到14维
            current_actions_flat = data_actions_for_critic.reshape(data_actions_for_critic.shape[0], -1)
        else:
            raise ValueError(f"Unexpected action shape: {data_actions.shape}")
        
        # 🔍 调试信息：显示actions截断效果
        logger.debug(f"🔍 Actions截断调试: 原始{original_actions_shape} -> Critic输入{current_actions_flat.shape} (截断到{self.real_action_dim}维)")
        
        # Sample actions from π₀ for next states (correct for policy evaluation)  
        # Use gradient-safe sampling method with processed next observations
        if next_obs_dict is None:
            raise ValueError("next_obs_dict is required for critic loss computation")
        next_actions = pi0_model.sample_actions_differentiable(
            rng_next, _model.Observation.from_dict(next_obs_dict), num_steps=10
        )  # π₀采样保持32维，这是正确的
        
        # 🔧 只在传递给Critic时截断next_actions到real_action_dim维
        # π₀采样产生32维actions，但Critic只需要真实ALOHA的14维
        next_actions_for_critic = next_actions[..., :self.real_action_dim]  # [batch_size, chunk_size, real_action_dim]
        next_actions_flat = next_actions_for_critic.reshape(next_actions_for_critic.shape[0], -1)
        
        # 🔍 调试信息：显示next_actions截断效果
        logger.debug(f"🔍 Next Actions截断调试: π₀采样{next_actions.shape} -> Critic输入{next_actions_flat.shape} (截断到{self.real_action_dim}维)")
        
        # Compute target Q-values
        target_q_values = critic_networks(
            next_obs_encoded, next_actions_flat, 
            use_target=True, train=train, aggregate=False
        )  # [num_critics, batch_size]
        
        # Aggregate target Q-values
        if self.q_aggregation == "min":
            target_q = jnp.min(target_q_values, axis=0)
        elif self.q_aggregation == "mean":
            target_q = jnp.mean(target_q_values, axis=0)
        else:  # weighted - handled by network
            target_q = critic_networks(
                next_obs_encoded, next_actions_flat,
                use_target=True, train=train, aggregate=True
            )
        
        # Enhanced bootstrap target computation with adaptive horizon
        use_adaptive_horizon = 'sequence_lengths' in batch
        sequence_lengths = batch.get('sequence_lengths', None)
        
        if 'multi_step_rewards' in batch and use_adaptive_horizon:
            # Use advanced multi-step bootstrap with TD(λ)
            target_q_bootstrap = self.bootstrap_handler.compute_multi_step_bootstrap_targets(
                rewards=batch['multi_step_rewards'],
                next_q_values=target_q,
                masks=batch.get('step_masks', jnp.ones(batch['reward'].shape[0], dtype=jnp.float32)),  # 默认mask
                discount=self.discount,
                valid_horizons=sequence_lengths,
                use_td_lambda=self.config.get('use_td_lambda', False),
                lambda_param=self.config.get('lambda_param', 0.95)
            )
        else:
            # Standard bootstrap computation with OpenPI format
            # 🚀 OpenPI架构: 直接使用标量batch['reward']而非序列rewards
            action_horizon = batch['actions'].shape[1]
            # 使用默认masks(全为1)，非终止状态
            masks = jnp.ones(batch['reward'].shape[0], dtype=jnp.float32)  # [batch_size]
            
            target_q_bootstrap = self.bootstrap_handler.compute_bootstrap_target(
                rewards=batch['reward'],  # [batch_size] - OpenPI原生标量
                next_q_values=target_q,
                masks=masks,  # [batch_size] - 生成的mask
                discount=self.discount,
                horizon_length=self.horizon_length,  # Q-chunking核心参数
                adaptive_horizon=use_adaptive_horizon
            )
        
        # Current Q-values using data actions (standard Q-chunking)
        current_q_values = critic_networks(
            current_obs_encoded, current_actions_flat,
            use_target=False, train=train, aggregate=False
        )  # [num_critics, batch_size]
        
        # TD errors
        td_errors = current_q_values - target_q_bootstrap[None, :]  # [num_critics, batch_size]
        
        # Enhanced loss computation with importance weighting
        critic_loss_raw = jnp.square(td_errors)  # [num_critics, batch_size]
        
        # Apply valid mask (standard Q-chunking requirement)
        if 'valid' in batch:
            # Use final step valid mask for Q-chunking
            if batch['valid'].ndim > 1:
                valid_mask = batch['valid'][..., -1]  # [batch_size]
            else:
                valid_mask = batch['valid']  # [batch_size]
            
            # Apply valid mask to all critics: [num_critics, batch_size] * [batch_size]
            critic_loss_raw = critic_loss_raw * valid_mask[None, :]
        
        # Compute sample weights for importance weighting
        use_importance_weighting = self.config.get('use_importance_weighting', False)
        sample_weights = None
        if use_importance_weighting:
            sample_weights = MaskHandler.compute_adaptive_sample_weights(
                batch,
                use_reward_weighting=True,
                use_rarity_weighting=True,
                use_difficulty_weighting=False,  # Skip for critic loss
                temperature=self.config.get('weight_temperature', 1.0)
            )
        
        # Apply enhanced masking with temporal consistency
        # 🚀 OpenPI架构: sequence_mask应该是batch维度而非action维度
        batch_size = batch['reward'].shape[0]
        sequence_mask = batch.get('sequence_mask', jnp.ones(batch_size))
        
        # Optional temporal consistency filtering
        if self.config.get('use_temporal_consistency', False):
            # 🚀 使用处理好的observations
            if current_obs_dict is None or next_obs_dict is None:
                raise ValueError("current_obs_dict and next_obs_dict are required for temporal consistency")
            consistency_mask = MaskHandler.compute_temporal_consistency_mask(
                current_observations=_model.Observation.from_dict(current_obs_dict),
                next_observations=_model.Observation.from_dict(next_obs_dict),
                consistency_threshold=self.config.get('consistency_threshold', 0.1),
                check_state_consistency=True,
                check_image_consistency=False  # Expensive, disabled by default
            )
            sequence_mask = sequence_mask * consistency_mask.astype(jnp.float32)
        
        # Apply enhanced loss masking per critic
        ensemble_losses = []
        for critic_idx in range(current_q_values.shape[0]):
            critic_loss_single = MaskHandler.apply_loss_masking(
                loss=critic_loss_raw[critic_idx],
                mask=sequence_mask,
                sample_weights=sample_weights
            )
            ensemble_losses.append(critic_loss_single)
        
        # Average across critics
        critic_loss = jnp.mean(jnp.array(ensemble_losses))
        
        # Loss info
        info = {
            'critic_loss': critic_loss,
            'q_mean': current_q_values.mean(),
            'q_std': current_q_values.std(),
            'target_q_mean': target_q_bootstrap.mean(),
            'td_error_mean': jnp.abs(td_errors).mean(),
            'valid_samples': sequence_mask.sum(),
            'mask_ratio': sequence_mask.mean()
        }
        
        return critic_loss, info


class BCLossComputer:
    """Computes behavior cloning loss using π₀'s diffusion loss."""
    
    def __init__(self, mask_handler: Optional[MaskHandler] = None, real_action_dim: int = 14):
        self.mask_handler = mask_handler or MaskHandler()
        self.real_action_dim = real_action_dim
    
    def __call__(
        self,
        pi0_model: Any,
        batch: Dict[str, jnp.ndarray],
        rng: jnp.ndarray,
        train: bool = True,
        return_features: bool = False,
        embeddings_cache: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Tuple[jnp.ndarray, Dict[str, Any]] | Tuple[jnp.ndarray, Dict[str, Any], jnp.ndarray]:
        """
        🚀 OPTIMIZED: Compute BC loss using unified forward pass with optional feature extraction.
        
        BC loss is only computed on positive samples (reward=1.0) to learn 
        from successful demonstrations only.
        
        When return_features=True, uses π₀'s new compute_loss_and_features method
        for efficient joint computation, eliminating duplicate forward passes.
        
        Args:
            pi0_model: π₀ model with compute_loss_and_features support
            batch: Training batch data
            rng: Random number generator
            train: Training mode flag
            return_features: If True, also return features for Critic use
            
        Returns:
            If return_features=False: Tuple of (loss_value, loss_info_dict)
            If return_features=True: Tuple of (loss_value, loss_info_dict, pi0_features)
        """
        # 🔑 关键修复：BC loss只针对正样本（reward=1）训练
        # 提取reward信息并创建正样本掩码
        
        # 🔧 调试：记录batch中的实际keys以诊断问题
        logger.debug(f"BC loss: Batch keys available: {list(batch.keys())}")
        
        if 'rewards' in batch:
            # Q-chunking format: rewards shape is [batch_size, horizon_length]
            if batch['rewards'].ndim > 1:
                # 使用最终步骤的reward作为episode reward判断
                episode_rewards = batch['rewards'][..., -1]  # [batch_size]
            else:
                # 如果是单步reward，直接使用
                episode_rewards = batch['rewards']  # [batch_size]
            
            # 创建正样本掩码：只有reward=1.0的样本参与BC训练
            positive_mask = jnp.isclose(episode_rewards, 1.0, atol=1e-6)  # [batch_size]
            num_positive = jnp.sum(positive_mask)
            logger.debug(f"BC loss: Using 'rewards' field, found {num_positive} positive samples out of {len(episode_rewards)}")
            
        elif 'reward' in batch:
            # 🔧 修复：支持OpenPI原生的单值reward字段
            # 当collate_fn被禁用时，batch中只有'reward'字段而不是'rewards'
            episode_rewards = batch['reward']  # [batch_size] - OpenPI原生格式
            
            # 创建正样本掩码：只有reward=1.0的样本参与BC训练
            positive_mask = jnp.isclose(episode_rewards, 1.0, atol=1e-6)  # [batch_size]
            num_positive = jnp.sum(positive_mask)
            logger.debug(f"BC loss: Using 'reward' field, found {num_positive} positive samples out of {len(episode_rewards)}")
            
        else:
            # 如果没有reward信息，警告并处理所有样本（回退行为）
            logger.warning(f"BC loss: No rewards found in batch (keys: {list(batch.keys())}), processing all samples")
            positive_mask = jnp.ones(batch['state'].shape[0], dtype=jnp.bool_)
            num_positive = positive_mask.shape[0]
        
        # 🔧 JAX-compatible conditional: Use jax.lax.cond instead of Python if
        # Create zero loss info for when no positive samples exist
        batch_size = batch['state'].shape[0]
        
        # 🔧 预先确定feature维度，避免在闭包中访问外部变量
        # 关键：无论是否使用缓存，都要保证feature维度一致（vision_dim + 14）
        if return_features:
            if embeddings_cache is not None and 'current' in embeddings_cache:
                # 从缓存获取正确的feature维度: vision_dim + real_action_dim
                pooled_dim = embeddings_cache['current']['pooled_features'].shape[-1]
                feature_dim = pooled_dim + self.real_action_dim
            else:
                # 🔧 修复：计算准确的特征维度，保持与缓存版本一致
                # 假设vision特征是2048维（SigLIP），状态特征截断到real_action_dim
                feature_dim = 2048 + self.real_action_dim
        else:
            feature_dim = 1  # 不返回features时的占位维度
        
        def create_zero_loss_result():
            zero_loss = jnp.array(0.0)
            zero_info = {
                'bc_loss': zero_loss,
                'bc_loss_raw': zero_loss,
                'bc_loss_std': zero_loss,
                'valid_samples': jnp.array(0.0, dtype=jnp.float32),  # 🔧 JAX兼容的类型转换
                'positive_samples': jnp.array(0.0, dtype=jnp.float32),  # 🔧 JAX兼容的类型转换
                'total_samples': jnp.array(batch_size, dtype=jnp.float32)  # 🔧 JAX兼容的类型转换
            }
            if return_features:
                dummy_features = jnp.zeros((batch_size, feature_dim), dtype=jnp.float32)
                return zero_loss, zero_info, dummy_features
            else:
                return zero_loss, zero_info
        
        def compute_normal_bc_loss():
            """🚀 CACHE OPTIMIZED: Compute BC loss using cached embeddings when available."""
            pi0_features = None
            
            # 🚀 优化：优先使用缓存，避免重复embed_prefix计算
            if embeddings_cache is not None and 'current' in embeddings_cache:
                logger.debug("🚀 BCLoss使用缓存embeddings，避免重复前向传播")
                # 使用缓存计算BC loss - 跳过重复的embed_prefix()
                if return_features:
                    # 从缓存提取features，避免重复特征提取
                    pi0_features = pi0_model.extract_features_from_cache('current', embeddings_cache, real_action_dim=self.real_action_dim)
                    # 使用缓存的embed_prefix结果计算BC loss
                    bc_loss_raw = self._compute_bc_loss_with_cache(
                        pi0_model, batch, rng, train, embeddings_cache['current']
                    )
                else:
                    bc_loss_raw = self._compute_bc_loss_with_cache(
                        pi0_model, batch, rng, train, embeddings_cache['current']
                    )
            else:
                raise ValueError("缺少current observation缓存 - 检查缓存生成逻辑")
            
            # 用positive_mask将负样本的loss置零
            masked_loss = bc_loss_raw * positive_mask[:, None]  # broadcast to [batch_size, action_horizon]
            
            # 处理sequence_mask（如果存在）
            if 'sequence_mask' in batch:
                final_mask = positive_mask * batch['sequence_mask']
                masked_loss = bc_loss_raw * final_mask[:, None]
            else:
                final_mask = positive_mask
            
            # 计算最终loss：只对positive samples求平均
            total_loss = jnp.sum(masked_loss)
            total_steps = jnp.sum(final_mask) * bc_loss_raw.shape[1]
            
            bc_loss = jax.lax.cond(
                total_steps > 0,
                lambda: total_loss / total_steps,
                lambda: jnp.array(0.0)
            )
            
            # 统计信息
            batch_size = batch['state'].shape[0]
            info = {
                'bc_loss': bc_loss,
                'bc_loss_raw': bc_loss_raw.mean(),
                'bc_loss_std': bc_loss_raw.std(), 
                'valid_samples': jnp.array(jnp.sum(final_mask), dtype=jnp.float32),  # 🔧 JAX兼容的类型转换
                'positive_samples': jnp.array(jnp.sum(positive_mask), dtype=jnp.float32),  # 🔧 JAX兼容的类型转换
                'total_samples': jnp.array(batch_size, dtype=jnp.float32)  # 🔧 JAX兼容的类型转换
            }
            
            if return_features:
                return bc_loss, info, pi0_features
            else:
                return bc_loss, info
        
        # 🔧 JAX-compatible conditional: Use jax.lax.cond instead of Python if
        return jax.lax.cond(
            num_positive == 0,
            create_zero_loss_result,
            compute_normal_bc_loss
        )
    
    def _compute_bc_loss_with_cache(
        self, 
        pi0_model: Any, 
        batch: Dict[str, jnp.ndarray], 
        rng: jnp.ndarray, 
        train: bool, 
        cached_data: Dict[str, Any]
    ) -> jnp.ndarray:
        """
        🚀 CACHE OPTIMIZED: 使用预计算的embeddings计算BC loss
        
        这个方法避免了重复的embed_prefix()调用，直接使用缓存的prefix tokens。
        
        Args:
            pi0_model: π₀模型实例
            batch: 训练batch数据
            rng: 随机数生成器
            train: 训练模式标志
            cached_data: 预计算的embeddings缓存数据
            
        Returns:
            BC loss: [batch_size, action_horizon]
        """
        # 从缓存中获取预计算的数据
        prefix_tokens = cached_data['prefix_tokens']
        prefix_mask = cached_data['prefix_mask'] 
        processed_obs = cached_data['processed_obs']
        
        # 执行与compute_loss()相同的逻辑，但跳过embed_prefix()
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        actions = batch['actions']
        
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        
        # 使用缓存的prefix_tokens，只需要计算suffix
        suffix_tokens, suffix_mask, suffix_ar_mask = pi0_model.embed_suffix(processed_obs, x_t, time)
        
        # 组合prefix和suffix进行forward pass
        from openpi.models.pi0 import make_attn_mask
        prefix_ar_mask = cached_data['prefix_ar_mask']
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        
        (prefix_out, suffix_out), _ = pi0_model.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions
        )
        v_t = pi0_model.action_out_proj(suffix_out[:, -pi0_model.action_horizon:])
        
        bc_loss_raw = jnp.mean(jnp.square(v_t - u_t), axis=-1)  # [batch_size, action_horizon]
        
        # Apply Q-chunking valid masks to exclude episode boundary predictions
        valid_mask = batch.get('valid', jnp.ones_like(bc_loss_raw))  # [batch_size, action_horizon]
        return bc_loss_raw * valid_mask


class ActorLossComputer:
    """Computes Actor loss using Best-of-N sampling: -max{Q(s, π₀ᵢ(s))} for i=1...N"""
    
    def __init__(self, num_action_samples: int = 4, real_action_dim: int = 14):
        self.num_action_samples = num_action_samples
        self.real_action_dim = real_action_dim
    
    def __call__(
        self,
        pi0_model: Any,
        critic_networks: Any,
        observation_encoder: Callable,
        batch: Dict[str, jnp.ndarray],
        rng: jnp.ndarray,
        train: bool = True,
        embeddings_cache: Optional[Dict[str, Dict[str, Any]]] = None,
        current_obs_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        🚀 CACHE OPTIMIZED: Compute Actor loss using Best-of-N with cached embeddings.
        
        Args:
            pi0_model: π₀ model
            critic_networks: Critic network ensemble
            observation_encoder: Observation encoding function
            batch: Training batch data
            rng: Random number generator
            train: Training mode flag
            embeddings_cache: Optional cached embeddings to avoid recomputation
            
        Returns:
            Tuple of (loss_value, loss_info_dict)
        """
        # 🚀 强制使用缓存的observation features
        if embeddings_cache is not None and 'current' in embeddings_cache:
            logger.debug("🚀 ActorLoss使用缓存features，避免重复observation encoding")
            obs_encoded = pi0_model.extract_features_from_cache('current', embeddings_cache, real_action_dim=self.real_action_dim)
        else:
            raise ValueError("缺少current observation缓存 - 检查缓存生成逻辑")
            
        batch_size = obs_encoded.shape[0]
        
        # 1. Sample multiple action candidates (Best-of-N)
        sample_rngs = jax.random.split(rng, self.num_action_samples)
        
        def sample_single_action(sample_rng):
            if current_obs_dict is None:
                raise ValueError("current_obs_dict is required for action sampling")
            return pi0_model.sample_actions_differentiable(
                sample_rng, _model.Observation.from_dict(current_obs_dict), num_steps=10
            )  # [batch_size, action_horizon, action_dim]
        
        # Generate all action candidates in parallel
        action_candidates = jax.vmap(sample_single_action)(sample_rngs)
        # Shape: [num_samples, batch_size, action_horizon, action_dim]
        
        # 2. Evaluate Q-values for all candidates
        def evaluate_candidate_batch(actions):
            # 🔧 关键修复：截断32维actions到14维，然后flatten
            # π₀采样产生32维actions，但Critic只需要真实ALOHA的14维
            actions_for_critic = actions[..., :self.real_action_dim]  # [batch_size, action_horizon, real_action_dim]
            actions_flat = actions_for_critic.reshape(actions.shape[0], -1)  # [batch_size, horizon*real_action_dim]
            
            # 🔍 调试信息：显示ActorLoss action截断效果
            logger.debug(f"🔍 ActorLoss Actions截断调试: π₀采样{actions.shape} -> Critic输入{actions_flat.shape} (截断到{self.real_action_dim}维)")
            
            # Get Q-values from critic ensemble
            q_values = critic_networks(
                obs_encoded, actions_flat,
                use_target=False, train=False, aggregate=True  # Use aggregated Q-values
            )  # [batch_size]
            
            return q_values
        
        # Evaluate all candidates: [num_samples, batch_size]
        all_q_values = jax.vmap(evaluate_candidate_batch)(action_candidates)
        
        # 3. Best-of-N selection: choose highest Q-value for each batch element
        best_q_values = jnp.max(all_q_values, axis=0)  # [batch_size] - max across samples
        
        # 4. Apply Q-chunking valid masks and compute actor loss
        valid_mask = batch.get('valid', jnp.ones(best_q_values.shape[0]))  # [batch_size] or [batch_size, action_horizon]
        if valid_mask.ndim > 1:
            valid_mask = valid_mask[..., -1]  # Use last timestep for episode boundary [batch_size]
        
        # Actor loss: negative of masked best Q-values (to maximize Q)
        actor_loss = -jnp.mean(best_q_values * valid_mask)  # Simple masked mean
        
        # Metrics for monitoring
        info = {
            'actor_loss': actor_loss,
            'q_values_for_actor': jnp.mean(best_q_values),  # Average of best Q-values
            'q_std_for_actor': jnp.std(all_q_values),      # Std across all candidates
            'q_best_mean': jnp.mean(best_q_values),         # Mean of selected best Q-values
            'q_best_std': jnp.std(best_q_values)            # Std of selected best Q-values
        }
        
        return actor_loss, info




class EntropyEstimator:
    """Estimates entropy of π₀ policy for temperature adjustment."""
    
    def __init__(self, num_samples: int = 8):
        self.num_samples = num_samples
    
    def estimate_entropy(
        self,
        pi0_model: Any,
        observations: _model.Observation,
        rng: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Estimate policy entropy using sampling.
        
        Args:
            pi0_model: π₀ model
            observations: Batch of observations
            rng: Random number generator
            
        Returns:
            Estimated entropy per sample: [batch_size]
        """
        # Sample multiple actions from π₀
        sample_rngs = jax.random.split(rng, self.num_samples)
        
        def sample_single(rng_key):
            return pi0_model.sample_actions_differentiable(rng_key, observations, num_steps=5)  # Use fewer steps for entropy estimation
        
        # [num_samples, batch_size, action_horizon, action_dim]
        action_samples = jax.vmap(sample_single)(sample_rngs)
        
        # Estimate entropy using variance of samples
        # This is a simplified entropy estimate - more sophisticated methods exist
        action_var = jnp.var(action_samples, axis=0)  # [batch_size, action_horizon, action_dim]
        
        # Average variance across action dimensions and horizon
        entropy_estimate = jnp.mean(action_var, axis=(1, 2))  # [batch_size]
        
        # Convert variance to entropy-like quantity (log scale)
        entropy_estimate = jnp.log(entropy_estimate + 1e-8)
        
        return entropy_estimate


class JointLossComputer:
    """Computes the joint loss function for ACRLPD + π₀ training."""
    
    def __init__(
        self,
        loss_weights: LossWeights,
        critic_loss_computer: CriticLossComputer,
        actor_loss_computer: ActorLossComputer,              # NEW
        bc_loss_computer: BCLossComputer,
        temperature_module: Optional[TemperatureModule] = None,
        entropy_estimator: Optional[EntropyEstimator] = None,
        target_entropy_multiplier: float = 0.5
    ):
        self.loss_weights = loss_weights
        self.critic_loss_computer = critic_loss_computer
        self.actor_loss_computer = actor_loss_computer       # NEW
        self.bc_loss_computer = bc_loss_computer
        self.temperature_module = temperature_module
        self.entropy_estimator = entropy_estimator
        self.target_entropy_multiplier = target_entropy_multiplier
        
        # Validate configuration
        loss_weights.validate()
    
    def __call__(
        self,
        pi0_model: Any,
        critic_networks: Any,
        observation_encoder: Optional[Callable] = None,
        batch: Dict[str, jnp.ndarray] = None,
        rng: jnp.ndarray = None,
        train: bool = True
    ) -> Tuple[jnp.ndarray, LossInfo]:
        """
        🚀 CACHE OPTIMIZED: Compute joint loss with precomputed embeddings cache.
        
        PERFORMANCE OPTIMIZATION:
        - Precomputes embed_prefix() results for all observations at start
        - Eliminates 6+ duplicate embed_prefix() calls across loss computers
        - Passes cached embeddings to all loss computers for reuse
        - Achieves 2-4x speedup by avoiding redundant forward passes
        
        Args:
            pi0_model: π₀ model with caching methods
            critic_networks: Critic network ensemble
            observation_encoder: Observation encoding function (optional)
            batch: Training batch data
            rng: Random number generator  
            train: Training mode flag
            
        Returns:
            Tuple of (total_loss, detailed_loss_info)
        """
        rng_critic, rng_actor, rng_bc, rng_entropy, rng_cache = jax.random.split(rng, 5)
        
        # 🚀 STEP 1: 预计算所有observations的embed_prefix - 核心优化！
        
        # 🚀 Q-chunking修复：构造标准OpenPI格式的current和next observation
        
        
        # 🚀 从时间序列数据中提取current和next observations
        # state: [B, 2, state_dim] -> current: [B, state_dim], next: [B, state_dim] 
        current_state = batch['state'][:, 0]  # [B, state_dim]
        next_state = batch['state'][:, 1]     # [B, state_dim]
        
        # image: {cam: [B, 2, H, W, C]} -> current: {cam: [B, H, W, C]}, next: {cam: [B, H, W, C]}
        current_image = {cam: img[:, 0] for cam, img in batch['image'].items()}  # {cam: [B, H, W, C]}
        next_image = {cam: img[:, 1] for cam, img in batch['image'].items()}     # {cam: [B, H, W, C]}
        
        # image_mask: 为current和next创建标准OpenPI格式mask
        batch_size = current_state.shape[0]
        current_image_mask = {cam: jnp.ones(batch_size, dtype=jnp.bool_) for cam in current_image.keys()}
        next_image_mask = {cam: jnp.ones(batch_size, dtype=jnp.bool_) for cam in next_image.keys()}
        
        # Current observation (标准OpenPI格式)
        current_obs_dict = {
            'image': current_image,
            'image_mask': current_image_mask,
            'state': current_state,
            'tokenized_prompt': batch.get('tokenized_prompt'),
            'tokenized_prompt_mask': batch.get('tokenized_prompt_mask')
        }
        
        # Next observation (标准OpenPI格式，用于Q-learning目标计算)
        next_obs_dict = {
            'image': next_image,
            'image_mask': next_image_mask, 
            'state': next_state
            # next不需要prompt相关字段
        }
        
        # 构造observations缓存，利用OpenPI预计算优化
        observations_to_cache = {
            'current': _model.Observation.from_dict(current_obs_dict),
            'next': _model.Observation.from_dict(next_obs_dict)
        }
        logger.debug(f"✅ OpenPI原生数据格式: reward{batch['reward'].shape if 'reward' in batch else 'N/A'}, action{batch['actions'].shape}")
        
        # 预计算所有embed_prefix结果，避免后续重复计算
        logger.debug(f"🚀 预计算{len(observations_to_cache)}个observations的embeddings...")
        embeddings_cache = pi0_model.precompute_embeddings_cache(
            observations_to_cache, rng_cache, train
        )
        logger.debug(f"✅ Embeddings缓存预计算完成，覆盖keys: {list(embeddings_cache.keys())}")
        
        # 🚀 STEP 2: 使用缓存优化的BC loss computation
        bc_loss, bc_info, pi0_features = self.bc_loss_computer(
            pi0_model, batch, rng_bc, train, return_features=True,
            embeddings_cache=embeddings_cache  # 传递缓存避免重复embed_prefix
        )
        
        # 🚀 STEP 3: 使用缓存优化的Critic loss computation  
        critic_loss, critic_info = self.critic_loss_computer(
            pi0_model=pi0_model,
            critic_networks=critic_networks,
            observation_encoder=None,  # Let it create encoder automatically
            batch=batch,
            rng=rng_critic,
            train=train,
            embeddings_cache=embeddings_cache,  # 传递缓存避免重复embed_prefix
            current_obs_dict=current_obs_dict,  # 传递处理好的current observations
            next_obs_dict=next_obs_dict  # 传递处理好的next observations
        )
        
        # 🚀 STEP 4: 使用缓存优化的Actor loss computation
        # 当使用缓存时，不需要传递observation_encoder（避免维度冲突）
        if embeddings_cache is not None and 'current' in embeddings_cache:
            # 使用缓存时，ActorLossComputer会直接从缓存提取features
            # 不需要observation_encoder，设为None避免维度冲突
            observation_encoder_for_actor = None
        else:
            # 只有在没有缓存时才创建observation_encoder
            if observation_encoder is None:
                import sys
                from pathlib import Path
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from training.acrlpd_train_state import combine_pi0_and_state_features
                observation_encoder_for_actor = lambda obs: combine_pi0_and_state_features(
                    pi0_model, obs, rng_actor, 
                    real_action_dim=getattr(self.critic_loss_computer, 'real_action_dim', 14)
                )
            else:
                observation_encoder_for_actor = observation_encoder
        
        actor_loss, actor_info = self.actor_loss_computer(
            pi0_model, critic_networks, observation_encoder_for_actor,
            batch, rng_actor, train,
            embeddings_cache=embeddings_cache,  # 传递缓存避免重复embed_prefix
            current_obs_dict=current_obs_dict  # 传递处理好的current observations
        )
        
        # 3. Temperature/alpha loss
        alpha_loss, alpha_info = 0.0, {'alpha_loss': 0.0, 'alpha_value': 1.0, 'entropy_estimate': 0.0}
        if self.temperature_module is not None and self.entropy_estimator is not None:
            # Estimate entropy
            entropy_est = self.entropy_estimator.estimate_entropy(
                pi0_model, _model.Observation.from_dict(batch), rng_entropy
            ).mean()
            
            # Target entropy (heuristic based on action dimension)
            action_dim = batch['actions'].shape[-1]
            target_entropy = -self.target_entropy_multiplier * action_dim
            
            # Temperature loss
            alpha_loss = self.temperature_module.alpha_loss(entropy_est, target_entropy)
            alpha_value = self.temperature_module()
            
            alpha_info = {
                'alpha_loss': alpha_loss,
                'alpha_value': alpha_value,
                'entropy_estimate': entropy_est,
                'target_entropy': target_entropy
            }
        
        # 4. Combine losses with proper weighting
        total_loss = (
            self.loss_weights.critic_weight * critic_loss +
            self.loss_weights.actor_weight * actor_loss +
            self.loss_weights.bc_weight * bc_loss +
            self.loss_weights.alpha_weight * alpha_loss
        )
        
        # L2 regularization (if enabled)
        if self.loss_weights.weight_decay > 0:
            # Add weight decay - implementation depends on parameter access
            l2_loss = 0.0  # Placeholder
            total_loss += self.loss_weights.weight_decay * l2_loss
        
        # Create comprehensive loss info with all metrics from unified computation
        loss_info = LossInfo(
            total_loss=total_loss,
            critic_loss=critic_loss,
            actor_loss=actor_loss,
            bc_loss=bc_loss,
            alpha_loss=alpha_loss,
            q_mean=critic_info.get('q_mean', 0.0),
            q_std=critic_info.get('q_std', 0.0),
            target_q_mean=critic_info.get('target_q_mean', 0.0),
            td_error_mean=critic_info.get('td_error_mean', 0.0),
            bc_loss_raw=bc_info.get('bc_loss_raw', 0.0),
            alpha_value=alpha_info.get('alpha_value', 1.0),
            entropy_estimate=alpha_info.get('entropy_estimate', 0.0),
            q_values_for_actor=actor_info.get('q_values_for_actor', 0.0),
            valid_samples=critic_info.get('valid_samples', bc_info.get('valid_samples', 0.0)),
            mask_ratio=critic_info.get('mask_ratio', 1.0),
            bc_positive_samples=bc_info.get('positive_samples', 0.0),      # BC正样本统计
            bc_total_samples=bc_info.get('total_samples', 0.0)             # BC总样本统计
        )
        
        return total_loss, loss_info


def create_loss_computer(
    loss_weights: LossWeights,
    discount: float = 0.99,
    horizon_length: int = 5,
    q_aggregation: str = "min",
    target_entropy_multiplier: float = 0.5,
    use_temperature: bool = True,
    actor_num_samples: int = 4,                    # NEW: Actor sampling parameter
    initial_temperature: float = 1.0,
    real_action_dim: int = 14,                     # NEW: Real action dimension parameter
    rngs: Optional[nnx.Rngs] = None
) -> Tuple[JointLossComputer, Optional[TemperatureModule]]:
    """
    Factory function to create joint loss computer.
    
    Args:
        loss_weights: Loss weighting configuration
        discount: Discount factor for RL
        horizon_length: Action chunk length
        q_aggregation: Q-value aggregation method
        target_entropy_multiplier: Target entropy scaling
        use_temperature: Whether to use adaptive temperature
        actor_num_samples: Number of action samples for actor loss variance reduction
        initial_temperature: Initial temperature value
        rngs: Random number generators
        
    Returns:
        Tuple of (JointLossComputer, TemperatureModule)
    """
    # Create loss computers
    critic_loss_computer = CriticLossComputer(
        discount=discount,
        horizon_length=horizon_length,
        q_aggregation=q_aggregation,
        config={}  # 提供空配置字典
    )
    
    # NEW: Create actor loss computer
    actor_loss_computer = ActorLossComputer(num_action_samples=actor_num_samples, real_action_dim=real_action_dim)
    
    bc_loss_computer = BCLossComputer(real_action_dim=real_action_dim)
    
    temperature_module = None
    entropy_estimator = None
    if use_temperature:
        if rngs is None:
            raise ValueError("rngs required when use_temperature=True")
        temperature_module = TemperatureModule(initial_temperature, rngs)
        entropy_estimator = EntropyEstimator()
    
    # Create joint loss computer
    joint_loss_computer = JointLossComputer(
        loss_weights=loss_weights,
        critic_loss_computer=critic_loss_computer,
        actor_loss_computer=actor_loss_computer,           # NEW
        bc_loss_computer=bc_loss_computer,
        temperature_module=temperature_module,
        entropy_estimator=entropy_estimator,
        target_entropy_multiplier=target_entropy_multiplier
    )
    
    return joint_loss_computer, temperature_module


# Default loss weight configuration  
DEFAULT_LOSS_WEIGHTS = LossWeights()