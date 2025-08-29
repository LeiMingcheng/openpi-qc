"""
ACRLPD + π₀ Agent Integration.

This module implements the core ACRLPDPi0Agent that integrates Q-chunking 
reinforcement learning with π₀ diffusion models. The agent combines:

- π₀ model as the direct policy (Actor)
- Critic network ensemble for Q-value estimation
- Joint loss functions for combined training
- Best-of-N sampling for policy optimization
- State management and training utilities

Key features:
- End-to-end differentiable training
- Multi-modal observation handling
- Action sequence generation and evaluation
- Adaptive temperature control
- Bootstrap handling for episode boundaries
"""

import logging
from typing import Dict, Any, Tuple, Optional, Callable, NamedTuple
import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import numpy as np

import openpi.models.model as _model
import openpi.models.pi0 as _pi0
from openpi.shared import array_typing as at
import openpi.training.optimizer as _optimizer
import openpi.training.utils as training_utils

from .critic_networks import CriticNetworks, CriticConfig, create_critic_networks
from .loss_functions import (
    JointLossComputer, LossWeights, LossInfo, TemperatureModule,
    create_loss_computer
)

logger = logging.getLogger(__name__)
# 确保logger能正确输出到console
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@dataclasses.dataclass(frozen=True)
class ACRLPDPi0Config:
    """Complete configuration for ACRLPD + π₀ agent."""
    
    # π₀ model configuration
    pi0_config: _pi0.Pi0Config = dataclasses.field(default_factory=_pi0.Pi0Config)
    freeze_pi0_backbone: bool = False      # Freeze π₀ vision/language backbone
    real_action_dim: int = 14              # Real action dimension (e.g., 14 for ALOHA)
    
    # ACRLPD core parameters
    horizon_length: int = 10               # Action chunk length  
    discount: float = 0.99                 # RL discount factor
    q_aggregation: str = "min"            # Q-value aggregation
    
    # Critic network configuration
    critic_config: CriticConfig = dataclasses.field(default_factory=CriticConfig)
    
    # Loss weighting
    loss_weights: LossWeights = dataclasses.field(default_factory=LossWeights)
    
    # Training parameters
    batch_size: int = 128                  # Training batch size
    target_update_tau: float = 0.005       # Target network soft update
    
    # EMA configuration
    use_ema: bool = True                   # Enable EMA for stabilization
    pi0_ema_decay: float = 0.999           # EMA decay for π₀ model (high protection)
    critic_ema_decay: float = 0.99         # EMA decay for Critic networks
    use_ema_for_inference: bool = True     # Use EMA params during inference
    
    # Sampling configuration
    best_of_n_samples: int = 32            # Best-of-N sample count
    diffusion_steps: int = 10              # π₀ diffusion sampling steps
    use_best_of_n: bool = True             # Enable Best-of-N sampling
    
    # Temperature control
    use_adaptive_temperature: bool = True   # Adaptive temperature
    initial_temperature: float = 1.0       # Initial α value
    target_entropy_multiplier: float = 0.5 # Target entropy scaling
    
    # OpenPI optimizer configurations
    pi0_lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(
        default_factory=lambda: _optimizer.CosineDecaySchedule(
            warmup_steps=1000, peak_lr=1e-5, decay_steps=30000, decay_lr=1e-6
        )
    )
    critic_lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(
        default_factory=lambda: _optimizer.CosineDecaySchedule(
            warmup_steps=1000, peak_lr=3e-4, decay_steps=30000, decay_lr=3e-5
        )
    )
    pi0_optimizer: _optimizer.OptimizerConfig = dataclasses.field(
        default_factory=lambda: _optimizer.AdamW(weight_decay=1e-4, clip_gradient_norm=1.0)
    )
    critic_optimizer: _optimizer.OptimizerConfig = dataclasses.field(
        default_factory=lambda: _optimizer.AdamW(weight_decay=1e-4, clip_gradient_norm=1.0)
    )
    
    # Training phases
    eval_frequency: int = 10000            # Evaluation frequency
    save_frequency: int = 50000            # Checkpoint save frequency
    
    def validate(self):
        """Validate configuration parameters."""
        assert self.horizon_length > 0
        assert 0 < self.discount <= 1.0
        assert self.q_aggregation in ["min", "mean", "weighted"]
        assert self.best_of_n_samples > 0
        assert self.diffusion_steps > 0
        assert self.batch_size > 0
        
        # CRITICAL: Ensure consistency between horizon_length and pi0_config.action_horizon
        assert self.horizon_length == self.pi0_config.action_horizon, \
            f"horizon_length ({self.horizon_length}) must equal pi0_config.action_horizon ({self.pi0_config.action_horizon}) for gradient-safe sampling consistency"
        
        # Validate EMA configuration
        if self.use_ema:
            assert 0.0 < self.pi0_ema_decay < 1.0, f"π₀ EMA decay must be in (0,1), got {self.pi0_ema_decay}"
            assert 0.0 < self.critic_ema_decay < 1.0, f"Critic EMA decay must be in (0,1), got {self.critic_ema_decay}"
        
        # Validate sub-configs
        self.loss_weights.validate()


class TrainingState(NamedTuple):
    """Training state container."""
    
    step: int
    pi0_optimizer_state: Any
    critic_optimizer_state: Any
    temperature_optimizer_state: Any
    metrics: Dict[str, float]
    rng: jnp.ndarray


class ACRLPDPi0Agent(nnx.Module):
    """
    ACRLPD + π₀ integrated agent.
    
    This agent combines Q-chunking reinforcement learning with π₀ diffusion
    models, enabling sample-efficient learning on robotic manipulation tasks.
    """
    
    def __init__(
        self,
        config: ACRLPDPi0Config,
        rngs: nnx.Rngs
    ):
        super().__init__()
        self.config = config
        config.validate()
        
        # Create π₀ model - extract raw RNG key from RngStream
        pi0_raw_rng = rngs.pi0()  # Get the actual jax random key
        self.pi0_model = config.pi0_config.create(pi0_raw_rng)
        
        # Create critic networks with π₀ integration
        critic_raw_rng = rngs.critic()  # Get the actual jax random key
        logger.info(f"🔍 [Agent创建] 配置检查: π₀_action_dim={config.pi0_config.action_dim}, real_action_dim={config.real_action_dim}")
        logger.info(f"🔍 [Agent创建] Critic网络创建参数: horizon={config.horizon_length}, action_dim={config.real_action_dim}")
        
        self.critic_networks = create_critic_networks(
            config=config.critic_config,
            pi0_model=self.pi0_model,
            action_horizon=config.horizon_length,
            action_dim=config.real_action_dim,  # Use real action dimension instead of π₀'s 32
            rngs=critic_raw_rng,
            pi0_config=config.pi0_config  # Pass config for fake_obs generation
        )
        
        logger.info(f"✅ [Agent创建] Critic网络创建完成")
        
        # Create joint loss computer
        temp_raw_rng = rngs.temperature() if config.use_adaptive_temperature else None
        self.loss_computer, self.temperature_module = create_loss_computer(
            loss_weights=config.loss_weights,
            discount=config.discount,
            horizon_length=config.horizon_length,
            q_aggregation=config.q_aggregation,
            target_entropy_multiplier=config.target_entropy_multiplier,
            use_temperature=config.use_adaptive_temperature,
            actor_num_samples=config.best_of_n_samples,  # 🔧 传递num_action_samples参数
            initial_temperature=config.initial_temperature,
            rngs=temp_raw_rng
        )
        
        # Create observation encoder using existing feature extraction functions
        self.observation_encoder = self._create_observation_encoder()
        
        # Create optimizers using OpenPI system
        pi0_weight_decay_mask = None
        if config.freeze_pi0_backbone:
            # Create weight decay mask that excludes frozen parameters
            pi0_weight_decay_mask = nnx.filterlib.to_predicate(nnx.Not(config.freeze_filter)) if hasattr(config, 'freeze_filter') else None
        
        self.pi0_optimizer = _optimizer.create_optimizer(
            config.pi0_optimizer, 
            config.pi0_lr_schedule,
            weight_decay_mask=pi0_weight_decay_mask
        )
        
        self.critic_optimizer = _optimizer.create_optimizer(
            config.critic_optimizer, 
            config.critic_lr_schedule
        )
        
        if self.temperature_module is not None:
            # Use same schedule as critic but with no weight decay
            temp_optimizer_config = _optimizer.AdamW(weight_decay=0.0, clip_gradient_norm=1.0)
            self.temperature_optimizer = _optimizer.create_optimizer(
                temp_optimizer_config,
                config.critic_lr_schedule
            )
        else:
            self.temperature_optimizer = None
        
        # Initialize training state
        self._step = 0
        
        # Initialize optimizer states using dummy parameters to get the structure
        # Mark these as non-trackable by nnx to avoid serialization issues
        dummy_pi0_params = nnx.state(self.pi0_model, nnx.Param)
        dummy_critic_params = nnx.state(self.critic_networks, nnx.Param)
        dummy_temp_params = nnx.state(self.temperature_module, nnx.Param) if self.temperature_module else {}
        
        # Use nnx.Variable with collection=False to exclude from state tracking
        self.pi0_optimizer_state = nnx.Variable(self.pi0_optimizer.init(dummy_pi0_params))
        self.critic_optimizer_state = nnx.Variable(self.critic_optimizer.init(dummy_critic_params))
        self.temperature_optimizer_state = nnx.Variable(self.temperature_optimizer.init(dummy_temp_params)) if self.temperature_optimizer else None
        
        logger.info(f"Created ACRLPDPi0Agent: π₀_action_dim={config.pi0_config.action_dim}, "
                   f"real_action_dim={config.real_action_dim}, horizon={config.horizon_length}, "
                   f"batch_size={config.batch_size}, EMA={'enabled' if config.use_ema else 'disabled'}")
           250 -      def sample_actions(
       251 -          self,
       252 -          observations: _model.Observation,
       253 -          rng: jnp.ndarray,
       254 -          *,
       255 -          use_best_of_n: Optional[bool] = None,
       256 -          num_samples: Optional[int] = None,
       257 -          deterministic: bool = False
       258 -      ) -> _model.Actions:
       259 -          """
       260 -          Sample actions from the π₀ policy with optional Best-of-N 
           - optimization.
       261 -          
       262 -          Args:
       263 -              observations: Multi-modal observations
       264 -              rng: Random number generator
       265 -              use_best_of_n: Whether to use Best-of-N sampling
       266 -              num_samples: Number of samples for Best-of-N
       267 -              deterministic: Use deterministic (greedy) sampling
       268 -              
       269 -          Returns:
       270 -              Action sequences: [batch_size, action_horizon, 
     action_dim]
       271 -          """
       272 -          if use_best_of_n is None:
       273 -              use_best_of_n = self.config.use_best_of_n and not 
           - deterministic
       274 -          
       275 -          if num_samples is None:
       276 -              num_samples = self.config.best_of_n_samples
       277 -          
       278 -          if not use_best_of_n or deterministic:
       279 -              # Standard π₀ sampling - use gradient-safe version for 
           - consistency
       280 -              if deterministic:
       281 -                  # For deterministic sampling, use fewer diffusion 
           - steps
       282 -                  return self.pi0_model.sample_actions_differentiable(
       283 -                      rng, observations, num_steps=5
       284 -                  )
       285 -              else:
       286 -                  return self.pi0_model.sample_actions_differentiable(
       287 -                      rng, observations, 
           - num_steps=self.config.diffusion_steps
       288 -                  )
       289 -          
       290 -          return self._best_of_n_sampling(observations, rng, 
           - num_samples)
    # ============================================================================
    # 训练相关方法 (推理代码已移除，AC_Training框架不需要推理)
    # ============================================================================
    
    def _best_of_n_sampling(
        self,
        observations: _model.Observation,
        rng: jnp.ndarray,
        num_samples: int,
        use_top_k: bool = False,
        top_k: int = 5,
        use_temperature_scaling: bool = True,
        temperature: float = 1.0,
        use_diversity_bonus: bool = False,
        diversity_weight: float = 0.1,
        use_parallel_evaluation: bool = True,
        chunk_size: int = 32
    ) -> _model.Actions:
        """
        Enhanced Best-of-N sampling with advanced selection strategies.
        
        Args:
            observations: Multi-modal observations
            rng: Random number generator
            num_samples: Number of candidate samples
            use_top_k: Whether to select from top-k candidates randomly
            top_k: Number of top candidates to consider
            use_temperature_scaling: Whether to use temperature in sampling
            temperature: Temperature for stochastic selection
            use_diversity_bonus: Whether to add diversity bonus to Q-values
            diversity_weight: Weight for diversity bonus
            
        Returns:
            Best action sequences based on Q-values
        """
        batch_size = observations.state.shape[0]
        
        # Adaptive strategy selection based on num_samples and available memory
        if num_samples > chunk_size and use_parallel_evaluation:
            return self._memory_efficient_best_of_n(
                observations, rng, num_samples, use_top_k, top_k, 
                use_temperature_scaling, temperature, chunk_size
            )
        elif not use_parallel_evaluation:
            # Sequential sampling for very large N or memory constraints
            return self._sequential_best_of_n(
                observations, rng, num_samples, use_top_k, top_k,
                use_temperature_scaling, temperature
            )
        
        # Standard parallel sampling for smaller N
        sample_rngs = jax.random.split(rng, num_samples)
        
        # Optimized sampling with variable diffusion steps based on quality needs
        def sample_single(rng_key):
            # Use fewer diffusion steps for candidate generation to speed up
            diffusion_steps = max(5, self.config.diffusion_steps // 2)
            return self.pi0_model.sample_actions_differentiable(
                rng_key, observations, num_steps=diffusion_steps
            )
        
        # Parallel candidate generation
        action_candidates = jax.vmap(sample_single)(sample_rngs)
        
        # Efficient Q-value evaluation
        q_values = self._evaluate_action_candidates(
            observations, action_candidates, use_diversity_bonus, diversity_weight
        )
        
        # Enhanced selection strategy
        selected_actions = self._select_best_actions(
            action_candidates, q_values, rng, use_top_k, top_k, 
            use_temperature_scaling, temperature
        )
        
        return selected_actions
    
    def _memory_efficient_best_of_n(
        self,
        observations: _model.Observation,
        rng: jnp.ndarray,
        num_samples: int,
        use_top_k: bool = False,
        top_k: int = 5,
        use_temperature_scaling: bool = True,
        temperature: float = 1.0,
        chunk_size: int = 32
    ) -> _model.Actions:
        """
        Memory-efficient Best-of-N sampling for large N.
        
        Processes candidates in chunks to avoid memory issues.
        """
        batch_size = observations.state.shape[0]
        best_q_values = jnp.full((batch_size,), -jnp.inf)
        best_actions = None
        
        # Use vectorized approach to avoid dynamic loops
        # For memory efficiency, we still limit to chunk_size but process statically
        effective_samples = min(num_samples, chunk_size)  # Limit samples for memory
        sample_rngs = jax.random.split(rng, effective_samples)
        
        def sample_and_evaluate(sample_rng):
            # Use gradient-safe sampling
            candidate_actions = self.pi0_model.sample_actions_differentiable(
                sample_rng, observations, 
                num_steps=max(5, self.config.diffusion_steps // 2)
            )
            
            # Evaluate candidate
            candidate_q_values = self._evaluate_single_candidate(
                observations, candidate_actions
            )
            
            return candidate_actions, candidate_q_values
        
        # Process all candidates in parallel (memory-limited)
        all_actions, all_q_values = jax.vmap(sample_and_evaluate)(sample_rngs)
        # all_actions: [effective_samples, batch_size, action_horizon, action_dim]
        # all_q_values: [effective_samples, batch_size]
        
        # Select best actions per batch element
        best_indices = jnp.argmax(all_q_values, axis=0)  # [batch_size]
        batch_indices = jnp.arange(all_actions.shape[1])
        
        # Gather best actions for each batch element
        best_actions = all_actions[best_indices, batch_indices]
        
        return best_actions
    
    def _sequential_best_of_n(
        self,
        observations: _model.Observation,
        rng: jnp.ndarray,
        num_samples: int,
        use_top_k: bool = False,
        top_k: int = 5,
        use_temperature_scaling: bool = True,
        temperature: float = 1.0
    ) -> _model.Actions:
        """
        Gradient-safe Best-of-N sampling using static loops.
        
        Simplified version that avoids dynamic Python loops.
        """
        # Use vectorized approach instead of sequential processing
        # This is more efficient and avoids dynamic loops entirely
        sample_rngs = jax.random.split(rng, num_samples)
        
        def sample_and_evaluate(sample_rng):
            # Use gradient-safe sampling
            candidate_actions = self.pi0_model.sample_actions_differentiable(
                sample_rng, observations, 
                num_steps=max(5, self.config.diffusion_steps // 2)
            )
            
            # Evaluate candidate
            candidate_q_values = self._evaluate_single_candidate(
                observations, candidate_actions
            )
            
            return candidate_actions, candidate_q_values
        
        # Process all candidates in parallel (vectorized)
        all_actions, all_q_values = jax.vmap(sample_and_evaluate)(sample_rngs)
        # all_actions: [num_samples, batch_size, action_horizon, action_dim]  
        # all_q_values: [num_samples, batch_size]
        
        # Select best actions per batch element
        best_indices = jnp.argmax(all_q_values, axis=0)  # [batch_size]
        batch_indices = jnp.arange(all_actions.shape[1])
        
        # Gather best actions for each batch element
        best_actions = all_actions[best_indices, batch_indices]
        
        return best_actions
    
    def _create_observation_encoder(self):
        """
        Create observation encoder wrapper method using existing feature extraction.
        
        This method bridges the architectural transition from agent-internal encoding
        to training-loop encoding by wrapping the existing combine_pi0_and_state_features
        function for compatibility.
        
        Returns:
            Callable that encodes observations using π₀ and state features
        """
        def observation_encoder_fn(observations: _model.Observation) -> jnp.ndarray:
            """
            Encode observations using π₀ visual features and state features.
            
            Args:
                observations: Multi-modal observations
                
            Returns:
                Encoded features: [batch_size, llm_dim + state_dim]
            """
            # Delayed import to avoid circular dependencies
            try:
                from training.acrlpd_train_state import combine_pi0_and_state_features
                return combine_pi0_and_state_features(self.pi0_model, observations)
            except ImportError as e:
                logger.error(f"Failed to import combine_pi0_and_state_features: {e}")
                # Fallback: create dummy features with correct dimensions
                batch_size = observations.state.shape[0] if observations.state is not None else 1
                llm_dim = getattr(self.pi0_model.config, 'width', 1024)  # Default to 1024
                state_dim = self.config.pi0_config.action_dim
                total_dim = llm_dim + state_dim
                return jnp.zeros((batch_size, total_dim))
        
        return observation_encoder_fn
    
    def _evaluate_single_candidate(
        self,
        observations: _model.Observation,
        actions: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Evaluate a single action candidate efficiently.
        
        Args:
            observations: Multi-modal observations
            actions: Action sequence [batch_size, action_horizon, action_dim]
            
        Returns:
            Q-values: [batch_size]
        """
        # Encode observations
        obs_encoded = self.observation_encoder(observations)
        
        # Flatten actions for critic
        actions_flat = actions.reshape(actions.shape[0], -1)
        
        # Evaluate with critic
        q_values = self.critic_networks(
            obs_encoded, actions_flat,
            use_target=False, train=False, aggregate=True
        )
        
        return q_values
    
    def _evaluate_action_candidates(
        self,
        observations: _model.Observation,
        action_candidates: jnp.ndarray,
        use_diversity_bonus: bool = False,
        diversity_weight: float = 0.1
    ) -> jnp.ndarray:
        """
        Efficiently evaluate Q-values for action candidates.
        
        Args:
            observations: Multi-modal observations
            action_candidates: [num_samples, batch_size, action_horizon, action_dim]
            use_diversity_bonus: Whether to add diversity bonus
            diversity_weight: Weight for diversity bonus
            
        Returns:
            Q-values: [num_samples, batch_size]
        """
        num_samples, batch_size = action_candidates.shape[:2]
        
        # Encode observations once
        obs_encoded = self.observation_encoder(observations)  # [batch_size, obs_dim]
        
        # Reshape for batch evaluation
        obs_encoded_expanded = jnp.tile(
            obs_encoded[None, :, :], (num_samples, 1, 1)
        )  # [num_samples, batch_size, obs_dim]
        
        # Flatten actions for critic
        action_candidates_flat = action_candidates.reshape(
            num_samples, batch_size, -1
        )  # [num_samples, batch_size, flattened_action_dim]
        
        # Vectorized Q-value evaluation
        def evaluate_batch(obs_batch, action_batch):
            return self.critic_networks(
                obs_batch, action_batch, 
                use_target=False, train=False, aggregate=True
            )
        
        # Parallel evaluation across samples
        q_values = jax.vmap(evaluate_batch)(obs_encoded_expanded, action_candidates_flat)
        
        # Add diversity bonus if requested
        if use_diversity_bonus:
            diversity_bonus = self._compute_diversity_bonus(
                action_candidates, diversity_weight
            )
            q_values = q_values + diversity_bonus
        
        return q_values
    
    def _compute_diversity_bonus(
        self,
        action_candidates: jnp.ndarray,
        diversity_weight: float
    ) -> jnp.ndarray:
        """
        Compute diversity bonus to encourage action diversity.
        
        Args:
            action_candidates: [num_samples, batch_size, action_horizon, action_dim]
            diversity_weight: Weight for diversity bonus
            
        Returns:
            Diversity bonus: [num_samples, batch_size]
        """
        num_samples, batch_size = action_candidates.shape[:2]
        
        # Flatten actions for distance computation
        actions_flat = action_candidates.reshape(num_samples, batch_size, -1)
        
        # Compute pairwise distances within each batch
        def compute_batch_diversity(batch_actions):
            # batch_actions: [num_samples, action_dim]
            # Compute pairwise L2 distances
            diff = batch_actions[:, None, :] - batch_actions[None, :, :]  # [num_samples, num_samples, action_dim]
            distances = jnp.linalg.norm(diff, axis=-1)  # [num_samples, num_samples]
            
            # Average distance to other candidates (excluding self)
            mask = 1.0 - jnp.eye(num_samples)
            avg_distance = jnp.sum(distances * mask, axis=1) / jnp.maximum(num_samples - 1, 1)
            
            return avg_distance
        
        # Apply across batch dimension
        diversity_scores = jax.vmap(compute_batch_diversity, in_axes=1, out_axes=1)(actions_flat)
        
        return diversity_weight * diversity_scores
    
    def _select_best_actions(
        self,
        action_candidates: jnp.ndarray,
        q_values: jnp.ndarray,
        rng: jnp.ndarray,
        use_top_k: bool = False,
        top_k: int = 5,
        use_temperature_scaling: bool = True,
        temperature: float = 1.0
    ) -> _model.Actions:
        """
        Select best actions using various strategies.
        
        Args:
            action_candidates: [num_samples, batch_size, action_horizon, action_dim]
            q_values: [num_samples, batch_size]
            rng: Random number generator
            use_top_k: Whether to use top-k selection
            top_k: Number of top candidates
            use_temperature_scaling: Whether to use temperature scaling
            temperature: Temperature parameter
            
        Returns:
            Selected actions: [batch_size, action_horizon, action_dim]
        """
        num_samples, batch_size = q_values.shape
        
        if use_top_k:
            # Top-k selection with optional temperature scaling
            if use_temperature_scaling and temperature > 0:
                # Stochastic selection from top-k
                top_k_indices = jnp.argsort(q_values, axis=0)[-top_k:]  # [top_k, batch_size]
                top_k_q_values = q_values[top_k_indices, jnp.arange(batch_size)[None, :]]
                
                # Temperature-scaled probabilities
                probs = jax.nn.softmax(top_k_q_values / temperature, axis=0)  # [top_k, batch_size]
                
                # Sample from top-k for each batch element
                sample_rngs = jax.random.split(rng, batch_size)
                
                def sample_from_topk(prob_vec, sample_rng):
                    return jax.random.categorical(sample_rng, jnp.log(prob_vec + 1e-8))
                
                selected_topk_indices = jax.vmap(sample_from_topk)(probs.T, sample_rngs)
                selected_indices = top_k_indices[selected_topk_indices, jnp.arange(batch_size)]
            else:
                # Deterministic top-1 selection
                selected_indices = jnp.argmax(q_values, axis=0)
        else:
            if use_temperature_scaling and temperature > 0:
                # Full stochastic selection with temperature
                probs = jax.nn.softmax(q_values / temperature, axis=0)
                sample_rngs = jax.random.split(rng, batch_size)
                
                def sample_from_all(prob_vec, sample_rng):
                    return jax.random.categorical(sample_rng, jnp.log(prob_vec + 1e-8))
                
                selected_indices = jax.vmap(sample_from_all)(probs.T, sample_rngs)
            else:
                # Deterministic best selection
                selected_indices = jnp.argmax(q_values, axis=0)
        
        # Gather selected actions
        selected_actions = action_candidates[selected_indices, jnp.arange(batch_size)]
        
        return selected_actions
    
    def compute_loss(
        self,
        batch: Dict[str, jnp.ndarray],
        rng: jnp.ndarray,
        train: bool = True
    ) -> Tuple[jnp.ndarray, LossInfo]:
        """
        Compute joint training loss.
        
        Args:
            batch: Training batch data
            rng: Random number generator
            train: Training mode flag
            
        Returns:
            Tuple of (total_loss, detailed_loss_info)
        """
        return self.loss_computer(
            pi0_model=self.pi0_model,
            critic_networks=self.critic_networks,
            observation_encoder=self.observation_encoder,
            batch=batch,
            rng=rng,
            train=train
        )
    
    def train_step(
        self,
        batch: Dict[str, jnp.ndarray],
        rng: jnp.ndarray
    ) -> Tuple["ACRLPDPi0Agent", LossInfo]:
        """
        Perform a single training step.
        
        Args:
            batch: Training batch data
            rng: Random number generator
            
        Returns:
            Tuple of (updated_agent, loss_info)
        """
        def loss_fn(agent_vars):
            # Create temporary agent with updated variables
            temp_agent = agent_vars
            loss, loss_info = temp_agent.compute_loss(batch, rng, train=True)
            return loss, loss_info
        
        # Compute gradients
        (loss, loss_info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(self)
        
        # Apply gradients to π₀ model
        if not self.config.freeze_pi0_backbone:
            pi0_grads = grads.pi0_model
            pi0_params = nnx.state(self.pi0_model, nnx.Param)
            pi0_updates, new_pi0_opt_state = self.pi0_optimizer.update(pi0_grads, self.pi0_optimizer_state.value, pi0_params)
            new_pi0_params = optax.apply_updates(pi0_params, pi0_updates)
            nnx.update(self.pi0_model, new_pi0_params)
            self.pi0_optimizer_state.value = new_pi0_opt_state
        
        # Apply gradients to critic networks
        critic_grads = grads.critic_networks
        critic_params = nnx.state(self.critic_networks, nnx.Param)
        critic_updates, new_critic_opt_state = self.critic_optimizer.update(critic_grads, self.critic_optimizer_state.value, critic_params)
        new_critic_params = optax.apply_updates(critic_params, critic_updates)
        nnx.update(self.critic_networks, new_critic_params)
        self.critic_optimizer_state.value = new_critic_opt_state
        
        # Apply gradients to temperature (if used)
        if self.temperature_module is not None:
            temp_grads = grads.temperature_module
            temp_params = nnx.state(self.temperature_module, nnx.Param)
            temp_updates, new_temp_opt_state = self.temperature_optimizer.update(temp_grads, self.temperature_optimizer_state.value, temp_params)
            new_temp_params = optax.apply_updates(temp_params, temp_updates)
            nnx.update(self.temperature_module, new_temp_params)
            self.temperature_optimizer_state.value = new_temp_opt_state
        
        # Update target networks
        if self._step % 1 == 0:  # Update every step (soft update)
            self.critic_networks.soft_update_target_networks(
                tau=self.config.target_update_tau
            )
        
        # Update step counter
        self._step += 1
        
        return self, loss_info
    
    def _update_ema_params(self, ema_params: nnx.State) -> nnx.State:
        """Update EMA parameters for π₀ model only (OpenPI compatible).
        
        Args:
            ema_params: Current π₀ EMA parameters (nnx.State)
            
        Returns:
            Updated π₀ EMA parameters (nnx.State)
        """
        if not self.config.use_ema or ema_params is None:
            return ema_params
            
        # Extract current π₀ parameters
        current_pi0_params = nnx.state(self.pi0_model, nnx.Param)
        
        # Update π₀ EMA parameters (high protection of pretrained knowledge)
        updated_ema_params = jax.tree.map(
            lambda ema, current: self.config.pi0_ema_decay * ema + (1 - self.config.pi0_ema_decay) * current,
            ema_params,
            current_pi0_params
        )
        
        return updated_ema_params
    
    def set_inference_mode(self, use_ema: bool = None):
        """Switch between EMA and current parameters for π₀ inference.
        
        Args:
            use_ema: If True, use EMA parameters. If None, use config setting.
        """
        if use_ema is None:
            use_ema = self.config.use_ema_for_inference
            
        if use_ema and self.config.use_ema:
            # Get current train state with EMA parameters
            train_state = self.create_train_state()
            if train_state.ema_params is not None:
                # Apply π₀ EMA parameters for inference
                nnx.update(self.pi0_model, train_state.ema_params)
                logger.debug("Switched to π₀ EMA parameters for inference")
            else:
                logger.warning("EMA enabled but no π₀ EMA parameters found")
        else:
            logger.debug("Using current π₀ parameters for inference")
    
    def save_component_checkpoints(self, checkpoint_dir: str, step: int):
        """Save each component independently using orbax for better compatibility."""
        import orbax.checkpoint as ocp
        from pathlib import Path
        import json
        
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Use orbax for consistent checkpoint format
        with ocp.PyTreeCheckpointer() as ckptr:
            
            # Save π₀ model
            pi0_dir = checkpoint_path / "pi0"
            pi0_dir.mkdir(exist_ok=True)
            
            pi0_params = nnx.state(self.pi0_model, nnx.Param)
            ckptr.save(pi0_dir / "params", {"params": pi0_params})
            ckptr.save(pi0_dir / "optimizer_state", {"opt_state": self.pi0_optimizer_state.value})
            
            with open(pi0_dir / "metadata.json", "w") as f:
                json.dump({
                    "step": step,
                    "component": "pi0_model",
                    "config": dataclasses.asdict(self.config.pi0_config)
                }, f, indent=2)
            
            # Save critic networks
            critic_dir = checkpoint_path / "critic"
            critic_dir.mkdir(exist_ok=True)
            
            critic_params = nnx.state(self.critic_networks, nnx.Param)
            ckptr.save(critic_dir / "params", {"params": critic_params})
            ckptr.save(critic_dir / "optimizer_state", {"opt_state": self.critic_optimizer_state.value})
            
            with open(critic_dir / "metadata.json", "w") as f:
                json.dump({
                    "step": step,
                    "component": "critic_networks", 
                    "config": dataclasses.asdict(self.config.critic_config)
                }, f, indent=2)
            
            # Save temperature module if exists
            if self.temperature_module is not None:
                temp_dir = checkpoint_path / "temperature"
                temp_dir.mkdir(exist_ok=True)
                
                temp_params = nnx.state(self.temperature_module, nnx.Param)
                ckptr.save(temp_dir / "params", {"params": temp_params})
                ckptr.save(temp_dir / "optimizer_state", {"opt_state": self.temperature_optimizer_state.value})
                
                with open(temp_dir / "metadata.json", "w") as f:
                    json.dump({
                        "step": step,
                        "component": "temperature_module",
                        "initial_temperature": self.config.initial_temperature
                    }, f, indent=2)
        
        # Save training metadata
        with open(checkpoint_path / "training_metadata.json", "w") as f:
            json.dump({
                "step": step,
                "agent_config": dataclasses.asdict(self.config),
                "components": ["pi0", "critic"] + (["temperature"] if self.temperature_module else [])
            }, f, indent=2)
        
        logger.info(f"Saved component checkpoints to {checkpoint_path} using orbax")
    
    def load_component_checkpoints(self, checkpoint_dir: str, components: list[str] = None):
        """Load specific components from independent orbax checkpoints with robust error handling."""
        import orbax.checkpoint as ocp
        from pathlib import Path
        import json
        
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint directory {checkpoint_path} does not exist")
        
        # Load metadata
        metadata_path = checkpoint_path / "training_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Training metadata not found at {metadata_path}")
            
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        available_components = metadata["components"]
        if components is None:
            components = available_components
        
        loaded_components = []
        
        # Use orbax for consistent loading
        with ocp.PyTreeCheckpointer() as ckptr:
            
            # Load π₀ model if requested
            if "pi0" in components and "pi0" in available_components:
                try:
                    pi0_dir = checkpoint_path / "pi0"
                    if not (pi0_dir / "params").exists():
                        logger.warning(f"π₀ params not found at {pi0_dir / 'params'}")
                    else:
                        pi0_params = ckptr.restore(pi0_dir / "params")["params"]
                        pi0_opt_state = ckptr.restore(pi0_dir / "optimizer_state")["opt_state"]
                        
                        nnx.update(self.pi0_model, pi0_params)
                        self.pi0_optimizer_state.value = pi0_opt_state
                        loaded_components.append("pi0")
                        logger.info("Loaded π₀ model checkpoint using orbax")
                except Exception as e:
                    logger.error(f"Failed to load π₀ component: {e}")
            
            # Load critic networks if requested  
            if "critic" in components and "critic" in available_components:
                try:
                    critic_dir = checkpoint_path / "critic"
                    if not (critic_dir / "params").exists():
                        logger.warning(f"Critic params not found at {critic_dir / 'params'}")
                    else:
                        critic_params = ckptr.restore(critic_dir / "params")["params"]
                        critic_opt_state = ckptr.restore(critic_dir / "optimizer_state")["opt_state"]
                        
                        nnx.update(self.critic_networks, critic_params)
                        self.critic_optimizer_state.value = critic_opt_state
                        loaded_components.append("critic")
                        logger.info("Loaded critic networks checkpoint using orbax")
                except Exception as e:
                    logger.error(f"Failed to load critic component: {e}")
            
            # Load temperature module if requested and exists
            if "temperature" in components and "temperature" in available_components and self.temperature_module:
                try:
                    temp_dir = checkpoint_path / "temperature"
                    if not (temp_dir / "params").exists():
                        logger.warning(f"Temperature params not found at {temp_dir / 'params'}")
                    else:
                        temp_params = ckptr.restore(temp_dir / "params")["params"]
                        temp_opt_state = ckptr.restore(temp_dir / "optimizer_state")["opt_state"]
                        
                        nnx.update(self.temperature_module, temp_params)
                        self.temperature_optimizer_state.value = temp_opt_state
                        loaded_components.append("temperature")
                        logger.info("Loaded temperature module checkpoint using orbax")
                except Exception as e:
                    logger.error(f"Failed to load temperature component: {e}")
        
        # Update step counter
        self._step = metadata["step"]
        
        # Report loading results
        if loaded_components:
            logger.info(f"Successfully loaded components: {loaded_components} from step {self._step}")
        else:
            logger.warning(f"No components were loaded from {checkpoint_path}")
            
        if set(components) - set(loaded_components):
            failed_components = set(components) - set(loaded_components)
            logger.warning(f"Failed to load components: {failed_components}")
    
    def load_legacy_component_checkpoints(self, checkpoint_dir: str, components: list[str] = None):
        """Load components from legacy pickle format for backward compatibility."""
        import pickle
        from pathlib import Path
        import json
        
        checkpoint_path = Path(checkpoint_dir)
        logger.info(f"Attempting to load legacy pickle format from {checkpoint_path}")
        
        # Load metadata
        with open(checkpoint_path / "training_metadata.json", "r") as f:
            metadata = json.load(f)
        
        available_components = metadata["components"]
        if components is None:
            components = available_components
        
        loaded_components = []
        
        # Load π₀ model if requested (legacy format)
        if "pi0" in components and "pi0" in available_components:
            try:
                pi0_dir = checkpoint_path / "pi0"
                if (pi0_dir / "model_params.pkl").exists():
                    with open(pi0_dir / "model_params.pkl", "rb") as f:
                        pi0_params = pickle.load(f)
                    with open(pi0_dir / "optimizer_state.pkl", "rb") as f:
                        pi0_opt_state = pickle.load(f)
                    
                    nnx.update(self.pi0_model, pi0_params)
                    self.pi0_optimizer_state.value = pi0_opt_state
                    loaded_components.append("pi0")
                    logger.info("Loaded π₀ model from legacy pickle format")
            except Exception as e:
                logger.error(f"Failed to load legacy π₀ component: {e}")
        
        # Load critic networks if requested (legacy format)
        if "critic" in components and "critic" in available_components:
            try:
                critic_dir = checkpoint_path / "critic"
                if (critic_dir / "model_params.pkl").exists():
                    with open(critic_dir / "model_params.pkl", "rb") as f:
                        critic_params = pickle.load(f)
                    with open(critic_dir / "optimizer_state.pkl", "rb") as f:
                        critic_opt_state = pickle.load(f)
                    
                    nnx.update(self.critic_networks, critic_params)
                    self.critic_optimizer_state.value = critic_opt_state
                    loaded_components.append("critic")
                    logger.info("Loaded critic networks from legacy pickle format")
            except Exception as e:
                logger.error(f"Failed to load legacy critic component: {e}")
        
        # Load temperature module if requested (legacy format)
        if "temperature" in components and "temperature" in available_components and self.temperature_module:
            try:
                temp_dir = checkpoint_path / "temperature"
                if (temp_dir / "model_params.pkl").exists():
                    with open(temp_dir / "model_params.pkl", "rb") as f:
                        temp_params = pickle.load(f)
                    with open(temp_dir / "optimizer_state.pkl", "rb") as f:
                        temp_opt_state = pickle.load(f)
                    
                    nnx.update(self.temperature_module, temp_params)
                    self.temperature_optimizer_state.value = temp_opt_state
                    loaded_components.append("temperature")
                    logger.info("Loaded temperature module from legacy pickle format")
            except Exception as e:
                logger.error(f"Failed to load legacy temperature component: {e}")
        
        self._step = metadata["step"]
        logger.info(f"Loaded legacy components {loaded_components} from step {self._step}")
    
    def create_train_state(self) -> training_utils.TrainState:
        """
        Create OpenPI-compatible training state for checkpointing.
        
        IMPORTANT: This creates a TrainState that is fully compatible with OpenPI inference.
        - params contains ONLY π₀ model weights (no critic/temperature)
        - EMA params (if enabled) are the preferred weights for inference
        - Other components are saved separately via save_component_checkpoints()
        """
        # Extract ONLY π₀ parameters for OpenPI compatibility
        pi0_params = nnx.state(self.pi0_model, nnx.Param)
        
        # Create minimal dummy optimizer and state for OpenPI TrainState compatibility
        # Use a lightweight optimizer to avoid memory allocation issues
        dummy_tx = optax.sgd(learning_rate=1e-4)  # SGD has minimal memory overhead
        
        # Create dummy state structure without full allocation
        # Use jax.eval_shape to get structure without allocating memory
        def create_dummy_opt_state():
            trainable_filter = lambda path, x: True
            filtered_params = pi0_params.filter(trainable_filter)
            return dummy_tx.init(filtered_params)
        
        # In FSDP context, this will be properly sharded
        try:
            dummy_opt_state = create_dummy_opt_state()
        except (ValueError, RuntimeError) as e:
            # If we still get OOM, create an empty dummy state
            logger.warning(f"Creating minimal dummy optimizer state due to memory constraints: {e}")
            # Create minimal dummy state structure with floating point dtypes
            def create_float_placeholder(x):
                if x.dtype in [jnp.uint32, jnp.int32, jnp.uint64, jnp.int64]:
                    return jnp.zeros((), dtype=jnp.float32)
                elif x.dtype in [jnp.bool_]:
                    return jnp.zeros((), dtype=jnp.float32)
                else:
                    return jnp.zeros((), dtype=x.dtype)
                    
            dummy_opt_state = jax.tree.map(
                create_float_placeholder, 
                jax.eval_shape(create_dummy_opt_state)
            )
        
        # Initialize EMA parameters if enabled (π₀ only for inference compatibility)
        ema_params = None
        if self.config.use_ema:
            # EMA contains only π₀ weights - this is what gets saved to params/ for inference
            ema_params = pi0_params
        
        return training_utils.TrainState(
            step=self._step,
            params=pi0_params,  # ONLY π₀ weights - OpenPI inference compatible
            model_def=nnx.graphdef(self.pi0_model),  # π₀ model definition
            opt_state=dummy_opt_state,  # Dummy state - actual states managed in agent
            tx=dummy_tx,
            ema_decay=self.config.pi0_ema_decay if self.config.use_ema else None,
            ema_params=ema_params  # π₀ EMA weights for inference (if enabled)
        )
    
    def update_train_state_with_ema(self, train_state: training_utils.TrainState) -> training_utils.TrainState:
        """Update TrainState with fresh EMA parameters after training step.
        
        Args:
            train_state: Current training state
            
        Returns:
            Updated training state with refreshed EMA parameters
        """
        if not self.config.use_ema or train_state.ema_params is None:
            return train_state
            
        # Update EMA parameters using component-specific logic
        updated_ema_params = self._update_ema_params(train_state.ema_params)
        
        # Return updated train state
        return dataclasses.replace(train_state, ema_params=updated_ema_params)
    
    def update_from_train_state(self, train_state: training_utils.TrainState):
        """Update agent from OpenPI training state (π₀ only)."""
        # Update step counter
        self._step = train_state.step
        
        # Update π₀ model parameters (TrainState.params now contains only π₀ weights)
        nnx.update(self.pi0_model, train_state.params)
        
        # Note: Critic and temperature states are NOT updated from TrainState
        # Use load_component_checkpoints() for complete state recovery
        logger.info(f"Updated π₀ model from training state (step {self._step})")
    
    @property
    def step(self) -> int:
        """Current training step."""
        return self._step
    
    # ===============================================================================
    # ACRLPD TRAINSTATE CONVERSION FOR FSDP COMPATIBILITY
    # ===============================================================================
    
    def to_train_state(self, step: Optional[int] = None) -> "ACRLPDTrainState":
        """
        Convert agent to ACRLPDTrainState for FSDP training.
        
        This creates a pure JAX pytree containing all components that can be 
        sharded and JIT-compiled for distributed training.
        
        Args:
            step: Override step count (if None, uses current step)
            
        Returns:
            ACRLPDTrainState containing all component states
        """
        try:
            from ..training.acrlpd_train_state import create_train_state_from_components
        except ImportError:
            # Handle case when called from different contexts
            from training.acrlpd_train_state import create_train_state_from_components
        
        current_step = step if step is not None else self._step
        
        # 🔍 调试Agent转换为TrainState时的配置信息
        logger.info(f"🔍 [Agent->TrainState] 转换开始: π₀_action_dim={self.config.pi0_config.action_dim}, real_action_dim={self.config.real_action_dim}")
        logger.info(f"🔍 [Agent->TrainState] horizon={self.config.horizon_length}")
        
        # Create training state from all components
        train_state = create_train_state_from_components(
            step=current_step,
            pi0_model=self.pi0_model,
            pi0_tx=self.pi0_optimizer,
            critic_networks=self.critic_networks,
            critic_tx=self.critic_optimizer,
            temperature_module=self.temperature_module,
            temperature_tx=self.temperature_optimizer if self.temperature_module else None,
            pi0_ema_decay=self.config.pi0_ema_decay if self.config.use_ema else None,
            config=dataclasses.asdict(self.config)
        )
        
        logger.info(f"✅ [Agent->TrainState] 转换完成 (step {current_step})")
        return train_state
    
    def from_train_state(self, train_state: "ACRLPDTrainState") -> "ACRLPDPi0Agent":
        """
        Update agent from ACRLPDTrainState after FSDP training.
        
        This applies the updated parameters and optimizer states from the 
        training state back to the agent components.
        
        Args:
            train_state: Updated ACRLPDTrainState from training
            
        Returns:
            Self (for method chaining)
        """
        # Update step counter
        self._step = int(train_state.step)
        
        # Update π₀ model parameters
        nnx.update(self.pi0_model, train_state.pi0_params)
        self.pi0_optimizer_state.value = train_state.pi0_opt_state
        
        # Update critic networks parameters
        nnx.update(self.critic_networks, train_state.critic_params)
        self.critic_optimizer_state.value = train_state.critic_opt_state
        
        # Update temperature module (if present)
        if self.temperature_module is not None and train_state.temperature_params is not None:
            nnx.update(self.temperature_module, train_state.temperature_params)
            if train_state.temperature_opt_state is not None:
                self.temperature_optimizer_state.value = train_state.temperature_opt_state
        
        logger.info(f"✅ Updated agent from ACRLPDTrainState (step {self._step})")
        return self
    
    def create_fsdp_compatible_train_state(self, step: Optional[int] = None) -> "ACRLPDTrainState":
        """
        Alias for to_train_state() for clarity in FSDP contexts.
        
        Args:
            step: Override step count (if None, uses current step)
            
        Returns:
            ACRLPDTrainState for FSDP training
        """
        return self.to_train_state(step=step)


def create_acrlpd_pi0_agent(
    config: ACRLPDPi0Config,
    rng: jnp.ndarray
) -> ACRLPDPi0Agent:
    """
    Factory function to create ACRLPD + π₀ agent.
    
    Args:
        config: Agent configuration
        rng: Random number generator
        
    Returns:
        Initialized ACRLPDPi0Agent
    """
    # Split RNG for different components
    pi0_rng, critic_rng, temp_rng = jax.random.split(rng, 3)
    rngs = nnx.Rngs(pi0=pi0_rng, critic=critic_rng, temperature=temp_rng)
    agent = ACRLPDPi0Agent(config, rngs)
    
    logger.info(f"Created ACRLPD Pi0 Agent with {agent.pi0_model.action_dim} action dim, "
                f"{config.horizon_length} horizon length")
    
    return agent


# Predefined configurations for different scenarios
def get_droid_config() -> ACRLPDPi0Config:
    """Configuration optimized for DROID robot data with large-scale dataset tuning."""
    return ACRLPDPi0Config(
        # π₀ model configuration optimized for DROID
        pi0_config=_pi0.Pi0Config(
            action_dim=8,  # DROID action space
            action_horizon=10,
            max_token_len=180
        ),
        
        # ACRLPD parameters tuned for DROID's rich data
        horizon_length=10,
        batch_size=64,  # Smaller batch for memory efficiency with large dataset
        best_of_n_samples=16,  # Moderate sampling for speed
        use_best_of_n=True,
        diffusion_steps=8,  # Faster sampling during training
        
        # Critic configuration for complex state-action spaces
        critic_config=CriticConfig(
            num_critics=10,
            hidden_dims=[512, 512, 256],
            use_layer_norm=True,
            dropout_rate=0.1
        ),
        
        # Loss weights optimized for diverse DROID tasks
        loss_weights=LossWeights(
            bc_weight=0.01,   # Moderate BC regularization 
            critic_weight=1.0,
            actor_weight=1.0,  # Standard actor loss weight
            alpha_weight=0.5   # Moderate temperature learning
        ),
        
        # Training phases tuned for large dataset

        
        # OpenPI optimizer configurations optimized for DROID
        pi0_lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2000, peak_lr=3e-5, decay_steps=100000, decay_lr=3e-6
        ),
        critic_lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2000, peak_lr=5e-4, decay_steps=80000, decay_lr=5e-5
        ),
        pi0_optimizer=_optimizer.AdamW(
            b1=0.9, b2=0.95, weight_decay=1e-4, clip_gradient_norm=1.0
        ),
        critic_optimizer=_optimizer.AdamW(
            b1=0.9, b2=0.99, weight_decay=5e-4, clip_gradient_norm=2.0
        ),
        
        # Advanced sampling and temperature
        use_adaptive_temperature=True,
        initial_temperature=0.8,
        target_entropy_multiplier=0.6
    )


def get_aloha_config() -> ACRLPDPi0Config:
    """Configuration optimized for ALOHA bimanual manipulation tasks."""
    return ACRLPDPi0Config(
        # π₀ model configuration for bimanual ALOHA
        pi0_config=_pi0.Pi0Config(
            action_dim=14,  # Bimanual ALOHA action space (7 per arm)
            action_horizon=20,  # Longer sequences for complex manipulation
            max_token_len=250   # More tokens for bimanual tasks
        ),
        
        # ACRLPD parameters optimized for complex manipulation
        horizon_length=20,  # Longer horizons for multi-step manipulation
        batch_size=128,     # Balanced batch size for fine-tuning
        best_of_n_samples=32,  # More samples for precision tasks
        use_best_of_n=True,
        diffusion_steps=12,  # More steps for precision
        
        # Critic configuration for bimanual coordination
        critic_config=CriticConfig(
            num_critics=8,  # Smaller ensemble for faster training
            hidden_dims=[256, 256, 128],  # Smaller networks for fine-tuning
            use_layer_norm=True,
            dropout_rate=0.05  # Lower dropout for stability
        ),
        
        # Loss weights for fine-tuning regime
        loss_weights=LossWeights(
            bc_weight=0.001,  # Very weak BC for fine-tuning
            critic_weight=1.0,
            actor_weight=1.0,  # Standard actor loss weight
            alpha_weight=0.3   # Conservative temperature learning
        ),
        
        # Conservative training for fine-tuning
        freeze_pi0_backbone=True,  # Preserve pretrained features

        
        # OpenPI optimizer configurations for fine-tuning
        pi0_lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500, peak_lr=5e-6, decay_steps=30000, decay_lr=5e-7
        ),
        critic_lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500, peak_lr=1e-4, decay_steps=25000, decay_lr=1e-5
        ),
        pi0_optimizer=_optimizer.AdamW(
            b1=0.9, b2=0.999, weight_decay=5e-5, clip_gradient_norm=0.5  # Conservative settings
        ),
        critic_optimizer=_optimizer.AdamW(
            b1=0.9, b2=0.999, weight_decay=1e-4, clip_gradient_norm=1.0
        ),
        
        # Temperature settings for precision tasks
        use_adaptive_temperature=True,
        initial_temperature=0.5,  # Lower initial temperature for precision
        target_entropy_multiplier=0.3  # Conservative exploration
    )


def get_libero_config() -> ACRLPDPi0Config:
    """Configuration optimized for Libero simulation benchmarks."""
    return ACRLPDPi0Config(
        # π₀ model configuration for single-arm simulation
        pi0_config=_pi0.Pi0Config(
            action_dim=7,  # Single-arm action space
            action_horizon=5,  # Shorter horizons for reactive tasks
            max_token_len=180  # Standard token length
        ),
        
        # ACRLPD parameters tuned for simulation efficiency
        horizon_length=5,   # Shorter horizons for fast simulation feedback
        batch_size=256,     # Large batches for computational efficiency
        best_of_n_samples=64,  # More samples for exploration in simulation
        use_best_of_n=True,
        diffusion_steps=6,  # Fewer steps for speed in simulation
        
        # Critic configuration for simulation learning
        critic_config=CriticConfig(
            num_critics=12,  # Larger ensemble for simulation stability
            hidden_dims=[512, 512, 256, 128],  # Deeper networks for learning
            use_layer_norm=True,
            dropout_rate=0.15  # Higher dropout for regularization
        ),
        
        # Loss weights for simulation training
        loss_weights=LossWeights(
            bc_weight=0.1,   # Stronger BC constraint for simulation
            critic_weight=1.0,
            actor_weight=1.0,  # Standard actor loss weight
            alpha_weight=1.0   # Active temperature learning
        ),
        
        # Aggressive training for simulation
        freeze_pi0_backbone=False,  # Allow full fine-tuning in simulation

        
        # OpenPI optimizer configurations for simulation
        pi0_lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000, peak_lr=3e-4, decay_steps=50000, decay_lr=3e-5
        ),
        critic_lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000, peak_lr=1e-3, decay_steps=40000, decay_lr=1e-4
        ),
        pi0_optimizer=_optimizer.AdamW(
            b1=0.9, b2=0.95, weight_decay=1e-3, clip_gradient_norm=2.0  # Aggressive settings
        ),
        critic_optimizer=_optimizer.AdamW(
            b1=0.9, b2=0.95, weight_decay=1e-3, clip_gradient_norm=3.0
        ),
        
        # Temperature settings for exploration
        use_adaptive_temperature=True,
        initial_temperature=1.2,  # Higher initial temperature for exploration
        target_entropy_multiplier=0.8  # Encourage exploration
    )


# =================================================================================
# RLTrainConfig Integration - Factory Function
# =================================================================================

def create_acrlpd_pi0_agent_from_rl_config(
    rl_config,  # RLTrainConfig 
    rng: jnp.ndarray
) -> ACRLPDPi0Agent:
    """
    从统一RLTrainConfig创建ACRLPD + π₀ agent
    
    Args:
        rl_config: RLTrainConfig统一配置 
        rng: Random number generator
        
    Returns:
        初始化的ACRLPDPi0Agent
    """
    logger.info(f"Creating ACRLPD π₀ Agent from unified config: {rl_config.name}")
    
    # 提取关键参数
    action_dim = rl_config.qchunking.action_dim
    horizon_length = rl_config.qchunking.horizon_length
    batch_size = rl_config.batch_size
    
    logger.info(f"🔍 [工厂函数] 参数提取: action_dim={action_dim}, horizon={horizon_length}, batch_size={batch_size}")
    logger.info(f"🔍 [工厂函数] π₀配置检查: π₀_action_dim={rl_config.model.action_dim}")
    
    # 构建ACRLPDPi0Config
    agent_config = ACRLPDPi0Config(
        # π₀模型配置 - 直接使用RLTrainConfig的模型配置
        pi0_config=rl_config.model,
        real_action_dim=action_dim,  # 设置真实动作维度，来自QChunkingConfig
        
        # Q-chunking参数
        horizon_length=horizon_length,
        discount=rl_config.acrlpd.discount,
        q_aggregation=rl_config.acrlpd.q_aggregation,
        batch_size=batch_size,
        target_update_tau=rl_config.acrlpd.target_update_tau,
        
        # EMA配置
        use_ema=rl_config.acrlpd.use_ema,
        pi0_ema_decay=rl_config.acrlpd.pi0_ema_decay,
        critic_ema_decay=rl_config.acrlpd.critic_ema_decay,
        use_ema_for_inference=rl_config.acrlpd.use_ema_for_inference,
        
        # 采样配置
        best_of_n_samples=rl_config.acrlpd.num_action_samples,
        diffusion_steps=rl_config.acrlpd.diffusion_steps,
        use_best_of_n=rl_config.acrlpd.use_best_of_n,
        
        # Critic配置
        critic_config=CriticConfig(
            num_critics=rl_config.acrlpd.num_critics,
            hidden_dims=rl_config.acrlpd.critic_hidden_dims,
            use_layer_norm=True,
            dropout_rate=0.1
        ),
        
        # 损失权重
        loss_weights=LossWeights(
            bc_weight=rl_config.acrlpd.bc_loss_weight,
            critic_weight=1.0,
            actor_weight=1.0,
            alpha_weight=1.0
        ),
        
        # 温度控制
        use_adaptive_temperature=rl_config.acrlpd.use_adaptive_temperature,
        initial_temperature=rl_config.acrlpd.action_sampling_temperature,
        target_entropy_multiplier=rl_config.acrlpd.target_entropy_multiplier,
        
        # 训练阶段参数
        freeze_pi0_backbone=False,
        eval_frequency=rl_config.acrlpd.eval_frequency,
        save_frequency=rl_config.acrlpd.save_frequency,
        
        # 优化器配置 - 使用RLTrainConfig的学习率调度
        pi0_lr_schedule=rl_config.actor_lr_schedule,
        critic_lr_schedule=rl_config.critic_lr_schedule,
        pi0_optimizer=rl_config.actor_optimizer,
        critic_optimizer=rl_config.critic_optimizer
    )
    
    # 创建agent
    agent = create_acrlpd_pi0_agent(agent_config, rng)
    
    logger.info(f"✓ Created Agent from config '{rl_config.name}':")
    logger.info(f"  Action dim: {action_dim}")
    logger.info(f"  Horizon: {horizon_length}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  π₀ model: {type(rl_config.model).__name__}")
    
    return agent