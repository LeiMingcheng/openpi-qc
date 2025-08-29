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
    
    # Training phases (注意：现在使用RLTrainConfig.num_train_steps)
    # eval_frequency已移除 - 当前无真实环境评估实现
    # save_frequency已移除 - 使用RLTrainConfig.save_interval
    
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
        rngs: nnx.Rngs,
        lazy_init: bool = False
    ):
        super().__init__()
        self.config = config
        config.validate()
        
        # Lazy initialization mode - avoid creating duplicate models
        self.lazy_init = lazy_init
        self._initialized_from_fsdp = False
        
        if lazy_init:
            logger.info(" [Agent初始化] lazy initialization...")
            # Only save configuration, models will be set from FSDP state
            self.pi0_model = None
            self.critic_networks = None
            self.loss_computer = None
            self.temperature_module = None
            self.observation_encoder = None
            # Optimizers will be set in setup_from_fsdp_state if needed
            self.pi0_optimizer = None
            self.critic_optimizer = None
            self.temperature_optimizer = None
        else:
            logger.info(" [Agent完整初始化] 创建所有模型组件")
            # Create π₀ model - extract raw RNG key from RngStream
            pi0_raw_rng = rngs.pi0()  # Get the actual jax random key
            self.pi0_model = config.pi0_config.create(pi0_raw_rng)
            
            # Create critic networks with π₀ integration
            critic_raw_rng = rngs.critic()  # Get the actual jax random key
            logger.info(f" [Agent创建] 配置检查: π₀_action_dim={config.pi0_config.action_dim}, real_action_dim={config.real_action_dim}")
            logger.info(f" [Agent创建] Critic网络创建参数: horizon={config.horizon_length}, action_dim={config.real_action_dim}")
            
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
                actor_num_samples=config.best_of_n_samples,  #  传递num_action_samples参数
                initial_temperature=config.initial_temperature,
                real_action_dim=getattr(config, 'real_action_dim', 14),  # 添加real_action_dim参数
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
        
        if not lazy_init:
            # Initialize optimizer states using dummy parameters to get the structure
            # Mark these as non-trackable by nnx to avoid serialization issues
            dummy_pi0_params = nnx.state(self.pi0_model, nnx.Param)
            dummy_critic_params = self.critic_networks.online_params  # Use Linen params directly
            dummy_temp_params = nnx.state(self.temperature_module, nnx.Param) if self.temperature_module else {}
            
            # Use nnx.Variable with collection=False to exclude from state tracking
            self.pi0_optimizer_state = nnx.Variable(self.pi0_optimizer.init(dummy_pi0_params))
            self.critic_optimizer_state = nnx.Variable(self.critic_optimizer.init(dummy_critic_params))
            self.temperature_optimizer_state = nnx.Variable(self.temperature_optimizer.init(dummy_temp_params)) if self.temperature_optimizer else None
        else:
            # Lazy init mode - optimizer states will be set from FSDP state
            self.pi0_optimizer_state = None
            self.critic_optimizer_state = None
            self.temperature_optimizer_state = None
        
        logger.info(f"Created ACRLPDPi0Agent: π₀_action_dim={config.pi0_config.action_dim}, "
                   f"real_action_dim={config.real_action_dim}, horizon={config.horizon_length}, "
                   f"batch_size={config.batch_size}, EMA={'enabled' if config.use_ema else 'disabled'}")

    def setup_from_fsdp_state(self, fsdp_train_state):
        """
        从FSDP训练状态设置Agent模型组件，避免重复创建。
        
        Args:
            fsdp_train_state: FSDP训练状态 (ACRLPDTrainState)
        """
        if not self.lazy_init:
            logger.warning("Agent不是以lazy_init模式创建的，跳过FSDP状态设置")
            return
            
        if self._initialized_from_fsdp:
            logger.debug("Agent已经从FSDP状态设置过，跳过重复设置")
            return
        
        logger.debug(" [Agent FSDP设置] 开始从FSDP训练状态设置模型组件")
        
        # 重构π₀模型从FSDP状态
        self.pi0_model = nnx.merge(
            fsdp_train_state.pi0_model_def, 
            fsdp_train_state.pi0_params
        )
        logger.debug("✅ [Agent FSDP设置] π₀模型已从FSDP状态设置")
        
        # 重构Critic网络从FSDP状态 (Linen版本)
        # Create new critic networks and set parameters
        from agents.critic_networks import create_critic_networks
        self.critic_networks = create_critic_networks(
            config=self.config.critic_config,
            pi0_model=self.pi0_model,
            action_horizon=self.config.horizon_length,
            action_dim=self.config.real_action_dim,
            rngs=jax.random.PRNGKey(42),  # Dummy rng for structure
            pi0_config=self.config.pi0_config
        )
        self.critic_networks.online_params = fsdp_train_state.critic_params
        # Target params will be synced when needed
        logger.debug("✅ [Agent FSDP设置] Critic网络已从FSDP状态设置")
        
        # 创建observation encoder和loss computer（依赖于模型）
        self.observation_encoder = self._create_observation_encoder()
        
        # 创建loss computer和temperature module
        self.loss_computer, self.temperature_module = create_loss_computer(
            loss_weights=self.config.loss_weights,
            discount=self.config.discount,
            horizon_length=self.config.horizon_length,
            q_aggregation=self.config.q_aggregation,
            target_entropy_multiplier=self.config.target_entropy_multiplier,
            use_temperature=self.config.use_adaptive_temperature,
            actor_num_samples=self.config.best_of_n_samples,
            initial_temperature=self.config.initial_temperature,
            real_action_dim=getattr(self.config, 'real_action_dim', 14),  # 添加real_action_dim参数
            rngs=None  # Use None since we're not training the temperature module separately
        )
        logger.debug("✅ [Agent FSDP设置] Loss computer和temperature module已创建")
        
        # 创建必要的优化器（用于checkpoint兼容性）
        pi0_weight_decay_mask = None
        if self.config.freeze_pi0_backbone:
            pi0_weight_decay_mask = nnx.filterlib.to_predicate(nnx.Not(self.config.freeze_filter)) if hasattr(self.config, 'freeze_filter') else None
        
        self.pi0_optimizer = _optimizer.create_optimizer(
            self.config.pi0_optimizer, 
            self.config.pi0_lr_schedule,
            weight_decay_mask=pi0_weight_decay_mask
        )
        
        self.critic_optimizer = _optimizer.create_optimizer(
            self.config.critic_optimizer, 
            self.config.critic_lr_schedule
        )
        
        if self.temperature_module is not None:
            temp_optimizer_config = _optimizer.AdamW(weight_decay=0.0, clip_gradient_norm=1.0)
            self.temperature_optimizer = _optimizer.create_optimizer(
                temp_optimizer_config,
                self.config.critic_lr_schedule
            )
        else:
            self.temperature_optimizer = None
        
        # 初始化优化器状态（用于checkpoint兼容性）
        dummy_pi0_params = nnx.state(self.pi0_model, nnx.Param)
        dummy_critic_params = self.critic_networks.online_params  # Use Linen params directly
        dummy_temp_params = nnx.state(self.temperature_module, nnx.Param) if self.temperature_module else {}
        
        self.pi0_optimizer_state = nnx.Variable(self.pi0_optimizer.init(dummy_pi0_params))
        self.critic_optimizer_state = nnx.Variable(self.critic_optimizer.init(dummy_critic_params))
        self.temperature_optimizer_state = nnx.Variable(self.temperature_optimizer.init(dummy_temp_params)) if self.temperature_optimizer else None
        
        self._initialized_from_fsdp = True
        logger.debug("✅ [Agent FSDP设置] 完成，Agent现在可以用于checkpoint和状态管理")
    
    # ============================================================================
    # 状态管理方法 (训练功能已迁移到FSDP系统)
    # ============================================================================
    
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
            
            # Save both online and target parameters for Linen version
            ckptr.save(critic_dir / "online_params", {"params": self.critic_networks.online_params})
            ckptr.save(critic_dir / "target_params", {"params": self.critic_networks.target_params})
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
                    # Try new Linen format first (separate online/target)
                    if (critic_dir / "online_params").exists() and (critic_dir / "target_params").exists():
                        online_params = ckptr.restore(critic_dir / "online_params")["params"]
                        target_params = ckptr.restore(critic_dir / "target_params")["params"]
                        critic_opt_state = ckptr.restore(critic_dir / "optimizer_state")["opt_state"]
                        
                        self.critic_networks.online_params = online_params
                        self.critic_networks.target_params = target_params
                        self.critic_optimizer_state.value = critic_opt_state
                        loaded_components.append("critic")
                    # Fallback to old format
                    elif (critic_dir / "params").exists():
                        logger.warning("Loading old critic checkpoint format - target networks will be synced from online")
                        critic_params = ckptr.restore(critic_dir / "params")["params"]
                        critic_opt_state = ckptr.restore(critic_dir / "optimizer_state")["opt_state"]
                        
                        # For old format, assume it's online params and sync to target
                        self.critic_networks.online_params = critic_params
                        self.critic_networks.sync_target_networks()
                        self.critic_optimizer_state.value = critic_opt_state
                        loaded_components.append("critic")
                    else:
                        logger.warning(f"No critic params found at {critic_dir}")
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
    
    # Legacy checkpoint loading removed - now using orbax format exclusively
    
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
        
        #  调试Agent转换为TrainState时的配置信息
        logger.info(f" [Agent->TrainState] 转换开始: π₀_action_dim={self.config.pi0_config.action_dim}, real_action_dim={self.config.real_action_dim}")
        logger.info(f" [Agent->TrainState] horizon={self.config.horizon_length}")
        
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
        
        # Update critic networks parameters (Linen version)
        self.critic_networks.online_params = train_state.critic_params
        # Note: target_params are managed by the training loop's sync calls
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
            # Handle lazy_init case where pi0_model might be None
            if self.pi0_model is None:
                logger.warning("observation_encoder_fn called before pi0_model is set - using fallback")
                batch_size = observations.state.shape[0] if observations.state is not None else 1
                llm_dim = 1024  # Default dimension
                state_dim = self.config.pi0_config.action_dim
                total_dim = llm_dim + state_dim
                return jnp.zeros((batch_size, total_dim))
            
            try:
                from training.acrlpd_train_state import combine_pi0_and_state_features
                return combine_pi0_and_state_features(self.pi0_model, observations)
            except ImportError as e:
                logger.error(f"Failed to import combine_pi0_and_state_features: {e}")
                # Fallback: create dummy features with correct dimensions
                batch_size = observations.state.shape[0] if observations.state is not None else 1
                llm_dim = getattr(self.pi0_model.config, 'width', 1024) if self.pi0_model else 1024  # Default to 1024
                state_dim = self.config.pi0_config.action_dim
                total_dim = llm_dim + state_dim
                return jnp.zeros((batch_size, total_dim))
        
        return observation_encoder_fn


def create_acrlpd_pi0_agent(
    config: ACRLPDPi0Config,
    rng: jnp.ndarray,
    lazy_init: bool = False
) -> ACRLPDPi0Agent:
    """
    Factory function to create ACRLPD + π₀ agent.
    
    Args:
        config: Agent configuration
        rng: Random number generator
        lazy_init: If True, skip model creation and wait for FSDP state setup
        
    Returns:
        Initialized ACRLPDPi0Agent
    """
    # Split RNG for different components
    pi0_rng, critic_rng, temp_rng = jax.random.split(rng, 3)
    rngs = nnx.Rngs(pi0=pi0_rng, critic=critic_rng, temperature=temp_rng)
    agent = ACRLPDPi0Agent(config, rngs, lazy_init=lazy_init)
    
    if not lazy_init and agent.pi0_model is not None:
        logger.info(f"Created ACRLPD Pi0 Agent with {agent.pi0_model.action_dim} action dim, "
                    f"{config.horizon_length} horizon length")
    else:
        logger.info(f"Created ACRLPD Pi0 Agent in lazy_init mode with "
                    f"{config.horizon_length} horizon length")
    
    return agent


# Note: Legacy predefined configurations removed.
# Use unified RLTrainConfig system with create_acrlpd_pi0_agent_from_rl_config() instead.


# =================================================================================
# RLTrainConfig Integration - Factory Function
# =================================================================================

def create_acrlpd_pi0_agent_from_rl_config(
    rl_config,  # RLTrainConfig 
    rng: jnp.ndarray,
    lazy_init: bool = False
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
    
    logger.info(f" 参数提取: action_dim={action_dim}, horizon={horizon_length}, batch_size={batch_size}")
    logger.info(f" π₀配置检查: π₀_action_dim={rl_config.model.action_dim}")
    
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
        
        # 训练阶段参数 (现在使用RLTrainConfig.num_train_steps)
        freeze_pi0_backbone=False,
        # eval_frequency和save_frequency已移除
        
        # 优化器配置 - 使用RLTrainConfig的学习率调度
        pi0_lr_schedule=rl_config.actor_lr_schedule,
        critic_lr_schedule=rl_config.critic_lr_schedule,
        pi0_optimizer=rl_config.actor_optimizer,
        critic_optimizer=rl_config.critic_optimizer
    )
    
    # 创建agent（支持lazy_init模式）
    agent = create_acrlpd_pi0_agent(agent_config, rng, lazy_init=lazy_init)
    
    logger.info(f"✓ Created Agent from config '{rl_config.name}':")
    logger.info(f"  Action dim: {action_dim}")
    logger.info(f"  Horizon: {horizon_length}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  π₀ model: {type(rl_config.model).__name__}")
    
    return agent