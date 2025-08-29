"""
Training Loop for ACRLPD + π₀ Integration.

This module provides the complete training system for ACRLPD + π₀ agents,
including offline pretraining, online fine-tuning, evaluation, and monitoring.

Key features:
- Offline data pretraining with behavior cloning regularization
- Online environment interaction and fine-tuning
- Comprehensive evaluation and metrics tracking
- Checkpoint management and recovery
- WandB integration for experiment monitoring
- Multi-GPU training support
"""

import logging
import os
import time
from typing import Dict, Any, Optional, Callable, Tuple
import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import wandb
# tqdm removed for professional logging
import math

import sys

# Add ac_training root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.acrlpd_pi0_agent import ACRLPDPi0Agent
from agents.loss_functions import LossInfo
from data import ACRLPDDataLoader
from config import RLTrainConfig
from utils.memory_monitor import enable_memory_monitoring, log_memory_usage
import openpi.models.model as _model
import openpi.training.checkpoints as _checkpoints
import openpi.training.sharding as sharding

logger = logging.getLogger(__name__)
# 确保logger能正确输出到console
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def create_dynamic_lr_schedule(
    base_lr: float, 
    total_steps: int, 
    warmup_epochs: int, 
    total_epochs: int, 
    steps_per_epoch: int,
    lr_min_factor: float = 0.1,
    intra_epoch_min_factor: float = 0.9,
    lr_absolute_min: float = 1e-7  # 学习率绝对下限
):
    """
    创建双层cosine学习率调节函数：Epoch间 + Epoch内
    
    Args:
        base_lr: 基础学习率（peak值）
        total_steps: 总训练步数
        warmup_epochs: 预热epoch数
        total_epochs: 总epoch数
        steps_per_epoch: 每个epoch步数
        lr_min_factor: Epoch间最小学习率因子
        intra_epoch_min_factor: Epoch内最小学习率因子
        lr_absolute_min: 学习率绝对下限
    
    Returns:
        学习率调节函数 step -> learning_rate
    """
    def schedule(step):
        import jax
        import jax.numpy as jnp
        
        # 计算当前epoch和epoch内步数
        current_epoch = step // steps_per_epoch
        step_in_epoch = step % steps_per_epoch
        
        # Epoch间调节：使用JAX条件函数
        def warmup_lr():
            return base_lr * (current_epoch + 1) / warmup_epochs
        
        def normal_lr():
            effective_epoch = current_epoch - warmup_epochs
            effective_total_epochs = total_epochs - warmup_epochs
            
            # 使用JAX条件函数避免布尔转换
            cosine_progress = jnp.minimum(effective_epoch / jnp.maximum(effective_total_epochs, 1.0), 1.0)
            epoch_factor = lr_min_factor + (1 - lr_min_factor) * \
                0.5 * (1 + jnp.cos(jnp.pi * cosine_progress))
            return base_lr * epoch_factor
        
        # 使用JAX条件函数
        epoch_lr_base = jax.lax.cond(
            current_epoch < warmup_epochs,
            warmup_lr,
            normal_lr
        )
        
        # Epoch内调节：使用JAX函数
        intra_progress = jnp.minimum(step_in_epoch / jnp.maximum(steps_per_epoch, 1.0), 1.0)
        intra_factor = intra_epoch_min_factor + (1 - intra_epoch_min_factor) * \
            0.5 * (1 + jnp.cos(jnp.pi * intra_progress))
        
        # 最终学习率
        final_lr = epoch_lr_base * intra_factor
        
        # 应用绝对下限
        return jnp.maximum(final_lr, lr_absolute_min)
    
    return schedule


@dataclasses.dataclass
class ACRLPDTrainingConfig:
    """额外的训练循环参数，配合RLTrainConfig使用"""
    
    # 评估和日志 (注意：现在使用RLTrainConfig.num_train_steps作为总步数)
    # eval_frequency已移除 - 当前无真实环境评估实现
    num_eval_episodes: int = 10            # 每次评估的episode数（当前未使用）
    eval_batch_size: int = 64              # 评估批次大小
    
    # 训练行为配置（已移至RLTrainConfig.acrlpd中）
    early_stopping_patience: int = 50      # 早停耐心值
    
    # 环境配置（在线训练用）
    env_name: Optional[str] = None         # 环境名称
    max_episode_steps: int = 200           # 每episode最大步数


class TrainingMetrics:
    """Tracks training metrics and statistics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.step_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_eval_reward = -float('inf')
        self.steps_since_improvement = 0
    
    def update(self, step: int, loss_info: LossInfo, **kwargs):
        """Update metrics with new training step data."""
        self.step_count = step
        
        # Loss metrics
        self.metrics.update({
            'train/total_loss': float(loss_info.total_loss),
            'train/critic_loss': float(loss_info.critic_loss),
            'train/bc_loss': float(loss_info.bc_loss),
            # π₀ loss is included in bc_loss
            'train/alpha_loss': float(loss_info.alpha_loss),
            'train/q_mean': float(loss_info.q_mean),
            'train/q_std': float(loss_info.q_std),
            'train/target_q_mean': float(loss_info.target_q_mean),
            'train/td_error_mean': float(loss_info.td_error_mean),
            'train/alpha_value': float(loss_info.alpha_value),
            'train/entropy_estimate': float(loss_info.entropy_estimate),
            'train/valid_samples': float(loss_info.valid_samples),
            'train/mask_ratio': float(loss_info.mask_ratio)
        })
        
        # Additional metrics
        self.metrics.update(kwargs)
        self.metrics['step'] = step
    
    def add_episode_result(self, reward: float, length: int):
        """Add episode result for evaluation."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
    
    def get_episode_stats(self) -> Dict[str, float]:
        """Get episode statistics."""
        if not self.episode_rewards:
            return {}
        
        return {
            'eval/episode_reward_mean': np.mean(self.episode_rewards),
            'eval/episode_reward_std': np.std(self.episode_rewards),
            'eval/episode_reward_max': np.max(self.episode_rewards),
            'eval/episode_reward_min': np.min(self.episode_rewards),
            'eval/episode_length_mean': np.mean(self.episode_lengths),
            'eval/num_episodes': len(self.episode_rewards)
        }
    
    def check_improvement(self, eval_reward: float) -> bool:
        """Check if evaluation reward improved."""
        if eval_reward > self.best_eval_reward:
            self.best_eval_reward = eval_reward
            self.steps_since_improvement = 0
            return True
        else:
            self.steps_since_improvement += 1
            return False


class ACRLPDCheckpointManager:
    """ACRLPD-specific wrapper for OpenPI checkpoint management."""
    
    def __init__(self, checkpoint_dir: str, keep_period: int = 10000, overwrite: bool = False, resume: bool = False):
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Initialize OpenPI checkpoint manager
        self.manager, self.resuming = _checkpoints.initialize_checkpoint_dir(
            checkpoint_dir=self.checkpoint_dir,
            keep_period=keep_period,
            overwrite=overwrite,
            resume=resume
        )
        
        # 🔍 新增：自动检测可resume的checkpoint
        self.auto_resume_path = None
        if resume and not overwrite:
            self.auto_resume_path = self._find_latest_checkpoint()
    
    def save_checkpoint(
        self,
        agent: ACRLPDPi0Agent,
        dataloader: Any,
        step: int
    ) -> str:
        """
        Save agent checkpoint with dual strategy:
        1. OpenPI-compatible format (π₀ only) for inference 
        2. Complete component state for training recovery
        """
        try:
            # 🔍 关键修复：检查checkpoint路径是否已存在
            checkpoint_path = str(self.checkpoint_dir / str(step))
            if os.path.exists(checkpoint_path):
                logger.info(f"Checkpoint {checkpoint_path} already exists, skipping duplicate save")
                return checkpoint_path
            # === STRATEGY 1: OpenPI-compatible format (π₀ only) ===
            # This creates standard params/ directory that can be used for inference
            train_state = agent.create_train_state()
            
            _checkpoints.save_state(
                checkpoint_manager=self.manager,
                state=train_state,
                data_loader=dataloader,
                step=step
            )
            logger.info(f"Saved OpenPI-compatible π₀ weights at step {step}")
            
            # === STRATEGY 2: Complete component state ===
            # Save all components independently for training recovery
            components_dir = self.checkpoint_dir / str(step) / "components"
            agent.save_component_checkpoints(str(components_dir), step)
            logger.info(f"Saved complete component state at step {step}")
            
            checkpoint_path = str(self.checkpoint_dir / str(step))
            logger.info(f"Dual checkpoint saved: {checkpoint_path}")
            logger.info("  - params/: π₀ weights (OpenPI inference compatible)")
            logger.info("  - components/: All components (training recovery)")
            
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint at step {step}: {e}")
            return ""
    
    def load_checkpoint(
        self, 
        agent: ACRLPDPi0Agent, 
        dataloader: Any,
        step: Optional[int] = None
    ) -> Tuple[ACRLPDPi0Agent, int, Optional[Dict]]:
        """
        Load agent checkpoint with intelligent strategy:
        1. Try loading complete component state first (preferred for training)
        2. Fallback to OpenPI format (π₀ only, for inference compatibility)
        
        Returns:
            Tuple[agent, step, saved_metrics]: Updated agent, checkpoint step, and optional metrics
        """
        try:
            # Determine step to load
            if step is None:
                available_steps = list(self.manager.all_steps())
                if not available_steps:
                    raise ValueError("No checkpoints found")
                step = max(available_steps)
            
            components_dir = self.checkpoint_dir / str(step) / "components"
            
            # === STRATEGY 1: Try complete component state first ===
            if components_dir.exists():
                logger.info(f"Found complete component state at step {step}")
                
                # Check if it's orbax format (preferred) or legacy pickle format
                pi0_dir = components_dir / "pi0"
                if (pi0_dir / "params").exists():
                    # New orbax format
                    agent.load_component_checkpoints(str(components_dir))
                    logger.info(f"Loaded complete training state (orbax format) from step {step}")
                elif (pi0_dir / "model_params.pkl").exists():
                    # Legacy pickle format
                    logger.warning("Loading from legacy pickle format - consider migrating to orbax")
                    agent.load_legacy_component_checkpoints(str(components_dir))
                    logger.info(f"Loaded complete training state (legacy format) from step {step}")
                else:
                    logger.error(f"Unrecognized checkpoint format at {components_dir}")
                    raise ValueError(f"Cannot determine checkpoint format at {components_dir}")
                
                # 🔍 尝试加载metrics (容错处理)
                saved_metrics = None
                try:
                    metrics_path = self.checkpoint_dir / str(step) / "metrics" / "metrics"
                    if metrics_path.exists():
                        # TODO: 实现metrics加载逻辑
                        logger.debug(f"Found metrics file at {metrics_path}")
                    else:
                        logger.debug(f"No metrics file found at {metrics_path}")
                except Exception as e:
                    logger.warning(f"Failed to load metrics for step {step}: {e}")
                    
                return agent, step, saved_metrics
            
            # === STRATEGY 2: Fallback to OpenPI format ===
            logger.info(f"Complete component state not found, using OpenPI format")
            logger.warning("Loading π₀ only - critic and temperature states will be reinitialized")
            
            # Create template training state
            template_state = agent.create_train_state()
            
            # Restore using OpenPI checkpoint system
            restored_state = _checkpoints.restore_state(
                checkpoint_manager=self.manager,
                state=template_state,
                data_loader=dataloader,
                step=step
            )
            
            # Update agent from restored state (π₀ only)
            agent.update_from_train_state(restored_state)
            
            logger.info(f"Loaded π₀ weights from OpenPI format at step {restored_state.step}")
            
            # 🔍 尝试加载metrics (容错处理)
            saved_metrics = None
            try:
                metrics_path = self.checkpoint_dir / str(restored_state.step) / "metrics" / "metrics"
                if metrics_path.exists():
                    # TODO: 实现metrics加载逻辑
                    logger.debug(f"Found metrics file at {metrics_path}")
                else:
                    logger.debug(f"No metrics file found at {metrics_path}")
            except Exception as e:
                logger.warning(f"Failed to load metrics for step {restored_state.step}: {e}")
            
            return agent, restored_state.step, saved_metrics
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def _find_latest_checkpoint(self) -> Optional[str]:
        """查找最新的可用checkpoint"""
        if not self.checkpoint_dir.exists():
            return None
            
        available_steps = []
        for step_dir in self.checkpoint_dir.iterdir():
            if step_dir.is_dir() and step_dir.name.isdigit():
                step = int(step_dir.name)
                # 验证checkpoint完整性
                if self._validate_checkpoint(step_dir):
                    available_steps.append(step)
        
        if available_steps:
            latest_step = max(available_steps)
            checkpoint_path = str(self.checkpoint_dir / str(latest_step))
            logger.info(f"🔍 Found latest checkpoint: {checkpoint_path}")
            return checkpoint_path
        else:
            logger.info("🔍 No valid checkpoints found for auto-resume")
            return None
    
    def _validate_checkpoint(self, checkpoint_dir: Path) -> bool:
        """验证checkpoint完整性"""
        # 检查必需的文件/目录是否存在
        components_dir = checkpoint_dir / "components"
        params_dir = checkpoint_dir / "params"
        
        is_valid = components_dir.exists() or params_dir.exists()
        if is_valid:
            logger.debug(f"✅ Valid checkpoint: {checkpoint_dir.name}")
        else:
            logger.debug(f"❌ Invalid checkpoint: {checkpoint_dir.name}")
        
        return is_valid


class ACRLPDTrainer:
    """Complete training system for ACRLPD + π₀ agents with OpenPI integration."""
    
    def __init__(
        self,
        agent: Optional[ACRLPDPi0Agent],
        dataloader: ACRLPDDataLoader,
        rl_config: RLTrainConfig,
        training_config: ACRLPDTrainingConfig,
        eval_fn: Optional[Callable] = None,
        # FSDP参数
        mesh: Optional[jax.sharding.Mesh] = None,
        data_sharding: Optional[jax.sharding.Sharding] = None,
        replicated_sharding: Optional[jax.sharding.Sharding] = None,
        agent_rng: Optional[jax.Array] = None,
        # 全局优化器参数（修复pytree一致性）
        global_pi0_tx: Optional[Any] = None,
        global_critic_tx: Optional[Any] = None
    ):
        self.agent = agent  # May be None initially
        self.dataloader = dataloader
        self.rl_config = rl_config
        self.training_config = training_config
        self.eval_fn = eval_fn
        self.agent_rng = agent_rng if agent_rng is not None else jax.random.PRNGKey(42)
        
        # FSDP support
        self.mesh = mesh
        self.data_sharding = data_sharding
        self.replicated_sharding = replicated_sharding
        self.use_fsdp = mesh is not None
        
        # 全局优化器（修复pytree一致性）
        self.global_pi0_tx = global_pi0_tx
        self.global_critic_tx = global_critic_tx
        
        # Training state (initialize before FSDP setup)
        self.rng = jax.random.PRNGKey(42)
        self.current_step = 0
        self.is_online_phase = False
        
        if self.use_fsdp:
            logger.info(f"FSDP enabled with mesh: {mesh}")
            # FSDP setup will be done later after main script provides components
            logger.info("FSDP components will be provided by main script")
        else:
            logger.info("FSDP disabled - creating agent for single device training")
            if self.agent is None:
                from agents.acrlpd_pi0_agent import create_acrlpd_pi0_agent_from_rl_config
                self.agent = create_acrlpd_pi0_agent_from_rl_config(rl_config, self.agent_rng)
        
        # Initialize systems
        self.metrics = TrainingMetrics()
        
        # Cache learning rate schedules for efficient logging (avoid recreating optax schedules)
        self._cached_actor_schedule = self.rl_config.get_effective_actor_lr_schedule().create()
        self._cached_critic_schedule = self.rl_config.get_effective_critic_lr_schedule().create()
        self.checkpoint_manager = ACRLPDCheckpointManager(
            checkpoint_dir=str(rl_config.checkpoint_dir),
            keep_period=rl_config.keep_period or 10000,
            overwrite=rl_config.overwrite,
            resume=rl_config.resume
        )
        
        # Initialize WandB if enabled
        if rl_config.wandb_enabled:
            self._init_wandb()
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        try:
            wandb.init(
                project=self.rl_config.project_name,
                name=self.rl_config.exp_name,
                config={
                    'rl_config': dataclasses.asdict(self.rl_config),
                    'training_config': dataclasses.asdict(self.training_config),
                    'agent_config': dataclasses.asdict(self.agent.config) if hasattr(self.agent, 'config') else {}
                }
            )
            logger.info("Initialized WandB logging")
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")
            # Disable wandb in rl_config by creating a copy
            object.__setattr__(self.rl_config, 'wandb_enabled', False)
    
    def _get_current_learning_rates(self):
        """获取真实的动态学习率"""
        acrlpd_config = self.rl_config.acrlpd
        current_step = self.current_step
        
        if acrlpd_config.enable_epoch_based_lr_schedule:
            # 使用动态计算
            actor_schedule = create_dynamic_lr_schedule(
                base_lr=acrlpd_config.actor_lr,  # 使用配置中的actor_lr
                total_steps=self.rl_config.num_train_steps,
                warmup_epochs=acrlpd_config.warmup_epochs,
                total_epochs=acrlpd_config.total_epochs,
                steps_per_epoch=acrlpd_config.steps_per_epoch,
                lr_min_factor=acrlpd_config.lr_min_factor,
                intra_epoch_min_factor=acrlpd_config.intra_epoch_min_factor,
                lr_absolute_min=getattr(acrlpd_config, 'lr_absolute_min', 1e-7)
            )
            
            critic_schedule = create_dynamic_lr_schedule(
                base_lr=acrlpd_config.critic_lr,  # 使用配置中的critic_lr
                total_steps=self.rl_config.num_train_steps,
                warmup_epochs=acrlpd_config.warmup_epochs,
                total_epochs=acrlpd_config.total_epochs,
                steps_per_epoch=acrlpd_config.steps_per_epoch,
                lr_min_factor=acrlpd_config.lr_min_factor,
                intra_epoch_min_factor=acrlpd_config.intra_epoch_min_factor,
                lr_absolute_min=getattr(acrlpd_config, 'lr_absolute_min', 1e-7)
            )
            
            actor_lr = actor_schedule(current_step)
            critic_lr = critic_schedule(current_step)
            
            # 计算调节因子用于详细日志
            current_epoch = current_step // acrlpd_config.steps_per_epoch
            step_in_epoch = current_step % acrlpd_config.steps_per_epoch
            
            # 计算epoch因子
            if current_epoch < acrlpd_config.warmup_epochs:
                epoch_factor = (current_epoch + 1) / acrlpd_config.warmup_epochs
            else:
                effective_epoch = current_epoch - acrlpd_config.warmup_epochs
                effective_total_epochs = acrlpd_config.total_epochs - acrlpd_config.warmup_epochs
                
                if effective_total_epochs > 0:
                    cosine_progress = min(effective_epoch / effective_total_epochs, 1.0)
                    epoch_factor = acrlpd_config.lr_min_factor + (1 - acrlpd_config.lr_min_factor) * \
                        0.5 * (1 + math.cos(math.pi * cosine_progress))
                else:
                    epoch_factor = 1.0
            
            # 计算intra-epoch因子
            if acrlpd_config.steps_per_epoch > 0:
                intra_progress = min(step_in_epoch / acrlpd_config.steps_per_epoch, 1.0)
                intra_epoch_factor = acrlpd_config.intra_epoch_min_factor + \
                    (1 - acrlpd_config.intra_epoch_min_factor) * 0.5 * (1 + math.cos(math.pi * intra_progress))
            else:
                intra_epoch_factor = 1.0
                
        else:
            # 使用OpenPI标准学习率调度（step-based调节）
            # 从缓存的调度器获取当前步骤的真实学习率
            actor_lr = float(self._cached_actor_schedule(current_step))
            critic_lr = float(self._cached_critic_schedule(current_step))
            epoch_factor = 1.0  # OpenPI调度不使用epoch概念
            intra_epoch_factor = 1.0  # OpenPI调度不使用intra-epoch概念
            
        return {
            'actor_lr': actor_lr,
            'critic_lr': critic_lr,
            'epoch_factor': epoch_factor,
            'intra_epoch_factor': intra_epoch_factor,
            'total_factor': epoch_factor * intra_epoch_factor
        }
    
    def train(
        self,
        resume_from: Optional[str] = None
    ) -> ACRLPDPi0Agent:
        """
        Run complete training pipeline.
        
        Args:
            resume_from: Path to checkpoint to resume from
            
        Returns:
            Trained agent
        """
        logger.info("Starting ACRLPD + π₀ training")
        
        # 启用详细内存监控
        enable_memory_monitoring()
        
        # 🔍 增强resume逻辑
        if resume_from:
            self._resume_training(resume_from)
        elif hasattr(self.checkpoint_manager, 'auto_resume_path') and self.checkpoint_manager.auto_resume_path:
            # 自动resume模式
            logger.info(f"🤖 Auto-resume detected: {self.checkpoint_manager.auto_resume_path}")
            self._resume_training(self.checkpoint_manager.auto_resume_path)
        
        # Phase 1: Offline pretraining (now uses num_train_steps for total training)
        logger.info("Starting offline pretraining phase")
        self.agent = self._train_offline()
        

        
        # Final evaluation and checkpoint
        self._final_evaluation()
        
        logger.info("Training completed successfully")
        return self.agent
    
    def _train_offline(self) -> ACRLPDPi0Agent:
        """Run offline pretraining phase with gradient accumulation support."""
        import time  # 在函数开始就导入time模块
        start_step = self.current_step
        target_steps = self.rl_config.num_train_steps  # 使用正确的训练步数
        
        # 🔧 从配置中获取梯度积累参数（用户要求的完整功能）
        gradient_accumulation_steps = getattr(
            self.rl_config.acrlpd, 'gradient_accumulation_steps', 4
        )
        enable_gradient_accumulation = getattr(
            self.rl_config.acrlpd, 'enable_gradient_accumulation', True
        )
        max_grad_norm = getattr(
            self.rl_config.acrlpd, 'max_grad_norm', 1.0
        )
        
        if not enable_gradient_accumulation:
            gradient_accumulation_steps = 1
            logger.info("🔧 梯度积累已禁用，使用标准训练模式")
        else:
            logger.info(f"🔧 梯度积累已启用: {gradient_accumulation_steps}步积累，梯度裁剪阈值: {max_grad_norm}")
        
        # Add system monitoring for tqdm
        import psutil
        
        def get_memory_gpu_info():
            """Get current memory and GPU usage for monitoring."""
            try:
                # Memory usage
                process = psutil.Process()
                memory_gb = process.memory_info().rss / (1024**3)
                
                # GPU usage (fallback to nvidia-smi if GPUtil not available)
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Primary GPU
                        gpu_mem_used = gpu.memoryUsed / 1024  # Convert MB to GB
                        gpu_mem_total = gpu.memoryTotal / 1024
                        gpu_util = gpu.load * 100
                        return f"RAM:{memory_gb:.1f}GB GPU:{gpu_mem_used:.1f}/{gpu_mem_total:.1f}GB({gpu_util:.0f}%)"
                except ImportError:
                    # Fallback: Use nvidia-smi for GPU info
                    try:
                        import subprocess
                        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], 
                                              capture_output=True, text=True, timeout=2)
                        if result.returncode == 0:
                            lines = result.stdout.strip().split('\n')
                            if lines and lines[0]:
                                gpu_mem_used, gpu_mem_total, gpu_util = lines[0].split(', ')
                                gpu_mem_used_gb = float(gpu_mem_used) / 1024
                                gpu_mem_total_gb = float(gpu_mem_total) / 1024
                                return f"RAM:{memory_gb:.1f}GB GPU:{gpu_mem_used_gb:.1f}/{gpu_mem_total_gb:.1f}GB({gpu_util}%)"
                    except Exception:
                        pass
                
                return f"RAM:{memory_gb:.1f}GB GPU:N/A"
            except Exception:
                return "Monitor:Error"
        
        # 初始化专业日志和Epoch机制
        logger.info(f"=== Starting Offline Training ===")
        logger.info(f"Steps: {start_step} -> {target_steps} ({target_steps - start_step} steps)")
        logger.info(f"Batch Size: {self.rl_config.batch_size}")
        
        # 学习率信息
        acrlpd_config = self.rl_config.acrlpd
        
        if acrlpd_config.enable_epoch_based_lr_schedule:
            steps_per_epoch = getattr(acrlpd_config, 'steps_per_epoch', 10000)
            total_epochs = getattr(acrlpd_config, 'total_epochs', 100)
            logger.info(f"Learning Rate Schedule: Epoch-based {acrlpd_config.lr_decay_strategy} decay enabled")
            logger.info(f"Epoch Configuration: {steps_per_epoch} steps/epoch, {total_epochs} total epochs")
            logger.info(f"Warmup: {acrlpd_config.warmup_epochs} epochs, Min factor: {acrlpd_config.lr_min_factor}")
            if acrlpd_config.intra_epoch_lr_decay:
                logger.info(f"Intra-epoch: {acrlpd_config.intra_epoch_strategy} decay, Min factor: {acrlpd_config.intra_epoch_min_factor}")
            
            # Epoch机制初始化（仅在启用时）
            current_epoch = 0
            steps_in_current_epoch = 0
            epoch_losses = []
            epoch_start_time = time.time()
        else:
            logger.info("Learning Rate Schedule: Using step-based schedule only")
        
        # 记录训练开始时的内存状态
        log_memory_usage(start_step, self.agent.create_train_state() if hasattr(self.agent, 'create_train_state') else None, "offline_training_start")
        
        # Gradient accumulation state
        accumulated_gradients = None
        accumulation_count = 0
        
        # 预编译JIT函数（一次编译，多次使用）
        if not hasattr(self, '_jit_compute_gradients'):
            logger.info("Precompiling JIT gradient computation function...")
            self._jit_compute_gradients = self._create_jit_compute_gradients()
            logger.info("JIT gradient computation function compiled successfully")
        
        if not hasattr(self, '_jit_apply_gradients'):
            logger.info("Precompiling JIT gradient application function...")
            self._jit_apply_gradients = self._create_jit_apply_gradients()
            logger.info("JIT gradient application function compiled successfully")
        
        # 梯度累积信息记录
        if gradient_accumulation_steps > 1:
            logger.info(f"Gradient accumulation enabled: {gradient_accumulation_steps} steps")
            logger.info(f"Effective batch_size: {self.rl_config.batch_size} x {gradient_accumulation_steps} = {self.rl_config.batch_size * gradient_accumulation_steps}")
        
        # # 🎯 AC_Training性能分析：详细时间统计初始化
        # ac_perf_stats = {
        #     'total_iter_times': [],
        #     'grad_accumulation_times': [],
        #     'data_loading_times': [],
        #     'grad_compute_times': [],
        #     'grad_process_times': [],
        #     'param_update_times': [],
        #     'config_convert_times': [],
        #     'jax_sync_times': [],
        #     'memory_monitor_times': [],
        #     'logging_times': [],
        #     'existing_perf_analysis_times': []
        # }
        
        # # 🔍 JIT编译时间分析：追踪第一次编译
        # jit_compilation_stats = {
        #     'first_grad_compute_compilation': None,
        #     'first_grad_apply_compilation': None,
        #     'first_model_forward_compilation': None,
        #     'compilation_completed': False
        # }
        
        # logger.info("🔍 AC_Training性能分析：开始详细计时...")
        # logger.info(f"预期性能瓶颈：梯度积累({gradient_accumulation_steps}x)，性能分析代码，内存监控")
        # logger.info("🔍 JIT编译分析：将详细监控第一次编译时间")
        # enable_perf_analysis = getattr(self.rl_config, 'enable_perf_analysis', False)

        while self.current_step < target_steps:
            # # 🎯 AC_Training性能分析：总迭代时间开始
            # ac_iter_start_time = time.perf_counter()
            step_start_time = time.time()  # 保持原有计时
            
            # Debug: Log every 100 steps to show progress (reduced frequency)
            if self.current_step % 100 == 0:
                logger.debug(f"Training step {self.current_step}/{target_steps} - Sampling batch...")
            
            
            # === 🚀 高效梯度累积逻辑（预编译JIT）===
            if gradient_accumulation_steps > 1:
                # 高效梯度累积模式：分离计算和更新，避免重复编译
                accumulated_gradients = None
                total_loss_info = None
                accumulated_aux_info = []
                
                # 复用预创建的训练配置（避免重复创建，提高JIT缓存命中率）
                if not hasattr(self, '_cached_training_config'):
                    self._cached_training_config = {
                        'critic_weight': getattr(self.rl_config.acrlpd, 'critic_weight', 1.0),
                        'actor_weight': getattr(self.rl_config.acrlpd, 'actor_weight', 1.0),
                        'bc_loss_weight': self.rl_config.acrlpd.bc_loss_weight,
                        'alpha_weight': getattr(self.rl_config.acrlpd, 'alpha_weight', 1.0),
                        'freeze_pi0_backbone': getattr(self.rl_config, 'freeze_pi0_backbone', False),
                        'target_update_tau': getattr(self.rl_config.acrlpd, 'target_update_tau', 0.005),
                        'horizon_length': self.rl_config.qchunking.horizon_length,
                        'discount': getattr(self.rl_config.acrlpd, 'discount', 0.99),
                        'q_aggregation': getattr(self.rl_config.acrlpd, 'q_aggregation', 'min'),
                        'real_action_dim': self.rl_config.qchunking.action_dim  # 从QChunkingConfig传递真实动作维度
                    }
                training_config = self._cached_training_config
                
                # 🎯 性能分析：初始化计时器
                #perf_timings = {}  # 总是初始化避免作用域问题
                #if enable_perf_analysis:
                #step_start_time = time.time()
                
                # 🎯 AC_Training性能分析：梯度积累循环开始
                #grad_accum_start_time = time.perf_counter()
                # existing_perf_start_time = time.perf_counter()
                
                # 梯度积累循环：只计算梯度，不更新参数
                for accumulation_step in range(gradient_accumulation_steps):
                    # 🎯 AC_Training性能分析：数据加载
                    #ac_data_load_start = time.perf_counter()
                    
                    # Sample training batch
                    '''if enable_perf_analysis:
                        data_load_start = time.time()'''
                    self.rng, batch_rng = jax.random.split(self.rng)
                    batch = self.dataloader.sample_batch()
                    '''if enable_perf_analysis:
                        data_load_time = time.time() - data_load_start'''
                    
                    #ac_data_load_time = time.perf_counter() - ac_data_load_start
                    
                    # Extract positive/negative sample counts for logging
                    if 'positive_samples' in batch and 'negative_samples' in batch:
                        self._last_batch_pos_samples = int(batch['positive_samples'])
                        self._last_batch_neg_samples = int(batch['negative_samples'])
                    
                    # 🔍 调试：检查批次数据结构（只在debug模式下输出）
                    if logger.isEnabledFor(logging.DEBUG) and self.current_step < 3 and accumulation_step == 0:
                        logger.debug(f"=== 批次数据调试 (步骤 {self.current_step}, 累积步骤 {accumulation_step}) ===")
                        batch_keys = list(batch.keys()) if isinstance(batch, dict) else "Not a dict"
                        logger.debug(f"批次键: {batch_keys}")
                        
                        # 检查关键的RL字段
                        rl_required_keys = ['next_observations', 'rewards', 'masks']
                        missing_keys = [key for key in rl_required_keys if key not in batch]
                        if missing_keys:
                            logger.debug(f"❌ 缺少RL必需字段: {missing_keys}")
                        else:
                            logger.debug(f"✅ RL字段完整: {rl_required_keys}")
                            # 打印字段形状
                            for key in rl_required_keys:
                                if hasattr(batch[key], 'shape'):
                                    logger.debug(f"  {key}: {batch[key].shape}")
                                else:
                                    logger.debug(f"  {key}: {type(batch[key])}")
                        
                        # 检查基础字段
                        basic_keys = ['observation', 'actions', 'action']
                        for key in basic_keys:
                            if key in batch:
                                if hasattr(batch[key], 'shape'):
                                    logger.debug(f"  {key}: {batch[key].shape}")
                                elif isinstance(batch[key], dict):
                                    logger.debug(f"  {key}: dict with keys {list(batch[key].keys())}")
                        logger.debug("=== 批次数据调试结束 ===")
                    
                    self.rng, train_rng = jax.random.split(self.rng)
                    
                    if self.use_fsdp:
                        # 🎯 AC_Training性能分析：配置转换
                        #ac_config_convert_start = time.perf_counter()
                        
                        # 🚀 使用预编译的JIT梯度计算（快速！）
                        # 转换dict config为可哈希的frozen dataclass
                        #if enable_perf_analysis:
                        #    config_convert_start = time.time()
                        from training.acrlpd_train_state import ACRLPDJITConfig
                        jit_config = ACRLPDJITConfig.from_dict(training_config)
                        #if enable_perf_analysis:
                        #    config_convert_time = time.time() - config_convert_start
                        
                        #ac_config_convert_time = time.perf_counter() - ac_config_convert_start
                        
                        # 🎯 AC_Training性能分析：梯度计算
                        #ac_grad_compute_start = time.perf_counter()
                        
                        # 🔍 JIT编译分析：监控第一次梯度计算编译
                        # if jit_compilation_stats['first_grad_compute_compilation'] is None:
                        #     logger.info(f"🔍 开始第一次JIT编译 - 梯度计算函数 (步骤 {self.current_step}, 积累 {accumulation_step})")
                        #     jit_first_compile_start = time.perf_counter()
                        
                        # 🎯 梯度计算性能分析
                        #if enable_perf_analysis:
                        #    grad_compute_start = time.time()
                        step_gradients, step_loss_info, step_aux_info = self._jit_compute_gradients(
                            self.fsdp_train_state, batch, train_rng, jit_config
                        )
                        
                        # 🔍 JIT编译分析：记录第一次编译时间
                        #if jit_compilation_stats['first_grad_compute_compilation'] is None:
                        #     # JAX同步以确保编译完成
                        #    jax.block_until_ready((step_gradients, step_loss_info, step_aux_info))
                        #     jit_first_compile_time = time.perf_counter() - jit_first_compile_start
                        #     jit_compilation_stats['first_grad_compute_compilation'] = jit_first_compile_time
                        #     logger.info(f"✅ 第一次JIT编译完成 - 梯度计算: {jit_first_compile_time:.2f}秒")
                        #else:
                        #     # 非第一次调用，正常同步
                        jax.block_until_ready((step_gradients, step_loss_info, step_aux_info))
                        
                        #ac_grad_compute_time = time.perf_counter() - ac_grad_compute_start
                        
                        #if enable_perf_analysis:
                        #    grad_compute_time = time.time() - grad_compute_start
                        
                        # 🎯 AC_Training性能分析：梯度处理
                        #ac_grad_process_start = time.perf_counter()
                        
                        # 🎯 梯度处理性能分析
                        #if enable_perf_analysis:
                        #    grad_process_start = time.time()
                        # 缩放梯度（梯度积累的关键步骤）
                        scaled_gradients = jax.tree.map(
                            lambda g: g / gradient_accumulation_steps if g is not None else None,
                            step_gradients
                        )
                        
                        # 积累梯度
                        if accumulated_gradients is None:
                            accumulated_gradients = scaled_gradients
                        else:
                            accumulated_gradients = jax.tree.map(
                                lambda acc, new: acc + new if (acc is not None and new is not None) 
                                                  else (acc if new is None else new),
                                accumulated_gradients, 
                                scaled_gradients
                            )
                        
                        # JAX同步确保梯度处理完成
                        jax.block_until_ready(accumulated_gradients)
                        #ac_grad_process_time = time.perf_counter() - ac_grad_process_start
                        
                        #if enable_perf_analysis:
                        #    grad_process_time = time.time() - grad_process_start
                        
                        # 积累loss信息（用于监控）
                        if total_loss_info is None:
                            total_loss_info = step_loss_info
                        else:
                            # 累积loss信息（保持数值稳定性）
                            total_loss_info = {
                                key: (total_loss_info[key] * accumulation_step + step_loss_info[key]) / (accumulation_step + 1)
                                for key in total_loss_info.keys()
                            }
                        
                        accumulated_aux_info.append(step_aux_info)
                        
                        # 🎯 AC_Training性能分析：收集本次积累步骤的时间统计
                        '''if accumulation_step == 0:
                            # 初始化累积时间统计
                            total_data_load_time = ac_data_load_time
                            total_config_convert_time = ac_config_convert_time
                            total_grad_compute_time = ac_grad_compute_time
                            total_grad_process_time = ac_grad_process_time
                        else:
                            # 累积时间统计
                            total_data_load_time += ac_data_load_time
                            total_config_convert_time += ac_config_convert_time
                            total_grad_compute_time += ac_grad_compute_time
                            total_grad_process_time += ac_grad_process_time
                        
                        # 🎯 累积计时信息（保持原有逻辑）
                        if enable_perf_analysis:
                            if accumulation_step == 0:
                                perf_timings['data_loading'] = data_load_time
                                perf_timings['config_convert'] = config_convert_time
                                perf_timings['grad_compute'] = grad_compute_time  
                                perf_timings['grad_process'] = grad_process_time
                            else:
                                perf_timings['data_loading'] += data_load_time
                                perf_timings['config_convert'] += config_convert_time
                                perf_timings['grad_compute'] += grad_compute_time
                                perf_timings['grad_process'] += grad_process_time'''
                    
                    else:
                        # 非FSDP模式的兼容实现
                        step_loss_info = self.agent.train_step(batch, train_rng)
                        if total_loss_info is None:
                            total_loss_info = step_loss_info._asdict()
                        else:
                            # 累积loss信息
                            step_dict = step_loss_info._asdict()
                            total_loss_info = {
                                key: (total_loss_info[key] * accumulation_step + step_dict[key]) / (accumulation_step + 1)
                                for key in total_loss_info.keys()
                            }
                
                # 🎯 AC_Training性能分析：梯度积累循环总时间
                #total_grad_accum_time = time.perf_counter() - grad_accum_start_time
                
                # 应用积累的梯度（一次参数更新）
                if self.use_fsdp and accumulated_gradients is not None:
                    # 🎯 AC_Training性能分析：参数更新
                    #ac_param_update_start = time.perf_counter()
                    
                    # 🔍 JIT编译分析：监控第一次参数更新编译
                    if jit_compilation_stats['first_grad_apply_compilation'] is None:
                        logger.info(f"🔍 开始第一次JIT编译 - 参数更新函数 (步骤 {self.current_step})")
                        jit_apply_compile_start = time.perf_counter()
                    
                    # 🎯 参数更新性能分析
                    '''if enable_perf_analysis:
                        param_update_start = time.time()'''
                    # 🚀 使用预编译的JIT梯度应用（快速！）
                    # 转换dict config为可哈希的frozen dataclass
                    from training.acrlpd_train_state import ACRLPDJITConfig
                    jit_config = ACRLPDJITConfig.from_dict(training_config)
                    
                    self.fsdp_train_state = self._jit_apply_gradients(
                        self.fsdp_train_state, accumulated_gradients, jit_config
                    )
                    
                    # 🔍 JIT编译分析：记录第一次参数更新编译时间
                    if jit_compilation_stats['first_grad_apply_compilation'] is None:
                        # JAX同步确保参数更新和编译完成
                        jax.block_until_ready(self.fsdp_train_state)
                        jit_apply_compile_time = time.perf_counter() - jit_apply_compile_start
                        jit_compilation_stats['first_grad_apply_compilation'] = jit_apply_compile_time
                        logger.info(f"✅ 第一次JIT编译完成 - 参数更新: {jit_apply_compile_time:.2f}秒")
                    else:
                        # JAX同步确保参数更新完成
                        jax.block_until_ready(self.fsdp_train_state)
                    
                    ac_param_update_time = time.perf_counter() - ac_param_update_start
                    
                    # Update current step from FSDP train state
                    self.current_step = int(self.fsdp_train_state.step)
                    '''if enable_perf_analysis:
                        param_update_time = time.time() - param_update_start
                        perf_timings['param_update'] = param_update_time'''
                
                # 转换loss info为标准格式
                from agents.loss_functions import LossInfo
                loss_info = LossInfo(
                    total_loss=total_loss_info.get('total_loss', 0.0),
                    critic_loss=total_loss_info.get('critic_loss', 0.0),
                    actor_loss=total_loss_info.get('actor_loss', 0.0),
                    bc_loss=total_loss_info.get('bc_loss', 0.0),
                    alpha_loss=total_loss_info.get('alpha_loss', 0.0),
                    q_mean=total_loss_info.get('q_mean', 0.0),
                    q_std=total_loss_info.get('q_std', 0.0),
                    target_q_mean=total_loss_info.get('target_q_mean', 0.0),
                    td_error_mean=total_loss_info.get('td_error_mean', 0.0),
                    bc_loss_raw=total_loss_info.get('bc_loss_raw', 0.0),
                    alpha_value=total_loss_info.get('alpha_value', 0.0),
                    entropy_estimate=total_loss_info.get('entropy_estimate', 0.0),
                    q_values_for_actor=total_loss_info.get('q_values_for_actor', 0.0),
                    valid_samples=total_loss_info.get('valid_samples', 0.0),
                    mask_ratio=total_loss_info.get('mask_ratio', 1.0)
                )
                
                # Extract positive/negative sample counts from accumulated loss_info (梯度积累模式)
                self._last_batch_pos_samples = int(float(total_loss_info.get('bc_positive_samples', 0.0)))
                total_samples = int(float(total_loss_info.get('bc_total_samples', 0.0)))
                self._last_batch_neg_samples = total_samples - self._last_batch_pos_samples
                
            else:
                # 🚀 OPTIMIZED: Streamlined training step (非梯度积累)
                # 🎯 AC_Training性能分析：数据加载
                #ac_data_load_start = time.perf_counter()
                
                #if enable_perf_analysis:
                #    data_load_start = time.time()
                batch = self.dataloader.sample_batch()
                #if enable_perf_analysis:
                #    data_load_time = time.time() - data_load_start
                
                #ac_data_load_time = time.perf_counter() - ac_data_load_start
                
                # Extract positive/negative sample counts for logging (from loss_info instead of batch)
                # Note: In non-gradient accumulation mode, sample info comes from loss computation output
                self._last_batch_pos_samples = 0  # Will be updated from loss_info_dict after training step
                self._last_batch_neg_samples = 0  # Will be updated from loss_info_dict after training step
                
                self.rng, train_rng = jax.random.split(self.rng)
                
                # 🎯 性能分析：初始化计时器（确保在所有分支中都定义）
                # existing_perf_start_time = time.perf_counter()
                perf_timings = {}  # 初始化性能计时字典
                
                if self.use_fsdp:
                    # 🔍 JIT编译分析：监控第一次直接训练步编译
                    '''if jit_compilation_stats.get('first_direct_train_compilation') is None:
                        logger.info(f"🔍 开始第一次JIT编译 - 直接训练步函数 (步骤 {self.current_step})")
                        jit_direct_compile_start = time.perf_counter()'''
                    
                    # 🎯 直接训练步性能分析
                    '''if enable_perf_analysis:
                        direct_train_start = time.time()'''
                    
                    # Direct FSDP training with optimized acrlpd_train_step
                    self.fsdp_train_state, loss_info_dict = self.fsdp_train_step(
                        self.fsdp_train_state, batch, train_rng
                    )
                    
                    # Extract positive/negative sample counts from loss_info_dict (FIXED)
                    self._last_batch_pos_samples = int(float(loss_info_dict.get('bc_positive_samples', 0.0)))
                    total_samples = int(float(loss_info_dict.get('bc_total_samples', 0.0)))
                    self._last_batch_neg_samples = total_samples - self._last_batch_pos_samples
                    
                    # 🔍 JIT编译分析：记录第一次直接训练步编译时间
                    '''if jit_compilation_stats.get('first_direct_train_compilation') is None:
                        # JAX同步确保编译完成
                        jax.block_until_ready((self.fsdp_train_state, loss_info_dict))
                        jit_direct_compile_time = time.perf_counter() - jit_direct_compile_start
                        jit_compilation_stats['first_direct_train_compilation'] = jit_direct_compile_time
                        logger.info(f"✅ 第一次JIT编译完成 - 直接训练步: {jit_direct_compile_time:.2f}秒")
                        
                        # 更新编译完成状态检查逻辑
                        if not jit_compilation_stats.get('compilation_completed', False):
                            jit_compilation_stats['compilation_completed'] = True
                            logger.info("=" * 60)
                            logger.info("🔍 AC_Training JIT编译完成 - 直接训练模式")
                            logger.info("=" * 60)
                            logger.info(f"直接训练步函数编译: {jit_direct_compile_time:.2f}秒")
                            logger.info("JIT编译完成！后续训练步骤将使用已编译的优化代码。")
                            logger.info("=" * 60)
                    else:'''
                        # 非第一次调用，正常同步
                    jax.block_until_ready((self.fsdp_train_state, loss_info_dict))
                    
                    #if enable_perf_analysis:
                    #    direct_train_time = time.time() - direct_train_start
                    #    perf_timings['direct_train'] = direct_train_time
                    
                    # 🎯 AC_Training性能分析：损失信息转换
                    #ac_loss_convert_start = time.perf_counter()
                    
                    # 🚀 OPTIMIZED: Simplified loss info conversion
                    #if enable_perf_analysis:
                    #    loss_convert_start = time.time()
                    from agents.loss_functions import LossInfo
                    loss_info = LossInfo(**{
                        **{k: loss_info_dict.get(k, 0.0) for k in [
                            'total_loss', 'critic_loss', 'actor_loss', 'bc_loss', 'alpha_loss',
                            'q_mean', 'q_std', 'target_q_mean', 'td_error_mean', 'bc_loss_raw',
                            'alpha_value', 'entropy_estimate', 'q_values_for_actor'
                        ]},
                        'valid_samples': float(loss_info_dict.get('valid_samples', 16.0)),
                        'mask_ratio': loss_info_dict.get('mask_ratio', 1.0)
                    })
                    #if enable_perf_analysis:
                    #    loss_convert_time = time.time() - loss_convert_start
                    #    perf_timings['loss_convert'] = loss_convert_time
                    
                    #ac_loss_convert_time = time.perf_counter() - ac_loss_convert_start
                    
                    # Update current step from FSDP train state
                    self.current_step = int(self.fsdp_train_state.step)
                    
                    # # 🎯 AC_Training性能分析：收集非梯度积累模式的时间统计
                    # # 初始化非梯度积累专用的统计字典
                    # if not hasattr(ac_perf_stats, 'non_grad_accum_data_load'):
                    #     ac_perf_stats['non_grad_accum_data_load'] = []
                    #     ac_perf_stats['non_grad_accum_direct_train'] = []
                    #     ac_perf_stats['non_grad_accum_loss_convert'] = []
                    
                    # ac_perf_stats['non_grad_accum_data_load'].append(ac_data_load_time)
                    # ac_perf_stats['non_grad_accum_loss_convert'].append(ac_loss_convert_time)
                    
                    # 设置性能分析的基本时间统计
                    '''if enable_perf_analysis:
                        perf_timings['data_loading'] = data_load_time
                        perf_timings['config_convert'] = 0.0  # 非梯度积累模式无配置转换
                        perf_timings['grad_compute'] = 0.0   # 非梯度积累模式无分离的梯度计算
                        perf_timings['grad_process'] = 0.0   # 非梯度积累模式无分离的梯度处理
                        perf_timings['param_update'] = direct_train_time  # 直接训练包含了参数更新'''
                    
                else:
                    # 原始方式：直接调用（保持兼容性）
                    #if enable_perf_analysis:
                    #    agent_train_start = time.time()
                    self.agent, loss_info = self.agent.train_step(batch, train_rng)
                    '''if enable_perf_analysis:
                        agent_train_time = time.time() - agent_train_start
                        perf_timings = {
                            'data_loading': data_load_time,
                            'agent_train': agent_train_time,
                            'config_convert': 0.0,
                            'grad_compute': 0.0,
                            'grad_process': 0.0,
                            'param_update': agent_train_time
                        }'''
            
            # 🎯 AC_Training性能分析：收集本次迭代的统计数据
            # ac_total_iter_time = time.perf_counter() - ac_iter_start_time
            
            # # 收集现有性能分析代码的时间开销
            # existing_perf_analysis_time = time.perf_counter() - existing_perf_start_time if enable_perf_analysis else 0
            
            # 添加到统计数组
            # if gradient_accumulation_steps > 1:
            #     # 梯度积累模式的统计
            #     ac_perf_stats['total_iter_times'].append(ac_total_iter_time)
            #    ac_perf_stats['grad_accumulation_times'].append(total_grad_accum_time)
            #    ac_perf_stats['data_loading_times'].append(total_data_load_time)
            #    ac_perf_stats['grad_compute_times'].append(total_grad_compute_time)
            #    ac_perf_stats['grad_process_times'].append(total_grad_process_time)
            #    ac_perf_stats['param_update_times'].append(ac_param_update_time if 'ac_param_update_time' in locals() else 0)
            #    ac_perf_stats['config_convert_times'].append(total_config_convert_time)
            #    ac_perf_stats['existing_perf_analysis_times'].append(existing_perf_analysis_time)
            # else:
            #     # 非梯度积累模式的统计
            #     ac_perf_stats['total_iter_times'].append(ac_total_iter_time)
                
                # 为非梯度积累模式收集对应的统计
                #if 'ac_data_load_time' in locals():
                #    ac_perf_stats['data_loading_times'].append(ac_data_load_time)
                #if 'ac_loss_convert_time' in locals():
                    # 将损失转换时间作为处理时间
                #    ac_perf_stats.setdefault('loss_convert_times', []).append(ac_loss_convert_time)
                
                # 对于非梯度积累模式，直接训练步包含了所有计算
                # direct_train_time = ac_total_iter_time - ac_data_load_time - ac_loss_convert_time if 'ac_data_load_time' in locals() else 0
                #ac_perf_stats.setdefault('direct_train_times', []).append(direct_train_time)
                
                # # 设置占位值以保持统计一致性
                # ac_perf_stats.setdefault('grad_accumulation_times', []).append(0)
                # ac_perf_stats.setdefault('grad_compute_times', []).append(direct_train_time * 0.7)  # 估算大部分时间在计算
                # ac_perf_stats.setdefault('grad_process_times', []).append(0)  # 非梯度积累无单独处理
                # ac_perf_stats.setdefault('param_update_times', []).append(direct_train_time * 0.3)  # 估算参数更新
                # ac_perf_stats.setdefault('config_convert_times', []).append(0)  # 非梯度积累无配置转换
                # ac_perf_stats['existing_perf_analysis_times'].append(existing_perf_analysis_time)
            
            # # 🎯 其他开销计时：GPU监控
            # ac_memory_monitor_start = time.perf_counter()
            # if enable_perf_analysis:
            #     gpu_monitor_start = time.time()
                
            # 🔧 定期监控显存使用和FSDP效果
            '''if self.current_step % 100 == 0:
                try:
                    import subprocess
                    logger.info(f"🔍 训练步骤 {self.current_step} - 显存监控:")
                    result = subprocess.run(['nvidia-smi',
            '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
                        capture_output=True, text=True)
                    if result.returncode == 0:
                        for line in result.stdout.strip().split('\n'):
                            parts = line.split(', ')
                            if len(parts) == 3:
                                gpu_id, used, total = parts
                                usage_pct = (int(used) / int(total)) * 100
                                logger.info(f"  GPU {gpu_id}: {used}MB/{total}MB ({usage_pct:.1f}%)")

                    # 梯度累积效果记录
                    if gradient_accumulation_steps > 1:
                        logger.info(f"🔧 梯度累积效果: 有效batch_size={self.rl_config.batch_size * gradient_accumulation_steps}")
                except Exception as e:
                    logger.warning(f"显存监控失败: {e}")'''
            
            # # 收集内存监控时间
            # ac_memory_monitor_time = time.perf_counter() - ac_memory_monitor_start
            # ac_perf_stats['memory_monitor_times'].append(ac_memory_monitor_time)
            
            # === 🔍 JIT编译完成检查和报告 ===
            '''if not jit_compilation_stats['compilation_completed']:
                all_compiled = (
                    jit_compilation_stats['first_grad_compute_compilation'] is not None and 
                    jit_compilation_stats['first_grad_apply_compilation'] is not None
                )
                
                if all_compiled:
                    jit_compilation_stats['compilation_completed'] = True
                    total_compile_time = (
                        jit_compilation_stats['first_grad_compute_compilation'] + 
                        jit_compilation_stats['first_grad_apply_compilation']
                    )
                    
                    logger.info("=" * 60)
                    logger.info("🔍 AC_Training JIT编译完成 - 详细时间报告")
                    logger.info("=" * 60)
                    logger.info(f"总JIT编译时间: {total_compile_time:.2f}秒")
                    logger.info(f"├─ 梯度计算函数编译: {jit_compilation_stats['first_grad_compute_compilation']:.2f}秒 ({jit_compilation_stats['first_grad_compute_compilation']/total_compile_time*100:.1f}%)")
                    logger.info(f"└─ 参数更新函数编译: {jit_compilation_stats['first_grad_apply_compilation']:.2f}秒 ({jit_compilation_stats['first_grad_apply_compilation']/total_compile_time*100:.1f}%)")
                    logger.info("")
                    logger.info("JIT编译完成！后续训练步骤将使用已编译的优化代码。")
                    logger.info("=" * 60)'''
            
            # === AC_Training性能分析报告（调试已完成，注释掉详细输出）===
            # if (self.current_step % 2 == 0 or self.current_step == start_step) and ac_perf_stats['total_iter_times']:
            #     recent_samples = min(2, len(ac_perf_stats['total_iter_times']))
            #     
            #     avg_total = np.mean(ac_perf_stats['total_iter_times'][-recent_samples:])
            #     avg_grad_accum = np.mean(ac_perf_stats['grad_accumulation_times'][-recent_samples:])
            #     avg_data_load = np.mean(ac_perf_stats['data_loading_times'][-recent_samples:])
            #     avg_grad_compute = np.mean(ac_perf_stats['grad_compute_times'][-recent_samples:])
            #     avg_grad_process = np.mean(ac_perf_stats['grad_process_times'][-recent_samples:])
            #     avg_param_update = np.mean([t for t in ac_perf_stats['param_update_times'][-recent_samples:] if t > 0])
            #     avg_config_convert = np.mean(ac_perf_stats['config_convert_times'][-recent_samples:])
            #     avg_memory_monitor = np.mean(ac_perf_stats['memory_monitor_times'][-recent_samples:])
            #     avg_existing_perf = np.mean(ac_perf_stats['existing_perf_analysis_times'][-recent_samples:])
            #     
            #     # 根据梯度积累模式显示不同的性能分析
            #     if gradient_accumulation_steps > 1:
            #         # 梯度积累模式
            #         logger.info(f"🔍 AC_Training性能分析 (Step {self.current_step}) - 梯度积累模式:")
            #         logger.info(f"  总迭代时间: {avg_total*1000:.2f}ms")
            #         logger.info(f"  ├─ 梯度积累循环: {avg_grad_accum*1000:.2f}ms ({avg_grad_accum/avg_total*100:.1f}%)")
            #         logger.info(f"  │  ├─ 数据加载(x{gradient_accumulation_steps}): {avg_data_load*1000:.2f}ms ({avg_data_load/avg_total*100:.1f}%)")
            #         logger.info(f"  │  ├─ 梯度计算(x{gradient_accumulation_steps}): {avg_grad_compute*1000:.2f}ms ({avg_grad_compute/avg_total*100:.1f}%)")
            #         logger.info(f"  │  ├─ 梯度处理(x{gradient_accumulation_steps}): {avg_grad_process*1000:.2f}ms ({avg_grad_process/avg_total*100:.1f}%)")
            #         logger.info(f"  │  └─ 配置转换(x{gradient_accumulation_steps}): {avg_config_convert*1000:.2f}ms ({avg_config_convert/avg_total*100:.1f}%)")
            #         logger.info(f"  ├─ 参数更新: {avg_param_update*1000:.2f}ms ({avg_param_update/avg_total*100:.1f}%)")
            #         logger.info(f"  ├─ 内存监控: {avg_memory_monitor*1000:.2f}ms ({avg_memory_monitor/avg_total*100:.1f}%)")
            #         logger.info(f"  └─ 现有性能分析: {avg_existing_perf*1000:.2f}ms ({avg_existing_perf/avg_total*100:.1f}%)")
            #         logger.info(f"  每秒迭代数: {1.0/avg_total:.2f} iter/s (有效batch: {self.rl_config.batch_size * gradient_accumulation_steps})")
            #     else:
            #         # 非梯度积累模式 - 直接训练
            #         avg_direct_train = np.mean(ac_perf_stats.get('direct_train_times', [0])[-recent_samples:])
            #         avg_loss_convert = np.mean(ac_perf_stats.get('loss_convert_times', [0])[-recent_samples:])
            #         
            #         logger.info(f"🔍 AC_Training性能分析 (Step {self.current_step}) - 直接训练模式:")
            #         logger.info(f"  总迭代时间: {avg_total*1000:.2f}ms")
            #         logger.info(f"  ├─ 数据加载: {avg_data_load*1000:.2f}ms ({avg_data_load/avg_total*100:.1f}%)")
            #         logger.info(f"  ├─ 直接训练步: {avg_direct_train*1000:.2f}ms ({avg_direct_train/avg_total*100:.1f}%)")
            #         logger.info(f"  │  ├─ 梯度计算+参数更新: {avg_grad_compute*1000:.2f}ms ({avg_grad_compute/avg_total*100:.1f}%)")
            #         logger.info(f"  │  └─ 内部参数更新: {avg_param_update*1000:.2f}ms ({avg_param_update/avg_total*100:.1f}%)")
            #         logger.info(f"  ├─ 损失信息转换: {avg_loss_convert*1000:.2f}ms ({avg_loss_convert/avg_total*100:.1f}%)")
            #         logger.info(f"  ├─ 内存监控: {avg_memory_monitor*1000:.2f}ms ({avg_memory_monitor/avg_total*100:.1f}%)")
            #         logger.info(f"  └─ 现有性能分析: {avg_existing_perf*1000:.2f}ms ({avg_existing_perf/avg_total*100:.1f}%)")
            #         logger.info(f"  每秒迭代数: {1.0/avg_total:.2f} iter/s (batch: {self.rl_config.batch_size})")
            
            '''if enable_perf_analysis and self.current_step % 100 == 0:
                gpu_monitor_time = time.time() - gpu_monitor_start
                if 'gpu_monitor' not in perf_timings:
                    perf_timings['gpu_monitor'] = 0
                perf_timings['gpu_monitor'] += gpu_monitor_time
                
            # 🎯 其他开销计时：EMA更新
            if enable_perf_analysis:
                ema_start = time.time()
                
            # 🚀 EMA更新优化：在FSDP模式下，EMA更新已在gradient application中自动完成
            # 删除多余的EMA更新，避免重建训练状态的16.6s开销
            if enable_perf_analysis:
                ema_time = time.time() - ema_start
                # EMA更新时间现在应该接近0，因为已经在gradient application中处理
                if 'ema_update' not in perf_timings:
                    perf_timings['ema_update'] = 0
                perf_timings['ema_update'] += ema_time'''
                
            # 注意：对于非FSDP模式，EMA更新在agent.train_step中处理
                    
            # 🎯 其他开销计时：指标更新
            '''if enable_perf_analysis:
                metrics_start = time.time()'''
            
            # Update metrics
            step_time = time.time() - step_start_time
            effective_batch_size = self.rl_config.batch_size * gradient_accumulation_steps
            self.metrics.update(
                self.current_step,
                loss_info,
                timing_step_duration=step_time,
                timing_samples_per_sec=effective_batch_size / step_time,
                effective_batch_size=effective_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps
            )
            
            '''if enable_perf_analysis:
                metrics_time = time.time() - metrics_start
                if 'metrics_update' not in perf_timings:
                    perf_timings['metrics_update'] = 0
                perf_timings['metrics_update'] += metrics_time'''
            
            # 详细内存监控（每100步记录一次）
            if self.current_step % 100 == 0 or self.current_step <= 5:
                log_memory_usage(self.current_step, self.agent.create_train_state() if hasattr(self.agent, 'create_train_state') else None, "offline_training")
            
            # 仅在启用epoch机制时进行epoch统计
            if acrlpd_config.enable_epoch_based_lr_schedule:
                steps_in_current_epoch += 1
                epoch_losses.append(float(loss_info.total_loss))
                
                # 使用配置中的steps_per_epoch参数
                steps_per_epoch = getattr(self.rl_config.acrlpd, 'steps_per_epoch', 10000)
                
                if steps_in_current_epoch >= steps_per_epoch:
                    # Epoch结束统计
                    epoch_time = time.time() - epoch_start_time
                    avg_epoch_loss = np.mean(epoch_losses)
                    lr_info = self._get_current_learning_rates()
                    
                    logger.info(f"=== Epoch {current_epoch} Complete ===")
                    logger.info(f"Steps: {steps_in_current_epoch} | Avg Loss: {avg_epoch_loss:.4f} | Time: {epoch_time:.1f}s | Speed: {steps_in_current_epoch/epoch_time:.1f} steps/s")
                    logger.info(f"Learning Rates: Actor={lr_info['actor_lr']:.6f}, Critic={lr_info['critic_lr']:.6f} (Epoch factor: {lr_info['epoch_factor']:.3f})")
                    
                    # 重置epoch计数器
                    current_epoch += 1
                    steps_in_current_epoch = 0
                    epoch_losses = []
                    epoch_start_time = time.time()
            
            # Professional Training Logging
            if self.current_step % self.rl_config.log_interval == 0:
                # Calculate progress and timing
                progress_pct = (self.current_step / target_steps) * 100
                elapsed_time = time.time() - (time.time() - step_time * self.current_step)  # Approximate
                samples_per_sec = effective_batch_size / step_time if step_time > 0 else 0
                
                # Get current learning rates
                lr_info = self._get_current_learning_rates()
                
                # Extract positive/negative sample info from last batch  
                pos_samples = getattr(self, '_last_batch_pos_samples', 0)
                neg_samples = getattr(self, '_last_batch_neg_samples', 0)
                total_samples = pos_samples + neg_samples
                pos_ratio = pos_samples / max(total_samples, 1) * 100
                
                # Main training log with learning rates and sample composition
                logger.info(
                    f"Step {self.current_step:6d}/{target_steps} ({progress_pct:5.1f}%) | "
                    f"Loss: {float(loss_info.total_loss):.4f} | "
                    f"Critic: {float(loss_info.critic_loss):.4f} | "
                    f"Actor: {float(loss_info.actor_loss):.4f} | "
                    f"BC: {float(loss_info.bc_loss):.4f} | "
                    f"Q_mean: {float(loss_info.q_mean):.3f} | "
                    f"Samples: +{pos_samples}/-{neg_samples} ({pos_ratio:.1f}%) | "
                    f"LR_Actor: {lr_info['actor_lr']:.6f} | "
                    f"LR_Critic: {lr_info['critic_lr']:.6f} | "
                    f"Time: {step_time:.3f}s | "
                    f"Speed: {samples_per_sec:.1f} samples/s | "
                )
                
                self._log_metrics()
            
            # Detailed metrics every 1000 steps
            if self.current_step % (self.rl_config.log_interval * 10) == 0:
                lr_info = self._get_current_learning_rates()
                logger.info(f"=== Detailed Metrics at Step {self.current_step} ===")
                logger.info(
                    f"Losses - Total: {float(loss_info.total_loss):.6f} | "
                    f"Critic: {float(loss_info.critic_loss):.6f} | "
                    f"Actor: {float(loss_info.actor_loss):.6f} | "
                    f"BC: {float(loss_info.bc_loss):.6f} | "
                    f"Alpha: {float(loss_info.alpha_loss):.6f}"
                )
                logger.info(
                    f"Q-Values - Mean: {float(loss_info.q_mean):.3f} | "
                    f"Std: {float(loss_info.q_std):.3f} | "
                    f"Target: {float(loss_info.target_q_mean):.3f} | "
                    f"TD_Error: {float(loss_info.td_error_mean):.4f}"
                )
                logger.info(
                    f"Learning Rates - Actor: {lr_info['actor_lr']:.6f} | "
                    f"Critic: {lr_info['critic_lr']:.6f} | "
                    f"Epoch_Factor: {lr_info['epoch_factor']:.4f} | "
                    f"Intra_Epoch_Factor: {lr_info['intra_epoch_factor']:.4f}"
                )
            
            # Evaluation已移除 - 当前无真实环境评估实现
            
            # Checkpointing
            if self.current_step % self.rl_config.save_interval == 0:
                # Sync agent state from FSDP train state before checkpointing
                if self.use_fsdp:
                    self.agent.from_train_state(self.fsdp_train_state)
                
                self.checkpoint_manager.save_checkpoint(
                    self.agent, self.dataloader, self.current_step
                )
            
            # 🎯 完整性能分析输出
            '''if enable_perf_analysis:
                total_step_time = time.time() - step_start_time
                perf_timings['total_step'] = total_step_time
                perf_timings.setdefault('param_update', 0.0)  # 如果没有参数更新
                
                # 计算主要阶段时间
                main_stages_time = (perf_timings['data_loading'] + 
                                   perf_timings['grad_compute'] + 
                                   perf_timings['grad_process'] + 
                                   perf_timings['param_update'])
                
                # 计算其他开销细分
                other_stages_time = 0
                other_stages = ['gpu_monitor', 'ema_update', 'metrics_update']
                for stage in other_stages:
                    other_stages_time += perf_timings.get(stage, 0)
                
                # 未分类的开销
                unaccounted_time = total_step_time - main_stages_time - other_stages_time
                
                # 每5步输出一次详细性能分析
                if self.current_step % 5 == 0:
                    logger.info("=" * 80)
                    logger.info(f"🎯 性能分析 (步骤 {self.current_step}) - 梯度累积: {gradient_accumulation_steps}步")
                    logger.info(f"📊 总耗时: {total_step_time:.3f}s ({1.0/total_step_time:.1f} sample/s)")
                    
                    # 主要阶段
                    logger.info(f"📈 数据加载: {perf_timings['data_loading']:.3f}s ({perf_timings['data_loading']/total_step_time*100:.1f}%)")
                    logger.info(f"📈 配置转换: {perf_timings['config_convert']:.3f}s ({perf_timings['config_convert']/total_step_time*100:.1f}%)")
                    logger.info(f"📈 梯度计算: {perf_timings['grad_compute']:.3f}s ({perf_timings['grad_compute']/total_step_time*100:.1f}%)")
                    logger.info(f"📈 梯度处理: {perf_timings['grad_process']:.3f}s ({perf_timings['grad_process']/total_step_time*100:.1f}%)")
                    logger.info(f"📈 参数更新: {perf_timings['param_update']:.3f}s ({perf_timings['param_update']/total_step_time*100:.1f}%)")
                    
                    # 其他开销细分
                    if other_stages_time > 0:
                        logger.info(f"📈 其他开销细分:")
                        for stage in other_stages:
                            if stage in perf_timings and perf_timings[stage] > 0:
                                logger.info(f"    {stage}: {perf_timings[stage]:.3f}s ({perf_timings[stage]/total_step_time*100:.1f}%)")
                    
                    # 未分类开销
                    if unaccounted_time > 0:
                        logger.info(f"📈 未分类开销: {unaccounted_time:.3f}s ({unaccounted_time/total_step_time*100:.1f}%)")
                    
                    logger.info("=" * 80)'''
            
            self.current_step += 1
        
        # 训练完成统计
        total_training_time = time.time() - (time.time() - (self.current_step - start_step) * 0.5)  # 基于step粗略估算
        logger.info(f"=== Offline Training Complete ===")
        logger.info(f"Total Steps: {self.current_step - start_step}")
        if acrlpd_config.enable_epoch_based_lr_schedule:
            logger.info(f"Total Epochs: {current_epoch}")
        logger.info(f"Total Time: {total_training_time/3600:.2f}h | Avg Speed: {(self.current_step - start_step)/(total_training_time/60):.1f} steps/min")
        
        # # === 最终AC_Training性能报告 ===
        # if ac_perf_stats['total_iter_times']:
        #     logger.info("=" * 60)
        #     logger.info("🔍 AC_Training训练完成 - 最终性能报告")
        #     logger.info("=" * 60)
        #     
        #     # JIT编译时间报告
        #     if jit_compilation_stats.get('compilation_completed', False):
        #         if gradient_accumulation_steps > 1:
        #             # 梯度积累模式的编译统计
        #             total_compile_time = (
        #                 jit_compilation_stats.get('first_grad_compute_compilation', 0) + 
        #                 jit_compilation_stats.get('first_grad_apply_compilation', 0)
        #             )
        #             logger.info("JIT编译时间统计 (梯度积累模式):")
        #             logger.info(f"  总编译时间: {total_compile_time:.2f}秒")
        #             logger.info(f"  梯度计算函数: {jit_compilation_stats.get('first_grad_compute_compilation', 0):.2f}秒")
        #             logger.info(f"  参数更新函数: {jit_compilation_stats.get('first_grad_apply_compilation', 0):.2f}秒")
        #         else:
        #             # 直接训练模式的编译统计
        #             direct_compile_time = jit_compilation_stats.get('first_direct_train_compilation', 0)
        #             logger.info("JIT编译时间统计 (直接训练模式):")
        #             logger.info(f"  直接训练步函数: {direct_compile_time:.2f}秒")
        #         logger.info("")
        #     
        #     total_samples = len(ac_perf_stats['total_iter_times'])
        #     avg_total = np.mean(ac_perf_stats['total_iter_times'])
        #     std_total = np.std(ac_perf_stats['total_iter_times'])
        #     
        #     avg_grad_accum = np.mean(ac_perf_stats['grad_accumulation_times'])
        #     avg_data_load = np.mean(ac_perf_stats['data_loading_times'])
        #     avg_grad_compute = np.mean(ac_perf_stats['grad_compute_times'])
        #     avg_grad_process = np.mean(ac_perf_stats['grad_process_times'])
        #     avg_param_update = np.mean([t for t in ac_perf_stats['param_update_times'] if t > 0])
        #     avg_config_convert = np.mean(ac_perf_stats['config_convert_times'])
        #     avg_memory_monitor = np.mean(ac_perf_stats['memory_monitor_times'])
        #     avg_existing_perf = np.mean(ac_perf_stats['existing_perf_analysis_times'])
        #     
        #     logger.info(f"样本数量: {total_samples} 次迭代")
        #     logger.info(f"平均迭代时间: {avg_total*1000:.2f} ± {std_total*1000:.2f}ms")
        #     logger.info(f"平均每秒迭代数: {1.0/avg_total:.2f} iter/s")
        #     
        #     # 根据模式显示不同的批次信息
        #     if gradient_accumulation_steps > 1:
        #         logger.info(f"有效批次大小: {self.rl_config.batch_size * gradient_accumulation_steps} (梯度积累模式)")
        #     else:
        #         logger.info(f"批次大小: {self.rl_config.batch_size} (直接训练模式)")
        #     logger.info("")
        
        """
        # Performance analysis section commented out
        # # # 根据模式显示不同的时间分解
        # # if gradient_accumulation_steps > 1:
        #     logger.info("时间分解 (梯度积累模式):")
                logger.info(f"  梯度积累循环: {avg_grad_accum*1000:.2f}ms ({avg_grad_accum/avg_total*100:.1f}%)")
                logger.info(f"    ├─ 数据加载(x{gradient_accumulation_steps}): {avg_data_load*1000:.2f}ms ({avg_data_load/avg_total*100:.1f}%)")
                logger.info(f"    ├─ 梯度计算(x{gradient_accumulation_steps}): {avg_grad_compute*1000:.2f}ms ({avg_grad_compute/avg_total*100:.1f}%)")
                logger.info(f"    ├─ 梯度处理(x{gradient_accumulation_steps}): {avg_grad_process*1000:.2f}ms ({avg_grad_process/avg_total*100:.1f}%)")
                logger.info(f"    └─ 配置转换(x{gradient_accumulation_steps}): {avg_config_convert*1000:.2f}ms ({avg_config_convert/avg_total*100:.1f}%)")
                logger.info(f"  参数更新: {avg_param_update*1000:.2f}ms ({avg_param_update/avg_total*100:.1f}%)")
                logger.info(f"  内存监控: {avg_memory_monitor*1000:.2f}ms ({avg_memory_monitor/avg_total*100:.1f}%)")
                logger.info(f"  现有性能分析: {avg_existing_perf*1000:.2f}ms ({avg_existing_perf/avg_total*100:.1f}%)")
                
                logger.info("")
                logger.info("性能瓶颈分析:")
                bottleneck_info = [
                    ("梯度积累循环", avg_grad_accum, avg_grad_accum/avg_total*100),
                    ("数据加载", avg_data_load, avg_data_load/avg_total*100),
                    ("梯度计算", avg_grad_compute, avg_grad_compute/avg_total*100),
                    ("梯度处理", avg_grad_process, avg_grad_process/avg_total*100),
                    ("参数更新", avg_param_update, avg_param_update/avg_total*100),
                    ("内存监控", avg_memory_monitor, avg_memory_monitor/avg_total*100),
                ]
            else:
                # 非梯度积累模式的统计
                avg_direct_train = np.mean(ac_perf_stats.get('direct_train_times', [0]))
                avg_loss_convert = np.mean(ac_perf_stats.get('loss_convert_times', [0]))
                
                logger.info("时间分解 (直接训练模式):")
                logger.info(f"  数据加载: {avg_data_load*1000:.2f}ms ({avg_data_load/avg_total*100:.1f}%)")
                logger.info(f"  直接训练步: {avg_direct_train*1000:.2f}ms ({avg_direct_train/avg_total*100:.1f}%)")
                logger.info(f"    ├─ 梯度计算+参数更新: {avg_grad_compute*1000:.2f}ms ({avg_grad_compute/avg_total*100:.1f}%)")
                logger.info(f"    └─ 内部参数更新: {avg_param_update*1000:.2f}ms ({avg_param_update/avg_total*100:.1f}%)")
                logger.info(f"  损失信息转换: {avg_loss_convert*1000:.2f}ms ({avg_loss_convert/avg_total*100:.1f}%)")
                logger.info(f"  内存监控: {avg_memory_monitor*1000:.2f}ms ({avg_memory_monitor/avg_total*100:.1f}%)")
                logger.info(f"  现有性能分析: {avg_existing_perf*1000:.2f}ms ({avg_existing_perf/avg_total*100:.1f}%)")
                
                logger.info("")
                logger.info("性能瓶颈分析:")
                bottleneck_info = [
                    ("直接训练步", avg_direct_train, avg_direct_train/avg_total*100),
                    ("数据加载", avg_data_load, avg_data_load/avg_total*100),
                    ("梯度计算+参数更新", avg_grad_compute, avg_grad_compute/avg_total*100),
                    ("损失信息转换", avg_loss_convert, avg_loss_convert/avg_total*100),
                    ("内存监控", avg_memory_monitor, avg_memory_monitor/avg_total*100),
                ]
            
            bottleneck_info.sort(key=lambda x: x[2], reverse=True)
            for i, (name, time_val, pct) in enumerate(bottleneck_info[:3]):
                logger.info(f"  {i+1}. {name}: {time_val*1000:.2f}ms ({pct:.1f}%)")
            logger.info("=" * 60)
        """
        
        return self.agent
    
    def _create_jit_compute_gradients(self):
        """创建JIT编译的梯度计算函数。"""
        from training.acrlpd_train_state import acrlpd_compute_gradients
        import openpi.training.sharding as sharding
        
        jit_compute_gradients = jax.jit(
            acrlpd_compute_gradients, 
            static_argnames=['config']
        )
        
        logger.debug("✅ JIT梯度计算函数编译完成")
        return jit_compute_gradients
    
    def _create_jit_apply_gradients(self):
        """创建JIT编译的梯度应用函数。"""
        from training.acrlpd_train_state import acrlpd_apply_gradients
        import openpi.training.sharding as sharding
        
        jit_apply_gradients = jax.jit(
            acrlpd_apply_gradients,
            static_argnames=['config']
        )
        
        logger.debug("✅ JIT梯度应用函数编译完成")
        return jit_apply_gradients
    
    # _run_evaluation方法已移除 - 当前无真实环境评估实现
    def _log_metrics(self):
        """Log current metrics."""
        if self.rl_config.wandb_enabled:
            wandb.log(self.metrics.metrics, step=self.current_step)
        
        # Console logging
        if self.current_step % (self.rl_config.log_interval * 10) == 0:
            logger.info(f"Step {self.current_step}: "
                       f"Loss={self.metrics.metrics.get('train/total_loss', 0):.4f}, "
                       f"Q_mean={self.metrics.metrics.get('train/q_mean', 0):.3f}")
    
    def _final_evaluation(self):
        """Run final comprehensive evaluation."""
        logger.info("Running final evaluation")
        self._run_evaluation()
        
        # Sync agent state from FSDP train state before final checkpoint
        if self.use_fsdp:
            self.agent.from_train_state(self.fsdp_train_state)
        
        # Save final checkpoint
        final_checkpoint = self.checkpoint_manager.save_checkpoint(
            self.agent, self.dataloader, self.current_step
        )
        
        logger.info(f"Final checkpoint saved: {final_checkpoint}")
        
        if self.rl_config.wandb_enabled:
            wandb.finish()
    
    def _resume_training(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        logger.info(f"🔄 Resuming training from {checkpoint_path}")
        try:
            # 🔍 修复参数传递：从checkpoint_path中提取step
            from pathlib import Path
            checkpoint_dir = Path(checkpoint_path)
            if checkpoint_dir.name.isdigit():
                step_to_load = int(checkpoint_dir.name)
            else:
                step_to_load = None
            
            # Load checkpoint using the checkpoint manager
            self.agent, self.current_step, saved_metrics = self.checkpoint_manager.load_checkpoint(
                self.agent, self.dataloader, step=step_to_load
            )
            
            # 🔍 更新FSDP train state
            if self.use_fsdp and hasattr(self, 'fsdp_train_state'):
                # 从恢复的agent同步FSDP state
                logger.info("🔄 Synchronizing FSDP train state from resumed agent")
                # 重要修复：使用正确的ACRLPD FSDP初始化函数
                # 不调用agent.create_train_state()因为它返回OpenPI TrainState而非ACRLPDTrainState
                from training.acrlpd_train_state import init_acrlpd_fsdp_training
                self.fsdp_train_state, self.train_state_sharding, self.jit_train_step_fn = init_acrlpd_fsdp_training(
                    rl_config=self.rl_config,
                    mesh=self.mesh,
                    rng=self.rng,
                    data_sharding=self.data_sharding,
                    step=self.current_step,
                    global_pi0_tx=self.global_pi0_tx,
                    global_critic_tx=self.global_critic_tx
                )
                logger.info("✅ FSDP train state synchronized successfully")
            
            # Update metrics with saved values
            if saved_metrics:
                self.metrics.metrics.update(saved_metrics)
                logger.info("✅ Restored training metrics")
            else:
                logger.info("ℹ️ No metrics found, starting fresh metrics tracking")
            
            logger.info(f"✅ Successfully resumed training from step {self.current_step}")
            
        except Exception as e:
            logger.error(f"❌ Failed to resume training from {checkpoint_path}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise


# create_simple_eval_fn函数已移除 - 虚假eval实现已删除