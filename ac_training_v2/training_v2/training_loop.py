"""
ACRLPD Training Loop

基于OpenPI模式的简化训练循环，优化点：
- 移除冗余的性能监控代码
- 统一的JIT编译训练步骤  
- 最小化的日志记录开销
- 高效的checkpoint管理

保持完整的RL算法功能的同时大幅提升训练效率
"""

import logging
import time
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Iterator
import jax
import jax.numpy as jnp
import numpy as np
import wandb

# AC Training v2 imports - fix relative import
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents_v2.acrlpd_pi0_agent import ACRLPDPi0Agent
from agents_v2.loss_functions import UnifiedLossComputer, create_loss_computer, create_loss_config_from_rl_config
from training_v2.train_state import (
    ACRLPDTrainState, init_train_state, unified_train_step_jit, 
    save_checkpoint, load_checkpoint, get_latest_checkpoint_path, TrainingMetrics
)
from training_v2.optimizer import create_pi0_optimizer, create_critic_optimizer

logger = logging.getLogger(__name__)


@dataclass
class ACRLPDTrainingConfig:
    """ACRLPD训练配置"""
    # 基础训练配置
    max_steps: int = 100000
    batch_size: int = 128
    
    # 日志和保存配置
    log_interval: int = 100         # 日志记录间隔（简化）
    eval_interval: int = 1000       # 评估间隔  
    save_interval: int = 5000       # checkpoint保存间隔
    
    # Checkpoint配置
    checkpoint_dir: str = "./checkpoints"
    resume_from_checkpoint: bool = True
    max_checkpoints_to_keep: int = 5
    
    # 优化器配置
    pi0_lr: float = 1e-5           # π₀学习率（小模型需要小lr）
    critic_lr: float = 1e-3        # Critic学习率
    warmup_steps: int = 1000
    
    # wandb配置
    use_wandb: bool = True
    wandb_project: str = "acrlpd_v2"
    wandb_run_name: Optional[str] = None
    
    # 评估配置
    eval_episodes: int = 10
    
    # 其他配置
    seed: int = 42
    debug_mode: bool = False


class ACRLPDTrainer:
    """
    ACRLPD主训练器
    
    整合所有组件，提供简化高效的训练循环
    """
    
    def __init__(self,
                 agent: ACRLPDPi0Agent,
                 data_loader: Any,  # ACRLPDDataLoaderV2
                 config: ACRLPDTrainingConfig):
        """
        初始化训练器
        
        Args:
            agent: ACRLPD Agent
            data_loader: 数据加载器
            config: 训练配置
        """
        self.agent = agent
        self.data_loader = data_loader
        self.config = config
        
        logger.info("=== ACRLPDTrainer 初始化 ===")
        logger.info(f"最大步数: {config.max_steps}")
        logger.info(f"批次大小: {config.batch_size}")
        logger.info(f"π₀学习率: {config.pi0_lr}")
        logger.info(f"Critic学习率: {config.critic_lr}")
        
        # 1. 创建损失计算器
        # 这里需要从agent的rl_config创建loss_config，暂时使用默认配置
        self.loss_computer = create_loss_computer(agent)
        
        # 2. 创建优化器
        self.pi0_optimizer, self.critic_optimizer = self._create_optimizers()
        
        # 3. 初始化训练状态  
        self.train_state = self._init_training_state()
        
        # 4. 初始化wandb（如果启用）
        if config.use_wandb:
            self._init_wandb()
        
        # 5. 性能统计
        self.start_time = time.time()
        self.step_times = []
        
        logger.info("✅ ACRLPDTrainer初始化完成")
    
    def _create_optimizers(self):
        """创建优化器"""
        logger.info("创建优化器...")
        
        # 创建专用优化器
        pi0_optimizer = create_pi0_optimizer(
            pi0_lr=self.config.pi0_lr,
            total_steps=self.config.max_steps,
            warmup_steps=self.config.warmup_steps
        )
        
        critic_optimizer = create_critic_optimizer(
            critic_lr=self.config.critic_lr,
            total_steps=self.config.max_steps,
            warmup_steps=self.config.warmup_steps
        )
        
        logger.info("✅ 优化器创建完成")
        return pi0_optimizer, critic_optimizer
    
    def _init_training_state(self) -> ACRLPDTrainState:
        """初始化训练状态"""
        logger.info("初始化训练状态...")
        
        # 尝试从checkpoint恢复
        if self.config.resume_from_checkpoint:
            latest_checkpoint = get_latest_checkpoint_path(self.config.checkpoint_dir)
            if latest_checkpoint:
                logger.info(f"从checkpoint恢复: {latest_checkpoint}")
                train_state, metadata = load_checkpoint(latest_checkpoint)
                logger.info(f"恢复到步数: {train_state.step}")
                return train_state
        
        # 创建新的训练状态
        rng = jax.random.PRNGKey(self.config.seed)
        train_state = init_train_state(
            agent=self.agent,
            pi0_optimizer=self.pi0_optimizer,
            critic_optimizer=self.critic_optimizer,
            rng=rng,
            config=self.config.__dict__
        )
        
        logger.info("✅ 训练状态初始化完成")
        return train_state
    
    def _init_wandb(self):
        """初始化wandb"""
        wandb_config = {
            **self.config.__dict__,
            **self.agent.get_model_config()
        }
        
        wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            config=wandb_config,
            resume="allow"
        )
        
        logger.info("✅ wandb初始化完成")
    
    def train(self):
        """
        主训练循环 - 基于OpenPI的简化模式
        
        核心优化：
        - 移除冗余的性能监控
        - 统一的JIT编译训练步骤
        - 最小化日志开销
        - 批量I/O操作
        """
        
        logger.info("=== 开始ACRLPD训练 ===")
        logger.info(f"起始步数: {self.train_state.step}")
        logger.info(f"目标步数: {self.config.max_steps}")
        
        start_step = self.train_state.step
        training_start_time = time.time()
        
        # 主训练循环
        for step in range(start_step, self.config.max_steps):
            step_start_time = time.time()
            
            # 1. 采样训练batch
            batch = self.data_loader.sample_batch()
            
            # 2. 执行统一训练步骤（完全JIT编译）
            self.train_state, metrics = unified_train_step_jit(
                train_state=self.train_state,
                batch=batch,
                agent=self.agent,
                loss_computer=self.loss_computer,
                pi0_optimizer=self.pi0_optimizer,
                critic_optimizer=self.critic_optimizer
            )
            
            # 3. 记录步骤时间（最小化开销）
            step_time = time.time() - step_start_time
            self.step_times.append(step_time)
            
            # 4. 简化的日志记录
            if step % self.config.log_interval == 0:
                self._log_training_metrics(metrics, step, step_time)
            
            # 5. 保存checkpoint
            if step % self.config.save_interval == 0:
                self._save_checkpoint(step)
            
            # 6. 评估（可选）
            if step % self.config.eval_interval == 0:
                self._evaluate(step)
        
        # 训练完成
        total_time = time.time() - training_start_time
        avg_time_per_step = np.mean(self.step_times[-1000:])  # 最后1000步的平均时间
        
        logger.info("=== ACRLPD训练完成 ===")
        logger.info(f"总训练时间: {total_time:.2f}秒")
        logger.info(f"平均步骤时间: {avg_time_per_step:.4f}秒/步")
        logger.info(f"训练效率: {1.0/avg_time_per_step:.2f} 步/秒")
        
        # 保存最终checkpoint
        self._save_checkpoint(self.config.max_steps, final=True)
    
    def _log_training_metrics(self, metrics: TrainingMetrics, step: int, step_time: float):
        """
        记录训练指标 - 最小化版本
        
        只记录核心指标，避免过度的I/O开销
        """
        
        # 计算平均步骤时间
        recent_step_times = self.step_times[-self.config.log_interval:]
        avg_step_time = np.mean(recent_step_times)
        steps_per_sec = 1.0 / avg_step_time
        
        # 核心指标日志
        log_dict = {
            'step': step,
            'total_loss': float(metrics.total_loss),
            'critic_loss': float(metrics.critic_loss), 
            'actor_loss': float(metrics.actor_loss),
            'bc_loss': float(metrics.bc_loss),
            'critic_q_mean': float(metrics.critic_q_mean),
            'actor_q_best': float(metrics.actor_q_best),
            'steps_per_sec': steps_per_sec,
            'step_time': step_time
        }
        
        # 控制台输出（简化）
        logger.info(
            f"Step {step}: "
            f"Loss={log_dict['total_loss']:.4f} "
            f"(C={log_dict['critic_loss']:.4f}, "
            f"A={log_dict['actor_loss']:.4f}, "
            f"BC={log_dict['bc_loss']:.4f}) "
            f"Q={log_dict['critic_q_mean']:.3f} "
            f"Speed={steps_per_sec:.2f}it/s"
        )
        
        # wandb日志（如果启用）
        if self.config.use_wandb:
            wandb.log(log_dict, step=step)
    
    def _save_checkpoint(self, step: int, final: bool = False):
        """保存checkpoint"""
        checkpoint_name = f"checkpoint_{step}.pkl"
        if final:
            checkpoint_name = f"final_checkpoint_{step}.pkl"
        
        checkpoint_path = os.path.join(self.config.checkpoint_dir, checkpoint_name)
        
        metadata = {
            'step': step,
            'timestamp': time.time(),
            'config': self.config.__dict__,
            'agent_config': self.agent.get_model_config()
        }
        
        save_checkpoint(self.train_state, checkpoint_path, metadata)
        
        # 清理旧checkpoint（保持最新的几个）
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """清理旧的checkpoint文件"""
        import os
        import glob
        
        if not os.path.exists(self.config.checkpoint_dir):
            return
        
        checkpoints = glob.glob(os.path.join(self.config.checkpoint_dir, "checkpoint_*.pkl"))
        checkpoints.sort(key=lambda x: os.path.getmtime(x))
        
        # 保持最新的N个checkpoint
        if len(checkpoints) > self.config.max_checkpoints_to_keep:
            for old_checkpoint in checkpoints[:-self.config.max_checkpoints_to_keep]:
                try:
                    os.remove(old_checkpoint)
                    logger.debug(f"删除旧checkpoint: {old_checkpoint}")
                except OSError:
                    pass
    
    def _evaluate(self, step: int):
        """
        模型评估 - 简化版本
        
        由于评估需要实际的环境，这里只做占位实现
        """
        if self.config.debug_mode:
            logger.info(f"评估占位 - Step {step}")
            
            # 简单的合成评估指标
            eval_metrics = {
                'eval/reward_mean': 0.5 + 0.1 * np.random.randn(),
                'eval/success_rate': 0.7 + 0.05 * np.random.randn(),
                'eval/episode_length': 100 + 10 * np.random.randn()
            }
            
            if self.config.use_wandb:
                wandb.log(eval_metrics, step=step)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        if not self.step_times:
            return {}
        
        recent_times = self.step_times[-100:]  # 最近100步
        
        return {
            'current_step': self.train_state.step,
            'total_steps': self.config.max_steps,
            'avg_step_time': np.mean(recent_times),
            'steps_per_sec': 1.0 / np.mean(recent_times),
            'total_training_time': time.time() - self.start_time,
            'estimated_time_remaining': (self.config.max_steps - self.train_state.step) * np.mean(recent_times)
        }


def create_trainer_from_rl_config(rl_config: Any,
                                 data_loader: Any) -> ACRLPDTrainer:
    """
    从RLConfig创建训练器
    
    Args:
        rl_config: AC Training的RLConfig
        data_loader: 数据加载器
        
    Returns:
        ACRLPDTrainer实例
    """
    
    logger.info("从RLConfig创建ACRLPDTrainer...")
    
    # 1. 创建Agent
    from ..agents_v2.acrlpd_pi0_agent import create_acrlpd_pi0_agent_from_rl_config
    agent = create_acrlpd_pi0_agent_from_rl_config(rl_config)
    
    # 2. 创建训练配置
    training_config = ACRLPDTrainingConfig(
        max_steps=getattr(rl_config, 'max_steps', 100000),
        batch_size=getattr(rl_config, 'batch_size', 128),
        pi0_lr=getattr(rl_config, 'pi0_lr', 1e-5),
        critic_lr=getattr(rl_config, 'critic_lr', 1e-3),
        log_interval=getattr(rl_config, 'log_interval', 100),
        save_interval=getattr(rl_config, 'save_interval', 5000),
        checkpoint_dir=getattr(rl_config, 'checkpoint_dir', './checkpoints'),
        use_wandb=getattr(rl_config, 'use_wandb', True),
        wandb_project=getattr(rl_config, 'wandb_project', 'acrlpd_v2'),
        seed=getattr(rl_config, 'seed', 42)
    )
    
    # 3. 创建训练器
    trainer = ACRLPDTrainer(
        agent=agent,
        data_loader=data_loader,
        config=training_config
    )
    
    logger.info("✅ 从RLConfig成功创建ACRLPDTrainer")
    return trainer