#!/usr/bin/env python3
"""
ACRLPD π₀ Training Script - AC Training v2

主训练入口脚本，整合所有v2组件：
- Agent v2: 高效的π₀ + Critic管理
- Loss Functions v2: 统一损失计算与特征共享
- Training Loop v2: 基于OpenPI的简化训练循环
- Data Loader v2: 基于OpenPI的高效数据加载

使用方法:
python train_acrlpd_pi0.py --config rl_fold_box --exp_name my_experiment

性能提升预期:
- 编译时间: 233s -> <10s  
- 训练速度: 40s/it -> <5s/it
- 整体吞吐: 0.02it/s -> >0.2it/s (10x+)
"""

import logging
import sys
import os
import argparse
from pathlib import Path
from typing import Any, Optional

# 添加路径以导入AC Training v2模块
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/dev/shm/lmc/openpi')

import jax

# AC Training v2 imports
from training_v2.training_loop import create_trainer_from_rl_config, ACRLPDTrainingConfig
from agents_v2.acrlpd_pi0_agent import create_acrlpd_pi0_agent_from_rl_config

# 原有AC Training配置和数据加载器
sys.path.insert(0, '/dev/shm/lmc/openpi/ac_training')
from config import RLTrainConfig, get_config
from data.acrlpd_data_loader_v2 import create_acrlpd_data_loader_v2

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/dev/shm/lmc/openpi/ac_training_v2/train.log')
    ]
)
logger = logging.getLogger(__name__)


def setup_jax_environment():
    """设置JAX环境"""
    # 确保使用足够的GPU内存
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.9')
    
    # 日志JAX配置
    logger.info("=== JAX环境配置 ===")
    logger.info(f"JAX版本: {jax.__version__}")
    logger.info(f"可用设备: {jax.devices()}")
    logger.info(f"默认后端: {jax.default_backend()}")
    
    # GPU内存设置
    if 'XLA_PYTHON_CLIENT_MEM_FRACTION' in os.environ:
        logger.info(f"GPU内存比例: {os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']}")


def create_v2_training_config(rl_config: RLTrainConfig, args: Any) -> ACRLPDTrainingConfig:
    """
    从原有RLConfig创建v2训练配置
    
    Args:
        rl_config: 原有的RLTrainConfig
        args: 命令行参数
        
    Returns:
        ACRLPDTrainingConfig: v2训练配置
    """
    
    # 基于原有配置创建v2配置
    v2_config = ACRLPDTrainingConfig(
        # 从原配置提取参数
        max_steps=getattr(rl_config, 'max_steps', 100000),
        batch_size=getattr(rl_config, 'batch_size', 128),
        
        # 优化后的日志间隔（减少I/O开销）
        log_interval=getattr(args, 'log_interval', 100),
        save_interval=getattr(args, 'save_interval', 5000),
        eval_interval=getattr(args, 'eval_interval', 10000),
        
        # 学习率（v2优化后的推荐值）
        pi0_lr=getattr(args, 'pi0_lr', 1e-5),      # π₀需要小学习率
        critic_lr=getattr(args, 'critic_lr', 1e-3), # Critic可以用大学习率
        warmup_steps=getattr(args, 'warmup_steps', 1000),
        
        # Checkpoint配置
        checkpoint_dir=getattr(args, 'checkpoint_dir', f'./checkpoints_v2/{args.exp_name}'),
        resume_from_checkpoint=not getattr(args, 'overwrite', False),
        max_checkpoints_to_keep=5,
        
        # wandb配置
        use_wandb=not getattr(args, 'no_wandb', False),
        wandb_project=getattr(args, 'wandb_project', 'acrlpd_v2'),
        wandb_run_name=getattr(args, 'exp_name', 'acrlpd_v2_experiment'),
        
        # 其他配置
        seed=getattr(rl_config, 'seed', 42),
        debug_mode=getattr(args, 'debug', False)
    )
    
    return v2_config


def train_acrlpd_pi0(config_name: str,
                    exp_name: str,
                    # 训练配置
                    max_steps: int = 100000,
                    batch_size: int = 128,
                    pi0_lr: float = 1e-5,
                    critic_lr: float = 1e-3,
                    
                    # 日志和保存配置  
                    log_interval: int = 100,
                    save_interval: int = 5000,
                    eval_interval: int = 10000,
                    
                    # 系统配置
                    overwrite: bool = False,
                    no_wandb: bool = False,
                    debug: bool = False,
                    
                    # 其他配置
                    checkpoint_dir: Optional[str] = None,
                    wandb_project: str = "acrlpd_v2",
                    dry_run: bool = False,
                    **kwargs):
    """
    ACRLPD π₀ 训练主函数
    
    Args:
        config_name: 配置名称 (如 'rl_fold_box')
        exp_name: 实验名称
        max_steps: 最大训练步数
        batch_size: 批次大小
        pi0_lr: π₀学习率
        critic_lr: Critic学习率
        log_interval: 日志记录间隔
        save_interval: checkpoint保存间隔
        eval_interval: 评估间隔
        overwrite: 是否覆盖已有checkpoint
        no_wandb: 是否禁用wandb
        debug: 是否启用debug模式
        checkpoint_dir: checkpoint目录
        wandb_project: wandb项目名称
        dry_run: 是否只是测试配置（不实际训练）
    """
    
    logger.info("=" * 60)
    logger.info("🚀 ACRLPD π₀ Training v2 启动")
    logger.info("=" * 60)
    logger.info(f"配置: {config_name}")
    logger.info(f"实验名称: {exp_name}")
    logger.info(f"最大步数: {max_steps}")
    logger.info(f"π₀学习率: {pi0_lr}")
    logger.info(f"Critic学习率: {critic_lr}")
    logger.info(f"批次大小: {batch_size}")
    logger.info("-" * 60)
    
    try:
        # 1. 设置JAX环境
        setup_jax_environment()
        
        # 2. 加载原有配置
        logger.info("加载配置...")
        rl_config = get_config(config_name)
        
        # 设置命令行覆盖参数
        if max_steps != 100000:
            rl_config.max_steps = max_steps
        if batch_size != 128:
            rl_config.batch_size = batch_size
            
        logger.info(f"✅ 配置加载完成: {config_name}")
        logger.info(f"  数据仓库: {rl_config.data.repo_id if hasattr(rl_config.data, 'repo_id') else 'N/A'}")
        logger.info(f"  动作维度: {rl_config.qchunking.action_dim}")
        logger.info(f"  动作horizon: {rl_config.model.action_horizon}")
        
        if dry_run:
            logger.info("🧪 Dry run模式 - 配置验证完成，退出")
            return
        
        # 3. 创建v2训练配置
        class Args:
            """临时参数容器"""
            pass
        
        args = Args()
        args.exp_name = exp_name
        args.log_interval = log_interval
        args.save_interval = save_interval
        args.eval_interval = eval_interval
        args.pi0_lr = pi0_lr
        args.critic_lr = critic_lr
        args.checkpoint_dir = checkpoint_dir or f'./checkpoints_v2/{exp_name}'
        args.overwrite = overwrite
        args.no_wandb = no_wandb
        args.debug = debug
        args.wandb_project = wandb_project
        
        v2_config = create_v2_training_config(rl_config, args)
        logger.info(f"✅ v2训练配置创建完成")
        
        # 4. 创建数据加载器
        logger.info("创建数据加载器...")
        data_loader = create_acrlpd_data_loader_v2(
            rl_config=rl_config,
            batch_size=batch_size,
            debug_mode=debug
        )
        logger.info(f"✅ 数据加载器创建完成")
        
        # 5. 创建Agent
        logger.info("创建ACRLPD Agent v2...")
        agent = create_acrlpd_pi0_agent_from_rl_config(rl_config)
        logger.info(f"✅ Agent创建完成")
        
        # 6. 创建训练器
        logger.info("创建训练器...")
        from training_v2.training_loop import ACRLPDTrainer
        trainer = ACRLPDTrainer(
            agent=agent,
            data_loader=data_loader,
            config=v2_config
        )
        logger.info(f"✅ 训练器创建完成")
        
        # 7. 开始训练
        logger.info("🚀 开始训练...")
        logger.info("-" * 60)
        
        trainer.train()
        
        logger.info("-" * 60)
        logger.info("✅ 训练完成！")
        
        # 8. 输出最终统计
        stats = trainer.get_training_stats()
        logger.info("=== 训练统计 ===")
        for key, value in stats.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")
        
    except KeyboardInterrupt:
        logger.info("⚠️  训练被用户中断")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(description="ACRLPD π₀ Training v2")
    parser.add_argument('--config', required=True, help='配置名称')
    parser.add_argument('--exp_name', required=True, help='实验名称')
    parser.add_argument('--max_steps', type=int, default=100000, help='最大训练步数')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--pi0_lr', type=float, default=1e-5, help='π₀学习率')
    parser.add_argument('--critic_lr', type=float, default=1e-3, help='Critic学习率')
    parser.add_argument('--log_interval', type=int, default=100, help='日志记录间隔')
    parser.add_argument('--save_interval', type=int, default=5000, help='checkpoint保存间隔')
    parser.add_argument('--eval_interval', type=int, default=10000, help='评估间隔')
    parser.add_argument('--overwrite', action='store_true', help='是否覆盖已有checkpoint')
    parser.add_argument('--no_wandb', action='store_true', help='是否禁用wandb')
    parser.add_argument('--debug', action='store_true', help='是否启用debug模式')
    parser.add_argument('--checkpoint_dir', type=str, help='checkpoint目录')
    parser.add_argument('--wandb_project', type=str, default='acrlpd_v2', help='wandb项目名称')
    parser.add_argument('--dry_run', action='store_true', help='是否只是测试配置')
    
    args = parser.parse_args()
    
    train_acrlpd_pi0(
        config_name=args.config,
        exp_name=args.exp_name,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        pi0_lr=args.pi0_lr,
        critic_lr=args.critic_lr,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        overwrite=args.overwrite,
        no_wandb=args.no_wandb,
        debug=args.debug,
        checkpoint_dir=args.checkpoint_dir,
        wandb_project=args.wandb_project,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()