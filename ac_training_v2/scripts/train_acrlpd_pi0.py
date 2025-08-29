#!/usr/bin/env python3
"""
Training script for ACRLPD + π₀ integration.

This script provides a complete command-line interface for training ACRLPD
agents with π₀ models on robotic manipulation tasks.

Usage:
    # Basic training with H5 data
    python train_acrlpd_pi0.py --data_dir /path/to/data --config droid

    # Training with LeRobot dataset
    python train_acrlpd_pi0.py --lerobot_repo_id lerobot/aloha_sim_insertion_human --config aloha

    # Resume from checkpoint
    python train_acrlpd_pi0.py --data_dir /path/to/data --config aloha --resume /path/to/checkpoint

    # Custom hyperparameters
    python train_acrlpd_pi0.py --lerobot_repo_id lerobot/pusht --config libero \
        --horizon_length 10 --bc_alpha 0.01 --batch_size 128

    # Disable WandB logging
    python train_acrlpd_pi0.py --data_dir /path/to/data --config droid --no_wandb
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

# CRITICAL: Set HuggingFace cache to data disk, not system disk
# This prevents "No space left on device" errors and enables local dataset access
os.environ['HF_HOME'] = '/era-ai/lm/dataset/huggingface'
os.environ['HF_CACHE_HOME'] = '/era-ai/lm/dataset/huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = '/era-ai/lm/dataset/huggingface_cache/datasets'
os.environ['HF_LEROBOT_HOME'] = '/era-ai/lm/dataset/huggingface_cache/lerobot'

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# ✅ Stage 1.3 修复：更新为agents_v2和training_v2导入
from config import get_config, RLTrainConfig
from agents_v2.acrlpd_pi0_agent import ACRLPDPi0Agent, create_acrlpd_pi0_agent_from_rl_config
from training_v2.training_loop import ACRLPDTrainer, ACRLPDTrainingConfig, create_simple_eval_fn
from data_v2.acrlpd_data_loader import create_acrlpd_data_loader

import openpi.models.pi0 as _pi0
import openpi.training.config as openpi_config
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding

# Reduce verbose output before imports
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")      # Suppress TF info logs
os.environ.setdefault("JAX_LOG_COMPILES", "0")          # Disable JAX compilation logs

# Setup logging - 减少详细输出
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress verbose output from JAX/Flax/transformers libraries
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("flax").setLevel(logging.WARNING)  
logging.getLogger("optax").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("absl").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser for unified config system."""
    parser = argparse.ArgumentParser(
        description="Train ACRLPD + π₀ agents using unified configuration system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration selection (main argument)
    parser.add_argument(
        "--config", type=str, required=True,
        choices=["rl_aloha_fold", "rl_fold_box", "rl_libero", "rl_droid"],
        help="Predefined configuration for different robot platforms"
    )
    
    # Experiment setup
    parser.add_argument(
        "--exp_name", type=str, required=True,
        help="Experiment name for checkpoints and logging"
    )
    
    # Training control
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--overwrite", action="store_true", 
        help="Overwrite existing experiment"
    )
    
    # Environment (for online training)
    parser.add_argument(
        "--env_name", type=str, default=None,
        help="Environment name for online training evaluation"
    )
    
    # System configuration
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    
    # Debug options
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--no_wandb", action="store_true",
        help="Disable WandB logging"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Setup without training"
    )
    
    # FSDP修复方案选择
    parser.add_argument(
        "--fsdp_fix", type=str, default="direct", 
        choices=["direct", "manual"],
        help="选择FSDP修复方案: direct(默认) 或 manual"
    )
    
    return parser


def load_rl_config(args: argparse.Namespace) -> RLTrainConfig:
    """加载并定制RLTrainConfig"""
    # 获取预定义配置
    rl_config = get_config(args.config)
    
    # 使用dataclasses.replace更新实验名和选项
    import dataclasses
    rl_config = dataclasses.replace(
        rl_config,
        exp_name=args.exp_name,
        resume=args.resume,
        overwrite=args.overwrite,
        seed=args.seed,
        wandb_enabled=not args.no_wandb
    )
    
    return rl_config


def log_gpu_memory(step_name: str = "", alert_threshold: float = 95.0):
    """Enhanced GPU memory logging with step tracking and alerts."""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            step_prefix = f"🔍 GPU Memory - {step_name}: " if step_name else "🔍 GPU Memory Status:"
            logger.info(step_prefix)
            total_used = 0
            total_capacity = 0
            usage_data = []
            max_usage_gpu = 0
            max_usage_mb = 0
            
            for line in result.stdout.strip().split('\n'):
                parts = line.split(', ')
                if len(parts) == 3:
                    gpu_id, used, total = parts
                    used_mb = int(used)
                    total_mb = int(total)
                    usage_pct = (used_mb / total_mb) * 100
                    total_used += used_mb
                    total_capacity += total_mb
                    usage_data.append(used_mb)
                    
                    # 记录最高使用GPU
                    if used_mb > max_usage_mb:
                        max_usage_mb = used_mb
                        max_usage_gpu = int(gpu_id)
                    
                    logger.info(f"  GPU {gpu_id}: {used}MB/{total}MB ({usage_pct:.1f}%)")
            
            # 计算统计数据
            overall_pct = (total_used / total_capacity) * 100 if total_capacity > 0 else 0
            avg_per_gpu = total_used / len(usage_data) if len(usage_data) > 0 else 0
            std_dev = (sum((x - avg_per_gpu) ** 2 for x in usage_data) / len(usage_data)) ** 0.5 if len(usage_data) > 0 else 0
            
            logger.info(f"  Total: {total_used}MB/{total_capacity}MB ({overall_pct:.1f}%) | Avg/GPU: {avg_per_gpu:.0f}MB")
            logger.info(f"  Max GPU: GPU{max_usage_gpu} ({max_usage_mb}MB) | StdDev: {std_dev:.0f}MB")
            
            # FSDP效果分析
            analyze_fsdp_effectiveness(usage_data, avg_per_gpu, std_dev)
            
            # 内存异常报警
            if overall_pct > alert_threshold:
                logger.error(f"🚨 GPU内存警告: {overall_pct:.1f}% > {alert_threshold}% 阈值!")
                if max_usage_mb > 76000:  # 76GB alarm (95% of 80GB)
                    logger.error(f"🚨 严重警告: GPU{max_usage_gpu}使用{max_usage_mb}MB，接近80GB上限!")
            
            # 返回数据用于进一步分析
            return {
                'usage_data': usage_data,
                'avg_per_gpu': avg_per_gpu,
                'max_usage': max_usage_mb,
                'max_usage_gpu': max_usage_gpu,
                'std_dev': std_dev,
                'overall_pct': overall_pct
            }
                    
    except Exception as e:
        logger.warning(f"Failed to get GPU memory info: {e}")
        return None


def analyze_fsdp_effectiveness(usage_data, avg_per_gpu, std_dev):
    """分析FSDP分片效果"""
    if len(usage_data) < 2 or avg_per_gpu == 0:
        return
        
    # 计算变异系数 (CV = std_dev / mean)
    cv = std_dev / avg_per_gpu
    
    # 检查是否有明显的内存不均
    min_usage = min(usage_data)
    max_usage = max(usage_data)
    usage_range = max_usage - min_usage
    
    logger.info(f"  FSDP分析: CV={cv:.3f}, Range={usage_range:.0f}MB ({min_usage:.0f}-{max_usage:.0f})")
    
    if cv < 0.1 and usage_range < 5000:  # 变异系数<0.1且范围<5GB
        logger.info("  ✅ FSDP效果优秀：内存使用均匀，分片工作良好")
    elif cv < 0.3 and usage_range < 15000:  # 变异系数<0.3且范围<15GB
        logger.info("  ⚠️  FSDP效果一般：存在一定内存不均")
    elif max_usage > 50000:  # 任何GPU超过50GB
        logger.info("  ❌ FSDP分片可能失效：存在高内存使用GPU")
        # 检测是否为复制而非分片
        high_usage_count = sum(1 for x in usage_data if x > 40000)
        if high_usage_count >= len(usage_data) * 0.8:  # 80%以上GPU高内存使用
            logger.error("  🚨 FSDP完全失效：参数被复制到多个GPU而非分片!")
    else:
        logger.info("  ✅ FSDP基本正常：内存使用在合理范围内")


def verify_fsdp_sharding(train_state, step_name: str = ""):
    """验证FSDP分片是否正确工作"""
    step_prefix = f"🔍 FSDP验证 - {step_name}: " if step_name else "🔍 FSDP分片验证:"
    logger.info(step_prefix)
    
    try:
        # 检查关键参数的分片情况
        if hasattr(train_state, 'pi0_params'):
            param_count = 0
            sharded_count = 0
            replicated_count = 0
            
            def check_sharding(param_path, param):
                nonlocal param_count, sharded_count, replicated_count
                if hasattr(param, 'sharding') and hasattr(param, 'shape'):
                    param_count += 1
                    size_mb = param.nbytes / (1024 * 1024)
                    
                    # 检查分片状态
                    if hasattr(param.sharding, 'spec'):
                        spec = param.sharding.spec
                        is_sharded = any(axis is not None for axis in spec)
                        
                        if is_sharded:
                            sharded_count += 1
                            if size_mb > 10:  # 只记录大参数
                                logger.info(f"  ✅ SHARDED: {param_path} ({param.shape}, {size_mb:.1f}MB)")
                        else:
                            replicated_count += 1
                            if size_mb > 1:  # 记录中等大小参数
                                logger.info(f"  🔄 REPLICATED: {param_path} ({param.shape}, {size_mb:.1f}MB)")
            
            # 遍历参数
            def traverse_params(params, prefix=""):
                if isinstance(params, dict):
                    for key, value in params.items():
                        new_prefix = f"{prefix}.{key}" if prefix else key
                        traverse_params(value, new_prefix)
                elif hasattr(params, 'shape'):
                    check_sharding(prefix, params)
                elif hasattr(params, 'value') and hasattr(params.value, 'shape'):
                    check_sharding(prefix, params.value)
            
            traverse_params(train_state.pi0_params, "pi0_params")
            
            # 总结分片效果
            if param_count > 0:
                sharded_pct = (sharded_count / param_count) * 100
                logger.info(f"  📊 分片统计: {sharded_count}/{param_count} 参数被分片 ({sharded_pct:.1f}%)")
                
                if sharded_pct > 70:
                    logger.info("  ✅ FSDP分片效果良好")
                elif sharded_pct > 30:
                    logger.info("  ⚠️  FSDP分片效果中等")
                else:
                    logger.info("  ❌ FSDP分片可能失效")
            else:
                logger.info("  ⚠️  未找到可检查的参数")
                
    except Exception as e:
        logger.warning(f"FSDP验证失败: {e}")


def setup_jax_environment_with_fsdp(args: argparse.Namespace, rl_config: RLTrainConfig):
    """Setup JAX environment with OpenPI FSDP support."""
    logger.info(f"JAX version: {jax.__version__}")
    logger.info(f"Available devices: {jax.devices()}")
    
    # Log initial GPU memory
    log_gpu_memory("INITIAL")
    
    # OpenPI标准：检查批次大小与设备数量的兼容性
    if rl_config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {rl_config.batch_size} must be divisible by device count {jax.device_count()}"
        )
    
    # 创建OpenPI标准mesh和sharding
    mesh = sharding.make_mesh(rl_config.fsdp_devices)
    
    # OpenPI标准：始终使用DATA_AXIS进行分片
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    
    logger.info(f"Created FSDP mesh: {mesh}")
    logger.info(f"Mesh shape: {mesh.shape}")
    logger.info(f"Data sharding: {data_sharding}")
    
    # 🔍 验证OpenPI标准FSDP配置
    batch_per_device = rl_config.batch_size // jax.device_count()
    logger.info("✅ OpenPI标准FSDP配置验证:")
    logger.info(f"   总batch_size: {rl_config.batch_size}")
    logger.info(f"   设备数量: {jax.device_count()}")
    logger.info(f"   每设备batch: {batch_per_device}")
    logger.info(f"   数据分片策略: {data_sharding}")
    
    # Log GPU memory after FSDP setup
    log_gpu_memory("FSDP_SETUP")
    
    # Set memory preallocation
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    
    if args.debug:
        jax.config.update("jax_debug_nans", True)
        jax.config.update("jax_debug_infs", True)
    
    return mesh, data_sharding, replicated_sharding


def create_acrlpd_data_loader_with_sharding(rl_config: RLTrainConfig, batch_size: int, data_sharding: jax.sharding.Sharding):
    """Create ACRLPD data loader with FSDP sharding support."""
    # 基于现有create_acrlpd_data_loader，添加sharding参数
    dataloader = create_acrlpd_data_loader(
        rl_config=rl_config,
        batch_size=batch_size,
        episodes_per_memory_pool=rl_config.episodes_per_memory_pool
    )
    
    # 包装数据加载器以支持分片
    class ShardedACRLPDDataLoader:
        def __init__(self, base_loader, sharding):
            self.base_loader = base_loader
            self.sharding = sharding
        
        def __iter__(self):
            for batch in self.base_loader:
                # OpenPI标准：使用make_array_from_process_local_data进行分片
                yield jax.tree.map(
                    lambda x: jax.make_array_from_process_local_data(self.sharding, np.asarray(x)), 
                    batch
                )
        
        def __len__(self):
            return len(self.base_loader)
            
        def sample_batch(self):
            """OpenPI标准的FSDP分片应用方法"""
            batch = self.base_loader.sample_batch()
            return jax.tree.map(
                lambda x: jax.make_array_from_process_local_data(self.sharding, np.asarray(x)), 
                batch
            )
            
        def data_config(self):
            """Proxy data_config method."""
            if hasattr(self.base_loader, 'data_config'):
                return self.base_loader.data_config
            return None
            
        def __getattr__(self, name):
            """Proxy any other attributes to the base loader."""
            return getattr(self.base_loader, name)
    
    return ShardedACRLPDDataLoader(dataloader, data_sharding)


# create_agent_with_fsdp function removed to avoid nested mesh context
# FSDP initialization now handled directly in training loop using OpenPI patterns


def main():
    """Main training function using unified configuration system."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    logger.info("🚀 Starting ACRLPD + π₀ training with OpenPI FSDP")
    logger.info(f"Arguments: {vars(args)}")
    
    # Load unified configuration first
    logger.info(f"Loading configuration: {args.config}")
    rl_config = load_rl_config(args)
    
    # Setup JAX environment with FSDP support
    mesh, data_sharding, replicated_sharding = setup_jax_environment_with_fsdp(args, rl_config)
    
    try:
        # Create training config for additional parameters
        training_config = ACRLPDTrainingConfig(
            env_name=args.env_name
        )
        
        logger.info(f"✓ Configuration loaded successfully")
        logger.info(f"  Model: {type(rl_config.model).__name__}")
        logger.info(f"  Dataset: {rl_config.data.repo_id if hasattr(rl_config.data, 'repo_id') else 'Unknown'}")
        logger.info(f"  Batch size: {rl_config.batch_size}")
        logger.info(f"  Training steps: {rl_config.num_train_steps}")
        logger.info(f"  FSDP devices: {rl_config.fsdp_devices}")
        
        if args.dry_run:
            logger.info("✓ Dry run completed successfully")
            return
        
        # Set random seed
        np.random.seed(rl_config.seed)
        rng = jax.random.PRNGKey(rl_config.seed)
        
        # Create data loader with FSDP sharding support
        logger.info("Creating data loader with FSDP sharding...")
        dataloader = create_acrlpd_data_loader_with_sharding(
            rl_config=rl_config,
            batch_size=rl_config.batch_size,
            data_sharding=data_sharding
        )
        logger.info(f"Data loader created with FSDP sharding support")
        
        # Create agent with proper FSDP support directly
        logger.info("Creating ACRLPD + π₀ agent with FSDP...")
        logger.info(f"Debug: About to split RNG...")
        rng, agent_rng = jax.random.split(rng)
        logger.info(f"Debug: RNG split complete, entering mesh context...")
        
        # **关键修复：创建全局优化器避免pytree元数据不匹配**
        logger.info("🔧 创建全局优化器实例以确保pytree一致性...")
        
        # 在FSDP上下文外创建全局优化器（确保一致性）
        import openpi.training.optimizer as _optimizer
        global_pi0_tx = _optimizer.create_optimizer(rl_config.actor_optimizer, rl_config.actor_lr_schedule)
        global_critic_tx = _optimizer.create_optimizer(rl_config.critic_optimizer, rl_config.critic_lr_schedule)
        global_temp_tx = None  # 暂时不使用温度优化器
        
        logger.info("✅ 全局优化器创建成功，将传递给FSDP初始化")
        
        # **验证优化器一致性（调试pytree问题）**
        logger.info("🔍 验证优化器pytree一致性...")
        from utils.pytree_checker import diagnose_pytree_structure, check_optimizer_consistency
        
        # 诊断优化器结构
        diagnose_pytree_structure(global_pi0_tx, "global_pi0_tx")
        diagnose_pytree_structure(global_critic_tx, "global_critic_tx")
        
        # 创建测试优化器验证一致性
        test_pi0_tx = _optimizer.create_optimizer(rl_config.actor_optimizer, rl_config.actor_lr_schedule)
        consistency_ok = check_optimizer_consistency(global_pi0_tx, test_pi0_tx, "global_pi0_tx", "test_pi0_tx")
        
        if not consistency_ok:
            logger.warning("⚠️ 优化器一致性检查失败，但继续训练...")
        else:
            logger.info("✅ 优化器一致性检查通过")
        
        # **修复8卡FSDP内存爆炸问题**
        # 问题：参数在所有GPU上复制而不是分片，导致每个GPU使用61GB
        # 解决方案：选择适当的FSDP修复方案，并传递全局优化器
        
        logger.info(f"🔧 使用 {args.fsdp_fix.upper()} FSDP修复方案")
        
        # **统一使用init_acrlpd_fsdp_training避免模型重复创建**
        logger.info("🚀 使用统一FSDP初始化（避免模型重复创建）")
        # ✅ Stage 1.3 修复：使用training_v2导入
        from training_v2.acrlpd_train_state import init_acrlpd_fsdp_training
        
        train_state, state_sharding, lazy_jit_creator = init_acrlpd_fsdp_training(
            rl_config=rl_config,
            mesh=mesh,
            rng=agent_rng,
            data_sharding=data_sharding,
            step=0,
            global_pi0_tx=global_pi0_tx,
            global_critic_tx=global_critic_tx
        )
        
        # 延迟创建JIT函数
        logger.info("🔧 创建JIT训练函数...")
        fsdp_train_step_fn = lazy_jit_creator()
        
        # 验证FSDP是否真正工作
        logger.info("验证FSDP分片效果...")
        
        # 检查实际显存使用
        memory_stats = log_gpu_memory("FSDP验证")
        if memory_stats:
            avg_memory = memory_stats['avg_per_gpu']
            max_memory = memory_stats['max_usage']
            logger.info(f"💾 FSDP内存使用: {avg_memory:.0f}MB/GPU (最大{max_memory:.0f}MB)")
        
        # 创建轻量级agent用于trainer（不包含训练状态）
        with jax.default_device(jax.devices('cpu')[0]):
            agent = create_acrlpd_pi0_agent_from_rl_config(rl_config, agent_rng)
        
        logger.info("FSDP验证完成")
        logger.info("直接FSDP方式已绕过agent.to_train_state()的复杂转换")
        
        # Log GPU memory after FSDP creation and verify sharding
        logger.info("🔍 GPU内存使用 - FSDP训练状态创建后:")
        log_gpu_memory("FSDP_STATE_CREATED")
        
        # 🔧 验证FSDP分片是否正确工作
        verify_fsdp_sharding(train_state, "FSDP_STATE_CREATED")
        
        # Create evaluation function if needed
        eval_fn = None
        if args.env_name:
            eval_fn = create_simple_eval_fn(args.env_name)
            logger.info(f"📈 Created evaluation function for {args.env_name}")
        
        # Create trainer with OpenPI标准FSDP agent和train_state  
        logger.info("Creating trainer with OpenPI标准FSDP agent...")
        trainer = ACRLPDTrainer(
            agent=agent,  # 传递OpenPI标准方式创建的agent
            dataloader=dataloader,
            rl_config=rl_config,
            training_config=training_config,
            eval_fn=eval_fn,
            # FSDP参数
            mesh=mesh,
            data_sharding=data_sharding,
            replicated_sharding=replicated_sharding,
            # 全局优化器参数（修复pytree一致性）
            global_pi0_tx=global_pi0_tx,
            global_critic_tx=global_critic_tx
        )
        
        # 设置训练器的FSDP状态（统一方式创建）
        trainer.fsdp_train_state = train_state
        trainer.train_state_sharding = state_sharding
        trainer.fsdp_train_step = fsdp_train_step_fn
        trainer.use_fsdp = True
        
        # 验证FSDP组件设置成功
        logger.info("✅ FSDP components set successfully")
        
        # **添加pytree一致性验证**
        logger.info("🔍 验证训练状态pytree一致性...")
        from utils.pytree_checker import diagnose_pytree_structure, validate_fsdp_compatibility
        
        # 诊断训练状态结构
        diagnose_pytree_structure(train_state, "train_state")
        
        # 验证FSDP兼容性
        fsdp_compatible = validate_fsdp_compatibility(train_state, mesh, data_sharding)
        if not fsdp_compatible:
            logger.warning("⚠️ FSDP兼容性验证失败，但继续训练...")
        else:
            logger.info("✅ FSDP兼容性验证通过")
        
        # 确保PyTree结构一致性 - 在JIT编译前预热train_state
        logger.info("🔧 预热训练状态确保PyTree一致性...")
        jax.block_until_ready(train_state)
        
        logger.info("✅ 训练器创建成功，使用OpenPI标准8卡FSDP配置")
        
        # Log GPU memory before training starts
        logger.info("🔍 GPU内存使用 - 训练开始前:")
        log_gpu_memory("BEFORE_TRAINING")
        
        # Start training
        logger.info("Starting training...")
        trained_agent = trainer.train()
        
        logger.info("Training completed successfully!")
        
        # Final evaluation
        if eval_fn:
            logger.info("Running final evaluation...")
            rng, eval_rng = jax.random.split(rng) 
            final_reward, final_length = eval_fn(trained_agent, eval_rng, deterministic=True)
            logger.info(f"Final evaluation: reward={final_reward:.3f}, length={final_length}")
        
    except KeyboardInterrupt:
        logger.info(" Training interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f" Training failed with error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()