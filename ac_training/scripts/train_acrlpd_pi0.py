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

from config import get_config, RLTrainConfig
from agents.acrlpd_pi0_agent import ACRLPDPi0Agent, create_acrlpd_pi0_agent, create_acrlpd_pi0_agent_from_rl_config
from training.training_loop import ACRLPDTrainer, ACRLPDTrainingConfig
from data import create_acrlpd_data_loader

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
        "--debug-transforms", action="store_true",
        help="Enable detailed transforms debugging output"
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


def create_acrlpd_data_loader_with_sharding(rl_config: RLTrainConfig, batch_size: int, data_sharding: jax.sharding.Sharding, replicated_sharding: jax.sharding.Sharding, debug_transforms: bool = False):
    """Create ACRLPD data loader with FSDP sharding support."""
    # 基于现有create_acrlpd_data_loader，添加sharding参数
    dataloader = create_acrlpd_data_loader(
        rl_config=rl_config,
        batch_size=batch_size,
        episodes_per_memory_pool=rl_config.episodes_per_memory_pool,
        debug_transforms=debug_transforms  # 🔑 传递debug参数
    )
    
    # 传递性能分析参数到data loader
    if hasattr(rl_config, 'enable_perf_analysis') and rl_config.enable_perf_analysis:
        dataloader.enable_perf_analysis = True
    
    # 包装数据加载器以支持分片
    class ShardedACRLPDDataLoader:
        def __init__(self, base_loader, sharding, replicated_sharding):
            self.base_loader = base_loader
            self.sharding = sharding
            self.replicated_sharding = replicated_sharding
        
        def __iter__(self):
            for batch in self.base_loader:
                # OpenPI标准：使用make_array_from_process_local_data进行分片
                yield jax.tree.map(
                    lambda x: jax.make_array_from_process_local_data(self.sharding, np.asarray(x)), 
                    batch
                )
        
        def __len__(self):
            return len(self.base_loader)
            
        def sample_batch(self, max_retries: int = 3):
            """OpenPI标准的FSDP分片应用方法 - 修复标量字段分片问题，支持异常重试"""
            import time
            
            # 🎯 性能分析：FSDP分片细粒度计时
            enable_perf_analysis = getattr(self.base_loader, 'enable_perf_analysis', False)
            if enable_perf_analysis:
                total_start = time.time()
                perf_timings = {}
            
            # Step 1: 基础数据加载（支持重试）
            if enable_perf_analysis:
                base_load_start = time.time()
            
            # 🚀 关键修复：添加异常捕获和重试机制
            for retry_count in range(max_retries):
                try:
                    batch = self.base_loader.sample_batch()
                    break  # 成功获取batch，跳出重试循环
                except ValueError as e:
                    if "检测到极端异常数据，当前batch的episodes已加入灰名单" in str(e):
                        logger.warning(f"🔄 训练层检测到数据异常，重试采样 ({retry_count + 1}/{max_retries}): {e}")
                        
                        # 记录episode统计信息
                        if hasattr(self.base_loader, 'log_episode_statistics'):
                            self.base_loader.log_episode_statistics(level='warning')
                        
                        if retry_count < max_retries - 1:
                            time.sleep(0.1)  # 短暂延迟
                            continue
                        else:
                            logger.error(f"❌ 训练层重试次数用完 ({max_retries}), 数据采样失败")
                            raise RuntimeError(f"训练无法继续：数据采样重试 {max_retries} 次后仍然失败") from e
                    else:
                        # 其他类型的ValueError，直接抛出
                        raise
                except Exception as e:
                    logger.error(f"❌ 数据加载时发生未预期异常: {e}")
                    raise
            
            if enable_perf_analysis:
                perf_timings['base_data_loading'] = time.time() - base_load_start
            
            # Step 2: FSDP分片应用
            if enable_perf_analysis:
                sharding_start = time.time()
            
            # 🔧 修复：区分标量和张量字段的分片策略
            def apply_appropriate_sharding(path, x):
                """根据字段类型应用合适的分片策略"""
                path_str = '.'.join(str(p) for p in path) if isinstance(path, tuple) else str(path)
                x_array = np.asarray(x)
                
                # 检查是否为标量字段（rank 0或形状为()）
                is_scalar_field = (
                    x_array.ndim == 0 or  # 标量数组
                    x_array.shape == () or  # 空形状
                    # 已知的标量字段名
                    any(scalar_name in path_str for scalar_name in [
                        'reward', 'done', 'terminated', 'truncated', 
                        'episode_id', 'step_count', 'timestamp',
                        'success', 'failure', 'timeout',
                        # ACRLPD特定的标量字段
                        'negative_samples', 'positive_samples',
                        'sample_count', 'episode_count'
                    ])
                )
                
                
                if is_scalar_field:
                    # 标量字段使用replicated分片
                    return jax.make_array_from_process_local_data(self.replicated_sharding, x_array)
                else:
                    # 张量字段使用data分片
                    return jax.make_array_from_process_local_data(self.sharding, x_array)
            
            result = jax.tree_util.tree_map_with_path(apply_appropriate_sharding, batch)
            
            if enable_perf_analysis:
                perf_timings['fsdp_sharding'] = time.time() - sharding_start
                total_time = time.time() - total_start
                perf_timings['total'] = total_time
                
                # 每10个batch输出一次FSDP分片分析（减少日志量）
                if not hasattr(self, '_fsdp_perf_counter'):
                    self._fsdp_perf_counter = 0
                self._fsdp_perf_counter += 1
                
                if self._fsdp_perf_counter % 5 == 0:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"🔍 FSDP分片细粒度分析: 总耗时 {total_time:.3f}s")
                    for stage, timing in perf_timings.items():
                        if stage != 'total':
                            logger.info(f"    {stage}: {timing:.3f}s ({timing/total_time*100:.1f}%)")
            
            return result
            
        def data_config(self):
            """Proxy data_config method."""
            if hasattr(self.base_loader, 'data_config'):
                return self.base_loader.data_config
            return None
            
        def __getattr__(self, name):
            """Proxy any other attributes to the base loader."""
            return getattr(self.base_loader, name)
    
    return ShardedACRLPDDataLoader(dataloader, data_sharding, replicated_sharding)


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
            data_sharding=data_sharding,
            replicated_sharding=replicated_sharding,
            debug_transforms=getattr(args, 'debug_transforms', False)  # 🔑 使用专门的transforms debug参数
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
        
        if rl_config.acrlpd.enable_epoch_based_lr_schedule:
            # 使用动态双层学习率调节
            import optax
            from training.training_loop import create_dynamic_lr_schedule
            
            # 创建动态学习率调节
            actor_schedule = create_dynamic_lr_schedule(
                base_lr=rl_config.acrlpd.actor_lr,  # 使用配置中的actor_lr
                total_steps=rl_config.num_train_steps,
                warmup_epochs=rl_config.acrlpd.warmup_epochs,
                total_epochs=rl_config.acrlpd.total_epochs,
                steps_per_epoch=rl_config.acrlpd.steps_per_epoch,
                lr_min_factor=rl_config.acrlpd.lr_min_factor,
                intra_epoch_min_factor=rl_config.acrlpd.intra_epoch_min_factor,
                lr_absolute_min=getattr(rl_config.acrlpd, 'lr_absolute_min', 1e-7)
            )
            
            critic_schedule = create_dynamic_lr_schedule(
                base_lr=rl_config.acrlpd.critic_lr,  # 使用配置中的critic_lr
                total_steps=rl_config.num_train_steps,
                warmup_epochs=rl_config.acrlpd.warmup_epochs,
                total_epochs=rl_config.acrlpd.total_epochs,
                steps_per_epoch=rl_config.acrlpd.steps_per_epoch,
                lr_min_factor=rl_config.acrlpd.lr_min_factor,
                intra_epoch_min_factor=rl_config.acrlpd.intra_epoch_min_factor,
                lr_absolute_min=getattr(rl_config.acrlpd, 'lr_absolute_min', 1e-7)
            )
            
            # 使用optax直接创建优化器
            global_pi0_tx = optax.adamw(learning_rate=actor_schedule, weight_decay=1e-6)
            global_critic_tx = optax.adamw(learning_rate=critic_schedule, weight_decay=1e-5)
            
            logger.info("✅ 使用动态双层学习率调节")
        else:
            # 使用基于acrlpd静态学习率的调度器（优雅方案：单一数据源）
            global_pi0_tx = _optimizer.create_optimizer(rl_config.actor_optimizer, rl_config.get_effective_actor_lr_schedule())
            global_critic_tx = _optimizer.create_optimizer(rl_config.critic_optimizer, rl_config.get_effective_critic_lr_schedule())
            logger.info(f"✅ 使用基于acrlpd学习率的调度器 (Actor: {rl_config.acrlpd.actor_lr}, Critic: {rl_config.acrlpd.critic_lr})")
        global_temp_tx = None  # 暂时不使用温度优化器
        
        logger.info("✅ 全局优化器创建成功，将传递给FSDP初始化")
        
        # **验证优化器一致性（调试pytree问题）**
        logger.info("🔍 验证优化器pytree一致性...")
        from utils.pytree_checker import diagnose_pytree_structure, check_optimizer_consistency
        
        # 诊断优化器结构
        diagnose_pytree_structure(global_pi0_tx, "global_pi0_tx")
        diagnose_pytree_structure(global_critic_tx, "global_critic_tx")
        
        # 创建测试优化器验证一致性  
        test_pi0_tx = _optimizer.create_optimizer(rl_config.actor_optimizer, rl_config.get_effective_actor_lr_schedule())
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
        from training.acrlpd_train_state import init_acrlpd_fsdp_training
        
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
        
        # 创建轻量级agent用于trainer（避免重复创建模型）
        logger.info("🔧 创建轻量级Agent (lazy_init=True) 避免重复创建模型")
        with jax.default_device(jax.devices('cpu')[0]):
            agent = create_acrlpd_pi0_agent_from_rl_config(rl_config, agent_rng, lazy_init=True)
        
        # 从FSDP状态设置Agent模型组件
        logger.info("🔧 从FSDP训练状态设置Agent模型组件")
        agent.setup_from_fsdp_state(train_state)
        
        logger.info("FSDP验证完成")
        logger.info("直接FSDP方式已绕过agent.to_train_state()的复杂转换")
        
        # Log GPU memory after FSDP creation and verify sharding
        logger.info("🔍 GPU内存使用 - FSDP训练状态创建后:")
        log_gpu_memory("FSDP_STATE_CREATED")
        
        # 🔧 验证FSDP分片是否正确工作
        verify_fsdp_sharding(train_state, "FSDP_STATE_CREATED")
        
        # Evaluation function已移除 - 当前无真实环境评估实现
        
        # Create trainer with OpenPI标准FSDP agent和train_state  
        logger.info("Creating trainer with OpenPI标准FSDP agent...")
        trainer = ACRLPDTrainer(
            agent=agent,  # 传递OpenPI标准方式创建的agent
            dataloader=dataloader,
            rl_config=rl_config,
            training_config=training_config,
            eval_fn=None,  # 评估功能已移除
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
        
        # 🔍 关键修复：根据rl_config.resume自动确定resume路径
        resume_from = None
        if rl_config.resume:
            from pathlib import Path
            
            # 自动寻找最新checkpoint
            checkpoint_dir = Path(rl_config.checkpoint_dir)
            if checkpoint_dir.exists():
                # 寻找最新的checkpoint
                available_steps = []
                for step_dir in checkpoint_dir.iterdir():
                    if step_dir.is_dir() and step_dir.name.isdigit():
                        step = int(step_dir.name)
                        # 验证checkpoint完整性
                        components_dir = step_dir / "components"
                        params_dir = step_dir / "params"
                        if components_dir.exists() or params_dir.exists():
                            available_steps.append(step)
                
                if available_steps:
                    latest_step = max(available_steps)
                    resume_from = str(checkpoint_dir / str(latest_step))
                    logger.info(f"🔄 Auto-resuming from step {latest_step}: {resume_from}")
                else:
                    logger.warning(f"⚠️ Resume requested but no valid checkpoints found in {checkpoint_dir}")
            else:
                logger.warning(f"⚠️ Resume requested but checkpoint directory does not exist: {checkpoint_dir}")
        
        trained_agent = trainer.train(resume_from=resume_from)
        
        logger.info("Training completed successfully!")
        
        # Final evaluation已移除 - 当前无真实环境评估实现
        
    except KeyboardInterrupt:
        logger.info(" Training interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        # 删除无用的错误包装，直接抛出原始异常获得完整stack trace
        if args.debug:
            import traceback
            traceback.print_exc()
        raise


if __name__ == "__main__":
    main()