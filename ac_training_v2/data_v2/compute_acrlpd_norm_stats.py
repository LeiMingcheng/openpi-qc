"""
Compute normalization statistics for ACRLPD datasets.

This script computes the normalization statistics for ACRLPD training data,
compatible with OpenPI's normalization architecture.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import tqdm

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms
from openpi.shared import normalize as _normalize

logger = logging.getLogger(__name__)


class RemoveStrings(transforms.DataTransformFn):
    """Remove string fields that are not needed for normalization stats."""
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def compute_acrlpd_norm_stats(
    rl_config: Any,  # RLTrainConfig (avoiding circular import)
    output_dir: Path,
    max_samples: int = 2000,  # 降低默认样本数，提高稳定性
    batch_size: int = 32      # 降低默认batch大小
) -> Dict[str, _normalize.NormStats]:
    """
    Compute normalization statistics using OpenPI native methods.
    
    Args:
        rl_config: RLTrainConfig configuration
        output_dir: Directory to save norm_stats.json
        max_samples: Maximum number of samples to use for computing stats
        batch_size: Batch size for processing
    
    Returns:
        Dictionary of normalization statistics
    """
    
    logger.info("🚀 使用OpenPI原生方法计算normalization statistics")
    
    # 创建data config（使用OpenPI标准方法）
    print("🔍 创建OpenPI DataConfig...")
    data_config = rl_config.data.create(rl_config.assets_dirs, rl_config.model)
    
    repo_id = data_config.repo_id
    logger.info(f"📊 数据集: {repo_id}")
    print(f"✅ DataConfig创建成功: {repo_id}")
    
    # 创建dataset（使用OpenPI标准torch dataset，但跳过视频处理）
    print("🔍 创建torch dataset（跳过视频处理）...")
    
    # 为norm stats计算创建一个只包含必要数据的轻量版dataset
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    
    # 创建简化的LeRobot dataset，用于norm stats计算
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    
    print("🔍 创建简化LeRobot dataset...")
    
    # 🔧 CRITICAL FIX: Apply torch.stack Column compatibility patch first
    import torch
    if not hasattr(torch, '_openpi_column_patch'):
        logger.info("🔧 应用torch.stack Column兼容性patch for norm stats")
        original_stack = torch.stack
        
        def patched_stack(tensors, dim=0, *, out=None):
            # 检查是否是HF Dataset的Column对象
            if hasattr(tensors, '__class__') and 'Column' in tensors.__class__.__name__:
                # 将Column转换为tensor列表
                tensor_list = [torch.as_tensor(item) for item in tensors]
                return original_stack(tensor_list, dim=dim, out=out)
            else:
                return original_stack(tensors, dim=dim, out=out)
        
        torch.stack = patched_stack
        torch._openpi_column_patch = True
    
    try:
        # 先尝试不使用delta_timestamps，避免时间戳问题
        lerobot_dataset = LeRobotDataset(
            data_config.repo_id,
            tolerance_s=1e-4,
            video_backend="pyav",  # 使用PyAV backend
            skip_problematic_episodes=True
        )
        print("✅ PyAV backend 初始化成功")
        
        # 直接使用原生LeRobot dataset，不需要OpenPI包装
        dataset = lerobot_dataset
        
    except Exception as e:
        print(f"⚠️ 视频backend失败: {e}")
        # 再次尝试，这次完全跳过视频处理
        print("🔍 尝试跳过所有视频处理...")
        try:
            lerobot_dataset = LeRobotDataset(
                data_config.repo_id,
                tolerance_s=1e-4,
                skip_problematic_episodes=True
                # 不指定video_backend，使用默认或跳过
            )
            dataset = lerobot_dataset
            print("✅ 跳过视频处理成功")
        except Exception as e2:
            raise RuntimeError(f"无法创建数据集: {e2}") from e2
    
    print(f"✅ Dataset创建成功，总长度: {len(dataset)}")
    
    # 简化的批处理方式，直接从dataset采样而不使用复杂的DataLoader
    logger.info(f"📊 开始直接采样数据计算统计信息")
    logger.info(f"   - 数据集大小: {len(dataset)}")
    logger.info(f"   - 最大样本数: {max_samples}")
    
    # 初始化统计收集器
    keys = ["state", "actions"]  
    stats = {key: _normalize.RunningStats() for key in keys}
    
    logger.info(f"🔄 开始计算normalization stats for keys: {keys}")
    
    # 🔑 完全按照OpenPI原生方法重写
    
    # 🔑 关键修复：对于norm stats，只使用基础transforms，跳过复杂的model transforms
    print("🔍 应用简化的transforms（仅用于norm stats计算）...")
    
    # 只使用repack_transforms，跳过可能导致维度问题的data_transforms
    essential_transforms = [
        *data_config.repack_transforms.inputs,
        RemoveStrings(),  # 移除字符串字段
    ]
    
    dataset = _data_loader.TransformedDataset(dataset, essential_transforms)
    
    # 计算batch数量
    if max_samples is not None and max_samples < len(dataset):
        num_batches = max_samples // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    
    # 创建DataLoader（OpenPI标准方式）
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=0,  # 单进程避免视频问题
        shuffle=shuffle,
        num_batches=num_batches,
    )
    
    # 🔑 OpenPI原生统计计算循环（带debug信息）
    batch_count = 0
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing norm stats"):
        batch_count += 1
        
        # Debug前几个batch的详细信息
        if batch_count <= 3:
            print(f"\n🔍 DEBUG Batch {batch_count}:")
            print(f"  - Batch keys: {list(batch.keys())}")
            for key in keys:
                if key in batch:
                    batch_data = batch[key]
                    print(f"  - {key}: type={type(batch_data)}, len={len(batch_data) if hasattr(batch_data, '__len__') else 'no len'}")
                    if hasattr(batch_data, '__len__') and len(batch_data) > 0:
                        first_elem = batch_data[0]
                        print(f"    - [0]: type={type(first_elem)}, shape={getattr(first_elem, 'shape', 'no shape')}")
        
        for key in keys:
            if key in batch:
                try:
                    # 🔑 完全按照OpenPI方式：batch[key][0]然后reshape
                    values = np.asarray(batch[key][0])
                    reshaped_values = values.reshape(-1, values.shape[-1])
                    
                    # Debug维度信息
                    if batch_count <= 3:
                        print(f"  - {key} 处理成功: original shape={values.shape} -> reshaped shape={reshaped_values.shape}")
                    
                    stats[key].update(reshaped_values)
                    
                except Exception as e:
                    # 详细的错误debug信息
                    print(f"\n❌ ERROR processing {key} in batch {batch_count}:")
                    print(f"  - Exception: {e}")
                    try:
                        values = np.asarray(batch[key][0])
                        print(f"  - values shape: {values.shape}")
                        print(f"  - values dtype: {values.dtype}")
                        print(f"  - values.shape[-1]: {values.shape[-1]}")
                        print(f"  - reshape target: (-1, {values.shape[-1]})")
                        if stats[key]._mean is not None:
                            print(f"  - current stats mean shape: {stats[key]._mean.shape}")
                    except Exception as e2:
                        print(f"  - Could not extract debug info: {e2}")
                    raise
    
    # 计算最终统计信息（使用OpenPI标准方法）
    logger.info("🔄 计算最终统计信息...")
    
    norm_stats = {}
    for key, running_stats in stats.items():
        logger.info(f"处理 '{key}': {running_stats._count} 个样本")
        
        if running_stats._count > 1:
            try:
                final_stats = running_stats.get_statistics()
                norm_stats[key] = final_stats
                
                logger.info(f"✅ '{key}' 统计信息:")
                logger.info(f"  - shape: {final_stats.mean.shape}")
                logger.info(f"  - mean range: [{final_stats.mean.min():.6f}, {final_stats.mean.max():.6f}]")
                logger.info(f"  - std range: [{final_stats.std.min():.6f}, {final_stats.std.max():.6f}]")
                
            except ValueError as e:
                logger.warning(f"计算 '{key}' 统计信息失败: {e}")
        else:
            logger.warning(f"'{key}' 样本数不足: 只有 {running_stats._count} 个样本")
    
    if not norm_stats:
        raise RuntimeError("未能计算出任何有效的统计信息！请检查数据加载器。")
    
    # 保存统计信息（使用OpenPI标准方法）
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"💾 保存统计信息到: {output_dir}")
    _normalize.save(output_dir, norm_stats)
    
    # 输出总结
    print(f"\n🎉 成功计算并保存normalization statistics!")
    print(f"📁 输出目录: {output_dir}")
    print(f"📊 统计信息概要:")
    
    for key, stats_obj in norm_stats.items():
        print(f"  🔹 {key}:")
        print(f"    - Shape: {stats_obj.mean.shape}")
        print(f"    - Mean range: [{stats_obj.mean.min():.6f}, {stats_obj.mean.max():.6f}]")
        print(f"    - Std range: [{stats_obj.std.min():.6f}, {stats_obj.std.max():.6f}]")
        if hasattr(stats_obj, 'q01'):
            print(f"    - Q01 range: [{stats_obj.q01.min():.6f}, {stats_obj.q01.max():.6f}]")
            print(f"    - Q99 range: [{stats_obj.q99.min():.6f}, {stats_obj.q99.max():.6f}]")
    
    return norm_stats


def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compute ACRLPD normalization statistics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--repo-id", required=True, 
                       help="LeRobot dataset repository ID (local or remote)")
    parser.add_argument("--output-dir", required=True, 
                       help="Output directory for norm_stats.json") 
    parser.add_argument("--max-samples", type=int, default=10000, 
                       help="Maximum samples to process")
    parser.add_argument("--batch-size", type=int, default=32, 
                       help="Batch size for processing")
    parser.add_argument("--action-horizon", type=int, default=10, 
                       help="Action horizon for sequence generation")
    parser.add_argument("--tolerance-s", type=float, default=1e-4,
                       help="Tolerance for timestamp synchronization")
    parser.add_argument("--skip-problematic-episodes", action="store_true",
                       help="Skip episodes with timestamp issues")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check if we have a local dataset path instead of remote repo_id
    local_dataset_path = Path(f"/era-ai/lm/dataset/huggingface_cache/lerobot/{args.repo_id}")
    if local_dataset_path.exists():
        logger.info(f"✓ Found local dataset at: {local_dataset_path}")
        # Use local path as repo_id for LeRobot dataset loading
        actual_repo_id = str(local_dataset_path)
    else:
        logger.info(f"Using remote repo_id: {args.repo_id}")
        actual_repo_id = args.repo_id

    logger.info(f"🚀 Starting ACRLPD normalization statistics computation")
    logger.info(f"📊 Configuration:")
    logger.info(f"  - repo_id: {args.repo_id} -> {actual_repo_id}")
    logger.info(f"  - output_dir: {args.output_dir}")
    logger.info(f"  - max_samples: {args.max_samples}")
    logger.info(f"  - batch_size: {args.batch_size}")
    logger.info(f"  - action_horizon: {args.action_horizon}")
    
    # Use unified config system but override repo_id with command line arg
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import get_config
    
    logger.info("Loading rl_fold_box config...")
    rl_config = get_config("rl_fold_box")
    
    # 🔑 CRITICAL FIX: Override repo_id with command line argument
    print(f"🔄 STEP 1: Original config repo_id: {rl_config.data.repo_id}")
    print(f"🔄 STEP 2: Overriding with command line repo_id: {actual_repo_id}")
    
    # Create a modified config with the correct repo_id
    import dataclasses
    from openpi.training import config as openpi_config
    
    # Get the original data config factory
    original_data_factory = rl_config.data
    print(f"🔄 STEP 3: Got original data factory")
    
    # Create a new LeRobotAlohaDataConfig with the correct repo_id
    modified_data_config = openpi_config.LeRobotAlohaDataConfig(
        repo_id=actual_repo_id,  # Use command line repo_id
        default_prompt=original_data_factory.default_prompt,
        adapt_to_pi=original_data_factory.adapt_to_pi,
        assets=original_data_factory.assets,
        repack_transforms=original_data_factory.repack_transforms,
        base_config=original_data_factory.base_config
    )
    print(f"🔄 STEP 4: Created modified data config with repo_id: {modified_data_config.repo_id}")
    
    # Create modified RL config with new data config and NO weight loader (for norm stats only)
    rl_config = dataclasses.replace(
        rl_config, 
        data=modified_data_config,
        weight_loader=None  # Skip weight loading for norm stats computation
    )
    print(f"🔄 STEP 5: Replaced RL config, new repo_id: {rl_config.data.repo_id}")
    print(f"🔄 STEP 6: Disabled weight loader for norm stats computation")
    
    print(f"✓ Using dataset: {actual_repo_id}")
    
    try:
        # Compute and save statistics using unified config
        compute_acrlpd_norm_stats(
            rl_config=rl_config,
            output_dir=Path(args.output_dir),
            max_samples=args.max_samples,
            batch_size=args.batch_size
        )
        
        logger.info("🎉 ACRLPD normalization statistics computation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to compute normalization statistics: {e}")
        raise


if __name__ == "__main__":
    main()