#!/usr/bin/env python3
"""
数据集完整性检查和清理脚本

检查指定数据集中的所有episode，找出包含异常数据的episode并将其删除。

用法:
    python scripts/check_dataset_integrity.py --config rl_fold_box --fix
    python scripts/check_dataset_integrity.py --config rl_fold_box --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import shutil

import jax.numpy as jnp
import numpy as np
import pandas as pd

# Add ac_training root to path  
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from data.acrlpd_data_loader_v2 import ACRLPDDataLoaderV2

# 配置logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/dev/shm/lmc/openpi/ac_training/logs/dataset_integrity_check.log')
    ]
)
logger = logging.getLogger(__name__)


class DatasetIntegrityChecker:
    """数据集完整性检查器"""
    
    def __init__(self, config_name: str):
        self.config_name = config_name
        self.rl_config = get_config(config_name)
        self.problematic_episodes = []
        
        # 数值检查阈值
        self.max_abs_value = 1e6  # 最大绝对值
        self.max_std = 1000       # 最大标准差
        self.check_fields = ['state', 'actions', 'reward']  # 需要检查的字段
        
        logger.info(f"初始化数据集检查器: {config_name}")
        
    def _check_numerical_validity(self, data: Any, field_name: str, episode_path: str) -> List[str]:
        """
        检查数值有效性
        
        Returns:
            List[str]: 发现的问题列表
        """
        problems = []
        
        if isinstance(data, (np.ndarray, jnp.ndarray)):
            data_array = np.asarray(data)
            
            # 检查NaN
            if np.any(np.isnan(data_array)):
                problems.append(f"{field_name}: 包含NaN值")
                
            # 检查Inf  
            if np.any(np.isinf(data_array)):
                problems.append(f"{field_name}: 包含Inf值")
                
            # 检查过大数值
            max_abs = np.max(np.abs(data_array))
            if max_abs > self.max_abs_value:
                problems.append(f"{field_name}: 过大数值 (max_abs={max_abs:.2e})")
                
            # 检查异常标准差
            if data_array.size > 1:
                data_std = np.std(data_array)
                if data_std > self.max_std:
                    problems.append(f"{field_name}: 异常标准差 (std={data_std:.2f})")
                    
        elif isinstance(data, dict):
            for sub_key, sub_data in data.items():
                sub_problems = self._check_numerical_validity(sub_data, f"{field_name}.{sub_key}", episode_path)
                problems.extend(sub_problems)
                
        return problems
        
    def check_episode(self, episode_path: str) -> Dict[str, Any]:
        """
        检查单个episode文件
        
        Returns:
            Dict: 检查结果 {
                'path': str,
                'valid': bool, 
                'problems': List[str],
                'stats': Dict[str, Any]
            }
        """
        try:
            # 读取parquet文件
            df = pd.read_parquet(episode_path)
            
            result = {
                'path': episode_path,
                'valid': True,
                'problems': [],
                'stats': {
                    'num_steps': len(df),
                    'columns': list(df.columns)
                }
            }
            
            # 检查每个关键字段
            for i, row in df.iterrows():
                for field in self.check_fields:
                    if field in row:
                        field_problems = self._check_numerical_validity(
                            row[field], field, f"{episode_path}:step_{i}"
                        )
                        if field_problems:
                            result['problems'].extend([f"Step {i} - {p}" for p in field_problems])
                            result['valid'] = False
                            
                # 检查observation中的state和images
                if 'observation.state' in row:
                    state_problems = self._check_numerical_validity(
                        row['observation.state'], 'observation.state', f"{episode_path}:step_{i}"
                    )
                    if state_problems:
                        result['problems'].extend([f"Step {i} - {p}" for p in state_problems])
                        result['valid'] = False
                        
                # 检查action
                if 'action' in row:
                    action_problems = self._check_numerical_validity(
                        row['action'], 'action', f"{episode_path}:step_{i}"
                    )
                    if action_problems:
                        result['problems'].extend([f"Step {i} - {p}" for p in action_problems])
                        result['valid'] = False
                        
            return result
            
        except Exception as e:
            return {
                'path': episode_path,
                'valid': False,
                'problems': [f"无法读取文件: {e}"],
                'stats': {}
            }
            
    def find_dataset_files(self) -> List[str]:
        """找到数据集中所有的parquet文件"""
        dataset_paths = []
        
        # 从RL配置中获取数据路径
        data_config = self.rl_config.data
        if hasattr(data_config, 'repo_id'):
            # LeRobot数据集路径
            cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
            possible_paths = [
                cache_dir / f"datasets--lerobot--{data_config.repo_id}" / "snapshots",
                Path("/era-ai/lm/dataset/huggingface_cache/lerobot") / data_config.repo_id / "data"
            ]
            
            for base_path in possible_paths:
                if base_path.exists():
                    # 查找所有parquet文件
                    parquet_files = list(base_path.rglob("*.parquet"))
                    dataset_paths.extend([str(f) for f in parquet_files])
                    logger.info(f"在 {base_path} 中找到 {len(parquet_files)} 个文件")
                    
        return sorted(dataset_paths)
        
    def check_all_episodes(self, max_episodes: int = None) -> Dict[str, Any]:
        """
        检查所有episode
        
        Args:
            max_episodes: 最大检查episode数（用于测试）
            
        Returns:
            Dict: 完整检查结果
        """
        episode_files = self.find_dataset_files()
        
        if not episode_files:
            logger.error("未找到任何数据文件！")
            return {'valid_episodes': [], 'problematic_episodes': [], 'summary': {}}
            
        if max_episodes:
            episode_files = episode_files[:max_episodes]
            logger.info(f"限制检查前 {max_episodes} 个文件")
            
        logger.info(f"开始检查 {len(episode_files)} 个episode文件...")
        
        valid_episodes = []
        problematic_episodes = []
        
        for i, episode_path in enumerate(episode_files):
            if (i + 1) % 50 == 0:
                logger.info(f"进度: {i+1}/{len(episode_files)} ({100*(i+1)/len(episode_files):.1f}%)")
                
            result = self.check_episode(episode_path)
            
            if result['valid']:
                valid_episodes.append(result)
            else:
                problematic_episodes.append(result)
                logger.warning(f"发现问题文件: {episode_path}")
                for problem in result['problems'][:5]:  # 只显示前5个问题
                    logger.warning(f"  - {problem}")
                if len(result['problems']) > 5:
                    logger.warning(f"  ... 还有 {len(result['problems']) - 5} 个问题")
                    
        summary = {
            'total_episodes': len(episode_files),
            'valid_episodes': len(valid_episodes), 
            'problematic_episodes': len(problematic_episodes),
            'success_rate': len(valid_episodes) / len(episode_files) if episode_files else 0
        }
        
        return {
            'valid_episodes': valid_episodes,
            'problematic_episodes': problematic_episodes,
            'summary': summary
        }
        
    def delete_problematic_episodes(self, problematic_episodes: List[Dict], dry_run: bool = True):
        """
        删除有问题的episode文件
        
        Args:
            problematic_episodes: 有问题的episode列表
            dry_run: 是否为试运行（不实际删除）
        """
        if not problematic_episodes:
            logger.info("没有需要删除的文件")
            return
            
        logger.info(f"{'[DRY RUN] ' if dry_run else ''}准备删除 {len(problematic_episodes)} 个问题文件:")
        
        deleted_count = 0
        for episode_info in problematic_episodes:
            file_path = episode_info['path']
            logger.info(f"{'[DRY RUN] ' if dry_run else ''}删除: {file_path}")
            
            if not dry_run:
                try:
                    Path(file_path).unlink()
                    deleted_count += 1
                    logger.info(f"✅ 已删除: {file_path}")
                except Exception as e:
                    logger.error(f"❌ 删除失败 {file_path}: {e}")
                    
        if not dry_run:
            logger.info(f"✅ 成功删除 {deleted_count}/{len(problematic_episodes)} 个文件")


def main():
    parser = argparse.ArgumentParser(description="检查并清理数据集异常episode")
    parser.add_argument("--config", required=True, help="配置名称 (如: rl_fold_box)")
    parser.add_argument("--fix", action="store_true", help="删除有问题的文件（不加此参数则为试运行）")
    parser.add_argument("--dry-run", action="store_true", help="试运行，不删除文件（默认行为）")
    parser.add_argument("--max-episodes", type=int, help="限制检查的episode数量（用于测试）")
    
    args = parser.parse_args()
    
    # 创建检查器
    checker = DatasetIntegrityChecker(args.config)
    
    # 检查所有episodes
    logger.info("=" * 80)
    logger.info("开始数据集完整性检查...")
    logger.info("=" * 80)
    
    results = checker.check_all_episodes(max_episodes=args.max_episodes)
    
    # 输出摘要
    summary = results['summary']
    logger.info("=" * 80)
    logger.info("检查完成！摘要:")
    logger.info(f"  总文件数: {summary['total_episodes']}")
    logger.info(f"  正常文件: {summary['valid_episodes']}")
    logger.info(f"  问题文件: {summary['problematic_episodes']}")
    logger.info(f"  成功率: {summary['success_rate']*100:.1f}%")
    logger.info("=" * 80)
    
    if results['problematic_episodes']:
        logger.info("问题文件详情:")
        for episode_info in results['problematic_episodes']:
            logger.info(f"\n📁 {episode_info['path']}")
            for problem in episode_info['problems'][:3]:  # 只显示前3个问题
                logger.info(f"   ⚠️  {problem}")
            if len(episode_info['problems']) > 3:
                logger.info(f"   ... 还有 {len(episode_info['problems']) - 3} 个问题")
        
        # 删除处理
        if args.fix and not args.dry_run:
            logger.info("\n" + "=" * 80)
            logger.info("开始删除问题文件...")
            checker.delete_problematic_episodes(results['problematic_episodes'], dry_run=False)
        else:
            logger.info("\n" + "=" * 80)
            logger.info("试运行模式 - 如需实际删除，请添加 --fix 参数")
            checker.delete_problematic_episodes(results['problematic_episodes'], dry_run=True)
    else:
        logger.info("🎉 所有文件都正常！")


if __name__ == "__main__":
    main()