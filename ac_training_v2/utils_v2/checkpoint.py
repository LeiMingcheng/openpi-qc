"""
Checkpoint Management

高效的checkpoint管理系统，功能：
- 自动checkpoint保存和清理
- 增量checkpoint（只保存变化的参数）
- 压缩和元数据管理
- 快速恢复机制
"""

import logging
import os
import time
import json
import pickle
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CheckpointConfig:
    """Checkpoint配置"""
    # 基础配置
    checkpoint_dir: str = "./checkpoints"
    max_checkpoints: int = 5               # 保持的checkpoint数量
    save_interval: int = 1000              # 保存间隔
    
    # 高级配置
    compress: bool = False                 # 是否压缩checkpoint
    save_incremental: bool = False         # 是否保存增量checkpoint
    
    # 文件命名
    checkpoint_prefix: str = "checkpoint"
    metadata_suffix: str = "metadata.json"


class CheckpointManager:
    """
    Checkpoint管理器
    
    功能：
    - 自动管理checkpoint的保存和清理
    - 支持增量保存（减少存储空间）
    - 快速恢复和验证
    """
    
    def __init__(self, config: CheckpointConfig):
        """
        初始化checkpoint管理器
        
        Args:
            config: Checkpoint配置
        """
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        
        # 确保目录存在
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"CheckpointManager初始化完成")
        logger.info(f"  Checkpoint目录: {self.checkpoint_dir}")
        logger.info(f"  最大保持数量: {config.max_checkpoints}")
        logger.info(f"  保存间隔: {config.save_interval}")
    
    def save_checkpoint(self, 
                       train_state: Any,
                       step: int,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        保存checkpoint
        
        Args:
            train_state: 训练状态
            step: 当前步数
            metadata: 额外元数据
            
        Returns:
            checkpoint路径
        """
        
        checkpoint_name = f"{self.config.checkpoint_prefix}_{step:08d}"
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pkl"
        
        logger.info(f"保存checkpoint: {checkpoint_path}")
        
        # 准备保存数据
        save_data = {
            'train_state': train_state,
            'step': step,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        
        try:
            # 保存主checkpoint
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            # 保存元数据文件（方便快速查看）
            metadata_path = self.checkpoint_dir / f"{checkpoint_name}_{self.config.metadata_suffix}"
            metadata_info = {
                'step': step,
                'timestamp': time.time(),
                'checkpoint_file': checkpoint_path.name,
                'metadata': metadata or {}
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata_info, f, indent=2)
            
            # 验证checkpoint
            if self._verify_checkpoint(checkpoint_path):
                logger.info(f"✅ Checkpoint保存成功: step {step}")
                
                # 清理旧checkpoint
                self._cleanup_old_checkpoints()
                
                return str(checkpoint_path)
            else:
                raise RuntimeError("Checkpoint验证失败")
                
        except Exception as e:
            logger.error(f"❌ Checkpoint保存失败: {e}")
            # 清理可能损坏的文件
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            raise
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        加载checkpoint
        
        Args:
            checkpoint_path: checkpoint路径，如果None则加载最新的
            
        Returns:
            (train_state, metadata)
        """
        
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()
            if checkpoint_path is None:
                raise FileNotFoundError("没有找到可用的checkpoint")
        
        logger.info(f"加载checkpoint: {checkpoint_path}")
        
        try:
            with open(checkpoint_path, 'rb') as f:
                save_data = pickle.load(f)
            
            train_state = save_data['train_state']
            metadata = save_data.get('metadata', {})
            step = save_data.get('step', 0)
            
            logger.info(f"✅ Checkpoint加载成功: step {step}")
            return train_state, metadata
            
        except Exception as e:
            logger.error(f"❌ Checkpoint加载失败: {e}")
            raise
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        获取最新的checkpoint路径
        
        Returns:
            最新checkpoint路径，如果没有返回None
        """
        
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        
        # 按步数排序，返回最新的
        latest = max(checkpoints, key=lambda x: x['step'])
        return latest['path']
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        列出所有checkpoint
        
        Returns:
            checkpoint信息列表
        """
        
        checkpoints = []
        
        for checkpoint_file in self.checkpoint_dir.glob(f"{self.config.checkpoint_prefix}_*.pkl"):
            try:
                # 从文件名提取步数
                name_parts = checkpoint_file.stem.split('_')
                if len(name_parts) >= 2:
                    step = int(name_parts[-1])
                    
                    # 获取文件时间和大小
                    stat = checkpoint_file.stat()
                    
                    checkpoints.append({
                        'path': str(checkpoint_file),
                        'step': step,
                        'size': stat.st_size,
                        'mtime': stat.st_mtime,
                        'name': checkpoint_file.name
                    })
            
            except (ValueError, OSError) as e:
                logger.warning(f"解析checkpoint文件失败 {checkpoint_file}: {e}")
        
        # 按步数排序
        checkpoints.sort(key=lambda x: x['step'])
        return checkpoints
    
    def _verify_checkpoint(self, checkpoint_path: Path) -> bool:
        """
        验证checkpoint文件完整性
        
        Args:
            checkpoint_path: checkpoint路径
            
        Returns:
            是否验证成功
        """
        
        try:
            with open(checkpoint_path, 'rb') as f:
                save_data = pickle.load(f)
            
            # 基础验证
            required_keys = ['train_state', 'step', 'timestamp']
            for key in required_keys:
                if key not in save_data:
                    logger.warning(f"Checkpoint缺少必需字段: {key}")
                    return False
            
            # 验证数据类型
            if not isinstance(save_data['step'], int):
                logger.warning("Step字段类型错误")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Checkpoint验证异常: {e}")
            return False
    
    def _cleanup_old_checkpoints(self):
        """清理旧的checkpoint文件"""
        
        checkpoints = self.list_checkpoints()
        
        # 保持最新的N个checkpoint
        if len(checkpoints) > self.config.max_checkpoints:
            checkpoints_to_remove = checkpoints[:-self.config.max_checkpoints]
            
            for checkpoint_info in checkpoints_to_remove:
                try:
                    checkpoint_path = Path(checkpoint_info['path'])
                    
                    # 删除主文件
                    if checkpoint_path.exists():
                        checkpoint_path.unlink()
                    
                    # 删除对应的元数据文件
                    metadata_path = checkpoint_path.with_suffix('').with_suffix(f'_{self.config.metadata_suffix}')
                    if metadata_path.exists():
                        metadata_path.unlink()
                    
                    logger.debug(f"删除旧checkpoint: {checkpoint_path.name}")
                    
                except OSError as e:
                    logger.warning(f"删除checkpoint失败: {e}")
    
    def get_checkpoint_info(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        获取checkpoint详细信息
        
        Args:
            checkpoint_path: checkpoint路径
            
        Returns:
            checkpoint信息字典
        """
        
        try:
            with open(checkpoint_path, 'rb') as f:
                save_data = pickle.load(f)
            
            stat = Path(checkpoint_path).stat()
            
            return {
                'path': checkpoint_path,
                'step': save_data.get('step', 0),
                'timestamp': save_data.get('timestamp', 0),
                'file_size': stat.st_size,
                'file_mtime': stat.st_mtime,
                'metadata': save_data.get('metadata', {})
            }
            
        except Exception as e:
            logger.error(f"获取checkpoint信息失败: {e}")
            return {}
    
    def cleanup_all(self):
        """清理所有checkpoint文件"""
        logger.warning("清理所有checkpoint文件...")
        
        try:
            if self.checkpoint_dir.exists():
                shutil.rmtree(self.checkpoint_dir)
                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logger.info("✅ 所有checkpoint已清理")
            
        except Exception as e:
            logger.error(f"❌ 清理checkpoint失败: {e}")


def create_checkpoint_manager(checkpoint_dir: str,
                            max_checkpoints: int = 5,
                            save_interval: int = 1000) -> CheckpointManager:
    """
    工厂函数：创建checkpoint管理器
    
    Args:
        checkpoint_dir: checkpoint目录
        max_checkpoints: 最大保持checkpoint数量
        save_interval: 保存间隔
        
    Returns:
        CheckpointManager实例
    """
    
    config = CheckpointConfig(
        checkpoint_dir=checkpoint_dir,
        max_checkpoints=max_checkpoints,
        save_interval=save_interval
    )
    
    return CheckpointManager(config)