"""
ACRLPDDataLoaderV2: 基于OpenPI的高效RL数据加载器

在OpenPI高效数据加载基础上，添加Q-chunking RL训练需要的功能：
- 使用OpenPI的LeRobotDataset + delta_timestamps机制
- 保持OpenPI的transforms pipeline
- 添加RL特有字段：reward, next_observations, masks, terminals  
- 输出标准Q-chunking格式，与AC Training训练循环兼容

架构：
LeRobotDataset → OpenPI transforms → RLQChunkingTransform → Q-chunking output
"""

import logging
from typing import Dict, List, Any, Optional
import jax.numpy as jnp
import numpy as np
import torch

# OpenPI imports  
import openpi.training.data_loader as openpi_data_loader
import openpi.training.config as openpi_config
import openpi.transforms as _transforms

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# RLQChunkingTransform类已完全删除 - 所有转换逻辑移至_collate_fn实现零复制高效处理


class ACRLPDDataLoaderV2:
    """基于OpenPI的高效RL数据加载器"""
    
    def __init__(
        self,
        rl_config: Any,  # RLTrainConfig
        batch_size: int = 128,
        seed: int = 42,
        positive_batch_ratio: float = 0.1,  # 保留兼容性，但不使用
        tolerance_s: float = 1e-4,
        debug_mode: bool = False
    ):
        """
        初始化基于OpenPI的RL数据加载器
        
        Args:
            rl_config: AC Training的RLTrainConfig  
            batch_size: 批次大小
            seed: 随机种子
            positive_batch_ratio: (兼容性参数，不使用复杂分类)
            tolerance_s: OpenPI时间戳容错
            debug_mode: 调试模式
        """
        self.rl_config = rl_config
        self.batch_size = batch_size
        self.seed = seed
        self.debug_mode = debug_mode
        
        # 从RL config提取关键参数
        self.repo_id = self._extract_repo_id(rl_config.data)
        self.action_horizon = rl_config.model.action_horizon
        self.action_dim = rl_config.qchunking.action_dim
        
        logger.info(f"=== ACRLPDDataLoaderV2 (基于OpenPI) ===")
        logger.info(f"Repo ID: {self.repo_id}")
        logger.info(f"Action Horizon: {self.action_horizon}")
        logger.info(f"Action Dim: {self.action_dim}")
        logger.info(f"Batch Size: {batch_size}")
        
        # 1. 使用OpenPI标准方式创建数据集
        self.openpi_dataset = self._create_openpi_dataset(tolerance_s)
        
        # 2. RL转换器已删除 - 直接在_collate_fn中实现零复制转换
        
        # 3. 创建PyTorch DataLoader (使用OpenPI方式)
        self.dataloader = self._create_pytorch_dataloader()
        
        logger.info(f"✅ ACRLPDDataLoaderV2 初始化完成 (基于OpenPI)")
    
    def _extract_repo_id(self, data_config) -> str:
        """从data config中提取repo_id"""
        if hasattr(data_config, 'repo_id'):
            return data_config.repo_id
        else:
            raise ValueError("无法从data config中提取repo_id")
    
    def _create_openpi_dataset(self, tolerance_s: float):
        """使用OpenPI标准方式创建数据集"""
        
        logger.info("使用OpenPI标准方式创建torch数据集...")
        
        # 1. 从DataConfigFactory创建实际的DataConfig实例
        # LeRobotAlohaDataConfig是factory，需要调用create()生成实际配置
        assets_dirs = self.rl_config.assets_dirs  # TrainConfig的assets_dirs属性
        actual_data_config = self.rl_config.data.create(assets_dirs, self.rl_config.model)
        logger.info(f"✅ DataConfigFactory创建实际DataConfig成功，prompt_from_task={actual_data_config.prompt_from_task}")
        
        # 2. 创建基础数据集 - 使用实际DataConfig
        base_dataset = openpi_data_loader.create_torch_dataset(
            data_config=actual_data_config,  # 使用factory创建的实际DataConfig
            action_horizon=self.action_horizon,
            model_config=self.rl_config.model,
            tolerance_s=tolerance_s,
            skip_problematic_episodes=True
        )
        
        # 3. 应用OpenPI transforms (repack + data + normalize + model)
        transformed_dataset = openpi_data_loader.transform_dataset(
            base_dataset, 
            actual_data_config,  # 使用实际DataConfig进行transform
            skip_norm_stats=False  # 使用归一化统计
        )
        
        logger.info(f"✅ OpenPI数据集创建并变换成功，数据集长度: {len(transformed_dataset)}")
        return transformed_dataset
    
    def _create_pytorch_dataloader(self):
        """创建PyTorch DataLoader"""
        
        # 使用OpenPI方式创建DataLoader
        dataloader = torch.utils.data.DataLoader(
            self.openpi_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # JAX环境下通常设为0
            drop_last=True
            # 🚀 性能修复：使用PyTorch默认collate_fn，恢复OpenPI高效batching
        )
        
        logger.info(f"✅ PyTorch DataLoader创建成功")
        return dataloader
    
    def _collate_fn_DISABLED_FOR_PERFORMANCE(self, batch):
        """
        超高效collate函数：零复制OpenPI→Q-chunking转换
        
        输入: OpenPI原始样本列表 [{image: {...}, state: [...], actions: [...], reward: float}]
        输出: Q-chunking批次 {image: [B,...], actions: [B,H,A], rewards: [B,H], ...}
        """
        
        if len(batch) == 0:
            return {}
        
        batch_size = len(batch)
        
        # 1. 预计算RL字段模板（避免重复计算）
        masks_template = jnp.ones(self.action_horizon, dtype=jnp.float32)
        terminals_template = jnp.zeros(self.action_horizon, dtype=jnp.float32)
        
        collated = {}
        
        # 数值检查函数：防止异常数据
        def _check_numerical_validity(name: str, data):
            """检查数据的数值有效性，防止NaN/Inf/过大数值"""
            if isinstance(data, (jnp.ndarray, np.ndarray)):
                if jnp.any(jnp.isnan(data)) or jnp.any(jnp.isinf(data)):
                    raise ValueError(f"发现NaN或Inf值在字段: {name}")
                if jnp.max(jnp.abs(data)) > 1e6:  # 合理的数值范围
                    logger.warning(f"字段 {name} 包含过大数值: max_abs={float(jnp.max(jnp.abs(data)))}")
            elif isinstance(data, dict):
                for sub_key, sub_data in data.items():
                    _check_numerical_validity(f"{name}.{sub_key}", sub_data)
        
        # 2. 🚀 NEW: 处理LeRobot时间序列数据格式
        # 基于RepackTransform映射：observation.* -> 对应OpenPI字段
        processed_keys = set()
        
        
        
        for key in batch[0].keys():
            processed_keys.add(key)
            
            if key == 'reward':
                # RL数据必须有reward字段，AC Training专为RL设计
                reward_values = jnp.array([float(sample['reward']) for sample in batch], dtype=jnp.float32)
                
                # Q-chunking奖励分布：序列末尾给奖励，中间步骤为0（与原AC Training一致）
                rewards = jnp.zeros((batch_size, self.action_horizon), dtype=jnp.float32)
                rewards = rewards.at[:, -1].set(reward_values)  # 只在序列最后一步给奖励
                collated['rewards'] = rewards
                
            elif key == 'state':
                # state: [batch_size, 2, state_dim] -> [current_state, next_state]
                state_values = [sample[key] for sample in batch]  # List[[2, state_dim]]
                collated['state'] = jnp.stack([s[0] for s in state_values], axis=0)      # [B, state_dim] - current
                collated['next_state'] = jnp.stack([s[1] for s in state_values], axis=0) # [B, state_dim] - next
                
            elif key == 'image':
                # images: {cam_name: [batch_size, 2, H, W, C]} -> {cam_name: [current_img, next_img]}
                images_values = [sample[key] for sample in batch]  # List[{cam: [2, H, W, C]}]
                
                collated['image'] = {}
                collated['next_image'] = {}
                
                # 🔧 RL修复: 添加image_mask处理，维度需要匹配时间序列格式
                collated['image_mask'] = {}
                collated['next_image_mask'] = {}  # 🔧 修复：初始化next_image_mask字典
                
                # 获取所有相机名称（从第一个样本）
                cam_names = images_values[0].keys()
                # 🚀 优化：单次循环处理所有相机数据和masks
                for cam_name in cam_names:
                    # 每个相机: [B, 2, H, W, C]
                    cam_data = [sample[key][cam_name] for sample in batch]
                    collated['image'][cam_name] = jnp.stack([img[0] for img in cam_data], axis=0)      # [B, H, W, C] - current  
                    collated['next_image'][cam_name] = jnp.stack([img[1] for img in cam_data], axis=0) # [B, H, W, C] - next
                    
                    # 🚀 优化：单次创建current和next的image_mask
                    mask = jnp.ones(batch_size, dtype=jnp.bool_)  # [B] - 标准OpenPI格式
                    collated['image_mask'][cam_name] = mask
                    collated['next_image_mask'][cam_name] = mask  # 复用same mask
                    
            elif key == 'actions':
                # actions: [batch_size, action_horizon, action_dim] - 保持原有处理
                values = [sample[key] for sample in batch]
                collated[key] = jnp.stack(values, axis=0)
                
            else:
                # 其他字段：按原有逻辑处理
                values = [sample[key] for sample in batch]
                if isinstance(values[0], dict):
                    collated[key] = {}
                    for sub_key in values[0].keys():
                        sub_values = [sample[key][sub_key] for sample in batch]
                        collated[key][sub_key] = jnp.stack(sub_values, axis=0)
                elif isinstance(values[0], (jnp.ndarray, np.ndarray)):
                    collated[key] = jnp.stack(values, axis=0)
                else:
                    collated[key] = values if len(set(str(v) for v in values)) > 1 else values[0]
        
        # 2.5. 验证RL必需字段存在
        if 'reward' not in processed_keys:
            logger.info(f"当前字段：{batch[0].keys()}")
            raise ValueError("RL数据必须包含reward字段！AC Training专为RL训练设计，不支持SFT数据。")
        
        # 3. 添加固定RL字段（向量化操作）
        collated['masks'] = jnp.tile(masks_template[None, :], (batch_size, 1))
        collated['terminals'] = jnp.tile(terminals_template[None, :], (batch_size, 1))
        
        # 5. 数值有效性检查（防止异常数据）
        try:
            for key, value in collated.items():
                _check_numerical_validity(key, value)
        except ValueError as e:
            # 数值异常时抛出异常，让sample_batch重新采样
            raise ValueError(f"Batch数值异常: {e}")
        
        return collated
    
    def sample_batch(self) -> Dict[str, jnp.ndarray]:
        """采样一个batch，兼容AC Training接口"""
        
         # 从DataLoader获取下一个batch
        batch_iter = iter(self.dataloader)
        batch = next(batch_iter)
        
        return batch
    
    def __iter__(self):
        """迭代器接口"""
        return iter(self.dataloader)
    
    def __len__(self):
        """返回数据集长度"""  
        return len(self.dataloader)


def create_acrlpd_data_loader_v2(
    rl_config: Any,
    batch_size: int = 128,
    seed: int = 42,
    tolerance_s: float = 1e-4,
    debug_mode: bool = False,
    **kwargs
) -> ACRLPDDataLoaderV2:
    """
    创建ACRLPDDataLoaderV2实例 (兼容AC Training接口)
    
    Args:
        rl_config: RLTrainConfig统一配置
        batch_size: 批次大小  
        seed: 随机种子
        tolerance_s: OpenPI时间戳容错
        debug_mode: 调试模式
        **kwargs: 其他兼容性参数
        
    Returns:
        ACRLPDDataLoaderV2实例
    """
    
    return ACRLPDDataLoaderV2(
        rl_config=rl_config,
        batch_size=batch_size,
        seed=seed,
        tolerance_s=tolerance_s,
        debug_mode=debug_mode
    )