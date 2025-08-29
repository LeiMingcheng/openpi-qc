"""
ACRLPDDataLoader: QC-ACT架构 + OpenPI格式输出的数据加载器

基于qc_ACT的完全随机遍历架构，专为内存受限环境设计：
- 内存池预加载：每epoch随机加载N个episodes
- 完全随机采样：忽略DataLoader索引，统计性遍历全数据集
- LeRobot输入 → OpenPI格式输出
- π₀变换管道集成
- 高性能内存访问，避免频繁I/O
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Iterator, Union
import dataclasses
from enum import Enum

import jax
import jax.numpy as jnp
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize
import openpi.training.data_loader as openpi_data_loader
import openpi.training.config as openpi_config

logger = logging.getLogger(__name__)
# 确保logger能正确输出到console
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# ===============================================================================
# BACKWARD COMPATIBILITY CLASSES FOR STAGE 1
# These classes are preserved for Stage 1 (H5 → LeRobot conversion) compatibility
# ===============================================================================

class SamplingStrategy(Enum):
    """Backward compatibility: Sampling strategies for Stage 1."""
    UNIFORM = "uniform"
    BALANCED_REWARD = "balanced_reward" 
    PRIORITY_POSITIVE = "priority_positive"
    BOOTSTRAP = "bootstrap"


@dataclasses.dataclass
class SamplingConfig:
    """Backward compatibility: Sampling configuration for Stage 1."""
    strategy: SamplingStrategy = SamplingStrategy.UNIFORM
    positive_weight: float = 1.0
    negative_weight: float = 1.0
    min_positive_ratio: float = 0.3
    max_positive_ratio: float = 0.7
    bootstrap_ratio: float = 0.1


@dataclasses.dataclass 
class RewardProcessingConfig:
    """Backward compatibility: Reward processing configuration for Stage 1."""
    use_sparse_rewards: bool = True
    use_dense_rewards: bool = False
    reward_shaping: bool = False
    discount_factor: float = 0.99
    reward_scale: float = 1.0
    success_bonus: float = 10.0
    failure_penalty: float = -1.0

# ===============================================================================


# ACRLPDDataConfig已删除 - 现在直接使用RLTrainConfig


class ACRLPDDataLoader:
    """
    Q-chunking RL数据加载器（基于qc_ACT架构）
    
    核心特性：
    - Q-chunking支持：返回RL训练所需的完整transition格式
    - 动作序列生成：H步动作序列和episode边界处理
    - Bootstrap机制：next_observations用于target Q计算
    - 内存池管理：每epoch随机加载episodes到内存
    - 完全随机采样：符合强化学习要求
    - π₀兼容：Observation格式兼容OpenPI模型
    
    输出格式（Q-chunking标准）：
    {
        'observations': π₀_Observation,      # 当前观测
        'next_observations': π₀_Observation,  # Bootstrap用
        'actions': [B, H, action_dim],       # 动作序列
        'rewards': [B, H],                   # 奖励序列
        'masks': [B, H],                     # Bootstrap mask
        'valid': [B, H],                     # 动作有效性
        'terminals': [B, H],                 # 步骤终止标志
        'next_terminal': [B],                # 下一状态terminal
        'sequence_mask': [B]                 # 序列有效性
    }
    """
    
    def __init__(
        self,
        rl_config: Any,  # RLTrainConfig (避免循环导入)
        batch_size: int = 128,
        episodes_per_memory_pool: int = 64,  # 内存池大小
        shuffle: bool = True,
        seed: int = 42,
        tolerance_s: float = 1e-4,  # 时间戳容错阈值
        device_sharding: Optional[jax.sharding.Sharding] = None,
        rank: int = 0,
        world_size: int = 1,
        skip_norm_stats: bool = False  # acrlpd_data_converter 新增：跳过norm stats加载，避免循环依赖
    ):
        """
        初始化统一的ACRLPD数据加载器
        
        Args:
            rl_config: RLTrainConfig统一配置
            batch_size: 批次大小
            episodes_per_memory_pool: 内存池大小（每epoch加载的episodes数量）
            shuffle: 是否启用随机性（实际总是随机的）
            seed: 随机种子
            tolerance_s: 时间戳容错阈值（秒）
            device_sharding: JAX设备分片配置
            rank: 分布式训练GPU编号
            world_size: 总GPU数量
            skip_norm_stats: 跳过norm stats加载，用于norm stats计算场景
        """
        self.rl_config = rl_config
        self.batch_size = batch_size
        self.episodes_per_memory_pool = episodes_per_memory_pool
        self.seed = seed
        self.tolerance_s = tolerance_s
        self.rank = rank
        self.world_size = world_size
        self.skip_norm_stats = skip_norm_stats
        
        # 从统一配置中提取关键参数
        self.qchunking_config = rl_config.qchunking
        self.acrlpd_config = rl_config.acrlpd
        
        # 从OpenPI数据配置中获取repo_id
        data_config_factory = rl_config.data
        if hasattr(data_config_factory, 'repo_id'):
            self.repo_id = data_config_factory.repo_id
        else:
            # 对于复杂的DataConfigFactory，需要创建后获取
            temp_data_config = data_config_factory.create(
                assets_dirs=rl_config.assets_dirs,
                model_config=rl_config.model
            )
            self.repo_id = temp_data_config.repo_id
        
        if not self.repo_id:
            raise ValueError("Cannot determine repo_id from RLTrainConfig.data")
        
        # qc_ACT风格的内存池
        self.memory_pool_episodes = []      # 当前内存池中的完整episodes数据
        self.memory_pool_lengths = []       # 每个episode的长度（transition数量）
        self.total_pool_transitions = 0     # 内存池中总transition数
        
        # 全局数据集信息
        self.all_episode_info = []          # 所有episodes的信息（边界等）
        self.current_epoch_seed = 0         # 当前epoch种子
        
        # 随机状态
        self.rng = np.random.RandomState(seed)
        
        # 设备分片
        if device_sharding is None:
            device_sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B")
            )
        self.device_sharding = device_sharding
        
        # 初始化步骤
        print(f"acrlpd_data_converter DATA LOADER DEBUG: repo_id={self.repo_id}")
        print(f"   - episodes_per_memory_pool={episodes_per_memory_pool}")
        print(f"   - batch_size={batch_size}")
        logger.info(f"初始化统一ACRLPD数据加载器: repo_id={self.repo_id}, "
                   f"episodes_per_memory_pool={episodes_per_memory_pool}, batch_size={batch_size}")
        
        self._discover_all_episodes()       # 发现所有episodes
        self._setup_transforms()            # 设置OpenPI变换管道
        self._load_current_memory_pool()    # 加载初始内存池
        
        # 统计信息
        self.stats = {
            'repo_id': self.repo_id,
            'total_episodes': len(self.all_episode_info),
            'episodes_per_memory_pool': episodes_per_memory_pool,
            'current_pool_transitions': self.total_pool_transitions,
            'batches_served': 0,
            'current_epoch': 0,
            'config_type': 'RLTrainConfig (unified)'
        }
        
        logger.info(f"✓ 统一数据加载器初始化完成: 发现{len(self.all_episode_info)}个episodes, "
                   f"内存池容量{episodes_per_memory_pool}, 当前加载{len(self.memory_pool_episodes)}个episodes, "
                   f"总计{self.total_pool_transitions}个transitions")
    
    def _discover_all_episodes(self):
        """发现LeRobot数据集中的所有episodes"""
        
        # 获取数据集元数据
        try:
            self.dataset_meta = LeRobotDatasetMetadata(self.repo_id, local_files_only=True)
        except TypeError:
            # 如果LeRobotDatasetMetadata不支持local_files_only，回退到默认方式
            self.dataset_meta = LeRobotDatasetMetadata(self.repo_id)
        
        # acrlpd_data_converter DEBUG: Print the repo_id being used to create LeRobot dataset
        print(f"acrlpd_data_converter LEROBOT DATASET: Creating dataset with repo_id='{self.repo_id}'")
        print(f"   - Type: {type(self.repo_id)}")
        print(f"   - Absolute path exists: {Path(self.repo_id).exists() if not self.repo_id.startswith('/') else 'checking...'}")
        
        # acrlpd_data_converter 修复1：强制禁用torchcodec，确保使用pyav/torchvision
        import lerobot.common.datasets.video_utils as video_utils
        
        if not hasattr(video_utils, '_openpi_patched'):
            logger.info("acrlpd_data_converter 应用PyAV/torchvision patch以修复torchcodec问题")
            original_decode_video_frames = video_utils.decode_video_frames
            
            def patched_decode_video_frames(video_path, timestamps, tolerance_s, backend="pyav"):
                # 使用torchvision+pyav backend，避免torchcodec问题
                try:
                    return video_utils.decode_video_frames_torchvision(video_path, timestamps, tolerance_s, backend="pyav")
                except Exception as e:
                    logger.error(f"PyAV/torchvision解码失败: {e}")
                    raise e
            
            video_utils.decode_video_frames = patched_decode_video_frames
            video_utils._openpi_patched = True
        
        # acrlpd_data_converter 修复2：torch.stack(Column)兼容性问题 - 经过验证的修复
        import torch
        if not hasattr(torch, '_openpi_column_patch'):
            logger.info("acrlpd_data_converter 应用torch.stack Column兼容性patch")
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
        
        # 创建LeRobot数据集 - 添加参数跳过有问题的episodes
        self.lerobot_dataset = lerobot_dataset.LeRobotDataset(
            self.repo_id,
            delta_timestamps={
                key: [t / self.dataset_meta.fps for t in range(self.qchunking_config.horizon_length)] 
                for key in ["action"]  # 简化为标准action key
            },
            tolerance_s=self.tolerance_s,  # 使用可配置的tolerance
            video_backend="pyav",  # 与OpenPI保持一致
            skip_problematic_episodes=True  # 跳过有时间同步问题的episodes
        )
        
        # acrlpd_data_converter DEBUG: Print dataset info after creation
        print(f"acrlpd_data_converter LEROBOT DATASET CREATED:")
        print(f"   - Dataset repo_id: {getattr(self.lerobot_dataset, 'repo_id', 'N/A')}")
        print(f"   - Dataset root: {getattr(self.lerobot_dataset, 'root', 'N/A')}")
        print(f"   - Dataset length: {len(self.lerobot_dataset)}")
        if hasattr(self.lerobot_dataset, 'episode_data_index'):
            print(f"   - Episode data index keys: {len(self.lerobot_dataset.episode_data_index)} episodes")
        
        # 提取episode边界信息
        if hasattr(self.lerobot_dataset, 'episode_data_index'):
            episode_index = self.lerobot_dataset.episode_data_index
            
            print(f"acrlpd_data_converter EPISODE_DATA_INDEX DEBUG:")
            print(f"   - Type: {type(episode_index)}")
            print(f"   - Length: {len(episode_index)}")
            print(f"   - Keys: {list(episode_index.keys())}...")
            for key, value in list(episode_index.items())[:1]:
                print(f"   - Sample: key={key} (type: {type(key)})")
                print(f"           value={value} (type: {type(value)})")
            
            # acrlpd_data_converter FIXED: Handle correct LeRobot episode_data_index format
            if 'from' in episode_index and 'to' in episode_index:
                # Correct LeRobot format: {'from': tensor([starts...]), 'to': tensor([ends...])}
                starts = episode_index['from']
                ends = episode_index['to']
                
                print(f"acrlpd_data_converter CORRECTED EPISODE PARSING:")
                print(f"   - Episodes found: {len(starts)}")
                print(f"   - Total transitions: {ends[-1].item() if len(ends) > 0 else 0}")
                print(f"   - First 3 episodes: starts={starts[:3].tolist()}, ends={ends[:3].tolist()}")
                
                # Create episode info for each episode
                for i in range(len(starts)):
                    episode_start = int(starts[i].item())
                    episode_end = int(ends[i].item())
                    episode_length = episode_end - episode_start
                    
                    if episode_length > 0:
                        self.all_episode_info.append({
                            'episode_id': i,
                            'start_idx': episode_start,
                            'end_idx': episode_end,
                            'length': episode_length
                        })
                        
            else:
                # Fallback: Legacy format handling (for compatibility)
                logger.warning("Using legacy episode_data_index format")
                for episode_id, episode_info in episode_index.items():
                    try:
                        if hasattr(episode_info, '__getitem__') and isinstance(episode_info, dict):
                            episode_start = episode_info['from']
                            episode_end = episode_info['to']
                        elif hasattr(episode_info, 'from_'):
                            episode_start = episode_info.from_
                            episode_end = episode_info.to
                        elif torch.is_tensor(episode_info) and episode_info.numel() >= 2:
                            episode_start = episode_info[0]
                            episode_end = episode_info[1]
                        else:
                            logger.warning(f"Unknown episode_info structure for episode {episode_id}: {type(episode_info)}")
                            continue
                        
                        # 转换为Python int
                        if torch.is_tensor(episode_start):
                            episode_start = int(episode_start.item())
                        if torch.is_tensor(episode_end):
                            episode_end = int(episode_end.item())
                        
                        episode_start = int(episode_start)
                        episode_end = int(episode_end)
                        episode_length = episode_end - episode_start
                        
                        if episode_length > 0:
                            self.all_episode_info.append({
                                'episode_id': episode_id,
                                'start_idx': episode_start,
                                'end_idx': episode_end,
                                'length': episode_length
                            })
                            
                    except Exception as e:
                        logger.warning(f"Failed to process episode {episode_id}: {e}")
                        continue
        else:
            # 备用方案：尝试通过遍历检测episode边界
            logger.warning("No episode_data_index found, using fallback detection")
            # 这里可以实现备用的episode检测逻辑
            # 暂时假设整个数据集是一个大episode
            dataset_len = len(self.lerobot_dataset)
            self.all_episode_info.append({
                'episode_id': 0,
                'start_idx': 0,
                'end_idx': dataset_len,
                'length': dataset_len
            })
        
        total_transitions = sum(ep['length'] for ep in self.all_episode_info)
        print(f"acrlpd_data_converter EPISODE DISCOVERY RESULTS:")
        print(f"   - Total episodes found: {len(self.all_episode_info)}")
        print(f"   - Total transitions: {total_transitions}")
        print(f"   - First 3 episodes:")
        for i, ep_info in enumerate(self.all_episode_info[:3]):
            print(f"     Episode {i}: ID={ep_info.get('episode_id', 'N/A')}, length={ep_info.get('length', 'N/A')}")
        
        logger.info(f"✓ 发现 {len(self.all_episode_info)} 个episodes，总计 {total_transitions} 个transitions")
        
        if len(self.all_episode_info) == 0:
            raise RuntimeError("No episodes found in dataset")
    
    def _setup_transforms(self):
        """设置OpenPI变换管道（延迟加载优化版本）"""
        
        print(f"acrlpd_data_converter TRANSFORM SETUP: Starting lazy initialization...")
        
        # 延迟data_config创建，仅存储factory
        self.data_config_factory = self.rl_config.data
        self.data_config = None
        print(f"acrlpd_data_converter TRANSFORM SETUP: Stored data config factory for lazy loading")
        
        # 预设transform为None，真正需要时才加载
        self.repack_transforms = None
        self.data_transforms = None  
        self.model_transforms = None
        self.norm_stats = None
        
        # 标记是否已初始化transforms
        self._transforms_initialized = False
        
        logger.info("✓ Transform系统延迟初始化完成（优化版本）")
    
    def _ensure_transforms_initialized(self):
        """延迟加载：仅在真正需要时才初始化transforms"""
        
        if self._transforms_initialized:
            return
            
        print(f"acrlpd_data_converter LAZY LOADING: First-time transform initialization...")
        
        # 创建实际的DataConfig
        try:
            print(f"acrlpd_data_converter LAZY LOADING: Creating data config...")
            
            self.data_config = self.data_config_factory.create(
                assets_dirs=self.rl_config.assets_dirs,
                model_config=self.rl_config.model
            )
            print(f"acrlpd_data_converter LAZY LOADING: Data config created successfully")
        except Exception as e:
            print(f"ERROR: data_config创建失败: {e}")
            import traceback
            traceback.print_exc()
            raise e
        
        # 提取变换管道
        print(f"acrlpd_data_converter LAZY LOADING: Extracting transform pipelines...")
        self.repack_transforms = self.data_config.repack_transforms
        self.data_transforms = self.data_config.data_transforms
        self.model_transforms = self.data_config.model_transforms
        
        # 归一化统计信息
        if self.skip_norm_stats:
            print(f"acrlpd_data_converter LAZY LOADING: Skipping norm_stats loading (norm computation mode)")
            self.norm_stats = None
        else:
            print(f"acrlpd_data_converter LAZY LOADING: Loading norm_stats...")
            try:
                self.norm_stats = self.data_config.norm_stats
                print(f"acrlpd_data_converter LAZY LOADING: Got norm_stats successfully")
            except Exception as e:
                print(f"acrlpd_data_converter LAZY LOADING: Failed to load norm_stats: {e}")
                print(f"acrlpd_data_converter LAZY LOADING: Setting norm_stats to None for now...")
                self.norm_stats = None
        
        self._transforms_initialized = True
        print(f"acrlpd_data_converter LAZY LOADING: Transform initialization completed")
        logger.info("✓ Transform系统延迟加载完成")
    
    def _load_current_memory_pool(self):
        """qc_ACT风格：随机加载episodes_per_memory_pool个episodes到内存池（性能优化版本）"""
        
        if self.rank == 0:
            logger.info(f" 加载内存池 (epoch {self.current_epoch_seed})")
        
        total_start_time = time.time()
        step_times = {}  # 详细计时各个步骤
        
        # Step 1: Episode选择
        step_start = time.time()
        # 基于epoch种子和GPU rank创建独立随机状态
        pool_rng = np.random.RandomState(self.current_epoch_seed + self.rank * 1000)
        
        # 随机选择episodes
        if len(self.all_episode_info) >= self.episodes_per_memory_pool:
            selected_indices = pool_rng.choice(
                len(self.all_episode_info), 
                size=self.episodes_per_memory_pool, 
                replace=False  # 不重复选择
            )
        else:
            # 如果总episode数不足，全部加载
            selected_indices = list(range(len(self.all_episode_info)))
            logger.warning(f"总episodes数({len(self.all_episode_info)})小于内存池大小({self.episodes_per_memory_pool})，全部加载")
        
        step_times['episode_selection'] = time.time() - step_start
        
        # Step 2: 并行加载episodes（Phase 2A优化）
        step_start = time.time()
        
        # 清空旧内存池
        self.memory_pool_episodes.clear()
        self.memory_pool_lengths.clear()
        
        # acrlpd_data_converter Phase 2A: 并行批量加载episodes
        loaded_episodes = 0
        failed_episodes = 0
        
        # 使用并行加载替代串行加载
        episode_results = self._parallel_load_episodes(selected_indices)
        
        # 处理加载结果
        for episode_data in episode_results:
            if episode_data is not None and len(episode_data) > 0:
                self.memory_pool_episodes.append(episode_data)
                self.memory_pool_lengths.append(len(episode_data))
                loaded_episodes += 1
            else:
                failed_episodes += 1
        
        step_times['parallel_loading'] = time.time() - step_start
        
        self.total_pool_transitions = sum(self.memory_pool_lengths)
        total_load_time = time.time() - total_start_time
        
        if self.rank == 0:
            logger.info(f"✓ 内存池加载完成 (epoch {self.current_epoch_seed}): "
                       f"成功加载{loaded_episodes}个episodes, 失败{failed_episodes}个, "
                       f"总计{self.total_pool_transitions}个transitions, 耗时{total_load_time:.2f}s")
            
            # acrlpd_data_converter 详细性能分析日志
            logger.info(f"acrlpd_data_converter Phase 2A性能分析:")
            for step_name, duration in step_times.items():
                percentage = (duration / total_load_time) * 100
                logger.info(f"   - {step_name}: {duration:.2f}s ({percentage:.1f}%)")
            
            # 性能指标
            avg_time_per_episode = total_load_time / len(selected_indices) if len(selected_indices) > 0 else 0
            avg_time_per_transition = total_load_time / self.total_pool_transitions if self.total_pool_transitions > 0 else 0
            logger.info(f"📈 性能指标: 平均每episode {avg_time_per_episode:.2f}s, 平均每transition {avg_time_per_transition*1000:.1f}ms")
        
        if self.total_pool_transitions == 0:
            raise RuntimeError("Memory pool is empty after loading")
    
    def _parallel_load_episodes(self, selected_indices):
        """acrlpd_data_converter Phase 2A: 并行加载多个episodes"""
        import concurrent.futures
        import time
        
        # acrlpd_data_converter 高性能服务器优化：充分利用64+核心和充足内存
        # 根据episode数量和服务器能力选择最优并行度
        total_episodes = len(selected_indices)
        max_workers = min(64, total_episodes)  # 大数据集：使用最多64线程


        episode_results = []
        
        if max_workers > 1 and len(selected_indices) > 1:
            # acrlpd_data_converter 高性能并行加载
            logger.info(f"acrlpd_data_converter 启动{max_workers}线程并行加载{total_episodes}个episodes...")
            parallel_start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有加载任务
                future_to_idx = {}
                for episode_idx in selected_indices:
                    episode_info = self.all_episode_info[episode_idx]
                    future = executor.submit(self._load_episode_optimized, episode_info)
                    future_to_idx[future] = episode_idx
                
                # 收集结果（保持顺序）并显示进度
                results_dict = {}
                completed_episodes = 0
                total_samples = 0
                
                for future in concurrent.futures.as_completed(future_to_idx):
                    episode_idx = future_to_idx[future]
                    try:
                        episode_data = future.result()
                        results_dict[episode_idx] = episode_data
                        if episode_data:
                            total_samples += len(episode_data)
                    except Exception as e:
                        logger.warning(f"加载episode {episode_idx}失败: {e}")
                        results_dict[episode_idx] = None
                    
                    completed_episodes += 1
                    
                    # 每完成10%显示详细进度和性能指标
                    if completed_episodes % max(1, total_episodes // 10) == 0 or completed_episodes == total_episodes:
                        elapsed = time.time() - parallel_start_time
                        progress_pct = (completed_episodes / total_episodes) * 100
                        avg_time_per_episode = elapsed / completed_episodes
                        eta = avg_time_per_episode * (total_episodes - completed_episodes)
                        episodes_per_sec = completed_episodes / max(elapsed, 0.001)
                        
                        logger.info(f"acrlpd_data_converter 进度: {completed_episodes}/{total_episodes} ({progress_pct:.1f}%), "
                                  f"累计{total_samples}个samples, "
                                  f"用时{elapsed:.1f}s, ETA:{eta:.1f}s, "
                                  f"速度{episodes_per_sec:.2f}episodes/s, 平均{avg_time_per_episode:.2f}s/episode")
                
                # 按原始顺序返回结果
                episode_results = [results_dict.get(idx, None) for idx in selected_indices]
                
                # acrlpd_data_converter 最终性能总结
                total_parallel_time = time.time() - parallel_start_time
                success_count = sum(1 for r in episode_results if r is not None)
                failure_count = total_episodes - success_count
                final_sample_count = sum(len(r) for r in episode_results if r is not None)
                
                logger.info(f"acrlpd_data_converter 并行加载完成: {success_count}/{total_episodes}成功 ({failure_count}失败), "
                          f"总计{final_sample_count}个samples, "
                          f"总耗时{total_parallel_time:.2f}s, "
                          f"最终速度{success_count/max(total_parallel_time,0.001):.2f}episodes/s, "
                          f"样本速度{final_sample_count/max(total_parallel_time,0.001):.1f}samples/s")
        else:
            # 串行加载（fallback）
            for episode_idx in selected_indices:
                episode_info = self.all_episode_info[episode_idx]
                try:
                    episode_data = self._load_episode_optimized(episode_info)
                    episode_results.append(episode_data)
                except Exception as e:
                    logger.warning(f"串行加载episode {episode_idx}失败: {e}")
                    episode_results.append(None)
        
        return episode_results
    
    def _load_episode_optimized(self, episode_info: dict) -> List[dict]:
        """acrlpd_data_converter Phase 2A: 优化的单个episode加载方法"""
        episode_data = []
        start_idx = episode_info['start_idx']
        end_idx = episode_info['end_idx']
        
        # acrlpd_data_converter 批量访问优化：尝试批量加载整个episode
        try:
            batch_indices = list(range(start_idx, end_idx))
            
            # acrlpd_data_converter 深入探索LeRobot批量加载的可能性
            # 虽然列表索引不支持，但可能有其他批量访问方法
            
            batch_loaded_data = None
            
            # 方法1: 深入探索LeRobot数据集的内部结构
            try:
                # 首次运行时详细分析数据集结构
                if not hasattr(self, '_dataset_structure_analyzed'):
                    self._dataset_structure_analyzed = True
                    
                    # 分析所有可能的数据访问属性
                    potential_data_attrs = []
                    for attr_name in ['hf_dataset', '_hf_dataset', 'dataset', '_dataset', 
                                    'data', '_data', 'episodes', '_episodes']:
                        if hasattr(self.lerobot_dataset, attr_name):
                            attr_obj = getattr(self.lerobot_dataset, attr_name)
                            potential_data_attrs.append((attr_name, type(attr_obj).__name__))
                    
                    logger.info(f"acrlpd_data_converter LeRobot内部数据属性: {potential_data_attrs}")
                
                # acrlpd_data_converter 关键修复：使用LeRobotDataset的正确__getitem__方法而不是绕过它
                # 这确保了视频解码逻辑被正确执行
                try:
                    # acrlpd_data_converter 高性能优化加载：减少视频解码开销 + 详细计时
                    episode_id = episode_info.get('episode_id', 'unknown')
                    episode_start_time = time.time()
                    episode_length = len(batch_indices)
                    
                    batch_loaded_data = []
                    logger.info(f"acrlpd_data_converter Episode {episode_id}:  {episode_length}帧")
                    # 批量加载并计时
                    load_count = 0
                    for idx in batch_indices:
                        try:
                            # 关键：使用LeRobotDataset的__getitem__方法，触发视频解码
                            sample = self.lerobot_dataset[idx]
                            batch_loaded_data.append(sample)
                            load_count += 1
                        except Exception as e:
                            logger.debug(f"LeRobotDataset[{idx}]加载失败: {e}")
                            continue
                    
                    episode_load_time = time.time() - episode_start_time
                    fps = load_count / max(episode_load_time, 0.001)
                    logger.info(f"acrlpd_data_converter Episode {episode_id}: 加载{load_count}帧, 耗时{episode_load_time:.2f}s, 速度{fps:.1f}帧/s")
                except Exception as e:
                    logger.debug(f"LeRobotDataset.__getitem__批量加载失败: {e}")
                    batch_loaded_data = None
                                
            except Exception as e:
                logger.debug(f"底层数据集批量访问失败: {e}")
            
            # 方法2: 多线程LeRobotDataset访问（如果主方法失败且数据量大）
            if batch_loaded_data is None and len(batch_indices) > 4:  # 只在足够多数据时使用多线程
                try:
                    import concurrent.futures
                    
                    def fetch_single_sample(idx):
                        # 确保使用LeRobotDataset的__getitem__方法
                        return self.lerobot_dataset[idx]
                    
                    logger.info(f"acrlpd_data_converter 使用多线程LeRobotDataset.__getitem__方法...")
                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                        future_to_idx = {executor.submit(fetch_single_sample, idx): idx for idx in batch_indices}
                        batch_loaded_data = [None] * len(batch_indices)
                        
                        for future in concurrent.futures.as_completed(future_to_idx):
                            idx = future_to_idx[future]
                            original_idx = batch_indices.index(idx)
                            try:
                                batch_loaded_data[original_idx] = future.result()
                            except Exception as e:
                                logger.warning(f"多线程样本{idx}失败: {e}")
                                batch_loaded_data[original_idx] = None
                    
                    # 过滤None值
                    batch_loaded_data = [sample for sample in batch_loaded_data if sample is not None]
                except Exception as e:
                    logger.debug(f"多线程LeRobotDataset访问失败: {e}")
                    batch_loaded_data = None
            
            # 如果批量加载成功，返回结果
            if batch_loaded_data and len(batch_loaded_data) > 0:
                return batch_loaded_data
            
            # 回退方法：逐个使用LeRobotDataset.__getitem__访问（确保视频解码）
            episode_data = [None] * len(batch_indices)
            
            for i, transition_idx in enumerate(batch_indices):
                try:
                    python_idx = int(transition_idx)
                    # 关键：使用LeRobotDataset的__getitem__方法，确保视频解码
                    sample = self.lerobot_dataset[python_idx]
                    episode_data[i] = sample
                except Exception as e:
                    logger.debug(f"LeRobotDataset[{transition_idx}]加载失败: {e}")
                    episode_data[i] = None
            
            # 过滤掉失败的transitions
            episode_data = [sample for sample in episode_data if sample is not None]
            
        except Exception as e:
            logger.warning(f"Episode {episode_info['episode_id']}批量加载失败，回退到原方法: {e}")
            # 完全回退到原来的方法
            episode_data = self._load_complete_episode_fallback(episode_info)
        
        # 显示episode加载结果
        if episode_data:
            logger.info(f"✅ Episode {episode_info['episode_id']}: {len(episode_data)} samples")
        else:
            logger.warning(f"acrlpd_data_converter Episode {episode_info['episode_id']}: 0 samples (failed)")
            
        return episode_data
    
    def _load_complete_episode_fallback(self, episode_info: dict) -> List[dict]:
        """完全回退的episode加载方法（确保使用LeRobotDataset.__getitem__）"""
        episode_data = []
        start_idx = episode_info['start_idx']
        end_idx = episode_info['end_idx']
        
        for transition_idx in range(start_idx, end_idx):
            try:
                python_idx = int(transition_idx)
                # 关键：使用LeRobotDataset的__getitem__方法，确保视频解码
                sample = self.lerobot_dataset[python_idx]
                episode_data.append(sample)
            except Exception as e:
                logger.debug(f"LeRobotDataset[{transition_idx}] in episode {episode_info['episode_id']}失败: {e}")
                continue
        
        # 显示episode加载结果（fallback方法）
        if episode_data:
            logger.info(f"✅ Episode {episode_info['episode_id']} (fallback): {len(episode_data)} samples")
        else:
            logger.warning(f"acrlpd_data_converter Episode {episode_info['episode_id']} (fallback): 0 samples (failed)")
            
        return episode_data
    
    def _load_complete_episode(self, episode_info: dict) -> List[dict]:
        """从LeRobot数据集加载完整episode数据（保持向后兼容，现在使用优化版本）"""
        # acrlpd_data_converter Phase 2A: 使用优化的加载方法
        return self._load_episode_optimized(episode_info)
    
    def sample_batch(self) -> Dict[str, jnp.ndarray]:
        """基于qc_ACT标准的Q-chunking batch生成"""
        
        if self.total_pool_transitions == 0:
            raise RuntimeError("Memory pool is empty. Call refresh_memory_pool() first.")
        
        # Step 1: 智能采样 - 确保可以构建完整action sequence
        valid_starts = self._sample_valid_sequence_starts(
            self.batch_size, 
            self.qchunking_config.horizon_length
        )
        
        # Step 2: 为每个起始点构建完整的Q-chunking transition
        batch_transitions = []
        for ep_idx, start_idx in valid_starts:
            transition = self._build_qc_transition_from_episode(
                ep_idx, start_idx, self.qchunking_config.horizon_length
            )
            batch_transitions.append(transition)
        
        # Step 3: 批次整理 - 将列表转换为JAX arrays
        batch_dict = self._collate_qc_batch(batch_transitions)
        
        # 更新统计
        self.stats['batches_served'] += 1
        
        return batch_dict
    
    def _sample_valid_sequence_starts(self, batch_size: int, horizon_length: int) -> List[Tuple[int, int]]:
        """acrlpd_data_converter CHUNK-AWARE采样：智能识别chunk型数据，只从有图像的chunk起始点开始"""
        
        valid_positions = []
        
        # 收集所有可能的起始位置（支持chunk型和正常数据）
        for ep_idx, ep_length in enumerate(self.memory_pool_lengths):
            episode_data = self.memory_pool_episodes[ep_idx]
            
            # acrlpd_data_converter CHECK: 检测是否为chunk型数据（通过metadata检查）
            if len(episode_data) > 0:
                sample_frame = episode_data[0]
                is_chunked_episode = not sample_frame.get('has_image', True)  # 如果第一帧没有图像，可能是chunk型
                
                # 进一步验证：检查is_chunk_start模式
                chunk_starts = [sample.get('is_chunk_start', True) for sample in episode_data]
                has_images = [sample.get('has_image', True) for sample in episode_data]
                is_chunked_episode = not all(has_images)  # 如果不是所有帧都有图像，则为chunk型
                
                if is_chunked_episode:
                    # acrlpd_data_converter CHUNK-TYPE: 只从有图像且为chunk开始的位置采样
                    logger.debug(f"Episode {ep_idx}: Detected chunk-type data")
                    for pos in range(ep_length):
                        if pos < len(episode_data):
                            sample = episode_data[pos]
                            # 必须同时满足：有图像 AND 是chunk开始
                            if sample.get('has_image', False) and sample.get('is_chunk_start', False):
                                valid_positions.append((ep_idx, pos))
                                logger.debug(f"  Valid chunk start: frame {pos}")
                else:
                    # acrlpd_data_converter NORMAL-TYPE: 原有逻辑，任何位置都可以开始
                    for pos in range(ep_length):
                        valid_positions.append((ep_idx, pos))
        
        if len(valid_positions) == 0:
            raise RuntimeError("No valid starting positions found - check chunk data or metadata")
        
        logger.debug(f"Found {len(valid_positions)} valid sequence starting positions")
        
        # 随机采样
        selected_indices = self.rng.choice(
            len(valid_positions), 
            size=batch_size, 
            replace=True if len(valid_positions) < batch_size else False
        )
        
        return [valid_positions[i] for i in selected_indices]
    
    def _find_next_image_frame(self, episode_data: List[dict], start_idx: int) -> Optional[int]:
        """acrlpd_data_converter CHUNK-HELPER: 寻找从start_idx开始的下一个有图像的帧"""
        for idx in range(start_idx, len(episode_data)):
            if episode_data[idx].get('has_image', True):
                return idx
        return None
    
    def _build_qc_transition_from_episode(self, ep_idx: int, start_idx: int, horizon_length: int) -> Dict[str, Any]:
        """基于qc_ACT的完整transition构建逻辑"""
        
        episode_data = self.memory_pool_episodes[ep_idx]
        episode_length = len(episode_data)
        
        # === 1. 当前观测构建 ===
        current_sample = episode_data[start_idx]
        current_obs = self._sample_to_pi0_observation(current_sample)
        
        # === 2. 动作序列构建（qc_ACT核心逻辑）===
        action_dim = self._get_action_dim(current_sample)
        actions = np.zeros((horizon_length, action_dim), dtype=np.float32)
        rewards = np.zeros(horizon_length, dtype=np.float32)
        masks = np.zeros(horizon_length, dtype=np.float32)
        valid = np.zeros(horizon_length, dtype=np.float32)
        terminals = np.zeros(horizon_length, dtype=np.bool_)
        
        # 构建序列的每一步
        for i in range(horizon_length):
            step_idx = start_idx + i
            
            if step_idx < episode_length:
                # === 在episode范围内 ===
                actions[i] = self._extract_action_from_sample(episode_data[step_idx])
                
                # qc_ACT风格奖励：通常在最后一步给奖励
                if i == horizon_length - 1:
                    rewards[i] = self._calculate_step_reward(episode_data, step_idx, i, horizon_length)
                else:
                    rewards[i] = 0.0  # 中间步骤依赖bootstrap
                    
                masks[i] = 1.0      # 有效的mask
                valid[i] = 1.0      # 有效的动作
                terminals[i] = (step_idx == episode_length - 1)  # episode最后一步
                
            else:
                # === 超出episode边界，使用padding ===
                if i > 0:
                    actions[i] = actions[i-1]  # 重复最后一个有效动作
                else:
                    actions[i] = self._extract_action_from_sample(episode_data[start_idx])
                    
                rewards[i] = 0.0    # Padding步骤奖励为0
                masks[i] = 0.0      # 无效的mask  
                valid[i] = 0.0      # 无效的动作
                terminals[i] = True # 超边界即为terminal
        
        # === 3. Next Observations构建（Bootstrap必需）===
        next_idx = start_idx + horizon_length
        next_is_terminal = False
        
        if next_idx < episode_length:
            # 下一帧在episode内
            next_sample = episode_data[next_idx]
            
            # acrlpd_data_converter CHUNK-TYPE HANDLING: 检查next_sample是否有图像
            if not next_sample.get('has_image', True):
                # Chunk型数据：next_idx位置没有图像，寻找下一个有图像的chunk开始
                next_with_image_idx = self._find_next_image_frame(episode_data, next_idx)
                if next_with_image_idx is not None and next_with_image_idx < episode_length:
                    next_sample = episode_data[next_with_image_idx]
                    next_obs = self._sample_to_pi0_observation(next_sample)
                    next_is_terminal = (next_with_image_idx == episode_length - 1)
                else:
                    # 没有找到下一个有图像的帧，使用current_obs
                    next_obs = current_obs
                    next_is_terminal = True
            else:
                # 正常情况：next_sample有图像
                next_obs = self._sample_to_pi0_observation(next_sample)
                next_is_terminal = (next_idx == episode_length - 1)
        else:
            # 超出边界，使用terminal处理
            next_obs = current_obs  # Terminal state convention
            next_is_terminal = True
        
        # === 4. 构建完整transition（qc_ACT格式）===
        return {
            'observations': current_obs,
            'next_observations': next_obs,
            'actions': actions,                    # [H, action_dim]
            'rewards': rewards,                    # [H]
            'masks': masks,                       # [H] - bootstrap mask
            'valid': valid,                       # [H] - action validity
            'terminals': terminals,               # [H] - step terminals  
            'next_terminal': next_is_terminal,    # scalar - next state terminal
            'sequence_mask': np.ones(1, dtype=np.bool_)[0]  # 序列级有效性
        }
    
    def _map_global_to_episode_transition(self, global_idx: int) -> Tuple[int, int]:
        """将全局索引映射到(episode_idx, transition_idx)"""
        
        if global_idx >= self.total_pool_transitions or global_idx < 0:
            raise IndexError(f"Global index {global_idx} out of range [0, {self.total_pool_transitions})")
        
        cumulative = 0
        for episode_idx, episode_length in enumerate(self.memory_pool_lengths):
            if global_idx < cumulative + episode_length:
                transition_idx = global_idx - cumulative
                return episode_idx, transition_idx
            cumulative += episode_length
        
        # 不应该到达这里
        raise IndexError(f"Failed to map global index {global_idx} to episode transition")
    
    def _sample_to_pi0_observation(self, sample: dict) -> _model.Observation:
        """将单个LeRobot样本转换为π₀ Observation格式"""
        
        # 应用OpenPI变换管道
        transformed_sample = self._apply_openpi_transforms(sample)
        
        # 转换为π₀格式
        observation, _ = self._to_openpi_format([transformed_sample])
        
        # 返回单个观测（去掉batch维度）
        return jax.tree.map(lambda x: x[0], observation)
    
    def _get_action_dim(self, sample: dict) -> int:
        """获取动作维度（优先使用模型配置）"""
        
        # 首先尝试从模型配置获取
        if hasattr(self.rl_config.model, 'action_dim'):
            return self.rl_config.model.action_dim
        
        # 标准键名尝试
        for action_key in ["action", "actions", "observation.action"]:
            if action_key in sample:
                action = sample[action_key]
                if isinstance(action, (list, np.ndarray)):
                    action_array = np.array(action)
                    if action_array.ndim >= 2:  # [H, action_dim]
                        return action_array.shape[-1]
                    else:  # [action_dim]
                        return len(action_array)
        
        # 默认值（ALOHA机器人）
        logger.warning("Could not determine action dimension, using default 14")
        return 14
    
    def _extract_action_from_sample(self, sample: dict) -> np.ndarray:
        """从单个样本中提取动作（使用OpenPI transform后的数据）"""
        
        # **关键修复：首先应用OpenPI transforms，确保14维→32维转换**
        transformed_sample = self._apply_openpi_transforms(sample)
        
        # 从已转换的样本中提取actions（应该已经是32维）
        if "actions" in transformed_sample:
            action = transformed_sample["actions"]
            # 处理torch.Tensor
            if hasattr(action, 'numpy'):
                action = action.numpy()
            # 转换为numpy数组
            if isinstance(action, (list, np.ndarray)):
                action_array = np.array(action, dtype=np.float32)
                # LeRobot action格式：[action_dim] 或 [horizon, action_dim]
                if action_array.ndim >= 2:  # [H, action_dim] - 取第一个动作
                    action_array = action_array[0]
                
                return action_array
        
        # 备用：尝试从transformed_sample中的其他键
        for action_key in ["action", "observation.action"]:
            if action_key in transformed_sample:
                action = transformed_sample[action_key]
                if hasattr(action, 'numpy'):
                    action = action.numpy()
                if isinstance(action, (list, np.ndarray)):
                    action_array = np.array(action, dtype=np.float32)
                    if action_array.ndim >= 2:
                        action_array = action_array[0]
                    
                    return action_array
        
        # 如果找不到动作，打印调试信息并返回零动作
        if not hasattr(self, '_debug_logged'):
            logger.warning(f"Could not extract action from transformed sample. Available keys: {list(transformed_sample.keys())}")
            if "actions" in transformed_sample:
                action_data = transformed_sample["actions"]
                logger.warning(f"Actions data type: {type(action_data)}, shape: {getattr(action_data, 'shape', 'no shape')}")
            self._debug_logged = True
        
        logger.warning("Could not extract action from sample, returning zero action")
        # 使用模型的action_dim（应该是32维）
        model_action_dim = getattr(self.rl_config.model, 'action_dim', 32)
        return np.zeros(model_action_dim, dtype=np.float32)
    
    def _calculate_step_reward(self, episode_data: List[dict], step_idx: int, seq_pos: int, horizon_length: int) -> float:
        """计算单步奖励（可根据具体任务定制）"""
        
        # 基础实现：从 LeRobot 数据中提取奖励
        if 'reward' in episode_data[step_idx]:
            return float(episode_data[step_idx]['reward'])
        
        # 备用方案：基于episode成功/失败的稀疏奖励
        if seq_pos == horizon_length - 1:  # 仅在序列最后给奖励
            # 可以基于episode metadata判断成功/失败
            return 1.0  # 简化为固定奖励
        
        return 0.0
    
    def _apply_openpi_transforms(self, raw_sample: dict) -> dict:
        """对单个样本应用OpenPI变换管道（延迟加载版本）"""
        
        # acrlpd_data_converter 关键：确保transforms在使用前已初始化
        self._ensure_transforms_initialized()
        
        transformed_sample = raw_sample.copy()
        
        # 应用三阶段变换（如果存在）
        try:
            if self.repack_transforms:
                if hasattr(self.repack_transforms, '__call__'):
                    transformed_sample = self.repack_transforms(transformed_sample)
                elif hasattr(self.repack_transforms, 'inputs'):  # 修复：Group对象使用inputs属性
                    for transform in self.repack_transforms.inputs:
                        transformed_sample = transform(transformed_sample)
                elif hasattr(self.repack_transforms, 'transforms'):  # 回退兼容性
                    for transform in self.repack_transforms.transforms:
                        transformed_sample = transform(transformed_sample)
                        
            if self.data_transforms:
                # Apply data transforms correctly
                if hasattr(self.data_transforms, 'inputs'):
                    for i, transform in enumerate(self.data_transforms.inputs):
                        transformed_sample = transform(transformed_sample)
                elif hasattr(self.data_transforms, '__call__'):
                    transformed_sample = self.data_transforms(transformed_sample)
                else:
                    logger.warning(f"Unknown data_transforms type: {type(self.data_transforms)}")
                        
            if self.model_transforms:
                if hasattr(self.model_transforms, '__call__'):
                    transformed_sample = self.model_transforms(transformed_sample)
                elif hasattr(self.model_transforms, 'inputs'):
                    for i, transform in enumerate(self.model_transforms.inputs):
                        transformed_sample = transform(transformed_sample)
                elif hasattr(self.model_transforms, 'transforms'):
                    for transform in self.model_transforms.transforms:
                        transformed_sample = transform(transformed_sample)
        except Exception as e:
            import traceback
            logger.error(f"Transform application failed, using raw sample: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            logger.error(f"Raw sample keys: {raw_sample.keys() if isinstance(raw_sample, dict) else type(raw_sample)}")
            
            # acrlpd_data_converter 详细调试：检查图像数据是否存在
            if isinstance(raw_sample, dict):
                image_keys = [k for k in raw_sample.keys() if 'image' in k.lower()]
                logger.error(f"Available image keys: {image_keys}")
                
                # 检查observation结构
                if 'observation' in raw_sample:
                    obs = raw_sample['observation']
                    logger.error(f"observation type: {type(obs)}")
                    if hasattr(obs, 'keys'):
                        logger.error(f"observation keys: {list(obs.keys())}")
                    elif isinstance(obs, dict):
                        logger.error(f"observation dict keys: {obs.keys()}")
            
            transformed_sample = raw_sample
        
        # 应用归一化（如果有norm_stats）
        if self.norm_stats:
            transformed_sample = self._apply_normalization(transformed_sample)
        
        return transformed_sample
    
    def _apply_normalization(self, sample: dict) -> dict:
        """应用归一化统计"""
        
        # 这里应该使用openpi的normalize模块
        # 具体实现取决于norm_stats的结构
        try:
            if hasattr(_normalize, 'apply') and callable(_normalize.apply):
                return _normalize.apply(sample, self.norm_stats)
            else:
                # 备用的手动归一化
                return self._manual_normalization(sample)
        except Exception as e:
            logger.warning(f"Normalization failed: {e}, returning original sample")
            return sample
    
    def _manual_normalization(self, sample: dict) -> dict:
        """手动归一化备用方案"""
        
        # 简单的状态归一化示例
        normalized_sample = sample.copy()
        
        if "observation.state" in sample and "qpos" in self.norm_stats:
            state = np.array(sample["observation.state"])
            qpos_stats = self.norm_stats["qpos"]
            
            if hasattr(qpos_stats, 'mean') and hasattr(qpos_stats, 'std'):
                normalized_state = (state - qpos_stats.mean) / qpos_stats.std
                normalized_sample["observation.state"] = normalized_state
        
        return normalized_sample
    
    def _collate_qc_batch(self, transitions: List[Dict[str, Any]]) -> Dict[str, jnp.ndarray]:
        """将Q-chunking transition列表整理成JAX batch格式"""
        
        if not transitions:
            raise ValueError("Empty transition list")
        
        # 准备批次字典
        batch = {}
        
        # === 处理π₀ Observation对象 ===
        observations = []
        next_observations = []
        
        for trans in transitions:
            observations.append(trans['observations'])
            next_observations.append(trans['next_observations'])
        
        # 使用OpenPI的Observation批次化
        batch['observations'] = self._batch_pi0_observations(observations)
        batch['next_observations'] = self._batch_pi0_observations(next_observations)
        
        # === 处理其他数组字段 ===
        array_keys = ['actions', 'rewards', 'masks', 'valid', 'terminals']
        for key in array_keys:
            arrays = [trans[key] for trans in transitions]
            batch[key] = jnp.array(arrays)  # [B, H, ...] or [B, H]
        
        # === 处理标量字段 ===  
        scalar_keys = ['next_terminal', 'sequence_mask']
        for key in scalar_keys:
            scalars = [trans[key] for trans in transitions]
            batch[key] = jnp.array(scalars)  # [B]
        
        return batch
    
    def _batch_pi0_observations(self, observations: List[_model.Observation]) -> _model.Observation:
        """批次化π₀ Observation对象"""
        
        if not observations:
            raise ValueError("Empty observation list")
        
        # acrlpd_data_converter 调试：检查传入的observations类型
        logger.debug(f"批次化observations: 数量={len(observations)}")
        for i, obs in enumerate(observations[:2]):  # 只检查前2个
            logger.debug(f"  obs[{i}] type: {type(obs)}")
            if hasattr(obs, 'keys'):
                logger.debug(f"  obs[{i}] keys: {list(obs.keys())}")
        
        # acrlpd_data_converter 防护措施：检查是否是Column对象
        for i, obs in enumerate(observations):
            if hasattr(obs, '__class__') and 'Column' in str(type(obs)):
                logger.error(f"acrlpd_data_converter 检测到Column对象 obs[{i}]: {type(obs)}")
                raise TypeError(f"Cannot batch Column objects. Transform failed for observation {i}. "
                               f"Expected _model.Observation, got {type(obs)}")
        
        # 使用JAX tree操作批次化
        def stack_observations(*obs_list):
            return jax.tree.map(lambda *arrays: jnp.stack(arrays, axis=0), *obs_list)
        
        return stack_observations(*observations)
    
    def _collate_batch(self, samples: List[dict]) -> dict:
        """将样本列表整理成批次格式"""
        
        if not samples:
            raise ValueError("Cannot collate empty sample list")
        
        # 使用JAX的tree工具进行批次整理
        def stack_arrays(*arrays):
            arrays = [np.asarray(arr) for arr in arrays]
            return np.stack(arrays, axis=0)
        
        return jax.tree.map(stack_arrays, *samples)
    
    def _to_openpi_format(self, batch_samples: List[dict]) -> Tuple[_model.Observation, jnp.ndarray]:
        """转换为OpenPI标准格式（仅用于_sample_to_pi0_observation）"""
        
        # 批次整理
        batch = self._collate_batch(batch_samples)
        
        # 提取动作序列（在这里不重要，主要是为Observation服务）
        actions = jnp.zeros((len(batch_samples), 1))  # 占位符
        
        # 构建图像字典
        image_dict = {}
        image_mask_dict = {}
        
        # 检查LeRobot标准图像格式
        for key in batch.keys():
            if key.startswith("observation.images."):
                camera_name = key.replace("observation.images.", "")
                image_dict[camera_name] = jnp.array(batch[key])
                batch_size = batch[key].shape[0]
                # 关键修复：强制确保image_masks只有batch维度 [batch_size,]
                image_mask_dict[camera_name] = jnp.ones((batch_size,), dtype=jnp.bool_)
        
        # 备用：直接的"image"格式
        if not image_dict and "image" in batch:
            if isinstance(batch["image"], dict):
                for cam_name, cam_data in batch["image"].items():
                    image_dict[cam_name] = jnp.array(cam_data)
                    batch_size = cam_data.shape[0]
                    # 关键修复：强制确保image_masks只有batch维度 [batch_size,]
                    image_mask_dict[cam_name] = jnp.ones((batch_size,), dtype=jnp.bool_)
            else:
                # 假设是单相机
                image_dict["camera"] = jnp.array(batch["image"])
                batch_size = batch["image"].shape[0]
                # 关键修复：强制确保image_masks只有batch维度 [batch_size,]
                image_mask_dict["camera"] = jnp.ones((batch_size,), dtype=jnp.bool_)
        
        # 如果仍然没有图像，创建占位符
        if not image_dict:
            batch_size = len(batch_samples)
            image_dict["placeholder"] = jnp.zeros((batch_size, 224, 224, 3), dtype=jnp.float32)
            # 关键修复：强制确保占位符image_masks只有batch维度 [batch_size,]
            image_mask_dict["placeholder"] = jnp.ones((batch_size,), dtype=jnp.bool_)
        
        # 构建状态
        state_data = None
        for state_key in ["observation.state", "state", "qpos"]:
            if state_key in batch:
                state_data = jnp.array(batch[state_key])
                break
        
        if state_data is None:
            batch_size = len(batch_samples)
            state_data = jnp.zeros((batch_size, 14), dtype=jnp.float32)  # 假设14维状态
        
        # OpenPI's AlohaInputs transform handles dimension padding automatically
        
        # 构建观测字典
        observation_dict = {
            "image": image_dict,
            "image_mask": image_mask_dict,
            "state": state_data,
        }
        
        # 添加语言相关字段（如果存在）
        if "tokenized_prompt" in batch:
            observation_dict["tokenized_prompt"] = jnp.array(batch["tokenized_prompt"])
        if "tokenized_prompt_mask" in batch:
            observation_dict["tokenized_prompt_mask"] = jnp.array(batch["tokenized_prompt_mask"])
        
        # 创建π₀ Observation对象
        observation = _model.Observation.from_dict(observation_dict)
        
        return observation, actions
    
    def refresh_memory_pool(self, epoch_seed: int):
        """qc_ACT风格：刷新内存池到新epoch"""
        
        old_epoch = self.current_epoch_seed
        self.current_epoch_seed = epoch_seed
        
        # 重新加载内存池
        self._load_current_memory_pool()
        
        # 更新统计
        self.stats['current_epoch'] = epoch_seed
        
        if self.rank == 0:
            logger.info(f"✓ 内存池已刷新: epoch {old_epoch} → {epoch_seed}")
    
    def __getitem__(self, idx: int) -> Dict[str, jnp.ndarray]:
        """qc_ACT核心：完全忽略idx参数，返回Q-chunking格式"""
        
        # 忽略传入的idx，直接调用sample_batch获取随机批次
        batch_data = self.sample_batch()
        
        # 返回批次中的第一个样本（但保持批次维度）
        single_sample = {}
        for key, value in batch_data.items():
            if isinstance(value, jnp.ndarray) and value.ndim > 0:
                single_sample[key] = value[0:1]  # 保持[1, ...] 形状
            else:
                single_sample[key] = value
        
        return single_sample
    
    def __len__(self) -> int:
        """返回内存池中的transition总数"""
        return self.total_pool_transitions
    
    def create_batch_iterator(self) -> Iterator[Dict[str, jnp.ndarray]]:
        """创建无限Q-chunking batch迭代器"""
        while True:
            yield self.sample_batch()
    
    def __iter__(self) -> Iterator[Dict[str, jnp.ndarray]]:
        """创建 Q-chunking 迭代器，支持for循环"""
        return self.create_batch_iterator()
    
    def get_dataset_statistics(self) -> dict:
        """获取统计信息"""
        return {
            **self.stats,
            'total_episodes_in_dataset': len(self.all_episode_info),
            'episodes_in_memory_pool': len(self.memory_pool_episodes),
            'transitions_in_memory_pool': self.total_pool_transitions,
            'memory_pool_utilization': len(self.memory_pool_episodes) / self.episodes_per_memory_pool * 100,
            'model_action_dim': self.rl_config.model.action_dim,
            'qchunking_horizon': self.qchunking_config.horizon_length,
            'batch_size': self.batch_size,
            'config_name': self.rl_config.name
        }


# 添加time导入
import time


def create_acrlpd_data_loader(
    rl_config: Any,  # RLTrainConfig
    batch_size: int = 128,
    episodes_per_memory_pool: int = 64,
    skip_norm_stats: bool = False,
    **kwargs
) -> ACRLPDDataLoader:
    """
    创建统一的ACRLPD数据加载器
    
    Args:
        rl_config: RLTrainConfig统一配置
        batch_size: 批次大小
        episodes_per_memory_pool: 内存池大小
        skip_norm_stats: 跳过norm stats加载（用于计算norm stats时）
        **kwargs: 其他参数
        
    Returns:
        ACRLPDDataLoader实例
    """
    
    return ACRLPDDataLoader(
        rl_config=rl_config,
        batch_size=batch_size,
        episodes_per_memory_pool=episodes_per_memory_pool,
        skip_norm_stats=skip_norm_stats,
        **kwargs
    )


def load_acrlpd_norm_stats(
    repo_id: str,
    norm_stats_dir: Optional[Path] = None
) -> Optional[at.PyTree[_normalize.NormStats]]:
    """
    加载预计算的归一化统计信息
    
    Args:
        repo_id: 数据集repository ID
        norm_stats_dir: 自定义统计目录路径
        
    Returns:
        加载的归一化统计信息或None
    """
    if norm_stats_dir is None:
        dataset_name = Path(repo_id).name if "/" in repo_id else repo_id
        norm_stats_dir = Path(f"/tmp/acrlpd_norm_stats/{dataset_name}")
    
    if norm_stats_dir.exists():
        try:
            logger.info(f" Loading norm_stats from: {norm_stats_dir}")
            norm_stats = _normalize.load(norm_stats_dir)
            logger.info(f"✓ Successfully loaded norm_stats: {list(norm_stats.keys())}")
            return norm_stats
        except Exception as e:
            logger.warning(f"Failed to load norm_stats from {norm_stats_dir}: {e}")
            return None
    else:
        logger.warning(f"Norm_stats directory not found: {norm_stats_dir}")
        logger.info(f" Run compute_acrlpd_norm_stats.py first to generate norm_stats")
        return None


if __name__ == "__main__":
    # 测试代码
    import argparse
    
    parser = argparse.ArgumentParser(description="Test qc_ACT ACRLPDDataLoader")
    parser.add_argument("--repo-id", type=str, required=True, help="LeRobot dataset repo ID")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--episodes-per-memory-pool", type=int, default=4, help="Memory pool size")
    parser.add_argument("--num-batches", type=int, default=3, help="Number of test batches")
    
    args = parser.parse_args()
    
    try:
        # 需要使用RLTrainConfig，这里需要从config导入
        print("acrlpd_data_converter 请使用新的统一配置系统测试:")
        print("   cd /dev/shm/lmc/openpi/ac_training")
        print("   /era-ai/conda_envs/openpi/bin/uv run python data/acrlpd_data_loader.py --config rl_aloha_fold")
        exit(0)
        
        # 打印统计信息
        stats = data_loader.get_dataset_statistics()
        print(" Dataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # 测试Q-chunking batch采样
        print(f"\n Testing {args.num_batches} Q-chunking batches:")
        for i in range(args.num_batches):
            batch_dict = data_loader.sample_batch()
            
            print(f"Batch {i}:")
            print(f"  Q-chunking format keys: {list(batch_dict.keys())}")
            
            # 检查观测
            if 'observations' in batch_dict:
                obs = batch_dict['observations']
                print(f"  Observations type: {type(obs)}")
                if hasattr(obs, 'state') and obs.state is not None:
                    print(f"    State shape: {obs.state.shape}")
                if hasattr(obs, 'image') and obs.image is not None:
                    for cam_name, cam_data in obs.image.items():
                        print(f"    {cam_name} shape: {cam_data.shape}")
            
            # 检查Q-chunking关键字段
            for key in ['actions', 'rewards', 'masks', 'valid', 'terminals']:
                if key in batch_dict:
                    print(f"  {key} shape: {batch_dict[key].shape}")
            
            for key in ['next_terminal', 'sequence_mask']:
                if key in batch_dict:
                    print(f"  {key} shape: {batch_dict[key].shape}")
        
        # 测试epoch刷新
        print("\n Testing epoch refresh:")
        old_pool_size = data_loader.total_pool_transitions
        data_loader.refresh_memory_pool(1)
        new_pool_size = data_loader.total_pool_transitions
        print(f"Pool size before refresh: {old_pool_size}")
        print(f"Pool size after refresh: {new_pool_size}")
        
        print("\n Q-chunking RL ACRLPDDataLoader test completed successfully!")
        
    except Exception as e:
        print(f" Test failed: {e}")
        import traceback
        traceback.print_exc()