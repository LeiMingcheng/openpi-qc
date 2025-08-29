"""
Q-Chunking数据集加载器 V3 - 极简版
核心改进：
1. 移除max_batches_in_memory，每个Epoch只维护一批episodes  
2. 每个GPU独立随机采样，允许部分重叠
3. 消除复杂的索引管理和异步刷新机制
4. 基于随机种子确保训练随机性和可重现性
"""
import numpy as np
import torch
import os
import h5py
from torch.utils.data import Dataset, DataLoader
import cv2
import time
from typing import List, Dict, Tuple, Optional
import gc
import psutil


class QCSimpleDatasetV3(Dataset):
    """
    Q-Chunking简化数据集 V3 - 动态随机采样版
    核心思想：缓存池存储完整episodes，每次随机采样创建transitions
    """
    
    def __init__(self,
                 dataset_dir: str,
                 camera_names: List[str], 
                 norm_stats: Dict,
                 chunk_size: int,
                 use_depth_image: bool = False,
                 use_robot_base: bool = False,
                 positive_scores: List[int] = [5, 4],
                 negative_scores: List[int] = [1, 2],
                 positive_ratio: float = 0.6,
                 episodes_per_epoch: int = 32,  # 每个Epoch的episodes数量
                 # 分布式参数
                 rank: int = 0,
                 world_size: int = 1,
                 # 初始种子
                 epoch_seed: int = 0,
                 # 标识数据集类型
                 dataset_type: str = "train",
                 ):
        
        super().__init__()
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.chunk_size = chunk_size
        self.use_depth_image = use_depth_image
        self.use_robot_base = use_robot_base
        self.positive_ratio = positive_ratio
        self.episodes_per_epoch = episodes_per_epoch
        self.dataset_type = dataset_type
        
        # 分布式参数
        self.rank = rank
        self.world_size = world_size
        self.epoch_seed = epoch_seed
        
        # 收集所有episode路径
        self.all_positive_paths = self._get_episode_paths(positive_scores)
        self.all_negative_paths = self._get_episode_paths(negative_scores)
        
        if self.rank == 0:
            print(f"[{self.dataset_type.upper()}] 找到正样本episodes: {len(self.all_positive_paths)}")
            print(f"[{self.dataset_type.upper()}] 找到负样本episodes: {len(self.all_negative_paths)}")
            print(f"[{self.dataset_type.upper()}] 每个Epoch将随机采样 {episodes_per_epoch} episodes")
        
        # 当前Epoch的缓存池数据
        self.loaded_episodes = []      # 完整episodes数据
        self.episode_lengths = []      # 每个episode长度
        self.episode_labels = []       # 正负标签
        self.total_starts = 0          # 总的可能起始点数量
        
        # 加载初始数据
        self._load_current_epoch()
    
    def _get_episode_paths(self, scores: List[int]) -> List[str]:
        """获取指定分数的所有episode路径"""
        paths = []
        for score in scores:
            score_dir = os.path.join(self.dataset_dir, f'score_{score}')
            if os.path.exists(score_dir):
                episode_files = [f for f in os.listdir(score_dir) if f.endswith('.hdf5')]
                episode_paths = [os.path.join(score_dir, f) for f in episode_files]
                paths.extend(episode_paths)
        return paths
    
    def _load_current_epoch(self):
        """加载当前Epoch的完整episodes到缓存池"""
        if self.rank == 0:
            print(f"[{self.dataset_type.upper()}] 加载Epoch {self.epoch_seed} 数据...")
        start_time = time.time()
        
        # 每个GPU使用不同的随机种子：epoch_seed + rank * 1000
        gpu_seed = self.epoch_seed + self.rank * 1000
        rng = np.random.default_rng(gpu_seed)
        
        if self.rank == 0:
            print(f"[{self.dataset_type.upper()}] GPU {self.rank} 使用随机种子: {gpu_seed}")
        
        # 计算正负样本数量
        num_positive = int(self.episodes_per_epoch * self.positive_ratio)
        num_negative = self.episodes_per_epoch - num_positive
        
        # 从所有数据中随机采样episodes（允许GPU间重叠）
        if len(self.all_positive_paths) > 0:
            sampled_pos_paths = rng.choice(self.all_positive_paths, size=num_positive, replace=True)
        else:
            sampled_pos_paths = []
            
        if len(self.all_negative_paths) > 0:
            sampled_neg_paths = rng.choice(self.all_negative_paths, size=num_negative, replace=True)
        else:
            sampled_neg_paths = []
        
        # 清空缓存池
        self.loaded_episodes = []
        self.episode_lengths = []
        self.episode_labels = []
        
        # 加载正样本episodes
        for episode_path in sampled_pos_paths:
            try:
                episode_data = self._load_complete_episode(episode_path)
                self.loaded_episodes.append(episode_data)
                self.episode_lengths.append(len(episode_data['qpos']))
                self.episode_labels.append(True)
            except Exception as e:
                if self.rank == 0:
                    print(f"加载正样本episode失败 {episode_path}: {e}")
        
        # 加载负样本episodes
        for episode_path in sampled_neg_paths:
            try:
                episode_data = self._load_complete_episode(episode_path)
                self.loaded_episodes.append(episode_data)
                self.episode_lengths.append(len(episode_data['qpos']))
                self.episode_labels.append(False)
            except Exception as e:
                if self.rank == 0:
                    print(f"加载负样本episode失败 {episode_path}: {e}")
        
        # 计算总的可能起始点数量
        self.total_starts = sum(self.episode_lengths)
        
        load_time = time.time() - start_time
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        
        if self.rank == 0:
            pos_episodes = sum(1 for label in self.episode_labels if label)
            neg_episodes = len(self.episode_labels) - pos_episodes
            print(f"[{self.dataset_type.upper()}] Epoch {self.epoch_seed} 加载完成:")
            print(f"  正样本episodes: {pos_episodes} (总长度: {sum(length for i, length in enumerate(self.episode_lengths) if self.episode_labels[i])})")
            print(f"  负样本episodes: {neg_episodes} (总长度: {sum(length for i, length in enumerate(self.episode_lengths) if not self.episode_labels[i])})")
            print(f"  总可采样起始点: {self.total_starts}")
            print(f"  加载耗时: {load_time:.2f}s")
            print(f"  内存使用: {memory_mb:.1f}MB")
    
    def _load_complete_episode(self, episode_path: str) -> Dict:
        """加载完整的episode数据到内存"""
        with h5py.File(episode_path, 'r') as root:
            is_compress = root.attrs.get('compress', False)
            # 直接调用现有方法加载完整episode
            episode_data = self._preload_episode_data(root, is_compress)
        return episode_data
    
    def _preload_episode_data(self, root: h5py.File, is_compress: bool) -> Dict:
        """预加载episode的所有必要数据"""
        episode_data = {
            'qpos': root['/observations/qpos'][()],
            'action': root['/action'][()],
        }
        
        # 加载图像数据
        for cam_name in self.camera_names:
            if is_compress:
                episode_data[f'{cam_name}_images'] = root[f'/observations/images/{cam_name}'][()]
            else:
                episode_data[f'{cam_name}_images'] = root[f'/observations/images/{cam_name}'][()]
        
        # 深度图像（如果需要）
        if self.use_depth_image:
            for cam_name in self.camera_names:
                depth_key = f'/observations/depth_images/{cam_name}'
                if depth_key in root:
                    episode_data[f'{cam_name}_depth'] = root[depth_key][()]
        
        # 机器人底座动作（如果需要）
        if self.use_robot_base and '/base_action' in root:
            episode_data['base_action'] = root['/base_action'][()]
        
        return episode_data
    
    def _create_transition_from_preloaded(self, episode_data: Dict, start_idx: int, is_positive: bool) -> Dict:
        """从预加载的数据创建transition"""
        episode_len = len(episode_data['qpos'])
        
        # 当前状态
        qpos = episode_data['qpos'][start_idx]
        qpos_norm = (qpos - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        
        # 收集图像
        images = []
        depth_images = []
        
        for cam_name in self.camera_names:
            # RGB图像
            if f'{cam_name}_images' in episode_data:
                img_data = episode_data[f'{cam_name}_images'][start_idx]
                if len(img_data.shape) == 1:  # 压缩数据
                    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img = img_data
                
                # 归一化到[0,1]并转换为PyTorch格式 [C, H, W]
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
                images.append(img)
            
            # 深度图像（如果需要）
            if self.use_depth_image and f'{cam_name}_depth' in episode_data:
                depth_img = episode_data[f'{cam_name}_depth'][start_idx]
                # 深度图像通常是单通道，确保格式为 [1, H, W]
                if len(depth_img.shape) == 2:
                    depth_img = depth_img[np.newaxis, ...]  # H, W -> 1, H, W
                depth_images.append(depth_img)
        
        # 动作序列（Q-Chunking核心）
        sequence_length = self.chunk_size
        actions = np.zeros((sequence_length, episode_data['action'].shape[1]), dtype=np.float32)
        rewards = np.zeros(sequence_length, dtype=np.float32)
        masks = np.zeros(sequence_length, dtype=np.float32)
        valid = np.zeros(sequence_length, dtype=np.float32) 
        terminals = np.zeros(sequence_length, dtype=np.bool_)
        
        # 基础奖励
        base_reward = 1.0 if is_positive else 0.0
        discount = 0.99
        
        # 构建动作序列
        for i in range(sequence_length):
            step_idx = start_idx + i
            
            if step_idx < episode_len:
                # 在episode范围内
                actions[i] = episode_data['qpos'][step_idx]
                if self.use_robot_base and 'base_action' in episode_data:
                    base_act = episode_data['base_action'][step_idx]
                    actions[i] = np.concatenate([actions[i], base_act])
                
                # 标准QC-FQL奖励计算：只在序列末尾给奖励，其他步骤依赖bootstrap
                if i == sequence_length - 1:  # 只在最后一步给奖励
                    rewards[i] = base_reward
                else:
                    rewards[i] = 0.0  # 中间步骤奖励为0，依赖bootstrap机制
                masks[i] = 1.0
                    
                valid[i] = 1.0
                terminals[i] = (step_idx == episode_len - 1)
                
            else:
                # 超出边界，使用padding
                if i > 0:
                    actions[i] = actions[i-1]
                    rewards[i] = 0.0  # 超出边界的奖励为0
                    masks[i] = 0.0
                    valid[i] = 0.0
                else:
                    actions[i] = episode_data['qpos'][start_idx]
                    if self.use_robot_base and 'base_action' in episode_data:
                        base_act = episode_data['base_action'][start_idx]
                        actions[i] = np.concatenate([actions[i], base_act])
                    rewards[i] = 0.0  # 首步超出边界的奖励也为0
                    masks[i] = 0.0
                    valid[i] = 0.0
                    
                terminals[i] = True
        
        # 归一化动作序列
        actions_norm = (actions - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        
        # 构建观测字典
        obs_dict = {
            'qpos': qpos_norm.astype(np.float32),
            'images': np.stack(images, axis=0),
        }
        
        if self.use_depth_image and depth_images:
            obs_dict['depth_images'] = np.stack(depth_images, axis=0)
        
        # 构建next_observations（Bootstrap机制必需）
        next_obs_dict = None
        next_idx = start_idx + sequence_length  # 序列后的下一帧
        next_is_terminal = False  # 下一状态是否为terminal（用于正确的mask设置）
        
        if next_idx < episode_len:
            # 下一帧在episode范围内，构建真实的next_observations
            next_qpos = episode_data['qpos'][next_idx]
            next_qpos_norm = (next_qpos - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
            
            # 收集下一帧的图像
            next_images = []
            next_depth_images = []
            
            for cam_name in self.camera_names:
                # RGB图像
                if f'{cam_name}_images' in episode_data:
                    img_data = episode_data[f'{cam_name}_images'][next_idx]
                    if len(img_data.shape) == 1:  # 压缩数据
                        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    else:
                        img = img_data
                    
                    # 归一化到[0,1]并转换为PyTorch格式 [C, H, W]
                    img = img.astype(np.float32) / 255.0
                    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
                    next_images.append(img)
                
                # 深度图像（如果需要）
                if self.use_depth_image and f'{cam_name}_depth' in episode_data:
                    depth_img = episode_data[f'{cam_name}_depth'][next_idx]
                    # 深度图像通常是单通道，确保格式为 [1, H, W]
                    if len(depth_img.shape) == 2:
                        depth_img = depth_img[np.newaxis, ...]  # H, W -> 1, H, W
                    next_depth_images.append(depth_img)
            
            # 构建next_observations字典
            next_obs_dict = {
                'qpos': next_qpos_norm.astype(np.float32),
                'images': np.stack(next_images, axis=0),
            }
            
            if self.use_depth_image and next_depth_images:
                next_obs_dict['depth_images'] = np.stack(next_depth_images, axis=0)
            
            # 检查下一帧是否为episode的最后一帧（terminal state）
            next_is_terminal = (next_idx == episode_len - 1)
        else:
            # 序列超出episode边界，下一状态就是terminal
            next_is_terminal = True
            
            # 使用当前观测作为next_observations（Terminal state的标准处理）
            next_obs_dict = {
                'qpos': qpos_norm.astype(np.float32),
                'images': np.stack(images, axis=0),
            }
            
            if self.use_depth_image and depth_images:
                next_obs_dict['depth_images'] = np.stack(depth_images, axis=0)
        
        # 构建transition（添加next_terminal用于正确的Bootstrap mask设置）
        transition = {
            'observations': obs_dict,
            'next_observations': next_obs_dict,  # 添加next_observations用于Bootstrap
            'actions': actions_norm.astype(np.float32),
            'rewards': rewards.astype(np.float32),
            'masks': masks.astype(np.float32),
            'valid': valid.astype(np.float32),
            'terminals': terminals.astype(np.bool_),
            'next_terminal': next_is_terminal,  # 下一状态是否为terminal（标准Q-learning mask）
            'is_positive': is_positive,
            'final_reward': rewards[-1] if sequence_length > 0 else base_reward,
        }
        
        return transition
    
    def refresh_epoch(self, new_epoch_seed: int):
        """刷新到新Epoch，重新加载完整episodes"""
        if self.rank == 0:
            print(f"[{self.dataset_type.upper()}] 刷新到Epoch {new_epoch_seed}")
        
        self.epoch_seed = new_epoch_seed
        # 清理旧数据
        self.clear_cache()
        # 加载新数据
        self._load_current_epoch()
    
    def clear_cache(self):
        """清理缓存数据"""
        self.loaded_episodes.clear()
        self.episode_lengths.clear()
        self.episode_labels.clear()
        gc.collect()
    
    def __getitem__(self, idx: int) -> Dict:
        """
        动态随机采样transition
        注意：忽略传入的idx参数，每次都随机选择全局起始点
        """
        if self.total_starts == 0:
            raise RuntimeError("No episodes loaded. Please call refresh_epoch() first.")
        
        # 核心：忽略idx，直接随机选择全局索引
        random_global_idx = torch.randint(0, self.total_starts, (1,)).item()
        
        # 映射到对应的episode和起始点
        cumulative = 0
        for episode_idx, length in enumerate(self.episode_lengths):
            if random_global_idx < cumulative + length:
                start_idx = random_global_idx - cumulative
                episode_data = self.loaded_episodes[episode_idx]
                is_positive = self.episode_labels[episode_idx]
                # 使用现有方法动态创建transition
                return self._create_transition_from_preloaded(episode_data, start_idx, is_positive)
            cumulative += length
        
        # 理论上不应该走到这里
        raise RuntimeError(f"Invalid random index {random_global_idx} for total_starts {self.total_starts}")
    
    def __len__(self) -> int:
        """返回总的可采样起始点数量"""
        return self.total_starts
    
    def get_stats(self) -> Dict:
        """获取数据集统计信息"""
        pos_episodes = sum(1 for label in self.episode_labels if label)
        neg_episodes = len(self.episode_labels) - pos_episodes
        pos_frames = sum(length for i, length in enumerate(self.episode_lengths) if self.episode_labels[i])
        neg_frames = sum(length for i, length in enumerate(self.episode_lengths) if not self.episode_labels[i])
        
        return {
            'total_possible_starts': self.total_starts,
            'loaded_episodes': len(self.loaded_episodes),
            'positive_episodes': pos_episodes,
            'negative_episodes': neg_episodes,
            'positive_frames': pos_frames,
            'negative_frames': neg_frames,
            'positive_ratio': pos_episodes / len(self.episode_labels) if len(self.episode_labels) > 0 else 0,
            'episodes_per_epoch': self.episodes_per_epoch,
            'epoch_seed': self.epoch_seed,
        }


def create_qc_dataloader_v3(dataset_dir: str,
                           camera_names: List[str],
                           norm_stats: Dict,
                           chunk_size: int,
                           episodes_per_epoch: int = 32,
                           batch_size: int = 128,
                           num_workers: int = 8,
                           positive_scores: List[int] = [5, 4],
                           negative_scores: List[int] = [1, 2],
                           positive_ratio: float = 0.6,
                           use_depth_image: bool = False,
                           use_robot_base: bool = False,
                           rank: int = 0,
                           world_size: int = 1,
                           epoch_seed: int = 0,
                           dataset_type: str = "train",
                           shuffle: bool = True) -> Tuple[QCSimpleDatasetV3, DataLoader]:
    """创建Q-Chunking V3数据加载器"""
    
    # 创建数据集
    dataset = QCSimpleDatasetV3(
        dataset_dir=dataset_dir,
        camera_names=camera_names,
        norm_stats=norm_stats,
        chunk_size=chunk_size,
        episodes_per_epoch=episodes_per_epoch,
        positive_scores=positive_scores,
        negative_scores=negative_scores,
        positive_ratio=positive_ratio,
        use_depth_image=use_depth_image,
        use_robot_base=use_robot_base,
        rank=rank,
        world_size=world_size,
        epoch_seed=epoch_seed,
        dataset_type=dataset_type,
    )
    
    # 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # 确保批次大小一致
    )
    
    return dataset, dataloader


if __name__ == '__main__':
    # 测试代码
    print("测试QCSimpleDatasetV3...")
    
    # 模拟参数
    dataset_dir = "/era-ai/lm/dataset/lmc/aloha_pp"
    camera_names = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    
    # 模拟norm_stats
    norm_stats = {
        'qpos_mean': np.zeros(14),
        'qpos_std': np.ones(14),
    }
    
    try:
        dataset, dataloader = create_qc_dataloader_v3(
            dataset_dir=dataset_dir,
            camera_names=camera_names,
            norm_stats=norm_stats,
            chunk_size=16,
            episodes_per_epoch=8,  # 小数量测试
            batch_size=4,
            num_workers=2,
            rank=0,
            world_size=1,
            epoch_seed=0,
        )
        
        print("数据集创建成功！")
        print("统计信息:", dataset.get_stats())
        
        # 测试数据加载
        print("\n测试数据加载:")
        for i, batch in enumerate(dataloader):
            print(f"Batch {i}:")
            if 'observations' in batch:
                print(f"  observations.qpos: {batch['observations']['qpos'].shape}")
                print(f"  observations.images: {batch['observations']['images'].shape}")
            print(f"  actions: {batch['actions'].shape}")
            print(f"  is_positive: {sum(batch['is_positive'])}/{len(batch['is_positive'])}")
            if i >= 2:  # 只测试前3个batch
                break
        
        # 测试随机采样特性
        print("\n测试随机采样:")
        sample1 = dataset[0]
        sample2 = dataset[0]
        qpos1 = torch.tensor(sample1['observations']['qpos'])
        qpos2 = torch.tensor(sample2['observations']['qpos'])
        same_qpos = torch.equal(qpos1, qpos2)
        print(f"连续两次采样是否相同: {same_qpos} (应该大概率为False)")
        print(f"sample1 qpos前5位: {qpos1[:5]}")
        print(f"sample2 qpos前5位: {qpos2[:5]}")
        
        # 测试epoch刷新
        print("\n测试epoch刷新...")
        old_total = dataset.total_starts
        dataset.refresh_epoch(1)
        new_total = dataset.total_starts
        print(f"刷新前总起始点: {old_total}, 刷新后: {new_total}")
        print("刷新后统计:", dataset.get_stats())
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()