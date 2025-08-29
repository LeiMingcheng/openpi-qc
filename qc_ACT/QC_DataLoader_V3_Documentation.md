# Q-Chunking DataLoader V3 说明文档

## 概述

QCSimpleDatasetV3 是为Q-Chunking强化学习设计的数据加载器，采用"内存池+动态随机采样"架构，实现真正的随机性训练数据遍历。

## 核心设计理念

### 1. 内存池架构
- **缓存池**：每个Epoch加载N个完整episodes到内存
- **动态采样**：每次`__getitem__()`调用时实时创建transition
- **无预计算**：不预先创建固定数量的transitions

### 2. 真随机采样
- **忽略索引**：`__getitem__(idx)`完全忽略传入的idx参数
- **全局随机**：从所有episode的所有帧中随机选择起始点
- **统计遍历**：通过大量随机采样统计性地遍历整个数据集

### 3. 掩码边界处理
- **完整采样空间**：episode的每一帧都可作为起始点
- **自动填充**：超出边界的部分使用padding + valid/masks标记
- **训练时忽略**：通过掩码机制在损失计算时忽略无效部分

## 关键类和方法

### QCSimpleDatasetV3类

```python
class QCSimpleDatasetV3(Dataset):
    def __init__(self,
                 dataset_dir: str,
                 camera_names: List[str], 
                 norm_stats: Dict,
                 chunk_size: int,
                 episodes_per_epoch: int = 32,  # 每个Epoch加载的episodes数量
                 positive_ratio: float = 0.6,   # 正样本比例
                 rank: int = 0,                  # GPU编号
                 world_size: int = 1,            # 总GPU数
                 epoch_seed: int = 0,            # Epoch种子
                 dataset_type: str = "train"     # 数据集类型
                 )
```

#### 核心数据结构
```python
self.loaded_episodes = []      # 存储完整episode数据
self.episode_lengths = []      # 每个episode的实际长度
self.episode_labels = []       # 正负样本标签
self.total_starts = 0          # 总的可采样起始点数量
```

#### 关键方法

**1. `__getitem__(idx) -> Dict`**
```python
def __getitem__(self, idx):
    # 核心：忽略idx，每次都随机选择全局起始点
    random_global_idx = torch.randint(0, self.total_starts, (1,)).item()
    
    # 映射到对应episode和起始点
    # 动态创建transition
    return self._create_transition_from_preloaded(episode_data, start_idx, is_positive)
```

**2. `__len__() -> int`**
```python
def __len__(self):
    return self.total_starts  # 返回所有episode长度之和
```

**3. `_load_current_epoch()`**
- GPU独立随机采样：`gpu_seed = epoch_seed + rank * 1000`
- 加载N个完整episodes到内存池
- 计算总可采样起始点数量

**4. `refresh_epoch(new_epoch_seed)`**
- 清理旧数据，加载新的随机采样episodes
- 每个训练Epoch开始时调用

## 数据采样逻辑

### 采样空间计算
```python
# 示例：32个episodes
Episode 1: 600帧 → 600个可采样起始点
Episode 2: 580帧 → 580个可采样起始点  
Episode 3: 620帧 → 620个可采样起始点
...
总计：所有episode长度之和 = total_starts
```

### 随机映射机制
```python
# 从[0, total_starts)中随机选择
random_idx = torch.randint(0, self.total_starts, (1,)).item()

# 映射到具体episode
cumulative = 0
for episode_idx, length in enumerate(self.episode_lengths):
    if random_idx < cumulative + length:
        start_idx = random_idx - cumulative
        break
    cumulative += length
```

### Transition创建
使用现有的`_create_transition_from_preloaded()`方法：
- 处理图像解码和归一化
- 构建chunk_size长度的动作序列
- 生成valid/masks/terminals等掩码信息
- 计算累积折扣奖励

## 训练集成

### DataLoader配置
```python
dataloader = DataLoader(
    dataset,
    batch_size=128,           # 批次大小
    shuffle=True,             # DataLoader层面的shuffle（实际被忽略）
    num_workers=8,            # 并行加载线程
    pin_memory=True,          # GPU传输优化
    drop_last=True            # 保持批次大小一致
)
```

### 训练循环
```python
# 每个epoch开始时刷新数据
train_dataset.refresh_epoch(epoch)

# 固定步数训练（steps_per_epoch > total_starts）
for step in range(steps_per_epoch):  # 如5000步
    batch = next(dataloader_iter)    # 随机采样
    loss = model.update(batch)
```

## 配置参数

### 核心参数 (qc_config.py)
```python
QC_CONFIG = {
    'episodes_per_epoch': 32,    # 每个Epoch加载的episodes数量
    'steps_per_epoch': 5000,     # 每个Epoch固定训练步数
    'positive_ratio': 0.6,       # 正样本episodes比例
    'batch_size': 128,           # DataLoader批次大小
    'num_workers': 8,            # 数据加载线程数
}
```

### 关键设计要求
- `steps_per_epoch > total_possible_starts`：确保充分的随机遍历
- `episodes_per_epoch`：平衡内存使用和数据多样性
- `positive_ratio`：控制正负样本比例

## 性能特征

### 内存使用
```python
# 估算公式
memory_per_episode ≈ episode_length × 11MB/frame
total_memory ≈ episodes_per_epoch × average_episode_length × 11MB

# 典型配置 (32 episodes)
32 episodes × 400 frames/episode × 11MB ≈ 140GB
```

### 加载性能
- **初始加载**：60-80秒（32个episodes）
- **Epoch刷新**：类似时间（重新采样episodes）
- **运行时采样**：毫秒级（内存中动态创建）

### 随机性保证
- **GPU独立**：每个GPU使用不同种子，允许部分重叠
- **Epoch变化**：每个Epoch重新随机采样episodes
- **步骤随机**：每个训练步骤都是真正随机采样

## 数据结构

### Transition格式 (输出)
```python
transition = {
    'observations': {
        'qpos': np.array([14]),           # 当前机器人状态
        'images': np.array([3, H, W, 3]), # RGB图像：3个相机视角
        'depth_images': None              # 深度图像（可选）
    },
    'actions': np.array([chunk_size, 14]),     # 动作序列：chunk×14维
    'rewards': np.array([chunk_size]),         # 奖励序列：累积折扣
    'masks': np.array([chunk_size]),           # 掩码：区分有效/无效
    'valid': np.array([chunk_size]),           # 有效性：真实/填充
    'terminals': np.array([chunk_size]),       # 终止标记
    'is_positive': bool,                       # 正负样本标记
    'final_reward': float                      # 最终累积奖励
}
```

### 批处理后 (DataLoader输出)
所有字段增加batch维度：`[batch_size, ...]`

## 使用示例

### 创建数据加载器
```python
from qc_dataset_v3 import create_qc_dataloader_v3

train_dataset, train_dataloader = create_qc_dataloader_v3(
    dataset_dir="/path/to/data",
    camera_names=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
    norm_stats=norm_stats,
    chunk_size=50,
    episodes_per_epoch=32,
    batch_size=128,
    rank=0,
    world_size=8,
    epoch_seed=0
)
```

### 训练集成
```python
for epoch in range(num_epochs):
    # 刷新数据
    train_dataset.refresh_epoch(epoch)
    
    # 固定步数训练
    for step in range(steps_per_epoch):
        batch = next(dataloader_iter)
        loss = model.update(batch)
```

### 统计信息
```python
stats = train_dataset.get_stats()
print(f"总可采样起始点: {stats['total_possible_starts']}")
print(f"加载episodes: {stats['loaded_episodes']}")
print(f"正样本比例: {stats['positive_ratio']:.2f}")
```

## 优势特点

1. **真正的随机性**：每次采样都是完全随机的，避免固定的遍历顺序
2. **内存高效**：只存储原始episode数据，动态创建transitions
3. **最大化数据利用**：episode的每一帧都可作为起始点
4. **简化设计**：用随机性代替复杂的索引管理
5. **充分遍历**：通过足够的训练步数统计性地覆盖所有数据
6. **边界处理**：完善的掩码机制自动处理chunk边界问题

## 注意事项

1. **内存管理**：根据GPU内存调整`episodes_per_epoch`
2. **训练步数**：确保`steps_per_epoch`足够大以充分遍历数据
3. **随机种子**：分布式训练中每个GPU使用独立种子
4. **数据格式**：要求HDF5格式，支持压缩图像
5. **掩码依赖**：训练代码必须正确处理valid/masks信息

这个设计完美实现了"用随机性遍历数据"的核心理念，为强化学习训练提供了高效、灵活的数据加载解决方案。