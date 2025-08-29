# AC_Training_v2 Framework

AC_Training_v2 是基于OpenPI模式重构的ACRLPD训练框架，保持完整RL算法功能的同时优化训练效率。

## 架构概述

### 核心组件

- **agents_v2**: Agent核心实现，包含π₀模型集成、Critic网络ensemble和统一损失计算
- **training_v2**: 训练循环和状态管理，基于OpenPI模式简化
- **data_v2**: 数据加载组件（基于已重构的OpenPI数据pipeline）
- **scripts_v2**: 训练启动脚本
- **utils_v2**: 支持工具模块

### 技术架构

```
ac_training_v2/
├── agents_v2/
│   ├── acrlpd_pi0_agent.py      # Agent核心类，管理π₀和Critic组件
│   ├── loss_functions.py        # 统一损失计算，特征共享优化
│   └── critic_networks.py       # Critic网络ensemble实现
├── training_v2/
│   ├── train_state.py           # 训练状态管理，JIT编译优化
│   ├── training_loop.py         # 简化训练循环
│   └── optimizer.py             # 优化器配置
├── scripts_v2/
│   └── train_acrlpd_pi0.py     # 主训练入口
└── utils_v2/
    ├── metrics.py              # 简化指标记录
    ├── checkpoint.py           # checkpoint管理
    └── performance.py          # 性能监控
```

## 核心优化

### 1. 特征共享架构
- 在Actor、Critic、BC损失计算间共享encoder特征
- 单次feature computation，多处使用
- 减少重复的观察编码计算

### 2. 统一JIT编译
- 整合所有损失计算到单个JIT编译函数
- 避免训练时重复编译开销
- 统一的梯度计算和参数更新

### 3. 简化训练循环
- 移除冗余性能监控代码
- 最小化日志记录开销
- 批量I/O操作

## 使用方法

### 基础训练
```bash
cd /dev/shm/lmc/openpi/ac_training_v2
python scripts_v2/train_acrlpd_pi0.py --config rl_fold_box --exp_name test_v2
```

### 配置参数
```bash
python scripts_v2/train_acrlpd_pi0.py \
    --config rl_fold_box \
    --exp_name my_experiment \
    --max_steps 50000 \
    --pi0_lr 1e-5 \
    --critic_lr 1e-3 \
    --batch_size 128
```

### 环境设置
```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
```

## 算法实现

### ACRLPD核心组件

1. **Actor (π₀)**: 3.2B参数的diffusion模型
2. **Critic Ensemble**: 2个20M参数的Q网络
3. **Best-of-N采样**: Actor策略优化机制
4. **Q-learning**: 支持动作序列的时序差分学习
5. **Behavior Cloning**: 正则化组件

### 损失函数

- **Critic Loss**: Huber损失的TD error
- **Actor Loss**: 基于Best-of-N的策略梯度
- **BC Loss**: MSE行为克隆损失

### 网络架构

- **观察编码**: 复用π₀模型的encoder
- **状态特征**: MLP编码器
- **图像特征**: 多相机SiGLIP特征
- **动作预测**: Diffusion采样机制

## 兼容性

### 向后兼容
- 支持原有AC Training配置格式
- 兼容现有checkpoint结构
- 保持相同的RL算法语义

### 数据格式
- 使用OpenPI标准数据pipeline
- Q-chunking输出格式
- LeRobot数据集兼容

## 配置系统

框架使用原有的AC Training配置系统，主要配置项：

```python
# π₀模型配置
pi0_checkpoint_path: str
action_horizon: int = 20
action_dim: int = 14

# Critic配置  
num_critics: int = 2
critic_hidden_dims: List[int] = [256, 256]

# 训练配置
best_of_n: int = 4
bc_weight: float = 0.1
```

## 实现细节

### JIT编译策略
- 统一的`unified_train_step_jit`函数
- 一次编译包含所有训练逻辑
- 参数和状态更新合并

### 内存管理
- FSDP支持分布式训练
- 梯度accumulation和分离
- Target network软更新机制

### 数据流
```
LeRobot Dataset → OpenPI transforms → Q-chunking format → Shared features → Loss computation
```

## 监控和日志

- 简化的指标记录系统
- 批量日志写入
- 可选的wandb集成
- 性能统计和报告

## 开发和调试

### Debug模式
```bash
python scripts_v2/train_acrlpd_pi0.py --config rl_fold_box --exp_name debug_test --debug
```

### Dry run
```bash
python scripts_v2/train_acrlpd_pi0.py --config rl_fold_box --exp_name test --dry_run
```

## 依赖要求

- JAX ecosystem (JAX, Flax, Optax)
- OpenPI framework  
- LeRobot数据处理
- 标准ML库 (numpy, wandb等)

## 扩展性

框架设计支持：
- 新的优化器类型
- 额外的损失组件  
- 自定义数据增强
- 不同的网络架构