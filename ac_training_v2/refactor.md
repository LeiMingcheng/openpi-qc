# AC Training V2 系统性重构方案

**基于全面架构分析的系统重构计划**

## 🚨 现状分析

### 架构断裂危机
ac_training_v2目前存在严重的接口断裂和架构混乱：

**✅ agents_v2已重构完成**：
- 删除ACRLPDPi0Config，统一使用config.py
- 损失计算逻辑内化到Agent类
- 参数化构造函数，清晰的接口

**❌ 其他模块仍为v1复制**：
- training_v2仍导入`agents.acrlpd_pi0_agent`，期望旧接口
- scripts导入旧模块，依赖已删除的ACRLPDPi0Config
- 功能重复，职责边界模糊

### 具体问题
1. **接口不匹配**：training_v2期望`ACRLPDPi0Config`，但agents_v2已删除
2. **依赖错乱**：scripts导入agents而非agents_v2
3. **功能重复**：training_v2和agents_v2存在重复的损失计算逻辑
4. **职责混乱**：缺乏清晰的模块边界定义

## 🎯 目标架构设计

### 设计原则
1. **清晰职责分工**：agents负责算法，training负责基础设施
2. **简化接口**：统一的参数传递和调用方式
3. **消除重复**：删除重复的配置、损失计算等功能
4. **独立性**：不考虑向后兼容，完全独立的系统

### 目标目录结构与详细功能定义

```
ac_training_v2/
├── agents_v2/           # ✅ 核心算法模块（已重构完成）
│   ├── acrlpd_pi0_agent.py      # Agent类，内化损失计算
│   ├── critic_networks.py       # Critic网络定义
│   └── loss_functions.py        # 简化的工具函数
├── training_v2/         # 🔄 训练基础设施（需要完全重构）
│   ├── trainer.py              # 简化训练器，适配agents_v2
│   ├── fsdp_support.py         # FSDP分布式训练
│   └── checkpointing.py        # Checkpoint管理
├── data_v2/            # 📦 数据处理（保持相对独立）
│   └── acrlpd_data_loader.py   # LeRobot→OpenPI格式转换
├── scripts/            # 🔄 训练脚本（更新接口）
│   └── train_acrlpd_v2.py      # 统一训练入口
├── config.py           # ✅ 统一配置系统
└── utils/              # 🛠️ 工具函数
    ├── memory_monitor.py       # GPU内存监控工具
    └── pytree_checker.py       # PyTree结构验证
```

## 📋 详细模块功能定义

### agents_v2/acrlpd_pi0_agent.py ✅已重构完成
**职责**：ACRLPD算法核心逻辑，π₀模型集成，损失计算

#### 核心类定义
```python
class ACRLPDPi0Agent:
    """ACRLPD + π₀ 集成Agent，已完成重构"""
    
    def __init__(
        self,
        # 核心模型组件
        pi0_model: Any,                     # π₀扩散模型
        critic_networks: CriticNetworks,    # Critic网络ensemble
        
        # Q-learning核心参数（从config中解包）
        num_action_samples: int,            # Best-of-N采样数量
        horizon_length: int,                # Q-chunking序列长度  
        real_action_dim: int,               # 真实动作维度
        discount: float = 0.99,             # 折扣因子
        target_update_rate: float = 0.005,  # Target网络更新率
        
        # 损失权重
        bc_loss_weight: float = 0.1,        # BC正则化权重
        
        # EMA配置
        use_ema: bool = True,               # 启用EMA
        pi0_ema_decay: float = 0.999,       # π₀ EMA衰减率
        critic_ema_decay: float = 0.99,     # Critic EMA衰减率
        
        # 优化器状态（外部创建传入）
        pi0_opt_state: optax.OptState,      # π₀优化器状态
        critic_opt_state: optax.OptState,   # Critic优化器状态
        temp_opt_state: Optional[optax.OptState] = None,  # 温度优化器状态
        
        # 其他配置
        **kwargs
    ):
```

#### 核心方法接口
```python
# === 损失计算方法（已内化） ===
def compute_critic_loss(
    self, 
    batch: Dict[str, jnp.ndarray], 
    rng: jax.random.PRNGKey
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """计算Critic损失（Q-learning + Target网络）"""

def compute_actor_loss(
    self, 
    batch: Dict[str, jnp.ndarray], 
    rng: jax.random.PRNGKey
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """计算Actor损失（AWR + Best-of-N采样）"""

def compute_bc_loss(
    self, 
    batch: Dict[str, jnp.ndarray], 
    rng: jax.random.PRNGKey
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """计算Behavior Cloning正则化损失"""

def compute_loss(
    self, 
    batch: Dict[str, jnp.ndarray], 
    rng: jax.random.PRNGKey
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """统一损失计算接口（组合所有损失）"""

# === 训练步骤方法 ===
def train_step(
    self, 
    batch: Dict[str, jnp.ndarray], 
    rng: jax.random.PRNGKey
) -> Tuple['ACRLPDPi0Agent', Dict[str, Any]]:
    """单步训练更新（包含梯度计算和参数更新）"""

# === OpenPI兼容性方法 ===
def create_train_state(self) -> openpi_training.TrainState:
    """创建OpenPI兼容的训练状态"""

def create_openpi_train_state(self) -> openpi_training.TrainState:
    """创建仅包含π₀的OpenPI格式checkpoint"""

def to_train_state(self) -> openpi_training.TrainState:
    """转换为OpenPI TrainState格式"""

# === 状态管理方法 ===
def update_target_networks(self) -> 'ACRLPDPi0Agent':
    """更新Target网络（软更新）"""

def update_ema_parameters(self) -> 'ACRLPDPi0Agent':
    """更新EMA参数"""
```

#### 工厂函数接口
```python
def create_acrlpd_pi0_agent_from_rl_config(
    rl_config: RLTrainConfig,
    rng: jax.random.PRNGKey,
    pi0_opt_state: Optional[optax.OptState] = None,
    critic_opt_state: Optional[optax.OptState] = None,
    temp_opt_state: Optional[optax.OptState] = None
) -> ACRLPDPi0Agent:
    """从RLTrainConfig创建Agent实例（统一入口）"""

def create_acrlpd_pi0_agent(
    pi0_model: Any,
    critic_networks: CriticNetworks,
    **kwargs
) -> ACRLPDPi0Agent:
    """直接参数创建Agent实例"""
```

### training_v2/trainer.py 🔄需要完全重构
**职责**：训练循环管理，FSDP集成，性能监控

#### 核心类定义
```python
class ACRLPDTrainer:
    """简化的ACRLPD训练器，适配agents_v2接口"""
    
    def __init__(
        self,
        agent: ACRLPDPi0Agent,              # agents_v2的Agent实例
        dataloader: ACRLPDDataLoader,       # 数据加载器
        rl_config: RLTrainConfig,           # 统一配置
        training_config: ACRLPDTrainingConfig,  # 训练专用配置
        eval_fn: Optional[Callable] = None, # 评估函数
        
        # FSDP相关参数
        mesh: Optional[jax.sharding.Mesh] = None,
        data_sharding: Optional[jax.sharding.Sharding] = None,
        replicated_sharding: Optional[jax.sharding.Sharding] = None,
        
        # 全局优化器（解决pytree一致性）
        global_pi0_tx: Optional[optax.GradientTransformation] = None,
        global_critic_tx: Optional[optax.GradientTransformation] = None
    ):
```

#### 核心方法接口
```python
# === 训练循环方法 ===
def train(self) -> ACRLPDPi0Agent:
    """主训练循环，返回训练完成的Agent"""

def train_epoch(
    self, 
    epoch: int, 
    num_steps: int
) -> Dict[str, float]:
    """单个epoch的训练循环"""

def train_step_wrapper(
    self,
    agent: ACRLPDPi0Agent,
    batch: Dict[str, jnp.ndarray],
    rng: jax.random.PRNGKey
) -> Tuple[ACRLPDPi0Agent, Dict[str, Any]]:
    """训练步骤包装器（用于JIT编译）"""

# === FSDP支持方法 ===
def setup_fsdp(
    self,
    mesh: jax.sharding.Mesh,
    data_sharding: jax.sharding.Sharding
) -> None:
    """设置FSDP分布式训练"""

def create_fsdp_train_step(self) -> Callable:
    """创建FSDP兼容的训练步骤函数"""

# === Checkpoint方法 ===
def save_checkpoint(
    self, 
    step: int, 
    agent: ACRLPDPi0Agent,
    checkpoint_dir: str
) -> None:
    """保存训练checkpoint"""

def save_openpi_checkpoint(
    self, 
    step: int, 
    agent: ACRLPDPi0Agent,
    checkpoint_dir: str
) -> None:
    """保存π₀ OpenPI格式checkpoint"""

def load_checkpoint(
    self, 
    checkpoint_path: str
) -> ACRLPDPi0Agent:
    """加载训练checkpoint"""

# === 评估方法 ===
def evaluate(
    self, 
    agent: ACRLPDPi0Agent, 
    eval_steps: int = 100
) -> Dict[str, float]:
    """模型评估"""

# === 监控和日志方法 ===
def log_training_metrics(
    self, 
    step: int, 
    metrics: Dict[str, Any]
) -> None:
    """记录训练指标"""

def log_memory_usage(
    self, 
    step: int, 
    phase: str = "training"
) -> None:
    """记录GPU内存使用"""
```

### training_v2/fsdp_support.py 🔄需要创建
**职责**：FSDP分布式训练支持，内存优化

#### 核心函数接口
```python
def create_fsdp_mesh(
    fsdp_devices: int,
    data_parallel_devices: Optional[int] = None
) -> jax.sharding.Mesh:
    """创建FSDP训练mesh"""

def create_fsdp_sharding_strategy(
    mesh: jax.sharding.Mesh,
    model_config: Any
) -> Tuple[jax.sharding.Sharding, jax.sharding.Sharding]:
    """创建FSDP分片策略（数据分片 + 复制分片）"""

def init_fsdp_train_state(
    rl_config: RLTrainConfig,
    mesh: jax.sharding.Mesh,
    rng: jax.random.PRNGKey,
    global_optimizers: Dict[str, optax.GradientTransformation]
) -> Tuple[ACRLPDPi0Agent, jax.sharding.Sharding]:
    """初始化FSDP训练状态"""

def create_fsdp_train_step_fn(
    agent: ACRLPDPi0Agent,
    mesh: jax.sharding.Mesh,
    data_sharding: jax.sharding.Sharding
) -> Callable:
    """创建FSDP优化的训练步骤函数"""

def validate_fsdp_sharding(
    train_state: Any,
    mesh: jax.sharding.Mesh
) -> bool:
    """验证FSDP分片是否正确工作"""

def optimize_fsdp_memory(
    batch_size: int,
    model_config: Any,
    available_devices: int
) -> Dict[str, int]:
    """FSDP内存优化建议"""
```

### training_v2/checkpointing.py 🔄需要重构
**职责**：Checkpoint保存和加载，OpenPI格式兼容

#### 核心函数接口
```python
def save_acrlpd_checkpoint(
    agent: ACRLPDPi0Agent,
    step: int,
    checkpoint_dir: str,
    save_openpi: bool = True,
    save_full: bool = True
) -> Dict[str, str]:
    """保存ACRLPD checkpoint（支持多种格式）"""

def save_openpi_pi0_checkpoint(
    agent: ACRLPDPi0Agent,
    checkpoint_path: str
) -> str:
    """保存π₀模型的OpenPI兼容checkpoint"""

def load_acrlpd_checkpoint(
    checkpoint_path: str,
    rl_config: RLTrainConfig,
    rng: jax.random.PRNGKey
) -> Tuple[ACRLPDPi0Agent, int]:
    """加载ACRLPD checkpoint"""

def load_pi0_weights_from_openpi(
    openpi_checkpoint_path: str,
    target_agent: ACRLPDPi0Agent
) -> ACRLPDPi0Agent:
    """从OpenPI checkpoint加载π₀权重"""

def create_checkpoint_metadata(
    rl_config: RLTrainConfig,
    training_metrics: Dict[str, Any],
    step: int
) -> Dict[str, Any]:
    """创建checkpoint元数据"""

def validate_checkpoint_compatibility(
    checkpoint_path: str,
    expected_config: RLTrainConfig
) -> bool:
    """验证checkpoint与配置的兼容性"""
```

### data_v2/acrlpd_data_loader.py ✅已存在，需要接口优化
**职责**：LeRobot→OpenPI数据格式转换，Q-chunking数据生成

#### 核心类接口（已实现，需要接口标准化）
```python
class ACRLPDDataLoader:
    """Q-chunking RL数据加载器，统一接口版本"""
    
    def __init__(
        self,
        rl_config: RLTrainConfig,           # 统一配置接口
        batch_size: int = 128,
        episodes_per_memory_pool: int = 64,
        skip_norm_stats: bool = False,
        device_sharding: Optional[jax.sharding.Sharding] = None,
        **kwargs
    ):
    
    # === Q-chunking数据生成 ===
    def sample_batch(self) -> Dict[str, jnp.ndarray]:
        """生成Q-chunking格式batch"""
    
    def refresh_memory_pool(self, epoch_seed: int) -> None:
        """刷新内存池（新epoch）"""
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""

# === 工厂函数 ===
def create_acrlpd_data_loader(
    rl_config: RLTrainConfig,
    batch_size: int = 128,
    episodes_per_memory_pool: int = 64,
    skip_norm_stats: bool = False,
    **kwargs
) -> ACRLPDDataLoader:
    """创建统一的ACRLPD数据加载器"""
```

### scripts/train_acrlpd_v2.py 🔄需要重构
**职责**：训练脚本入口，命令行接口，实验管理

#### 核心函数接口
```python
def create_argument_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""

def load_rl_config(args: argparse.Namespace) -> RLTrainConfig:
    """从命令行参数加载RLTrainConfig"""

def setup_jax_environment(
    rl_config: RLTrainConfig,
    args: argparse.Namespace
) -> Tuple[jax.sharding.Mesh, jax.sharding.Sharding, jax.sharding.Sharding]:
    """设置JAX环境和FSDP配置"""

def create_training_components(
    rl_config: RLTrainConfig,
    mesh: jax.sharding.Mesh,
    data_sharding: jax.sharding.Sharding,
    rng: jax.random.PRNGKey
) -> Tuple[ACRLPDPi0Agent, ACRLPDDataLoader, ACRLPDTrainer]:
    """创建训练组件（Agent, DataLoader, Trainer）"""

def run_training_experiment(
    trainer: ACRLPDTrainer,
    rl_config: RLTrainConfig,
    args: argparse.Namespace
) -> ACRLPDPi0Agent:
    """运行完整训练实验"""

def main() -> None:
    """主入口函数"""

# === 辅助函数 ===
def log_gpu_memory(step_name: str) -> Dict[str, float]:
    """GPU内存使用监控"""

def verify_fsdp_sharding(train_state: Any, step_name: str) -> None:
    """验证FSDP分片效果"""

def analyze_fsdp_effectiveness(usage_data: List[float]) -> None:
    """分析FSDP分片效果"""
```

### config.py ✅已完成
**职责**：统一配置系统，预定义配置

#### 核心接口（已实现）
```python
@dataclasses.dataclass(frozen=True)
class RLTrainConfig(openpi_config.TrainConfig):
    """统一的强化学习训练配置"""
    acrlpd: ACRLPDHyperparams
    qchunking: QChunkingConfig
    # ... 其他字段

def get_config(config_name: str) -> RLTrainConfig:
    """获取预定义配置"""

def list_configs() -> Dict[str, str]:
    """列出所有可用配置"""

def cli() -> RLTrainConfig:
    """命令行配置选择接口"""
```

### utils/memory_monitor.py ✅已存在
**职责**：GPU内存监控和分析

#### 核心接口（已实现）
```python
class GPUMemoryMonitor:
    """GPU内存使用监控器"""
    
    def checkpoint_memory(
        self, 
        name: str, 
        train_state: Optional[Any] = None
    ) -> None:
        """创建内存使用检查点"""
    
    def analyze_train_state_memory(
        self, 
        train_state: Any
    ) -> Dict[str, Dict[str, float]]:
        """分析训练状态的详细内存组成"""
    
    def get_memory_report(self) -> str:
        """生成详细的内存使用报告"""

# === 全局函数 ===
def log_memory_usage(
    step: int, 
    train_state: Any = None, 
    phase: str = "training"
) -> None:
    """便捷的内存使用记录函数"""

def enable_memory_monitoring() -> None:
    """启用内存监控系统"""
```

### utils/pytree_checker.py 🔄需要创建
**职责**：PyTree结构验证，FSDP兼容性检查

#### 核心函数接口
```python
def diagnose_pytree_structure(
    pytree: Any, 
    name: str = "pytree"
) -> Dict[str, Any]:
    """诊断PyTree结构"""

def check_optimizer_consistency(
    opt1: optax.GradientTransformation,
    opt2: optax.GradientTransformation,
    name1: str = "opt1",
    name2: str = "opt2"
) -> bool:
    """检查优化器一致性"""

def validate_fsdp_compatibility(
    train_state: Any,
    mesh: jax.sharding.Mesh,
    data_sharding: jax.sharding.Sharding
) -> bool:
    """验证FSDP兼容性"""

def check_pytree_leaf_dtypes(
    pytree: Any,
    expected_dtype: jnp.dtype = jnp.bfloat16
) -> Dict[str, List[str]]:
    """检查PyTree叶子节点数据类型"""

def compare_pytree_structures(
    tree1: Any,
    tree2: Any,
    name1: str = "tree1", 
    name2: str = "tree2"
) -> Dict[str, Any]:
    """比较两个PyTree的结构差异"""
```

## 📋 重构阶段规划

### Stage 1: Training-Agents接口统一 (High Priority)
**目标**：使training_v2完全适配agents_v2的新架构

- **1.1 更新training_v2导入和接口**
  - 修改training_v2导入agents_v2而非agents
  - 适配agents_v2的参数化构造函数
  - 调用agents_v2的内化损失计算方法

- **1.2 简化training职责**
  - 删除training_v2中重复的损失计算逻辑
  - 专注于训练循环、FSDP支持、checkpoint管理
  - 通过简单接口调用agents_v2方法

### Stage 2: Checkpoint机制统一 (Medium Priority)
**目标**：复用现有的OpenPI兼容checkpoint机制

- **2.1 分析现有机制**
  - agents中的`create_train_state()`已支持OpenPI格式
  - 无需重新设计，直接移植到agents_v2

- **2.2 简化checkpoint保存**
  - 删除复杂的多格式支持
  - 专注π₀ OpenPI格式和完整ACRLPD格式

### Stage 3: Scripts接口更新 (Medium Priority)
**目标**：更新训练脚本使用新的接口

- **3.1 更新imports和工厂函数调用**
- **3.2 适配新的参数传递方式**
- **3.3 统一配置系统使用**

### Stage 4: 架构清理优化 (Low Priority)
**目标**：最终清理和优化

- **4.1 删除残留的重复代码**
- **4.2 优化import依赖关系**
- **4.3 文档和测试完善**

## 🔧 接口设计规范和数据流

### 核心数据格式标准

#### Q-chunking Batch格式
```python
# ACRLPDDataLoader.sample_batch() 输出格式
batch: Dict[str, jnp.ndarray] = {
    # π₀观测格式（支持多相机）
    'observations': _model.Observation = {
        'image': Dict[str, jnp.ndarray],      # {"cam_high": [B,224,224,3], ...}
        'image_mask': Dict[str, jnp.ndarray], # {"cam_high": [B,], ...}
        'state': jnp.ndarray,                 # [B, state_dim]
        'tokenized_prompt': jnp.ndarray,      # [B, prompt_len]
        'tokenized_prompt_mask': jnp.ndarray, # [B, prompt_len]
    },
    'next_observations': _model.Observation,  # 同上格式，用于bootstrap
    
    # Q-chunking动作序列
    'actions': jnp.ndarray,          # [B, H, action_dim] - H步动作序列
    'rewards': jnp.ndarray,          # [B, H] - 奖励序列
    'masks': jnp.ndarray,           # [B, H] - Bootstrap mask
    'valid': jnp.ndarray,           # [B, H] - 动作有效性
    'terminals': jnp.ndarray,       # [B, H] - 步骤终止标志
    'next_terminal': jnp.ndarray,   # [B] - 下一状态terminal标志
    'sequence_mask': jnp.ndarray,   # [B] - 序列有效性
}
```

#### Agent训练状态格式
```python
# ACRLPDPi0Agent内部状态结构
agent_state = {
    # 模型参数
    'pi0_params': PyTree,           # π₀模型参数
    'critic_params': PyTree,        # Critic网络参数
    'target_critic_params': PyTree, # Target Critic参数
    
    # EMA参数
    'pi0_ema_params': PyTree,       # π₀ EMA参数
    'critic_ema_params': PyTree,    # Critic EMA参数
    
    # 优化器状态
    'pi0_opt_state': optax.OptState,    # π₀优化器状态
    'critic_opt_state': optax.OptState, # Critic优化器状态
    'temp_opt_state': optax.OptState,   # 温度优化器状态（可选）
    
    # 训练计数器
    'step': int,                    # 训练步数
    'epoch': int,                   # 当前epoch
}
```

### agents_v2 → training_v2 接口契约

#### 初始化接口
```python
# training_v2中的Agent创建模式
agent = create_acrlpd_pi0_agent_from_rl_config(
    rl_config=rl_config,           # RLTrainConfig统一配置
    rng=rng,                       # JAX随机状态
    pi0_opt_state=None,            # 可选：预先创建的优化器状态
    critic_opt_state=None,         # 可选：预先创建的优化器状态
    temp_opt_state=None            # 可选：温度优化器状态
) -> ACRLPDPi0Agent
```

#### 训练循环接口
```python
# 单步训练的标准调用模式
def standard_train_step(
    agent: ACRLPDPi0Agent,
    batch: Dict[str, jnp.ndarray],
    rng: jax.random.PRNGKey
) -> Tuple[ACRLPDPi0Agent, Dict[str, Any]]:
    """标准训练步骤接口"""
    
    # 1. 计算损失（已内化到Agent）
    total_loss, loss_info = agent.compute_loss(batch, rng)
    
    # 2. 执行训练步骤
    updated_agent, train_info = agent.train_step(batch, rng)
    
    # 3. 合并信息
    train_info.update(loss_info)
    
    return updated_agent, train_info

# FSDP训练步骤的调用模式
@jax.jit
def fsdp_train_step(
    agent: ACRLPDPi0Agent,
    batch: Dict[str, jnp.ndarray],
    rng: jax.random.PRNGKey
) -> Tuple[ACRLPDPi0Agent, Dict[str, Any]]:
    """FSDP优化的训练步骤"""
    return agent.train_step(batch, rng)
```

#### Checkpoint接口
```python
# OpenPI兼容checkpoint保存
def save_checkpoint_interface(
    agent: ACRLPDPi0Agent,
    step: int,
    checkpoint_dir: str
) -> Dict[str, str]:
    """Checkpoint保存接口"""
    
    # 1. 保存完整ACRLPD状态
    full_path = f"{checkpoint_dir}/acrlpd_full_step_{step}.pkl"
    save_acrlpd_checkpoint(agent, step, checkpoint_dir, save_full=True)
    
    # 2. 保存π₀ OpenPI格式
    openpi_path = f"{checkpoint_dir}/pi0_openpi_step_{step}"
    openpi_state = agent.create_openpi_train_state()  # 只包含π₀
    save_openpi_pi0_checkpoint(agent, openpi_path)
    
    return {
        'full_checkpoint': full_path,
        'openpi_checkpoint': openpi_path
    }
```

### training_v2 → data_v2 接口契约

#### 数据加载接口
```python
# 统一数据加载器创建
dataloader = create_acrlpd_data_loader(
    rl_config=rl_config,                    # 包含数据配置
    batch_size=rl_config.batch_size,        # 批次大小
    episodes_per_memory_pool=64,            # 内存池大小
    device_sharding=data_sharding           # FSDP数据分片
) -> ACRLPDDataLoader

# 批次采样接口
batch = dataloader.sample_batch()  # -> Dict[str, jnp.ndarray]

# Epoch管理接口
dataloader.refresh_memory_pool(epoch_seed)  # 新epoch刷新数据
```

### scripts → 全系统 接口契约

#### 统一训练入口
```python
def unified_training_interface(
    config_name: str,
    exp_name: str,
    **training_args
) -> ACRLPDPi0Agent:
    """统一训练接口"""
    
    # 1. 加载配置
    rl_config = get_config(config_name)
    rl_config = customize_config(rl_config, **training_args)
    
    # 2. 设置环境
    mesh, data_sharding, replicated_sharding = setup_jax_environment(rl_config)
    
    # 3. 创建组件
    agent, dataloader, trainer = create_training_components(
        rl_config, mesh, data_sharding, rng
    )
    
    # 4. 执行训练
    trained_agent = trainer.train()
    
    return trained_agent
```

### FSDP集成接口契约

#### FSDP初始化接口
```python
def fsdp_integration_interface(
    rl_config: RLTrainConfig,
    mesh: jax.sharding.Mesh,
    rng: jax.random.PRNGKey
) -> Tuple[ACRLPDPi0Agent, Callable]:
    """FSDP集成接口"""
    
    # 1. 创建全局优化器（解决pytree一致性）
    global_optimizers = create_global_optimizers(rl_config)
    
    # 2. 初始化FSDP训练状态
    agent, state_sharding = init_fsdp_train_state(
        rl_config, mesh, rng, global_optimizers
    )
    
    # 3. 创建FSDP训练函数
    fsdp_train_step_fn = create_fsdp_train_step_fn(
        agent, mesh, data_sharding
    )
    
    return agent, fsdp_train_step_fn
```

### 职责边界明确定义

#### agents_v2职责
- **算法逻辑**：ACRLPD算法实现，Q-learning + AWR
- **损失计算**：Critic损失、Actor损失、BC损失的内化计算
- **模型管理**：π₀模型、Critic网络、EMA参数管理
- **状态更新**：参数更新、Target网络更新、优化器状态管理
- **OpenPI兼容**：TrainState格式转换，π₀ checkpoint生成

#### training_v2职责  
- **训练循环**：Epoch循环、批次迭代、性能监控
- **FSDP分片**：分布式训练setup、内存优化
- **Checkpoint管理**：完整状态保存/加载、实验管理
- **系统集成**：Agent-DataLoader协调、JIT编译管理

#### data_v2职责
- **数据格式转换**：LeRobot → OpenPI格式转换
- **Q-chunking生成**：动作序列构建、Bootstrap处理
- **内存池管理**：Episode缓存、随机采样策略
- **性能优化**：并行加载、Transform pipeline优化

#### scripts职责
- **命令行接口**：参数解析、配置定制
- **实验管理**：checkpoint路径、WandB集成
- **环境设置**：JAX配置、FSDP mesh创建
- **组件协调**：Agent-Trainer-DataLoader的统一管理

### 数据流向图
```
用户命令 → scripts/train_acrlpd_v2.py
    ↓
config.py (RLTrainConfig加载)
    ↓
data_v2/acrlpd_data_loader.py (LeRobot→OpenPI转换)
    ↓
agents_v2/acrlpd_pi0_agent.py (算法逻辑)
    ↓
training_v2/trainer.py (训练循环 + FSDP)
    ↓
training_v2/checkpointing.py (OpenPI兼容保存)
```

## ✅ 验证标准

### 功能验证
- [ ] agents_v2和training_v2协同工作正常
- [ ] FSDP和标准训练模式都能运行
- [ ] π₀ checkpoint可保存为OpenPI格式
- [ ] 训练收敛性与原系统一致

### 架构验证
- [ ] 模块职责边界清晰
- [ ] 接口简单统一
- [ ] 无重复功能
- [ ] 依赖关系明确

## 🎉 预期收益

1. **架构简化**：清晰的模块职责，简单的接口设计
2. **维护性提升**：消除重复代码，统一配置管理
3. **扩展性提升**：标准化接口便于添加新功能
4. **性能优化**：消除不必要的参数传递和计算
5. **兼容性保证**：与OpenPI生态系统完全兼容

---

**下一步：详细实施计划请参考 `refactor_detail.md`**