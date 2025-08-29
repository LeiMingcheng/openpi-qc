# AC Training V2 系统性重构方案 (完整版)

**基于全面代码分析的详细实施计划**

**目标**: 解决ac_training_v2中的严重接口断裂问题，实现agents_v2与training_v2/scripts的完全协同，确保与OpenPI和FSDP的完全兼容性。

**重构原则**:
1. **不考虑向后兼容** - ac_training_v2完全独立
2. **删除推理相关代码** - 专注训练功能  
3. **OpenPI π₀ checkpoint兼容** - 便于复制到ALOHA推理
4. **消除功能重复** - 清晰的模块边界

---

基于深入代码分析，ac_training_v2系统存在严重的接口断裂危机：

### ✅ 已重构完成：agents_v2
- **ACRLPDPi0Agent类**: 删除ACRLPDPi0Config，参数化构造函数
- **损失计算内化**: compute_critic_loss, compute_actor_loss, compute_bc_loss
- **统一工厂函数**: create_acrlpd_pi0_agent_from_rl_config
- **OpenPI兼容**: create_openpi_train_state, to_train_state方法

### ❌ 严重接口不匹配问题
1. **training_v2/training_loop.py**:
   ```python
   # 错误：仍导入旧接口
   from agents.acrlpd_pi0_agent import ACRLPDPi0Agent, ACRLPDPi0Config
   ```
   
2. **scripts/train_acrlpd_pi0.py**:
   ```python 
   # 错误：导入已删除的配置
   from agents.acrlpd_pi0_agent import ACRLPDPi0Agent, create_acrlpd_pi0_agent
   ```
   
3. **功能重复**: training_v2中仍有损失计算逻辑，与agents_v2内化方法冲突

### 🎯 解决策略：5阶段系统性重构

**目标**: 修复严重的接口断裂，使agents_v2与training_v2/scripts协同工作

### 1.1 修复training_v2导入和接口适配

#### 关键修复：training_v2/training_loop.py
```python
# 修复前（错误）：
from agents.acrlpd_pi0_agent import ACRLPDPi0Agent, ACRLPDPi0Config

# 修复后（正确）：
from agents_v2.acrlpd_pi0_agent import ACRLPDPi0Agent, create_acrlpd_pi0_agent_from_rl_config
```

#### 删除重复的损失计算逻辑
training_v2中删除以下重复功能：
- ✂️ **删除**: `compute_critic_losses()` - 已内化到Agent  
- ✂️ **删除**: `compute_actor_losses()` - 已内化到Agent
- ✂️ **删除**: `compute_bc_losses()` - 已内化到Agent
- ✅ **保留**: 训练循环、FSDP管理、checkpoint保存

### 1.2 修复scripts/train_acrlpd_pi0.py接口

#### 关键修复：导入和工厂函数调用
```python
# 修复前（错误）：
from agents.acrlpd_pi0_agent import ACRLPDPi0Agent, create_acrlpd_pi0_agent

# 修复后（正确）：  
from agents_v2.acrlpd_pi0_agent import create_acrlpd_pi0_agent_from_rl_config
```

#### 统一Agent创建接口
```python
# 新的统一调用方式
agent = create_acrlpd_pi0_agent_from_rl_config(
    rl_config=rl_config,
    rng=agent_rng
)
```

### 1.3 训练器类重构：ACRLPDTrainer

#### 新的简化训练器接口
```python
class ACRLPDTrainer:
    def __init__(
        self,
        agent: ACRLPDPi0Agent,              # agents_v2的统一Agent
        dataloader: ACRLPDDataLoader,       # 数据加载器
        rl_config: RLTrainConfig,           # 统一配置
        training_config: ACRLPDTrainingConfig,
        # FSDP支持
        mesh: Optional[jax.sharding.Mesh] = None,
        data_sharding: Optional[jax.sharding.Sharding] = None,
    ):
    
    def train(self) -> ACRLPDPi0Agent:
        """主训练循环 - 直接调用Agent的内化方法"""
        for step in range(self.rl_config.num_train_steps):
            batch = self.dataloader.sample_batch()
            
            # 关键：直接调用agents_v2的内化方法
            updated_agent, train_info = self.agent.train_step(batch, rng)
            
            # 训练器专注于基础设施
            self._log_metrics(step, train_info)
            self._save_checkpoint_if_needed(step, updated_agent)
            self.agent = updated_agent
```

## 阶段 2: FSDP集成统一

**目标**: 统一FSDP支持，消除pytree一致性问题和内存优化

### 2.1 创建training_v2/fsdp_support.py

#### 统一FSDP初始化接口
```python
def init_acrlpd_fsdp_training(
    rl_config: RLTrainConfig,
    mesh: jax.sharding.Mesh,
    rng: jax.random.PRNGKey,
    data_sharding: jax.sharding.Sharding,
    step: int = 0,
    global_pi0_tx: Optional[optax.GradientTransformation] = None,
    global_critic_tx: Optional[optax.GradientTransformation] = None
) -> Tuple[Any, jax.sharding.Sharding, Callable]:
    """
    统一FSDP训练初始化 - 避免模型重复创建
    
    Returns:
        - train_state: FSDP分片的训练状态
        - state_sharding: 状态分片配置
        - lazy_jit_creator: 延迟JIT函数创建器
    """
```

### 2.2 解决PyTree一致性问题

#### 全局优化器创建
```python
# 在FSDP上下文外创建全局优化器（确保一致性）
global_pi0_tx = _optimizer.create_optimizer(rl_config.actor_optimizer, rl_config.actor_lr_schedule)
global_critic_tx = _optimizer.create_optimizer(rl_config.critic_optimizer, rl_config.critic_lr_schedule)

# 传递给FSDP初始化避免pytree元数据不匹配
train_state, state_sharding, lazy_jit_creator = init_acrlpd_fsdp_training(
    rl_config, mesh, rng, data_sharding, 
    global_pi0_tx=global_pi0_tx,
    global_critic_tx=global_critic_tx
)
```

### 2.3 GPU内存优化和监控

#### 集成memory_monitor.py
```python
from utils.memory_monitor import memory_monitor, log_memory_usage

# 训练过程中的内存监控
log_memory_usage(step, train_state, phase="training")
memory_monitor.checkpoint_memory(f"training_step_{step}", train_state)
```

## 阶段 3: Checkpoint机制统一

**目标**: 复用agents_v2的OpenPI兼容checkpoint机制

### 3.1 创建training_v2/checkpointing.py

#### OpenPI兼容保存接口
```python
def save_unified_checkpoints(
    agent: ACRLPDPi0Agent,
    step: int,
    checkpoint_dir: str
) -> Dict[str, str]:
    """统一checkpoint保存 - 支持OpenPI和ACRLPD格式"""
    
    # 1. 保存完整ACRLPD训练状态
    full_checkpoint_path = f"{checkpoint_dir}/acrlpd_full_step_{step}"
    save_acrlpd_full_state(agent, full_checkpoint_path)
    
    # 2. 保存π₀ OpenPI兼容格式（关键：便于推理）
    openpi_checkpoint_path = f"{checkpoint_dir}/pi0_openpi_step_{step}"
    openpi_train_state = agent.create_openpi_train_state()
    save_openpi_train_state(openpi_train_state, openpi_checkpoint_path)
    
    return {
        'full_checkpoint': full_checkpoint_path,
        'openpi_checkpoint': openpi_checkpoint_path
    }
```

### 3.2 OpenPI π₀格式验证
```python
def verify_openpi_checkpoint_compatibility(checkpoint_path: str) -> bool:
    """验证π₀ checkpoint与OpenPI库的兼容性"""
    
    # 加载checkpoint并验证结构
    loaded_state = load_openpi_train_state(checkpoint_path)
    
    # 验证必要字段存在
    required_fields = ['params', 'step', 'config']
    return all(field in loaded_state for field in required_fields)
```

## 阶段 4: 数据流统一

**目标**: data_v2与新架构的完全集成

### 4.1 data_v2接口标准化

#### 统一数据加载器接口
```python
# data_v2/acrlpd_data_loader.py接口标准化
def create_acrlpd_data_loader(
    rl_config: RLTrainConfig,           # 统一配置入口
    batch_size: int = 128,
    episodes_per_memory_pool: int = 64,
    device_sharding: Optional[jax.sharding.Sharding] = None,
    **kwargs
) -> ACRLPDDataLoader:
    """创建与agents_v2兼容的数据加载器"""
```

### 4.2 Q-chunking格式验证
```python
# 确保数据格式与agents_v2期望一致
def validate_qchunking_batch_format(batch: Dict[str, jnp.ndarray]) -> bool:
    """验证Q-chunking批次格式与agents_v2的兼容性"""
    
    required_keys = [
        'observations', 'next_observations', 
        'actions', 'rewards', 'masks', 'valid', 
        'terminals', 'next_terminal', 'sequence_mask'
    ]
    return all(key in batch for key in required_keys)
```

## 阶段 5: 脚本和配置清理

**目标**: 完善scripts和配置系统，删除推理相关代码

### 5.1 scripts/train_acrlpd_v2.py重构

#### 统一训练入口
```python
def main():
    # 1. 加载统一配置
    rl_config = load_rl_config(args)
    
    # 2. 设置JAX+FSDP环境
    mesh, data_sharding, replicated_sharding = setup_jax_environment_with_fsdp(args, rl_config)
    
    # 3. 创建全局优化器（解决pytree一致性）
    global_pi0_tx, global_critic_tx = create_global_optimizers(rl_config)
    
    # 4. 统一FSDP初始化
    train_state, state_sharding, fsdp_train_step_fn = init_acrlpd_fsdp_training(
        rl_config, mesh, rng, data_sharding, 
        global_pi0_tx=global_pi0_tx, global_critic_tx=global_critic_tx
    )
    
    # 5. 创建轻量级agent和trainer
    agent = create_acrlpd_pi0_agent_from_rl_config(rl_config, rng)
    dataloader = create_acrlpd_data_loader(rl_config, rl_config.batch_size, data_sharding=data_sharding)
    trainer = ACRLPDTrainer(agent, dataloader, rl_config)
    
    # 6. 设置FSDP状态
    trainer.setup_fsdp_state(train_state, state_sharding, fsdp_train_step_fn)
    
    # 7. 执行训练
    trained_agent = trainer.train()
```

### 5.2 删除推理相关代码

#### 清理推理功能
- ✂️ **删除**: 所有`inference`相关函数和类
- ✂️ **删除**: 环境评估代码（evaluation函数）  
- ✂️ **删除**: 策略服务器相关代码
- ✅ **保留**: π₀ checkpoint保存（OpenPI格式，便于推理时使用）

## 📋 详细实施检查清单

### 阶段1检查清单：接口统一
- [ ] 修复training_v2/training_loop.py的导入错误
- [ ] 删除training_v2中重复的损失计算函数
- [ ] 修复scripts/train_acrlpd_pi0.py的导入和工厂函数调用
- [ ] 重构ACRLPDTrainer类以适配agents_v2接口
- [ ] 验证Agent创建和训练步骤的正常工作

### 阶段2检查清单：FSDP统一
- [ ] 创建training_v2/fsdp_support.py统一FSDP初始化
- [ ] 解决pytree一致性问题（全局优化器）
- [ ] 集成GPU内存监控和分析
- [ ] 验证8卡FSDP分片正确工作
- [ ] 测试内存使用优化效果

### 阶段3检查清单：Checkpoint统一
- [ ] 创建training_v2/checkpointing.py统一保存接口
- [ ] 实现OpenPI π₀格式兼容保存
- [ ] 验证checkpoint与OpenPI库兼容性
- [ ] 测试checkpoint加载和恢复训练
- [ ] 确认π₀ checkpoint可直接复制到ALOHA

### 阶段4检查清单：数据流统一
- [ ] 标准化data_v2/acrlpd_data_loader.py接口
- [ ] 验证Q-chunking格式与agents_v2兼容
- [ ] 优化数据加载性能和FSDP分片
- [ ] 测试数据流完整性
- [ ] 验证LeRobot→OpenPI转换正确性

### 阶段5检查清单：最终清理
- [ ] 重构scripts/train_acrlpd_v2.py为统一入口
- [ ] 删除所有推理相关代码
- [ ] 清理残留的重复功能
- [ ] 优化import依赖关系
- [ ] 完善文档和使用说明

## 🎯 成功验证标准

### 功能验证
1. **agents_v2 ↔ training_v2协同**: 无接口错误，正常训练
2. **FSDP分片**: 8卡训练内存均匀，无复制问题
3. **π₀ checkpoint**: 可直接复制到ALOHA进行推理
4. **训练收敛**: 与原系统一致的训练效果
5. **内存优化**: GPU内存使用合理，无OOM错误

### 架构验证
1. **无重复功能**: 损失计算、配置管理等无重复
2. **清晰职责**: agents算法、training基础设施边界明确
3. **接口简单**: 统一的参数传递和调用方式
4. **依赖清晰**: import关系简单明了

---

**实施优先级**: Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5

**预计完成时间**: Stage 1 (高优先级，立即开始) → 其他阶段按需推进

#### 解决方案：

**文件: `training_v2/acrlpd_train_state.py`**
```python
# 重构：统一的状态转换接口

@dataclasses.dataclass(frozen=True)
class ACRLPDTrainState:
    """统一的ACRLPD训练状态 - 兼容agents_v2新架构"""
    
    # 基础状态
    step: int
    
    # π₀ 组件（OpenPI兼容）
    pi0_params: nnx.State
    pi0_opt_state: Any
    pi0_ema_params: Optional[nnx.State] = None
    
    # Critic 组件（FSDP分片兼容）
    critic_params: nnx.State  
    critic_opt_state: Any
    critic_target_params: Optional[nnx.State] = None  # Target网络参数
    
    # Temperature 组件（可选）
    temperature_params: Optional[nnx.State] = None
    temperature_opt_state: Optional[Any] = None
    
    # 配置信息（JIT兼容）
    config: ACRLPDJITConfig
    
    # π₀ checkpoint保存兼容性
    openpi_checkpoint_params: Optional[nnx.State] = None  # π₀参数用于OpenPI格式保存

def create_acrlpd_train_state_from_agent(
    agent: "ACRLPDPi0Agent",  # 来自agents_v2
    config_dict: Dict[str, Any]
) -> ACRLPDTrainState:
    """
    从重构后的Agent创建FSDP兼容的训练状态
    
    这个函数是agents_v2和training_v2之间的桥梁
    """
    # 提取Agent的所有组件状态
    pi0_params = nnx.state(agent.pi0_model, nnx.Param)
    critic_params = nnx.state(agent.critic_networks, nnx.Param)
    
    # 创建JIT兼容配置
    jit_config = ACRLPDJITConfig(
        critic_weight=agent.critic_weight,
        actor_weight=agent.actor_weight,
        bc_loss_weight=agent.bc_weight,
        horizon_length=agent.horizon_length,
        discount=agent.discount,
        q_aggregation=agent.q_aggregation,
        real_action_dim=agent.real_action_dim,
        target_update_tau=agent.target_update_tau
    )
    
    # 创建OpenPI checkpoint兼容的参数（仅π₀）
    openpi_checkpoint_params = pi0_params
    
    return ACRLPDTrainState(
        step=agent.step,
        pi0_params=pi0_params,
        pi0_opt_state=agent.pi0_optimizer_state.value,
        pi0_ema_params=getattr(agent, '_pi0_ema_params', None),
        critic_params=critic_params,
        critic_opt_state=agent.critic_optimizer_state.value,
        critic_target_params=nnx.state(agent.critic_networks.target_networks, nnx.Param),
        temperature_params=nnx.state(agent.temperature_module, nnx.Param) if agent.temperature_module else None,
        temperature_opt_state=agent.temperature_optimizer_state.value if agent.temperature_module else None,
        config=jit_config,
        openpi_checkpoint_params=openpi_checkpoint_params
    )

def update_agent_from_acrlpd_train_state(
    agent: "ACRLPDPi0Agent",
    train_state: ACRLPDTrainState
) -> "ACRLPDPi0Agent":
    """
    将FSDP训练后的状态更新回Agent
    """
    # 更新步数
    agent._step = int(train_state.step)
    
    # 更新π₀组件
    nnx.update(agent.pi0_model, train_state.pi0_params)
    agent.pi0_optimizer_state.value = train_state.pi0_opt_state
    if train_state.pi0_ema_params is not None:
        agent._pi0_ema_params = train_state.pi0_ema_params
    
    # 更新Critic组件
    nnx.update(agent.critic_networks, train_state.critic_params)
    agent.critic_optimizer_state.value = train_state.critic_opt_state
    if train_state.critic_target_params is not None:
        nnx.update(agent.critic_networks.target_networks, train_state.critic_target_params)
    
    # 更新Temperature组件
    if agent.temperature_module and train_state.temperature_params:
        nnx.update(agent.temperature_module, train_state.temperature_params)
        if train_state.temperature_opt_state:
            agent.temperature_optimizer_state.value = train_state.temperature_opt_state
    
    return agent
```

### 1.2 Training Loop 重构

**文件: `training_v2/training_loop.py`**
```python
# 重构：适配agents_v2的新架构

class ACRLPDTrainingLoop:
    """统一的ACRLPD训练循环 - 协同agents_v2和training_v2"""
    
    def __init__(
        self,
        agent: "ACRLPDPi0Agent",  # 来自agents_v2，已内化损失计算
        data_loader: Any,
        enable_fsdp: bool = False,
        enable_wandb: bool = True
    ):
        self.agent = agent
        self.data_loader = data_loader
        self.enable_fsdp = enable_fsdp
        self.enable_wandb = enable_wandb
        
        # 根据是否启用FSDP选择训练模式
        if enable_fsdp:
            self._setup_fsdp_training()
        else:
            self._setup_standard_training()
    
    def _setup_fsdp_training(self):
        """设置FSDP分布式训练"""
        # 创建FSDP兼容的训练状态
        config_dict = {
            'critic_weight': self.agent.critic_weight,
            'actor_weight': self.agent.actor_weight,
            'bc_loss_weight': self.agent.bc_weight,
            'horizon_length': self.agent.horizon_length,
            'discount': self.agent.discount,
            'real_action_dim': self.agent.real_action_dim
        }
        
        self.train_state = create_acrlpd_train_state_from_agent(self.agent, config_dict)
        
        # 设置FSDP分片
        from . import acrlpd_sharding
        self.train_state = acrlpd_sharding.setup_fsdp_sharding(self.train_state)
    
    def _setup_standard_training(self):
        """设置标准单机训练"""
        self.train_state = None  # 直接使用Agent的内部状态
    
    def train_step(self, batch: Dict[str, jnp.ndarray], rng: jnp.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        统一的训练步骤 - 自动选择FSDP或标准模式
        """
        if self.enable_fsdp:
            return self._fsdp_train_step(batch, rng)
        else:
            return self._standard_train_step(batch, rng)
    
    def _standard_train_step(self, batch: Dict[str, jnp.ndarray], rng: jnp.ndarray) -> Tuple[float, Dict[str, Any]]:
        """标准训练步骤 - 直接使用Agent的内化损失计算"""
        
        # 使用Agent内化的损失计算和梯度更新
        updated_agent, loss_info = self.agent.train_step(batch, rng)
        self.agent = updated_agent
        
        return loss_info['total_loss'], loss_info
    
    def _fsdp_train_step(self, batch: Dict[str, jnp.ndarray], rng: jnp.ndarray) -> Tuple[float, Dict[str, Any]]:
        """FSDP训练步骤 - 使用分片状态"""
        
        # 在FSDP上下文中使用Agent的损失计算
        def fsdp_loss_fn(train_state):
            # 临时从训练状态重建Agent以计算损失
            temp_agent = update_agent_from_acrlpd_train_state(self.agent, train_state)
            total_loss, loss_info = temp_agent.compute_loss(batch, rng)
            return total_loss, loss_info
        
        # 应用FSDP梯度更新
        (total_loss, loss_info), updated_train_state = self._apply_fsdp_gradients(
            fsdp_loss_fn, self.train_state, batch, rng
        )
        
        # 更新训练状态
        self.train_state = updated_train_state
        
        # 同步Agent状态（可选，用于监控）
        if hasattr(self, '_sync_agent_state') and self._sync_agent_state:
            self.agent = update_agent_from_acrlpd_train_state(self.agent, self.train_state)
        
        return total_loss, loss_info
    


### 1.3 FSDP Sharding 适配

**文件: `training_v2/acrlpd_sharding.py`**
```python
# 重构：适配新的ACRLPDTrainState结构

def setup_fsdp_sharding(train_state: ACRLPDTrainState) -> ACRLPDTrainState:
    """
    为新的训练状态设置FSDP分片
    
    适配agents_v2的多组件架构
    """
    
    # π₀组件分片（大模型，需要细粒度分片）
    pi0_sharding_spec = sharding.create_fsdp_sharding_spec(
        train_state.pi0_params,
        shard_strategy='model_parallel',  # 模型并行分片
        min_shard_size=1024 * 1024  # 1MB最小分片
    )
    
    # Critic组件分片（相对较小，可以复制或粗粒度分片）
    critic_sharding_spec = sharding.create_fsdp_sharding_spec(
        train_state.critic_params,
        shard_strategy='data_parallel',  # 数据并行复制
        min_shard_size=512 * 1024  # 512KB最小分片
    )
    
    # 应用分片
    sharded_train_state = jax.tree.map(
        lambda x, spec: sharding.apply_sharding(x, spec),
        train_state,
        {
            'pi0_params': pi0_sharding_spec,
            'pi0_opt_state': pi0_sharding_spec,
            'critic_params': critic_sharding_spec,
            'critic_opt_state': critic_sharding_spec,
            'temperature_params': None,  # 小参数不分片
            'temperature_opt_state': None
        }
    )
    
    return sharded_train_state

def create_fsdp_mesh() -> jax.sharding.Mesh:
    """
    创建适合ACRLPD的FSDP mesh
    """
    devices = jax.devices()
    if len(devices) == 1:
        # 单机模式
        mesh = jax.sharding.Mesh(devices, ('data',))
    elif len(devices) <= 8:
        # 单节点多GPU
        mesh = jax.sharding.Mesh(devices, ('fsdp',))
    else:
        # 多节点
        mesh = jax.sharding.Mesh(
            devices.reshape((-1, 8)), 
            ('data', 'model')
        )
    
    return mesh
```

---

## 阶段 2: 复用现有Checkpoint机制 ✅ **已确认满足要求**

### 🎯 实施状态：COMPLETED

**分析结果**: 当前agents中的checkpoint保存机制已经满足OpenPI兼容性要求，无需重新设计。

**现有机制分析**:
- ✅ **`create_train_state()`** - 创建OpenPI兼容TrainState，仅包含π₀参数和模型定义
- ✅ **`save_component_checkpoints()`** - 分别保存π₀、critic等各组件的完整checkpoint
- ✅ **格式兼容**: TrainState格式与OpenPI库标准完全一致
- ✅ **EMA支持**: 自动使用EMA参数作为推理权重（如果启用）
- ✅ **部署友好**: π₀ checkpoint可直接拷贝到ALOHA机器人

### 2.1 π₀ Checkpoint OpenPI格式保存

**文件: `training_v2/pi0_checkpoint_saver.py`** (新建)
```python
"""
π₀ Checkpoint OpenPI格式保存器

专门用于保存π₀模型为OpenPI标准格式，方便拷贝到ALOHA机器人
"""

import openpi.training.utils as training_utils
from typing import Optional

def save_pi0_openpi_checkpoint(
    acrlpd_state: ACRLPDTrainState,
    checkpoint_path: str,
    step: int
):
    """
    保存π₀模型为OpenPI标准格式
    
    Args:
        acrlpd_state: ACRLPD训练状态
        checkpoint_path: 保存路径
        step: 训练步数
    """
    
    # 提取π₀组件（仅π₀参数）
    pi0_params = acrlpd_state.pi0_params
    pi0_ema_params = acrlpd_state.pi0_ema_params
    
    # 使用EMA参数作为主要参数（如果存在）
    main_params = pi0_ema_params if pi0_ema_params is not None else pi0_params
    
    # 创建最小优化器（OpenPI格式要求）
    dummy_tx = optax.sgd(learning_rate=1e-4)
    dummy_opt_state = jax.tree.map(
        lambda x: jnp.zeros((), dtype=x.dtype),
        jax.eval_shape(lambda: dummy_tx.init(main_params))
    )
    
    # 创建OpenPI TrainState
    openpi_train_state = training_utils.TrainState(
        step=step,
        params=main_params,  # π₀主要参数
        model_def=acrlpd_state.pi0_model_def,  # π₀模型定义
        opt_state=dummy_opt_state,  # 占位符优化器状态
        tx=dummy_tx,  # 占位符优化器
        ema_decay=0.999 if pi0_ema_params is not None else None,
        ema_params=pi0_ema_params  # EMA参数（如果有）
    )
    
    # 保存为OpenPI标准格式
    checkpoint_dir = Path(checkpoint_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用OpenPI标准保存函数
    from openpi.training.checkpoints import save_train_state
    save_train_state(openpi_train_state, str(checkpoint_dir))
    
    # 保存简单元数据
    metadata = {
        'step': step,
        'model_type': 'pi0',
        'openpi_compatible': True,
        'has_ema': pi0_ema_params is not None
    }
    
    with open(checkpoint_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ π₀ OpenPI格式checkpoint已保存: {checkpoint_dir}")
```

### 2.2 统一的配置管理

**文件: `config.py`** (更新)
```python
# 扩展：添加OpenPI兼容性配置

@dataclasses.dataclass(frozen=True)
class OpenPICompatibilityConfig:
    """OpenPI兼容性配置"""
    
    # 推理相关
    enable_ema_for_inference: bool = True
    ema_decay_rate: float = 0.999
    
    # Checkpoint格式
    save_openpi_checkpoints: bool = True
    validate_openpi_compatibility: bool = True
    

@dataclasses.dataclass(frozen=True) 
class ACRLPDv2Config:
    """AC Training V2 统一配置"""
    
    # 基础配置
    name: str
    batch_size: int
    
    # 组件配置
    model: Any  # π₀模型配置
    critic: CriticConfig
    acrlpd: ACRLPDConfig
    qchunking: QChunkingConfig
    
    # 训练配置
    actor_lr_schedule: Any
    critic_lr_schedule: Any
    actor_optimizer: Any
    critic_optimizer: Any
    
    # 兼容性配置
    openpi_compatibility: OpenPICompatibilityConfig
    
    # FSDP配置
    enable_fsdp: bool = False
    fsdp_config: Optional[FSDPConfig] = None

# 预定义配置（更新）
ACRLPD_V2_ALOHA_FOLD = ACRLPDv2Config(
    name="acrlpd_v2_aloha_fold",
    batch_size=256,
    model=PI0_CONFIG_FOLD,
    critic=CriticConfig(
        num_critics=10,
        hidden_dims=[512, 512, 512],
        use_layer_norm=True,
        dropout_rate=0.1
    ),
    acrlpd=ACRLPDConfig(
        num_action_samples=4,  # Best-of-N
        discount=0.99,
        bc_loss_weight=0.05,
        use_adaptive_temperature=True
    ),
    qchunking=QChunkingConfig(
        action_dim=14,  # ALOHA真实动作维度
        horizon_length=20
    ),
    actor_lr_schedule=create_cosine_schedule(3e-5, 50000),
    critic_lr_schedule=create_cosine_schedule(3e-4, 50000),
    actor_optimizer=_optimizer.AdamW(weight_decay=1e-4),
    critic_optimizer=_optimizer.AdamW(weight_decay=1e-4),
    openpi_compatibility=OpenPICompatibilityConfig(
        enable_ema_for_inference=True,
        save_openpi_checkpoints=True,
        validate_openpi_compatibility=True
    ),
    enable_fsdp=False
)
```

---

## 阶段 3: FSDP 分布式训练优化 (Medium Priority) 

### 🎯 实施状态：PENDING

**目标**: 优化FSDP分布式训练性能，确保多GPU和多节点训练的稳定性。

### 3.1 内存优化的FSDP策略

**文件: `training_v2/fsdp_optimization.py`** (新建)
```python
"""
FSDP分布式训练优化

针对ACRLPD多组件架构的FSDP优化策略
"""

class ACRLPDFSDPOptimizer:
    """ACRLPD专用的FSDP优化器"""
    
    def __init__(self, config: ACRLPDv2Config):
        self.config = config
        self.mesh = self._create_optimal_mesh()
        
    def _create_optimal_mesh(self) -> jax.sharding.Mesh:
        """
        根据硬件配置创建最优的mesh
        """
        devices = jax.devices()
        total_devices = len(devices)
        
        if total_devices == 1:
            # 单GPU：不分片
            return jax.sharding.Mesh([devices[0]], ('replica',))
        elif total_devices <= 8:
            # 单节点：π₀模型并行 + Critic数据并行
            return jax.sharding.Mesh(devices, ('model',))
        else:
            # 多节点：2D mesh
            nodes = total_devices // 8
            gpus_per_node = 8
            mesh_shape = (nodes, gpus_per_node)
            return jax.sharding.Mesh(
                devices.reshape(mesh_shape), 
                ('data', 'model')
            )
    
    def create_optimized_sharding_strategy(
        self, 
        train_state: ACRLPDTrainState
    ) -> Dict[str, jax.sharding.NamedSharding]:
        """
        创建优化的分片策略
        """
        strategies = {}
        
        # π₀组件：大模型，需要模型并行
        pi0_strategy = jax.sharding.NamedSharding(
            self.mesh, 
            jax.sharding.PartitionSpec('model',)
        )
        strategies['pi0_params'] = pi0_strategy
        strategies['pi0_opt_state'] = pi0_strategy
        strategies['pi0_ema_params'] = pi0_strategy
        
        # Critic组件：小模型，数据并行复制
        critic_strategy = jax.sharding.NamedSharding(
            self.mesh,
            jax.sharding.PartitionSpec(None,)  # 复制到所有设备
        )
        strategies['critic_params'] = critic_strategy
        strategies['critic_opt_state'] = critic_strategy
        
        # Temperature：非常小，复制
        temp_strategy = jax.sharding.NamedSharding(
            self.mesh,
            jax.sharding.PartitionSpec(None,)
        )
        strategies['temperature_params'] = temp_strategy
        strategies['temperature_opt_state'] = temp_strategy
        
        return strategies
    
    def optimize_gradient_accumulation(
        self,
        global_batch_size: int,
        num_devices: int
    ) -> Tuple[int, int]:
        """
        优化梯度累积策略
        
        Returns:
            (per_device_batch_size, gradient_accumulation_steps)
        """
        # 计算每设备批次大小
        per_device_batch_size = global_batch_size // num_devices
        
        # 内存限制检查
        max_per_device_batch = self._estimate_max_batch_size()
        
        if per_device_batch_size <= max_per_device_batch:
            # 无需梯度累积
            return per_device_batch_size, 1
        else:
            # 需要梯度累积
            grad_acc_steps = (per_device_batch_size + max_per_device_batch - 1) // max_per_device_batch
            adjusted_batch_size = per_device_batch_size // grad_acc_steps
            return adjusted_batch_size, grad_acc_steps
    
    def _estimate_max_batch_size(self) -> int:
        """
        估算单设备最大批次大小
        """
        # 基于GPU内存和模型大小的启发式估算
        gpu_memory_gb = self._get_gpu_memory_gb()
        
        if gpu_memory_gb >= 40:  # A100
            return 64
        elif gpu_memory_gb >= 24:  # 4090
            return 32
        elif gpu_memory_gb >= 16:  # V100
            return 16
        else:
            return 8

    def setup_fsdp_training(
        self, 
        train_state: ACRLPDTrainState
    ) -> Tuple[ACRLPDTrainState, Dict[str, Any]]:
        """
        设置优化的FSDP训练
        """
        # 创建分片策略
        sharding_strategies = self.create_optimized_sharding_strategy(train_state)
        
        # 应用分片
        sharded_train_state = self._apply_sharding(train_state, sharding_strategies)
        
        # 创建训练配置
        training_config = {
            'mesh': self.mesh,
            'sharding_strategies': sharding_strategies,
            'gradient_accumulation': self.optimize_gradient_accumulation(
                self.config.batch_size, 
                len(jax.devices())
            )
        }
        
        return sharded_train_state, training_config
```

### 3.2 Multi-Host 训练支持

**文件: `training_v2/multi_host_training.py`** (新建)
```python
"""
多节点分布式训练支持
"""

class MultiHostACRLPDTrainer:
    """多节点ACRLPD训练器"""
    
    def __init__(self, config: ACRLPDv2Config):
        self.config = config
        self.setup_multi_host()
    
    def setup_multi_host(self):
        """设置多节点环境"""
        # 初始化JAX分布式
        jax.distributed.initialize()
        
        # 获取主机信息
        self.process_index = jax.process_index()
        self.process_count = jax.process_count()
        self.local_device_count = jax.local_device_count()
        
        logger.info(f"Multi-host setup: process {self.process_index}/{self.process_count}, "
                   f"local devices: {self.local_device_count}")
    
    def create_global_mesh(self) -> jax.sharding.Mesh:
        """创建全局mesh"""
        global_devices = jax.devices()
        
        # 重塑为2D mesh: [num_hosts, devices_per_host]
        devices_per_host = len(global_devices) // self.process_count
        mesh_shape = (self.process_count, devices_per_host)
        
        global_mesh = jax.sharding.Mesh(
            global_devices.reshape(mesh_shape),
            ('hosts', 'devices')
        )
        
        return global_mesh
    
    def sync_train_state_across_hosts(
        self, 
        train_state: ACRLPDTrainState
    ) -> ACRLPDTrainState:
        """跨主机同步训练状态"""
        
        # 使用JAX的集合通信同步参数
        def sync_component(component):
            return jax.experimental.multihost_utils.sync_global_devices(component)
        
        synced_train_state = jax.tree.map(sync_component, train_state)
        
        return synced_train_state
    
    def save_global_checkpoint(
        self, 
        train_state: ACRLPDTrainState, 
        checkpoint_path: str
    ):
        """保存全局checkpoint（仅在主进程）"""
        
        if self.process_index == 0:
            # 收集所有主机的状态
            global_train_state = self.sync_train_state_across_hosts(train_state)
            
            # 保存checkpoint
            checkpoint_manager = OpenPICheckpointManager(checkpoint_path)
            checkpoint_manager.save_openpi_checkpoint(
                global_train_state, 
                int(global_train_state.step)
            )
            
            logger.info(f"Global checkpoint saved: {checkpoint_path}")
        
        # 等待主进程完成保存
        jax.experimental.multihost_utils.sync_global_devices("checkpoint_sync")
```

---

## 阶段 4: 数据处理和脚本系统重构 (Medium Priority)

### 🎯 实施状态：PENDING

**目标**: 重构data_v2和scripts，确保与新的agents_v2和training_v2系统协同工作。

### 4.1 数据加载器适配

**文件: `data_v2/acrlpd_data_loader_v2.py`** (新建)
```python
"""
ACRLPD V2 数据加载器

适配新的agents_v2架构和training_v2需求
"""

class ACRLPDv2DataLoader:
    """
    ACRLPD V2 数据加载器
    
    特点：
    - 支持动态批次大小（FSDP适配）
    - 内存优化的数据管道
    - OpenPI兼容的数据格式
    """
    
    def __init__(self, config: ACRLPDv2Config):
        self.config = config
        self.setup_data_pipeline()
    
    def setup_data_pipeline(self):
        """设置优化的数据管道"""
        
        # 计算最优的数据加载参数
        num_devices = len(jax.devices())
        global_batch_size = self.config.batch_size
        
        # FSDP下的批次配置
        if self.config.enable_fsdp:
            per_device_batch_size = global_batch_size // num_devices
            prefetch_factor = 2  # 减少预取降低内存使用
        else:
            per_device_batch_size = global_batch_size
            prefetch_factor = 4
        
        self.per_device_batch_size = per_device_batch_size
        self.prefetch_factor = prefetch_factor
        
        # 设置数据转换管道
        self.transforms = self._create_data_transforms()
    
    def _create_data_transforms(self) -> List[Callable]:
        """创建数据转换管道"""
        transforms = []
        
        # 1. 基础格式转换
        transforms.append(self._convert_to_acrlpd_format)
        
        # 2. 动作维度处理（OpenPI 32维 -> 真实动作维度）
        transforms.append(partial(
            self._process_action_dimensions,
            target_dim=self.config.qchunking.action_dim
        ))
        
        # 3. Q-chunking准备
        transforms.append(partial(
            self._prepare_qchunking_data,
            horizon_length=self.config.qchunking.horizon_length
        ))
        
        # 4. 观察编码预处理（如果启用）
        if self.config.enable_observation_preprocessing:
            transforms.append(self._preprocess_observations)
        
        return transforms
    
    def _convert_to_acrlpd_format(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """转换为ACRLPD格式"""
        
        # 确保有必要的键
        required_keys = ['observations', 'actions', 'rewards', 'next_observations', 'masks']
        for key in required_keys:
            if key not in batch:
                raise ValueError(f"Missing required key: {key}")
        
        # 类型转换
        converted_batch = {}
        for key, value in batch.items():
            if isinstance(value, np.ndarray):
                converted_batch[key] = jnp.array(value)
            else:
                converted_batch[key] = value
        
        return converted_batch
    
    def _process_action_dimensions(
        self, 
        batch: Dict[str, Any], 
        target_dim: int
    ) -> Dict[str, Any]:
        """处理动作维度兼容性"""
        
        actions = batch['actions']
        
        # 检查当前动作维度
        if actions.shape[-1] > target_dim:
            # 从32维截断到目标维度
            batch['actions'] = actions[..., :target_dim]
            logger.debug(f"Action dimensions truncated: {actions.shape[-1]} -> {target_dim}")
        elif actions.shape[-1] < target_dim:
            # 填充到目标维度
            padding_shape = actions.shape[:-1] + (target_dim - actions.shape[-1],)
            padding = jnp.zeros(padding_shape)
            batch['actions'] = jnp.concatenate([actions, padding], axis=-1)
            logger.debug(f"Action dimensions padded: {actions.shape[-1]} -> {target_dim}")
        
        return batch
    
    def create_data_iterator(
        self, 
        dataset_path: str, 
        split: str = "train",
        shuffle: bool = True
    ) -> Iterator[Dict[str, jnp.ndarray]]:
        """
        创建数据迭代器
        
        支持FSDP和标准训练模式
        """
        
        # 加载LeRobot数据集
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        
        dataset = LeRobotDataset(
            repo_id=dataset_path,
            split=split,
            batch_size=self.per_device_batch_size,
            shuffle=shuffle
        )
        
        # 创建数据迭代器
        data_iterator = iter(dataset)
        
        # 应用变换管道
        def transform_batch(batch):
            for transform in self.transforms:
                batch = transform(batch)
            return batch
        
        # 包装迭代器
        transformed_iterator = map(transform_batch, data_iterator)
        
        return transformed_iterator
```

### 4.2 统一训练脚本

**文件: `scripts/train_acrlpd_v2.py`** (新建)
```python
"""
ACRLPD V2 统一训练脚本

支持单机和FSDP分布式训练
"""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="ACRLPD V2 Training")
    
    # 基础配置
    parser.add_argument("--config", required=True, help="配置名称")
    parser.add_argument("--exp_name", required=True, help="实验名称")
    parser.add_argument("--data_path", required=True, help="数据路径")
    
    # 训练模式
    parser.add_argument("--enable_fsdp", action="store_true", help="启用FSDP分布式训练")
    parser.add_argument("--num_steps", type=int, default=50000, help="训练步数")
    parser.add_argument("--save_interval", type=int, default=5000, help="保存间隔")
    
    # 系统配置
    parser.add_argument("--no_wandb", action="store_true", help="禁用W&B")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    parser.add_argument("--overwrite", action="store_true", help="覆盖现有实验")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(debug=args.debug)
    
    # 加载配置
    config = load_config(args.config)
    if args.enable_fsdp:
        config = dataclasses.replace(config, enable_fsdp=True)
    
    logger.info(f"🚀 Starting ACRLPD V2 training: {args.exp_name}")
    logger.info(f"📊 Config: {config.name}")
    logger.info(f"🔧 FSDP: {'enabled' if args.enable_fsdp else 'disabled'}")
    
    # 设置实验目录
    exp_dir = Path("experiments") / args.exp_name
    if exp_dir.exists() and not args.overwrite:
        raise ValueError(f"Experiment {args.exp_name} exists. Use --overwrite to overwrite.")
    
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化系统
    rng = jax.random.PRNGKey(42)
    
    # 创建Agent（来自agents_v2）
    from agents_v2 import create_acrlpd_pi0_agent_from_rl_config
    agent = create_acrlpd_pi0_agent_from_rl_config(config, rng)
    
    # 创建数据加载器
    from data_v2 import ACRLPDv2DataLoader
    data_loader = ACRLPDv2DataLoader(config)
    train_iterator = data_loader.create_data_iterator(args.data_path)
    
    # 创建训练循环（来自training_v2）
    from training_v2 import ACRLPDTrainingLoop
    training_loop = ACRLPDTrainingLoop(
        agent=agent,
        data_loader=data_loader,
        enable_fsdp=args.enable_fsdp,
        enable_wandb=not args.no_wandb
    )
    
    # 设置W&B（如果启用）
    if not args.no_wandb:
        setup_wandb(args.exp_name, config)
    
    # 主训练循环
    logger.info("🎯 Starting training loop")
    
    for step in range(args.num_steps):
        # 获取批次
        batch = next(train_iterator)
        step_rng = jax.random.fold_in(rng, step)
        
        # 训练步骤
        total_loss, loss_info = training_loop.train_step(batch, step_rng)
        
        # 记录日志
        if step % 100 == 0:
            log_training_metrics(step, total_loss, loss_info)
            
            if not args.no_wandb:
                wandb.log({
                    "step": step,
                    "total_loss": total_loss,
                    **loss_info
                })
        
        # 保存checkpoint
        if step % args.save_interval == 0 and step > 0:
            checkpoint_path = exp_dir / "checkpoints" / f"step_{step}"
            
            # 保存ACRLPD checkpoint
            training_loop.save_acrlpd_checkpoint(checkpoint_path)
            
            # 保存OpenPI兼容checkpoint
            training_loop.create_openpi_checkpoint(checkpoint_path / "openpi")
            
            logger.info(f"💾 Saved checkpoint: step {step}")
    
    logger.info("✅ Training completed successfully")

def load_config(config_name: str):
    """加载预定义配置"""
    from config import (
        ACRLPD_V2_ALOHA_FOLD, 
        ACRLPD_V2_DROID, 
        ACRLPD_V2_LIBERO
    )
    
    config_map = {
        "aloha_fold": ACRLPD_V2_ALOHA_FOLD,
        "droid": ACRLPD_V2_DROID,
        "libero": ACRLPD_V2_LIBERO
    }
    
    if config_name not in config_map:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(config_map.keys())}")
    
    return config_map[config_name]

if __name__ == "__main__":
    main()
```

---

## 阶段 5: 测试和验证系统 (Medium Priority)

### 🎯 实施状态：PENDING

**目标**: 建立完整的测试和验证系统，确保重构后的系统功能正确性和性能。

### 5.1 单元测试框架

**文件: `tests/test_agents_v2.py`** (新建)
```python
"""
Agents V2 单元测试
"""

import pytest
import jax
import jax.numpy as jnp
from agents_v2 import ACRLPDPi0Agent, create_acrlpd_pi0_agent_from_rl_config
from config import ACRLPD_V2_ALOHA_FOLD

class TestACRLPDPi0Agent:
    
    @pytest.fixture
    def agent(self):
        """创建测试Agent"""
        rng = jax.random.PRNGKey(42)
        config = ACRLPD_V2_ALOHA_FOLD
        return create_acrlpd_pi0_agent_from_rl_config(config, rng)
    
    @pytest.fixture
    def test_batch(self):
        """创建测试批次"""
        batch_size = 4
        horizon_length = 20
        action_dim = 14
        
        return {
            'observations': {
                'state': jnp.ones((batch_size, action_dim)),
                'image': {
                    'base_camera': jnp.ones((batch_size, 224, 224, 3)),
                    'wrist_camera': jnp.ones((batch_size, 224, 224, 3))
                }
            },
            'actions': jnp.ones((batch_size, horizon_length, action_dim)),
            'rewards': jnp.ones((batch_size,)),
            'next_observations': {
                'state': jnp.ones((batch_size, action_dim)),
                'image': {
                    'base_camera': jnp.ones((batch_size, 224, 224, 3)),
                    'wrist_camera': jnp.ones((batch_size, 224, 224, 3))
                }
            },
            'masks': jnp.ones((batch_size,))
        }
    
    def test_agent_creation(self, agent):
        """测试Agent创建"""
        assert isinstance(agent, ACRLPDPi0Agent)
        assert agent.num_action_samples == 4
        assert agent.horizon_length == 20
        assert agent.real_action_dim == 14
    
    def test_loss_computation(self, agent, test_batch):
        """测试损失计算"""
        rng = jax.random.PRNGKey(42)
        
        total_loss, loss_info = agent.compute_loss(test_batch, rng)
        
        # 验证损失值
        assert jnp.isfinite(total_loss)
        assert total_loss > 0
        
        # 验证损失组件
        assert 'critic_loss' in loss_info
        assert 'actor_loss' in loss_info
        assert 'bc_loss' in loss_info
        
        # 验证数值稳定性
        assert jnp.isfinite(loss_info['critic_loss'])
        assert jnp.isfinite(loss_info['actor_loss'])
        assert jnp.isfinite(loss_info['bc_loss'])
    
    def test_train_step(self, agent, test_batch):
        """测试训练步骤"""
        rng = jax.random.PRNGKey(42)
        initial_step = agent.step
        
        updated_agent, loss_info = agent.train_step(test_batch, rng)
        
        # 验证步数更新
        assert updated_agent.step == initial_step + 1
        
        # 验证返回的损失信息
        assert 'total_loss' in loss_info
        assert jnp.isfinite(loss_info['total_loss'])
    
    def test_openpi_compatibility(self, agent):
        """测试OpenPI兼容性"""
        
        # 创建OpenPI训练状态
        openpi_state = agent.create_openpi_train_state()
        
        # 验证状态结构
        assert hasattr(openpi_state, 'params')
        assert hasattr(openpi_state, 'step')
        assert hasattr(openpi_state, 'model_def')
        
        # 验证推理兼容性
        test_obs = create_dummy_observation()
        try:
            model = openpi_state.model_def.apply(openpi_state.params)
            actions = model.sample_actions(jax.random.PRNGKey(42), test_obs)
            assert actions.shape[-1] == 32  # OpenPI标准动作维度
        except Exception as e:
            pytest.fail(f"OpenPI兼容性测试失败: {e}")

def create_dummy_observation():
    """创建测试观察"""
    return {
        'state': jnp.ones((1, 14)),
        'image': {
            'base_camera': jnp.ones((1, 224, 224, 3)),
            'wrist_camera': jnp.ones((1, 224, 224, 3))
        }
    }
```

### 5.2 集成测试

**文件: `tests/test_training_integration.py`** (新建)
```python
"""
训练系统集成测试
"""

import pytest
import tempfile
from pathlib import Path

class TestTrainingIntegration:
    
    def test_standard_training_pipeline(self):
        """测试标准训练管道"""
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 设置配置
            config = ACRLPD_V2_ALOHA_FOLD
            
            # 创建组件
            rng = jax.random.PRNGKey(42)
            agent = create_acrlpd_pi0_agent_from_rl_config(config, rng)
            
            data_loader = ACRLPDv2DataLoader(config)
            training_loop = ACRLPDTrainingLoop(
                agent=agent,
                data_loader=data_loader,
                enable_fsdp=False
            )
            
            # 创建测试数据
            test_batch = create_test_batch()
            
            # 运行几个训练步骤
            for step in range(5):
                step_rng = jax.random.fold_in(rng, step)
                total_loss, loss_info = training_loop.train_step(test_batch, step_rng)
                
                assert jnp.isfinite(total_loss)
                assert total_loss > 0
            
            # 测试checkpoint保存
            checkpoint_path = Path(tmp_dir) / "checkpoint"
            training_loop.create_openpi_checkpoint(str(checkpoint_path))
            
            assert checkpoint_path.exists()
    
    @pytest.mark.skipif(len(jax.devices()) < 2, reason="需要多GPU进行FSDP测试")
    def test_fsdp_training_pipeline(self):
        """测试FSDP训练管道"""
        
        # 设置FSDP配置
        config = dataclasses.replace(ACRLPD_V2_ALOHA_FOLD, enable_fsdp=True)
        
        # 创建组件
        rng = jax.random.PRNGKey(42)
        agent = create_acrlpd_pi0_agent_from_rl_config(config, rng)
        
        data_loader = ACRLPDv2DataLoader(config)
        training_loop = ACRLPDTrainingLoop(
            agent=agent,
            data_loader=data_loader,
            enable_fsdp=True
        )
        
        # 创建测试数据
        test_batch = create_test_batch()
        
        # 运行FSDP训练步骤
        step_rng = jax.random.PRNGKey(42)
        total_loss, loss_info = training_loop.train_step(test_batch, step_rng)
        
        assert jnp.isfinite(total_loss)
        assert total_loss > 0
```

### 5.3 性能基准测试

**文件: `tests/benchmark_performance.py`** (新建)
```python
"""
性能基准测试
"""

import time
import jax

class PerformanceBenchmark:
    
    def benchmark_training_step(self, config, num_steps=100):
        """训练步骤性能基准"""
        
        # 创建组件
        rng = jax.random.PRNGKey(42)
        agent = create_acrlpd_pi0_agent_from_rl_config(config, rng)
        test_batch = create_test_batch()
        
        # 预热
        for _ in range(10):
            step_rng = jax.random.fold_in(rng, 0)
            agent.compute_loss(test_batch, step_rng)
        
        # 同步设备
        jax.block_until_ready(jnp.ones(1))
        
        # 基准测试
        start_time = time.time()
        
        for step in range(num_steps):
            step_rng = jax.random.fold_in(rng, step)
            total_loss, _ = agent.compute_loss(test_batch, step_rng)
            jax.block_until_ready(total_loss)
        
        end_time = time.time()
        
        avg_time_per_step = (end_time - start_time) / num_steps
        steps_per_second = 1.0 / avg_time_per_step
        
        return {
            'avg_time_per_step': avg_time_per_step,
            'steps_per_second': steps_per_second,
            'total_time': end_time - start_time
        }
    
    def benchmark_memory_usage(self, config):
        """内存使用基准"""
        
        # 获取初始内存
        initial_memory = get_gpu_memory_usage()
        
        # 创建Agent和数据
        rng = jax.random.PRNGKey(42)
        agent = create_acrlpd_pi0_agent_from_rl_config(config, rng)
        test_batch = create_test_batch()
        
        # 运行训练步骤
        step_rng = jax.random.PRNGKey(42)
        total_loss, _ = agent.compute_loss(test_batch, step_rng)
        jax.block_until_ready(total_loss)
        
        # 获取峰值内存
        peak_memory = get_gpu_memory_usage()
        
        return {
            'initial_memory_mb': initial_memory,
            'peak_memory_mb': peak_memory,
            'memory_increase_mb': peak_memory - initial_memory
        }

def get_gpu_memory_usage() -> float:
    """获取GPU内存使用量（MB）"""
    try:
        import nvidia_ml_py3 as nvml
        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByIndex(0)
        info = nvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024**2  # 转换为MB
    except:
        return 0.0  # 如果无法获取，返回0

if __name__ == "__main__":
    benchmark = PerformanceBenchmark()
    
    # 基准测试配置
    configs = [
        ("ALOHA_FOLD", ACRLPD_V2_ALOHA_FOLD),
    ]
    
    for name, config in configs:
        print(f"\n=== {name} 性能基准 ===")
        
        # 训练速度基准
        perf_results = benchmark.benchmark_training_step(config)
        print(f"训练步骤/秒: {perf_results['steps_per_second']:.2f}")
        print(f"平均每步时间: {perf_results['avg_time_per_step']:.3f}s")
        
        # 内存使用基准
        memory_results = benchmark.benchmark_memory_usage(config)
        print(f"内存使用: {memory_results['peak_memory_mb']:.1f}MB")
        print(f"内存增长: {memory_results['memory_increase_mb']:.1f}MB")
```

---

## 🔄 实施时间表和验证计划

### 实施优先级：

**第1周 - 接口重构 (High Priority)**
- [ ] 统一agents_v2和training_v2的状态接口
- [ ] 重构训练循环适配新Agent架构  
- [ ] 验证标准训练管道正常工作

**第2周 - OpenPI兼容性 (High Priority)**
- [ ] 实现系统级OpenPI兼容性模块
- [ ] 统一checkpoint格式和状态转换
- [ ] 验证与OpenPI推理流程的完全兼容

**第3周 - FSDP优化 (Medium Priority)**
- [ ] 优化FSDP分片策略
- [ ] 实现多节点训练支持
- [ ] 验证分布式训练性能和稳定性

**第4周 - 系统集成 (Medium Priority)**
- [ ] 重构数据加载和脚本系统
- [ ] 建立完整的测试和验证框架
- [ ] 性能基准测试和优化

### 验证检查列表：

**功能验证：**
- [ ] agents_v2和training_v2协同工作正常
- [ ] FSDP和标准训练模式都能正常运行
- [ ] OpenPI兼容性完全验证
- [ ] 训练收敛性与原系统一致

**性能验证：**
- [ ] 训练速度不低于原系统
- [ ] 内存使用优化
- [ ] 分布式训练线性扩展性

**兼容性验证：**
- [ ] 现有数据集直接可用
- [ ] OpenPI推理无需修改  
- [ ] checkpoint格式向后兼容

---

## 📈 预期收益

### 架构收益：
1. **统一接口** - agents和training系统标准化协同
2. **简化设计** - 消除过度工程化，清晰的职责分工
3. **完全兼容** - OpenPI和FSDP在系统层面的无缝集成
4. **扩展性** - 易于添加新功能和优化

### 性能收益：
1. **内存优化** - FSDP分片策略和梯度累积优化
2. **训练加速** - 多节点扩展和内存管理优化
3. **系统稳定性** - 统一状态管理和错误处理

### 开发收益：
1. **可维护性** - 清晰的模块边界和接口
2. **可测试性** - 完整的单元测试和集成测试
3. **调试友好** - 统一日志和性能监控

重构后的ac_training_v2将成为一个**现代化、高性能、完全兼容**的ACRLPD训练系统，为未来的扩展和优化奠定坚实基础。