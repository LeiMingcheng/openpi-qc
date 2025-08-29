# Q-Chunking强化学习算法技术文档

## 概述

Q-Chunking（带动作分块的Q学习）是一种新颖的强化学习算法，通过在时间扩展的动作空间上运行RL算法，利用先验数据提高探索和在线样本效率。本文档详细介绍算法的数学原理、实现细节及代码复现方法。

## 核心概念

### 1. 动作分块（Action Chunking）

动作分块是Q-Chunking的核心创新，其基本思想是预测未来多个时间步的动作序列，而非传统的单步动作。

**数学定义：**

- 传统RL：在状态 $s_t$ 下选择动作 $a_t \in \mathbb{R}^d$
- Q-Chunking：在状态 $s_t$ 下选择动作序列 $\mathbf{a}_t = [a_t, a_{t+1}, ..., a_{t+H-1}] \in \mathbb{R}^{H \times d}$

其中 $H$ 是horizon_length（动作分块长度），$d$ 是单步动作维度。

**实现机制：**

```python
# 动作重塑：从(batch_size, horizon_length, action_dim)到(batch_size, horizon_length * action_dim)
if self.config["action_chunking"]:
    batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
else:
    batch_actions = batch["actions"][..., 0, :] # 取第一个动作
```

### 2. Critic网络输入输出机制

**关键设计决策：虽然Policy输出动作块序列，但Critic返回单个标量Q值**

**Policy输出：** 动作块序列 $\mathbf{a}_t = [a_t, a_{t+1}, ..., a_{t+H-1}] \in \mathbb{R}^{H \times d}$

**Critic输入：** 状态 + flatten后的动作块

```python
# 动作处理
if self.config["action_chunking"]:
    # 从 (batch_size, horizon_length, action_dim) 
    # 到 (batch_size, horizon_length * action_dim)
    batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
else:
    batch_actions = batch["actions"][..., 0, :] # 只取第一个动作

# Critic调用：Q(s, a) - 输入状态和动作
q = self.network.select('critic')(
    batch['observations'],    # 状态: (batch_size, obs_dim)
    actions=batch_actions,    # 动作: (batch_size, H * action_dim)
    params=grad_params
)
# 内部会将状态和动作连接: concat([obs, actions], axis=-1)
```

**Critic输出：** 单个标量Q值，表示执行整个动作块的累积价值

```python
# 网络定义：最后一层输出维度为1
value_net = mlp_class((*self.hidden_dims, 1), activate_final=False)
# 输出标量
v = self.value_net(inputs).squeeze(-1)
```

### 3. 时间折扣修正

传统RL的时间折扣为 $\gamma$，Q-Chunking中需要修正为 $\gamma^H$：

**原理：**

- 传统：$Q(s_t, a_t) = r_t + \gamma Q(s_{t+1}, a_{t+1})$
- Q-Chunking：$Q(s_t, \mathbf{a}_t) = \sum_{i=0}^{H-1} \gamma^i r_{t+i} + \gamma^H Q(s_{t+H}, \mathbf{a}_{t+H})$

**实现：**

```python
target_q = batch['rewards'][..., -1] + \  # 使用累积折扣奖励
    (self.config['discount'] ** self.config["horizon_length"]) * batch['masks'][..., -1] * next_q
```

## Flow Q-Learning（FQL）算法

### 1. 算法原理

Flow Q-Learning结合了流匹配（Flow Matching）和Q学习，用于在连续动作空间中进行高效策略学习。

**核心思想：**

- 使用流匹配学习从噪声分布到目标动作分布的连续映射
- 结合Q函数指导策略改进

### 2. 流匹配过程

**数学公式：**

```
x_t = (1-t) * x_0 + t * x_1
```

其中：

- $x_0$：初始噪声（高斯分布采样）
- $x_1$：目标动作（来自数据集）
- $t \in [0,1]$：插值参数
- $x_t$：插值轨迹上的点

**速度场学习：**

```
v_θ(s, x_t, t) = x_1 - x_0
```

### 3. Actor损失函数

```python
def actor_loss(self, batch, grad_params, rng):
    # BC流损失
    x_0 = jax.random.normal(x_rng, (batch_size, action_dim))  # 噪声
    x_1 = batch_actions  # 目标动作
    t = jax.random.uniform(t_rng, (batch_size, 1))  # 时间
    x_t = (1 - t) * x_0 + t * x_1  # 插值
    vel = x_1 - x_0  # 真实速度场
    
    # 预测速度场
    pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)
    
    # 损失计算（考虑动作分块的有效性掩码）
    if self.config["action_chunking"]:
        bc_flow_loss = jnp.mean(
            jnp.reshape(
                (pred - vel) ** 2, 
                (batch_size, self.config["horizon_length"], self.config["action_dim"]) 
            ) * batch["valid"][..., None]
        )
    else:
        bc_flow_loss = jnp.mean(jnp.square(pred - vel))
```

### 4. 动作采样策略

**Best-of-N采样：**

```python
elif self.config["actor_type"] == "best-of-n":
    # 生成多个候选动作
    noises = jax.random.normal(rng, (batch_size, num_samples, action_dim))
    actions = self.compute_flow_actions(observations, noises)  # 通过流模型生成
    
    # Q值评估
    q = self.network.select("critic")(observations, actions)
    if self.config["q_agg"] == "mean":
        q = q.mean(axis=0)
    else:
        q = q.min(axis=0)
    
    # 选择Q值最高的动作
    indices = jnp.argmax(q, axis=-1)
    selected_actions = actions[indices]
```

**欧拉方法积分：**

```python
def compute_flow_actions(self, observations, noises):
    actions = noises
    for i in range(self.config['flow_steps']):
        t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
        vels = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)
        actions = actions + vels / self.config['flow_steps']  # 欧拉步长
    return jnp.clip(actions, -1, 1)
```

## ACRLPD算法（Action-Chunked RLPD）

### 1. 算法原理

ACRLPD（Action-Chunked Reinforcement Learning with Prior Data）是基于SAC（Soft Actor-Critic）的算法，结合了行为克隆（BC）正则化和动作分块机制。相比ACFQL，ACRLPD提供了更简洁高效的实现方式。

**核心特点：**

- 基于成熟的SAC框架，训练稳定性更好
- 直接从TanhNormal分布采样动作序列
- 使用BC正则化利用先验数据
- 支持自适应温度参数调节探索-利用平衡

### 2. 动作采样机制

**简洁的采样策略：**

```python
def sample_actions(self, observations, rng=None):
    # 从TanhNormal分布直接采样动作序列
    dist = self.network.select('actor')(observations)
    actions = dist.sample(seed=rng)
    actions = jnp.clip(actions, -1, 1)
    return actions
```

**网络架构：**

```python
# Actor网络：输出动作序列的均值和标准差
actor_def = TanhNormal(
    base_cls=MLP(hidden_dims=config["actor_hidden_dims"]), 
    action_dim=full_action_dim  # horizon_length * action_dim
)

# 自适应温度网络
alpha_def = Temperature(init_temperature=config["init_temp"])
```

### 3. 损失函数设计

**完整的Actor损失：**

```python
def actor_loss(self, batch, grad_params, rng):
    if self.config["action_chunking"]:
        batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
    else:
        batch_actions = batch["actions"][..., 0, :]
    
    dist = self.network.select('actor')(batch['observations'], params=grad_params)
    actions = dist.sample(seed=rng)
    log_probs = dist.log_prob(actions)
    
    # 1. SAC Actor损失：最大化Q值，保持熵正则化
    qs = self.network.select('critic')(batch['observations'], actions)
    q = jnp.mean(qs, axis=0)
    actor_loss = (log_probs * self.network.select('alpha')() - q).mean()
    
    # 2. 熵损失：自适应温度参数学习
    alpha = self.network.select('alpha')(params=grad_params)
    entropy = -jax.lax.stop_gradient(log_probs).mean()
    alpha_loss = (alpha * (entropy - self.config['target_entropy'])).mean()
    
    # 3. BC正则化损失：利用先验数据约束策略
    bc_loss = -dist.log_prob(jnp.clip(batch_actions, -1 + 1e-5, 1 - 1e-5)).mean() * self.config["bc_alpha"]
    
    total_loss = actor_loss + alpha_loss + bc_loss
    return total_loss
```

**Critic损失（与ACFQL相同）：**

```python
def critic_loss(self, batch, grad_params, rng):
    if self.config["action_chunking"]:
        batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
    else:
        batch_actions = batch["actions"][..., 0, :]
    
    # 下一状态动作采样
    next_dist = self.network.select('actor')(batch['next_observations'][..., -1, :])
    next_actions = next_dist.sample(seed=rng)
    
    # 目标Q值计算
    next_qs = self.network.select('target_critic')(batch['next_observations'][..., -1, :], next_actions)
    if self.config['q_agg'] == 'min':
        next_q = next_qs.min(axis=0)
    else:
        next_q = next_qs.mean(axis=0)
    
    # TD目标：考虑动作分块的时间折扣
    target_q = batch['rewards'][..., -1] + \
        (self.config['discount'] ** self.config["horizon_length"]) * batch['masks'][..., -1] * next_q
    
    # 当前Q值
    q = self.network.select('critic')(batch['observations'], batch_actions, params=grad_params)
    
    # TD误差：应用有效性掩码处理轨迹终止
    critic_loss = (jnp.square(q - target_q) * batch['valid'][..., -1]).mean()
    
    return critic_loss
```

### 4. 网络架构对比

**ACRLPD网络组件：**

```python
# Critic：集成Q网络（通常10个）
critic_def = Ensemble(
    partial(StateActionValue, base_cls=MLP), 
    num=config["num_qs"]
)

# Actor：TanhNormal分布网络
actor_def = TanhNormal(
    base_cls=MLP(hidden_dims=config["actor_hidden_dims"]),
    action_dim=full_action_dim
)

# 温度参数：自适应调节
alpha_def = Temperature(initial_temperature=config["init_temp"])
```

### 5. 关键超参数

```python
config = {
    'agent_name': 'acrlpd',
    'lr': 3e-4,
    'batch_size': 256,
    'num_qs': 10,  # 更多Critic集成（vs ACFQL的2个）
    'target_entropy': None,  # 自动设为-0.5*action_dim
    'target_entropy_multiplier': 0.5,
    'bc_alpha': 0.0,  # BC正则化系数，online时设为0.01
    'horizon_length': 5,
    'action_chunking': True,
    'init_temp': 1.0,
}
```

## ACFQL vs ACRLPD：算法对比分析

### 1. 核心算法范式

| 算法       | 基础框架                   | 动作生成方式               | 主要优势                     |
| ---------- | -------------------------- | -------------------------- | ---------------------------- |
| **ACFQL**  | Flow Matching + Q-Learning | 流匹配积分 + Best-of-N采样 | 高质量动作序列，适合复杂任务 |
| **ACRLPD** | SAC + BC正则化             | TanhNormal直接采样         | 训练稳定，计算高效           |

### 2. 动作采样策略对比

**ACFQL - Best-of-N采样：**

```python
# 生成多个候选序列，选择Q值最高的
noises = jax.random.normal(rng, (batch_size, num_samples, action_dim))
actions = self.compute_flow_actions(observations, noises)  # 流匹配生成
q_values = self.network.select("critic")(observations, actions)
best_actions = actions[jnp.argmax(q_values, axis=-1)]
```

**ACRLPD - 直接采样：**

```python
# 从学习的分布直接采样
dist = self.network.select('actor')(observations)
actions = dist.sample(seed=rng)
```

### 3. 计算复杂度对比

| 指标         | ACFQL                    | ACRLPD             |
| ------------ | ------------------------ | ------------------ |
| **训练时间** | 较慢（流积分+Best-of-N） | 较快（直接采样）   |
| **推理时间** | 慢（需要多次前向传播）   | 快（单次前向传播） |
| **内存占用** | 高（存储多个候选）       | 低（单个动作序列） |
| **网络参数** | 多（Flow网络+编码器）    | 少（标准SAC网络）  |

### 4. 算法特性分析

**ACFQL特性：**

- 基于流匹配的连续动作生成
- Best-of-N采样策略提供动作多样性
- 较高的计算复杂度（多步积分+多候选采样）
- 较少的Critic集成（2个）

**ACRLPD特性：**

- 基于成熟SAC框架的直接采样
- TanhNormal分布建模动作空间
- 较低的计算复杂度（单次前向传播）
- 较多的Critic集成（10个）用于稳定训练

## 训练Pipeline设计

### 1. 两种实验设置

**Pipeline 1：离线预训练+在线微调（main.py）**

- **默认智能体：** ACFQL
- **训练流程：** 先从离线数据集预训练（1M步），再进行在线环境交互（1M步）
- **设计目的：** 利用离线数据快速获得合理策略，然后通过在线交互进一步优化

**Pipeline 2：纯在线训练（main_online.py）**

- **默认智能体：** ACRLPD
- **训练流程：** 直接从随机策略开始在线学习（1M步）
- **设计目的：** 测试算法的纯在线学习能力

### 2. 算法兼容性

**重要说明：ACFQL和ACRLPD都支持离线训练和在线训练**

- 两种算法都可以在任一pipeline中使用

- 选择不同的算法只需修改配置文件路径：

  ```bash
  # 使用ACRLPD进行离线+在线训练
  python main.py --agent agents/acrlpd.py
  
  # 使用ACFQL进行纯在线训练  
  python main_online.py --agent agents/acfql.py
  ```

### 3. BC正则化参数调节

通过调整BC系数可以控制对离线数据的依赖程度：

```python
# 强离线学习设置
config['bc_alpha'] = 0.1    # ACRLPD
config['alpha'] = 100.0     # ACFQL

# 弱离线学习设置
config['bc_alpha'] = 0.01   # ACRLPD  
config['alpha'] = 10.0      # ACFQL

# 纯在线学习设置
config['bc_alpha'] = 0.0    # ACRLPD
config['alpha'] = 1.0       # ACFQL（仍需非零值用于流匹配）
```

## 数据处理管道

### 1. 序列采样机制

```python
def sample_sequence(self, batch_size, sequence_length, discount):
    idxs = np.random.randint(self.size - sequence_length + 1, size=batch_size)
    
    # 初始化序列数据结构
    rewards = np.zeros((batch_size, sequence_length))
    masks = np.ones((batch_size, sequence_length))
    valid = np.ones((batch_size, sequence_length))
    
    # 填充每个时间步的数据
    for i in range(sequence_length):
        cur_idxs = idxs + i
        
        if i == 0:
            rewards[..., 0] = self['rewards'][cur_idxs]
            masks[..., 0] = self["masks"][cur_idxs]
        else:
            # 累积折扣奖励
            rewards[..., i] = rewards[..., i - 1] + self['rewards'][cur_idxs] * (discount ** i)
            # 累积掩码（遇到终止状态后全部为0）
            masks[..., i] = np.minimum(masks[..., i-1], self["masks"][cur_idxs])
            # 有效性掩码（终止后的动作无效）
            valid[..., i] = (1.0 - terminals[..., i - 1])
```

### 2. 动作有效性处理

为了处理轨迹终止后的无效动作，引入有效性掩码：

```python
# 在损失计算中应用有效性掩码
if self.config["action_chunking"]:
    bc_flow_loss = jnp.mean(
        jnp.reshape(
            (pred - vel) ** 2, 
            (batch_size, self.config["horizon_length"], self.config["action_dim"]) 
        ) * batch["valid"][..., None]  # 有效性掩码
    )
```

## 网络架构设计

### 1. Critic网络架构

**完全独立的Critic网络：**

```python
# 网络定义
critic_def = Value(
    hidden_dims=config['value_hidden_dims'],  # 通常为(512, 512, 512, 512)
    layer_norm=config['layer_norm'],  # Layer Normalization
    num_ensembles=config['num_qs'],  # 集成Q网络数量，通常为2
    encoder=encoders.get('critic'),  # 独立的视觉编码器
)

# 网络初始化参数
network_info = dict(
    critic=(critic_def, (ex_observations, full_actions)),  # 独立初始化
    target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),  # 目标网络
    actor_bc_flow=(actor_bc_flow_def, ...),  # 独立的Actor网络
)
```

**Critic内部架构：**

```python
class Value(nn.Module):
    def __call__(self, observations, actions=None):
        # 输入处理
        if self.encoder is not None:
            inputs = [self.encoder(observations)]  # 独立编码器处理状态
        else:
            inputs = [observations]  # 直接使用状态 (batch_size, obs_dim)
        
        if actions is not None:
            inputs.append(actions)  # 添加动作块 (batch_size, H * action_dim)
        
        # 状态-动作连接
        inputs = jnp.concatenate(inputs, axis=-1)  # (batch_size, obs_dim + H * action_dim)
        
        # MLP处理
        # 输入 → 隐藏层1(512) → 隐藏层2(512) → 隐藏层3(512) → 隐藏层4(512) → 输出(1)
        v = self.value_net(inputs).squeeze(-1)  # (batch_size,)
        
        return v  # 返回标量Q值
```

**集成机制（Ensemble）：**

```python
# 当 num_ensembles > 1 时
mlp_class = ensemblize(mlp_class, self.num_ensembles)
# 创建多个并行的MLP，每个有独立参数
# 输出形状: (num_ensembles, batch_size) → 用于Q值聚合
```

### 2. Actor向量场网络

```python
actor_bc_flow_def = ActorVectorField(
    hidden_dims=config['actor_hidden_dims'],
    action_dim=full_action_dim,  # horizon_length * action_dim
    layer_norm=config['actor_layer_norm'],
    encoder=encoders.get('actor_bc_flow'),
    use_fourier_features=config["use_fourier_features"],  # 时间嵌入
    fourier_feature_dim=config["fourier_feature_dim"],
)
```

### 3. 傅里叶特征时间嵌入

```python
class FourierFeatures(nn.Module):
    output_size: int = 64
    
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        half_dim = self.output_size // 2
        f = jnp.log(10000) / (half_dim - 1)
        f = jnp.exp(jnp.arange(half_dim) * -f)
        f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)
```

## 完整算法实现

### 1. 双模式训练循环

**模式一：ACFQL离线-在线训练（main.py）**

```python
# 离线预训练阶段
for i in range(FLAGS.offline_steps):  # 默认1M步
    # 采样序列批次
    batch = train_dataset.sample_sequence(
        config['batch_size'], 
        sequence_length=FLAGS.horizon_length, 
        discount=discount
    )
    
    # 更新ACFQL智能体（流匹配+BC损失）
    agent, offline_info = agent.update(batch)
    
    # 定期评估
    if i % FLAGS.eval_interval == 0:
        eval_info = evaluate(agent, eval_env, ...)

# 在线学习阶段：使用预训练模型进行环境交互  
for i in range(FLAGS.online_steps):
    # 动作队列管理：执行完整的动作序列
    if len(action_queue) == 0:
        # ACFQL使用Best-of-N采样生成高质量动作序列
        action = agent.sample_actions(observations=ob, rng=key)
        action_chunk = np.array(action).reshape(-1, action_dim)
        for action in action_chunk:
            action_queue.append(action)
    
    action = action_queue.pop(0)
    next_ob, reward, terminated, truncated, info = env.step(action)
    
    # 存储转移并更新智能体
    replay_buffer.add_transition(transition)
    if i >= FLAGS.start_training:
        batch = replay_buffer.sample_sequence(...)
        agent, update_info = agent.update(batch)
```

**模式二：ACRLPD纯在线训练（main_online.py）**

```python
# 纯在线学习：从随机策略开始
for i in range(FLAGS.online_steps):  # 默认1M步
    # 动作采样
    if len(action_queue) == 0:
        if i <= FLAGS.start_training:
            # 初期随机探索
            action = jax.random.uniform(key, shape=(action_dim,), minval=-1, maxval=1)
        else:
            # ACRLPD直接从TanhNormal分布采样
            action = agent.sample_actions(observations=ob, rng=key)
        
        action_chunk = np.array(action).reshape(-1, action_dim)
        for action in action_chunk:
            action_queue.append(action)
    
    action = action_queue.pop(0)
    next_ob, reward, terminated, truncated, info = env.step(action)
    
    # 在线学习更新
    replay_buffer.add_transition(transition)
    if i >= FLAGS.start_training:
        batch = replay_buffer.sample_sequence(...)
        agent, update_info = agent.update(batch)  # SAC+BC更新
```

### 2. 算法超参数对比

**ACFQL关键配置：**

```python
config = {
    'agent_name': 'acfql',
    'lr': 3e-4,
    'batch_size': 256,
    'actor_hidden_dims': (512, 512, 512, 512),
    'value_hidden_dims': (512, 512, 512, 512),
    'discount': 0.99,
    'tau': 0.005,  # 目标网络软更新系数
    'alpha': 100.0,  # BC系数，需要针对环境调优
    'num_qs': 2,  # Critic集成数量（较少）
    'flow_steps': 10,  # 流积分步数
    'horizon_length': 5,  # 动作分块长度
    'action_chunking': True,
    'actor_type': "best-of-n",  # 推荐使用Best-of-N
    'actor_num_samples': 32,  # Best-of-N的候选数量
    'use_fourier_features': False,  # 时间嵌入
}
```

**ACRLPD关键配置：**

```python
config = {
    'agent_name': 'acrlpd',
    'lr': 3e-4,
    'batch_size': 256,
    'actor_hidden_dims': (512, 512, 512, 512),
    'value_hidden_dims': (512, 512, 512, 512),
    'discount': 0.99,
    'tau': 0.005,
    'num_qs': 10,  # 更多的Critic集成（提高稳定性）
    'target_entropy': None,  # 自动设定为-0.5*action_dim
    'target_entropy_multiplier': 0.5,
    'bc_alpha': 0.01,  # BC正则化系数（online时使用）
    'horizon_length': 5,
    'action_chunking': True,
    'init_temp': 1.0,  # 初始温度参数
}
```

## 环境接口实现

### 1. 动作执行机制

```python
# 动作队列管理
action_queue = []
action_dim = example_batch["actions"].shape[-1]

for step in range(online_steps):
    # 当动作队列为空时，生成新的动作序列
    if len(action_queue) == 0:
        action = agent.sample_actions(observations=ob, rng=key)
        action_chunk = np.array(action).reshape(-1, action_dim)
        for action in action_chunk:
            action_queue.append(action)
    
    # 执行队列中的下一个动作
    action = action_queue.pop(0)
    next_ob, reward, terminated, truncated, info = env.step(action)
    
    # 处理轨迹终止
    if terminated or truncated:
        action_queue = []  # 清空动作队列
```

### 2. 奖励处理

```python
# 不同环境的奖励调整
if 'antmaze' in env_name:
    # D4RL antmaze环境奖励调整
    reward = reward - 1.0
elif is_robomimic_env(env_name):
    # Robomimic环境奖励调整
    reward = reward - 1.0

# 稀疏奖励处理
if sparse_reward:
    reward = (reward != 0.0) * -1.0
```

## 复现指南

### 1. 环境搭建

```bash
# 安装依赖
pip install -r requirements.txt

# 关键依赖版本
jax==0.6.0
flax==0.10.5
ml_collections==1.1.0
distrax==0.1.5
optax==0.2.4
wandb==0.19.9
```

### 2. 数据准备

**Robomimic数据集：**

```bash
# 下载到指定目录
mkdir -p ~/.robomimic/{lift,can,square}/mh/
# 下载 low_dim_v15.hdf5 文件
```

**OGBench数据集：**

```bash
wget -r -np -nH --cut-dirs=2 -A "*.npz" \
    https://rail.eecs.berkeley.edu/datasets/ogbench/cube-quadruple-play-100m-v0/
```

### 3. 运行实验

**ACFQL算法实验（离线-在线模式）：**

```bash
# 标准QC算法：Best-of-N采样
MUJOCO_GL=egl python main.py \
    --run_group=reproduce \
    --agent.actor_type=best-of-n \
    --agent.actor_num_samples=32 \
    --env_name=cube-triple-play-singletask-task2-v0 \
    --sparse=False \
    --horizon_length=5 \
    --offline_steps=1000000 \
    --online_steps=1000000

# QC-FQL变体：更高的BC系数
MUJOCO_GL=egl python main.py \
    --run_group=reproduce \
    --agent.alpha=100 \
    --env_name=cube-triple-play-singletask-task2-v0 \
    --sparse=False \
    --horizon_length=5

# 禁用动作分块的对比实验
MUJOCO_GL=egl python main.py \
    --run_group=reproduce \
    --agent.actor_type=best-of-n \
    --agent.actor_num_samples=4 \
    --env_name=cube-triple-play-singletask-task2-v0 \
    --sparse=False \
    --horizon_length=5 \
    --agent.action_chunking=False
```

**ACRLPD算法实验（纯在线模式）：**

```bash
# 标准RLPD：无动作分块
MUJOCO_GL=egl python main_online.py \
    --env_name=cube-triple-play-singletask-task2-v0 \
    --sparse=False \
    --horizon_length=1 \
    --online_steps=1000000

# RLPD-AC：启用动作分块
MUJOCO_GL=egl python main_online.py \
    --env_name=cube-triple-play-singletask-task2-v0 \
    --sparse=False \
    --horizon_length=5 \
    --online_steps=1000000

# QC-RLPD：动作分块+BC正则化
MUJOCO_GL=egl python main_online.py \
    --env_name=cube-triple-play-singletask-task2-v0 \
    --sparse=False \
    --horizon_length=5 \
    --agent.bc_alpha=0.01 \
    --online_steps=1000000
```

**稀疏奖励环境实验：**

```bash
# scene和puzzle-3x3域使用稀疏奖励
MUJOCO_GL=egl python main.py \
    --env_name=scene-v0 \
    --sparse=True \
    --horizon_length=5

MUJOCO_GL=egl python main_online.py \
    --env_name=puzzle-3x3-v0 \
    --sparse=True \
    --horizon_length=5
```

### 4. 调试技巧

1. **检查动作分块形状：**

   ```python
   print("Original actions shape:", batch["actions"].shape)
   print("Chunked actions shape:", batch_actions.shape)
   ```

2. **验证有效性掩码：**

   ```python
   print("Valid mask:", batch["valid"])
   print("Terminals:", batch["terminals"])
   ```

3. **监控训练指标：**

   - `critic/q_mean`：Q值平均值
   - `actor/bc_flow_loss`：BC流损失
   - `eval/success_rate`：成功率

### 5. 常见问题

1. **内存不足：** 调整`batch_size`和`dataset_replace_interval`
2. **训练不稳定：** 调整`alpha`（BC系数）和学习率
3. **性能不佳：** 检查`horizon_length`设置和环境奖励处理

## 总结

Q-Chunking算法通过动作分块机制扩展了传统RL的动作空间，本项目提供了两种高质量的实现方案：ACFQL和ACRLPD。

### 核心创新点

1. **时间扩展动作空间**：将单步动作预测扩展为多步动作序列预测
2. **动作分块执行**：通过`horizon_length`控制预测未来H步动作
3. **时间折扣修正**：将标准折扣因子γ修正为γ^H
4. **有效性掩码处理**：优雅处理轨迹终止后的无效动作

### 两种算法实现对比

| 特性             | ACFQL                        | ACRLPD                   |
| ---------------- | ---------------------------- | ------------------------ |
| **核心框架**     | Flow Matching + Q-Learning   | SAC + BC正则化           |
| **动作生成**     | 流匹配积分 + Best-of-N       | TanhNormal直接采样       |
| **主要优势**     | 高质量动作序列，适合复杂任务 | 训练稳定，计算高效       |
| **默认Pipeline** | main.py（离线+在线）         | main_online.py（纯在线） |
| **计算复杂度**   | 高（多步积分+多候选）        | 低（单次前向传播）       |
| **网络组件**     | Flow网络+时间嵌入            | 标准SAC网络              |
| **Critic集成**   | 2个（轻量级）                | 10个（高稳定性）         |


### 关键技术要点

1. **动作分块处理**：两种算法都使用相同的reshape机制

   ```python
   if config["action_chunking"]:
       batch_actions = jnp.reshape(batch["actions"], (batch_size, -1))
   ```

2. **时间折扣修正**：考虑动作分块长度的折扣计算

   ```python
   discount_factor = config['discount'] ** config["horizon_length"]
   ```

3. **有效性掩码**：处理轨迹终止后的无效动作

   ```python
   loss = loss * batch["valid"][..., None]  # 应用掩码
   ```

4. **动作队列执行**：确保完整动作序列的顺序执行

   ```python
   action_chunk = np.array(action).reshape(-1, action_dim)
   for action in action_chunk:
       action_queue.append(action)
   ```

### 实验复现要点

1. **环境配置**：确保MUJOCO_GL=egl用于无头渲染
2. **数据准备**：Robomimic和OGBench数据集的正确下载和放置
3. **超参数调优**：alpha（ACFQL）和bc_alpha（ACRLPD）需要针对具体环境调节
4. **监控指标**：关注critic/q_mean、actor损失和eval/success_rate

### 扩展方向

1. **更多采样策略**：可以为ACRLPD添加Best-of-N采样支持
2. **混合训练**：结合两种算法的优势进行ensemble学习
3. **自适应分块长度**：根据任务复杂度动态调整horizon_length
4. **层次化动作分块**：多时间尺度的动作预测

本文档详细分析了Q-Chunking的两种实现方案：ACFQL基于流匹配提供高质量动作生成，ACRLPD基于SAC提供稳定高效的训练。两种算法在动作分块机制上保持一致，在动作生成策略上形成互补，为Action Chunking这一重要概念提供了完整的技术实现方案。



 