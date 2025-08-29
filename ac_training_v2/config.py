"""
统一的ACRLPD训练配置系统

完全基于OpenPI TrainConfig架构的Q-chunking强化学习配置系统。
消除了所有命名冲突和架构兼容性问题。

主要特性：
- 继承OpenPI TrainConfig，完全兼容现有基础设施
- 集成Q-chunking RL专用参数（ACRLPD、Q-chunking）
- 简洁统一的配置接口
- 支持π₀模型与Q-chunking RL的完整集成
"""

import dataclasses
from typing import Any, Dict, Optional, Tuple, Union

import tyro
import jax
import jax.numpy as jnp

# OpenPI核心导入
import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.models.pi0_fast as pi0_fast
import openpi.training.config as openpi_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
# normalize导入已移除，使用OpenPI原生assets机制自动加载norm_stats
# aloha_policy不需要手动import，LeRobotAlohaDataConfig会自动处理


# ===============================================================================
# Q-chunking RL算法参数配置
# ===============================================================================

@dataclasses.dataclass(frozen=True)
class ACRLPDHyperparams:
    """ACRLPD算法超参数配置"""
    
    # Q-learning核心参数
    discount: float = 0.99
    target_update_rate: float = 0.005  # Target网络软更新率
    
    # Critic网络参数
    critic_lr: float = 3e-4
    critic_hidden_dims: Tuple[int, ...] = (256, 256, 256)
    num_critics: int = 2  # Critic ensemble大小
    
    # Actor (π₀) 参数
    actor_lr: float = 1e-4
    actor_update_freq: int = 2  # Actor更新频率
    
    # Best-of-N采样参数
    num_action_samples: int = 32  # 候选动作数量
    action_sampling_temperature: float = 1.0
    
    # Behavior Cloning正则化
    bc_loss_weight: float = 0.1
    
    # Q-chunking参数
    chunk_length: int = 5  # 动作序列长度
    bootstrap_length: int = 5  # Bootstrap目标长度
    
    # 训练动态参数
    batch_size: int = 256
    utd_ratio: int = 1  # Update-to-data比率
    
    # ACRLPDPi0Config映射的额外参数
    q_aggregation: str = "min"  # Q-value aggregation method
    target_update_tau: float = 0.005  # Target network soft update rate
    
    # EMA参数
    use_ema: bool = True
    pi0_ema_decay: float = 0.999
    critic_ema_decay: float = 0.99
    use_ema_for_inference: bool = True
    
    # 采样参数
    diffusion_steps: int = 10
    use_best_of_n: bool = True
    
    # 温度控制参数
    use_adaptive_temperature: bool = True
    target_entropy_multiplier: float = 0.5
    
    # 训练阶段参数 (注意：现在使用RLTrainConfig.num_train_steps作为总步数)
    eval_frequency: int = 10000
    save_frequency: int = 50000
    
    # 🔧 梯度积累配置（完整功能，用户要求不简化）
    gradient_accumulation_steps: int = 4  # 梯度积累步数，有效batch_size = batch_size × gradient_accumulation_steps
    max_grad_norm: float = 1.0           # 梯度裁剪阈值
    enable_gradient_accumulation: bool = True  # 启用梯度积累
    
    # 🔧 数据采样比例控制（新增：正负样本episode比例控制）
    positive_episode_ratio: float = 0.4  # 内存池中正样本episode的比例（默认60%）
    enable_reward_balanced_episodes: bool = True  # 启用episode级别的正负样本比例控制
    
    # 🔧 批量采样比例控制（新增）
    positive_batch_ratio: float = 0.5  # batch中正样本的比例（默认50%）
    enable_batch_ratio_control: bool = True  # 启用batch级别比例控制
    
    # 🔧 Epoch和学习率调节配置
    enable_epoch_based_lr_schedule: bool = False  # 使用OpenPI标准学习率控制，不需要复杂双层调节
    lr_decay_strategy: str = "cosine"             # 学习率衰减策略: "cosine", "step", "exp"
    lr_decay_factor: float = 0.95                 # 学习率衰减因子（epoch间）
    lr_min_factor: float = 0.1                    # 最小学习率因子（相对初始学习率）
    warmup_epochs: int = 2                        # 学习率预热epoch数
    total_epochs: int = 100                       # 总epoch数（用于cosine衰减计算）
    
    # Epoch内学习率调节（配合基于step的调节）
    intra_epoch_lr_decay: bool = True             # 启用epoch内学习率衰减
    intra_epoch_strategy: str = "cosine"          # Epoch内衰减策略: "cosine", "linear", "exp"
    steps_per_epoch: int = 10000                  # 每个epoch的预期步数
    intra_epoch_min_factor: float = 0.9           # Epoch内最小学习率因子
    lr_absolute_min: float = 1e-7                 # 学习率绝对下限


@dataclasses.dataclass(frozen=True)
class QChunkingConfig:
    """Q-chunking特定配置"""
    
    # 动作序列参数
    horizon_length: int = 5
    action_dim: int = 14  # 机器人动作维度
    
    # 序列生成参数
    discount: float = 0.99
    reward_scale: float = 1.0
    use_sparse_rewards: bool = True
    terminal_reward: float = 1.0
    
    # Bootstrap和掩码参数
    bootstrap_type: str = "standard"  # "standard", "n_step", "lambda"
    mask_invalid_actions: bool = True
    
    # Episode边界处理
    pad_episodes: bool = True
    episode_timeout_penalty: float = 0.0


# ===============================================================================
# 统一训练配置系统
# ===============================================================================

@dataclasses.dataclass(frozen=True)
class RLTrainConfig(openpi_config.TrainConfig):
    """
    统一的强化学习训练配置
    
    继承OpenPI TrainConfig，添加Q-chunking RL专用参数。
    完全兼容OpenPI基础设施，消除所有配置冲突。
    """
    
    # === Q-chunking RL专用参数 ===
    acrlpd: ACRLPDHyperparams = dataclasses.field(default_factory=ACRLPDHyperparams)
    qchunking: QChunkingConfig = dataclasses.field(default_factory=QChunkingConfig)
    
    # === 数据加载参数 ===
    episodes_per_memory_pool: int = 32  # 🚀 优化：从2增加到8，减少频繁重载开销
    
    # === 异步加载参数 ===
    async_loading_enabled: bool = True          # 启用异步加载
    async_trigger_ratio: float = 0.6            # 在epoch多少进度时开始预加载（75%）
    
    # === 性能分析参数 ===
    enable_perf_analysis: bool = False  # 是否启用详细性能分析
    
    # === RL专用优化器配置 ===
    # 覆盖基类的单一optimizer，支持Actor-Critic双优化器
    critic_optimizer: _optimizer.OptimizerConfig = dataclasses.field(
        default_factory=lambda: _optimizer.AdamW(weight_decay=1e-5)
    )
    actor_optimizer: _optimizer.OptimizerConfig = dataclasses.field(
        default_factory=lambda: _optimizer.AdamW(weight_decay=1e-6)
    )
    
    # === RL专用学习率调度（与Epoch-based兼容） ===
    critic_lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(
        default_factory=lambda: _optimizer.CosineDecaySchedule(
            warmup_steps=20000, peak_lr=3e-4, decay_steps=200_000  # 匹配epoch设置：2 epochs warmup, 200k total steps
        )
    )
    actor_lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(
        default_factory=lambda: _optimizer.CosineDecaySchedule(
            warmup_steps=20000, peak_lr=1e-4, decay_steps=200_000   # 匹配epoch设置：2 epochs warmup, 200k total steps
        )
    )
    
    # === π₀模型专用权重加载器 ===
    pi0_weight_loader: weight_loaders.WeightLoader = dataclasses.field(
        default_factory=lambda: weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi0_base/params"
        )
    )
    
    def get_effective_actor_lr_schedule(self) -> _optimizer.LRScheduleConfig:
        """获取有效的Actor学习率调度器，基于acrlpd中的静态学习率"""
        return _optimizer.CosineDecaySchedule(
            warmup_steps=2000,
            peak_lr=self.acrlpd.actor_lr,  # 使用acrlpd中的学习率作为peak_lr
            decay_steps=200_000
        )

    def get_effective_critic_lr_schedule(self) -> _optimizer.LRScheduleConfig:
        """获取有效的Critic学习率调度器，基于acrlpd中的静态学习率"""
        return _optimizer.CosineDecaySchedule(
            warmup_steps=2000,
            peak_lr=self.acrlpd.critic_lr,  # 使用acrlpd中的学习率作为peak_lr
            decay_steps=200_000
        )

    def validate_rl_config(self):
        """Q-chunking RL专用配置验证"""
        # 调用基类验证
        super().__post_init__()
        
        # 验证动作维度配置合理性
        # 注意：qchunking.action_dim是真实机器人动作维度
        # model.action_dim是模型内部处理维度，两者可以不同
        # OpenPI的transforms会自动处理维度转换
        if hasattr(self.model, 'action_dim'):
            # 确保维度都是正数
            assert self.qchunking.action_dim > 0, f"QChunking action_dim must be positive: {self.qchunking.action_dim}"
            assert self.model.action_dim > 0, f"Model action_dim must be positive: {self.model.action_dim}"
        
        # 验证Q-chunking参数
        assert self.qchunking.horizon_length > 0, "Horizon length must be positive"
        assert 0 < self.qchunking.discount <= 1.0, "Discount must be in (0, 1]"
        
        # 验证ACRLPD参数
        assert self.batch_size > 0, "Batch size must be positive"  # 使用主配置的batch_size
        assert self.acrlpd.num_action_samples > 0, "Action samples must be positive"
        
        # FSDP兼容性检查：batch_size必须能被设备数量整除
        if hasattr(self, 'fsdp_devices') and self.fsdp_devices > 1:
            if self.batch_size % self.fsdp_devices != 0:
                suggested_batch_size = ((self.batch_size // self.fsdp_devices) + 1) * self.fsdp_devices
                raise ValueError(
                    f"FSDP要求batch_size ({self.batch_size}) 能被设备数量 ({self.fsdp_devices}) 整除。"
                    f"建议使用 batch_size={suggested_batch_size}"
                )


# ===============================================================================
# 预定义配置
# ===============================================================================

# ALOHA折叠任务配置
RL_ALOHA_FOLD = RLTrainConfig(
    name="rl_aloha_fold",
    project_name="acrlpd_pi0",
    
    # π₀模型配置 - 使用OpenPI默认32维以保持预训练权重兼容
    model=pi0.Pi0Config(
        action_horizon=50  # AlohaInputs会自动处理14→32维转换
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi0_base/params"
    ),
    
    # 数据配置
    data=openpi_config.LeRobotAlohaDataConfig(
        repo_id="aloha_fold",
        default_prompt="fold the clothes on the table",
        adapt_to_pi=False,
        assets=openpi_config.AssetsConfig(asset_id="aloha_fold")
    ),
    
    # 基础训练参数
    batch_size=256,
    num_train_steps=50_000,
    log_interval=100,
    save_interval=5000,
    
    # Q-chunking RL参数
    acrlpd=ACRLPDHyperparams(
        chunk_length=5,
        num_action_samples=32,
        bc_loss_weight=0.1,
        batch_size=256
    ),
    qchunking=QChunkingConfig(
        horizon_length=5,
        action_dim=14
    )
)

# Libero仿真配置
RL_LIBERO = RLTrainConfig(
    name="rl_libero",
    project_name="acrlpd_pi0_libero",
    
    # π₀模型配置
    model=pi0.Pi0Config(
        action_dim=7, 
        action_horizon=5
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi0_base/params"
    ),
    
    # 数据配置
    data=openpi_config.LeRobotLiberoDataConfig(
        repo_id="physical-intelligence/libero",
        base_config=openpi_config.DataConfig(prompt_from_task=True)
    ),
    
    # 仿真训练参数
    batch_size=256,
    num_train_steps=100_000,
    log_interval=100,
    save_interval=5000,
    
    # Q-chunking RL参数（针对仿真优化）
    acrlpd=ACRLPDHyperparams(
        chunk_length=5,
        num_action_samples=64,  # 仿真中使用更多采样
        bc_loss_weight=0.1,
        critic_lr=1e-3,  # 更激进的学习率
        batch_size=256
    ),
    qchunking=QChunkingConfig(
        horizon_length=5,
        action_dim=7
    )
)

# Fold Box配置 - 第一阶段优化：梯度累积 + 激进FSDP + 恢复训练质量参数
RL_FOLD_BOX = RLTrainConfig(
    name="rl_fold_box",
    project_name="acrlpd_pi0_fold_box",
    
    # π₀模型配置 - 使用OpenPI默认32维以保持预训练权重兼容
    model=pi0.Pi0Config(
        action_horizon=20,  # AlohaInputs会自动处理14→32维padding
        dtype="bfloat16"    # 重要：使用bfloat16减少内存使用
    ),
    # 修复权重加载路径 - 使用正确的完整检查点
    weight_loader=weight_loaders.CheckpointWeightLoader(
        #"/dev/shm/lmc/openpi/checkpoints/pi0_base/openpi-assets/checkpoints/pi0_base/pi0_base/params"
        "/era-ai/lm/weight/pi0/pi0_dual_box_full/yzy_fold_box/90000/params/"
    ),
    
    # 数据配置 - 使用OpenPI标准ALOHA配置
    data=openpi_config.LeRobotAlohaDataConfig(
        repo_id="fold_box_unified",  # 使用新转换的fold_box数据集
        #repo_id="aloha_test_dataset",
        default_prompt="fold the box",  # 与用户配置一致
        adapt_to_pi=False,  # 与用户配置一致：使用标准ALOHA数据空间
        assets=openpi_config.AssetsConfig(
            assets_dir="/era-ai/lm/weight/pi0/pi0_dual_box_full/yzy_fold_box/90000/assets/yzysmile",
            asset_id="aloha_fold_box"
        ),
        
        # 自定义repack transforms用于相机映射
        # 关键修复：将我们的相机名称映射到AlohaInputs期望的标准名称
        repack_transforms=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.cam_high",
                            "cam_left_wrist": "observation.images.cam_left_wrist", 
                            "cam_right_wrist": "observation.images.cam_right_wrist",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "reward": "reward",  # 添加reward字段映射，确保RL数据完整性
                    }
                )
            ]
        ),
        # 不传递data_transforms - 让LeRobotAlohaDataConfig自动创建
        # 它会自动创建：AlohaInputs(action_dim=14, adapt_to_pi=False) + AlohaOutputs
        base_config=openpi_config.DataConfig(
            prompt_from_task=False  # 与用户配置一致：不根据LeRobot数据集名称给语言指令
            # norm_stats由AssetsConfig的assets_dir自动加载，无需手动配置
        )
    ),
    
    # 训练参数 - 第一阶段梯度累积版本
    batch_size=32,   # 物理batch_size保持小以节省瞬时内存
    num_train_steps=20000,  # 与用户配置一致
    log_interval=2,
    save_interval=1000,
    
    # 多GPU配置 - 最大化FSDP分片减少内存
    fsdp_devices=8,  # 最大分片：(1,8) = 1个数据并行组 × 8设备FSDP分片
    
    # Q-chunking RL参数（第一阶段：恢复关键训练质量参数）
    acrlpd=ACRLPDHyperparams(
        # 基本RL参数
        discount=0.99,
        target_update_rate=0.005,
        q_aggregation="min",
        target_update_tau=0.005,
        
        # Critic网络参数（第一阶段恢复）
        critic_lr=3e-5,  # 保守的学习率，适合真实机器人数据
        critic_hidden_dims=(512, 512, 512),  # 第一阶段目标：部分恢复到(192,192)
        num_critics=6,  # 第一阶段目标：恢复ensemble到2
        
        # Actor (π₀) 参数
        actor_lr=1e-5,
        actor_update_freq=4,
        
        # Best-of-N采样参数（第一阶段部分恢复）
        num_action_samples=4,  # 第一阶段目标：从1恢复到8
        action_sampling_temperature=1.0,
        
        # Behavior Cloning正则化
        bc_loss_weight=200,  # 适中的BC权重，适合manipulation任务
        
        # Q-chunking参数（与action_horizon匹配）
        chunk_length=20,  # 与action_horizon匹配：恢复到20
        bootstrap_length=20,
        
        # 训练动态参数（梯度累积优化）
        batch_size=32,   # 与主配置RLTrainConfig.batch_size保持一致
        utd_ratio=1,
        
        # EMA参数
        use_ema=True,
        pi0_ema_decay=0.999,
        critic_ema_decay=0.99,
        use_ema_for_inference=True,
        
        # 采样参数
        diffusion_steps=10,
        use_best_of_n=True,
        
        # 温度控制参数（第一阶段仍禁用以节省内存）
        use_adaptive_temperature=False,  # 第一阶段仍禁用熵估计
        target_entropy_multiplier=0.5,
        
        # 训练阶段参数 (使用num_train_steps=200000作为总步数)
        eval_frequency=10000,
        save_frequency=1000,
        
        # 🔧 梯度积累配置（第一阶段优化：4步积累，有效batch_size=8×4=32）
        gradient_accumulation_steps=4,  # 梯度积累步数，提升训练效率
        max_grad_norm=0.1,             # 梯度裁剪，防止梯度爆炸
        enable_gradient_accumulation=False,  # 启用梯度积累
        
        # 🔧 Epoch和学习率调节配置（应用修改）
        enable_epoch_based_lr_schedule=False,  # 使用OpenPI标准学习率控制
        lr_decay_strategy="cosine",            # 学习率衰减策略: cosine
        lr_decay_factor=0.95,                  # 学习率衰减因子（epoch间）
        lr_min_factor=0.1,                     # 最小学习率因子（相对初始学习率）
        warmup_epochs=2,                       # 学习率预热epoch数
        total_epochs=100,                       # 总epoch数
        
        # Epoch内学习率调节（应用cosine调节）
        intra_epoch_lr_decay=True,             # 启用epoch内学习率衰减
        intra_epoch_strategy="cosine",         # Epoch内衰减策略: cosine
        steps_per_epoch=200,                 # 每个epoch的步数
        intra_epoch_min_factor=0.9,            # Epoch内最小学习率因子
        lr_absolute_min=3e-8                   # 学习率绝对下限
    ),
    qchunking=QChunkingConfig(
        horizon_length=20,  # 与action_horizon匹配：恢复到20
        action_dim=14  # ALOHA双臂机器人
    ),
    
    # === 异步加载参数 ===
    async_loading_enabled=True,          # 启用异步加载
    async_trigger_ratio=0.7             # 在epoch多少进度时开始预加载（75%）
)

# DROID数据集配置
RL_DROID = RLTrainConfig(
    name="rl_droid",
    project_name="acrlpd_pi0_droid",
    
    # π₀-FAST模型配置（更适合大规模数据）
    model=pi0_fast.Pi0FASTConfig(
        action_dim=8, 
        action_horizon=10,
        max_token_len=180
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi0_fast_base/params"
    ),
    
    # DROID数据配置
    data=openpi_config.SimpleDataConfig(
        assets=openpi_config.AssetsConfig(asset_id="droid"),
        data_transforms=lambda model: _transforms.Group(
            inputs=[],  # 将由DROID-specific transforms填充
            outputs=[]
        ),
        base_config=openpi_config.DataConfig(prompt_from_task=True)
    ),
    
    # 针对fold box任务的训练参数
    batch_size=32,  # 小批次以适应内存和快速测试
    num_train_steps=200_000,
    log_interval=200,
    save_interval=10_000,
    
    # Q-chunking RL参数（针对ALOHA fold box任务优化）
    acrlpd=ACRLPDHyperparams(
        chunk_length=20,  # 与action_horizon保持一致
        num_action_samples=16,  # 减少采样数量以提高速度
        bc_loss_weight=0.01,  # 更低的BC权重
        batch_size=32  # 与外部batch_size保持一致
    ),
    qchunking=QChunkingConfig(
        horizon_length=10,  # 与model.action_horizon保持一致：20→10
        action_dim=14       # ALOHA双臂机器人的真实动作维度
    )
)


# ===============================================================================
# 配置注册和管理
# ===============================================================================

_CONFIGS = {
    "rl_aloha_fold": RL_ALOHA_FOLD,
    "rl_fold_box": RL_FOLD_BOX,
    "rl_libero": RL_LIBERO,
    "rl_droid": RL_DROID,
}


def get_config(config_name: str) -> RLTrainConfig:
    """获取统一的RL训练配置"""
    if config_name not in _CONFIGS:
        available = ", ".join(_CONFIGS.keys())
        raise ValueError(f"Config '{config_name}' not found. Available: {available}")
    
    config = _CONFIGS[config_name]
    config.validate_rl_config()
    return config


def list_configs() -> Dict[str, str]:
    """列出所有可用的配置"""
    return {name: "RLTrainConfig (统一配置)" for name in _CONFIGS.keys()}


def cli() -> RLTrainConfig:
    """命令行配置选择接口"""
    return tyro.extras.overridable_config_cli(
        {k: (k, v) for k, v in _CONFIGS.items()}
    )


# ===============================================================================
# 测试和验证
# ===============================================================================

if __name__ == "__main__":
    print("🔧 测试统一ACRLPD配置系统...")
    
    print(f"\n📋 可用配置: {len(_CONFIGS)} 个")
    config_info = list_configs()
    for name, description in config_info.items():
        print(f"  {name}: {description}")
    
    print(f"\n✅ 配置验证:")
    for config_name in _CONFIGS.keys():
        try:
            config = get_config(config_name)
            print(f"  {config_name}: ✓")
            print(f"    模型: {type(config.model).__name__}")
            print(f"    动作维度: {config.model.action_dim}")
            print(f"    Q-chunking序列长度: {config.qchunking.horizon_length}")
            print(f"    批次大小: {config.batch_size}")
            print(f"    OpenPI兼容性: ✓ (继承TrainConfig)")
        except Exception as e:
            print(f"  {config_name}: ❌ {e}")
    
    print(f"\n🎯 配置系统特性:")
    print(f"  - 统一架构: 基于OpenPI TrainConfig")  
    print(f"  - 零冲突: 无命名冲突和架构问题")
    print(f"  - 完全兼容: 可直接用于OpenPI基础设施")
    print(f"  - Q-chunking支持: 完整的RL参数配置")
    
    print(f"\n✅ 统一配置系统测试完成!")