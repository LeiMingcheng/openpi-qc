"""
ACRLPD π₀ Agent Implementation

核心Agent类，整合Actor(π₀)和Critic组件，支持完整的ACRLPD算法
保持所有RL功能的同时优化特征计算和参数管理
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import numpy as np

# OpenPI imports
import openpi.models.pi0 as pi0_model
import openpi.models.model as model_lib
from openpi.models.model import Observation
import openpi.training.checkpoints as checkpoints_lib

logger = logging.getLogger(__name__)


@dataclass
class ACRLPDConfig:
    """ACRLPD Agent配置"""
    # π₀模型配置
    pi0_checkpoint_path: str
    action_horizon: int = 20
    action_dim: int = 14
    
    # Critic配置
    num_critics: int = 2
    critic_hidden_dims: List[int] = None
    critic_learning_rate: float = 1e-3
    
    # 训练配置
    best_of_n: int = 4  # Best-of-N采样数量
    bc_weight: float = 0.1  # 行为克隆权重
    
    # 观察编码器配置
    image_keys: List[str] = None
    state_dim: int = 14
    
    # FSDP配置
    use_fsdp: bool = True
    fsdp_sharding_strategy: str = "full_shard"
    
    def __post_init__(self):
        if self.critic_hidden_dims is None:
            self.critic_hidden_dims = [256, 256]
        if self.image_keys is None:
            self.image_keys = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]


class SharedFeatures(NamedTuple):
    """共享特征结构 - 核心优化点"""
    image_features: Dict[str, jnp.ndarray]  # 图像特征 {cam_name: [B, feat_dim]}
    state_features: jnp.ndarray             # 状态特征 [B, feat_dim]  
    fused_features: jnp.ndarray             # 融合特征 [B, total_feat_dim]


class ACRLPDPi0Agent:
    """ACRLPD π₀ Agent核心类"""
    
    def __init__(self, config: ACRLPDConfig):
        """
        初始化Agent，加载π₀模型和创建Critic网络
        
        Args:
            config: Agent配置
        """
        self.config = config
        logger.info(f"=== 初始化ACRLPDPi0Agent ===")
        logger.info(f"π₀ checkpoint: {config.pi0_checkpoint_path}")
        logger.info(f"Action horizon: {config.action_horizon}, Action dim: {config.action_dim}")
        logger.info(f"Best-of-N: {config.best_of_n}, BC weight: {config.bc_weight}")
        
        # 1. 加载π₀模型
        self.pi0_model, self.pi0_checkpoint_info = self._load_pi0_model()
        
        # 2. 创建观察编码器 (从π₀模型中提取)
        self.observation_encoder = self._create_observation_encoder()
        
        # 3. 创建Critic网络ensemble  
        from agents_v2.critic_networks import CriticEnsemble
        self.critic_ensemble = CriticEnsemble(
            num_critics=config.num_critics,
            obs_feat_dim=self._get_obs_feature_dim(),
            action_dim=config.action_dim,
            hidden_dims=config.critic_hidden_dims
        )
        
        # 4. 缓存常用配置
        self._action_horizon = config.action_horizon
        self._action_dim = config.action_dim
        self._best_of_n = config.best_of_n
        
        logger.info(f"✅ ACRLPDPi0Agent初始化完成")
        logger.info(f"  π₀模型参数: ~3.2B")
        logger.info(f"  Critic ensemble: {config.num_critics} x ~20M参数")
        logger.info(f"  观察特征维度: {self._get_obs_feature_dim()}")
    
    def _load_pi0_model(self) -> Tuple[Any, Dict]:
        """加载π₀ diffusion模型"""
        logger.info(f"加载π₀模型: {self.config.pi0_checkpoint_path}")
        
        # 使用OpenPI标准方式加载π₀模型
        checkpoint_info = checkpoints_lib.load_checkpoint_info(self.config.pi0_checkpoint_path)
        pi0_model_instance = checkpoints_lib.load_model(
            self.config.pi0_checkpoint_path,
            model_class=pi0_model.Pi0Model
        )
        
        logger.info(f"✅ π₀模型加载完成，参数量: ~3.2B")
        return pi0_model_instance, checkpoint_info
    
    def _create_observation_encoder(self) -> Any:
        """创建观察编码器（从π₀模型提取）"""
        # π₀模型包含完整的观察编码器
        # 我们复用其编码器组件以确保一致性
        return self.pi0_model.observation_encoder
    
    def _get_obs_feature_dim(self) -> int:
        """获取观察特征维度"""
        # 基于π₀模型的编码器配置计算特征维度
        # 图像特征 + 状态特征的总维度
        image_feat_dim = 512 * len(self.config.image_keys)  # 每个相机512维特征
        state_feat_dim = 256  # 状态编码特征
        return image_feat_dim + state_feat_dim
    
    def compute_shared_features(self, batch: Dict[str, jnp.ndarray]) -> SharedFeatures:
        """
        计算共享特征 - 核心优化点
        
        一次计算，在Actor/Critic/BC loss间共享使用，避免重复计算
        
        Args:
            batch: 输入batch，包含image, state等
            
        Returns:
            SharedFeatures: 包含图像、状态和融合特征
        """
        # 1. 图像特征编码
        image_features = {}
        for cam_name in self.config.image_keys:
            if cam_name in batch['image']:
                # 使用π₀的图像编码器
                cam_image = batch['image'][cam_name]  # [B, H, W, C]
                cam_features = self.observation_encoder.encode_image(cam_image)  # [B, feat_dim]
                image_features[cam_name] = cam_features
        
        # 2. 状态特征编码
        state = batch['state']  # [B, state_dim]
        state_features = self.observation_encoder.encode_state(state)  # [B, feat_dim]
        
        # 3. 特征融合
        # 将所有图像特征和状态特征concatenate
        all_features = [state_features]
        for cam_name in sorted(image_features.keys()):  # 保证顺序一致
            all_features.append(image_features[cam_name])
        
        fused_features = jnp.concatenate(all_features, axis=-1)  # [B, total_feat_dim]
        
        return SharedFeatures(
            image_features=image_features,
            state_features=state_features, 
            fused_features=fused_features
        )
    
    def get_action(self, observation: Observation, rng: jax.random.PRNGKey) -> jnp.ndarray:
        """
        推理：生成动作
        
        Args:
            observation: 观察
            rng: 随机数生成器
            
        Returns:
            actions: [action_horizon, action_dim]
        """
        # 使用π₀模型的标准推理接口
        actions = self.pi0_model.sample_actions(
            observation=observation,
            rng=rng,
            num_samples=1
        )[0]  # 取第一个样本
        
        return actions
    
    def sample_action_candidates(self, 
                                observation: Observation, 
                                rng: jax.random.PRNGKey,
                                num_candidates: Optional[int] = None) -> jnp.ndarray:
        """
        Best-of-N采样：生成多个动作候选
        
        Args:
            observation: 观察
            rng: 随机数生成器 
            num_candidates: 候选数量，默认使用config中的best_of_n
            
        Returns:
            action_candidates: [num_candidates, action_horizon, action_dim]
        """
        if num_candidates is None:
            num_candidates = self._best_of_n
        
        # 使用π₀模型生成多个动作候选
        action_candidates = self.pi0_model.sample_actions(
            observation=observation,
            rng=rng,
            num_samples=num_candidates
        )
        
        return action_candidates
    
    def evaluate_action_candidates(self, 
                                 shared_features: SharedFeatures,
                                 action_candidates: jnp.ndarray) -> jnp.ndarray:
        """
        使用Critic评估动作候选的Q值
        
        Args:
            shared_features: 共享特征
            action_candidates: [num_candidates, action_horizon, action_dim]
            
        Returns:
            q_values: [num_candidates, ] - 每个候选的Q值
        """
        # 使用Critic ensemble的最小Q值（conservative）
        q_values = self.critic_ensemble.compute_min_q_values(
            obs_features=shared_features.fused_features,
            actions=action_candidates
        )
        
        return q_values
    
    def get_trainable_params(self) -> Dict[str, Any]:
        """
        获取可训练参数
        
        Returns:
            params: {pi0_params, critic_params}
        """
        return {
            'pi0_params': self.pi0_model.params,
            'critic_params': self.critic_ensemble.params
        }
    
    def set_params(self, pi0_params: Any, critic_params: Any):
        """
        设置模型参数（用于训练时的参数更新）
        
        Args:
            pi0_params: π₀模型参数
            critic_params: Critic网络参数
        """
        self.pi0_model = self.pi0_model.replace(params=pi0_params)
        self.critic_ensemble = self.critic_ensemble.replace(params=critic_params)
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置信息"""
        return {
            'action_horizon': self._action_horizon,
            'action_dim': self._action_dim,
            'obs_feature_dim': self._get_obs_feature_dim(),
            'num_critics': self.config.num_critics,
            'best_of_n': self._best_of_n,
            'pi0_checkpoint': self.config.pi0_checkpoint_path
        }


def create_acrlpd_pi0_agent_from_rl_config(rl_config: Any) -> ACRLPDPi0Agent:
    """
    从RLConfig创建ACRLPDPi0Agent
    
    Args:
        rl_config: AC Training的RLConfig
        
    Returns:
        ACRLPDPi0Agent实例
    """
    logger.info("从RLConfig创建ACRLPDPi0Agent...")
    
    # 从rl_config提取配置信息
    agent_config = ACRLPDConfig(
        # π₀模型配置
        pi0_checkpoint_path=rl_config.pi0_checkpoint_path,
        action_horizon=rl_config.model.action_horizon,
        action_dim=rl_config.qchunking.action_dim,
        
        # Critic配置
        num_critics=getattr(rl_config, 'num_critics', 2),
        critic_hidden_dims=getattr(rl_config, 'critic_hidden_dims', [256, 256]),
        critic_learning_rate=getattr(rl_config, 'critic_lr', 1e-3),
        
        # 训练配置
        best_of_n=getattr(rl_config, 'best_of_n', 4),
        bc_weight=getattr(rl_config, 'bc_weight', 0.1),
        
        # 观察配置
        image_keys=getattr(rl_config, 'image_keys', ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]),
        state_dim=getattr(rl_config, 'state_dim', 14),
        
        # FSDP配置
        use_fsdp=getattr(rl_config, 'use_fsdp', True),
        fsdp_sharding_strategy=getattr(rl_config, 'fsdp_strategy', "full_shard")
    )
    
    agent = ACRLPDPi0Agent(agent_config)
    
    logger.info(f"✅ 从RLConfig成功创建ACRLPDPi0Agent")
    return agent