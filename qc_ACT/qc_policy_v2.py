"""
Q-Chunking Policy实现V2，修复高优先级问题：
1. 移除ACT特征提取的no_grad，允许backbone参数更新
2. 优化特征维度检测
3. 改进网络架构设计
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import torchvision.transforms as transforms
import sys
import os

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policy import ACTPolicy


class QCriticEncoder(nn.Module):
    """Q-Chunking Critic的独立视觉编码器，可从ACT模型继承"""
    
    def __init__(self, 
                 camera_names: list,
                 use_depth_image: bool = False,
                 use_robot_base: bool = False,
                 act_model=None,  # 用于继承ACT编码器
                 freeze_encoder: bool = False,
                 rank: int = 0):
        super().__init__()
        
        self.camera_names = camera_names
        self.use_depth_image = use_depth_image
        self.use_robot_base = use_robot_base
        self.freeze_encoder = freeze_encoder
        self.rank = rank
        
        # 机器人状态维度
        self.qpos_dim = 16 if use_robot_base else 14
        
        if act_model is not None:
            # 从ACT模型继承编码器
            self._inherit_from_act(act_model)
        else:
            # 创建新的编码器
            self._build_new_encoder()
        
        # 冻结编码器参数（如果需要）
        if freeze_encoder:
            self._freeze_parameters()
    
    def _inherit_from_act(self, act_model):
        """从ACT模型继承视觉编码器"""
        
        try:
            # 验证ACT模型结构 - ACTPolicy的backbones在model属性中
            if hasattr(act_model, 'model') and hasattr(act_model.model, 'backbones'):
                act_backbones = act_model.model.backbones
            elif hasattr(act_model, 'backbones'):
                act_backbones = act_model.backbones
            else:
                raise AttributeError("ACT模型缺少backbones属性")
            
            if len(act_backbones) != len(self.camera_names):
                raise ValueError(f"ACT模型backbone数量({len(act_backbones)})与相机数量({len(self.camera_names)})不匹配")
            
            # 复制backbone
            self.backbones = nn.ModuleList()
            for i, backbone in enumerate(act_backbones):
                try:
                    # 使用深度复制而非重新构造，避免构造参数问题
                    import copy
                    backbone_copy = copy.deepcopy(backbone)
                    self.backbones.append(backbone_copy)
                    if self.rank == 0:
                        print(f"成功复制backbone[{i}]: {type(backbone).__name__}")
                except Exception as e:
                    print(f"警告: 复制backbone[{i}]失败: {e}")
                    # 回退到状态字典复制
                    backbone_copy = type(backbone)(
                        getattr(backbone, 'backbone', None), 
                        getattr(backbone, 'train_backbone', True), 
                        getattr(backbone, 'num_channels', 512), 
                        getattr(backbone, 'return_interm_layers', False)
                    )
                    backbone_copy.load_state_dict(backbone.state_dict())
                    self.backbones.append(backbone_copy)
            
            # 复制深度backbone（如果有）
            if hasattr(act_model, 'depth_backbones') and act_model.depth_backbones is not None:
                self.depth_backbones = nn.ModuleList()
                for i, depth_backbone in enumerate(act_model.depth_backbones):
                    try:
                        import copy
                        depth_backbone_copy = copy.deepcopy(depth_backbone)
                        self.depth_backbones.append(depth_backbone_copy)
                        if self.rank == 0:
                            print(f"成功复制depth_backbone[{i}]: {type(depth_backbone).__name__}")
                    except Exception as e:
                        print(f"警告: 复制depth_backbone[{i}]失败: {e}")
                        # 回退实现
                        depth_backbone_copy = type(depth_backbone)()
                        depth_backbone_copy.load_state_dict(depth_backbone.state_dict())
                        self.depth_backbones.append(depth_backbone_copy)
            else:
                self.depth_backbones = None
                if self.rank == 0:
                    print("ACT模型无深度backbone")
            
            # 复制input_proj
            act_input_proj = None
            if hasattr(act_model, 'model') and hasattr(act_model.model, 'input_proj'):
                act_input_proj = act_model.model.input_proj
            elif hasattr(act_model, 'input_proj'):
                act_input_proj = act_model.input_proj
                
            if act_input_proj is not None:
                try:
                    import copy
                    self.input_proj = copy.deepcopy(act_input_proj)
                    if self.rank == 0:
                        print(f"成功复制input_proj: {act_input_proj}")
                except Exception as e:
                    print(f"警告: 复制input_proj失败: {e}")
                    # 回退实现
                    input_proj_copy = nn.Conv2d(
                        act_input_proj.in_channels,
                        act_input_proj.out_channels,
                        act_input_proj.kernel_size,
                        act_input_proj.stride,
                        act_input_proj.padding
                    )
                    input_proj_copy.load_state_dict(act_input_proj.state_dict())
                    self.input_proj = input_proj_copy
            else:
                raise AttributeError("ACT模型缺少input_proj属性")
            
            # 复制robot state投影
            act_robot_proj = None
            if hasattr(act_model, 'model') and hasattr(act_model.model, 'input_proj_robot_state'):
                act_robot_proj = act_model.model.input_proj_robot_state
            elif hasattr(act_model, 'input_proj_robot_state'):
                act_robot_proj = act_model.input_proj_robot_state
                
            if act_robot_proj is not None:
                try:
                    import copy
                    self.input_proj_robot_state = copy.deepcopy(act_robot_proj)
                    if self.rank == 0:
                        print(f"成功复制input_proj_robot_state: {act_robot_proj}")
                except Exception as e:
                    print(f"警告: 复制input_proj_robot_state失败: {e}")
                    # 回退实现
                    robot_state_proj_copy = nn.Linear(
                        act_robot_proj.in_features,
                        act_robot_proj.out_features
                    )
                    robot_state_proj_copy.load_state_dict(act_robot_proj.state_dict())
                    self.input_proj_robot_state = robot_state_proj_copy
            else:
                raise AttributeError("ACT模型缺少input_proj_robot_state属性")
            
            if self.rank == 0:
                print(f"成功从ACT模型继承视觉编码器:")
                print(f"  Backbones: {len(self.backbones)}")
                print(f"  Depth backbones: {len(self.depth_backbones) if self.depth_backbones else 0}")
                print(f"  Input proj: 存在")
                print(f"  Robot state proj: 存在")
            
        except Exception as e:
            if self.rank == 0:
                print(f"错误: 从ACT模型继承编码器失败: {e}")
                print("回退到创建新编码器")
            self._build_new_encoder()
    
    def _build_new_encoder(self):
        """构建新的视觉编码器（暂时简化实现）"""
        # 简化实现：使用预设的维度
        # 实际应用中需要根据具体需求构建
        hidden_dim = 768
        
        # 计算实际的图像展平维度
        # 假设图像尺寸为480x640，3通道
        image_flat_dim = len(self.camera_names) * 3 * 480 * 640  # 实际展平维度
        
        # 创建一个多层感知机来处理巨大的图像维度
        self.visual_proj = nn.Sequential(
            nn.Linear(image_flat_dim, 2048),  # 第一层大幅降维
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, hidden_dim)       # 第二层到目标维度
        )
        self.input_proj_robot_state = nn.Linear(self.qpos_dim, hidden_dim)
        
        if self.rank == 0:
            print("创建新的视觉编码器（简化版本）")
            print(f"  输入图像维度: {image_flat_dim}")
            print(f"  输出特征维度: {hidden_dim}")
    
    def _freeze_parameters(self):
        """冻结编码器参数"""
        for param in self.parameters():
            param.requires_grad = False
        if self.rank == 0:
            print("视觉编码器参数已冻结")
    
    def forward(self, images: torch.Tensor, depth_images: torch.Tensor, 
                robot_state: torch.Tensor) -> torch.Tensor:
        """
        编码器前向传播
        Args:
            images: [batch_size, num_cameras, C, H, W]
            depth_images: [batch_size, num_cameras, H, W] 或 None
            robot_state: [batch_size, robot_state_dim]
        Returns:
            encoded_features: [batch_size, feature_dim]
        """
        
        batch_size = images.shape[0]
        # 图像归一化
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
        images_norm = normalize(images)
        
        # 深度图像处理
        depth_norm = None
        if self.use_depth_image and depth_images is not None and depth_images.numel() > 1:
            depth_normalize = transforms.Normalize(mean=[0.5], std=[0.5])
            depth_norm = depth_normalize(depth_images)
        
        if hasattr(self, 'backbones'):
            # 使用继承的ACT编码器
            all_cam_features = []
            for cam_id, cam_name in enumerate(self.camera_names):
                # 正确的backbone调用方式
                features, src_pos = self.backbones[cam_id](images_norm[:, cam_id])
                features = features[0]  # 取最后一层特征
                
                # 处理深度图像（如果有）
                if self.depth_backbones is not None and depth_norm is not None:
                    features_depth = self.depth_backbones[cam_id](depth_norm[:, cam_id].unsqueeze(dim=1))
                    combined_features = torch.cat([features, features_depth], axis=1)
                    projected_features = self.input_proj(combined_features)
                else:
                    projected_features = self.input_proj(features)
                
                # 将空间特征展平为固定维度向量（类似DETR_VAE的做法）
                # projected_features shape: [batch, hidden_dim, H, W]
                # 使用全局平均池化将空间维度压缩为固定长度向量
                pooled_features = torch.mean(projected_features, dim=[2, 3])  # [batch, hidden_dim]
                all_cam_features.append(pooled_features)
            
            # 直接拼接所有相机的固定维度特征向量
            visual_features_flat = torch.cat(all_cam_features, dim=1)  # [batch, hidden_dim * num_cameras]
            
            # 处理机器人状态
            robot_state_features = self.input_proj_robot_state(robot_state)
        else:
            # 使用简化的新编码器
            # 这里假设images已经是展平的特征（占位符实现）
            visual_features_flat = images.reshape(batch_size, -1)
            visual_features_flat = self.visual_proj(visual_features_flat)
            robot_state_features = self.input_proj_robot_state(robot_state)
        
        # 拼接所有特征
        encoded_features = torch.cat([visual_features_flat, robot_state_features], dim=-1)
        
        return encoded_features


class QCriticNetwork(nn.Module):
    """Q-Chunking的独立Critic网络，符合算法文档规范"""
    
    def __init__(self, 
                 camera_names: list,
                 action_chunk_dim: int,  # 动作块维度：chunk_size * action_dim
                 hidden_dims: list = [512, 512, 512, 512],
                 num_ensembles: int = 6,  # 增加到6个，参考ACRLPD的在线训练最佳实践
                 use_depth_image: bool = False,
                 use_robot_base: bool = False,
                 act_model=None,  # 用于继承ACT编码器
                 freeze_encoder: bool = False,
                 use_layer_norm: bool = True,
                 dropout_rate: float = 0.0,  # 新增dropout支持
                 activation: str = "relu",  # 新增激活函数选择
                 rank: int = 0):
        super().__init__()
        
        self.num_ensembles = num_ensembles
        self.action_chunk_dim = action_chunk_dim
        self.use_layer_norm = use_layer_norm
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.rank = rank
        
        # 激活函数映射
        activation_map = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
            "tanh": nn.Tanh()
        }
        self.activation_fn = activation_map.get(activation.lower(), nn.ReLU())
        
        # 独立的视觉编码器
        self.encoder = QCriticEncoder(
            camera_names=camera_names,
            use_depth_image=use_depth_image,
            use_robot_base=use_robot_base,
            act_model=act_model,
            freeze_encoder=freeze_encoder,
            rank=rank
        )
        
        # 估算编码器输出维度
        self.encoded_dim = self._estimate_encoded_dim()
        
        # 构建Q网络集成
        self.q_networks = nn.ModuleList()
        for _ in range(num_ensembles):
            layers = []
            input_dim = self.encoded_dim + action_chunk_dim
            
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(input_dim, hidden_dim))
                
                # LayerNorm位置参考ACRLPD：在Dense后、Dropout前、Activation前
                if use_layer_norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                
                # Dropout（如果启用）
                if dropout_rate > 0.0:
                    layers.append(nn.Dropout(dropout_rate))
                
                # 可配置的激活函数
                layers.append(self.activation_fn)
                input_dim = hidden_dim
            
            # 输出层：输出标量Q值
            layers.append(nn.Linear(input_dim, 1))
            self.q_networks.append(nn.Sequential(*layers))
        
        if self.rank == 0:
            print(f"QCriticNetwork初始化完成 (优化用于在线训练):")
            print(f"  编码器输出维度: {self.encoded_dim}")
            print(f"  动作块维度: {action_chunk_dim}")
            print(f"  隐藏层维度: {hidden_dims}")
            print(f"  集成网络数量: {num_ensembles} (ACRLPD风格)")
            print(f"  LayerNorm: {use_layer_norm}")
            print(f"  Dropout率: {dropout_rate}")
            print(f"  激活函数: {activation}")
    
    def _estimate_encoded_dim(self) -> int:
        """估算编码器输出维度"""
        try:
            # 创建dummy输入进行测试
            device = next(self.encoder.parameters()).device
            batch_size = 1
            
            dummy_images = torch.randn(batch_size, len(self.encoder.camera_names), 3, 480, 640, device=device)
            dummy_depth = torch.zeros(batch_size, 1, device=device) if not self.encoder.use_depth_image else torch.randn(batch_size, len(self.encoder.camera_names), 480, 640, device=device)
            dummy_robot_state = torch.randn(batch_size, self.encoder.qpos_dim, device=device)
            
            with torch.no_grad():
                encoded = self.encoder(dummy_images, dummy_depth, dummy_robot_state)
                encoded_dim = encoded.shape[-1]
            
            if self.rank == 0:
                print(f"自动检测编码器输出维度: {encoded_dim}")
            return encoded_dim
            
        except Exception as e:
            if self.rank == 0:
                print(f"无法检测编码器维度，使用默认值: {e}")
            # 使用默认估算
            if hasattr(self.encoder, 'backbones'):
                # ACT编码器估算：hidden_dim * num_cameras + robot_state_dim
                hidden_dim = 768  # 从预训练配置得到
                num_cameras = len(self.encoder.camera_names)
                return hidden_dim * num_cameras + self.encoder.qpos_dim
            else:
                # 简化编码器估算
                return 768 * len(self.encoder.camera_names) + self.encoder.qpos_dim
    
    def forward(self, observations_or_features, actions: torch.Tensor) -> torch.Tensor:
        """
        Critic网络前向传播 - 支持两种输入格式
        Args:
            observations_or_features: 
                1. Dict[str, torch.Tensor] - 包含图像和机器人状态的观察字典
                   - 'images': [batch_size, num_cameras, C, H, W]
                   - 'depth_images': [batch_size, num_cameras, H, W] (可选)
                   - 'robot_state': [batch_size, robot_state_dim]
                2. torch.Tensor - 预提取的特征 [batch_size, feature_dim]
            actions: [batch_size, action_chunk_dim] 展平的动作块
        Returns:
            q_values: [num_ensembles, batch_size] Q值
        """
        
        # 确保actions是展平的
        if len(actions.shape) == 3:  # [batch_size, chunk_size, action_dim]
            actions = actions.reshape(actions.shape[0], -1)
        
        # V2.8修复：支持两种输入格式 - observations字典或预提取的features
        if isinstance(observations_or_features, dict):
            # 输入是observations字典，需要提取特征
            images = observations_or_features['images']
            depth_images = observations_or_features.get('depth_images', torch.zeros(1, device=images.device))
            robot_state = observations_or_features['robot_state']
            
            # 编码观察
            encoded_obs = self.encoder(images, depth_images, robot_state)
        else:
            # 输入是预提取的features tensor
            encoded_obs = observations_or_features
        
        # 拼接观察和动作
        inputs = torch.cat([encoded_obs, actions], dim=-1)
        
        # 通过集成Q网络
        q_values = []
        for q_network in self.q_networks:
            q_value = q_network(inputs).squeeze(-1)  # 输出标量
            q_values.append(q_value)
        
        return torch.stack(q_values, dim=0)  # [num_ensembles, batch_size]


# 保留原有的QCriticHead作为向后兼容（标记为废弃）
class QCriticHead(nn.Module):
    """Q-Chunking的Critic头，附加在ACT编码器后 [DEPRECATED: 使用QCriticNetwork]"""
    
    def __init__(self, 
                 feature_dim: int,  # ACT编码器输出的特征维度
                 action_chunk_dim: int,  # 动作块维度
                 hidden_dims: list = [512, 512, 512],
                 num_ensembles: int = 2):
        super().__init__()
        
        print("警告: QCriticHead已废弃，建议使用QCriticNetwork")
        
        self.num_ensembles = num_ensembles
        self.q_heads = nn.ModuleList()
        
        # 创建集成Q头
        for _ in range(num_ensembles):
            layers = []
            input_dim = feature_dim + action_chunk_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                ])
                input_dim = hidden_dim
            
            layers.append(nn.Linear(input_dim, 1))
            self.q_heads.append(nn.Sequential(*layers))
    
    def forward(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            features: [batch_size, feature_dim] ACT编码器特征
            actions: [batch_size, action_chunk_dim] 动作块
        Returns:
            q_values: [num_ensembles, batch_size] Q值
        """
        # 确保actions是展平的
        if len(actions.shape) == 3:  # [batch_size, chunk_size, action_dim]
            actions = actions.reshape(actions.shape[0], -1)
        
        x = torch.cat([features, actions], dim=-1)
        
        q_values = []
        for q_head in self.q_heads:
            q_values.append(q_head(x).squeeze(-1))
        
        return torch.stack(q_values, dim=0)


class BCFlowActor(nn.Module):
    """BC Flow Actor网络，用于训练时的多步flow matching（Teacher模型）"""
    
    def __init__(self, 
                 feature_dim: int,  # ACT特征维度
                 action_dim: int,
                 hidden_dims: list = [512, 512, 512, 512],
                 use_fourier_features: bool = True,
                 fourier_feature_dim: int = 64):
        super().__init__()
        
        self.action_dim = action_dim
        self.use_fourier_features = use_fourier_features
        
        # 时间嵌入
        if use_fourier_features:
            self.fourier_features = FourierFeatures(fourier_feature_dim)
            time_dim = fourier_feature_dim
        else:
            time_dim = 1
        
        # 构建网络
        layers = []
        input_dim = feature_dim + action_dim + time_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        预测速度场
        Args:
            features: [batch_size, feature_dim] ACT特征
            x_t: [batch_size, action_dim] 插值轨迹点
            t: [batch_size, 1] 时间参数
        Returns:
            velocity: [batch_size, action_dim] 预测的速度场
        """
        # 时间嵌入
        if self.use_fourier_features:
            t_embed = self.fourier_features(t)
        else:
            t_embed = t
        
        # 拼接输入
        x = torch.cat([features, x_t, t_embed], dim=-1)
        
        return self.network(x)


class OnestepActor(nn.Module):
    """Onestep Actor网络，用于快速推理（Student模型）"""
    
    def __init__(self, 
                 feature_dim: int,  # ACT特征维度
                 action_dim: int,
                 hidden_dims: list = [512, 512, 512, 512]):
        super().__init__()
        
        self.action_dim = action_dim
        
        # 构建网络（无时间嵌入）
        layers = []
        input_dim = feature_dim + action_dim  # features + noise
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        直接预测动作（无时间维度）
        Args:
            features: [batch_size, feature_dim] ACT特征
            noise: [batch_size, action_dim] 输入噪声
        Returns:
            actions: [batch_size, action_dim] 预测的动作
        """
        # 拼接输入
        x = torch.cat([features, noise], dim=-1)
        return self.network(x)


class FourierFeatures(nn.Module):
    """傅里叶特征时间嵌入"""
    
    def __init__(self, output_size: int = 64):
        super().__init__()
        self.output_size = output_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.output_size // 2
        f = torch.log(torch.tensor(10000.0, device=x.device)) / (half_dim - 1)
        f = torch.exp(torch.arange(half_dim, device=x.device) * -f)
        f = x * f
        return torch.cat([torch.cos(f), torch.sin(f)], dim=-1)


class QChunkingPolicyV2(nn.Module):
    """
    Q-Chunking策略V2，修复高优先级问题
    """
    
    def __init__(self, 
                 act_policy: ACTPolicy,
                 chunk_size: int,
                 action_dim: int,
                 camera_names: list,
                 use_depth_image: bool = False,
                 use_robot_base: bool = False,
                 # QC-FQL参数
                 lr: float = 3e-4,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 100.0,  # BC系数
                 num_qs: int = 2,
                 flow_steps: int = 10,
                 actor_type: str = "best-of-n",
                 actor_num_samples: int = 32,
                 q_agg: str = "min",
                 # 新增：算法架构模式选择
                 actor_architecture: str = "flow_dual",  # 'flow_dual' 或 'act_direct'
                 update_act_directly: bool = False,       # 是否直接优化ACT参数
                 # 新增：ACT backbone更新控制
                 update_act_backbone: bool = True,
                 act_lr_scale: float = 0.1,  # ACT学习率缩放因子
                 # 新增：Critic编码器相关参数
                 use_independent_critic: bool = True,  # 是否使用独立的Critic网络
                 inherit_act_encoder: bool = True,    # 是否从ACT继承编码器
                 freeze_critic_encoder: bool = False, # 是否冻结Critic编码器
                 critic_hidden_dims: list = None,     # Critic隐藏层维度
                 use_layer_norm: bool = True,         # 是否使用LayerNorm
                 # 新增：Critic网络高级配置（参考ACRLPD）
                 critic_use_layer_norm: bool = True,  # Critic专用LayerNorm设置
                 critic_dropout_rate: float = 0.0,    # Critic Dropout率
                 critic_activation: str = "relu",     # Critic激活函数
                 # 新增：ACT模式专用权重参数
                 act_bc_weight: float = 100.0,        # ACT模式下BC损失权重
                 act_q_weight: float = 1.0,           # ACT模式下Q损失权重
                 # 新增：学习率调度器配置
                 lr_scheduler_config: dict = None,    # 学习率调度器配置
                 total_epochs: int = 20000,           # 总训练轮数
                 steps_per_epoch: int = 5000,         # 每个epoch的步数
                 rank: int = 0                        # 进程rank，用于控制打印
                 ):
        
        super().__init__()
        
        # 保存进程rank
        self.rank = rank
        
        # 保存ACT策略用于action生成
        self.act_policy = act_policy
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.camera_names = camera_names
        self.use_depth_image = use_depth_image
        self.use_robot_base = use_robot_base
        
        # 算法架构模式
        self.actor_architecture = actor_architecture
        self.update_act_directly = update_act_directly
        
        # QC-FQL参数
        self.discount = discount
        self.tau = tau
        self.alpha = alpha
        self.num_qs = num_qs
        self.flow_steps = flow_steps
        self.actor_type = actor_type
        self.actor_num_samples = actor_num_samples
        self.q_agg = q_agg
        
        # ACT更新控制
        self.update_act_backbone = update_act_backbone
        self.act_lr_scale = act_lr_scale
        
        # Critic网络参数
        self.use_independent_critic = use_independent_critic
        self.inherit_act_encoder = inherit_act_encoder
        self.freeze_critic_encoder = freeze_critic_encoder
        self.critic_hidden_dims = critic_hidden_dims or [512, 512, 512, 512]
        self.use_layer_norm = use_layer_norm
        # 新增：Critic网络高级配置（参考ACRLPD）
        self.critic_use_layer_norm = critic_use_layer_norm
        self.critic_dropout_rate = critic_dropout_rate
        self.critic_activation = critic_activation
        
        # ACT模式专用权重参数
        self.act_bc_weight = act_bc_weight
        self.act_q_weight = act_q_weight
        
        # 学习率调度器配置
        self.lr_scheduler_config = lr_scheduler_config
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        
        # 计算维度
        qpos_dim = 14  # 从ACT配置或数据中获取
        if use_robot_base:
            qpos_dim += 2
        self.qpos_dim = qpos_dim
        self.action_chunk_dim = chunk_size * action_dim
        
        # 获取ACT特征维度
        self.act_feature_dim = self._determine_act_feature_dim()
        
        # 构建网络
        self._build_networks()
        
        # 设置优化器
        self._setup_optimizers(lr)
        
        # 创建目标网络
        self._create_target_networks()
        
        # 创建学习率调度器
        self._setup_lr_scheduler()
        
        if self.rank == 0:
            print(f"QChunkingPolicyV2初始化完成:")
            print(f"  ACT特征维度: {self.act_feature_dim}")
            print(f"  动作块维度: {self.action_chunk_dim}")
            print(f"  更新ACT backbone: {self.update_act_backbone}")
            if self.actor_architecture == 'act_direct':
                print(f"  ACT模式权重配置:")
                print(f"    BC损失权重: {self.act_bc_weight}")
                print(f"    Q损失权重: {self.act_q_weight}")
    
    def _determine_act_feature_dim(self) -> int:
        """确定ACT特征维度"""
        try:
            # 通过运行一次前向传播来确定维度
            device = next(self.act_policy.parameters()).device
            
            # 创建dummy输入
            dummy_images = torch.randn(1, len(self.camera_names), 3, 480, 640, device=device)
            dummy_depth = torch.zeros(1, 1, device=device) if not self.use_depth_image else torch.randn(1, len(self.camera_names), 480, 640, device=device)
            dummy_qpos = torch.randn(1, self.qpos_dim, device=device)
            
            # 提取特征
            features = self._extract_act_features_unsafe(dummy_images, dummy_depth, dummy_qpos)
            feature_dim = features.shape[-1]
            
            if self.rank == 0:
                print(f"自动检测ACT特征维度: {feature_dim}")
            return feature_dim
            
        except Exception as e:
            if self.rank == 0:
                print(f"无法自动检测特征维度，使用默认值: {e}")
            # 根据ACT配置估算：hidden_dim * num_cameras + qpos维度
            backbone_dim = getattr(self.act_policy.model, 'hidden_dim', 768)
            estimated_dim = backbone_dim * len(self.camera_names) + self.qpos_dim
            if self.rank == 0:
                print(f"估算ACT特征维度: {estimated_dim} (hidden_dim={backbone_dim} * cameras={len(self.camera_names)} + qpos={self.qpos_dim})")
            return estimated_dim
    
    def _build_networks(self):
        """构建Critic网络和Flow Actor"""
        
        if self.use_independent_critic:
            # 使用新的独立Critic网络（推荐）
            act_model = self.act_policy.model if self.inherit_act_encoder else None
            
            self.critic = QCriticNetwork(
                camera_names=self.camera_names,
                action_chunk_dim=self.action_chunk_dim,
                hidden_dims=self.critic_hidden_dims,
                num_ensembles=self.num_qs,
                use_depth_image=self.use_depth_image,
                use_robot_base=self.use_robot_base,
                act_model=act_model,
                freeze_encoder=self.freeze_critic_encoder,
                use_layer_norm=self.critic_use_layer_norm,  # 使用Critic专用LayerNorm设置
                dropout_rate=self.critic_dropout_rate,      # ACRLPD风格dropout
                activation=self.critic_activation,          # 可配置激活函数
                rank=self.rank
            )
            
            if self.rank == 0:
                print(f"使用独立QCriticNetwork，ACT编码器继承: {self.inherit_act_encoder}")
        else:
            # 使用传统的QCriticHead（向后兼容）
            self.critic = QCriticHead(
                feature_dim=self.act_feature_dim,
                action_chunk_dim=self.action_chunk_dim,
                hidden_dims=self.critic_hidden_dims[:3],  # 适配旧接口
                num_ensembles=self.num_qs
            )
            
            if self.rank == 0:
                print("使用传统QCriticHead（已废弃）")
        
        # 根据架构模式选择Actor网络
        if self.actor_architecture == 'flow_dual':
            # Flow模式：双Actor架构（BC Flow Actor + Onestep Actor）
            self.bc_flow_actor = BCFlowActor(
                feature_dim=self.act_feature_dim,
                action_dim=self.action_chunk_dim
            )
            
            self.onestep_actor = OnestepActor(
                feature_dim=self.act_feature_dim,
                action_dim=self.action_chunk_dim
            )
            
            if self.rank == 0:
                print("创建Flow双Actor架构:")
                print(f"  BC Flow Actor: {self.bc_flow_actor.__class__.__name__}")
                print(f"  Onestep Actor: {self.onestep_actor.__class__.__name__}")
                
        elif self.actor_architecture == 'act_direct':
            # ACT模式：直接使用预训练ACT模型作为Actor
            # 不需要创建额外的Actor网络，直接使用self.act_policy
            self.bc_flow_actor = None
            self.onestep_actor = None
            
            if self.rank == 0:
                print("使用ACT直接模式:")
                print(f"  Actor: 预训练ACT模型 ({type(self.act_policy).__name__})")
                print(f"  直接优化ACT: {self.update_act_directly}")
        else:
            raise ValueError(f"不支持的actor_architecture: {self.actor_architecture}")
    
    def _setup_optimizers(self, lr: float):
        """设置优化器"""
        
        # QC网络优化器
        if self.use_independent_critic:
            # 独立Critic网络：可能需要分别优化编码器和Q网络
            critic_params = []
            critic_encoder_params = []
            
            # 分离Critic编码器和Q网络参数
            if not self.freeze_critic_encoder:
                critic_encoder_params.extend(self.critic.encoder.parameters())
            
            # Q网络参数
            critic_params.extend(self.critic.q_networks.parameters())
            
            # 创建优化器
            if critic_encoder_params:
                encoder_param_count = sum(p.numel() for p in critic_encoder_params)
                self.critic_encoder_optimizer = torch.optim.Adam(critic_encoder_params, lr=lr * 0.1)  # 编码器使用较小学习率
                if self.rank == 0:
                    print(f"Critic编码器优化器: {encoder_param_count:,}个参数, lr={lr * 0.1}")
            else:
                self.critic_encoder_optimizer = None
                if self.rank == 0:
                    print("Critic编码器参数已冻结")
            
            critic_param_count = sum(p.numel() for p in critic_params)
            self.critic_optimizer = torch.optim.Adam(critic_params, lr=lr)
            if self.rank == 0:
                print(f"Critic网络优化器: {critic_param_count:,}个参数, lr={lr}")
        else:
            # 传统方式
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
            self.critic_encoder_optimizer = None
        
        # 根据架构模式设置Actor优化器
        if self.actor_architecture == 'flow_dual':
            # Flow模式：双Actor优化器
            bc_flow_params = list(self.bc_flow_actor.parameters())
            onestep_params = list(self.onestep_actor.parameters())
            
            # 合并所有Actor参数到单个优化器（也可以分开优化）
            all_actor_params = bc_flow_params + onestep_params
            self.actor_optimizer = torch.optim.Adam(all_actor_params, lr=lr)
            
            if self.rank == 0:
                bc_flow_count = sum(p.numel() for p in bc_flow_params)
                onestep_count = sum(p.numel() for p in onestep_params)
                print(f"BC Flow Actor优化器: {bc_flow_count:,}个参数")
                print(f"Onestep Actor优化器: {onestep_count:,}个参数")
                print(f"总Actor参数: {bc_flow_count + onestep_count:,}个参数, lr={lr}")
                
        elif self.actor_architecture == 'act_direct':
            # ACT模式：不需要额外的Actor优化器，ACT参数由专门的优化器管理
            self.actor_optimizer = None
            
            if self.rank == 0:
                print("ACT直接模式: 使用ACT专用优化器，无额外Actor优化器")
        else:
            raise ValueError(f"不支持的actor_architecture: {self.actor_architecture}")
        
        # ACT优化器设置（根据模式和配置）
        if self.update_act_backbone or (self.actor_architecture == 'act_direct' and self.update_act_directly):
            act_lr = lr * self.act_lr_scale
            act_params = []
            
            # 正确访问ACT模型的参数
            act_model = self.act_policy.model
            
            if self.actor_architecture == 'act_direct' and self.update_act_directly:
                # ACT直接模式：优化整个ACT模型参数（除了已被Critic继承的部分）
                if self.rank == 0:
                    print("配置ACT直接模式优化器（整个ACT模型）...")
                
                # 收集所有ACT模型参数
                for name, param in act_model.named_parameters():
                    act_params.append(param)
                
                act_param_count = sum(p.numel() for p in act_params)
                if self.rank == 0:
                    print(f"ACT直接模式参数总数: {act_param_count:,}")
                    
            elif self.update_act_backbone:
                # 传统Flow模式：只优化backbone部分
                if self.rank == 0:
                    print("配置ACT backbone优化器（仅视觉编码器）...")
                
                # DETRVAE模型有backbones属性（ModuleList）
                if hasattr(act_model, 'backbones') and act_model.backbones is not None:
                    for backbone in act_model.backbones:
                        act_params.extend(backbone.parameters())
                    
                    backbone_param_count = sum(p.numel() for p in act_params)
                    if self.rank == 0:
                        print(f"检测到{len(act_model.backbones)}个backbone，参数数量: {backbone_param_count:,}")
                    
                    # 同时包含depth_backbones（如果有）
                    if hasattr(act_model, 'depth_backbones') and act_model.depth_backbones is not None:
                        depth_params = []
                        for depth_backbone in act_model.depth_backbones:
                            depth_params.extend(depth_backbone.parameters())
                        act_params.extend(depth_params)
                        depth_param_count = sum(p.numel() for p in depth_params)
                        if self.rank == 0:
                            print(f"额外包含{len(act_model.depth_backbones)}个depth backbone，参数数量: {depth_param_count:,}")
                    
                    # 包含input_proj层参数
                    if hasattr(act_model, 'input_proj'):
                        input_proj_params = list(act_model.input_proj.parameters())
                        act_params.extend(input_proj_params)
                        input_proj_param_count = sum(p.numel() for p in input_proj_params)
                        if self.rank == 0:
                            print(f"包含input_proj层参数数量: {input_proj_param_count:,}")
                            
                    # 包含robot state投影层
                    if hasattr(act_model, 'input_proj_robot_state'):
                        robot_proj_params = list(act_model.input_proj_robot_state.parameters())
                        act_params.extend(robot_proj_params)
                        robot_proj_param_count = sum(p.numel() for p in robot_proj_params)
                        if self.rank == 0:
                            print(f"包含robot state投影层参数数量: {robot_proj_param_count:,}")
                else:
                    if self.rank == 0:
                        print("警告: 无法找到backbones属性，跳过ACT backbone优化器设置")
                    self.update_act_backbone = False
                    self.act_optimizer = None
                    return
            
            if not act_params:
                if self.rank == 0:
                    print("警告: ACT参数列表为空，跳过ACT优化器设置")
                self.act_optimizer = None
                return
                
            total_param_count = sum(p.numel() for p in act_params)
            self.act_optimizer = torch.optim.Adam(act_params, lr=act_lr)
            
            if self.rank == 0:
                mode_desc = "ACT直接模式" if (self.actor_architecture == 'act_direct' and self.update_act_directly) else "ACT backbone模式"
                print(f"{mode_desc}优化器设置完成:")
                print(f"  学习率: {act_lr:.2e} (缩放因子: {self.act_lr_scale})")
                print(f"  参数数量: {total_param_count:,}")
        else:
            self.act_optimizer = None
            if self.rank == 0:
                print("ACT参数将被冻结")
    
    def _create_target_networks(self):
        """创建目标网络"""
        
        if self.use_independent_critic:
            # 创建独立的目标Critic网络
            act_model = self.act_policy.model if self.inherit_act_encoder else None
            
            self.target_critic = QCriticNetwork(
                camera_names=self.camera_names,
                action_chunk_dim=self.action_chunk_dim,
                hidden_dims=self.critic_hidden_dims,
                num_ensembles=self.num_qs,
                use_depth_image=self.use_depth_image,
                use_robot_base=self.use_robot_base,
                act_model=act_model,
                freeze_encoder=self.freeze_critic_encoder,
                use_layer_norm=self.critic_use_layer_norm,  # 使用Critic专用LayerNorm设置
                dropout_rate=self.critic_dropout_rate,      # ACRLPD风格dropout
                activation=self.critic_activation,          # 可配置激活函数
                rank=self.rank
            )
        else:
            # 传统目标网络
            self.target_critic = QCriticHead(
                feature_dim=self.act_feature_dim,
                action_chunk_dim=self.action_chunk_dim,
                hidden_dims=self.critic_hidden_dims[:3],
                num_ensembles=self.num_qs
            )
        
        # 初始化目标网络
        self._update_target_networks(tau=1.0)
        if self.rank == 0:
            print(f"目标网络创建完成，类型: {'QCriticNetwork' if self.use_independent_critic else 'QCriticHead'}")
    
    def compute_flow_actions(self, features: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        通过BC Flow Actor使用多步Euler积分生成动作（Teacher模型）
        Args:
            features: [batch_size, feature_dim] ACT特征
            noise: [batch_size, action_dim] 初始噪声
        Returns:
            actions: [batch_size, action_dim] 生成的动作
        """
        actions = noise
        dt = 1.0 / self.flow_steps
        
        # Euler积分方法
        for i in range(self.flow_steps):
            t = torch.full((features.shape[0], 1), i * dt, device=features.device)
            velocity = self.bc_flow_actor(features, actions, t)
            actions = actions + velocity * dt
            
        # 裁剪动作到合理范围
        actions = torch.clamp(actions, -1, 1)
        return actions
    
    def _setup_lr_scheduler(self):
        """设置二层学习率调度器"""
        if self.lr_scheduler_config is None or not self.lr_scheduler_config.get('enable', False):
            self.lr_scheduler = None
            if self.rank == 0:
                print("学习率调度器已禁用")
            return
        
        # 导入调度器模块
        try:
            from lr_scheduler import create_two_level_scheduler
        except ImportError as e:
            if self.rank == 0:
                print(f"警告: 无法导入学习率调度器模块: {e}")
            self.lr_scheduler = None
            return
        
        # 收集所有优化器
        optimizers_dict = {}
        if self.critic_optimizer is not None:
            optimizers_dict['critic'] = self.critic_optimizer
        if self.critic_encoder_optimizer is not None:
            optimizers_dict['critic_encoder'] = self.critic_encoder_optimizer
        if self.actor_optimizer is not None:
            optimizers_dict['actor'] = self.actor_optimizer
        if self.act_optimizer is not None:
            optimizers_dict['act'] = self.act_optimizer
        
        # 创建调度器
        self.lr_scheduler = create_two_level_scheduler(
            optimizers_dict=optimizers_dict,
            config=self.lr_scheduler_config,
            total_epochs=self.total_epochs,
            steps_per_epoch=self.steps_per_epoch,
            rank=self.rank
        )
        
        if self.rank == 0 and self.lr_scheduler is not None:
            print(f"二层学习率调度器创建成功")
    
    def step_epoch(self, epoch: int):
        """epoch开始时调用，更新全局学习率"""
        if self.lr_scheduler is not None:
            self.lr_scheduler.step_epoch(epoch)
    
    def step_batch(self, step_in_epoch: int):
        """每个batch后调用，更新局部学习率"""
        if self.lr_scheduler is not None:
            self.lr_scheduler.step_batch(step_in_epoch)
    
    def get_current_lrs(self) -> dict:
        """获取当前学习率，用于日志记录"""
        if self.lr_scheduler is not None:
            lrs = self.lr_scheduler.get_current_lrs()
            # 如果调度器返回空或全零，直接从优化器获取
            if not lrs or all(lr == 0 for lr in lrs.values()):
                return self._get_lrs_from_optimizers()
            return lrs
        return self._get_lrs_from_optimizers()
    
    def get_lr_info(self) -> str:
        """获取学习率信息字符串，用于打印"""
        if self.lr_scheduler is not None:
            lr_info = self.lr_scheduler.get_lr_info()
            # 如果调度器返回空或全零信息，直接从优化器获取
            if not lr_info or "0.00e+00" in lr_info:
                return self._get_lr_info_from_optimizers()
            return lr_info
        return self._get_lr_info_from_optimizers()
    
    def _get_lrs_from_optimizers(self) -> dict:
        """直接从优化器获取学习率"""
        lrs = {}
        if self.critic_optimizer is not None:
            lrs['critic'] = self.critic_optimizer.param_groups[0]['lr']
        if self.critic_encoder_optimizer is not None:
            lrs['critic_encoder'] = self.critic_encoder_optimizer.param_groups[0]['lr']
        if self.actor_optimizer is not None:
            lrs['actor'] = self.actor_optimizer.param_groups[0]['lr']
        if self.act_optimizer is not None:
            lrs['act'] = self.act_optimizer.param_groups[0]['lr']
        return lrs
    
    def _get_lr_info_from_optimizers(self) -> str:
        """直接从优化器获取学习率信息字符串"""
        lrs = self._get_lrs_from_optimizers()
        lr_strs = []
        for name, lr in lrs.items():
            lr_strs.append(f"{name}_lr={lr:.2e}")
        return ", ".join(lr_strs)
    
    def _extract_act_features_unsafe(self, images: torch.Tensor, depth_images: torch.Tensor,
                                   qpos: torch.Tensor) -> torch.Tensor:
        """不安全的特征提取（用于维度检测）"""
        with torch.no_grad():
            return self._extract_act_features(images, depth_images, qpos)
    
    def _extract_act_features(self, images: torch.Tensor, depth_images: torch.Tensor, 
                             qpos: torch.Tensor) -> torch.Tensor:
        """
        使用ACT模型提取多模态特征
        关键修复：按照ACT框架的正确方式提取backbone特征
        """
        
        # 图像归一化
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
        
        batch_size = images.shape[0]
        images_norm = normalize(images)  # 直接归一化
        
        # 深度图像处理
        depth_norm = None
        if self.use_depth_image and depth_images is not None and depth_images.numel() > 1:
            depth_normalize = transforms.Normalize(mean=[0.5], std=[0.5])
            depth_norm = depth_normalize(depth_images)
        
        # 通过ACT的backbone提取视觉特征 - 遵循ACT框架规范
        act_model = self.act_policy.model
        
        # 按照DETRVAE.forward的方式提取特征，但使用全局平均池化（与QCriticEncoder一致）
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            # 正确的backbone调用方式: features, src_pos = backbone(image)
            features, src_pos = act_model.backbones[cam_id](images_norm[:, cam_id])
            features = features[0]  # 取最后一层特征
            
            # 处理深度图像（如果有）
            if act_model.depth_backbones is not None and depth_norm is not None:
                features_depth = act_model.depth_backbones[cam_id](depth_norm[:, cam_id].unsqueeze(dim=1))
                combined_features = torch.cat([features, features_depth], axis=1)
                projected_features = act_model.input_proj(combined_features)
            else:
                projected_features = act_model.input_proj(features)
            
            # 关键修复：使用全局平均池化替代直接展平，与QCriticEncoder保持一致
            # projected_features shape: [batch, hidden_dim, H, W]
            # 使用全局平均池化将空间维度压缩为固定长度向量
            pooled_features = torch.mean(projected_features, dim=[2, 3])  # [batch, hidden_dim]
            all_cam_features.append(pooled_features)
        
        # 直接拼接所有相机的固定维度特征向量（与QCriticEncoder一致）
        visual_features_flat = torch.cat(all_cam_features, dim=1)  # [batch, hidden_dim * num_cameras]
        
        # 处理机器人状态特征
        robot_state_features = act_model.input_proj_robot_state(qpos)
        
        # 拼接所有特征
        combined_features = torch.cat([visual_features_flat, robot_state_features], dim=-1)
        
        return combined_features
    
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """更新策略网络"""
        
        # 从嵌套的observations结构中提取数据
        observations = batch['observations']
        images = observations['images']
        depth_images = observations.get('depth_images', None)
        qpos = observations['qpos']
        
        actions = batch['actions']  # [batch_size, chunk_size, action_dim]
        rewards = batch['rewards']  # [batch_size, sequence_length] 累积折扣奖励序列
        
        batch_size = images.shape[0]
        
        # 展平动作块
        if len(actions.shape) == 3:
            actions_flat = actions.reshape(batch_size, -1)
        else:
            actions_flat = actions
        
        # 构建观察字典（用于新的QCriticNetwork）
        observations = {
            'images': images,
            'depth_images': depth_images,
            'robot_state': qpos
        }
        
        # 使用独立Critic网络，直接传递observations
        critic_loss = self._update_critic_independent(observations, actions_flat, batch)
        
        # 更新Actor（仍使用ACT特征）
        if not hasattr(self, '_act_features_cache'):
            self._act_features_cache = self._extract_act_features(images, depth_images, qpos)
        actor_losses = self._update_actor(self._act_features_cache, actions_flat, batch)
        
        # 更新ACT backbone（如果允许）
        if self.update_act_backbone and self.act_optimizer is not None:
            self._update_act_backbone()
        
        # 更新Critic编码器（如果有）
        if self.critic_encoder_optimizer is not None:
            self.critic_encoder_optimizer.step()
            self.critic_encoder_optimizer.zero_grad()
        
        # 更新目标网络
        self._update_target_networks()
        
        # 清理缓存
        if hasattr(self, '_act_features_cache'):
            delattr(self, '_act_features_cache')
        
        # 合并所有损失信息
        result = {
            'critic_loss': critic_loss,
        }
        result.update(actor_losses)  # 包含详细的Actor损失分解
        
        return result
    
    def _update_critic_independent(self, observations: Dict[str, torch.Tensor], 
                                  actions: torch.Tensor, batch: Dict[str, torch.Tensor]) -> float:
        """更新独立Critic网络（QCriticNetwork）"""
        
        # 当前Q值预测
        q_values = self.critic(observations, actions)  # [num_ensembles, batch_size]
        
        # Q-Chunking目标值计算 - 使用bootstrap机制（与标准ACFQL一致）
        with torch.no_grad():
            # 获取奖励和mask（标准ACFQL格式）
            rewards = batch['rewards']  # [batch_size, chunk_size] 
            masks = batch['masks']      # [batch_size, chunk_size]
            
            # 标准ACFQL：使用序列最后一步的奖励和mask
            final_step_rewards = rewards[:, -1]  # [batch_size] - 对应batch['rewards'][..., -1]
            final_step_masks = masks[:, -1]      # [batch_size] - 对应batch['masks'][..., -1]
            
            # 实现完整的Bootstrap机制：target_q = reward + γ^H * mask * next_q
            next_observations = batch.get('next_observations', None)  # 数据加载器提供的单帧next_obs
            
            if next_observations is not None:
                # next_observations来自数据加载器，已经是单帧格式
                next_obs_dict = {
                    'images': next_observations['images'],     # [batch_size, num_cameras, C, H, W]
                    'robot_state': next_observations['qpos'],  # [batch_size, qpos_dim] - QCriticNetwork期望robot_state字段
                }
                if 'depth_images' in next_observations:
                    next_obs_dict['depth_images'] = next_observations['depth_images']  # [batch_size, num_cameras, 1, H, W]
                
                # 使用当前策略采样下一步动作
                next_actions = self.sample_actions(next_obs_dict, evaluation=True)  # [batch_size, action_dim]
                
                # 使用target critic计算next Q值
                next_q_values = self.target_critic(next_obs_dict, next_actions)  # [num_ensembles, batch_size]
                
                # Q值聚合（按配置选择min或mean）
                if self.q_agg == 'min':
                    next_q = next_q_values.min(dim=0)[0]  # [batch_size]
                else:
                    next_q = next_q_values.mean(dim=0)    # [batch_size]
                
                # Q-Chunking的γ^H折扣计算（与标准ACFQL一致）
                gamma_h = self.discount ** self.chunk_size
                
                # 正确的Bootstrap mask：基于下一状态是否为terminal
                next_terminal = batch.get('next_terminal', False)  # 数据加载器提供的terminal信息
                if isinstance(next_terminal, bool):
                    # 单个bool值，扩展为batch
                    bootstrap_mask = torch.tensor([1.0 - next_terminal] * len(final_step_rewards), 
                                                device=final_step_rewards.device, dtype=torch.float32)
                else:
                    # batch的bool数组
                    bootstrap_mask = (1.0 - next_terminal.float()).to(final_step_rewards.device)
                
                # 标准ACFQL Bootstrap公式：target_q = reward + γ^H * (1-terminal) * next_q
                target_q_single = final_step_rewards + gamma_h * bootstrap_mask * next_q  # [batch_size]
                target_q = target_q_single.unsqueeze(0).expand(self.num_qs, -1)  # [num_ensembles, batch_size]
            else:
                # 如果没有next_observations，使用简化版本（仅用于向后兼容）
                if self.rank == 0:
                    print("警告: 没有next_observations，使用简化的Bootstrap计算")
                gamma_h = self.discount ** self.chunk_size
                target_q_single = final_step_rewards * gamma_h  # 简单折扣
                target_q = target_q_single.unsqueeze(0).expand(self.num_qs, -1)  # [num_ensembles, batch_size]
        
        # Critic损失 - 应用valid掩码（标准Q-chunking要求）
        critic_loss_raw = F.mse_loss(q_values, target_q, reduction='none').mean(dim=0)  # [batch_size]
        
        # 应用valid掩码（如果存在）
        if 'valid' in batch:
            # 使用最后一步的valid掩码
            valid_data = batch['valid']
            if len(valid_data.shape) > 1:
                valid_mask = valid_data[:, -1].float()  # [batch_size]
            else:
                valid_mask = valid_data.float()  # [batch_size]
            
            # 应用掩码并归一化
            valid_count = valid_mask.sum() + 1e-8
            critic_loss = (critic_loss_raw * valid_mask).sum() / valid_count
        else:
            # 回退到平均损失
            critic_loss = critic_loss_raw.mean()
        
        # 更新Critic网络
        self.critic_optimizer.zero_grad()
        if self.critic_encoder_optimizer is not None:
            self.critic_encoder_optimizer.zero_grad()
        
        critic_loss.backward(retain_graph=True)  # 保留计算图用于ACT更新
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def _update_critic_legacy(self, features: torch.Tensor, actions: torch.Tensor, 
                             batch: Dict[str, torch.Tensor]) -> float:
        """更新传统Critic网络（QCriticHead），向后兼容"""
        
        # 当前Q值预测
        q_values = self.critic(features, actions)  # [num_ensembles, batch_size]
        
        # Q-Chunking目标值计算 - 使用bootstrap机制（与标准ACFQL一致）
        with torch.no_grad():
            # 获取奖励和mask（标准ACFQL格式）
            rewards = batch['rewards']  # [batch_size, chunk_size] 
            masks = batch['masks']      # [batch_size, chunk_size]
            
            # 标准ACFQL：使用序列最后一步的奖励和mask
            final_step_rewards = rewards[:, -1]  # [batch_size] - 对应batch['rewards'][..., -1]
            final_step_masks = masks[:, -1]      # [batch_size] - 对应batch['masks'][..., -1]
            
            # 实现完整的Bootstrap机制：target_q = reward + γ^H * mask * next_q
            next_observations = batch.get('next_observations', None)  # 数据加载器提供的单帧next_obs
            
            if next_observations is not None:
                # next_observations来自数据加载器，已经是单帧格式
                next_obs_dict = {
                    'images': next_observations['images'],     # [batch_size, num_cameras, C, H, W]
                    'robot_state': next_observations['qpos'],  # [batch_size, qpos_dim] - QCriticNetwork期望robot_state字段
                }
                if 'depth_images' in next_observations:
                    next_obs_dict['depth_images'] = next_observations['depth_images']  # [batch_size, num_cameras, 1, H, W]
                
                # 使用当前策略采样下一步动作
                next_actions = self.sample_actions(next_obs_dict, evaluation=True)  # [batch_size, action_dim]
                
                # 使用target critic计算next Q值
                next_q_values = self.target_critic(next_obs_dict, next_actions)  # [num_ensembles, batch_size]
                
                # Q值聚合（按配置选择min或mean）
                if self.q_agg == 'min':
                    next_q = next_q_values.min(dim=0)[0]  # [batch_size]
                else:
                    next_q = next_q_values.mean(dim=0)    # [batch_size]
                
                # Q-Chunking的γ^H折扣计算（与标准ACFQL一致）
                gamma_h = self.discount ** self.chunk_size
                
                # 正确的Bootstrap mask：基于下一状态是否为terminal
                next_terminal = batch.get('next_terminal', False)  # 数据加载器提供的terminal信息
                if isinstance(next_terminal, bool):
                    # 单个bool值，扩展为batch
                    bootstrap_mask = torch.tensor([1.0 - next_terminal] * len(final_step_rewards), 
                                                device=final_step_rewards.device, dtype=torch.float32)
                else:
                    # batch的bool数组
                    bootstrap_mask = (1.0 - next_terminal.float()).to(final_step_rewards.device)
                
                # 标准ACFQL Bootstrap公式：target_q = reward + γ^H * (1-terminal) * next_q
                target_q_single = final_step_rewards + gamma_h * bootstrap_mask * next_q  # [batch_size]
                target_q = target_q_single.unsqueeze(0).expand(self.num_qs, -1)  # [num_ensembles, batch_size]
            else:
                # 如果没有next_observations，使用简化版本（仅用于向后兼容）
                if self.rank == 0:
                    print("警告: 没有next_observations，使用简化的Bootstrap计算")
                gamma_h = self.discount ** self.chunk_size
                target_q_single = final_step_rewards * gamma_h  # 简单折扣
                target_q = target_q_single.unsqueeze(0).expand(self.num_qs, -1)  # [num_ensembles, batch_size]
        
        # Critic损失
        critic_loss = F.mse_loss(q_values, target_q)
        
        # 更新
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)  # 保留计算图用于ACT更新
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def _update_actor(self, features: torch.Tensor, target_actions: torch.Tensor, 
                     batch: Dict[str, torch.Tensor]) -> float:
        """根据架构模式更新Actor网络"""
        
        if self.actor_architecture == 'flow_dual':
            return self._update_flow_actors(features, target_actions, batch)
        elif self.actor_architecture == 'act_direct':
            return self._update_act_actor(features, target_actions, batch)
        else:
            raise ValueError(f"不支持的actor_architecture: {self.actor_architecture}")
    
    def _update_flow_actors(self, features: torch.Tensor, target_actions: torch.Tensor, 
                           batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Flow模式：更新双Actor网络，实现BC Flow + 蒸馏机制"""
        
        batch_size = features.shape[0]
        
        # 1. BC Flow Loss（BC Flow Actor学习行为克隆）
        # 生成噪声 x_0
        x_0 = torch.randn(batch_size, self.action_chunk_dim, device=features.device)
        
        # 目标动作 x_1 (已经是展平的动作序列)
        x_1 = target_actions
        
        # 随机时间插值参数
        t = torch.rand(batch_size, 1, device=features.device)
        
        # 流匹配插值轨迹: x_t = (1-t) * x_0 + t * x_1
        x_t = (1 - t) * x_0 + t * x_1
        
        # 真实速度场: v = x_1 - x_0
        vel_true = x_1 - x_0
        
        # BC Flow Actor预测速度场
        vel_pred = self.bc_flow_actor(features, x_t, t)
        
        # 应用有效性掩码
        valid_mask = batch['valid']  # [batch_size, sequence_length]
        
        if len(vel_pred.shape) == 2 and len(valid_mask.shape) == 2:
            # 重塑掩码以匹配动作维度
            action_dim = self.action_chunk_dim // self.chunk_size
            valid_mask_expanded = valid_mask.unsqueeze(-1).expand(-1, -1, action_dim)
            valid_mask_flat = valid_mask_expanded.reshape(batch_size, -1)
            
            # 获取正样本mask (基于final step rewards)
            final_rewards = batch['rewards'][:, -1]  # [batch_size]
            positive_mask = (final_rewards > 0.5).float()  # [batch_size]
            positive_mask_expanded = positive_mask.unsqueeze(-1).expand(-1, valid_mask_flat.shape[1])  # [batch_size, action_chunk_dim]
            
            # 计算带掩码的BC流损失 - 同时考虑valid_mask和positive_mask
            combined_mask = valid_mask_flat * positive_mask_expanded  # 只对正样本的有效步骤计算BC损失
            if combined_mask.sum() > 0:
                bc_flow_loss_raw = (vel_pred - vel_true) ** 2
                bc_flow_loss = (bc_flow_loss_raw * combined_mask).sum() / combined_mask.sum()
            else:
                bc_flow_loss = torch.tensor(0.0, device=vel_pred.device)
        else:
            # 获取正样本mask (基于final step rewards)  
            final_rewards = batch['rewards'][:, -1]  # [batch_size]
            positive_mask = (final_rewards > 0.5).float()  # 正样本mask
            
            # 只对正样本计算BC损失
            if positive_mask.sum() > 0:
                bc_flow_loss_raw = F.mse_loss(vel_pred, vel_true, reduction='none').mean(dim=1)  # [batch_size]
                bc_flow_loss = (bc_flow_loss_raw * positive_mask).sum() / positive_mask.sum()
            else:
                bc_flow_loss = torch.tensor(0.0, device=vel_pred.device)
        
        # 2. 蒸馏损失（Onestep Actor学习BC Flow Actor的输出）
        distill_loss = torch.tensor(0.0, device=features.device)
        q_loss = torch.tensor(0.0, device=features.device)
        
        if self.actor_type == "distill" or self.actor_type == "distill-ddpg":
            # 生成Teacher动作（通过BC Flow Actor）
            with torch.no_grad():
                teacher_noise = torch.randn(batch_size, self.action_chunk_dim, device=features.device)
                teacher_actions = self.compute_flow_actions(features, teacher_noise)
            
            # Student学习Teacher的输出
            student_noise = teacher_noise  # 使用相同的噪声
            student_actions = self.onestep_actor(features, student_noise)
            student_actions = torch.clamp(student_actions, -1, 1)
            
            # 蒸馏损失
            distill_loss = F.mse_loss(student_actions, teacher_actions)
            
            # Q损失（如果是distill-ddpg模式）
            if self.actor_type == "distill-ddpg":
                # 使用Student动作进行Q值最大化
                if self.use_independent_critic:
                    # 简化处理：暂时跳过Q损失，因为需要完整的观测
                    pass
                else:
                    q_values = self.critic(features, student_actions)
                    q_value = q_values.mean(dim=0)  # 平均ensemble
                    q_loss = -q_value.mean()  # 最大化Q值
        
        # 3. 总损失
        total_actor_loss = bc_flow_loss + self.alpha * distill_loss + q_loss
        
        # 更新
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()
        
        # 返回详细损失信息
        return {
            'actor_loss': total_actor_loss.item(),
            'bc_flow_loss': bc_flow_loss.item(),
            'distill_loss': distill_loss.item(),
            'q_loss': q_loss.item()
        }
    
    def _update_act_actor(self, features: torch.Tensor, target_actions: torch.Tensor,
                         batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """ACT模式：直接使用ACT模型作为Actor，用RL增强ACT能力"""
        
        # 1. 准备观测数据（重构为ACT输入格式）
        # 从嵌套的observations结构中提取数据
        observations = batch['observations']
        images = observations['images']  # [batch_size, num_cameras, 3, H, W]
        qpos = observations['qpos']      # [batch_size, qpos_dim]
        depth_images = observations.get('depth_images')  # 可选
        
        if images is None or qpos is None:
            # 如果没有完整观测，回退到纯BC损失
            bc_loss = torch.tensor(0.0, device=features.device)
            q_loss = torch.tensor(0.0, device=features.device)
            
            if self.rank == 0:
                print("警告: ACT模式缺少完整观测数据，跳过Actor更新")
            
            return {
                'actor_loss': 0.0,
                'bc_loss': bc_loss.item(),
                'q_loss': q_loss.item()
            }
        
        # 2. ACT前向传播（注意：需要梯度用于优化）
        # 重构观测为Critic期望的格式
        observations = {
            'robot_state': qpos,  # QCriticNetwork期望robot_state字段
            'images': images
        }
        if depth_images is not None:
            observations['depth_images'] = depth_images
        
        # ACT输出动作序列 [batch_size, chunk_size, action_dim]
        act_actions = self.act_policy(
            images,
            depth_images,
            qpos,
            actions=None,
            action_is_pad=None,
            z_vector=None
        )
        act_actions_flat = act_actions.reshape(act_actions.shape[0], -1)  # 展平为 [batch_size, chunk_size * action_dim]
        
        # 3. BC损失（L2正则化）- 只对正样本计算
        # 获取正样本mask (基于final step rewards)
        final_rewards = batch['rewards'][:, -1]  # [batch_size]
        positive_mask = (final_rewards > 0.5).float()  # 正样本mask
        
        # 只对正样本计算BC损失
        if positive_mask.sum() > 0:
            bc_loss_raw = F.mse_loss(act_actions_flat, target_actions, reduction='none').mean(dim=1)  # [batch_size]
            bc_loss = (bc_loss_raw * positive_mask).sum() / positive_mask.sum()
        else:
            bc_loss = torch.tensor(0.0, device=act_actions_flat.device)
        
        # 4. Q损失（RL增强）- 使用ACT输出的动作
        q_loss = torch.tensor(0.0, device=features.device)
        if self.update_act_directly:
            # 计算Q值
            if self.use_independent_critic:
                # 使用独立Critic，需要完整观测
                q_values = self.critic(observations, act_actions_flat)  # [num_qs, batch_size]
            else:
                # 使用特征版Critic
                q_values = self.critic(features, act_actions_flat)  # [num_qs, batch_size]
            
            # Q值聚合
            if self.q_agg == 'min':
                q_agg = q_values.min(dim=0)[0]  # [batch_size]
            else:
                q_agg = q_values.mean(dim=0)   # [batch_size]
            
            # Q损失：最大化Q值
            q_loss = -q_agg.mean()
        
        # 5. 总损失 - 应用ACT模式专用权重
        total_loss = self.act_bc_weight * bc_loss + self.act_q_weight * q_loss
        
        # 6. 更新ACT参数
        if self.update_act_directly and hasattr(self, 'act_optimizer') and self.act_optimizer is not None:
            self.act_optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            self.act_optimizer.step()
        
        return {
            'actor_loss': total_loss.item(),
            'bc_loss': bc_loss.item(),
            'q_loss': q_loss.item()
        }
    
    def _update_act_backbone(self) -> float:
        """更新ACT backbone参数"""
        if not self.update_act_backbone or self.act_optimizer is None:
            return 0.0
        
        # ACT backbone的梯度已通过特征计算链传递
        # 这里只需要执行优化步骤
        self.act_optimizer.step()
        self.act_optimizer.zero_grad()
        
        return 0.0  # 返回dummy值，实际损失已在critic/actor中计算
    
    def _update_target_networks(self, tau: Optional[float] = None):
        """软更新目标网络"""
        if tau is None:
            tau = self.tau
        
        for param, target_param in zip(self.critic.parameters(), 
                                     self.target_critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        checkpoint = {
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'act_policy': self.act_policy.state_dict(),
        }
        
        # Flow模式组件（可选）
        if self.bc_flow_actor is not None:
            checkpoint['bc_flow_actor'] = self.bc_flow_actor.state_dict()
        if self.onestep_actor is not None:
            checkpoint['onestep_actor'] = self.onestep_actor.state_dict()
        if self.actor_optimizer is not None:
            checkpoint['actor_optimizer'] = self.actor_optimizer.state_dict()
        if self.act_optimizer is not None:
            checkpoint['act_optimizer'] = self.act_optimizer.state_dict()
        
        # 学习率调度器状态
        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler'] = self.lr_scheduler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location='cpu')
        
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        # Flow模式组件（可选加载）
        if 'bc_flow_actor' in checkpoint and self.bc_flow_actor is not None:
            self.bc_flow_actor.load_state_dict(checkpoint['bc_flow_actor'])
        elif 'flow_actor' in checkpoint and self.bc_flow_actor is not None:
            # 兼容旧版本
            self.bc_flow_actor.load_state_dict(checkpoint['flow_actor'])
            print("警告: 加载旧版本单Actor模型到BC Flow Actor")
            
        if 'onestep_actor' in checkpoint and self.onestep_actor is not None:
            self.onestep_actor.load_state_dict(checkpoint['onestep_actor'])
            
        if 'actor_optimizer' in checkpoint and self.actor_optimizer is not None:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        
        if 'act_optimizer' in checkpoint and self.act_optimizer is not None:
            self.act_optimizer.load_state_dict(checkpoint['act_optimizer'])
        
        if 'act_policy' in checkpoint:
            self.act_policy.load_state_dict(checkpoint['act_policy'])
        
        # 学习率调度器状态
        if 'lr_scheduler' in checkpoint and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    
    def sample_actions(self, observations: Dict[str, torch.Tensor], 
                      rng=None, evaluation: bool = False) -> torch.Tensor:
        """根据架构模式采样动作"""
        
        if self.actor_architecture == 'flow_dual':
            return self._sample_flow_actions(observations, rng, evaluation)
        elif self.actor_architecture == 'act_direct':
            return self._sample_act_actions(observations, rng, evaluation)
        else:
            raise ValueError(f"不支持的actor_architecture: {self.actor_architecture}")
    
    def _sample_flow_actions(self, observations: Dict[str, torch.Tensor], 
                            rng=None, evaluation: bool = False) -> torch.Tensor:
        """Flow模式：使用Flow Actor采样动作"""
        
        batch_size = observations['images'].shape[0]
        device = observations['images'].device
        
        # 提取ACT特征
        features = self._extract_act_features(
            observations['images'], 
            observations.get('depth_images', torch.zeros(1, device=device)),
            observations['robot_state']
        )
        
        if self.actor_type == "best-of-n" and not evaluation:
            # Best-of-N采样策略
            num_candidates = self.actor_num_samples
            
            # 生成多个噪声候选
            noises = torch.randn(batch_size, num_candidates, 
                               self.action_chunk_dim, device=device)
            
            # 为每个样本生成动作候选
            all_actions = []
            for i in range(num_candidates):
                candidate_actions = self.compute_flow_actions(features, noises[:, i])
                all_actions.append(candidate_actions)
            
            # 堆叠所有候选: [batch_size, num_candidates, action_dim]
            action_candidates = torch.stack(all_actions, dim=1)
            
            # 使用Critic评估所有候选
            # 重塑以便批量评估: [batch_size * num_candidates, action_dim]
            features_expanded = features.unsqueeze(1).expand(-1, num_candidates, -1)
            features_flat = features_expanded.reshape(-1, features.shape[-1])
            actions_flat = action_candidates.reshape(-1, self.action_chunk_dim)
            
            # 计算Q值: [num_qs, batch_size * num_candidates] 
            if self.use_independent_critic:
                # 对于独立Critic，需要重构观测
                obs_expanded = {}
                for key, value in observations.items():
                    if key in ['images', 'qpos']:  # 只处理关键观测
                        expanded = value.unsqueeze(1).expand(-1, num_candidates, *value.shape[1:])
                        obs_expanded[key] = expanded.reshape(-1, *value.shape[1:])
                q_values = self.critic(obs_expanded, actions_flat)
            else:
                q_values = self.critic(features_flat, actions_flat)
            
            # 聚合Q值
            if self.q_agg == "mean":
                q_agg = q_values.mean(dim=0)  # [batch_size * num_candidates]
            else:  # min
                q_agg = q_values.min(dim=0)[0]  # [batch_size * num_candidates]
            
            # 重塑回: [batch_size, num_candidates]
            q_scores = q_agg.reshape(batch_size, num_candidates)
            
            # 选择Q值最高的动作
            best_indices = torch.argmax(q_scores, dim=-1)  # [batch_size]
            
            # 提取最佳动作
            selected_actions = action_candidates[torch.arange(batch_size), best_indices]
            
            return selected_actions
            
        else:
            # 单次采样（快速推理）- 使用Onestep Actor
            noises = torch.randn(batch_size, self.action_chunk_dim, device=device)
            if self.onestep_actor is not None:
                actions = self.onestep_actor(features, noises)
                actions = torch.clamp(actions, -1, 1)
            else:
                # 回退到Flow计算
                actions = self.compute_flow_actions(features, noises)
            return actions
    
    def _sample_act_actions(self, observations: Dict[str, torch.Tensor], 
                           rng=None, evaluation: bool = False) -> torch.Tensor:
        """ACT模式：使用ACT的VAE潜空间采样动作"""
        
        with torch.no_grad():  # 推理时不需要梯度
            if self.actor_type == "best-of-n" and not evaluation:
                # Best-of-N采样：在ACT的潜空间进行采样
                batch_size = observations['images'].shape[0]
                device = observations['images'].device
                num_candidates = self.actor_num_samples
                latent_dim = 32  # ACT的潜空间维度
                
                # 在潜空间生成多个候选z_vector
                z_vectors = torch.randn(num_candidates, batch_size, latent_dim, device=device) * 0.5  # 适当的采样强度
                
                # 为每个z_vector生成动作候选
                candidates = []
                for i in range(num_candidates):
                    z_vector = z_vectors[i]  # [batch_size, latent_dim]
                    
                    # 调用ACT模型，使用当前z_vector
                    candidate_actions = self.act_policy(
                        observations['images'],
                        observations.get('depth_images', None),
                        observations['robot_state'],
                        actions=None,  # 推理模式
                        action_is_pad=None,
                        z_vector=z_vector  # 关键：在潜空间采样
                    )  # [batch_size, chunk_size, action_dim]
                    
                    # 展平动作序列用于Critic评估
                    candidate_flat = candidate_actions.reshape(batch_size, -1)
                    candidates.append(candidate_flat)
                
                # 堆叠所有候选
                action_candidates = torch.stack(candidates, dim=1)  # [batch_size, num_candidates, action_dim_flat]
                
                # 使用Critic评估
                features = self._extract_act_features(
                    observations['images'], 
                    observations.get('depth_images', torch.zeros(1, device=device)),
                    observations['robot_state']
                )
                
                # 批量计算Q值
                features_expanded = features.unsqueeze(1).expand(-1, num_candidates, -1)
                features_flat = features_expanded.reshape(-1, features.shape[-1])
                actions_flat = action_candidates.reshape(-1, self.action_chunk_dim)
                
                if self.use_independent_critic:
                    # 重构观测
                    obs_expanded = {}
                    for key, value in observations.items():
                        if key in ['images', 'qpos']:
                            expanded = value.unsqueeze(1).expand(-1, num_candidates, *value.shape[1:])
                            obs_expanded[key] = expanded.reshape(-1, *value.shape[1:])
                    q_values = self.critic(obs_expanded, actions_flat)
                else:
                    q_values = self.critic(features_flat, actions_flat)
                
                # 选择最优动作
                if self.q_agg == "mean":
                    q_agg = q_values.mean(dim=0)
                else:
                    q_agg = q_values.min(dim=0)[0]
                
                q_scores = q_agg.reshape(batch_size, num_candidates)
                best_indices = torch.argmax(q_scores, dim=-1)
                
                selected_actions = action_candidates[torch.arange(batch_size), best_indices]
                return selected_actions
            
            else:
                # 确定性采样：使用默认的ACT行为（z_vector=None）
                act_actions = self.act_policy(
                    observations['images'],
                    observations.get('depth_images', None),
                    observations['robot_state'],
                    actions=None,  # 推理模式
                    action_is_pad=None,
                    z_vector=None  # 使用默认行为（零向量）
                )  # [batch_size, chunk_size, action_dim]
                
                return act_actions.reshape(act_actions.shape[0], -1)  # 展平返回
    
