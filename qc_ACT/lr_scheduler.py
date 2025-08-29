"""
二层学习率调度器：全局（跨epoch）+ 局部（epoch内）
支持多种调度策略，适用于qc-ACT算法的不同优化器
"""
import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, List, Optional, Any


class TwoLevelLRScheduler:
    """
    二层学习率调度器
    - 全局调度：跨epoch的学习率变化
    - 局部调度：每个epoch内部的学习率变化
    """
    
    def __init__(self, 
                 optimizers_dict: Dict[str, torch.optim.Optimizer],
                 config: Dict[str, Any],
                 total_epochs: int,
                 steps_per_epoch: int,
                 rank: int = 0):
        """
        Args:
            optimizers_dict: {name: optimizer} 字典
            config: 调度器配置
            total_epochs: 总训练轮数
            steps_per_epoch: 每个epoch的步数
            rank: 进程rank，用于控制打印
        """
        self.optimizers = {k: v for k, v in optimizers_dict.items() if v is not None}
        self.config = config
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.rank = rank
        
        # 记录初始学习率
        self.base_lrs = {}
        for name, optimizer in self.optimizers.items():
            self.base_lrs[name] = optimizer.param_groups[0]['lr']
        
        # 当前状态
        self.current_epoch = 0
        self.current_step_in_epoch = 0
        self.current_base_lrs = {}
        self.current_lrs = {}
        
        # 全局和局部调度器配置
        self.global_config = config.get('global', {})
        self.local_config = config.get('local', {})
        self.optimizer_configs = config.get('optimizer_configs', {})
        
        if self.rank == 0:
            print(f"二层学习率调度器初始化:")
            print(f"  管理优化器: {list(self.optimizers.keys())}")
            print(f"  全局调度策略: {self.global_config.get('type', 'constant')}")
            print(f"  局部调度策略: {self.local_config.get('type', 'constant')}")
            print(f"  总epochs: {total_epochs}, 每epoch步数: {steps_per_epoch}")
    
    def step_epoch(self, epoch: int):
        """epoch开始时调用，更新全局学习率"""
        self.current_epoch = epoch
        self.current_step_in_epoch = 0
        
        # 计算全局学习率因子
        global_factor = self._get_global_factor(epoch)
        
        # 为每个优化器计算基础学习率
        self.current_base_lrs = {}
        for name, optimizer in self.optimizers.items():
            base_lr = self.base_lrs[name]
            opt_config = self.optimizer_configs.get(name, {})
            global_scale = opt_config.get('global_scale', 1.0)
            
            new_base_lr = base_lr * global_factor * global_scale
            self.current_base_lrs[name] = new_base_lr
        
        # 立即应用epoch开始时的学习率（step_in_epoch=0）
        self.step_batch(0)
        
        if self.rank == 0 and epoch % 100 == 0:  # 每100个epoch打印一次
            lr_info = ", ".join([f"{name}={lr:.2e}" for name, lr in self.current_base_lrs.items()])
            print(f"Epoch {epoch} 全局学习率更新: {lr_info} (global_factor={global_factor:.4f})")
    
    def step_batch(self, step_in_epoch: int):
        """每个batch训练后调用，更新局部学习率"""
        self.current_step_in_epoch = step_in_epoch
        
        # 计算局部学习率因子
        local_factor = self._get_local_factor(step_in_epoch)
        
        # 为每个优化器应用局部学习率
        self.current_lrs = {}
        for name, optimizer in self.optimizers.items():
            base_lr = self.current_base_lrs.get(name, self.base_lrs[name])
            opt_config = self.optimizer_configs.get(name, {})
            local_scale = opt_config.get('local_scale', 1.0)
            
            new_lr = base_lr * local_factor * local_scale
            self.current_lrs[name] = new_lr
            
            # 更新优化器学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
    
    def get_current_lrs(self) -> Dict[str, float]:
        """获取当前学习率，用于日志记录"""
        return self.current_lrs.copy()
    
    def get_lr_info(self) -> str:
        """获取学习率信息字符串，用于打印"""
        lr_strs = []
        for name, lr in self.current_lrs.items():
            lr_strs.append(f"{name}_lr={lr:.2e}")
        return ", ".join(lr_strs)
    
    def _get_global_factor(self, epoch: int) -> float:
        """计算全局学习率因子（跨epoch调度）"""
        strategy = self.global_config.get('type', 'constant')
        
        if strategy == 'constant':
            return 1.0
            
        elif strategy == 'cosine':
            total_epochs = self.global_config.get('total_epochs', self.total_epochs)
            min_lr_ratio = self.global_config.get('min_lr_ratio', 0.01)
            
            factor = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * epoch / total_epochs)
            )
            return factor
            
        elif strategy == 'cosine_warmup':
            warmup_epochs = self.global_config.get('warmup_epochs', 500)
            total_epochs = self.global_config.get('total_epochs', self.total_epochs)
            min_lr_ratio = self.global_config.get('min_lr_ratio', 0.01)
            
            if epoch < warmup_epochs:
                # Warmup阶段：线性增长，从很小的值开始而不是0
                return (epoch + 1) / warmup_epochs
            else:
                # Cosine annealing阶段
                cosine_epochs = total_epochs - warmup_epochs
                cosine_progress = (epoch - warmup_epochs) / cosine_epochs
                factor = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
                    1.0 + math.cos(math.pi * cosine_progress)
                )
                return factor
                
        elif strategy == 'step':
            step_size = self.global_config.get('step_size', 2000)
            gamma = self.global_config.get('gamma', 0.5)
            return gamma ** (epoch // step_size)
            
        elif strategy == 'exponential':
            gamma = self.global_config.get('gamma', 0.95)
            return gamma ** epoch
            
        elif strategy == 'linear':
            total_epochs = self.global_config.get('total_epochs', self.total_epochs)
            min_lr_ratio = self.global_config.get('min_lr_ratio', 0.01)
            
            progress = min(epoch / total_epochs, 1.0)
            return 1.0 - progress * (1.0 - min_lr_ratio)
            
        else:
            if self.rank == 0:
                print(f"警告: 未知的全局调度策略 '{strategy}'，使用constant")
            return 1.0
    
    def _get_local_factor(self, step_in_epoch: int) -> float:
        """计算局部学习率因子（epoch内调度）"""
        strategy = self.local_config.get('type', 'constant')
        
        if strategy == 'constant':
            return 1.0
            
        elif strategy == 'linear_warmup':
            warmup_steps_ratio = self.local_config.get('warmup_steps_ratio', 0.1)
            warmup_steps = int(self.steps_per_epoch * warmup_steps_ratio)
            
            if step_in_epoch < warmup_steps:
                return (step_in_epoch + 1) / warmup_steps
            else:
                return 1.0
                
        elif strategy == 'cosine':
            min_lr_ratio = self.local_config.get('min_lr_ratio', 0.5)
            progress = step_in_epoch / self.steps_per_epoch
            
            factor = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
            return factor
            
        elif strategy == 'warmup_cosine':
            warmup_steps_ratio = self.local_config.get('warmup_steps_ratio', 0.1)
            min_lr_ratio = self.local_config.get('min_lr_ratio', 0.5)
            warmup_steps = int(self.steps_per_epoch * warmup_steps_ratio)
            
            if step_in_epoch < warmup_steps:
                # Warmup阶段
                return (step_in_epoch + 1) / warmup_steps
            else:
                # Cosine阶段
                cosine_steps = self.steps_per_epoch - warmup_steps
                cosine_progress = (step_in_epoch - warmup_steps) / cosine_steps
                factor = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
                    1.0 + math.cos(math.pi * cosine_progress)
                )
                return factor
                
        elif strategy == 'triangular':
            peak_step_ratio = self.local_config.get('peak_step_ratio', 0.5)
            peak_step = int(self.steps_per_epoch * peak_step_ratio)
            
            if step_in_epoch <= peak_step:
                # 上升阶段
                return (step_in_epoch + 1) / (peak_step + 1)
            else:
                # 下降阶段
                remaining_steps = self.steps_per_epoch - peak_step
                progress = (step_in_epoch - peak_step) / remaining_steps
                return 1.0 - progress * 0.5  # 下降到0.5
                
        else:
            if self.rank == 0:
                print(f"警告: 未知的局部调度策略 '{strategy}'，使用constant")
            return 1.0
    
    def state_dict(self) -> Dict[str, Any]:
        """保存调度器状态"""
        return {
            'current_epoch': self.current_epoch,
            'current_step_in_epoch': self.current_step_in_epoch,
            'current_base_lrs': self.current_base_lrs,
            'current_lrs': self.current_lrs,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """加载调度器状态"""
        self.current_epoch = state_dict.get('current_epoch', 0)
        self.current_step_in_epoch = state_dict.get('current_step_in_epoch', 0)
        self.current_base_lrs = state_dict.get('current_base_lrs', {})
        self.current_lrs = state_dict.get('current_lrs', {})
        
        # 恢复优化器学习率
        for name, lr in self.current_lrs.items():
            if name in self.optimizers:
                for param_group in self.optimizers[name].param_groups:
                    param_group['lr'] = lr


def create_two_level_scheduler(optimizers_dict: Dict[str, torch.optim.Optimizer],
                              config: Dict[str, Any],
                              total_epochs: int,
                              steps_per_epoch: int,
                              rank: int = 0) -> Optional[TwoLevelLRScheduler]:
    """
    创建二层学习率调度器的工厂函数
    """
    if not config.get('enable', False):
        if rank == 0:
            print("学习率调度器已禁用")
        return None
    
    return TwoLevelLRScheduler(
        optimizers_dict=optimizers_dict,
        config=config,
        total_epochs=total_epochs,
        steps_per_epoch=steps_per_epoch,
        rank=rank
    )