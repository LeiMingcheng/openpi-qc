"""
Simplified Metrics Logging

最小化的指标记录系统，优化点：
- 批量写入减少I/O开销
- 只记录核心指标
- 高效的数据结构
- 可选的wandb集成
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import json
import wandb

logger = logging.getLogger(__name__)


@dataclass
class MetricsConfig:
    """指标记录配置"""
    # 基础配置
    log_to_console: bool = True
    log_to_file: bool = True
    log_file_path: str = "training_metrics.log"
    
    # wandb配置
    use_wandb: bool = True
    
    # 性能优化配置
    buffer_size: int = 10               # 缓冲区大小（批量写入）
    history_length: int = 1000          # 保持的历史记录长度
    
    # 日志级别
    console_log_level: str = "INFO"     # "DEBUG", "INFO", "WARNING", "ERROR"


class MetricsLogger:
    """
    简化的指标记录器
    
    优化特点：
    - 批量I/O操作
    - 最小化内存使用
    - 高效的数据缓存
    """
    
    def __init__(self, config: MetricsConfig):
        """
        初始化指标记录器
        
        Args:
            config: 指标记录配置
        """
        self.config = config
        
        # 缓冲区
        self._buffer: List[Dict[str, Any]] = []
        self._last_flush_time = time.time()
        
        # 历史记录（用于计算移动平均等）
        self._history = defaultdict(lambda: deque(maxlen=config.history_length))
        
        # 文件句柄
        self._log_file = None
        if config.log_to_file:
            try:
                self._log_file = open(config.log_file_path, 'a')
            except Exception as e:
                logger.warning(f"无法打开日志文件 {config.log_file_path}: {e}")
        
        logger.info(f"MetricsLogger初始化完成")
        logger.info(f"  Console日志: {config.log_to_console}")
        logger.info(f"  文件日志: {config.log_to_file}")
        logger.info(f"  wandb: {config.use_wandb}")
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        记录指标
        
        Args:
            metrics: 指标字典
            step: 可选的步数
        """
        
        # 添加时间戳和步数
        log_entry = {
            'timestamp': time.time(),
            'step': step,
            **metrics
        }
        
        # 添加到缓冲区
        self._buffer.append(log_entry)
        
        # 更新历史记录
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self._history[key].append(value)
        
        # 检查是否需要刷新缓冲区
        if len(self._buffer) >= self.config.buffer_size:
            self.flush()
        
        # Console日志（实时）
        if self.config.log_to_console:
            self._log_to_console(log_entry)
        
        # wandb日志（实时）
        if self.config.use_wandb and step is not None:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                logger.debug(f"wandb日志失败: {e}")
    
    def flush(self):
        """刷新缓冲区到文件"""
        if not self._buffer:
            return
        
        # 文件日志（批量写入）
        if self._log_file:
            try:
                for entry in self._buffer:
                    json_line = json.dumps(entry) + '\n'
                    self._log_file.write(json_line)
                self._log_file.flush()
            except Exception as e:
                logger.warning(f"文件日志写入失败: {e}")
        
        # 清空缓冲区
        self._buffer.clear()
        self._last_flush_time = time.time()
    
    def _log_to_console(self, entry: Dict[str, Any]):
        """控制台日志输出"""
        step = entry.get('step', 'N/A')
        
        # 构建简洁的日志消息
        core_metrics = []
        
        # 提取核心指标
        if 'total_loss' in entry:
            core_metrics.append(f"Loss={entry['total_loss']:.4f}")
        
        if 'steps_per_sec' in entry:
            core_metrics.append(f"Speed={entry['steps_per_sec']:.2f}it/s")
        
        if 'critic_q_mean' in entry:
            core_metrics.append(f"Q={entry['critic_q_mean']:.3f}")
        
        # 组合消息
        message_parts = [f"Step {step}"]
        if core_metrics:
            message_parts.append(": " + ", ".join(core_metrics))
        
        message = "".join(message_parts)
        logger.info(message)
    
    def get_recent_average(self, metric_name: str, window: int = 100) -> Optional[float]:
        """
        获取最近N个值的平均值
        
        Args:
            metric_name: 指标名称
            window: 窗口大小
            
        Returns:
            平均值，如果没有数据返回None
        """
        if metric_name not in self._history:
            return None
        
        history = list(self._history[metric_name])
        if not history:
            return None
        
        recent_values = history[-window:] if len(history) >= window else history
        return sum(recent_values) / len(recent_values)
    
    def get_metric_summary(self, metric_name: str) -> Dict[str, float]:
        """
        获取指标的统计摘要
        
        Args:
            metric_name: 指标名称
            
        Returns:
            统计摘要字典
        """
        if metric_name not in self._history:
            return {}
        
        values = list(self._history[metric_name])
        if not values:
            return {}
        
        import numpy as np
        
        return {
            'count': len(values),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'recent_mean_100': self.get_recent_average(metric_name, 100) or 0.0,
            'recent_mean_10': self.get_recent_average(metric_name, 10) or 0.0
        }
    
    def close(self):
        """关闭日志记录器"""
        # 刷新最后的数据
        self.flush()
        
        # 关闭文件句柄
        if self._log_file:
            self._log_file.close()
            self._log_file = None
        
        logger.info("MetricsLogger已关闭")
    
    def __del__(self):
        """析构函数"""
        try:
            self.close()
        except:
            pass


def create_metrics_logger(use_wandb: bool = True,
                         log_file_path: str = "training_metrics.log") -> MetricsLogger:
    """
    工厂函数：创建指标记录器
    
    Args:
        use_wandb: 是否使用wandb
        log_file_path: 日志文件路径
        
    Returns:
        MetricsLogger实例
    """
    
    config = MetricsConfig(
        log_to_console=True,
        log_to_file=True,
        log_file_path=log_file_path,
        use_wandb=use_wandb,
        buffer_size=10,
        history_length=1000
    )
    
    return MetricsLogger(config)


class PerformanceTracker:
    """性能追踪器（轻量级）"""
    
    def __init__(self):
        self.step_times = deque(maxlen=100)  # 保持最近100步的时间
        self.start_time = time.time()
    
    def record_step_time(self, step_time: float):
        """记录步骤时间"""
        self.step_times.append(step_time)
    
    def get_stats(self) -> Dict[str, float]:
        """获取性能统计"""
        if not self.step_times:
            return {}
        
        import numpy as np
        
        times = list(self.step_times)
        return {
            'avg_step_time': float(np.mean(times)),
            'steps_per_sec': 1.0 / float(np.mean(times)),
            'total_time': time.time() - self.start_time,
            'recent_steps_count': len(times)
        }