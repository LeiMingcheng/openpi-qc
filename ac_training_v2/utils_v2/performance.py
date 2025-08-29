"""
Performance Monitoring

轻量级性能监控工具，功能：
- 训练速度跟踪
- 内存使用监控  
- JIT编译时间统计
- 系统资源监控
"""

import logging
import time
import psutil
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import jax

logger = logging.getLogger(__name__)


@dataclass
class PerformanceStats:
    """性能统计数据结构"""
    # 训练速度
    steps_per_sec: float = 0.0
    avg_step_time: float = 0.0
    total_steps: int = 0
    
    # 内存使用
    memory_usage_mb: float = 0.0
    gpu_memory_usage_mb: float = 0.0
    
    # JIT编译
    compilation_time: float = 0.0
    compilation_count: int = 0
    
    # 系统资源
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0
    
    # 时间统计
    total_time: float = 0.0


class PerformanceMonitor:
    """
    性能监控器
    
    轻量级设计，最小化对训练性能的影响
    """
    
    def __init__(self, history_length: int = 100):
        """
        初始化性能监控器
        
        Args:
            history_length: 保持的历史记录长度
        """
        self.history_length = history_length
        
        # 性能历史记录
        self.step_times = deque(maxlen=history_length)
        self.memory_usage = deque(maxlen=history_length)
        self.gpu_memory_usage = deque(maxlen=history_length)
        
        # 时间追踪
        self.start_time = time.time()
        self.last_step_time = time.time()
        
        # JIT编译统计
        self.compilation_times = []
        self.total_compilation_time = 0.0
        
        # 步数统计
        self.total_steps = 0
        
        # 系统信息
        self.process = psutil.Process(os.getpid())
        
        logger.info(f"PerformanceMonitor初始化完成")
        logger.info(f"  历史记录长度: {history_length}")
        
        # 初始GPU内存监控（如果可用）
        self._gpu_available = self._check_gpu_availability()
        if self._gpu_available:
            logger.info("✅ GPU监控可用")
        else:
            logger.info("⚠️  GPU监控不可用")
    
    def record_step(self, step_time: Optional[float] = None):
        """
        记录训练步骤
        
        Args:
            step_time: 步骤时间，如果None则自动计算
        """
        
        current_time = time.time()
        
        # 计算步骤时间
        if step_time is None:
            step_time = current_time - self.last_step_time
        
        self.step_times.append(step_time)
        self.last_step_time = current_time
        self.total_steps += 1
        
        # 定期记录内存使用（避免过于频繁）
        if self.total_steps % 10 == 0:
            self._record_memory_usage()
    
    def record_compilation_time(self, compilation_time: float):
        """
        记录JIT编译时间
        
        Args:
            compilation_time: 编译时间（秒）
        """
        self.compilation_times.append(compilation_time)
        self.total_compilation_time += compilation_time
        
        logger.info(f"JIT编译完成: {compilation_time:.2f}s "
                   f"(总计: {self.total_compilation_time:.2f}s, "
                   f"次数: {len(self.compilation_times)})")
    
    def get_current_stats(self) -> PerformanceStats:
        """
        获取当前性能统计
        
        Returns:
            PerformanceStats: 性能统计数据
        """
        
        # 计算训练速度
        steps_per_sec = 0.0
        avg_step_time = 0.0
        if self.step_times:
            avg_step_time = sum(self.step_times) / len(self.step_times)
            steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0.0
        
        # 获取当前内存使用
        current_memory = self._get_memory_usage()
        current_gpu_memory = self._get_gpu_memory_usage()
        
        # 获取CPU使用率
        cpu_usage = self.process.cpu_percent()
        
        # 总训练时间
        total_time = time.time() - self.start_time
        
        return PerformanceStats(
            steps_per_sec=steps_per_sec,
            avg_step_time=avg_step_time,
            total_steps=self.total_steps,
            memory_usage_mb=current_memory,
            gpu_memory_usage_mb=current_gpu_memory,
            compilation_time=self.total_compilation_time,
            compilation_count=len(self.compilation_times),
            cpu_usage=cpu_usage,
            gpu_usage=0.0,  # GPU使用率需要额外工具
            total_time=total_time
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        获取性能摘要报告
        
        Returns:
            性能摘要字典
        """
        
        stats = self.get_current_stats()
        
        # 计算历史统计
        step_time_stats = {}
        if self.step_times:
            times = list(self.step_times)
            step_time_stats = {
                'min': min(times),
                'max': max(times),
                'mean': sum(times) / len(times),
                'recent_10': sum(times[-10:]) / min(len(times), 10),
                'count': len(times)
            }
        
        memory_stats = {}
        if self.memory_usage:
            mem_values = list(self.memory_usage)
            memory_stats = {
                'min_mb': min(mem_values),
                'max_mb': max(mem_values),
                'mean_mb': sum(mem_values) / len(mem_values),
                'current_mb': mem_values[-1] if mem_values else 0
            }
        
        return {
            'training_speed': {
                'steps_per_sec': stats.steps_per_sec,
                'total_steps': stats.total_steps,
                'total_time_sec': stats.total_time,
                'step_time_stats': step_time_stats
            },
            'memory': {
                'current_usage_mb': stats.memory_usage_mb,
                'gpu_usage_mb': stats.gpu_memory_usage_mb,
                'memory_stats': memory_stats
            },
            'compilation': {
                'total_time_sec': stats.compilation_time,
                'count': stats.compilation_count,
                'avg_time_sec': (stats.compilation_time / stats.compilation_count 
                               if stats.compilation_count > 0 else 0)
            },
            'system': {
                'cpu_usage_percent': stats.cpu_usage,
                'process_id': os.getpid()
            }
        }
    
    def _record_memory_usage(self):
        """记录内存使用情况"""
        memory_mb = self._get_memory_usage()
        gpu_memory_mb = self._get_gpu_memory_usage()
        
        self.memory_usage.append(memory_mb)
        self.gpu_memory_usage.append(gpu_memory_mb)
    
    def _get_memory_usage(self) -> float:
        """获取当前内存使用量（MB）"""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / (1024 * 1024)  # 转换为MB
        except:
            return 0.0
    
    def _get_gpu_memory_usage(self) -> float:
        """获取GPU内存使用量（MB）"""
        if not self._gpu_available:
            return 0.0
        
        try:
            # 使用JAX获取GPU内存使用情况
            if jax.default_backend() == 'gpu':
                # 这是一个简化的实现，实际可能需要更复杂的方法
                return 0.0  # 占位实现
            return 0.0
        except:
            return 0.0
    
    def _check_gpu_availability(self) -> bool:
        """检查GPU监控是否可用"""
        try:
            devices = jax.devices()
            return any(device.device_kind == 'gpu' for device in devices)
        except:
            return False
    
    def log_performance_report(self):
        """输出性能报告到日志"""
        summary = self.get_performance_summary()
        
        logger.info("=== 性能报告 ===")
        
        # 训练速度
        speed = summary['training_speed']
        logger.info(f"训练速度: {speed['steps_per_sec']:.2f} 步/秒")
        logger.info(f"总步数: {speed['total_steps']}")
        logger.info(f"总训练时间: {speed['total_time_sec']:.1f}秒")
        
        # 内存使用
        memory = summary['memory']
        logger.info(f"内存使用: {memory['current_usage_mb']:.1f} MB")
        if memory['gpu_usage_mb'] > 0:
            logger.info(f"GPU内存使用: {memory['gpu_usage_mb']:.1f} MB")
        
        # JIT编译
        compilation = summary['compilation']
        if compilation['count'] > 0:
            logger.info(f"JIT编译: {compilation['count']}次, "
                       f"总时间: {compilation['total_time_sec']:.1f}秒")
        
        logger.info("================")
    
    def reset(self):
        """重置监控器"""
        self.step_times.clear()
        self.memory_usage.clear()  
        self.gpu_memory_usage.clear()
        self.compilation_times.clear()
        
        self.start_time = time.time()
        self.last_step_time = time.time()
        self.total_compilation_time = 0.0
        self.total_steps = 0
        
        logger.info("PerformanceMonitor已重置")


class CompilationTimer:
    """JIT编译时间计时器（上下文管理器）"""
    
    def __init__(self, monitor: PerformanceMonitor, operation_name: str = "compilation"):
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"开始JIT编译: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            compilation_time = time.time() - self.start_time
            self.monitor.record_compilation_time(compilation_time)


def create_performance_monitor(history_length: int = 100) -> PerformanceMonitor:
    """
    工厂函数：创建性能监控器
    
    Args:
        history_length: 历史记录长度
        
    Returns:
        PerformanceMonitor实例
    """
    
    return PerformanceMonitor(history_length=history_length)