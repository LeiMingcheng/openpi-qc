"""
详细的显存监控工具
基于FSDP调试报告的分析方法，提供组件级显存追踪
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional
import logging
import time

logger = logging.getLogger(__name__)

class GPUMemoryMonitor:
    """GPU内存使用监控器，支持组件级追踪"""
    
    def __init__(self):
        self.baseline_memory = self._get_gpu_memory_usage()
        self.checkpoints = {}
        
    def _get_gpu_memory_usage(self) -> Dict[str, float]:
        """获取GPU内存使用详情"""
        try:
            # JAX内存统计
            memory_info = {}
            
            # XLA内存统计
            for device in jax.devices():
                stats = device.memory_stats()
                device_id = f"GPU_{device.id}"
                memory_info[device_id] = {
                    'bytes_in_use': stats.get('bytes_in_use', 0),
                    'pool_bytes': stats.get('pool_bytes', 0), 
                    'peak_bytes_in_use': stats.get('peak_bytes_in_use', 0),
                    'bytes_reserved': stats.get('bytes_reserved', 0)
                }
                
            return memory_info
        except Exception as e:
            logger.warning(f"Failed to get GPU memory stats: {e}")
            return {}
    
    def _calculate_memory_size(self, pytree: Any, name: str = "") -> float:
        """计算pytree的实际内存占用（字节）"""
        try:
            total_bytes = 0
            count = 0
            
            def count_bytes(leaf):
                nonlocal total_bytes, count
                if hasattr(leaf, 'nbytes'):
                    # JAX数组
                    total_bytes += leaf.nbytes
                    count += 1
                elif hasattr(leaf, 'size') and hasattr(leaf, 'itemsize'):
                    # NumPy数组
                    total_bytes += leaf.size * leaf.itemsize
                    count += 1
                return leaf
                
            jax.tree_map(count_bytes, pytree)
            
            logger.debug(f"Memory analysis for {name}: {total_bytes/1e9:.3f}GB across {count} arrays")
            return total_bytes
            
        except Exception as e:
            logger.warning(f"Failed to calculate memory for {name}: {e}")
            return 0
    
    def analyze_train_state_memory(self, train_state: Any) -> Dict[str, Dict[str, float]]:
        """分析训练状态的详细内存组成"""
        analysis = {
            'π₀_parameters': {},
            'critic_parameters': {}, 
            'optimizer_states': {},
            'ema_parameters': {},
            'total_summary': {}
        }
        
        try:
            # π₀参数分析
            if hasattr(train_state, 'pi0_params') and train_state.pi0_params is not None:
                pi0_bytes = self._calculate_memory_size(train_state.pi0_params, "π₀_parameters")
                analysis['π₀_parameters'] = {
                    'bytes': pi0_bytes,
                    'GB': pi0_bytes / 1e9,
                    'percentage': pi0_bytes / (24.7e9) * 100  # 基于3.2B参数的理论值
                }
            
            # Critic参数分析
            if hasattr(train_state, 'critic_params') and train_state.critic_params is not None:
                critic_bytes = self._calculate_memory_size(train_state.critic_params, "critic_parameters")
                analysis['critic_parameters'] = {
                    'bytes': critic_bytes,
                    'GB': critic_bytes / 1e9,
                    'networks': len(train_state.critic_params) if isinstance(train_state.critic_params, (list, tuple)) else 1
                }
            
            # 优化器状态分析
            if hasattr(train_state, 'opt_state') and train_state.opt_state is not None:
                opt_bytes = self._calculate_memory_size(train_state.opt_state, "optimizer_states")
                analysis['optimizer_states'] = {
                    'bytes': opt_bytes,
                    'GB': opt_bytes / 1e9,
                    'components': self._analyze_optimizer_components(train_state.opt_state)
                }
            
            # EMA参数分析
            if hasattr(train_state, 'ema_params') and train_state.ema_params is not None:
                ema_bytes = self._calculate_memory_size(train_state.ema_params, "ema_parameters")
                analysis['ema_parameters'] = {
                    'bytes': ema_bytes,
                    'GB': ema_bytes / 1e9
                }
            
            # 总结分析
            total_bytes = sum([
                analysis['π₀_parameters'].get('bytes', 0),
                analysis['critic_parameters'].get('bytes', 0),
                analysis['optimizer_states'].get('bytes', 0),
                analysis['ema_parameters'].get('bytes', 0)
            ])
            
            analysis['total_summary'] = {
                'total_bytes': total_bytes,
                'total_GB': total_bytes / 1e9,
                'theoretical_GB': 6.2,  # 从FSDP报告
                'efficiency': (total_bytes / 1e9) / 6.2 * 100
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze train state memory: {e}")
        
        return analysis
    
    def _analyze_optimizer_components(self, opt_state: Any) -> Dict[str, float]:
        """分析优化器状态的组件分布"""
        components = {}
        try:
            # Adam优化器通常有mu(momentum)和nu(variance)
            if hasattr(opt_state, 'mu') and opt_state.mu is not None:
                components['momentum'] = self._calculate_memory_size(opt_state.mu, "momentum") / 1e9
            if hasattr(opt_state, 'nu') and opt_state.nu is not None:
                components['variance'] = self._calculate_memory_size(opt_state.nu, "variance") / 1e9
            
            # 处理嵌套的优化器状态
            if isinstance(opt_state, (list, tuple)):
                for i, state in enumerate(opt_state):
                    if hasattr(state, '__dict__'):
                        for attr_name, attr_value in state.__dict__.items():
                            if attr_value is not None:
                                size_gb = self._calculate_memory_size(attr_value, f"opt_component_{i}_{attr_name}") / 1e9
                                if size_gb > 0.001:  # 只记录大于1MB的组件
                                    components[f"component_{i}_{attr_name}"] = size_gb
                                    
        except Exception as e:
            logger.warning(f"Failed to analyze optimizer components: {e}")
        
        return components
    
    def checkpoint_memory(self, name: str, train_state: Optional[Any] = None):
        """创建内存使用检查点"""
        checkpoint = {
            'timestamp': time.time(),
            'gpu_memory': self._get_gpu_memory_usage(),
            'train_state_analysis': None
        }
        
        if train_state is not None:
            checkpoint['train_state_analysis'] = self.analyze_train_state_memory(train_state)
        
        self.checkpoints[name] = checkpoint
        
        # 打印内存摘要
        self._print_memory_summary(name, checkpoint)
        
    def _print_memory_summary(self, checkpoint_name: str, checkpoint: Dict[str, Any]):
        """打印内存使用摘要"""
        logger.info(f"🔍 内存检查点: {checkpoint_name}")
        
        # GPU内存
        for device_id, stats in checkpoint['gpu_memory'].items():
            bytes_in_use_gb = stats['bytes_in_use'] / 1e9
            pool_bytes_gb = stats['pool_bytes'] / 1e9
            peak_gb = stats['peak_bytes_in_use'] / 1e9
            
            logger.info(f"   {device_id}: 数据使用 {bytes_in_use_gb:.2f}GB | 内存池 {pool_bytes_gb:.2f}GB | 峰值 {peak_gb:.2f}GB")
        
        # 训练状态分析
        if checkpoint['train_state_analysis']:
            analysis = checkpoint['train_state_analysis']
            
            logger.info("   📊 组件内存分布:")
            for component, info in analysis.items():
                if component != 'total_summary' and info.get('GB', 0) > 0:
                    logger.info(f"      {component}: {info['GB']:.3f}GB")
            
            # 总结
            summary = analysis.get('total_summary', {})
            if summary:
                logger.info(f"   📈 总计: {summary.get('total_GB', 0):.2f}GB " +
                          f"(理论值: {summary.get('theoretical_GB', 0)}GB, " +
                          f"效率: {summary.get('efficiency', 0):.1f}%)")
    
    def compare_checkpoints(self, checkpoint1: str, checkpoint2: str):
        """比较两个检查点的内存变化"""
        if checkpoint1 not in self.checkpoints or checkpoint2 not in self.checkpoints:
            logger.error(f"Checkpoints {checkpoint1} or {checkpoint2} not found")
            return
        
        cp1 = self.checkpoints[checkpoint1]
        cp2 = self.checkpoints[checkpoint2]
        
        logger.info(f"🔄 内存变化对比: {checkpoint1} → {checkpoint2}")
        
        # 比较GPU内存
        for device_id in cp1['gpu_memory']:
            if device_id in cp2['gpu_memory']:
                usage1 = cp1['gpu_memory'][device_id]['bytes_in_use'] / 1e9
                usage2 = cp2['gpu_memory'][device_id]['bytes_in_use'] / 1e9
                delta = usage2 - usage1
                
                logger.info(f"   {device_id}: {usage1:.2f}GB → {usage2:.2f}GB (Δ{delta:+.2f}GB)")
    
    def get_memory_report(self) -> str:
        """生成详细的内存使用报告"""
        report = ["=" * 50, "GPU内存使用详细报告", "=" * 50]
        
        current_memory = self._get_gpu_memory_usage()
        
        for device_id, stats in current_memory.items():
            report.append(f"\n{device_id}:")
            report.append(f"  实际数据使用: {stats['bytes_in_use']/1e9:.2f}GB")
            report.append(f"  XLA内存池:   {stats['pool_bytes']/1e9:.2f}GB") 
            report.append(f"  峰值使用:    {stats['peak_bytes_in_use']/1e9:.2f}GB")
            report.append(f"  保留内存:    {stats.get('bytes_reserved', 0)/1e9:.2f}GB")
        
        report.append("\n基于FSDP调试报告的内存理解:")
        report.append("  • 实际数据使用 ~6GB: 模型参数 + 优化器状态")
        report.append("  • XLA内存池 ~60GB: 性能优化的预分配策略")
        report.append("  • 这是XLA大规模训练的正常行为")
        
        return "\n".join(report)


# 全局监控器实例
memory_monitor = GPUMemoryMonitor()


def log_memory_usage(step: int, train_state: Any = None, phase: str = "training"):
    """便捷的内存使用记录函数"""
    checkpoint_name = f"{phase}_step_{step}"
    memory_monitor.checkpoint_memory(checkpoint_name, train_state)


def log_gpu_memory(step_name: str = ""):
    """统一的 GPU 内存监控函数，客观输出数据"""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'], 
            capture_output=True, text=True
        )
        if result.returncode == 0:
            step_prefix = f"GPU Memory - {step_name}: " if step_name else "GPU Memory Status:"
            logger.info(step_prefix)
            total_used = 0
            total_capacity = 0
            usage_data = []
            max_usage_gpu = 0
            max_usage_mb = 0
            
            for line in result.stdout.strip().split('\n'):
                parts = line.split(', ')
                if len(parts) == 3:
                    gpu_id, used, total = parts
                    used_mb = int(used)
                    total_mb = int(total)
                    usage_pct = (used_mb / total_mb) * 100
                    total_used += used_mb
                    total_capacity += total_mb
                    usage_data.append(used_mb)
                    
                    if used_mb > max_usage_mb:
                        max_usage_mb = used_mb
                        max_usage_gpu = int(gpu_id)
                    
                    logger.info(f"  GPU {gpu_id}: {used}MB/{total}MB ({usage_pct:.1f}%)")
            
            # 计算统计数据
            overall_pct = (total_used / total_capacity) * 100 if total_capacity > 0 else 0
            avg_per_gpu = total_used / len(usage_data) if len(usage_data) > 0 else 0
            std_dev = (sum((x - avg_per_gpu) ** 2 for x in usage_data) / len(usage_data)) ** 0.5 if len(usage_data) > 0 else 0
            
            logger.info(f"  Total: {total_used}MB/{total_capacity}MB ({overall_pct:.1f}%)")
            logger.info(f"  Avg/GPU: {avg_per_gpu:.0f}MB, Max: {max_usage_mb:.0f}MB (GPU{max_usage_gpu}), StdDev: {std_dev:.0f}MB")
            
            # 返回数据用于进一步分析
            return {
                'usage_data': usage_data,
                'avg_per_gpu': avg_per_gpu,
                'max_usage': max_usage_mb,
                'max_usage_gpu': max_usage_gpu,
                'std_dev': std_dev,
                'overall_pct': overall_pct
            }
                    
    except Exception as e:
        logger.warning(f"Failed to get GPU memory info: {e}")
        return None


def enable_memory_monitoring():
    """启用内存监控（在训练循环开始时调用）"""
    logger.info("🔧 启用详细内存监控系统")
    memory_monitor.checkpoint_memory("training_start")
    
    # 打印系统信息
    logger.info(f"   GPU设备数: {len(jax.devices())}")
    logger.info(f"   JAX后端: {jax.lib.xla_bridge.get_backend().platform}")
    logger.info(memory_monitor.get_memory_report())