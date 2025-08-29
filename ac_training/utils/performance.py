"""
Performance optimization and memory management for ACRLPD data loading.

This module provides advanced optimization strategies for efficient processing
of large H5 datasets during ACRLPD training, including:

1. Memory-efficient data loading and caching
2. Parallel data processing and prefetching
3. JIT compilation optimization
4. GPU memory management
5. I/O optimization for H5 files
6. Profiling and monitoring tools

Key features:
- Adaptive caching strategies based on available memory
- Parallel episode loading and preprocessing
- JIT compilation cache management
- GPU memory optimization for large batches
- Real-time performance monitoring
- Automatic performance tuning
"""

import logging
import threading
import multiprocessing
import queue
import time
import gc
import psutil
import os
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
import h5py

# Removed unused imports from deleted modules

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    
    # Timing metrics (in seconds)
    episode_load_time: float = 0.0
    transform_time: float = 0.0
    batch_generation_time: float = 0.0
    device_transfer_time: float = 0.0
    
    # Memory metrics (in bytes)
    peak_memory_usage: int = 0
    current_memory_usage: int = 0
    cache_memory_usage: int = 0
    
    # Throughput metrics
    episodes_per_second: float = 0.0
    batches_per_second: float = 0.0
    samples_per_second: float = 0.0
    
    # Cache metrics
    cache_hit_rate: float = 0.0
    cache_eviction_rate: float = 0.0
    
    # Compilation metrics
    jit_compilation_count: int = 0
    jit_compilation_time: float = 0.0


class AdaptiveCacheManager:
    """
    Adaptive cache management for H5 episodes.
    
    This manager dynamically adjusts caching strategy based on:
    - Available system memory
    - Episode access patterns
    - Performance requirements
    """
    
    def __init__(
        self,
        initial_cache_size: int = 20,
        max_memory_usage_gb: float = 8.0,
        adaptation_interval: int = 100,  # Adapt every N accesses
        strategy: str = "lru_adaptive"  # "lru", "lfu", "lru_adaptive", "predictive"
    ):
        """
        Initialize AdaptiveCacheManager.
        
        Args:
            initial_cache_size: Initial number of episodes to cache
            max_memory_usage_gb: Maximum memory usage for caching
            adaptation_interval: How often to adapt cache strategy
            strategy: Caching strategy to use
        """
        self.initial_cache_size = initial_cache_size
        self.max_memory_bytes = int(max_memory_usage_gb * 1024**3)
        self.adaptation_interval = adaptation_interval
        self.strategy = strategy
        
        # Cache management
        self.cache = {}
        self.access_counts = {}
        self.access_times = {}
        self.cache_sizes = {}  # Track memory usage per episode
        self.access_history = []
        
        # Adaptive parameters
        self.current_cache_size = initial_cache_size
        self.access_counter = 0
        self.hit_count = 0
        self.miss_count = 0
        
        # Threading
        self.lock = threading.RLock()
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
    
    def get(self, episode_idx: int, loader_fn: Callable) -> Any:
        """
        Get episode data with adaptive caching.
        
        Args:
            episode_idx: Episode index
            loader_fn: Function to load episode if not cached
            
        Returns:
            Episode data
        """
        with self.lock:
            self.access_counter += 1
            current_time = time.time()
            
            # Update access tracking
            self.access_counts[episode_idx] = self.access_counts.get(episode_idx, 0) + 1
            self.access_times[episode_idx] = current_time
            self.access_history.append((episode_idx, current_time))
            
            # Check cache
            if episode_idx in self.cache:
                self.hit_count += 1
                self.metrics.cache_hit_rate = self.hit_count / self.access_counter
                return self.cache[episode_idx]
            
            # Cache miss - load episode
            self.miss_count += 1
            self.metrics.cache_hit_rate = self.hit_count / self.access_counter
            
            start_time = time.time()
            episode_data = loader_fn(episode_idx)
            load_time = time.time() - start_time
            self.metrics.episode_load_time = load_time
            
            # Estimate memory usage
            memory_usage = self._estimate_memory_usage(episode_data)
            self.cache_sizes[episode_idx] = memory_usage
            
            # Add to cache if there's space
            if self._can_cache(memory_usage):
                self.cache[episode_idx] = episode_data
                self._evict_if_necessary()
            
            # Adapt cache strategy periodically
            if self.access_counter % self.adaptation_interval == 0:
                self._adapt_cache_strategy()
            
            return episode_data
    
    def _estimate_memory_usage(self, episode_data: Any) -> int:
        """Estimate memory usage of episode data."""
        total_memory = 0
        
        def sum_memory(obj):
            nonlocal total_memory
            if isinstance(obj, np.ndarray):
                total_memory += obj.nbytes
            elif isinstance(obj, dict):
                for v in obj.values():
                    sum_memory(v)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    sum_memory(item)
        
        sum_memory(episode_data)
        return total_memory
    
    def _can_cache(self, additional_memory: int) -> bool:
        """Check if we can add more data to cache."""
        current_memory = sum(self.cache_sizes.values())
        return (current_memory + additional_memory) <= self.max_memory_bytes
    
    def _evict_if_necessary(self):
        """Evict episodes from cache if necessary."""
        current_memory = sum(self.cache_sizes.values())
        
        while current_memory > self.max_memory_bytes and self.cache:
            # Choose eviction strategy
            if self.strategy == "lru":
                episode_to_evict = min(self.cache.keys(), key=lambda x: self.access_times.get(x, 0))
            elif self.strategy == "lfu":
                episode_to_evict = min(self.cache.keys(), key=lambda x: self.access_counts.get(x, 0))
            else:  # lru_adaptive or predictive
                episode_to_evict = self._adaptive_eviction_choice()
            
            # Evict episode
            if episode_to_evict in self.cache:
                del self.cache[episode_to_evict]
                current_memory -= self.cache_sizes.get(episode_to_evict, 0)
                self.metrics.cache_eviction_rate += 1
    
    def _adaptive_eviction_choice(self) -> int:
        """Choose episode to evict using adaptive strategy."""
        # Combine LRU and LFU with access pattern analysis
        scores = {}
        current_time = time.time()
        
        for episode_idx in self.cache.keys():
            # Time since last access (LRU component)
            time_score = current_time - self.access_times.get(episode_idx, 0)
            
            # Inverse frequency (LFU component)
            freq_score = 1.0 / max(self.access_counts.get(episode_idx, 1), 1)
            
            # Memory usage (prefer evicting large episodes)
            memory_score = self.cache_sizes.get(episode_idx, 0) / (1024**2)  # Convert to MB
            
            # Combined score (higher = more likely to evict)
            scores[episode_idx] = time_score + freq_score + memory_score * 0.1
        
        return max(scores.keys(), key=lambda x: scores[x])
    
    def _adapt_cache_strategy(self):
        """Adapt cache strategy based on performance metrics."""
        # Analyze recent access patterns
        recent_history = self.access_history[-self.adaptation_interval:]
        unique_episodes = len(set(ep for ep, _ in recent_history))
        
        # Calculate temporal locality
        temporal_locality = len(recent_history) / max(unique_episodes, 1)
        
        # Adjust cache size based on hit rate and memory availability
        if self.metrics.cache_hit_rate < 0.5 and unique_episodes > self.current_cache_size:
            # Low hit rate and high unique episode count - increase cache
            available_memory = self.max_memory_bytes - sum(self.cache_sizes.values())
            if available_memory > 0:
                self.current_cache_size = min(
                    self.current_cache_size + 5,
                    unique_episodes,
                    self.initial_cache_size * 2
                )
        elif self.metrics.cache_hit_rate > 0.8 and temporal_locality > 2.0:
            # High hit rate and good temporal locality - can reduce cache size
            self.current_cache_size = max(
                self.current_cache_size - 2,
                self.initial_cache_size // 2
            )
        
        logger.debug(f"Adapted cache size to {self.current_cache_size}, hit rate: {self.metrics.cache_hit_rate:.3f}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                "cache_size": len(self.cache),
                "target_cache_size": self.current_cache_size,
                "memory_usage_mb": sum(self.cache_sizes.values()) / (1024**2),
                "hit_rate": self.metrics.cache_hit_rate,
                "total_accesses": self.access_counter,
                "unique_episodes_accessed": len(self.access_counts),
                "eviction_count": self.metrics.cache_eviction_rate,
            }


class ParallelDataProcessor:
    """
    Parallel data processing for H5 episodes.
    
    This processor uses multiple threads/processes to parallelize:
    - Episode loading from H5 files
    - Data transformation pipelines
    - Batch preprocessing
    """
    
    def __init__(
        self,
        num_workers: int = 4,
        prefetch_buffer_size: int = 10,
        use_processes: bool = False,  # Use threads by default for H5 file handling
        transformation_pipeline: Optional[Callable] = None
    ):
        """
        Initialize ParallelDataProcessor.
        
        Args:
            num_workers: Number of worker threads/processes
            prefetch_buffer_size: Size of prefetch buffer
            use_processes: Whether to use processes instead of threads
            transformation_pipeline: Optional transformation pipeline to apply
        """
        self.num_workers = num_workers
        self.prefetch_buffer_size = prefetch_buffer_size
        self.use_processes = use_processes
        self.transformation_pipeline = transformation_pipeline
        
        # Worker management
        self.executor = None
        self.prefetch_queue = queue.Queue(maxsize=prefetch_buffer_size)
        self.shutdown_event = threading.Event()
        
        # Performance tracking
        self.processed_episodes = 0
        self.total_processing_time = 0.0
        self.lock = threading.Lock()
    
    def start(self):
        """Start parallel processing workers."""
        if self.use_processes:
            self.executor = ProcessPoolExecutor(max_workers=self.num_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        
        logger.info(f"Started parallel data processor with {self.num_workers} {'processes' if self.use_processes else 'threads'}")
    
    def stop(self):
        """Stop parallel processing workers."""
        if self.executor:
            self.shutdown_event.set()
            self.executor.shutdown(wait=True)
            logger.info("Stopped parallel data processor")
    
    def process_episode_async(
        self,
        episode_loader: Callable,
        episode_idx: int,
        transform_params: Optional[Dict] = None
    ) -> Any:
        """
        Process episode asynchronously.
        
        Args:
            episode_loader: Function to load episode
            episode_idx: Episode index
            transform_params: Optional transformation parameters
            
        Returns:
            Future object for async result
        """
        if not self.executor:
            self.start()
        
        return self.executor.submit(
            self._process_episode_worker,
            episode_loader,
            episode_idx,
            transform_params or {}
        )
    
    def _process_episode_worker(
        self,
        episode_loader: Callable,
        episode_idx: int,
        transform_params: Dict
    ) -> Tuple[int, Any, float]:
        """Worker function for processing episodes."""
        start_time = time.time()
        
        try:
            # Load episode
            episode_data = episode_loader(episode_idx)
            
            # Apply transformation pipeline if provided
            if self.transformation_pipeline:
                episode_data = self.transformation_pipeline(episode_data, **transform_params)
            
            processing_time = time.time() - start_time
            
            # Update statistics
            with self.lock:
                self.processed_episodes += 1
                self.total_processing_time += processing_time
            
            return episode_idx, episode_data, processing_time
            
        except Exception as e:
            logger.error(f"Error processing episode {episode_idx}: {e}")
            raise
    
    def prefetch_episodes(
        self,
        episode_loader: Callable,
        episode_indices: List[int],
        transform_params: Optional[Dict] = None
    ):
        """
        Prefetch episodes in the background.
        
        Args:
            episode_loader: Function to load episodes
            episode_indices: List of episode indices to prefetch
            transform_params: Optional transformation parameters
        """
        def prefetch_worker():
            futures = []
            
            for episode_idx in episode_indices:
                if self.shutdown_event.is_set():
                    break
                
                future = self.process_episode_async(episode_loader, episode_idx, transform_params)
                futures.append(future)
                
                # Add completed futures to queue
                completed_futures = [f for f in futures if f.done()]
                for future in completed_futures:
                    try:
                        result = future.result()
                        if not self.prefetch_queue.full():
                            self.prefetch_queue.put(result)
                    except Exception as e:
                        logger.error(f"Prefetch error: {e}")
                    futures.remove(future)
        
        # Start prefetch thread
        prefetch_thread = threading.Thread(target=prefetch_worker)
        prefetch_thread.daemon = True
        prefetch_thread.start()
    
    def get_prefetched_episode(self, timeout: float = 1.0) -> Optional[Tuple[int, Any, float]]:
        """Get a prefetched episode from the queue."""
        try:
            return self.prefetch_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_stats(self) -> Dict[str, float]:
        """Get processing statistics."""
        with self.lock:
            avg_processing_time = (
                self.total_processing_time / max(self.processed_episodes, 1)
            )
            return {
                "processed_episodes": self.processed_episodes,
                "avg_processing_time": avg_processing_time,
                "episodes_per_second": self.processed_episodes / max(self.total_processing_time, 0.001),
                "total_processing_time": self.total_processing_time,
            }


class JITOptimizer:
    """
    JIT compilation optimization manager.
    
    This optimizer manages JAX JIT compilation to minimize compilation overhead
    and maximize runtime performance.
    """
    
    def __init__(self):
        """Initialize JIT optimizer."""
        self.compilation_cache = {}
        self.compilation_stats = {}
        self.warm_up_completed = set()
    
    def warm_up_functions(self, functions: Dict[str, Callable], sample_inputs: Dict[str, Any]):
        """
        Warm up JIT functions with sample inputs.
        
        Args:
            functions: Dictionary of function name -> function
            sample_inputs: Dictionary of function name -> sample input
        """
        logger.info("Warming up JIT functions...")
        
        for func_name, func in functions.items():
            if func_name in sample_inputs:
                start_time = time.time()
                
                try:
                    # Trigger compilation
                    sample_input = sample_inputs[func_name]
                    if isinstance(sample_input, (tuple, list)):
                        _ = func(*sample_input)
                    elif isinstance(sample_input, dict):
                        _ = func(**sample_input)
                    else:
                        _ = func(sample_input)
                    
                    compilation_time = time.time() - start_time
                    self.compilation_stats[func_name] = compilation_time
                    self.warm_up_completed.add(func_name)
                    
                    logger.debug(f"Warmed up {func_name} in {compilation_time:.3f}s")
                    
                except Exception as e:
                    logger.warning(f"Failed to warm up {func_name}: {e}")
        
        total_warmup_time = sum(self.compilation_stats.values())
        logger.info(f"JIT warm-up completed in {total_warmup_time:.3f}s for {len(self.warm_up_completed)} functions")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get JIT compilation statistics."""
        return {
            "warmed_up_functions": len(self.warm_up_completed),
            "total_compilation_time": sum(self.compilation_stats.values()),
            "compilation_times": self.compilation_stats.copy(),
        }


class GPUMemoryOptimizer:
    """
    GPU memory optimization for efficient batch processing.
    
    This optimizer manages GPU memory allocation and deallocation to prevent
    out-of-memory errors during training.
    """
    
    def __init__(self, reserved_memory_fraction: float = 0.1):
        """
        Initialize GPU memory optimizer.
        
        Args:
            reserved_memory_fraction: Fraction of GPU memory to keep reserved
        """
        self.reserved_memory_fraction = reserved_memory_fraction
        self.memory_pools = {}
        self.peak_memory_usage = 0
        
        # Set JAX memory preallocation
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(1.0 - reserved_memory_fraction))
    
    def optimize_batch_size(
        self,
        model_fn: Callable,
        sample_batch: Dict[str, Array],
        max_batch_size: int = 512,
        min_batch_size: int = 8
    ) -> int:
        """
        Find optimal batch size that fits in GPU memory.
        
        Args:
            model_fn: Model function to test
            sample_batch: Sample batch for testing
            max_batch_size: Maximum batch size to test
            min_batch_size: Minimum batch size to use
            
        Returns:
            Optimal batch size
        """
        logger.info("Optimizing batch size for GPU memory...")
        
        # Binary search for optimal batch size
        low, high = min_batch_size, max_batch_size
        optimal_batch_size = min_batch_size
        
        while low <= high:
            test_batch_size = (low + high) // 2
            
            # Create test batch
            test_batch = {}
            for key, value in sample_batch.items():
                if isinstance(value, jnp.ndarray):
                    # Replicate to test batch size
                    test_shape = (test_batch_size,) + value.shape[1:]
                    test_batch[key] = jnp.zeros(test_shape, dtype=value.dtype)
                else:
                    test_batch[key] = value
            
            try:
                # Test memory allocation
                _ = model_fn(test_batch)
                optimal_batch_size = test_batch_size
                low = test_batch_size + 1
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    high = test_batch_size - 1
                else:
                    raise
        
        logger.info(f"Optimal batch size: {optimal_batch_size}")
        return optimal_batch_size
    
    def clear_cache(self):
        """Clear GPU memory cache."""
        # Trigger garbage collection
        gc.collect()
        
        # Clear JAX cache if available
        if hasattr(jax, 'clear_backends'):
            jax.clear_backends()
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get GPU memory information."""
        try:
            # This would require proper GPU memory monitoring
            # For now, return placeholder info
            return {
                "reserved_fraction": self.reserved_memory_fraction,
                "peak_usage_mb": self.peak_memory_usage / (1024**2),
            }
        except Exception:
            return {"error": "GPU memory info not available"}


class PerformanceOptimizer:
    """
    Main performance optimization coordinator.
    
    This class coordinates all optimization strategies to maximize performance
    of the ACRLPD data loading pipeline.
    """
    
    def __init__(
        self,
        cache_manager: Optional[AdaptiveCacheManager] = None,
        parallel_processor: Optional[ParallelDataProcessor] = None,
        jit_optimizer: Optional[JITOptimizer] = None,
        gpu_optimizer: Optional[GPUMemoryOptimizer] = None,
        monitoring_interval: float = 30.0  # Monitor every 30 seconds
    ):
        """
        Initialize PerformanceOptimizer.
        
        Args:
            cache_manager: Adaptive cache manager
            parallel_processor: Parallel data processor
            jit_optimizer: JIT optimizer
            gpu_optimizer: GPU memory optimizer
            monitoring_interval: Performance monitoring interval
        """
        self.cache_manager = cache_manager or AdaptiveCacheManager()
        self.parallel_processor = parallel_processor or ParallelDataProcessor()
        self.jit_optimizer = jit_optimizer or JITOptimizer()
        self.gpu_optimizer = gpu_optimizer or GPUMemoryOptimizer()
        self.monitoring_interval = monitoring_interval
        
        # Performance monitoring
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.performance_history = []
        
        # System monitoring
        self.process = psutil.Process()
    
    def optimize_dataloader(self, dataloader) -> object:
        """
        Apply optimizations to ACRLPD dataloader.
        
        Args:
            dataloader: ACRLPD dataloader to optimize
            
        Returns:
            Optimized dataloader
        """
        logger.info("Applying performance optimizations to dataloader...")
        
        # Replace dataset cache manager with adaptive one
        if hasattr(dataloader.dataset, 'h5_reader'):
            original_load_episode = dataloader.dataset.h5_reader.load_episode
            
            def optimized_load_episode(episode_idx: int):
                return self.cache_manager.get(episode_idx, original_load_episode)
            
            dataloader.dataset.h5_reader.load_episode = optimized_load_episode
        
        # Start parallel processing
        self.parallel_processor.start()
        
        # Start performance monitoring
        self.start_monitoring()
        
        logger.info("Performance optimizations applied")
        return dataloader
    
    def warm_up_training(
        self,
        dataloader,
        model_functions: Dict[str, Callable],
        num_warmup_batches: int = 5
    ):
        """
        Warm up training with sample batches.
        
        Args:
            dataloader: ACRLPD dataloader
            model_functions: Dictionary of model functions to warm up
            num_warmup_batches: Number of batches for warm-up
        """
        logger.info("Starting training warm-up...")
        
        # Generate sample inputs for JIT warm-up
        sample_batch = dataloader.sample_single_step_batch()
        sample_inputs = {"batch_forward": sample_batch}
        
        # JIT warm-up
        if model_functions:
            self.jit_optimizer.warm_up_functions(model_functions, sample_inputs)
        
        # Batch processing warm-up
        start_time = time.time()
        for i in range(num_warmup_batches):
            batch = dataloader.sample_batch()
            logger.debug(f"Warm-up batch {i+1}/{num_warmup_batches} generated")
        
        warmup_time = time.time() - start_time
        avg_batch_time = warmup_time / num_warmup_batches
        
        logger.info(f"Training warm-up completed in {warmup_time:.3f}s (avg {avg_batch_time*1000:.1f}ms/batch)")
    
    def start_monitoring(self):
        """Start performance monitoring thread."""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.stop_monitoring.clear()
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            logger.info("Started performance monitoring")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.stop_monitoring.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Stopped performance monitoring")
    
    def _monitoring_loop(self):
        """Performance monitoring loop."""
        while not self.stop_monitoring.wait(self.monitoring_interval):
            try:
                # Collect performance metrics
                metrics = self._collect_metrics()
                self.performance_history.append(metrics)
                
                # Log performance summary
                self._log_performance_summary(metrics)
                
                # Keep history limited
                if len(self.performance_history) > 100:
                    self.performance_history = self.performance_history[-50:]
                    
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive performance metrics."""
        current_time = time.time()
        
        # System metrics
        cpu_usage = self.process.cpu_percent()
        memory_info = self.process.memory_info()
        memory_usage_mb = memory_info.rss / (1024**2)
        
        # Component metrics
        cache_stats = self.cache_manager.get_stats()
        parallel_stats = self.parallel_processor.get_stats()
        jit_stats = self.jit_optimizer.get_stats()
        gpu_stats = self.gpu_optimizer.get_memory_info()
        
        return {
            "timestamp": current_time,
            "system": {
                "cpu_usage_percent": cpu_usage,
                "memory_usage_mb": memory_usage_mb,
            },
            "cache": cache_stats,
            "parallel": parallel_stats,
            "jit": jit_stats,
            "gpu": gpu_stats,
        }
    
    def _log_performance_summary(self, metrics: Dict[str, Any]):
        """Log performance summary."""
        cache_hit_rate = metrics["cache"]["hit_rate"]
        memory_usage = metrics["system"]["memory_usage_mb"]
        eps = metrics["parallel"]["episodes_per_second"]
        
        logger.info(
            f"Performance: Cache hit rate {cache_hit_rate:.1%}, "
            f"Memory usage {memory_usage:.1f}MB, "
            f"Episodes/sec {eps:.1f}"
        )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        if not self.performance_history:
            return {"error": "No performance data available"}
        
        latest_metrics = self.performance_history[-1]
        
        # Calculate averages over recent history
        recent_history = self.performance_history[-10:]
        avg_cache_hit_rate = np.mean([m["cache"]["hit_rate"] for m in recent_history])
        avg_memory_usage = np.mean([m["system"]["memory_usage_mb"] for m in recent_history])
        avg_eps = np.mean([m["parallel"]["episodes_per_second"] for m in recent_history])
        
        return {
            "latest_metrics": latest_metrics,
            "averages": {
                "cache_hit_rate": avg_cache_hit_rate,
                "memory_usage_mb": avg_memory_usage,
                "episodes_per_second": avg_eps,
            },
            "history_length": len(self.performance_history),
            "recommendations": self._generate_recommendations(recent_history),
        }
    
    def _generate_recommendations(self, recent_history: List[Dict]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        if not recent_history:
            return recommendations
        
        # Analyze cache performance
        avg_hit_rate = np.mean([m["cache"]["hit_rate"] for m in recent_history])
        if avg_hit_rate < 0.5:
            recommendations.append("Consider increasing cache size for better hit rate")
        
        # Analyze memory usage  
        avg_memory = np.mean([m["system"]["memory_usage_mb"] for m in recent_history])
        if avg_memory > 8000:  # 8GB
            recommendations.append("High memory usage detected - consider reducing batch size")
        
        # Analyze processing performance
        avg_eps = np.mean([m["parallel"]["episodes_per_second"] for m in recent_history])
        if avg_eps < 1.0:
            recommendations.append("Low episode processing rate - consider increasing parallel workers")
        
        return recommendations
    
    def shutdown(self):
        """Shutdown performance optimizer."""
        logger.info("Shutting down performance optimizer...")
        
        self.stop_monitoring()
        self.parallel_processor.stop()
        
        logger.info("Performance optimizer shutdown complete")

if __name__ == "__main__":
    # Test performance optimization components
    print("Testing performance optimization components...")
    
    # Test adaptive cache manager
    cache_manager = AdaptiveCacheManager(initial_cache_size=5, max_memory_usage_gb=1.0)
    
    def dummy_loader(episode_idx):
        return {"data": np.random.randn(100, 14), "episode_idx": episode_idx}
    
    # Test cache performance
    start_time = time.time()
    for i in range(20):
        episode_idx = i % 8  # Some repetition for cache hits
        data = cache_manager.get(episode_idx, dummy_loader)
    cache_time = time.time() - start_time
    
    cache_stats = cache_manager.get_stats()
    print(f"Cache test completed in {cache_time:.3f}s")
    print(f"Cache stats: {cache_stats}")
    
    # Test parallel processor
    parallel_processor = ParallelDataProcessor(num_workers=2)
    parallel_processor.start()
    
    futures = []
    start_time = time.time()
    for i in range(10):
        future = parallel_processor.process_episode_async(dummy_loader, i)
        futures.append(future)
    
    # Wait for completion
    results = [future.result() for future in futures]
    parallel_time = time.time() - start_time
    
    parallel_stats = parallel_processor.get_stats()
    print(f"Parallel processing test completed in {parallel_time:.3f}s")
    print(f"Parallel stats: {parallel_stats}")
    
    parallel_processor.stop()
    
    # Test JIT optimizer
    jit_optimizer = JITOptimizer()
    
    @jax.jit
    def test_function(x):
        return jnp.sum(x**2)
    
    sample_input = jnp.ones((100, 14))
    jit_optimizer.warm_up_functions({"test_func": test_function}, {"test_func": sample_input})
    
    jit_stats = jit_optimizer.get_stats()
    print(f"JIT optimization stats: {jit_stats}")
    
    print("Performance optimization testing completed!")