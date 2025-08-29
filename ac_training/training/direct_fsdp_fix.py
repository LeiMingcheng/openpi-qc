"""
Direct FSDP Fix: 修复8卡FSDP内存爆炸问题
======================================

根据OpenPI标准实现和详细分析结果，核心问题是JAX FSDP在大模型上的底层失效。
本修复采用OpenPI的完全相同模式来解决内存爆炸问题：从53GB/GPU降到<8GB/GPU。

核心原理：
1. 使用jax.jit的out_shardings直接创建分片参数
2. 避免先创建完整参数再分片的模式（这会导致11.5x内存放大）
3. 完全模仿OpenPI的init_train_state实现

内存分析：
- 理论参数: π₀(12.4GB) + AdamW(24.7GB) + EMA(0GB) = 37GB
- 理论FSDP: 37GB ÷ 8 GPU = 4.6GB/GPU
- 实际失效: 53GB/GPU (11.5x放大)
- 修复目标: <8GB/GPU (可接受的分片效果)
"""

import logging
import jax
import jax.numpy as jnp
import dataclasses
from typing import Any, Tuple, Callable
from flax import nnx
import optax
import openpi.training.sharding as sharding
import openpi.training.optimizer as _optimizer
from openpi.training.utils import TrainState

logger = logging.getLogger(__name__)


def log_memory_usage(step_name: str):
    """记录GPU内存使用情况"""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            total_used = 0
            gpu_count = 0
            for line in result.stdout.strip().split('\n'):
                if ',' in line:
                    gpu_id, used, total = line.split(', ')
                    total_used += int(used)
                    gpu_count += 1
                    logger.info(f"  GPU {gpu_id}: {used}MB/{total}MB")
            
            if gpu_count > 0:
                avg_memory = total_used / gpu_count
                logger.info(f"🔍 {step_name}: 平均 {avg_memory:.0f}MB/GPU")
                return avg_memory
    except Exception as e:
        logger.warning(f"内存监控失败: {e}")
    return 0


def create_direct_fsdp_train_state(
    rl_config,
    mesh: jax.sharding.Mesh,
    rng: jax.Array,
    global_pi0_tx: optax.GradientTransformation = None,
    global_critic_tx: optax.GradientTransformation = None
) -> Tuple[Any, Any, Callable]:
    """
    使用OpenPI标准模式创建FSDP训练状态
    
    这是修复8卡FSDP内存爆炸的核心函数，完全采用OpenPI的init_train_state模式：
    1. 使用jax.eval_shape获取结构（无内存分配）
    2. 使用fsdp_sharding计算分片策略
    3. 使用jax.jit(out_shardings=...)直接创建分片参数
    """
    from agents.acrlpd_pi0_agent import create_acrlpd_pi0_agent_from_rl_config
    from training.acrlpd_train_state import ACRLPDTrainState, acrlpd_train_step
    
    logger.info("🚀 开始OpenPI标准FSDP初始化（修复内存爆炸）")
    log_memory_usage("FSDP初始化前")
    
    # **关键修复：使用全局优化器避免pytree元数据不匹配**
    if global_pi0_tx is None or global_critic_tx is None:
        logger.warning("警告：未提供全局优化器，将创建新实例（可能导致pytree问题）")
        pi0_tx = _optimizer.create_optimizer(rl_config.actor_optimizer, rl_config.get_effective_actor_lr_schedule())
        critic_tx = _optimizer.create_optimizer(rl_config.critic_optimizer, rl_config.get_effective_critic_lr_schedule())
    else:
        logger.info("✅ 使用全局优化器实例，确保pytree一致性")
        pi0_tx = global_pi0_tx
        critic_tx = global_critic_tx
    
    # Step 1: 定义初始化函数（完全模仿OpenPI）
    def init_fn(rng: jax.Array, partial_params: Any = None) -> ACRLPDTrainState:
        """
        初始化训练状态 - 这个函数会被JIT编译并直接输出分片参数
        
        关键点：当使用out_shardings时，这个函数的输出将直接以分片形式创建
        避免了先创建完整参数再分片的内存爆炸问题
        """
        # 创建agent（和之前一样）
        agent = create_acrlpd_pi0_agent_from_rl_config(rl_config, rng)
        
        # 如果有部分参数，合并进去（OpenPI模式）
        if partial_params is not None:
            # TODO: 实现参数合并逻辑，当前为None跳过
            pass
        
        # 获取参数和组件定义  
        pi0_params = nnx.state(agent.pi0_model)
        pi0_model_def = nnx.graphdef(agent.pi0_model)
        critic_params = nnx.state(agent.critic_networks)
        critic_model_def = nnx.graphdef(agent.critic_networks)
        
        # 关键修复：如果有预训练权重，合并进去
        if partial_params is not None:
            logger.info("🔄 合并预训练权重到π₀模型...")
            graphdef, state = nnx.split(agent.pi0_model)
            # 只合并π₀相关的权重
            pi0_weights = {k: v for k, v in partial_params.items() if 'pi0' in k or not any(x in k for x in ['critic', 'temperature'])}
            if pi0_weights:
                state.replace_by_pure_dict(pi0_weights)
                agent.pi0_model = nnx.merge(graphdef, state)
                pi0_params = nnx.state(agent.pi0_model)
                logger.info("✅ π₀预训练权重合并成功")
        
        # 创建优化器状态
        pi0_opt_state = pi0_tx.init(pi0_params)
        critic_opt_state = critic_tx.init(critic_params)
        
        # 温度模块（如果存在）
        temp_params = None
        temp_model_def = None
        temp_opt_state = None
        if hasattr(agent, 'temperature_module') and agent.temperature_module:
            temp_params = nnx.state(agent.temperature_module)
            temp_model_def = nnx.graphdef(agent.temperature_module)
            temp_tx = _optimizer.create_optimizer(rl_config.actor_optimizer, rl_config.get_effective_actor_lr_schedule())
            temp_opt_state = temp_tx.init(temp_params)
        else:
            temp_tx = None
        
        # 返回训练状态（关键：当用out_shardings时，这个返回值将直接分片创建）
        # 修复：确保EMA decay与其他创建点保持一致，包括use_ema检查
        use_ema = getattr(rl_config.acrlpd, 'use_ema', True)
        pi0_ema_decay_value = getattr(rl_config.acrlpd, 'pi0_ema_decay', 0.999) if use_ema else None
        return ACRLPDTrainState(
            step=0,
            pi0_params=pi0_params,
            pi0_model_def=pi0_model_def,
            pi0_opt_state=pi0_opt_state,
            pi0_tx=pi0_tx,
            critic_params=critic_params,
            critic_model_def=critic_model_def,
            critic_opt_state=critic_opt_state,
            critic_tx=critic_tx,
            pi0_ema_decay=pi0_ema_decay_value,  # 关键修复：添加缺失的EMA decay参数
            pi0_ema_params=pi0_params,  # EMA参数引用（无额外内存）
            temperature_params=temp_params,
            temperature_model_def=temp_model_def,
            temperature_opt_state=temp_opt_state,
            temperature_tx=temp_tx,
            config={}
        )
    
    logger.info("📐 Step 1: 使用eval_shape获取训练状态结构...")
    # Step 2: 获取结构（无内存分配，OpenPI标准模式）
    train_state_shape = jax.eval_shape(init_fn, rng, None)
    log_memory_usage("eval_shape后")
    
    logger.info("🎯 Step 2: 计算FSDP分片策略...")
    # Step 3: 计算FSDP分片策略（OpenPI标准模式）
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)
    
    logger.info("⚡ Step 3: JIT编译init函数与out_shardings...")
    # Step 4: 创建分片规格
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    
    # Step 5: 关键修复 - JIT编译时指定out_shardings
    # 这确保参数直接以分片形式创建，而不是先创建完整再分片
    init_fn_sharded = jax.jit(
        init_fn,
        donate_argnums=(1,),  # 如果有partial_params可以donate
        in_shardings=(replicated_sharding, replicated_sharding),  # RNG和partial_params都是replicated
        out_shardings=state_sharding  # 🔥 关键：输出直接分片创建！
    )
    
    logger.info("💫 Step 4: 创建分片训练状态（参数born sharded）...")
    # Step 6: 在mesh上下文中创建分片训练状态
    with sharding.set_mesh(mesh):
        train_state = init_fn_sharded(rng, None)
    
    # 等待完成并检查内存
    jax.block_until_ready(train_state)
    memory_after = log_memory_usage("FSDP创建后")
    
    # 记录FSDP内存使用情况
    logger.info(f"💾 FSDP内存使用: {memory_after:.0f}MB/GPU")
    
    # Step 7: 创建训练步骤函数
    def train_step_wrapper(train_state, batch, rng):
        """包装训练步骤，保持接口一致"""
        return acrlpd_train_step(train_state, batch, rng, {
            'critic_weight': getattr(rl_config.acrlpd, 'critic_weight', 1.0),
            'actor_weight': getattr(rl_config.acrlpd, 'actor_weight', 1.0),
            'bc_weight': getattr(rl_config.acrlpd, 'bc_loss_weight', 0.01),
            'alpha_weight': getattr(rl_config.acrlpd, 'alpha_weight', 1.0),
            'freeze_pi0_backbone': False,
            'target_update_tau': getattr(rl_config.acrlpd, 'target_update_tau', 0.005)
        })
    
    # 创建数据分片
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    
    # JIT编译训练步骤
    jit_train_step = jax.jit(
        train_step_wrapper,
        in_shardings=(state_sharding, data_sharding, replicated_sharding),
        out_shardings=(state_sharding, replicated_sharding),
        donate_argnums=()  # 避免donation问题
    )
    
    # 包装最终训练函数
    def fsdp_train_step(train_state, batch, rng):
        """FSDP训练步骤，在mesh上下文中执行"""
        with sharding.set_mesh(mesh):
            return jit_train_step(train_state, batch, rng)
    
    logger.info("🎉 OpenPI标准FSDP训练状态创建完成")
    
    return train_state, state_sharding, fsdp_train_step


def test_direct_fsdp_memory_usage():
    """
    测试direct FSDP修复效果
    
    这个测试创建一个简化的FSDP设置来验证内存使用是否正常
    """
    logger.info("🧪 测试Direct FSDP内存效果...")
    
    # 创建测试mesh
    mesh = sharding.make_mesh(8)
    
    # 测试用简单参数结构
    def create_test_params(rng):
        # 创建一个大参数数组模拟π₀参数
        return {
            'large_param': jnp.ones((4096, 4096), dtype=jnp.float32),  # 64MB
            'medium_param': jnp.ones((1024, 1024), dtype=jnp.float32),  # 4MB
            'small_param': jnp.ones((100, 100), dtype=jnp.float32),  # 0.04MB
        }
    
    # 获取结构
    param_shape = jax.eval_shape(create_test_params, jax.random.PRNGKey(0))
    
    # 计算分片策略
    param_sharding = sharding.fsdp_sharding(param_shape, mesh, log=True)
    
    # 用out_shardings创建
    create_sharded = jax.jit(
        create_test_params,
        in_shardings=(jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),),
        out_shardings=param_sharding
    )
    
    memory_before = log_memory_usage("测试前")
    
    with sharding.set_mesh(mesh):
        test_params = create_sharded(jax.random.PRNGKey(0))
    
    jax.block_until_ready(test_params)
    memory_after = log_memory_usage("测试后")
    
    memory_increase = memory_after - memory_before
    
    # 预期：68MB参数 / 8 GPU ≈ 8.5MB/GPU
    expected_per_gpu = 8.5
    
    if memory_increase < expected_per_gpu * 5:  # 允许5倍误差
        logger.info(f"✅ 测试通过: {memory_increase:.1f}MB/GPU (预期~{expected_per_gpu}MB)")
        return True
    else:
        logger.error(f"❌ 测试失败: {memory_increase:.1f}MB/GPU >> {expected_per_gpu}MB")
        return False


if __name__ == "__main__":
    """测试入口"""
    logging.basicConfig(level=logging.INFO)
    
    logger.info("=" * 60)
    logger.info("Direct FSDP修复测试")
    logger.info("=" * 60)
    
    test_direct_fsdp_memory_usage()