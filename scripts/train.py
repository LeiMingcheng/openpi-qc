import dataclasses
import functools
import logging
import platform
import time
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    try:
        images_to_log = [
            wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
            for i in range(min(5, len(next(iter(batch[0].images.values())))))
        ]
        wandb.log({"camera_views": images_to_log}, step=0)
    except Exception as e:
        logging.warning(f"Failed to log images: {e}")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")
    
    # 🔍 添加FSDP分片调试信息
    logging.info("=" * 60)
    logging.info("🔍 FSDP分片调试信息")
    logging.info("=" * 60)
    logging.info(f"Mesh: {mesh}")
    
    # 分析参数分片状态
    total_params_mb = 0
    sharded_params_mb = 0
    replicated_params_mb = 0
    
    def analyze_param_sharding(path, param):
        nonlocal total_params_mb, sharded_params_mb, replicated_params_mb
        if hasattr(param, 'sharding') and hasattr(param, 'shape'):
            size_mb = param.nbytes / (1024 * 1024)
            total_params_mb += size_mb
            
            if hasattr(param.sharding, 'spec'):
                spec = param.sharding.spec
                if spec == jax.sharding.PartitionSpec():
                    replicated_params_mb += size_mb
                    if size_mb > 10:  # 只记录大于10MB的参数
                        logging.info(f"📍 复制: {jax.tree_util.keystr(path)}: {param.shape} ({size_mb:.1f}MB)")
                else:
                    sharded_params_mb += size_mb
                    if size_mb > 10:  # 只记录大于10MB的参数
                        logging.info(f"✂️  分片: {jax.tree_util.keystr(path)}: {param.shape} -> {spec} ({size_mb:.1f}MB)")
    
    jax.tree_util.tree_map_with_path(analyze_param_sharding, train_state.params)
    
    logging.info("📊 参数分片统计:")
    logging.info(f"  总参数: {total_params_mb:.1f}MB")
    logging.info(f"  已分片: {sharded_params_mb:.1f}MB ({sharded_params_mb/total_params_mb*100:.1f}%)")
    logging.info(f"  复制型: {replicated_params_mb:.1f}MB ({replicated_params_mb/total_params_mb*100:.1f}%)")
    
    # 检查GPU内存分布
    import subprocess
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            logging.info("💾 GPU内存分布:")
            for line in result.stdout.strip().split('\n'):
                if ',' in line:
                    gpu_id, used, total = line.split(', ')
                    used_mb = int(used)
                    total_mb = int(total)
                    pct = used_mb / total_mb * 100
                    logging.info(f"  GPU {gpu_id}: {used}MB/{total}MB ({pct:.1f}%)")
    except:
        pass
    
    logging.info("=" * 60)

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    
    # 🎯 性能分析：时间统计初始化
    perf_stats = {
        'data_loading_times': [],
        'train_step_times': [],
        'logging_times': [],
        'total_iter_times': [],
        'jax_sync_times': []
    }
    
    logging.info("🔍 OpenPI性能分析：开始详细计时...")
    
    for step in pbar:
        iter_start_time = time.perf_counter()
        
        # === 训练步骤执行 ===
        train_step_start = time.perf_counter()
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        
        # JAX同步点 - 确保计算完成
        jax_sync_start = time.perf_counter()
        jax.block_until_ready((train_state, info))
        jax_sync_time = time.perf_counter() - jax_sync_start
        
        train_step_time = time.perf_counter() - train_step_start
        
        infos.append(info)
        
        # === 日志处理 ===
        if step % config.log_interval == 0:
            logging_start = time.perf_counter()
            
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            logging.info(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
            
            logging_time = time.perf_counter() - logging_start
            perf_stats['logging_times'].append(logging_time)
        else:
            logging_time = 0
        
        # === 数据加载 ===
        data_load_start = time.perf_counter()
        batch = next(data_iter)
        data_load_time = time.perf_counter() - data_load_start
        
        # === 性能统计收集 ===
        total_iter_time = time.perf_counter() - iter_start_time
        
        perf_stats['data_loading_times'].append(data_load_time)
        perf_stats['train_step_times'].append(train_step_time)
        perf_stats['total_iter_times'].append(total_iter_time)
        perf_stats['jax_sync_times'].append(jax_sync_time)
        
        # === 性能报告（每100步输出一次，第10步就开始）===
        if step % 100 == 0 or step == start_step:
            avg_data_load = np.mean(perf_stats['data_loading_times'][-100:])
            avg_train_step = np.mean(perf_stats['train_step_times'][-100:])
            avg_jax_sync = np.mean(perf_stats['jax_sync_times'][-100:])
            avg_total_iter = np.mean(perf_stats['total_iter_times'][-100:])
            avg_logging = np.mean([t for t in perf_stats['logging_times'] if t > 0][-10:]) if perf_stats['logging_times'] else 0
            
            logging.info(f"🔍 OpenPI性能分析 (Step {step}):")
            logging.info(f"  总迭代时间: {avg_total_iter*1000:.2f}ms")
            logging.info(f"  ├─ 训练步骤: {avg_train_step*1000:.2f}ms ({avg_train_step/avg_total_iter*100:.1f}%)")
            logging.info(f"  ├─ JAX同步:  {avg_jax_sync*1000:.2f}ms ({avg_jax_sync/avg_total_iter*100:.1f}%)")
            logging.info(f"  ├─ 数据加载: {avg_data_load*1000:.2f}ms ({avg_data_load/avg_total_iter*100:.1f}%)")
            logging.info(f"  └─ 日志处理: {avg_logging*1000:.2f}ms ({avg_logging/avg_total_iter*100:.1f}%)")
            logging.info(f"  每秒迭代数: {1.0/avg_total_iter:.2f} iter/s")

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    # === 最终性能报告 ===
    if perf_stats['total_iter_times']:
        logging.info("=" * 60)
        logging.info("🔍 OpenPI训练完成 - 最终性能报告")
        logging.info("=" * 60)
        
        total_samples = len(perf_stats['total_iter_times'])
        avg_data_load = np.mean(perf_stats['data_loading_times'])
        avg_train_step = np.mean(perf_stats['train_step_times'])
        avg_jax_sync = np.mean(perf_stats['jax_sync_times'])
        avg_total_iter = np.mean(perf_stats['total_iter_times'])
        avg_logging = np.mean([t for t in perf_stats['logging_times'] if t > 0]) if perf_stats['logging_times'] else 0
        
        std_total_iter = np.std(perf_stats['total_iter_times'])
        
        logging.info(f"样本数量: {total_samples} 次迭代")
        logging.info(f"平均迭代时间: {avg_total_iter*1000:.2f} ± {std_total_iter*1000:.2f}ms")
        logging.info(f"平均每秒迭代数: {1.0/avg_total_iter:.2f} iter/s")
        logging.info("")
        logging.info("时间分解:")
        logging.info(f"  训练步骤: {avg_train_step*1000:.2f}ms ({avg_train_step/avg_total_iter*100:.1f}%)")
        logging.info(f"  JAX同步:  {avg_jax_sync*1000:.2f}ms ({avg_jax_sync/avg_total_iter*100:.1f}%)")
        logging.info(f"  数据加载: {avg_data_load*1000:.2f}ms ({avg_data_load/avg_total_iter*100:.1f}%)")
        logging.info(f"  日志处理: {avg_logging*1000:.2f}ms ({avg_logging/avg_total_iter*100:.1f}%)")
        logging.info("=" * 60)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
