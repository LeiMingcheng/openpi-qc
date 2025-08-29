# ACRLPD + π₀ Complete Training Framework

This directory contains the complete ACRLPD (Action-Chunked Reinforcement Learning with Prior Data) integration with π₀ models for robotic manipulation tasks.

## Overview

The framework provides end-to-end training infrastructure combining Q-chunking reinforcement learning with π₀ diffusion models. It includes data loading, agent implementation, training loops, and comprehensive evaluation systems.

## Architecture

### Complete Training Pipeline

```
H5 Robot Data → Data Loading → ACRLPD+π₀ Agent → Training Loop → Trained Policy
      ↓              ↓              ↓               ↓             ↓
  H5DatasetReader  QChunkingDataset  ACRLPDPi0Agent  ACRLPDTrainer  Checkpoints
  CompositePipeline ACRLPDDataLoader  CriticNetworks  Evaluation    WandB Logs
  Performance Opts  JAX Batching     Joint Losses    Monitoring    Best Models
```

### Core Components

1. **Data Loading System** (`data_loader.py`, `qc_dataset.py`, `acrlpd_dataloader.py`)
   - Efficient H5 dataset processing with caching
   - π₀-compatible multi-modal observation formatting  
   - Q-chunking action sequence generation
   - Device-aware JAX array placement

2. **Agent System** (`agents/`)
   - **ACRLPDPi0Agent**: Main agent integrating π₀ as direct policy
   - **CriticNetworks**: 10-network ensemble for robust Q-estimation
   - **Joint Loss Functions**: Combined RL + BC + diffusion training
   - **Best-of-N Sampling**: Q-guided action sequence selection

3. **Training System** (`training/`)
   - **ACRLPDTrainer**: Complete offline + online training pipeline
   - **Checkpoint Management**: Model persistence and recovery
   - **Evaluation Framework**: Comprehensive performance assessment
   - **WandB Integration**: Experiment tracking and visualization

4. **Configuration & Scripts** (`config.py`, `scripts/`)
   - Platform-specific configurations (DROID, ALOHA, Libero)
   - Command-line training interface
   - Hyperparameter sweep generation
   - Debug and profiling tools

## Quick Start

### Installation

```bash
# Install dependencies (from project root)
uv sync
uv pip install -e .

# Set environment for training
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
```

### Basic Training

```bash
# Train on DROID dataset
python ac_training/scripts/train_acrlpd_pi0.py \
    --data_dir /path/to/droid/data \
    --config droid \
    --offline_steps 1000000 \
    --experiment_name "droid_acrlpd_baseline"

# Train on ALOHA with custom parameters
python ac_training/scripts/train_acrlpd_pi0.py \
    --data_dir /path/to/aloha/data \
    --config aloha \
    --horizon_length 20 \
    --bc_alpha 0.001 \
    --best_of_n_samples 32 \
    --experiment_name "aloha_long_horizon"

# Resume from checkpoint
python ac_training/scripts/train_acrlpd_pi0.py \
    --data_dir /path/to/data \
    --config libero \
    --resume /path/to/checkpoint.pkl
```

### Python API Usage

```python
from ac_training.agents import ACRLPDPi0Agent, get_droid_config
from ac_training.training import ACRLPDTrainer, TrainingConfig
from ac_training import create_acrlpd_dataloader
import jax

# Create configuration
config = get_droid_config()
config = config.replace(
    horizon_length=10,
    bc_alpha=0.01,
    batch_size=128
)

# Create data loader
dataloader = create_acrlpd_dataloader(
    data_dir="/path/to/data",
    batch_size=config.batch_size,
    horizon_length=config.horizon_length
)

# Create agent
rng = jax.random.PRNGKey(42)
agent = ACRLPDPi0Agent(config, nnx.Rngs(rng))

# Setup training
training_config = TrainingConfig(
    data_dir="/path/to/data",
    experiment_name="my_experiment",
    offline_steps=500000,
    eval_frequency=10000
)

trainer = ACRLPDTrainer(agent, dataloader, training_config)

# Train
trained_agent = trainer.train()
```

## Algorithm Details

### ACRLPD + π₀ Integration

**Core Idea**: Use π₀ directly as the policy network in ACRLPD, leveraging its multi-modal understanding and diffusion-based action generation while adding Q-learning for sample efficiency.

**Key Innovation**: 
- π₀ generates action sequences: `π₀(observation) → [a_t, a_{t+1}, ..., a_{t+H-1}]`
- Critic ensemble evaluates sequences: `Q(observation, action_sequence) → value`
- Best-of-N sampling: Generate N candidates, select highest Q-value sequence
- Joint training: `L_total = L_critic + λ_bc * L_bc + λ_α * L_α`

### Training Phases

1. **Offline Pretraining**
   - Train on robotic demonstration datasets
   - Strong BC regularization to preserve π₀'s behavior
   - Build Q-functions for action sequence evaluation

2. **Online Fine-tuning** (Optional)
   - Environment interaction with Best-of-N exploration
   - Adaptive temperature control for exploration
   - Early stopping based on evaluation performance

### Multi-Modal Processing

The framework handles π₀'s multi-modal inputs:
```python
observation = {
    'images': {
        'base_0_rgb': [B, 224, 224, 3],      # Main camera
        'left_wrist_0_rgb': [B, 224, 224, 3], # Left wrist camera  
        'right_wrist_0_rgb': [B, 224, 224, 3] # Right wrist camera
    },
    'state': [B, state_dim],                  # Robot proprioception
    'tokenized_prompt': [B, max_len],         # Language instruction
    'tokenized_prompt_mask': [B, max_len]     # Attention mask
}
```

## Configuration Presets

### DROID Configuration
Optimized for large-scale manipulation dataset:
```python
config = get_droid_config()
# horizon_length=10, bc_alpha=0.01, batch_size=64
# offline_steps=2000000, best_of_n_samples=16
```

### ALOHA Configuration  
Optimized for bimanual manipulation:
```python
config = get_aloha_config()
# horizon_length=20, bc_alpha=0.001, freeze_pi0_backbone=True
# batch_size=128, best_of_n_samples=32
```

### Libero Configuration
Optimized for simulation benchmarks:
```python
config = get_libero_config()  
# horizon_length=5, bc_alpha=0.1, batch_size=256
# best_of_n_samples=64, mixed_precision=False
```

## Advanced Features

### Best-of-N Sampling

The agent uses Q-guided sampling for improved performance:

```python
# Sample N action candidates from π₀
candidates = []
for i in range(N):
    actions = pi0_model.sample_actions(rng_i, observation)
    candidates.append(actions)

# Evaluate with critic ensemble
q_values = []
for actions in candidates:
    q_val = critic_networks(observation, actions)
    q_values.append(q_val.mean())  # or min() for conservative

# Select best sequence
best_idx = jnp.argmax(q_values)
best_actions = candidates[best_idx]
```

### Adaptive Temperature Control

Automatic entropy regularization:

```python
# Estimate π₀ policy entropy
entropy = estimate_entropy(pi0_model, observations)

# Adaptive temperature loss
target_entropy = -0.5 * action_dim
alpha_loss = alpha * (entropy - target_entropy)

# Update temperature parameter
alpha = alpha_optimizer.update(alpha, alpha_loss)
```

### Bootstrap Handling

Proper Q-learning across episode boundaries:

```python
# Compute n-step bootstrap targets
discount_factor = gamma ** horizon_length
target_q = rewards + discount_factor * next_q * masks

# Handle episode boundaries
masks = handle_episode_boundaries(batch, horizon_length)
```

## Performance Optimization

### Memory Management

```python
# Adaptive caching for large datasets
memory_manager = MemoryManager(max_memory_gb=16.0)
dataloader = create_acrlpd_dataloader(
    memory_manager=memory_manager,
    cache_strategy="adaptive"
)
```

### Multi-GPU Training

```python
# Automatic device sharding
device_manager = DeviceManager(strategy="data_parallel")
dataloader = create_acrlpd_dataloader(
    device_placement="gpu",
    device_manager=device_manager
)

# Sharded training step
sharded_batch = dataloader.sample_batch()
agent, loss_info = agent.train_step(sharded_batch, rng)
```

### Mixed Precision Training

```python
config = ACRLPDPi0Config(
    mixed_precision=True,
    gradient_clip=1.0,
    optimizer_type="adamw"
)
```

## File Structure

```
ac_training/
├── agents/                         # Core agent implementation
│   ├── __init__.py                 # Agent module exports
│   ├── acrlpd_pi0_agent.py        # Main ACRLPD+π₀ agent class
│   ├── critic_networks.py         # Q-network ensemble implementation
│   └── loss_functions.py          # Joint loss computation
├── training/                       # Training pipeline
│   ├── __init__.py                 # Training module exports
│   └── training_loop.py            # Complete training system
├── scripts/                        # Command-line interfaces
│   ├── train_acrlpd_pi0.py        # Main training script
│   └── generate_config.py         # Configuration generator
├── data_loader.py                  # H5 dataset reader
├── transforms.py                   # Data transformation pipeline
├── qc_dataset.py                   # QC Dataset interface adapter
├── acrlpd_dataloader.py           # ACRLPD data loading integration
├── batching.py                     # JAX batching and mask handling
├── config.py                       # Configuration management
├── performance.py                  # Performance optimization
├── test_data_loading.py           # Comprehensive test suite
└── README.md                       # This documentation
```

## Testing & Validation

### Unit Tests

```bash
# Test data loading
python -m pytest ac_training/test_data_loading.py -v

# Test agent components  
python -m pytest ac_training/agents/ -v

# Test training system
python -m pytest ac_training/training/ -v
```

### Integration Tests

```bash
# End-to-end training test
python ac_training/test_integration.py --quick

# Performance benchmarks
python ac_training/test_performance.py --profile
```

### Validation Checks

```python
# Validate configurations
config = get_droid_config()
config.validate()  # Raises if invalid

# Check data compatibility  
validate_h5_dataset("/path/to/data")

# Verify agent setup
agent = ACRLPDPi0Agent(config, rngs)
batch = dataloader.sample_batch()
loss, info = agent.compute_loss(batch, rng)
```

## Hyperparameter Sweeps

### Configuration Generation

```bash
# Generate sweep configs
python ac_training/scripts/generate_config.py \
    --platform droid \
    --sweep \
    --param_ranges horizon_length:5,10,15 bc_alpha:0.001,0.01,0.1 \
    --sweep_dir ./sweep_configs

# Train sweep
for config in sweep_configs/*.json; do
    python ac_training/scripts/train_acrlpd_pi0.py \
        --config_file $config \
        --experiment_name "sweep_$(basename $config .json)"
done
```

### Recommended Parameters

**Action Horizon Length**
- Short (5): Fast feedback, simple tasks
- Medium (10): Balanced performance  
- Long (20): Complex manipulation, bimanual tasks

**BC Regularization**
- Strong (0.1): Preserve demonstrations, low data regime
- Medium (0.01): Balanced exploration-exploitation
- Weak (0.001): Maximum RL learning, rich data

**Best-of-N Samples**
- Conservative (16): Fast inference
- Standard (32): Good performance-speed tradeoff  
- Aggressive (64): Maximum performance

## Troubleshooting

### Common Issues

**1. Out of Memory**
```bash
# Reduce batch size and increase gradient accumulation
--batch_size 64 --gradient_accumulation_steps 2

# Enable mixed precision
--mixed_precision

# Reduce critic ensemble size
--num_critics 5
```

**2. Training Instability**
```bash
# Increase BC regularization
--bc_alpha 0.1

# Reduce learning rates
--pi0_lr 1e-6 --critic_lr 1e-4

# Enable gradient clipping
--gradient_clip 0.5
```

**3. Poor Sample Efficiency**
```bash
# Increase Best-of-N samples
--best_of_n_samples 64

# Use larger critic ensemble
--num_critics 20

# Longer action horizons
--horizon_length 15
```

**4. Slow Training**
```bash
# Reduce diffusion steps
--diffusion_steps 5

# Disable Best-of-N during training
--no_best_of_n

# Increase batch size
--batch_size 256
```

### Debug Mode

```bash
# Enable detailed logging
python ac_training/scripts/train_acrlpd_pi0.py \
    --data_dir /path/to/data \
    --config droid \
    --debug \
    --dry_run

# Profile performance
python ac_training/scripts/train_acrlpd_pi0.py \
    --data_dir /path/to/data \
    --config droid \
    --profile \
    --offline_steps 1000
```

## API Reference

### Core Classes

#### `ACRLPDPi0Agent`
Main agent class integrating ACRLPD with π₀.

**Key Methods:**
- `sample_actions(obs, rng, use_best_of_n=True) -> Actions`
- `compute_loss(batch, rng) -> (loss, LossInfo)`
- `train_step(batch, rng) -> (updated_agent, LossInfo)`

#### `ACRLPDTrainer` 
Complete training system with offline and online phases.

**Key Methods:**
- `train(resume_from=None) -> ACRLPDPi0Agent`
- `_train_offline() -> ACRLPDPi0Agent`
- `_train_online() -> ACRLPDPi0Agent`

#### `CriticNetworks`
Ensemble of Q-networks for value estimation.

**Key Methods:**
- `__call__(obs, actions, use_target=False) -> Q_values`
- `soft_update_target_networks(tau=0.005) -> None`

### Configuration Classes

#### `ACRLPDPi0Config`
Complete agent configuration.

**Key Parameters:**
- `horizon_length: int` - Action chunking horizon
- `loss_weights: LossWeights` - Training loss weights
- `best_of_n_samples: int` - Sampling candidates
- `use_adaptive_temperature: bool` - Entropy control

#### `TrainingConfig`
Training pipeline configuration.

**Key Parameters:**
- `offline_steps: int` - Offline training duration
- `online_steps: int` - Online fine-tuning duration  
- `eval_frequency: int` - Evaluation interval
- `use_wandb: bool` - Experiment tracking

### Factory Functions

#### `create_acrlpd_pi0_agent(config, rng) -> ACRLPDPi0Agent`
Create agent with configuration.

#### `get_droid_config() -> ACRLPDPi0Config`
DROID platform preset.

#### `get_aloha_config() -> ACRLPDPi0Config` 
ALOHA platform preset.

#### `get_libero_config() -> ACRLPDPi0Config`
Libero simulation preset.

## Contributing

### Development Setup

```bash
# Install development dependencies
uv sync --dev

# Install pre-commit hooks
pre-commit install

# Run full test suite
python -m pytest ac_training/ -v --cov=ac_training
```

### Code Organization

- `agents/`: Core agent implementation
- `training/`: Training pipeline and utilities
- `scripts/`: Command-line interfaces
- `tests/`: Comprehensive test suite
- `configs/`: Configuration presets and examples

### Contributing Guidelines

1. **New Features**: Add comprehensive tests and documentation
2. **Performance**: Include benchmarks for optimization changes
3. **Configurations**: Test on multiple platforms
4. **API Changes**: Maintain backward compatibility where possible

## License

This project is part of the OpenPI framework. See the main project LICENSE for details.
