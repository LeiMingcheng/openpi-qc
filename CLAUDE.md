# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## OpenPI - Physical Intelligence Open Source Models

OpenPI is a repository for open-source robotics models from Physical Intelligence, primarily containing Vision-Language-Action (VLA) models:

- **π₀ (Pi-Zero)**: Flow-based diffusion VLA model trained on 10k+ hours of robot data
- **π₀-FAST**: Autoregressive VLA model using FAST action tokenizer for faster inference

## Development Environment

### Dependencies & Installation
```bash
# Clone with submodules (required for LeRobot dependency)
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
git submodule update --init --recursive

# Install using uv (preferred)
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Alternative Docker setup available (see docs/docker.md)
```

### Testing & Quality
```bash
# Run tests
uv run pytest src/

# Linting and formatting (configured in pyproject.toml)
uv run ruff check
uv run ruff format

# LeRobot-specific tests (from lerobot/ directory)
cd lerobot
make test-end-to-end
make test-act-ete-train
make test-diffusion-ete-train
make test-tdmpc-ete-train

# Run specific test files
uv run pytest src/openpi/models/pi0_test.py
uv run pytest src/openpi/policies/policy_test.py
```

### Environment Variables
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` - Enable JAX to use 90% GPU memory for training
- `OPENPI_DATA_HOME` - Override default checkpoint cache location (~/.cache/openpi)
- `GIT_LFS_SKIP_SMUDGE=1` - Required when installing with LeRobot dependency

## Core Architecture

### Model Components
- **src/openpi/models/**: Core model implementations
  - `pi0.py` - Diffusion-based π₀ model
  - `pi0_fast.py` - Autoregressive π₀-FAST model  
  - `model.py` - Base model interfaces and data structures
  - `tokenizer.py` - Action tokenization for FAST models
  - `siglip.py`, `vit.py` - Vision encoders
  - `gemma.py`, `gemma_fast.py` - Language model components

### Policy System
- **src/openpi/policies/**: Robot-specific policy implementations
  - `policy.py` - Base policy interface extending openpi-client BasePolicy
  - `aloha_policy.py` - ALOHA robot platform policies
  - `droid_policy.py` - DROID dataset/platform policies  
  - `libero_policy.py` - Libero simulation environment policies

### Training Infrastructure
- **src/openpi/training/**: Training pipeline components
  - `config.py` - Training configuration dataclasses and factory functions
  - `data_loader.py` - Data loading with LeRobot dataset integration
  - `checkpoints.py` - Model checkpointing and state management
  - `optimizer.py` - Optimizers and learning rate schedulers
  - `sharding.py` - Multi-GPU training support
  - `weight_loaders.py` - Pre-trained weight loading utilities

## Common Development Tasks

### Training Models
```bash
# Compute normalization statistics (required before training)
uv run scripts/compute_norm_stats.py --config-name pi0_fast_libero

# Fine-tune π₀-FAST on custom data
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_fast_libero --exp-name=my_experiment --overwrite

# Monitor training progress with W&B dashboard
```

### Running Inference
```bash
# Start policy server for remote inference
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid --policy.dir=checkpoints/path

# Test inference without robot hardware
uv run examples/simple_client/main.py
```

### Data Processing
```bash
# Convert custom datasets to LeRobot format (modify for your data)
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/data
```

## Key Configuration Patterns

### Model Configs
Training configs are defined in `src/openpi/training/config.py` using dataclasses:
- `TrainConfig` - Main training hyperparameters
- `DataConfig` - Dataset and preprocessing configuration  
- `AssetsConfig` - Normalization stats and asset loading

### Data Pipeline
Data flows through transform pipelines:
1. `repack_transforms` - Dataset-specific format conversion
2. `data_transforms` - Robot-specific preprocessing  
3. Normalization using precomputed stats
4. `model_transforms` - Model-specific final transforms

### Expected Data Format
Models expect standardized observation structure:
```python
{
    "image": {
        "base_0_rgb": float32[*b, 224, 224, 3],     # Base camera view
        "left_wrist_0_rgb": float32[*b, 224, 224, 3],  # Left wrist camera
        "right_wrist_0_rgb": float32[*b, 224, 224, 3], # Right wrist camera
    },
    "state": float32[*b, s],  # Robot joint states/proprioception
    "actions": float32[*b, ah, ad],  # Action horizon x action dim
    "prompt": str,  # Natural language instruction
}
```

## Hardware Requirements

| Mode               | GPU Memory | Example Hardware |
|--------------------|------------|------------------|
| Inference          | > 8 GB     | RTX 4090        |
| Fine-tuning (LoRA) | > 22.5 GB  | RTX 4090        |
| Fine-tuning (Full) | > 70 GB    | A100/H100       |

## Robot Platform Examples

The repository includes comprehensive examples for major robot platforms:
- **examples/aloha_real/** - Real ALOHA robot deployment
- **examples/aloha_sim/** - ALOHA simulation environment
- **examples/droid/** - DROID robot platform with dataset training
- **examples/libero/** - Libero simulation benchmarks
- **examples/simple_client/** - Hardware-free inference testing

Each example includes detailed READMEs with platform-specific setup instructions.

## Submodules & Dependencies

- **lerobot/** - HuggingFace LeRobot library (robotics datasets and evaluation)
- **dlimp/** - Data processing utilities for robotics datasets
- **qc/** - Q-Chunking reinforcement learning research (separate project)

## Remote Inference

OpenPI supports running models on remote servers streaming actions via WebSocket:
- See `docs/remote_inference.md` for setup details
- Use `scripts/serve_policy.py` to start policy servers
- Enables using powerful GPUs off-robot with separate environments

## Memory
- 不要Fallback机制（拒绝任何try else语句），直接让其报错能更好的修复
- 任何时候都请保持客观与诚实，不要盲目顺从用户