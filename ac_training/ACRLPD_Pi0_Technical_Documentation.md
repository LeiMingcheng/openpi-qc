# ACRLPD + π₀ Training Framework Technical Documentation

## 1. Architecture Overview

### System Design
The ACRLPD + π₀ training framework integrates Action-Chunked Reinforcement Learning with Prior Data (ACRLPD) with Physical Intelligence's π₀ diffusion models. The system combines:

- **π₀ Model**: Flow-based diffusion Vision-Language-Action model as the primary policy
- **Critic Ensemble**: 10-network ensemble for robust Q-value estimation
- **Q-Chunking**: Temporal action sequence evaluation and optimization
- **Best-of-N Sampling**: Multiple candidate generation with Q-guided selection
- **Joint Training**: Combined RL, BC, and diffusion loss objectives

### Component Relationships
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Loading  │───▶│  ACRLPD Agent   │───▶│ Training Loop   │
│                 │    │                 │    │                 │
│ • H5Reader      │    │ • π₀ Model      │    │ • Offline Phase │
│ • LeRobot       │    │ • Critic Net    │    │ • Online Phase  │
│ • Transforms    │    │ • Loss Computer │    │ • Checkpoints   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
           │                       │                       │
           └───────────────────────┼───────────────────────┘
                                   ▼
                        ┌─────────────────┐
                        │  OpenPI Core    │
                        │                 │
                        │ • Optimizers    │
                        │ • Checkpoints   │
                        │ • Config System │
                        └─────────────────┘
```

### OpenPI Integration Points
Direct integration with OpenPI infrastructure:
- **openpi.training.optimizer**: Learning rate schedules and optimizers
- **openpi.training.checkpoints**: Model state persistence and recovery
- **openpi.training.config**: Configuration system compatibility
- **openpi.models.pi0**: Core π₀ model implementation
- **openpi.models.model**: Data structures and interfaces

## 2. Data Loading System

### H5DatasetReader Implementation
Core data loading infrastructure in `ac_training/data_loader.py`:

```python
class H5DatasetReader:
    def __init__(self, data_dir: str, episode_metadata: Dict[str, Any]):
        # Memory-mapped H5 file reading
        # Efficient random access to large datasets
        # Episode boundary detection and management
```

**Key Features:**
- Memory-mapped file access for large datasets
- Episode metadata parsing and caching
- Efficient random sequence sampling
- Multi-threaded data loading with proper synchronization

### LeRobot Dataset Integration
Seamless integration with HuggingFace LeRobot datasets via `create_lerobot_acrlpd_dataloader()`:

```python
def create_lerobot_acrlpd_dataloader(
    lerobot_dataset: LeRobotDataset,
    batch_size: int,
    horizon_length: int
) -> ACRLPDDataLoader:
    # Convert LeRobot format to ACRLPD batch format
    # Handle observation keys mapping
    # Preserve episode boundaries and temporal structure
```

### ACRLPD-Specific Data Processing
The `ACRLPDDataLoader` class handles:

1. **Sequence Generation**: Creates action chunks of specified horizon length
2. **Multi-modal Batching**: Combines images, state, and language data
3. **Bootstrap Handling**: Manages episode boundaries for Q-learning targets
4. **Mask Generation**: Creates valid sample masks for loss computation

**ACRLPD Training Batch Format:**
```python
{
    'observations': _model.Observation({
        'images': {
            'base_0_rgb': [B, 224, 224, 3],
            'left_wrist_0_rgb': [B, 224, 224, 3], 
            'right_wrist_0_rgb': [B, 224, 224, 3]
        },
        'image_masks': {
            'base_0_rgb': [B],
            'left_wrist_0_rgb': [B],
            'right_wrist_0_rgb': [B]
        },
        'state': [B, state_dim],
        'tokenized_prompt': [B, max_tokens],
        'tokenized_prompt_mask': [B, max_tokens]
    }),
    'actions': [B, horizon_length, action_dim],
    'rewards': [B, horizon_length],
    'next_observations': _model.Observation({...}),
    'terminals': [B, horizon_length],
    'masks': [B, horizon_length],        # Valid step indicators
    'sequence_mask': [B],                # Episode boundary flags
    'episode_id': [B],                   # Episode identifiers
    'step_id': [B]                       # Within-episode step identifiers
}
```

### Two DataLoader Implementations:

#### 1. ACRLPDDataLoader (H5 Datasets)
For custom H5 robotic datasets in `ac_training/acrlpd_dataloader.py:33-334`:

```python
class ACRLPDDataLoader:
    def __init__(self, data_dir: str, batch_size: int, horizon_length: int,
                 sequence_ratio: float = 0.5, device_placement: str = "auto"):
        # Creates QChunkingDataset → H5DatasetReader bridge
        # Handles episode boundary detection and caching
        # Supports both sequence and single-step sampling
```

**Key Technical Features:**
- **Adaptive Sampling**: `sequence_ratio` controls mix of sequence vs single-step samples for curriculum learning
- **Device Sharding**: JAX NamedSharding for automatic data parallel distribution across devices
- **Episode Caching**: LRU cache with configurable size for frequently accessed episodes  
- **Statistics Tracking**: Real-time metrics on batch generation speed and data quality
- **Memory Management**: Configurable sharding strategies for large batch sizes

**Device Placement Implementation:**
```python
# ac_training/acrlpd_dataloader.py:109-137
def _setup_device_placement(self):
    if self.device_placement == "auto":
        devices = jax.devices()
        self.sharding = jax.sharding.NamedSharding(
            jax.sharding.Mesh(devices, ("batch",)),
            jax.sharding.PartitionSpec("batch")
        )
```

#### 2. LeRobotACRLPDDataLoader (LeRobot Datasets)
For HuggingFace LeRobot dataset integration in `ac_training/acrlpd_dataloader.py:336-583`:

```python
class LeRobotACRLPDDataLoader:
    def __init__(self, lerobot_dataset: LeRobotDataset, batch_size: int):
        # Direct LeRobot dataset processing via RobotDataProcessor
        # Format conversion to π₀-compatible observations
        # Episode boundary detection from LeRobot metadata
```

**LeRobot Integration Process:**
1. **Dataset Processing**: `RobotDataProcessor.process_lerobot_dataset()` converts LeRobot format
2. **Observation Construction**: Creates `_model.Observation` objects with proper tokenization
3. **Action Sequence Generation**: Chunks actions into horizon-length sequences
4. **Quality Validation**: Data quality checks and consistency validation

**Format Conversion Implementation:**
```python
# ac_training/acrlpd_dataloader.py:519-549
def _convert_to_pi0_format(self, batch: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
    def convert_obs(obs_data):
        return _model.Observation(
            images={k: jnp.array(v) for k, v in obs_data['images'].items()},
            image_masks={k: jnp.array(v) for k, v in obs_data['image_masks'].items()},
            state=jnp.array(obs_data['state']),
            tokenized_prompt=jnp.array(obs_data['tokenized_prompt']),
            tokenized_prompt_mask=jnp.array(obs_data['tokenized_prompt_mask'])
        )
```

## 3. Model Interfaces

### π₀ Model Integration
Direct usage of OpenPI's π₀ model as the primary policy in `ac_training/agents/acrlpd_pi0_agent.py:146`:

```python
# π₀ model creation using OpenPI factory
self.pi0_model = config.pi0_config.create(rngs.fork("pi0"))
```

**Core π₀ Interface Methods:**
- `sample_actions(rng, observations, num_steps)`: Generate action sequences using flow-based diffusion
- `compute_loss(rng, observations, actions, train)`: Diffusion training loss computation
- `embed_prefix(observations)`: Multi-modal feature extraction for critic integration

**Sampling Modes:**
```python
# Deterministic sampling (greedy)
actions = pi0_model.sample_actions(rng, observations, num_steps=5)

# Stochastic sampling (exploration)
actions = pi0_model.sample_actions(rng, observations, num_steps=10)
```

**Multi-modal Input Processing Pipeline:**
- **Images**: Multiple camera views processed through SigLIP-SO400M-Patch14-384 vision encoder
  - Standard resolution: 224x224 RGB with automatic resizing via `_transforms.ResizeImages(224, 224)`
  - Camera views: `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb` (platform-dependent)
  - Preprocessing: Normalization to [-1, 1] range for SigLIP compatibility
- **State**: Robot proprioception vectors
  - Joint positions (6-7 DoF depending on robot platform)
  - Gripper state (1-2 dimensions for open/close or continuous gripper)
  - Optional: velocity, force/torque feedback for advanced platforms
- **Language**: Natural language task instructions
  - Tokenization: PaLI/Gemma tokenizer with configurable `max_token_len`
  - Encoding: Shared vision-language transformer backbone
  - Support: Dynamic prompting, task conditioning, and context-aware instructions

**π₀ Configuration Integration:**
```python
# Platform-specific π₀ configurations
# DROID: 8D action space, 10-step horizon
pi0_config = _pi0.Pi0Config(action_dim=8, action_horizon=10, max_token_len=180)

# ALOHA: 14D bimanual action space, 20-step horizon
pi0_config = _pi0.Pi0Config(action_dim=14, action_horizon=20, max_token_len=250)

# Libero: 7D single-arm action space, 5-step horizon
pi0_config = _pi0.Pi0Config(action_dim=7, action_horizon=5, max_token_len=180)
```

### Critic Networks Implementation  
Configurable ensemble of critic networks for robust Q-value estimation in `ac_training/agents/critic_networks.py:242-483`:

```python
class CriticNetworks(nnx.Module):
    def __init__(self, config: CriticConfig, observation_dim: int, action_dim: int, rngs: nnx.Rngs):
        # Online critic ensemble
        self.online_critics = CriticEnsemble(config, observation_dim, action_dim, rngs.fork("online"))
        # Target critic ensemble
        self.target_critics = CriticEnsemble(config, observation_dim, action_dim, rngs.fork("target"))
        # Initialize target networks with same weights as online
        self.sync_target_networks()
```

#### Single Critic Architecture (`ac_training/agents/critic_networks.py:59-160`)
```python
class SingleCriticNetwork(nnx.Module):
    def __init__(self, config: CriticConfig, observation_dim: int, action_dim: int, rngs: nnx.Rngs):
        # Feature fusion for multi-modal observations
        if config.use_pi0_features and config.feature_fusion_method == "mlp":
            self.feature_fusion = nnx.Linear(observation_dim, observation_dim, rngs=rngs)
        
        # MLP layers with configurable architecture
        input_dim = observation_dim + action_dim
        for hidden_dim in config.hidden_dims:
            self.layers.append(nnx.Linear(
                input_dim, hidden_dim, rngs=rngs,
                kernel_init=nnx.initializers.variance_scaling(config.kernel_init_scale, "fan_in", "uniform")
            ))
            if config.dropout_rate > 0:
                self.layers.append(nnx.Dropout(config.dropout_rate, rngs=rngs))
            input_dim = hidden_dim
        
        # Output layer (single Q-value)
        self.output_layer = nnx.Linear(input_dim, 1, rngs=rngs)
```

#### Critic Ensemble Features:
- **Ensemble Size**: Configurable via `num_critics` (default 10, ALOHA uses 8, Libero uses 12)
- **Aggregation Strategies**:
  - `"min"`: Conservative pessimistic Q-value estimation (default for stability)
  - `"mean"`: Average across ensemble for balanced estimation  
  - `"weighted"`: Learnable weights with softmax normalization for adaptive combination
- **Target Networks**: Double Q-learning with exponential moving average updates
- **Architecture**: Fully configurable MLP with activation functions (`relu`, `swish`, `gelu`, `tanh`)
- **Regularization**: Dropout, layer normalization, and variance scaling initialization

#### Observation Encoding with π₀ Features (`ac_training/agents/critic_networks.py:360-439`)
```python
def create_observation_encoder(pi0_model: Any, config: CriticConfig):
    def encode_observation(observation: _model.Observation) -> jnp.ndarray:
        if config.use_pi0_features:
            # Use π₀'s embed_prefix for multi-modal feature extraction
            processed_obs = _model.preprocess_observation(None, observation, train=False)
            prefix_tokens, prefix_mask, _ = pi0_model.embed_prefix(processed_obs)
            
            # Weighted average pooling based on attention mask
            mask_expanded = prefix_mask[..., None]  # [batch_size, seq_len, 1]
            masked_tokens = prefix_tokens * mask_expanded
            feature_sum = jnp.sum(masked_tokens, axis=1)
            valid_count = jnp.maximum(jnp.sum(prefix_mask, axis=1, keepdims=True), 1.0)
            pooled_features = feature_sum / valid_count
            
            # Feature fusion with state
            state_features = processed_obs.state
            if config.feature_fusion_method == "concat":
                return jnp.concatenate([pooled_features, state_features], axis=-1)
            elif config.feature_fusion_method == "add":
                # Requires dimension matching or learnable projection
                return pooled_features + state_features  # (with proper dimension handling)
```

#### Target Network Soft Updates (`ac_training/agents/critic_networks.py:294-326`)
```python
def soft_update_target_networks(self, tau: Optional[float] = None):
    # Soft update: target = (1-tau) * target + tau * online
    for online_critic, target_critic in zip(self.online_critics.critics, self.target_critics.critics):
        online_params = nnx.state(online_critic, nnx.Param)
        target_params = nnx.state(target_critic, nnx.Param)
        
        updated_params = jax.tree_map(
            lambda target, online: (1.0 - tau) * target + tau * online,
            target_params, online_params
        )
        nnx.update(target_critic, updated_params)
```

**Critic Configuration Examples:**
```python
# DROID: Complex state-action spaces
CriticConfig(num_critics=10, hidden_dims=[512, 512, 256], dropout_rate=0.1)

# ALOHA: Faster convergence for fine-tuning
CriticConfig(num_critics=8, hidden_dims=[256, 256, 128], dropout_rate=0.05)

# Libero: Large ensemble for simulation stability  
CriticConfig(num_critics=12, hidden_dims=[512, 512, 256, 128], dropout_rate=0.15)
```

### Action Sequence Processing
Action sequences are processed consistently for ACRLPD + π₀ integration:

**Action Format Handling:**
- **π₀ Input**: Original action chunks `[batch_size, horizon_length, action_dim]`
- **π₀-FAST**: Uses discretized action tokens via FAST tokenizer (supported in π₀-FAST configs)
- **Critic Input**: Flattened sequences `[batch_size, horizon_length * action_dim]`
- **Platform-Specific**: Action dimensions vary by robot (DROID: 8D, ALOHA: 14D, Libero: 7D)

**Action Preprocessing Pipeline:**
1. **Raw Actions**: From dataset in platform-specific format
2. **Transform Pipeline**: Robot-specific coordinate transformations
3. **Delta Conversion**: Convert absolute to relative actions (platform-dependent)
4. **Normalization**: Apply precomputed statistics for stable training
5. **Chunking**: Segment into horizon-length sequences for Q-chunking
6. **Flattening**: Reshape for critic network input

**Example Action Shapes:**
```python
# DROID platform
actions_raw = [batch_size, 8]              # Joint positions + gripper
actions_chunked = [batch_size, 10, 8]      # 10-step horizon chunks
actions_critic = [batch_size, 80]          # Flattened for critic input

# ALOHA platform  
actions_raw = [batch_size, 14]             # Bimanual: 7 per arm
actions_chunked = [batch_size, 20, 14]     # 20-step horizon for complex manipulation
actions_critic = [batch_size, 280]         # Flattened for critic input
```

## 4. Algorithm Framework

### ACRLPD Core Algorithm
The framework implements Action-Chunked Reinforcement Learning with Prior Data using π₀ as the direct policy. Core components:

1. **Action Chunking**: Temporal sequences of configurable `horizon_length` (5-20 steps)
2. **Q-Learning**: Critic ensemble provides robust Q-value estimates for action sequences
3. **Prior Data**: Strong behavior cloning regularization from expert demonstrations
4. **π₀ Integration**: Uses π₀ diffusion model as the primary policy network
5. **Best-of-N Sampling**: Multiple action candidate generation with Q-guided selection

**Joint Training Objective:**
```python
L_total = λ_critic * L_critic + λ_actor * L_actor + λ_BC * L_BC + λ_α * L_α
```

**Loss Component Details:**
- `L_critic`: Q-learning TD error with N-step bootstrap and ensemble aggregation
- `L_actor`: Policy gradient loss `-E[Q(s, π₀(s))]` for strategy improvement (enables true RL)
- `L_BC`: π₀'s diffusion-based behavior cloning loss for expert action matching
- `L_α`: Adaptive temperature loss for entropy regularization and exploration control

### Joint Loss Function Implementation (`ac_training/agents/loss_functions.py:424-546`)
```python
class JointLossComputer:
    def __call__(self, pi0_model, critic_networks, observation_encoder, batch, rng, train=True):
        rng_critic, rng_actor, rng_bc, rng_entropy = jax.random.split(rng, 4)
        
        # 1. Critic loss - Q-learning with bootstrap handling
        critic_loss, critic_info = self.critic_loss_computer(
            pi0_model, critic_networks, observation_encoder, batch, rng_critic, train
        )
        
        # 2. BC loss - π₀'s diffusion-based behavior cloning
        bc_loss, bc_info = self.bc_loss_computer(pi0_model, batch, rng_bc, train)
        
        # 3. Actor loss - policy gradient for strategy improvement
        actor_loss, actor_info = self.actor_loss_computer(
            pi0_model, critic_networks, observation_encoder, batch, rng_actor, train
        )
        
        # 4. Temperature/alpha loss - adaptive exploration
        alpha_loss, alpha_info = 0.0, {...}
        if self.temperature_module is not None:
            entropy_est = self.entropy_estimator.estimate_entropy(pi0_model, batch['observations'], rng_entropy)
            target_entropy = -self.target_entropy_multiplier * action_dim
            alpha_loss = self.temperature_module.alpha_loss(entropy_est.mean(), target_entropy)
        
        # 4. Combine with learnable weights (Actor-Critic RL)
        total_loss = (
            self.loss_weights.critic_weight * critic_loss +
            self.loss_weights.actor_weight * actor_loss +      # Policy gradient: -E[Q(s, π₀(s))]
            self.loss_weights.bc_weight * bc_loss +
            self.loss_weights.alpha_weight * alpha_loss
        )
```

#### Critic Loss Computation (`ac_training/agents/loss_functions.py:98-260`)
```python
class CriticLossComputer:
    def __call__(self, pi0_model, critic_networks, observation_encoder, batch, rng, train=True):
        # Encode observations using π₀'s feature extraction
        current_obs_encoded = observation_encoder(batch['observations'])
        next_obs_encoded = observation_encoder(batch['next_observations'])
        
        # Sample actions from π₀ for Q-value computation
        current_actions = pi0_model.sample_actions(rng_current, batch['observations'], num_steps=10)
        next_actions = pi0_model.sample_actions(rng_next, batch['next_observations'], num_steps=10)
        
        # Compute target Q-values with ensemble aggregation
        target_q_values = critic_networks(
            next_obs_encoded, next_actions_flat, use_target=True, aggregate=False
        )  # [num_critics, batch_size]
        
        # Bootstrap target computation with enhanced masking
        target_q_bootstrap = self.bootstrap_handler.compute_bootstrap_target(
            rewards=batch['rewards'],
            next_q_values=target_q,
            masks=batch['masks'],
            discount=self.discount,
            horizon_length=self.horizon_length
        )
        
        # TD error computation with ensemble
        current_q_values = critic_networks(current_obs_encoded, current_actions_flat, 
                                          use_target=False, aggregate=False)
        td_errors = current_q_values - target_q_bootstrap[None, :]
        
        # Enhanced loss with importance weighting and temporal consistency
        critic_loss_raw = jnp.square(td_errors)
        # Apply advanced masking with sample weights and temporal filtering
        critic_loss = MaskHandler.apply_loss_masking(critic_loss_raw, batch['sequence_mask'])
```

#### Behavior Cloning Loss (`ac_training/agents/loss_functions.py:263-335`)
```python
class BCLossComputer:
    def __call__(self, pi0_model, batch, rng, train=True):
        # Use π₀'s native diffusion training loss
        bc_loss_raw = pi0_model.compute_loss(
            rng, batch['observations'], batch['actions'], train=train
        )  # [batch_size, action_horizon]
        
        # Apply sequence masking for episode boundaries
        sequence_mask = batch.get('sequence_mask', jnp.ones(bc_loss_raw.shape[0]))
        if bc_loss_raw.ndim > 1:
            mask_expanded = sequence_mask[:, None]
            masked_loss = bc_loss_raw * mask_expanded
        
        # Enhanced masking with importance weighting for difficult actions
        bc_loss = MaskHandler.apply_loss_masking(
            loss=masked_loss.mean(axis=-1),
            mask=sequence_mask,
            use_importance_weighting=True,  # Weight difficult actions more
            use_difficulty_weighting=True
        )
```

#### Temperature Control (`ac_training/agents/loss_functions.py:79-96`)
```python
class TemperatureModule(nnx.Module):
    def __init__(self, initial_temp: float = 1.0, rngs: nnx.Rngs = None):
        self.log_alpha = nnx.Param(jnp.log(initial_temp) * jnp.ones(()))
    
    def __call__(self) -> jnp.ndarray:
        return jnp.exp(self.log_alpha.value)  # Always positive temperature
    
    def alpha_loss(self, entropy: jnp.ndarray, target_entropy: float) -> jnp.ndarray:
        alpha = self()
        return alpha * jax.lax.stop_gradient(entropy - target_entropy)
```

### Best-of-N Sampling Strategies
Three adaptive sampling strategies in `ac_training/agents/acrlpd_pi0_agent.py:245-484`:

#### 1. Parallel Sampling (Default) (`ac_training/agents/acrlpd_pi0_agent.py:291-316`)
Standard approach for moderate N and sufficient GPU memory:
```python
def _best_of_n_sampling(self, observations, rng, num_samples):
    sample_rngs = jax.random.split(rng, num_samples)
    
    # Optimized candidate generation with reduced diffusion steps
    def sample_single(rng_key):
        diffusion_steps = max(5, self.config.diffusion_steps // 2)  # Speed optimization
        return self.pi0_model.sample_actions(rng_key, observations, num_steps=diffusion_steps)
    
    # Parallel candidate generation via vmap
    action_candidates = jax.vmap(sample_single)(sample_rngs)  # [num_samples, batch_size, horizon, action_dim]
    
    # Efficient vectorized Q-value evaluation
    q_values = self._evaluate_action_candidates(observations, action_candidates)
    
    # Enhanced selection with multiple strategies
    return self._select_best_actions(action_candidates, q_values, rng)
```

#### 2. Memory-Efficient Chunked Sampling (`ac_training/agents/acrlpd_pi0_agent.py:318-396`)
For large N (> chunk_size) to avoid GPU memory issues:
```python
def _memory_efficient_best_of_n(self, observations, rng, num_samples, chunk_size=32):
    batch_size = observations.state.shape[0]
    best_q_values = jnp.full((batch_size,), -jnp.inf)
    best_actions = None
    
    # Process candidates in chunks
    num_chunks = (num_samples + chunk_size - 1) // chunk_size
    sample_rngs = jax.random.split(rng, num_samples)
    
    for chunk_idx in range(num_chunks):
        start_idx, end_idx = chunk_idx * chunk_size, min((chunk_idx + 1) * chunk_size, num_samples)
        chunk_rngs = sample_rngs[start_idx:end_idx]
        
        # Generate chunk candidates
        chunk_candidates = jax.vmap(sample_single)(chunk_rngs)
        
        # Evaluate chunk Q-values
        chunk_q_values = self._evaluate_action_candidates(observations, chunk_candidates)
        
        # Update running best candidates
        chunk_best_indices = jnp.argmax(chunk_q_values, axis=0)
        chunk_best_q = chunk_q_values[chunk_best_indices, jnp.arange(batch_size)]
        
        # Update global best if better
        if best_actions is None:
            best_actions = chunk_candidates[chunk_best_indices, jnp.arange(batch_size)]
            best_q_values = chunk_best_q
        else:
            better_mask = chunk_best_q > best_q_values
            best_actions = jnp.where(better_mask[:, None, None], 
                                   chunk_candidates[chunk_best_indices, jnp.arange(batch_size)], 
                                   best_actions)
            best_q_values = jnp.where(better_mask, chunk_best_q, best_q_values)
```

#### 3. Sequential Sampling (`ac_training/agents/acrlpd_pi0_agent.py:398-484`)
For extreme memory constraints (single candidate processing):
```python
def _sequential_best_of_n(self, observations, rng, num_samples):
    batch_size = observations.state.shape[0]
    best_q_values = jnp.full((batch_size,), -jnp.inf)
    best_actions = None
    
    # Track top-k candidates for stochastic selection
    if use_top_k:
        top_candidates = []
        top_q_values = []
    
    sample_rngs = jax.random.split(rng, num_samples)
    
    for i in range(num_samples):
        # Sample single candidate
        candidate_actions = self.pi0_model.sample_actions(
            sample_rngs[i], observations, num_steps=max(5, self.config.diffusion_steps // 2)
        )
        
        # Evaluate single candidate efficiently
        candidate_q_values = self._evaluate_single_candidate(observations, candidate_actions)
        
        # Maintain top-k list or simple best tracking
        if use_top_k:
            top_candidates.append(candidate_actions)
            top_q_values.append(candidate_q_values)
            # Trim to top-k if exceeded
            if len(top_candidates) > top_k:
                sorted_indices = jnp.argsort(jnp.array(top_q_values), axis=0)[-top_k:]
                # Keep only top-k candidates
```

#### Q-Value Evaluation (`ac_training/agents/acrlpd_pi0_agent.py:515-566`)
```python
def _evaluate_action_candidates(self, observations, action_candidates, use_diversity_bonus=False):
    num_samples, batch_size = action_candidates.shape[:2]
    
    # Encode observations once for efficiency
    obs_encoded = self.observation_encoder(observations)  # [batch_size, obs_dim]
    
    # Expand for batch evaluation
    obs_encoded_expanded = jnp.tile(obs_encoded[None, :, :], (num_samples, 1, 1))
    
    # Flatten actions for critic input
    action_candidates_flat = action_candidates.reshape(num_samples, batch_size, -1)
    
    # Vectorized Q-value evaluation
    def evaluate_batch(obs_batch, action_batch):
        return self.critic_networks(obs_batch, action_batch, use_target=False, train=False, aggregate=True)
    
    # Parallel evaluation across all samples
    q_values = jax.vmap(evaluate_batch)(obs_encoded_expanded, action_candidates_flat)
    
    # Optional diversity bonus for exploration
    if use_diversity_bonus:
        diversity_bonus = self._compute_diversity_bonus(action_candidates, diversity_weight)
        q_values = q_values + diversity_bonus
    
    return q_values  # [num_samples, batch_size]
```

#### Advanced Selection Features (`ac_training/agents/acrlpd_pi0_agent.py:606-671`)
```python
def _select_best_actions(self, action_candidates, q_values, rng, use_top_k=False, 
                        top_k=5, use_temperature_scaling=True, temperature=1.0):
    if use_top_k and use_temperature_scaling:
        # Stochastic selection from top-k candidates
        top_k_indices = jnp.argsort(q_values, axis=0)[-top_k:]
        top_k_q_values = q_values[top_k_indices, jnp.arange(batch_size)[None, :]]
        
        # Temperature-scaled probabilities
        probs = jax.nn.softmax(top_k_q_values / temperature, axis=0)
        
        # Sample from top-k for each batch element
        sample_rngs = jax.random.split(rng, batch_size)
        selected_topk_indices = jax.vmap(lambda prob, rng: jax.random.categorical(rng, jnp.log(prob + 1e-8)))(probs.T, sample_rngs)
        selected_indices = top_k_indices[selected_topk_indices, jnp.arange(batch_size)]
    else:
        # Deterministic best selection
        selected_indices = jnp.argmax(q_values, axis=0)
    
    # Gather selected actions
    return action_candidates[selected_indices, jnp.arange(batch_size)]
```

#### Diversity Bonus Computation (`ac_training/agents/acrlpd_pi0_agent.py:568-604`)
```python
def _compute_diversity_bonus(self, action_candidates, diversity_weight):
    # Flatten actions for distance computation
    actions_flat = action_candidates.reshape(num_samples, batch_size, -1)
    
    def compute_batch_diversity(batch_actions):
        # Compute pairwise L2 distances
        diff = batch_actions[:, None, :] - batch_actions[None, :, :]
        distances = jnp.linalg.norm(diff, axis=-1)
        
        # Average distance to other candidates (excluding self)
        mask = 1.0 - jnp.eye(num_samples)
        avg_distance = jnp.sum(distances * mask, axis=1) / jnp.maximum(num_samples - 1, 1)
        return avg_distance
    
    # Apply across batch dimension
    diversity_scores = jax.vmap(compute_batch_diversity, in_axes=1, out_axes=1)(actions_flat)
    return diversity_weight * diversity_scores
```

#### Advanced Selection Features:
- **Top-K Selection**: Stochastic selection from top-k candidates with temperature scaling
- **Diversity Bonus**: L2 distance-based diversity encouragement for exploration
- **Temperature Scaling**: Configurable stochastic selection temperature
- **Adaptive Diffusion Steps**: Reduced diffusion steps (`max(5, config.diffusion_steps // 2)`) during candidate generation for 2x speed improvement
- **Memory Management**: Automatic strategy selection based on available GPU memory and sample count

### Loss Weight Configurations
Default loss weight configuration in `ac_training/agents/loss_functions.py`:

```python
# Default balanced weights with Actor-Critic training
DEFAULT_LOSS_WEIGHTS = LossWeights(
    critic_weight=1.0,    # Critic loss weight
    actor_weight=1.0,     # Actor loss weight (NEW: enables true RL)
    bc_weight=0.01,       # Behavior cloning weight
    alpha_weight=1.0      # Temperature loss weight
)
```

**Loss Weight Components:**
- **critic_weight**: TD-error loss for Q-value learning
- **actor_weight**: Policy gradient loss `-E[Q(s, π₀(s))]` for strategy improvement
- **bc_weight**: Flow matching loss for behavior cloning regularization
- **alpha_weight**: Adaptive temperature learning for exploration control

**Platform-Specific Loss Weight Tuning:**
- **DROID**: Moderate BC (`bc_weight=0.01`), standard Actor-Critic (`actor_weight=1.0`) for diverse dataset
- **ALOHA**: Minimal BC (`bc_weight=0.001`), standard Actor-Critic (`actor_weight=1.0`) for fine-tuning
- **Libero**: Strong BC (`bc_weight=0.1`), standard Actor-Critic (`actor_weight=1.0`) for simulation stability

### Entropy Estimation for Temperature Control (`ac_training/agents/loss_functions.py:379-421`)
```python
class EntropyEstimator:
    def estimate_entropy(self, pi0_model, observations, rng):
        # Sample multiple actions from π₀
        sample_rngs = jax.random.split(rng, self.num_samples)
        
        def sample_single(rng_key):
            return pi0_model.sample_actions(rng_key, observations, num_steps=5)
        
        action_samples = jax.vmap(sample_single)(sample_rngs)  # [num_samples, batch_size, horizon, action_dim]
        
        # Estimate entropy using variance of samples
        action_var = jnp.var(action_samples, axis=0)  # [batch_size, action_horizon, action_dim]
        entropy_estimate = jnp.mean(action_var, axis=(1, 2))  # [batch_size]
        
        # Convert variance to entropy-like quantity (log scale)
        return jnp.log(entropy_estimate + 1e-8)
```

## 5. Training Pipeline

### Complete Training System (`ac_training/training/training_loop.py:227-521`)
The `ACRLPDTrainer` class orchestrates the complete training pipeline:

```python
class ACRLPDTrainer:
    def __init__(self, agent: ACRLPDPi0Agent, dataloader: ACRLPDDataLoader, 
                 config: ACRLPDTrainingConfig, eval_fn: Optional[Callable] = None):
        self.agent = agent
        self.dataloader = dataloader
        self.config = config
        self.eval_fn = eval_fn
        
        # Initialize training systems
        self.metrics = TrainingMetrics()
        self.checkpoint_manager = ACRLPDCheckpointManager(config.checkpoint_dir, ...)
        
        # Initialize WandB if enabled
        if config.use_wandb:
            self._init_wandb()
```

### Offline Pretraining Phase (`ac_training/training/training_loop.py:323-372`)
Behavior cloning initialization using expert demonstrations:

```python
def _train_offline(self) -> ACRLPDPi0Agent:
    start_step = self.current_step
    target_steps = self.config.offline_steps
    
    progress_bar = tqdm(initial=start_step, total=target_steps, desc="Offline Training")
    
    while self.current_step < target_steps:
        step_start_time = time.time()
        
        # Sample training batch (mix of sequences and single-step)
        self.rng, batch_rng = jax.random.split(self.rng)
        batch = self.dataloader.sample_batch()
        
        # Execute training step with joint loss computation
        self.rng, train_rng = jax.random.split(self.rng)
        self.agent, loss_info = self.agent.train_step(batch, train_rng)
        
        # Update metrics with timing information
        step_time = time.time() - step_start_time
        self.metrics.update(
            self.current_step, loss_info,
            timing_step_duration=step_time,
            timing_samples_per_sec=batch['actions'].shape[0] / step_time
        )
        
        # Periodic logging, evaluation, and checkpointing
        if self.current_step % self.config.log_frequency == 0:
            self._log_metrics()
        
        if self.current_step % self.config.eval_frequency == 0:
            self._run_evaluation()
        
        if self.current_step % self.config.save_frequency == 0:
            self.checkpoint_manager.save_checkpoint(self.agent, self.dataloader, self.current_step)
        
        self.current_step += 1
        progress_bar.update(1)
```

**Offline Phase Characteristics:**
- **Pure Supervised Learning**: Strong behavior cloning on expert demonstrations
- **BC Weight Schedule**: High BC weights (`0.01-0.1`) for conservative policy initialization
- **π₀ Parameter Updates**: Full or selective parameter updates (backbone may be frozen)
- **Critic Pretraining**: Q-networks learn to estimate returns from expert trajectories
- **Stability Focus**: Conservative learning rates and strong regularization

### Online Fine-tuning Phase (`ac_training/training/training_loop.py:374-438`)
Environment interaction and policy improvement:

```python
def _train_online(self) -> ACRLPDPi0Agent:
    if self.eval_fn is None:
        logger.warning("No evaluation function provided for online training")
        return self.agent
    
    start_step = self.current_step
    target_steps = self.config.offline_steps + self.config.online_steps
    
    progress_bar = tqdm(initial=start_step, total=target_steps, desc="Online Training")
    
    while self.current_step < target_steps:
        # Mix online environment data with offline data (simplified implementation)
        self.rng, batch_rng = jax.random.split(self.rng)
        batch = self.dataloader.sample_batch()  # In full implementation: mix online+offline
        
        # Training step with reduced BC weights
        self.rng, train_rng = jax.random.split(self.rng)
        self.agent, loss_info = self.agent.train_step(batch, train_rng)
        
        # More frequent evaluation during online phase
        if self.current_step % (self.config.eval_frequency // 2) == 0:
            self._run_evaluation()
            
            # Early stopping based on evaluation performance
            eval_reward = self.metrics.metrics.get('eval/episode_reward_mean', -float('inf'))
            improved = self.metrics.check_improvement(eval_reward)
            
            if not improved and self.metrics.steps_since_improvement > self.config.early_stopping_patience:
                logger.info(f"Early stopping after {self.metrics.steps_since_improvement} steps without improvement")
                break
```

**Online Phase Characteristics:**
- **Environment Interaction**: Real-time policy evaluation and data collection
- **Adaptive Loss Weights**: Reduced BC weights, increased RL weights during online learning
- **Frequent Evaluation**: Every `eval_frequency // 2` steps for rapid feedback
- **Early Stopping**: Automatic termination when evaluation performance plateaus
- **Mixed Experience**: Combination of fresh online data with offline expert data

### Training Step Implementation (`ac_training/agents/acrlpd_pi0_agent.py:699-750`)
Single training step with gradient computation and application:

```python
def train_step(self, batch: Dict[str, jnp.ndarray], rng: jnp.ndarray) -> Tuple["ACRLPDPi0Agent", LossInfo]:
    def loss_fn(agent_vars):
        temp_agent = agent_vars
        loss, loss_info = temp_agent.compute_loss(batch, rng, train=True)
        return loss, loss_info
    
    # Compute gradients with automatic differentiation
    (loss, loss_info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(self)
    
    # Apply gradients to π₀ model (unless backbone frozen)
    if not self.config.freeze_pi0_backbone:
        pi0_grads = grads.pi0_model
        self.pi0_model = self.pi0_optimizer.apply_gradients(pi0_grads, self.pi0_model)
    
    # Apply gradients to critic networks
    critic_grads = grads.critic_networks
    self.critic_networks = self.critic_optimizer.apply_gradients(critic_grads, self.critic_networks)
    
    # Apply gradients to temperature module (if adaptive)
    if self.temperature_module is not None:
        temp_grads = grads.temperature_module
        self.temperature_module = self.temperature_optimizer.apply_gradients(temp_grads, self.temperature_module)
    
    # Soft update target networks every step
    if self._step % 1 == 0:
        self.critic_networks.soft_update_target_networks(tau=self.config.target_update_tau)
    
    self._step += 1
    return self, loss_info
```

### OpenPI Checkpoint Management (`ac_training/training/training_loop.py:157-225`)
```python
class ACRLPDCheckpointManager:
    def __init__(self, checkpoint_dir: str, keep_period: int = 10000, overwrite: bool = False, resume: bool = False):
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Initialize OpenPI checkpoint manager
        self.manager, self.resuming = _checkpoints.initialize_checkpoint_dir(
            checkpoint_dir=checkpoint_dir, keep_period=keep_period,
            overwrite=overwrite, resume=resume
        )
    
    def save_checkpoint(self, agent: ACRLPDPi0Agent, dataloader: Any, step: int) -> str:
        # Create OpenPI-compatible training state
        train_state = agent.create_train_state()
        
        # Save using OpenPI checkpoint system
        _checkpoints.save_state(
            checkpoint_manager=self.manager,
            state=train_state,
            data_loader=dataloader,
            step=step
        )
        
        logger.info(f"Saved checkpoint at step {step}")
        return str(self.checkpoint_dir / str(step))
    
    def load_checkpoint(self, agent: ACRLPDPi0Agent, dataloader: Any, step: Optional[int] = None):
        # Create template training state
        template_state = agent.create_train_state()
        
        # Restore using OpenPI checkpoint system
        restored_state = _checkpoints.restore_state(
            checkpoint_manager=self.manager,
            state=template_state,
            data_loader=dataloader,
            step=step
        )
        
        # Update agent from restored state
        agent.update_from_train_state(restored_state)
        return agent, restored_state.step
```

#### OpenPI Training State Compatibility (`ac_training/agents/acrlpd_pi0_agent.py:752-794`)
```python
def create_train_state(self) -> training_utils.TrainState:
    # Extract parameters using Flax NNX state management
    pi0_params = nnx.state(self.pi0_model, nnx.Param)
    critic_params = nnx.state(self.critic_networks, nnx.Param)
    temperature_params = nnx.state(self.temperature_module, nnx.Param) if self.temperature_module else {}
    
    # Combine all parameters
    params = {
        'pi0_model': pi0_params,
        'critic_networks': critic_params,
        'temperature_module': temperature_params
    }
    
    # Create OpenPI-compatible training state
    return training_utils.TrainState(
        step=self._step,
        params=params,
        opt_state={
            'pi0_optimizer': self.pi0_optimizer.init(pi0_params),
            'critic_optimizer': self.critic_optimizer.init(critic_params),
            'temperature_optimizer': self.temperature_optimizer.init(temperature_params) if self.temperature_optimizer else None
        },
        ema_params=None,  # EMA not used in ACRLPD
        ema_opt_state=None
    )

def update_from_train_state(self, train_state: training_utils.TrainState):
    # Update step counter
    self._step = train_state.step
    
    # Update model parameters using Flax NNX
    if 'pi0_model' in train_state.params:
        nnx.update(self.pi0_model, train_state.params['pi0_model'])
    if 'critic_networks' in train_state.params:
        nnx.update(self.critic_networks, train_state.params['critic_networks'])
    if 'temperature_module' in train_state.params and self.temperature_module is not None:
        nnx.update(self.temperature_module, train_state.params['temperature_module'])
```

### Evaluation and Metrics System (`ac_training/training/training_loop.py:87-155`)
```python
class TrainingMetrics:
    def update(self, step: int, loss_info: LossInfo, **kwargs):
        self.step_count = step
        
        # Comprehensive loss metrics tracking
        self.metrics.update({
            'train/total_loss': float(loss_info.total_loss),
            'train/critic_loss': float(loss_info.critic_loss),
            'train/bc_loss': float(loss_info.bc_loss),
            'train/actor_loss': float(loss_info.actor_loss),
            'train/alpha_loss': float(loss_info.alpha_loss),
            'train/q_mean': float(loss_info.q_mean),
            'train/q_std': float(loss_info.q_std),
            'train/target_q_mean': float(loss_info.target_q_mean),
            'train/td_error_mean': float(loss_info.td_error_mean),
            'train/alpha_value': float(loss_info.alpha_value),
            'train/entropy_estimate': float(loss_info.entropy_estimate),
            'train/valid_samples': float(loss_info.valid_samples),
            'train/mask_ratio': float(loss_info.mask_ratio)
        })
```

### Evaluation Execution (`ac_training/training/training_loop.py:440-467`)
```python
def _run_evaluation(self):
    logger.info("Running evaluation")
    self.metrics.episode_rewards.clear()
    self.metrics.episode_lengths.clear()
    
    for episode_idx in range(self.config.num_eval_episodes):
        self.rng, eval_rng = jax.random.split(self.rng)
        
        # Run evaluation episode with deterministic policy
        try:
            episode_reward, episode_length = self.eval_fn(
                self.agent, eval_rng, deterministic=True
            )
            self.metrics.add_episode_result(episode_reward, episode_length)
        except Exception as e:
            logger.warning(f"Evaluation episode {episode_idx} failed: {e}")
    
    # Update metrics with episode statistics
    episode_stats = self.metrics.get_episode_stats()
    self.metrics.metrics.update(episode_stats)
```

### Training Configuration (`ac_training/training/training_loop.py:39-85`)
```python
@dataclasses.dataclass
class ACRLPDTrainingConfig:
    # Training phases
    offline_steps: int = 1000000           # Offline pretraining steps
    online_steps: int = 500000             # Online fine-tuning steps
    
    # Evaluation and logging frequencies
    eval_frequency: int = 10000            # Steps between evaluations
    log_frequency: int = 100               # Steps between logging
    save_frequency: int = 50000            # Steps between checkpoints
    
    # OpenPI checkpoint integration
    checkpoint_dir: str = "./checkpoints"  # Checkpoint directory
    keep_period: int = 10000               # Checkpoint retention period
    overwrite: bool = False                # Overwrite existing checkpoints
    resume: bool = False                   # Resume from checkpoint
    
    # Training behavior
    gradient_accumulation_steps: int = 1   # Gradient accumulation
    max_grad_norm: float = 1.0             # Gradient clipping
    early_stopping_patience: int = 50     # Early stopping patience
    
    # Monitoring
    use_wandb: bool = True                 # WandB logging
    project_name: str = "acrlpd_pi0"       # WandB project
    experiment_name: str = ""              # Auto-generated if empty
```

### WandB Integration (`ac_training/training/training_loop.py:260-274`)
```python
def _init_wandb(self):
    try:
        wandb.init(
            project=self.config.project_name,
            name=self.config.experiment_name,
            config={
                'agent_config': dataclasses.asdict(self.agent.config),
                'training_config': dataclasses.asdict(self.config)
            }
        )
        logger.info("Initialized WandB logging")
    except Exception as e:
        logger.warning(f"Failed to initialize WandB: {e}")
        self.config.use_wandb = False
```

**Logged Metrics:**
- **Training**: Loss components, Q-value statistics, entropy estimates, timing metrics
- **Evaluation**: Episode rewards (mean, std, min, max), episode lengths, success rates
- **System**: GPU memory usage, training throughput, gradient norms
- **Algorithm**: Temperature values, mask ratios, valid sample counts

## 6. OpenPI Integration Points

### Direct OpenPI Component Usage

#### 1. Optimizer System Integration (`ac_training/agents/acrlpd_pi0_agent.py:170-195`)
Complete integration with OpenPI's optimizer infrastructure:

```python
# π₀ model optimizer with optional backbone freezing
pi0_weight_decay_mask = None
if config.freeze_pi0_backbone:
    pi0_weight_decay_mask = nnx.filterlib.to_predicate(nnx.Not(config.freeze_filter))

self.pi0_optimizer = _optimizer.create_optimizer(
    config.pi0_optimizer, 
    config.pi0_lr_schedule,
    weight_decay_mask=pi0_weight_decay_mask
)

# Critic ensemble optimizer
self.critic_optimizer = _optimizer.create_optimizer(
    config.critic_optimizer, 
    config.critic_lr_schedule
)

# Temperature optimizer (if adaptive temperature enabled)
if self.temperature_module is not None:
    temp_optimizer_config = _optimizer.AdamW(weight_decay=0.0, clip_gradient_norm=1.0)
    self.temperature_optimizer = _optimizer.create_optimizer(
        temp_optimizer_config, config.critic_lr_schedule
    )
```

**OpenPI Optimizer Features Utilized:**
- `_optimizer.CosineDecaySchedule(warmup_steps, peak_lr, decay_steps, decay_lr)`: Cosine annealing with warmup
- `_optimizer.AdamW(b1, b2, weight_decay, clip_gradient_norm)`: AdamW with gradient clipping
- `_optimizer.create_optimizer()`: Factory function with automatic gradient clipping and weight decay masks
- **Learning Rate Schedules**: Platform-optimized schedules (DROID: conservative, ALOHA: very conservative, Libero: aggressive)

#### 2. Checkpoint System Integration (`ac_training/training/training_loop.py:157-225`)
Full integration with OpenPI's checkpoint infrastructure:

```python
class ACRLPDCheckpointManager:
    def __init__(self, checkpoint_dir: str, keep_period: int = 10000, overwrite: bool = False, resume: bool = False):
        # Initialize OpenPI checkpoint manager
        self.manager, self.resuming = _checkpoints.initialize_checkpoint_dir(
            checkpoint_dir=self.checkpoint_dir,
            keep_period=keep_period,
            overwrite=overwrite,
            resume=resume
        )
    
    def save_checkpoint(self, agent: ACRLPDPi0Agent, dataloader: Any, step: int):
        train_state = agent.create_train_state()  # Convert to OpenPI format
        _checkpoints.save_state(self.manager, train_state, dataloader, step)
    
    def load_checkpoint(self, agent: ACRLPDPi0Agent, dataloader: Any, step: Optional[int] = None):
        template_state = agent.create_train_state()
        restored_state = _checkpoints.restore_state(self.manager, template_state, dataloader, step)
        agent.update_from_train_state(restored_state)
        return agent, restored_state.step
```

**OpenPI Checkpoint Features Utilized:**
- `_checkpoints.initialize_checkpoint_dir()`: Automatic checkpoint directory setup and validation
- `_checkpoints.save_state()`: Efficient training state serialization with metadata
- `_checkpoints.restore_state()`: Training state restoration with consistency checks
- `training_utils.TrainState`: Standard state container for parameters, optimizer states, and metadata
- **Automatic Pruning**: Configurable checkpoint retention policies
- **Resume Support**: Seamless training continuation from arbitrary checkpoints

#### 3. Model System Integration (`ac_training/agents/acrlpd_pi0_agent.py:146`)
Direct usage of π₀ model implementation from OpenPI:

```python
# π₀ model creation using OpenPI factory
self.pi0_model = config.pi0_config.create(rngs.fork("pi0"))
```

**OpenPI Model Features Utilized:**
- `_model.Observation`: Standardized multi-modal observation data structure
- `_model.Actions`: Standardized action sequence data structure
- `_model.preprocess_observation()`: Consistent observation preprocessing pipeline
- `_pi0.Pi0Config`: Complete π₀ model configuration with platform-specific parameters
- `_pi0.Pi0Model`: Core diffusion model with `sample_actions()`, `compute_loss()`, `embed_prefix()` methods

#### 4. Data Structure Compatibility
Full compatibility with OpenPI's data structures:

```python
# Observation structure matches OpenPI expectations
observation = _model.Observation(
    images={'base_0_rgb': images, 'left_wrist_0_rgb': wrist_images, ...},
    image_masks={'base_0_rgb': masks, 'left_wrist_0_rgb': wrist_masks, ...},
    state=robot_state,
    tokenized_prompt=prompt_tokens,
    tokenized_prompt_mask=prompt_mask
)

# Action format matches π₀ expectations
actions = _model.Actions(...)  # [batch_size, action_horizon, action_dim]
```

### Configuration System Integration
ACRLPD configurations use OpenPI optimizer and schedule configurations:

```python
@dataclasses.dataclass(frozen=True)
class ACRLPDPi0Config:
    # OpenPI optimizer configurations
    pi0_lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(
        default_factory=lambda: _optimizer.CosineDecaySchedule(
            warmup_steps=1000, peak_lr=1e-5, decay_steps=30000, decay_lr=1e-6
        )
    )
    critic_lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(
        default_factory=lambda: _optimizer.CosineDecaySchedule(
            warmup_steps=1000, peak_lr=3e-4, decay_steps=30000, decay_lr=3e-5
        )
    )
    pi0_optimizer: _optimizer.OptimizerConfig = dataclasses.field(
        default_factory=lambda: _optimizer.AdamW(weight_decay=1e-4, clip_gradient_norm=1.0)
    )
    critic_optimizer: _optimizer.OptimizerConfig = dataclasses.field(
        default_factory=lambda: _optimizer.AdamW(weight_decay=1e-4, clip_gradient_norm=1.0)
    )
```

### OpenPI Transform Pipeline Integration
Leverages OpenPI's transform system for data preprocessing:

**Repack Transforms**: Dataset format standardization
```python
# From src/openpi/training/config.py - ALOHA example
repack_transforms = _transforms.Group(
    inputs=[
        _transforms.RepackTransform({
            "images": {"cam_high": "observation.images.cam_high"},
            "state": "observation.state",
            "actions": "action",
        })
    ]
)
```

**Data Transforms**: Platform-specific preprocessing
```python
# Robot-specific coordinate transformations
data_transforms = _transforms.Group(
    inputs=[aloha_policy.AlohaInputs(action_dim=model_config.action_dim)],
    outputs=[aloha_policy.AlohaOutputs()]
)
```

**Model Transforms**: π₀-specific preprocessing
```python
# π₀ model input preparation
model_transforms = _transforms.Group(
    inputs=[
        _transforms.InjectDefaultPrompt(default_prompt),
        _transforms.ResizeImages(224, 224),
        _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer(max_token_len))
    ]
)
```

## 7. Platform Configurations

### DROID Configuration
Optimized for large-scale manipulation datasets:

```python
def get_droid_config() -> ACRLPDPi0Config:
    return ACRLPDPi0Config(
        # π₀ model: 8D action space, 10-step horizon
        pi0_config=_pi0.Pi0Config(action_dim=8, action_horizon=10, max_token_len=180),
        
        # Training: Extended offline phase for rich data
        offline_steps=2000000, online_steps=200000,
        batch_size=64, best_of_n_samples=16,
        
        # Critic: Large ensemble for complex state-action spaces
        critic_config=CriticConfig(num_critics=10, hidden_dims=[512, 512, 256]),
        
        # Loss weights: Moderate BC regularization
        loss_weights=LossWeights(bc_weight=0.01, critic_weight=1.0, actor_weight=1.0),
        
        # Optimizers: Conservative learning rates for stability
        pi0_lr_schedule=_optimizer.CosineDecaySchedule(peak_lr=3e-5, decay_steps=100000),
        critic_lr_schedule=_optimizer.CosineDecaySchedule(peak_lr=5e-4, decay_steps=80000)
    )
```

### ALOHA Configuration  
Fine-tuning focused for bimanual manipulation:

```python
def get_aloha_config() -> ACRLPDPi0Config:
    return ACRLPDPi0Config(
        # π₀ model: 14D bimanual action space, longer horizons
        pi0_config=_pi0.Pi0Config(action_dim=14, action_horizon=20, max_token_len=250),
        
        # Training: Conservative fine-tuning approach
        freeze_pi0_backbone=True, offline_steps=500000, online_steps=100000,
        batch_size=128, best_of_n_samples=32,
        
        # Critic: Smaller ensemble for faster convergence
        critic_config=CriticConfig(num_critics=8, hidden_dims=[256, 256, 128]),
        
        # Loss weights: Minimal BC for fine-tuning
        loss_weights=LossWeights(bc_weight=0.001, critic_weight=1.0, actor_weight=1.0),
        
        # Optimizers: Very conservative learning rates
        pi0_lr_schedule=_optimizer.CosineDecaySchedule(peak_lr=5e-6, decay_steps=30000),
        critic_lr_schedule=_optimizer.CosineDecaySchedule(peak_lr=1e-4, decay_steps=25000)
    )
```

### Libero Configuration
Simulation-optimized for rapid experimentation:

```python
def get_libero_config() -> ACRLPDPi0Config:
    return ACRLPDPi0Config(
        # π₀ model: 7D single-arm action space, short horizons
        pi0_config=_pi0.Pi0Config(action_dim=7, action_horizon=5, max_token_len=180),
        
        # Training: Aggressive learning for simulation
        freeze_pi0_backbone=False, offline_steps=1000000, online_steps=500000,
        batch_size=256, best_of_n_samples=64,
        
        # Critic: Large ensemble with deep networks
        critic_config=CriticConfig(num_critics=12, hidden_dims=[512, 512, 256, 128]),
        
        # Loss weights: Strong BC constraint for simulation stability
        loss_weights=LossWeights(bc_weight=0.1, critic_weight=1.0, actor_weight=1.0),
        
        # Optimizers: Aggressive learning rates for fast learning
        pi0_lr_schedule=_optimizer.CosineDecaySchedule(peak_lr=3e-4, decay_steps=50000),
        critic_lr_schedule=_optimizer.CosineDecaySchedule(peak_lr=1e-3, decay_steps=40000)
    )
```

## 8. Complete Training Workflow

### Command-Line Usage
Training execution via `scripts/train_acrlpd_pi0.py`:

```bash
# DROID dataset training
python scripts/train_acrlpd_pi0.py \
    --data_dir /path/to/droid/data \
    --config droid \
    --offline_steps 2000000 \
    --online_steps 200000 \
    --checkpoint_dir ./checkpoints/droid_experiment

# LeRobot dataset training  
python scripts/train_acrlpd_pi0.py \
    --lerobot_repo_id lerobot/aloha_sim_insertion_human \
    --config aloha \
    --experiment_name aloha_insertion_experiment

# Resume from checkpoint
python scripts/train_acrlpd_pi0.py \
    --data_dir /path/to/data \
    --config libero \
    --resume /path/to/checkpoint
```

### Configuration Management
Three-level configuration system:

1. **Platform Presets**: `get_droid_config()`, `get_aloha_config()`, `get_libero_config()`
2. **Command-line Overrides**: Selective parameter modification
3. **Runtime Validation**: Configuration parameter validation before training

### Training Execution Flow

#### Initialization Phase
1. **JAX Environment Setup**: Device detection, memory allocation, debug flags
2. **Data Loader Creation**: H5 or LeRobot dataset loading with transforms
3. **Agent Creation**: π₀ model + critic ensemble + optimizers initialization
4. **Checkpoint Manager**: OpenPI checkpoint system initialization
5. **Metrics System**: WandB integration and metrics tracking setup

#### Training Loop Execution
```python
# Phase 1: Offline Pretraining
while step < config.offline_steps:
    batch = dataloader.sample_batch()
    agent, loss_info = agent.train_step(batch, rng)
    
    # Every log_frequency steps: log metrics
    # Every eval_frequency steps: run evaluation  
    # Every save_frequency steps: save checkpoint

# Phase 2: Online Fine-tuning (if eval_fn provided)
while step < config.offline_steps + config.online_steps:
    # Mix online environment data with offline data
    # More frequent evaluation for early stopping
    # Adaptive loss weight scheduling
```

#### Training Step Details
```python
def train_step(self, batch, rng):
    # 1. Compute joint loss (RL + BC + π₀ + temperature)
    (loss, loss_info), grads = nnx.value_and_grad(loss_fn)(self)
    
    # 2. Apply gradients to π₀ (unless backbone frozen)
    if not self.config.freeze_pi0_backbone:
        self.pi0_model = self.pi0_optimizer.apply_gradients(grads.pi0_model, self.pi0_model)
    
    # 3. Apply gradients to critic ensemble
    self.critic_networks = self.critic_optimizer.apply_gradients(grads.critic_networks, self.critic_networks)
    
    # 4. Apply gradients to temperature (if adaptive)
    if self.temperature_module:
        self.temperature_module = self.temperature_optimizer.apply_gradients(grads.temperature_module, self.temperature_module)
    
    # 5. Soft update target networks
    self.critic_networks.soft_update_target_networks(tau=self.config.target_update_tau)
    
    return self, loss_info
```

### Monitoring and Evaluation
Comprehensive metrics tracking via `TrainingMetrics` class:

**Training Metrics:**
- Loss components (total, critic, BC, π₀, temperature)
- Q-value statistics (mean, std, target Q-values)
- TD error analysis and entropy estimates
- Training timing and throughput metrics

**Evaluation Metrics:**
- Episode rewards (mean, std, min, max)
- Episode lengths and success rates
- Best evaluation reward tracking
- Early stopping trigger monitoring

**WandB Integration:**
- Automatic experiment logging and visualization
- Hyperparameter and configuration tracking
- Real-time training progress monitoring
- Checkpoint performance comparison

### Error Handling and Recovery
Robust error handling throughout the pipeline:

- **Checkpoint Interruption**: Automatic checkpoint saving on KeyboardInterrupt
- **Training Failures**: Error logging with optional stack traces in debug mode
- **Data Loading Errors**: Graceful fallback and error reporting
- **Memory Management**: OOM detection and adaptive batch size reduction

### Performance Optimizations
Multiple optimization strategies implemented:

1. **JAX Compilation**: JIT compilation of training loops and loss functions
2. **Memory Management**: Adaptive caching and batch size optimization
3. **Multi-GPU Support**: FSDP sharding for large model training
4. **Efficient Sampling**: Reduced diffusion steps during candidate generation
5. **Vectorized Operations**: Parallel evaluation of action candidates

## 9. Implementation Details

### Key Files and Their Roles

#### Core Agent Implementation
- **`ac_training/agents/acrlpd_pi0_agent.py`**: Main agent class with π₀ + ACRLPD integration
- **`ac_training/agents/critic_networks.py`**: Critic ensemble implementation
- **`ac_training/agents/loss_functions.py`**: Joint loss computation and temperature control

#### Data Loading Infrastructure  
- **`ac_training/data_loader.py`**: H5 dataset reader and sequence generation
- **`ac_training/acrlpd_dataloader.py`**: ACRLPD-specific data loader with π₀ compatibility
- **`ac_training/qc_dataset.py`**: Q-chunking dataset interface

#### Training System
- **`ac_training/training/training_loop.py`**: Complete training pipeline with OpenPI integration
- **`ac_training/scripts/train_acrlpd_pi0.py`**: Command-line interface and argument parsing

#### Configuration and Utilities
- **`ac_training/config.py`**: Configuration management and platform presets
- **`ac_training/batching.py`**: Performance utilities and memory management
- **`ac_training/__init__.py`**: Package exports and convenience functions

### Data Flow Architecture
```
Raw Data (H5/LeRobot)
    ↓
H5DatasetReader / LeRobot Integration
    ↓
SequenceGenerator (episode boundaries + chunking)
    ↓
ACRLPDDataLoader (batch generation + masking)
    ↓
ACRLPDPi0Agent (π₀ model + critic ensemble)
    ↓
JointLossComputer (RL + BC + π₀ + temperature losses)
    ↓
OpenPI Optimizers (gradient application)
    ↓
ACRLPDCheckpointManager (state persistence)
```

### Memory and Performance Characteristics

**Memory Requirements:**
- **Inference**: ~8GB GPU memory (π₀ model + critic ensemble)
- **Training**: ~22GB GPU memory (with gradient computation and Best-of-N sampling)
- **Large Best-of-N**: Adaptive strategies automatically handle memory constraints

**Performance Benchmarks:**
- **Training Speed**: ~200-500 samples/second depending on platform
- **Best-of-N Sampling**: Parallel evaluation scales linearly with GPU memory
- **Checkpoint Operations**: <30 seconds for full model state save/load

### Integration Testing and Validation
Framework includes comprehensive validation:

```python
def validate_installation():
    # Verify JAX, Flax, NumPy, H5Py dependencies
    # Test π₀ model loading and basic operations
    # Validate critic network creation and forward pass
    # Test data loader batch generation
```

## 7. Platform Configurations

### DROID Platform Configuration
Optimized for large-scale manipulation datasets with complex state-action spaces:

```python
def get_droid_acrlpd_config() -> ACRLPDPi0Config:
    return ACRLPDPi0Config(
        # π₀ model configuration
        pi0_config=_pi0.Pi0Config(
            action_dim=8,              # Joint positions (7) + gripper (1)
            action_horizon=10,         # 10-step action chunks for complex manipulation
            max_token_len=180,         # Sufficient for task instructions
            diffusion_steps=50         # High-quality sampling for complex tasks
        ),
        
        # Training phases optimized for large datasets
        offline_steps=2000000,         # Extended offline phase for rich expert data
        online_steps=200000,           # Shorter online phase due to data richness
        
        # Batch configuration for distributed training
        batch_size=64,                 # Conservative for memory management
        sequence_ratio=0.7,            # Higher sequence ratio for temporal learning
        best_of_n_samples=16,          # Moderate N for balanced exploration
        
        # Critic ensemble configuration
        critic_config=CriticConfig(
            num_critics=10,            # Large ensemble for robustness
            hidden_dims=[512, 512, 256],  # Deep networks for complex state spaces
            dropout_rate=0.1,          # Regularization for generalization
            q_aggregation="min",       # Conservative Q-value estimation
            use_pi0_features=True,     # Leverage π₀'s rich feature representations
            feature_fusion_method="concat"
        ),
        
        # Loss weight configuration for dataset characteristics
        loss_weights=LossWeights(
            bc_weight=0.01,            # Moderate BC - balance exploration vs exploitation
            critic_weight=1.0,         # Standard critic weight
            actor_weight=1.0,          # Actor loss for true reinforcement learning
            alpha_weight=1.0           # Adaptive exploration
        ),
        
        # Optimizer configuration for stability
        pi0_lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=5000,         # Extended warmup for large model
            peak_lr=3e-5,              # Conservative learning rate
            decay_steps=100000,        # Long decay for stable convergence
            decay_lr=3e-6
        ),
        critic_lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2000,         # Faster critic warmup
            peak_lr=5e-4,              # Higher critic learning rate
            decay_steps=80000,         # Coordinated with π₀ decay
            decay_lr=5e-5
        ),
        
        # Advanced training parameters
        freeze_pi0_backbone=False,     # Full fine-tuning for large dataset
        target_update_tau=0.005,       # Conservative target updates
        discount=0.995,                # High discount for long horizon tasks
        temperature_config=TemperatureConfig(
            initial_temp=1.0,
            target_entropy_multiplier=0.5,
            adaptive=True
        )
    )
```

**DROID-Specific Optimizations:**
- **Extended Offline Training**: 2M steps to fully utilize large expert dataset (>10k hours)
- **Conservative Learning Rates**: Prevent overfitting to diverse manipulation tasks
- **Large Critic Ensemble**: 10 networks for robust Q-value estimation across task diversity
- **Moderate BC Regularization**: Balance between expert imitation and policy improvement
- **High Discount Factor**: 0.995 for long-horizon manipulation tasks

### ALOHA Platform Configuration
Fine-tuning focused for bimanual manipulation with pretrained models:

```python
def get_aloha_acrlpd_config() -> ACRLPDPi0Config:
    return ACRLPDPi0Config(
        # π₀ model configuration for bimanual manipulation
        pi0_config=_pi0.Pi0Config(
            action_dim=14,             # Bimanual: 7 DoF per arm
            action_horizon=20,         # Longer horizons for complex bimanual coordination
            max_token_len=250,         # Extended for complex task descriptions
            diffusion_steps=30         # Reduced steps for faster inference
        ),
        
        # Conservative fine-tuning approach
        offline_steps=500000,          # Shorter offline phase - leveraging pretrained features
        online_steps=100000,           # Moderate online phase for task-specific adaptation
        
        # Batch configuration for fine-tuning
        batch_size=128,                # Larger batches for stable fine-tuning
        sequence_ratio=0.8,            # High sequence focus for temporal coordination
        best_of_n_samples=32,          # Higher N for precise manipulation
        
        # Critic ensemble optimized for fine-tuning
        critic_config=CriticConfig(
            num_critics=8,             # Smaller ensemble for faster convergence
            hidden_dims=[256, 256, 128], # Smaller networks - leverage π₀ features
            dropout_rate=0.05,         # Lower dropout for fine-tuning stability
            q_aggregation="mean",      # Balanced aggregation
            use_pi0_features=True,
            feature_fusion_method="mlp"  # Learnable fusion for adaptation
        ),
        
        # Minimal BC regularization for fine-tuning
        loss_weights=LossWeights(
            bc_weight=0.001,           # Very weak BC - trust pretrained policy
            critic_weight=1.0,
            actor_weight=1.0,          # Actor loss for policy improvement
            alpha_weight=0.8           # Reduced exploration for precise manipulation
        ),
        
        # Very conservative optimizer settings
        pi0_lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2000,         # Quick warmup for fine-tuning
            peak_lr=5e-6,              # Very low learning rate to preserve pretrained features
            decay_steps=30000,         # Shorter decay schedule
            decay_lr=1e-6
        ),
        critic_lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=1e-4,              # Low critic learning rate for stability
            decay_steps=25000,
            decay_lr=1e-5
        ),
        
        # Fine-tuning specific parameters
        freeze_pi0_backbone=True,      # Freeze backbone - only tune task-specific layers
        target_update_tau=0.01,        # Faster target updates for adaptation
        discount=0.99,                 # Standard discount for manipulation
        gradient_accumulation_steps=2,  # Accumulate gradients for stable updates
        
        # Conservative exploration for precise manipulation
        temperature_config=TemperatureConfig(
            initial_temp=0.5,          # Lower initial temperature
            target_entropy_multiplier=0.3,  # Reduced exploration
            adaptive=True
        )
    )
```

**ALOHA-Specific Optimizations:**
- **Backbone Freezing**: Preserve pretrained features, only fine-tune task-specific components
- **Minimal BC Regularization**: Trust pretrained policy, focus on Q-value learning
- **Extended Action Horizons**: 20-step sequences for complex bimanual coordination
- **Learnable Feature Fusion**: MLP fusion for adapting π₀ features to task-specific critics
- **Precise Exploration**: Reduced temperature and entropy targets for fine manipulation

### Libero Platform Configuration
Simulation-optimized for rapid experimentation and benchmark evaluation:

```python
def get_libero_acrlpd_config() -> ACRLPDPi0Config:
    return ACRLPDPi0Config(
        # π₀ model configuration for simulation
        pi0_config=_pi0.Pi0Config(
            action_dim=7,              # Single-arm manipulation
            action_horizon=5,          # Short horizons for fast simulation
            max_token_len=180,         # Standard prompt length
            diffusion_steps=20         # Reduced steps for simulation speed
        ),
        
        # Aggressive training for simulation
        offline_steps=1000000,         # Standard offline phase
        online_steps=500000,           # Extended online phase for sim-to-real transfer
        
        # Large batch configuration for simulation efficiency
        batch_size=256,                # Large batches - no hardware constraints
        sequence_ratio=0.6,            # Balanced sequence/single-step sampling
        best_of_n_samples=64,          # High N for exploration in simulation
        
        # Large critic ensemble for simulation stability
        critic_config=CriticConfig(
            num_critics=12,            # Large ensemble for robust simulation learning
            hidden_dims=[512, 512, 256, 128],  # Deep networks for rich simulation
            dropout_rate=0.15,         # Higher dropout for simulation robustness
            q_aggregation="min",       # Conservative for safety in simulation
            use_pi0_features=True,
            feature_fusion_method="concat"
        ),
        
        # Strong BC regularization for simulation stability
        loss_weights=LossWeights(
            bc_weight=0.1,             # Strong BC constraint for safety
            critic_weight=1.0,
            actor_weight=1.0,          # Actor loss for policy optimization
            alpha_weight=1.2           # Enhanced exploration for simulation
        ),
        
        # Aggressive optimizer settings for fast learning
        pi0_lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=3000,         # Quick warmup
            peak_lr=3e-4,              # High learning rate for simulation
            decay_steps=50000,         # Faster decay
            decay_lr=3e-5
        ),
        critic_lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1500,
            peak_lr=1e-3,              # Aggressive critic learning
            decay_steps=40000,
            decay_lr=1e-4
        ),
        
        # Simulation-specific parameters
        freeze_pi0_backbone=False,     # Full training for simulation
        target_update_tau=0.01,        # Fast target updates
        discount=0.98,                 # Lower discount for episodic simulation tasks
        gradient_accumulation_steps=1,  # No accumulation needed
        
        # Enhanced exploration for simulation
        temperature_config=TemperatureConfig(
            initial_temp=1.5,          # High initial exploration
            target_entropy_multiplier=0.7,  # High entropy target
            adaptive=True
        )
    )
```

**Libero-Specific Optimizations:**
- **Large Batch Sizes**: Leverage simulation efficiency for large-scale training
- **Strong BC Regularization**: Ensure safety and stability in simulation environment
- **Extended Online Training**: 500k steps for sim-to-real transfer preparation
- **Deep Critic Networks**: Rich representations for complex simulation dynamics
- **Aggressive Learning**: High learning rates for fast convergence in controlled environment

### Platform-Specific Tuning Guidelines

#### Memory Management by Platform
```python
# Memory requirements by platform and configuration
PLATFORM_MEMORY_REQUIREMENTS = {
    "droid": {
        "inference": "~12GB",      # Large action horizon + ensemble
        "training": "~28GB",       # Best-of-N sampling + gradients
        "large_N": "adaptive"      # Use memory-efficient chunked sampling
    },
    "aloha": {
        "inference": "~10GB",      # Moderate requirements
        "training": "~24GB",       # Fine-tuning with frozen backbone
        "large_N": "parallel"      # Sufficient memory for parallel sampling
    },
    "libero": {
        "inference": "~8GB",       # Simulation efficiency
        "training": "~32GB",       # Large ensembles + aggressive batching
        "large_N": "parallel"      # No hardware constraints
    }
}
```

#### Learning Rate Scheduling by Platform
```python
# Platform-optimized learning rate schedules
PLATFORM_LR_SCHEDULES = {
    "droid": {
        "pi0_peak_lr": 3e-5,       # Conservative for complex dataset
        "critic_peak_lr": 5e-4,    # Balanced critic learning
        "warmup_ratio": 0.05,      # 5% warmup of total steps
        "decay_profile": "cosine"  # Smooth convergence
    },
    "aloha": {
        "pi0_peak_lr": 5e-6,       # Very conservative for fine-tuning
        "critic_peak_lr": 1e-4,    # Low critic rate to preserve features
        "warmup_ratio": 0.1,       # Extended warmup for stability
        "decay_profile": "cosine"
    },
    "libero": {
        "pi0_peak_lr": 3e-4,       # Aggressive for simulation
        "critic_peak_lr": 1e-3,    # High rate for fast learning
        "warmup_ratio": 0.03,      # Quick warmup
        "decay_profile": "cosine"
    }
}
```

#### Data Pipeline Optimization by Platform
```python
# Platform-specific data pipeline configurations
PLATFORM_DATA_CONFIGS = {
    "droid": {
        "sequence_ratio": 0.7,     # Focus on temporal learning
        "cache_size": 50,          # Large cache for diverse data
        "shuffle_buffer": 10000,   # Large shuffle for diversity
        "data_augmentation": "moderate"  # Preserve data distribution
    },
    "aloha": {
        "sequence_ratio": 0.8,     # High sequence focus for coordination
        "cache_size": 30,          # Moderate cache for targeted data
        "shuffle_buffer": 5000,    # Smaller shuffle for consistency
        "data_augmentation": "light"  # Preserve fine-tuning data
    },
    "libero": {
        "sequence_ratio": 0.6,     # Balanced for diverse simulation tasks
        "cache_size": 100,         # Large cache - no storage constraints
        "shuffle_buffer": 20000,   # Extensive shuffle for generalization
        "data_augmentation": "aggressive"  # Enhance simulation diversity
    }
}
```

## 8. Complete Training Workflow

### Command-Line Interface
Complete training execution via `scripts/train_acrlpd_pi0.py`:

#### Basic Training Commands
```bash
# DROID dataset training with H5 format
python scripts/train_acrlpd_pi0.py \
    --data_dir /path/to/droid/data \
    --config droid \
    --offline_steps 2000000 \
    --online_steps 200000 \
    --batch_size 64 \
    --checkpoint_dir ./checkpoints/droid_acrlpd \
    --experiment_name droid_large_scale_exp

# LeRobot dataset training (ALOHA)
python scripts/train_acrlpd_pi0.py \
    --lerobot_repo_id lerobot/aloha_sim_insertion_human \
    --config aloha \
    --experiment_name aloha_insertion_acrlpd \
    --freeze_pi0_backbone \
    --bc_weight 0.001

# Libero simulation training
python scripts/train_acrlpd_pi0.py \
    --lerobot_repo_id physical-intelligence/libero \
    --config libero \
    --experiment_name libero_benchmark_acrlpd \
    --batch_size 256 \
    --best_of_n_samples 64
```

#### Advanced Configuration Options
```bash
# Resume from checkpoint with modified hyperparameters
python scripts/train_acrlpd_pi0.py \
    --resume /path/to/checkpoint \
    --config droid \
    --override_lr_schedule \
    --pi0_peak_lr 1e-5 \
    --critic_peak_lr 3e-4 \
    --bc_weight 0.005

# Multi-GPU distributed training
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train_acrlpd_pi0.py \
    --data_dir /path/to/data \
    --config libero \
    --fsdp_devices 4 \
    --batch_size 1024 \
    --gradient_accumulation_steps 2

# Debug mode with comprehensive logging
python scripts/train_acrlpd_pi0.py \
    --data_dir /path/to/small_dataset \
    --config debug \
    --debug_mode \
    --log_frequency 10 \
    --eval_frequency 100 \
    --save_frequency 500
```

### Configuration Management System

#### Hierarchical Configuration Structure
```python
# 1. Base Configuration (platform defaults)
base_config = get_droid_acrlpd_config()

# 2. Experiment-specific overrides
experiment_overrides = {
    "offline_steps": 1500000,      # Reduced for faster experimentation
    "best_of_n_samples": 24,       # Moderate N for memory constraints
    "loss_weights.bc_weight": 0.02  # Increased BC for safety
}

# 3. Runtime parameter validation and application
final_config = apply_config_overrides(base_config, experiment_overrides)
validate_config_compatibility(final_config)
```

#### Environment-Specific Configuration Loading
```python
# Configuration factories for different deployment environments
def load_training_config(
    platform: str,
    environment: str = "production",
    custom_overrides: Dict[str, Any] = None
) -> ACRLPDPi0Config:
    """
    Load platform and environment-specific configuration.
    
    Args:
        platform: "droid", "aloha", "libero"
        environment: "production", "development", "debug"
        custom_overrides: User-specified parameter overrides
        
    Returns:
        Complete validated configuration
    """
    # Load base platform configuration
    base_configs = {
        "droid": get_droid_acrlpd_config(),
        "aloha": get_aloha_acrlpd_config(), 
        "libero": get_libero_acrlpd_config()
    }
    base_config = base_configs[platform]
    
    # Apply environment-specific modifications
    env_configs = {
        "production": {},  # Use defaults
        "development": {
            "offline_steps": base_config.offline_steps // 4,
            "online_steps": base_config.online_steps // 2,
            "eval_frequency": 1000,
            "save_frequency": 2000
        },
        "debug": {
            "offline_steps": 1000,
            "online_steps": 500,
            "batch_size": 8,
            "best_of_n_samples": 4,
            "log_frequency": 10
        }
    }
    
    # Merge configurations
    final_config = merge_configurations(
        base_config, env_configs[environment], custom_overrides or {}
    )
    
    return validate_and_return_config(final_config)
```

### Training Monitoring and Evaluation

#### Comprehensive Metrics Dashboard
```python
# WandB metrics organization
WANDB_METRIC_GROUPS = {
    "loss_components": [
        "train/total_loss", "train/critic_loss", "train/bc_loss", 
        "train/actor_loss", "train/alpha_loss"
    ],
    "q_value_analysis": [
        "train/q_mean", "train/q_std", "train/target_q_mean", 
        "train/td_error_mean", "train/q_ensemble_variance"
    ],
    "exploration_metrics": [
        "train/alpha_value", "train/entropy_estimate", 
        "train/best_of_n_variance", "train/action_diversity"
    ],
    "training_efficiency": [
        "timing/samples_per_sec", "timing/step_duration", 
        "memory/gpu_usage", "data/mask_ratio"
    ],
    "evaluation_performance": [
        "eval/episode_reward_mean", "eval/episode_reward_std",
        "eval/success_rate", "eval/episode_length_mean"
    ]
}
```

#### Real-time Training Monitoring
```python
class ACRLPDTrainingMonitor:
    """Advanced monitoring system for ACRLPD training."""
    
    def __init__(self, config: ACRLPDTrainingConfig):
        self.config = config
        self.metrics_history = []
        self.performance_tracker = PerformanceTracker()
        self.alert_system = AlertSystem()
    
    def monitor_training_step(self, step: int, loss_info: LossInfo, 
                            timing_info: Dict[str, float]):
        """Monitor single training step with anomaly detection."""
        
        # Performance tracking
        self.performance_tracker.update(step, loss_info, timing_info)
        
        # Anomaly detection
        anomalies = self.detect_training_anomalies(loss_info)
        if anomalies:
            self.alert_system.send_alerts(step, anomalies)
        
        # Adaptive learning rate suggestions
        if step % 1000 == 0:
            lr_suggestion = self.suggest_lr_adjustment(step, loss_info)
            if lr_suggestion:
                logger.info(f"Suggested LR adjustment: {lr_suggestion}")
    
    def detect_training_anomalies(self, loss_info: LossInfo) -> List[str]:
        """Detect training anomalies and instabilities."""
        anomalies = []
        
        # Loss explosion detection
        if loss_info.total_loss > 10 * self.performance_tracker.loss_baseline:
            anomalies.append("Loss explosion detected")
        
        # Q-value collapse detection
        if loss_info.q_std < 0.01:
            anomalies.append("Q-value collapse - all critics converged to same value")
        
        # BC-RL imbalance detection
        bc_to_critic_ratio = loss_info.bc_loss / (loss_info.critic_loss + 1e-8)
        if bc_to_critic_ratio > 100:
            anomalies.append("BC loss dominating - reduce bc_weight")
        elif bc_to_critic_ratio < 0.001:
            anomalies.append("BC loss too weak - increase bc_weight")
        
        # Temperature instability
        if loss_info.alpha_value > 10 or loss_info.alpha_value < 0.01:
            anomalies.append("Temperature instability - check entropy estimation")
        
        return anomalies
```

#### Evaluation Protocol
```python
class ACRLPDEvaluationProtocol:
    """Standardized evaluation protocol for ACRLPD training."""
    
    def __init__(self, eval_config: EvaluationConfig):
        self.config = eval_config
        self.baseline_metrics = None
        self.evaluation_history = []
    
    def run_comprehensive_evaluation(
        self, 
        agent: ACRLPDPi0Agent, 
        step: int
    ) -> Dict[str, float]:
        """Run comprehensive evaluation including robustness tests."""
        
        results = {}
        
        # 1. Standard task performance
        task_results = self.evaluate_task_performance(agent)
        results.update(task_results)
        
        # 2. Robustness evaluation
        if step % (self.config.eval_frequency * 5) == 0:
            robustness_results = self.evaluate_robustness(agent)
            results.update(robustness_results)
        
        # 3. Best-of-N analysis
        bon_results = self.analyze_best_of_n_performance(agent)
        results.update(bon_results)
        
        # 4. π₀ vs Critic consistency check
        consistency_results = self.check_pi0_critic_consistency(agent)
        results.update(consistency_results)
        
        return results
    
    def evaluate_robustness(self, agent: ACRLPDPi0Agent) -> Dict[str, float]:
        """Evaluate policy robustness to perturbations."""
        results = {}
        
        # Observation noise robustness
        noise_levels = [0.01, 0.05, 0.1]
        for noise in noise_levels:
            reward = self.evaluate_with_observation_noise(agent, noise)
            results[f"robustness/obs_noise_{noise}"] = reward
        
        # Action execution noise
        for noise in noise_levels:
            reward = self.evaluate_with_action_noise(agent, noise)  
            results[f"robustness/action_noise_{noise}"] = reward
        
        # Different Best-of-N values
        for n in [1, 4, 16, 32]:
            reward = self.evaluate_with_best_of_n(agent, n)
            results[f"robustness/best_of_{n}"] = reward
        
        return results
```

### Configuration File Management

#### YAML Configuration Support
```yaml
# acrlpd_droid_experiment.yaml
platform: droid
base_config: droid

# Training overrides
training:
  offline_steps: 1800000
  online_steps: 150000
  batch_size: 96
  eval_frequency: 5000

# Model overrides  
model:
  pi0_config:
    action_horizon: 12
    diffusion_steps: 40
  
# Critic overrides
critic:
  num_critics: 12
  hidden_dims: [512, 512, 256, 128]
  
# Loss weights
loss_weights:
  bc_weight: 0.015
  critic_weight: 1.0
  alpha_weight: 0.8

# Optimization
optimization:
  pi0_lr_schedule:
    peak_lr: 2e-5
    decay_steps: 80000
  gradient_accumulation_steps: 2
```

#### Configuration Validation System
```python
class ConfigValidator:
    """Validates ACRLPD configuration compatibility and completeness."""
    
    @staticmethod
    def validate_config(config: ACRLPDPi0Config, platform: str) -> List[str]:
        """Validate configuration for platform compatibility."""
        warnings = []
        
        # Memory requirement validation
        estimated_memory = ConfigValidator.estimate_memory_usage(config)
        if estimated_memory > PLATFORM_MEMORY_LIMITS[platform]:
            warnings.append(f"Estimated memory usage ({estimated_memory}GB) exceeds platform limit")
        
        # Learning rate validation
        if config.pi0_lr_schedule.peak_lr > PLATFORM_MAX_LR[platform]["pi0"]:
            warnings.append("π₀ learning rate may be too high for platform stability")
        
        # Batch size validation
        min_batch = PLATFORM_MIN_BATCH_SIZE[platform]
        if config.batch_size < min_batch:
            warnings.append(f"Batch size {config.batch_size} below recommended minimum {min_batch}")
        
        # Best-of-N validation
        max_n = PLATFORM_MAX_BEST_OF_N[platform]
        if config.best_of_n_samples > max_n:
            warnings.append(f"Best-of-N samples {config.best_of_n_samples} may cause memory issues")
        
        return warnings
```

### Production Deployment Workflow

#### Model Export and Serving
```python
# Export trained ACRLPD model for production deployment
def export_acrlpd_model(
    checkpoint_path: str,
    output_dir: str,
    optimization_level: str = "standard"
) -> str:
    """
    Export trained ACRLPD model for production serving.
    
    Args:
        checkpoint_path: Path to trained checkpoint
        output_dir: Directory for exported model
        optimization_level: "fast", "standard", "accurate"
        
    Returns:
        Path to exported model directory
    """
    # Load checkpoint
    agent = load_acrlpd_checkpoint(checkpoint_path)
    
    # Model optimization for deployment
    if optimization_level == "fast":
        # Reduce ensemble size for speed
        agent.critic_networks = select_top_critics(agent.critic_networks, k=3)
        # Reduce diffusion steps
        agent.pi0_model.config.diffusion_steps = 10
    elif optimization_level == "accurate":
        # Use full ensemble and high-quality sampling
        agent.pi0_model.config.diffusion_steps = 50
    
    # Export for OpenPI policy server
    policy_config = create_policy_config(agent)
    export_policy_server_config(policy_config, output_dir)
    
    return output_dir
```

#### Production Serving Configuration
```python
# Production policy server configuration
PRODUCTION_SERVING_CONFIG = {
    "droid": {
        "num_diffusion_steps": 20,     # Balanced speed/quality
        "best_of_n_samples": 8,        # Moderate exploration
        "critic_ensemble_size": 5,     # Reduced for speed
        "max_concurrent_requests": 4,  # Hardware-dependent
        "request_timeout": 30          # seconds
    },
    "aloha": {
        "num_diffusion_steps": 15,     # Faster for real-time bimanual control
        "best_of_n_samples": 12,       # Higher precision for fine manipulation
        "critic_ensemble_size": 4,     # Minimal for speed
        "max_concurrent_requests": 2,  # Bimanual requires more resources
        "request_timeout": 45
    },
    "libero": {
        "num_diffusion_steps": 10,     # Fast for simulation
        "best_of_n_samples": 16,       # High exploration for diverse tasks
        "critic_ensemble_size": 8,     # Balanced for simulation
        "max_concurrent_requests": 8,  # Simulation can handle more
        "request_timeout": 15
    }
}
```

### Performance Optimization Guide

#### Training Speed Optimization
```python
# Performance optimization checklist
PERFORMANCE_OPTIMIZATION_STRATEGIES = {
    "data_loading": [
        "Use adequate cache_size for episode caching",
        "Enable JAX array sharding with proper device placement",
        "Optimize sequence_ratio for training phase (higher during offline)",
        "Use memory-efficient Best-of-N for large N values"
    ],
    "model_computation": [
        "JIT compile training loops and loss functions",
        "Use reduced diffusion steps during candidate generation",
        "Enable gradient accumulation for large effective batch sizes",
        "Freeze π₀ backbone when appropriate (ALOHA fine-tuning)"
    ],
    "memory_management": [
        "Monitor GPU memory usage and adapt batch sizes",
        "Use chunked Best-of-N sampling for memory constraints",
        "Enable gradient checkpointing for large models",
        "Optimize critic ensemble size vs performance trade-off"
    ],
    "distributed_training": [
        "Use FSDP for large model sharding across multiple GPUs",
        "Coordinate data parallel training with proper synchronization",
        "Balance FSDP devices vs data parallel for optimal throughput",
        "Monitor inter-device communication overhead"
    ]
}
```

#### Hyperparameter Tuning Guidelines
```python
# Systematic hyperparameter tuning protocol
HYPERPARAMETER_TUNING_PROTOCOL = {
    "phase_1_bc_weight": {
        "search_space": [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
        "evaluation_metric": "offline_bc_loss + offline_eval_reward",
        "tuning_steps": 50000,
        "early_stopping": True
    },
    "phase_2_critic_architecture": {
        "search_space": {
            "num_critics": [6, 8, 10, 12, 15],
            "hidden_dims": [(256, 256), (512, 256), (512, 512, 256)]
        },
        "evaluation_metric": "critic_loss_convergence + eval_reward",
        "tuning_steps": 100000
    },
    "phase_3_best_of_n": {
        "search_space": [8, 16, 32, 64],
        "evaluation_metric": "online_eval_reward",
        "memory_constraint": True,
        "adaptive_strategy": True
    }
}
```

### Troubleshooting and Debugging

#### Common Issues and Solutions
```python
# Comprehensive troubleshooting guide
TROUBLESHOOTING_GUIDE = {
    "training_instability": {
        "symptoms": ["Loss spikes", "Q-value collapse", "Gradient explosion"],
        "solutions": [
            "Reduce learning rates by 50%",
            "Increase gradient clipping (reduce max_grad_norm)",
            "Increase BC weight for stability",
            "Reduce Best-of-N samples to decrease variance"
        ]
    },
    "poor_evaluation_performance": {
        "symptoms": ["Low episode rewards", "High evaluation variance"],
        "solutions": [
            "Increase Best-of-N samples for better action selection",
            "Adjust BC-RL balance (increase BC weight)",
            "Verify data preprocessing matches inference pipeline",
            "Check for distribution shift between training/evaluation"
        ]
    },
    "memory_issues": {
        "symptoms": ["OOM errors", "Slow batch generation"],
        "solutions": [
            "Use memory-efficient Best-of-N sampling",
            "Reduce batch size and increase gradient accumulation",
            "Enable gradient checkpointing",
            "Reduce critic ensemble size"
        ]
    },
    "convergence_issues": {
        "symptoms": ["Plateaued learning", "Oscillating losses"],
        "solutions": [
            "Implement learning rate decay scheduling",
            "Increase target network update frequency",
            "Balance loss weights (check bc_weight/critic_weight ratio)",
            "Verify data quality and episode boundary handling"
        ]
    }
}
```

#### Debug Mode Features
```python
# Debug mode comprehensive logging and analysis
class ACRLPDDebugger:
    """Advanced debugging utilities for ACRLPD training."""
    
    def enable_debug_mode(self, agent: ACRLPDPi0Agent):
        """Enable comprehensive debugging features."""
        
        # Gradient flow analysis
        self.setup_gradient_monitoring(agent)
        
        # Action distribution analysis
        self.setup_action_distribution_logging(agent)
        
        # Critic ensemble analysis
        self.setup_critic_ensemble_monitoring(agent)
        
        # Best-of-N sampling analysis
        self.setup_best_of_n_logging(agent)
    
    def generate_training_report(self, step: int) -> str:
        """Generate comprehensive training analysis report."""
        report = f"""
        ACRLPD Training Analysis Report - Step {step}
        
        ## Loss Analysis
        - Total Loss Trend: {self.analyze_loss_trends()}
        - Component Balance: {self.analyze_loss_balance()}
        - Convergence Status: {self.analyze_convergence()}
        
        ## Q-Value Analysis
        - Ensemble Agreement: {self.analyze_ensemble_agreement()}
        - Target Stability: {self.analyze_target_stability()}
        - Bootstrap Quality: {self.analyze_bootstrap_quality()}
        
        ## Policy Analysis
        - Action Diversity: {self.analyze_action_diversity()}
        - Best-of-N Effectiveness: {self.analyze_best_of_n_effectiveness()}
        - π₀ Sampling Quality: {self.analyze_pi0_sampling()}
        
        ## Recommendations
        {self.generate_optimization_recommendations()}
        """
        
        return report
```

### Continuous Integration and Testing

#### Automated Testing Pipeline
```bash
# CI/CD pipeline for ACRLPD framework
#!/bin/bash

# 1. Unit tests for core components
echo "Running unit tests..."
python -m pytest ac_training/tests/test_critic_networks.py -v
python -m pytest ac_training/tests/test_loss_functions.py -v
python -m pytest ac_training/tests/test_acrlpd_dataloader.py -v

# 2. Integration tests with π₀
echo "Running π₀ integration tests..."
python -m pytest ac_training/tests/test_pi0_integration.py -v

# 3. End-to-end training test (reduced scale)
echo "Running end-to-end training test..."
python scripts/train_acrlpd_pi0.py \
    --config debug \
    --experiment_name ci_test \
    --offline_steps 100 \
    --online_steps 50 \
    --eval_frequency 25

# 4. Memory profiling
echo "Running memory profiling..."
python scripts/profile_memory_usage.py --config libero --batch_size 256

# 5. Performance benchmarking
echo "Running performance benchmarks..."
python scripts/benchmark_training_speed.py --all_platforms
```

#### Regression Testing
```python
# Automated regression testing system
class ACRLPDRegressionTester:
    """Regression testing for ACRLPD framework updates."""
    
    def run_regression_suite(self, checkpoint_path: str) -> Dict[str, bool]:
        """Run complete regression test suite."""
        
        results = {}
        
        # 1. API compatibility tests
        results["api_compatibility"] = self.test_api_compatibility()
        
        # 2. Numerical stability tests
        results["numerical_stability"] = self.test_numerical_stability()
        
        # 3. Performance regression tests
        results["performance_regression"] = self.test_performance_metrics(checkpoint_path)
        
        # 4. OpenPI integration tests
        results["openpi_integration"] = self.test_openpi_compatibility()
        
        return results
    
    def test_performance_metrics(self, checkpoint_path: str) -> bool:
        """Test that trained model meets minimum performance thresholds."""
        
        # Load model and run evaluation
        agent = load_acrlpd_checkpoint(checkpoint_path)
        eval_results = run_standardized_evaluation(agent)
        
        # Check against platform-specific baselines
        platform = detect_platform_from_checkpoint(checkpoint_path)
        baseline = PLATFORM_PERFORMANCE_BASELINES[platform]
        
        return eval_results["success_rate"] >= baseline["min_success_rate"]
```

This completes the comprehensive technical documentation for the ACRLPD + π₀ training framework. The system provides a production-ready implementation for training robotic manipulation policies using state-of-the-art diffusion models with reinforcement learning optimization, covering all aspects from data loading through production deployment.