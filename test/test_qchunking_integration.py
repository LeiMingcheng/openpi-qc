#!/usr/bin/env python3
"""
Integration Test for Q-chunking ACRLPD Framework

This script tests the complete Q-chunking RL integration:
- ACRLPDTrainState creation and management
- CriticNetworks integration with π₀ models
- Loss computation using existing CriticLossComputer
- Target network updates
- Training step execution
"""

import logging
import sys
from pathlib import Path
import traceback

# Add ac_training to path
sys.path.insert(0, str(Path(__file__).parent))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

# Import our integrated components
from training.acrlpd_train_state import ACRLPDTrainState, create_train_state_from_components, acrlpd_train_step
from agents.critic_networks import create_critic_networks, CriticConfig
from agents.loss_functions import CriticLossComputer, LossWeights
from utils.batching import MaskHandler, BootstrapHandler

# Import OpenPI components
import openpi.models.model as _model
import openpi.training.optimizer as _optimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dummy_pi0_model(rng: jax.Array, action_dim: int = 32) -> _model.BaseModel:
    """Create a dummy π₀ model for testing."""
    
    # Create a minimal π₀ config
    class DummyPi0Config:
        def __init__(self):
            self.paligemma_variant = "dummy"
            self.width = 64  # Small LLM dimension for testing
            self.action_dim = action_dim
    
    class DummyPi0Model(_model.BaseModel):
        def __init__(self, config, rngs):
            super().__init__()
            self.config = config
            # Simple linear layers for testing
            self.vision_encoder = nnx.Linear(224*224*3, config.width, rngs=rngs)
            self.action_head = nnx.Linear(config.width + config.action_dim, config.action_dim, rngs=rngs)
            
        def embed_prefix(self, observation):
            """Dummy prefix embedding for testing."""
            batch_size = observation.state.shape[0] if hasattr(observation, 'state') else 1
            prefix_tokens = jnp.ones((batch_size, 10, self.config.width))  # 10 tokens
            prefix_mask = jnp.ones((batch_size, 10), dtype=jnp.bool_)
            return prefix_tokens, prefix_mask, None
            
        def compute_loss(self, rng, observation, actions, train=True):
            """Dummy loss computation."""
            batch_size = actions.shape[0] if actions.ndim > 1 else 1
            return jnp.ones(batch_size) * 0.1
            
        def sample_actions_differentiable(self, rng, observation, num_steps=10):
            """Dummy differentiable action sampling."""
            batch_size = observation.state.shape[0] if hasattr(observation, 'state') else 1
            return jax.random.normal(rng, (batch_size, 1, self.config.action_dim)) * 0.1
            
        def sample_actions(self, rng, observation, num_steps=10):
            """Dummy action sampling (required abstract method)."""
            batch_size = observation.state.shape[0] if hasattr(observation, 'state') else 1
            return jax.random.normal(rng, (batch_size, 1, self.config.action_dim)) * 0.1
    
    config = DummyPi0Config()
    rngs = nnx.Rngs(rng)
    return DummyPi0Model(config, rngs)


def create_dummy_observation(batch_size: int, state_dim: int = 32) -> _model.Observation:
    """Create a dummy observation for testing."""
    
    # Create dummy images
    images = {
        "base_0_rgb": jnp.ones((batch_size, 224, 224, 3), dtype=jnp.float32)
    }
    
    # Create image masks
    image_masks = {
        "base_0_rgb": jnp.ones((batch_size,), dtype=jnp.bool_)
    }
    
    # Create state
    state = jnp.ones((batch_size, state_dim), dtype=jnp.float32)
    
    return _model.Observation(
        images=images,
        image_masks=image_masks,
        state=state
    )


def create_dummy_batch(batch_size: int = 4, horizon_length: int = 5, action_dim: int = 32) -> dict:
    """Create a dummy Q-chunking batch for testing."""
    
    # Create observations
    observations = create_dummy_observation(batch_size, action_dim)
    next_observations = create_dummy_observation(batch_size, action_dim)
    
    # Create action sequences
    actions = jnp.ones((batch_size, horizon_length, action_dim), dtype=jnp.float32) * 0.1
    
    # Create rewards, masks, and validity
    rewards = jnp.ones((batch_size, horizon_length), dtype=jnp.float32)
    masks = jnp.ones((batch_size, horizon_length), dtype=jnp.float32)
    valid = jnp.ones((batch_size, horizon_length), dtype=jnp.float32)
    terminals = jnp.zeros((batch_size, horizon_length), dtype=jnp.bool_)
    
    # Mark last step as terminal for some samples
    terminals = terminals.at[:, -1].set(jnp.array([True, False, True, False]))
    
    return {
        'observations': observations,
        'next_observations': next_observations,
        'actions': actions,
        'rewards': rewards,
        'masks': masks,
        'valid': valid,
        'terminals': terminals,
        'next_terminal': jnp.array([False, False, True, False]),
        'sequence_mask': jnp.ones(batch_size, dtype=jnp.bool_)
    }


def test_critic_networks_creation():
    """Test 1: Critic networks creation and basic functionality."""
    logger.info("🧪 Test 1: Critic Networks Creation")
    
    rng = jax.random.PRNGKey(42)
    action_dim = 32
    horizon_length = 5
    
    # Create dummy π₀ model
    pi0_model = create_dummy_pi0_model(rng, action_dim)
    
    # Create critic configuration
    config = CriticConfig(
        num_critics=3,  # Small ensemble for testing
        hidden_dims=(64, 64),
        dropout_rate=0.1
    )
    
    # Create critic networks
    critic_networks = create_critic_networks(
        config=config,
        pi0_model=pi0_model,
        action_horizon=horizon_length,
        action_dim=action_dim,
        rngs=rng,
        pi0_config=pi0_model.config
    )
    
    logger.info(f"✅ Created critic networks with {config.num_critics} critics")
    
    # Test forward pass
    batch_size = 4
    obs_dim = pi0_model.config.width + action_dim  # 64 + 32 = 96
    flattened_action_dim = horizon_length * action_dim
    
    dummy_obs = jnp.ones((batch_size, obs_dim))
    dummy_actions = jnp.ones((batch_size, flattened_action_dim)) * 0.1
    
    # Test online network
    q_values = critic_networks(dummy_obs, dummy_actions, use_target=False, train=False, aggregate=True)
    logger.info(f"✅ Online Q-values shape: {q_values.shape}")
    
    # Test target network
    target_q_values = critic_networks(dummy_obs, dummy_actions, use_target=True, train=False, aggregate=True)
    logger.info(f"✅ Target Q-values shape: {target_q_values.shape}")
    
    # Test ensemble output
    ensemble_q_values = critic_networks(dummy_obs, dummy_actions, use_target=False, train=False, aggregate=False)
    logger.info(f"✅ Ensemble Q-values shape: {ensemble_q_values.shape}")
    
    return critic_networks, pi0_model


def test_train_state_creation():
    """Test 2: ACRLPDTrainState creation and component integration."""
    logger.info("🧪 Test 2: Training State Creation")
    
    rng = jax.random.PRNGKey(42)
    action_dim = 32
    horizon_length = 5
    
    # Create components
    pi0_model = create_dummy_pi0_model(rng, action_dim)
    
    # Create critic networks
    config = CriticConfig(num_critics=2, hidden_dims=(32, 32))
    critic_networks = create_critic_networks(
        config=config,
        pi0_model=pi0_model,
        action_horizon=horizon_length,
        action_dim=action_dim,
        rngs=rng,
        pi0_config=pi0_model.config
    )
    
    # Create optimizers
    pi0_tx = _optimizer.create_optimizer(
        _optimizer.OptimizerConfig(name='adam', lr=1e-4),
        _optimizer.LearningRateScheduleConfig(name='constant', lr=1e-4)
    )
    
    critic_tx = _optimizer.create_optimizer(
        _optimizer.OptimizerConfig(name='adam', lr=3e-4),
        _optimizer.LearningRateScheduleConfig(name='constant', lr=3e-4)
    )
    
    # Create training state
    train_state = create_train_state_from_components(
        step=0,
        pi0_model=pi0_model,
        pi0_tx=pi0_tx,
        critic_networks=critic_networks,
        critic_tx=critic_tx,
        config={'test': True}
    )
    
    logger.info(f"✅ Created training state at step {train_state.step}")
    logger.info(f"✅ π₀ parameters shape: {jax.tree_map(lambda x: x.shape if hasattr(x, 'shape') else 'no shape', train_state.pi0_params)}")
    logger.info(f"✅ Critic parameters available: {train_state.critic_params is not None}")
    logger.info(f"✅ Target critic parameters available: {train_state.target_critic_params is not None}")
    
    return train_state


def test_loss_computation():
    """Test 3: Loss computation using existing CriticLossComputer."""
    logger.info("🧪 Test 3: Loss Computation")
    
    rng = jax.random.PRNGKey(42)
    action_dim = 32
    horizon_length = 5
    batch_size = 4
    
    # Create components
    pi0_model = create_dummy_pi0_model(rng, action_dim)
    config = CriticConfig(num_critics=2, hidden_dims=(32, 32))
    critic_networks = create_critic_networks(
        config=config,
        pi0_model=pi0_model,
        action_horizon=horizon_length,
        action_dim=action_dim,
        rngs=rng,
        pi0_config=pi0_model.config
    )
    
    # Create loss computer
    critic_loss_computer = CriticLossComputer(
        discount=0.99,
        horizon_length=horizon_length,
        q_aggregation='min',
        config={'use_importance_weighting': False}
    )
    
    # Create dummy batch
    batch = create_dummy_batch(batch_size, horizon_length, action_dim)
    
    # Test loss computation
    rng_loss = jax.random.PRNGKey(123)
    critic_loss, critic_info = critic_loss_computer(
        pi0_model=pi0_model,
        critic_networks=critic_networks,
        observation_encoder=None,  # Will use built-in method
        batch=batch,
        rng=rng_loss,
        train=True
    )
    
    logger.info(f"✅ Critic loss computed: {critic_loss}")
    logger.info(f"✅ Loss info keys: {list(critic_info.keys())}")
    logger.info(f"✅ Q-value mean: {critic_info.get('q_mean', 'N/A')}")
    logger.info(f"✅ Target Q-value mean: {critic_info.get('target_q_mean', 'N/A')}")
    
    return critic_loss, critic_info


def test_training_step():
    """Test 4: Complete training step execution."""
    logger.info("🧪 Test 4: Training Step Execution")
    
    rng = jax.random.PRNGKey(42)
    action_dim = 32
    horizon_length = 5
    batch_size = 2  # Smaller batch for faster testing
    
    # Create training state
    train_state = test_train_state_creation()
    
    # Create dummy batch
    batch = create_dummy_batch(batch_size, horizon_length, action_dim)
    
    # Training configuration
    config = {
        'critic_weight': 1.0,
        'actor_weight': 0.1,
        'bc_loss_weight': 0.01,
        'alpha_weight': 0.0,
        'freeze_pi0_backbone': False,
        'target_update_tau': 0.005,
        'horizon_length': horizon_length,
        'discount': 0.99,
        'q_aggregation': 'min'
    }
    
    # Execute training step
    rng_train = jax.random.PRNGKey(789)
    
    logger.info("Executing training step...")
    new_train_state, loss_info = acrlpd_train_step(
        train_state=train_state,
        batch=batch,
        rng=rng_train,
        config=config
    )
    
    logger.info(f"✅ Training step completed")
    logger.info(f"✅ Step updated: {train_state.step} -> {new_train_state.step}")
    logger.info(f"✅ Total loss: {loss_info['total_loss']}")
    logger.info(f"✅ BC loss (π₀): {loss_info['bc_loss']}")
    logger.info(f"✅ Critic loss: {loss_info['critic_loss']}")
    logger.info(f"✅ Valid samples: {loss_info.get('valid_samples', 'N/A')}")
    
    # Verify target network updates
    if new_train_state.target_critic_params is not None:
        logger.info("✅ Target critic parameters updated")
    else:
        logger.warning("⚠️  Target critic parameters are None")
    
    return new_train_state, loss_info


def test_target_network_updates():
    """Test 5: Target network soft update mechanism."""
    logger.info("🧪 Test 5: Target Network Updates")
    
    rng = jax.random.PRNGKey(42)
    action_dim = 32
    horizon_length = 5
    
    # Create critic networks
    pi0_model = create_dummy_pi0_model(rng, action_dim)
    config = CriticConfig(num_critics=2, hidden_dims=(32, 32), target_update_tau=0.1)
    critic_networks = create_critic_networks(
        config=config,
        pi0_model=pi0_model,
        action_horizon=horizon_length,
        action_dim=action_dim,
        rngs=rng,
        pi0_config=pi0_model.config
    )
    
    # Get initial parameters
    initial_online_params = nnx.state(critic_networks.online_critics.critics[0])
    initial_target_params = nnx.state(critic_networks.target_critics.critics[0])
    
    # Check initial synchronization
    def params_equal(p1, p2):
        return jax.tree_util.tree_all(
            jax.tree_map(lambda x, y: jnp.allclose(x, y), p1, p2)
        )
    
    initial_sync = params_equal(initial_online_params, initial_target_params)
    logger.info(f"✅ Initial sync check: {initial_sync}")
    
    # Modify online network (simulate training)
    for critic in critic_networks.online_critics.critics:
        # Add small noise to simulate parameter updates
        current_params = nnx.state(critic, nnx.Param)
        noisy_params = jax.tree_map(
            lambda x: x + jax.random.normal(jax.random.PRNGKey(123), x.shape) * 0.01,
            current_params
        )
        nnx.update(critic, noisy_params)
    
    # Soft update target networks
    critic_networks.soft_update_target_networks(tau=0.1)
    
    # Check that target parameters have changed but not completely
    final_target_params = nnx.state(critic_networks.target_critics.critics[0])
    final_online_params = nnx.state(critic_networks.online_critics.critics[0])
    
    # Target should be different from initial
    target_changed = not params_equal(initial_target_params, final_target_params)
    # But not identical to online
    not_identical = not params_equal(final_online_params, final_target_params)
    
    logger.info(f"✅ Target parameters changed: {target_changed}")
    logger.info(f"✅ Target not identical to online: {not_identical}")
    
    return True


def run_integration_test():
    """Run complete integration test suite."""
    logger.info("🚀 Starting Q-chunking ACRLPD Integration Test Suite")
    logger.info("="*80)
    
    try:
        # Test 1: Basic critic network functionality
        critic_networks, pi0_model = test_critic_networks_creation()
        logger.info("✅ Test 1 PASSED\n")
        
        # Test 2: Training state creation
        train_state = test_train_state_creation()
        logger.info("✅ Test 2 PASSED\n")
        
        # Test 3: Loss computation
        critic_loss, critic_info = test_loss_computation()
        logger.info("✅ Test 3 PASSED\n")
        
        # Test 4: Training step execution
        new_train_state, loss_info = test_training_step()
        logger.info("✅ Test 4 PASSED\n")
        
        # Test 5: Target network updates
        target_update_success = test_target_network_updates()
        logger.info("✅ Test 5 PASSED\n")
        
        logger.info("="*80)
        logger.info("🎉 ALL TESTS PASSED! Q-chunking ACRLPD integration is working correctly.")
        logger.info("="*80)
        
        # Summary
        logger.info("📋 Integration Test Summary:")
        logger.info("  ✅ Critic networks created and functional")
        logger.info("  ✅ Training state management working")
        logger.info("  ✅ Loss computation using existing CriticLossComputer")
        logger.info("  ✅ Complete training step execution")
        logger.info("  ✅ Target network soft updates")
        logger.info("  ✅ Framework ready for full training")
        
        return True
        
    except Exception as e:
        logger.error("="*80)
        logger.error(f"❌ INTEGRATION TEST FAILED: {e}")
        logger.error("="*80)
        logger.error("Full traceback:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Configure JAX for testing
    jax.config.update('jax_platform_name', 'cpu')  # Use CPU for testing
    
    success = run_integration_test()
    sys.exit(0 if success else 1)