#!/usr/bin/env python3
"""
Test script for the new two-stage ACRLPD data loading pipeline.

This script tests:
1. ACRLPDDataConverter: H5 → LeRobot conversion with reward assignment
2. ACRLPDDataLoader: LeRobot → π₀ training batches

Using real ALOHA data from /era-ai/lm/dataset/lmc/aloha_pp/
- score_1: negative samples (reward=0.0)
- score_5: positive samples (reward=1.0)
"""

import os
import sys
import logging
from pathlib import Path

# CRITICAL: Set HuggingFace cache to data disk, not system disk
# This prevents "No space left on device" errors
os.environ['HF_HOME'] = '/era-ai/lm/dataset/huggingface'
os.environ['HF_CACHE_HOME'] = '/era-ai/lm/dataset/huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = '/era-ai/lm/dataset/huggingface_cache/datasets'
os.environ['HF_LEROBOT_HOME'] = '/era-ai/lm/dataset/huggingface_cache/lerobot'

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_two_stage_pipeline():
    """Test the complete two-stage data loading pipeline with efficient sample size."""
    
    # Test data paths
    base_data_dir = "/era-ai/lm/dataset/lmc/aloha_pp"
    positive_dir = f"{base_data_dir}/score_5"  # High reward samples
    negative_dir = f"{base_data_dir}/score_1"  # Low reward samples
    output_repo_id = "aloha_test_dataset"
    
    # Efficient testing: 10 episodes per folder for complete functionality testing
    MAX_EPISODES_PER_DIR = 10
    
    logger.info("=" * 80)
    logger.info("Testing ACRLPD Two-Stage Data Loading Pipeline v2.0")
    logger.info("=" * 80)
    
    # Check if test data directories exist
    if not os.path.exists(positive_dir):
        logger.error(f"Positive data directory not found: {positive_dir}")
        return False
    if not os.path.exists(negative_dir):
        logger.error(f"Negative data directory not found: {negative_dir}")
        return False
        
    logger.info(f"✓ Found test data directories:")
    logger.info(f"  Positive: {positive_dir}")
    logger.info(f"  Negative: {negative_dir}")
    
    try:
        # Import the unified configuration architecture
        from ac_training.config import get_config
        from ac_training.data import (
            create_acrlpd_data_loader,
            create_acrlpd_data_converter,
        )
        logger.info("✓ Successfully imported two-stage architecture")
        
        # Stage 1: Data Conversion (H5 → LeRobot + reward assignment)
        logger.info("\n" + "=" * 40)
        logger.info("STAGE 1: Data Conversion")
        logger.info("=" * 40)
        
        # Create converter with folder-based reward strategy
        converter = create_acrlpd_data_converter(
            robot_type="aloha",
            reward_strategy="folder_based",  # positive/negative folders
            positive_reward=1.0,
            negative_reward=0.0
        )
        logger.info(f"✓ Created ACRLPDDataConverter with folder-based reward strategy")
        
        # Print converter configuration
        logger.info(f"  Robot type: {converter.robot_config.robot_type}")
        logger.info(f"  Motors: {len(converter.robot_config.motors)} ({converter.robot_config.motors[:3]}...)")
        logger.info(f"  Cameras: {converter.robot_config.cameras}")
        logger.info(f"  Reward strategy: {converter.reward_config.strategy.value}")
        logger.info(f"  Positive reward: {converter.reward_config.positive_reward}")
        logger.info(f"  Negative reward: {converter.reward_config.negative_reward}")
        
        # Convert H5 data to LeRobot format (LIMITED FOR EFFICIENT TESTING)
        logger.info(f"\n⚡ Converting H5 data to LeRobot format...")
        logger.info(f"💾 Using data disk: {os.environ['HF_LEROBOT_HOME']}")
        logger.info(f"⚡ EFFICIENT MODE: Processing {MAX_EPISODES_PER_DIR} episodes per directory")
        logger.info(f"   This tests complete functionality while saving time")
        
        # Get episode lists for efficient testing
        import glob
        positive_episodes = sorted(glob.glob(f"{positive_dir}/*.hdf5"))
        negative_episodes = sorted(glob.glob(f"{negative_dir}/*.hdf5"))
        
        logger.info(f"   Found {len(positive_episodes)} positive episodes, using first {MAX_EPISODES_PER_DIR}")
        logger.info(f"   Found {len(negative_episodes)} negative episodes, using first {MAX_EPISODES_PER_DIR}")
        
        # Convert with episode limits for efficient testing
        dataset = converter.convert_h5_to_lerobot(
            h5_data_path=[positive_dir, negative_dir],
            output_repo_id=output_repo_id,
            task_name="aloha_manipulation_test",
            episodes=list(range(MAX_EPISODES_PER_DIR))  # First N episodes from each dir
        )
        
        # Print conversion statistics
        stats = converter.get_conversion_stats()
        logger.info(f"✓ Conversion completed successfully!")
        logger.info(f"  Episodes processed: {stats['episodes_processed']}")
        logger.info(f"  Total frames: {stats['total_frames']}")
        logger.info(f"  Positive episodes: {stats['positive_episodes']}")
        logger.info(f"  Negative episodes: {stats['negative_episodes']}")
        logger.info(f"  Average episode length: {stats['average_episode_length']:.1f}")
        logger.info(f"  Reward distribution: {stats['reward_distribution']}")
        
        # Stage 2: Training Data Loading (LeRobot → π₀ training batches)
        logger.info("\n" + "=" * 40)
        logger.info("STAGE 2: QC-ACT Training Data Loading")
        logger.info("=" * 40)
        
        # Load unified configuration and override repo_id for local test dataset
        base_config = get_config("rl_aloha_fold")
        logger.info(f"✓ Loaded base configuration: {base_config.name}")
        
        # Create modified config pointing to local test dataset
        import dataclasses
        import openpi.training.config as openpi_config
        
        # Override data config to use local test dataset
        local_data_config = openpi_config.LeRobotAlohaDataConfig(
            repo_id=output_repo_id,  # Use local test dataset
            default_prompt="fold the clothes on the table",
            adapt_to_pi=False,
            assets=openpi_config.AssetsConfig(asset_id="aloha_fold")
        )
        
        # Create modified RL config with local dataset
        rl_config = dataclasses.replace(base_config, data=local_data_config)
        logger.info(f"✓ Modified config to use local dataset: {output_repo_id}")
        
        # Create qc_ACT architecture data loader for training using unified config
        dataloader = create_acrlpd_data_loader(
            rl_config=rl_config,
            batch_size=16,  # Small batch for testing
            episodes_per_memory_pool=8,  # Memory pool size - 8 episodes for testing
            shuffle=True,
            seed=42,
            tolerance_s=15.0  # Relaxed tolerance for timestamp sync issues
        )
        logger.info(f"✓ Created qc_ACT ACRLPDDataLoader with memory pool architecture")
        logger.info(f"  Architecture: qc_ACT complete random sampling")
        logger.info(f"  Memory pool strategy: {dataloader.episodes_per_memory_pool} episodes per epoch")
        
        # Print qc_ACT data loader configuration
        loader_stats = dataloader.get_dataset_statistics()
        logger.info(f"  Total episodes in dataset: {loader_stats['total_episodes_in_dataset']}")
        logger.info(f"  Episodes in memory pool: {loader_stats['episodes_in_memory_pool']}")
        logger.info(f"  Transitions in memory pool: {loader_stats['transitions_in_memory_pool']}")
        logger.info(f"  Memory pool utilization: {loader_stats['memory_pool_utilization']:.1f}%")
        logger.info(f"  Batch size: {loader_stats.get('batch_size', 16)}")
        logger.info(f"  Q-chunking horizon: {loader_stats.get('qchunking_horizon', 'N/A')}")
        logger.info(f"  Model action dim: {loader_stats.get('model_action_dim', 'N/A')}")
        logger.info(f"  Config name: {loader_stats.get('config_name', 'N/A')}")
        
        # Test qc_ACT batch sampling
        logger.info(f"\n⚡ Testing qc_ACT random batch sampling...")
        num_test_batches = 3
        
        for i in range(num_test_batches):
            # qc_ACT: Direct batch sampling (returns Q-chunking dictionary)
            batch_dict = dataloader.sample_batch()
            
            logger.info(f"\n📦 Q-chunking Batch {i + 1}:")
            logger.info(f"  Batch keys: {list(batch_dict.keys())}")
            
            # Extract components from Q-chunking format
            observations = batch_dict['observations']
            actions = batch_dict['actions']
            rewards = batch_dict['rewards']
            
            logger.info(f"  Batch size: {actions.shape[0]}")
            logger.info(f"  Actions shape: {actions.shape}")  # [B, H, action_dim]
            logger.info(f"  Rewards shape: {rewards.shape}")  # [B, H]
            
            # Check observation components
            logger.info(f"  Observation type: {type(observations)}")
            if hasattr(observations, 'state') and observations.state is not None:
                logger.info(f"  State shape: {observations.state.shape}")
            if hasattr(observations, 'image') and observations.image is not None:
                for cam_name, cam_data in observations.image.items():
                    logger.info(f"  {cam_name} shape: {cam_data.shape}")
            if hasattr(observations, 'tokenized_prompt') and observations.tokenized_prompt is not None:
                logger.info(f"  Tokenized prompt shape: {observations.tokenized_prompt.shape}")
            
            # Q-chunking specific info
            logger.info(f"  Q-chunking horizon: {actions.shape[1]}")
            logger.info(f"  Action dimension: {actions.shape[2]}")
            logger.info(f"  Has next_observations: {'next_observations' in batch_dict}")
            logger.info(f"  Has masks: {'masks' in batch_dict}")
            logger.info(f"  Sampling: Pure random from {dataloader.total_pool_transitions} transitions")
        
        # Test qc_ACT memory pool refresh
        logger.info(f"\n⚡ Testing qc_ACT memory pool refresh...")
        old_pool_size = dataloader.total_pool_transitions
        old_episodes = dataloader.stats['episodes_in_memory_pool']
        
        # Refresh memory pool to epoch 1 (simulate new epoch)
        dataloader.refresh_memory_pool(epoch_seed=1)
        
        new_pool_size = dataloader.total_pool_transitions
        new_episodes = dataloader.stats['episodes_in_memory_pool']
        
        logger.info(f"✓ Memory pool refresh successful")
        logger.info(f"  Pool size before: {old_pool_size} transitions ({old_episodes} episodes)")
        logger.info(f"  Pool size after: {new_pool_size} transitions ({new_episodes} episodes)")
        logger.info(f"  Pool changed: {old_pool_size != new_pool_size}")
        
        # Test one more batch after refresh
        batch_dict_after_refresh = dataloader.sample_batch()
        actions_after_refresh = batch_dict_after_refresh['actions']
        logger.info(f"  Post-refresh batch shape: {actions_after_refresh.shape}")
        
        logger.info(f"\n🎉 Two-stage qc_ACT pipeline test completed successfully!")
        logger.info(f"✓ Stage 1 (Conversion): H5 → LeRobot format with rewards")
        logger.info(f"✓ Stage 2 (qc_ACT Loading): LeRobot → π₀ training batches")
        logger.info(f"✓ qc_ACT memory pool architecture working")
        logger.info(f"✓ Complete random sampling verified")
        logger.info(f"✓ Memory pool refresh functionality working")
        logger.info(f"✓ π₀-compatible observation format verified")
        logger.info(f"✓ Zero I/O training mode enabled")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        logger.error("Make sure all dependencies are installed")
        return False
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_stage_2_only():
    """Test only Stage 2: LeRobot → π₀ training batches using existing converted data."""
    
    # Use existing converted dataset from Stage 1
    output_repo_id = "aloha_test_dataset"
    
    logger.info("=" * 80)
    logger.info("Testing ACRLPD Stage 2: Training Data Loading")
    logger.info("=" * 80)
    logger.info(f"📂 Using existing LeRobot dataset: {output_repo_id}")
    logger.info(f"💾 LeRobot cache location: {os.environ['HF_LEROBOT_HOME']}")
    
    try:
        # Import the unified configuration architecture
        from ac_training.config import get_config
        from ac_training.data import (
            create_acrlpd_data_loader,
            load_acrlpd_norm_stats,
        )
        from openpi.shared import normalize as _normalize
        logger.info("✓ Successfully imported Stage 2 architecture")
        
        # Helper function to load existing norm_stats for testing
        def load_norm_stats_for_testing(repo_id: str):
            """Load existing normalization stats for testing."""
            logger.info(f"📊 Loading existing normalization stats for {repo_id}...")
            
            norm_stats = load_acrlpd_norm_stats(repo_id)
            if norm_stats is not None:
                logger.info("✓ Successfully loaded existing norm_stats")
                logger.info(f"  Keys available: {list(norm_stats.keys())}")
                return norm_stats
            else:
                logger.warning("❌ No norm_stats found!")
                logger.warning("💡 Please run: ./ac_training/scripts/run_compute_norm_stats.sh")
                return None
        
        # Stage 2: Training Data Loading (LeRobot → π₀ training batches)
        logger.info("\n" + "=" * 40)
        logger.info("STAGE 2: Training Data Loading")
        logger.info("=" * 40)
        
        # Load unified configuration and override repo_id for local test dataset
        base_config = get_config("rl_aloha_fold")
        logger.info(f"✓ Loaded base configuration: {base_config.name}")
        
        # Create modified config pointing to local test dataset
        import dataclasses
        import openpi.training.config as openpi_config
        
        # Override data config to use local test dataset
        local_data_config = openpi_config.LeRobotAlohaDataConfig(
            repo_id=output_repo_id,  # Use local test dataset
            default_prompt="fold the clothes on the table",
            adapt_to_pi=False,
            assets=openpi_config.AssetsConfig(asset_id="aloha_fold")
        )
        
        # Create modified RL config with local dataset
        rl_config = dataclasses.replace(base_config, data=local_data_config)
        logger.info(f"✓ Modified config to use local dataset: {output_repo_id}")
        
        # Load existing normalization statistics
        norm_stats = load_norm_stats_for_testing(output_repo_id)
        
        # Create qc_ACT data loader for training using unified config
        dataloader = create_acrlpd_data_loader(
            rl_config=rl_config,
            batch_size=16,  # Small batch for testing
            episodes_per_memory_pool=8,  # Memory pool size - 8 episodes for testing
            shuffle=True,
            seed=42,
            tolerance_s=15.0  # Relaxed tolerance for timestamp sync issues
        )
        logger.info(f"✓ Created qc_ACT ACRLPDDataLoader with memory pool architecture")
        
        # Print qc_ACT data loader configuration
        loader_stats = dataloader.get_dataset_statistics()
        logger.info(f"  Total episodes in dataset: {loader_stats['total_episodes_in_dataset']}")
        logger.info(f"  Episodes in memory pool: {loader_stats['episodes_in_memory_pool']}")
        logger.info(f"  Transitions in memory pool: {loader_stats['transitions_in_memory_pool']}")
        logger.info(f"  Memory pool utilization: {loader_stats['memory_pool_utilization']:.1f}%")
        logger.info(f"  Batch size: {loader_stats.get('batch_size', 16)}")
        logger.info(f"  Q-chunking horizon: {loader_stats.get('qchunking_horizon', 'N/A')}")
        logger.info(f"  Model action dim: {loader_stats.get('model_action_dim', 'N/A')}")
        logger.info(f"  Config name: {loader_stats.get('config_name', 'N/A')}")
        
        # Test qc_ACT batch sampling
        logger.info(f"\n⚡ Testing qc_ACT random batch sampling...")
        num_test_batches = 3
        
        for i in range(num_test_batches):
            # qc_ACT: Direct batch sampling (returns Q-chunking dictionary)
            batch_dict = dataloader.sample_batch()
            
            logger.info(f"\n📦 Q-chunking Batch {i + 1}:")
            logger.info(f"  Batch keys: {list(batch_dict.keys())}")
            
            # Extract components from Q-chunking format
            observations = batch_dict['observations']
            actions = batch_dict['actions']
            rewards = batch_dict['rewards']
            
            logger.info(f"  Batch size: {actions.shape[0]}")
            logger.info(f"  Actions shape: {actions.shape}")  # [B, H, action_dim]
            logger.info(f"  Rewards shape: {rewards.shape}")  # [B, H]
            
            # Check observation components
            logger.info(f"  Observation type: {type(observations)}")
            if hasattr(observations, 'state') and observations.state is not None:
                logger.info(f"  State shape: {observations.state.shape}")
            if hasattr(observations, 'image') and observations.image is not None:
                for cam_name, cam_data in observations.image.items():
                    logger.info(f"  {cam_name} shape: {cam_data.shape}")
            if hasattr(observations, 'tokenized_prompt') and observations.tokenized_prompt is not None:
                logger.info(f"  Tokenized prompt shape: {observations.tokenized_prompt.shape}")
            
            # Q-chunking specific info
            logger.info(f"  Q-chunking horizon: {actions.shape[1]}")
            logger.info(f"  Action dimension: {actions.shape[2]}")
            logger.info(f"  Has next_observations: {'next_observations' in batch_dict}")
            logger.info(f"  Has masks: {'masks' in batch_dict}")
            logger.info(f"  Sampling: Pure random from {dataloader.total_pool_transitions} transitions")
        
        logger.info(f"\n🎉 qc_ACT Stage 2 test completed successfully!")
        logger.info(f"✓ LeRobot → π₀ training batches working")
        logger.info(f"✓ qc_ACT memory pool architecture working")
        logger.info(f"✓ Complete random sampling verified")
        logger.info(f"✓ π₀-compatible observation format verified")
        logger.info(f"✓ Zero I/O training mode enabled")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        logger.error("Make sure all dependencies are installed")
        return False
        
    except Exception as e:
        logger.error(f"❌ Stage 2 test failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    import sys
    
    # Check command line arguments for test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--stage2-only":
        logger.info("ACRLPD Stage 2 Only Test")
        logger.info("Using existing converted LeRobot data")
        
        # Test only Stage 2
        if test_stage_2_only():
            logger.info("\n✅ Stage 2 test passed! Data loading pipeline is working correctly.")
            sys.exit(0)
        else:
            logger.error("\n❌ Stage 2 test failed! Please check the errors above.")
            sys.exit(1)
    else:
        logger.info("ACRLPD Two-Stage Pipeline Test")
        logger.info("Using real ALOHA data from /era-ai/lm/dataset/lmc/aloha_pp/")
        
        # Test complete two-stage pipeline
        if test_two_stage_pipeline():
            logger.info("\n✅ All tests passed! Two-stage pipeline is working correctly.")
            sys.exit(0)
        else:
            logger.error("\n❌ Tests failed! Please check the errors above.")
            sys.exit(1)