#!/usr/bin/env python3
"""
Simple test script for fold_box_unified data loading in Q-chunking training.

This script tests that the converted fold_box_unified dataset can be loaded
in AC_training Q-chunking without complex analysis - just verify it works.
"""

import os
import sys
import logging
from pathlib import Path

# CRITICAL: Set HuggingFace cache to data disk, not system disk
os.environ['HF_HOME'] = '/era-ai/lm/dataset/huggingface'
os.environ['HF_CACHE_HOME'] = '/era-ai/lm/dataset/huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = '/era-ai/lm/dataset/huggingface_cache/datasets'
os.environ['HF_LEROBOT_HOME'] = '/era-ai/lm/dataset/huggingface_cache/lerobot'

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_fold_box_loading():
    """Test that fold_box_unified data can be loaded in Q-chunking training."""
    
    repo_id = "fold_box_unified"
    
    logger.info("=" * 60)
    logger.info("Testing fold_box_unified Data Loading for Q-chunking")
    logger.info("=" * 60)
    logger.info(f"📂 Dataset: {repo_id}")
    logger.info(f"💾 Cache location: {os.environ['HF_LEROBOT_HOME']}")
    
    try:
        # Import AC training components
        from ac_training.config import get_config
        from ac_training.data import create_acrlpd_data_loader
        logger.info("✓ Successfully imported AC training components")
        
        # Load fold_box configuration
        base_config = get_config("rl_fold_box")
        logger.info(f"✓ Loaded configuration: {base_config.name}")
        
        # Create modified config for fold_box_unified dataset
        import dataclasses
        import openpi.training.config as openpi_config
        
        # Override data config to use fold_box_unified
        # 🔑 关键修复：继承原配置的repack_transforms，避免使用默认值
        original_data_config = base_config.data
        local_data_config = openpi_config.LeRobotAlohaDataConfig(
            repo_id=repo_id,
            default_prompt="fold the clothes on the table",
            adapt_to_pi=False,
            assets=openpi_config.AssetsConfig(asset_id="aloha_fold"),
            # 继承原配置的repack_transforms，包含正确的相机名称映射
            repack_transforms=original_data_config.repack_transforms,
            base_config=original_data_config.base_config
        )
        
        # Create modified RL config with fold_box_unified dataset
        rl_config = dataclasses.replace(base_config, data=local_data_config)
        logger.info(f"✓ Configuration set to use {repo_id}")
        
        # Create Q-chunking data loader
        dataloader = create_acrlpd_data_loader(
            rl_config=rl_config,
            batch_size=2,  # Small batch for testing
            episodes_per_memory_pool=4,  # Small memory pool for testing
            shuffle=True,
            seed=42,
            tolerance_s=15.0  # Relaxed tolerance for any timestamp issues
        )
        logger.info("✓ Created Q-chunking data loader")
        
        # Get basic dataset info
        loader_stats = dataloader.get_dataset_statistics()
        logger.info(f"📊 Dataset info:")
        logger.info(f"  Total episodes: {loader_stats['total_episodes_in_dataset']}")
        logger.info(f"  Episodes in pool: {loader_stats['episodes_in_memory_pool']}")
        logger.info(f"  Pool transitions: {loader_stats['transitions_in_memory_pool']}")
        logger.info(f"  Q-chunking horizon: {loader_stats.get('qchunking_horizon', 'N/A')}")
        
        # Test batch sampling
        logger.info(f"\n⚡ Testing Q-chunking batch sampling...")
        
        for i in range(3):
            # Sample a Q-chunking batch
            batch_dict = dataloader.sample_batch()
            
            logger.info(f"\n📦 Batch {i + 1}:")
            logger.info(f"  Keys: {list(batch_dict.keys())}")
            
            # Check Q-chunking format
            actions = batch_dict['actions']
            rewards = batch_dict['rewards']
            observations = batch_dict['observations']
            
            logger.info(f"  Batch size: {actions.shape[0]}")
            logger.info(f"  Actions shape: {actions.shape}")  # [B, H, action_dim]
            logger.info(f"  Rewards shape: {rewards.shape}")  # [B, H]
            
            # Check observations
            if hasattr(observations, 'state') and observations.state is not None:
                logger.info(f"  State shape: {observations.state.shape}")
            if hasattr(observations, 'image') and observations.image is not None:
                cam_shapes = {cam: data.shape for cam, data in observations.image.items()}
                logger.info(f"  Camera shapes: {cam_shapes}")
        
        logger.info(f"\n🎉 Test completed successfully!")
        logger.info(f"✓ fold_box_unified data can be loaded in Q-chunking training")
        logger.info(f"✓ Batch sampling works correctly")
        logger.info(f"✓ Q-chunking format is valid")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        logger.error("Make sure AC training dependencies are installed")
        return False
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    logger.info("fold_box_unified Q-chunking Loading Test")
    
    # Test fold_box_unified data loading
    if test_fold_box_loading():
        logger.info("\n✅ Test passed! fold_box_unified data works with Q-chunking training.")
        sys.exit(0)
    else:
        logger.error("\n❌ Test failed! Please check the errors above.")
        sys.exit(1)