#!/usr/bin/env python3
"""
Debug script to identify where the norm stats computation is hanging.
"""
import sys
import signal
import time
import logging
from pathlib import Path

# Set up timeout handler
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Script timed out")

# Set timeout to 120 seconds
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(120)

# Configure logging to be verbose
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

print("🔍 DEBUG: Starting norm stats hang diagnosis...")

try:
    # Add path and import config
    sys.path.insert(0, str(Path(__file__).parent / "ac_training"))
    from config import get_config
    
    print("🔍 DEBUG: Loading rl_fold_box config...")
    rl_config = get_config("rl_fold_box")
    print(f"🔍 DEBUG: Config loaded, repo_id = {rl_config.data.repo_id}")
    
    # Override repo_id 
    import dataclasses
    from openpi.training import config as openpi_config
    
    actual_repo_id = "/era-ai/lm/dataset/huggingface_cache/lerobot/fold_box_unified"
    print(f"🔍 DEBUG: Overriding with repo_id = {actual_repo_id}")
    
    modified_data_config = openpi_config.LeRobotAlohaDataConfig(
        repo_id=actual_repo_id,
        default_prompt=rl_config.data.default_prompt,
        adapt_to_pi=rl_config.data.adapt_to_pi,
        assets=rl_config.data.assets,
        repack_transforms=rl_config.data.repack_transforms,
        base_config=rl_config.data.base_config
    )
    
    rl_config = dataclasses.replace(
        rl_config, 
        data=modified_data_config,
        weight_loader=None
    )
    
    print("🔍 DEBUG: Creating ACRLPD data loader...")
    from ac_training.data.acrlpd_data_loader import create_acrlpd_data_loader
    
    # Test with very small batch size and memory pool
    dataloader = create_acrlpd_data_loader(
        rl_config=rl_config,
        batch_size=4,  # Very small
        episodes_per_memory_pool=2,  # Very small
        shuffle=True,
        seed=42,
        skip_norm_stats=True  # Skip norm stats loading
    )
    
    print("🔍 DEBUG: Data loader created successfully!")
    print(f"🔍 DEBUG: Episodes found: {len(dataloader.all_episode_info)}")
    print(f"🔍 DEBUG: Pool episodes: {len(dataloader.memory_pool_episodes)}")
    
    # Try to sample one batch
    print("🔍 DEBUG: Attempting to sample one batch...")
    batch = dataloader.sample_batch()
    print(f"🔍 DEBUG: Successfully sampled batch with keys: {list(batch.keys())}")
    
    print("✅ SUCCESS: No hang detected in data loader!")
    
except TimeoutError:
    print("❌ TIMEOUT: Data loader creation/sampling timed out after 120 seconds")
    print("   This indicates the hang is in the data loading process")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    
finally:
    signal.alarm(0)  # Cancel timeout

print("🔍 DEBUG: Script completed")