#!/usr/bin/env python3
"""
Test script with timeout and better error handling for data loader
"""
import signal
import sys
import time
from pathlib import Path

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

# Test with 60 second timeout for episode loading
signal.signal(signal.SIGALRM, timeout_handler)

try:
    sys.path.insert(0, str(Path(__file__).parent / "ac_training"))
    from config import get_config
    import dataclasses
    from openpi.training import config as openpi_config
    
    rl_config = get_config("rl_fold_box")
    actual_repo_id = "/era-ai/lm/dataset/huggingface_cache/lerobot/fold_box_unified"
    
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
    
    print("Testing data loader creation with timeout...")
    
    # Set 60 second timeout for data loader creation
    signal.alarm(60)
    
    from ac_training.data.acrlpd_data_loader import create_acrlpd_data_loader
    
    dataloader = create_acrlpd_data_loader(
        rl_config=rl_config,
        batch_size=2,  # Very small batch
        episodes_per_memory_pool=1,  # Only load 1 episode
        shuffle=True,
        seed=42,
        skip_norm_stats=True
    )
    
    signal.alarm(0)  # Cancel timeout
    print("✅ Data loader created successfully!")
    
    # Test batch sampling with timeout
    signal.alarm(30)
    batch = dataloader.sample_batch()
    signal.alarm(0)
    
    print(f"✅ Successfully sampled batch with keys: {list(batch.keys())}")
    
except TimeoutError:
    print("❌ TIMEOUT: Data loader operation timed out")
    print("The hang is in episode loading - dataset may have corrupted data")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    
finally:
    signal.alarm(0)