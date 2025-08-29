"""
ACRLPD + π₀ Training Framework v2.0

Two-stage data pipeline:
- Stage 1: H5 → LeRobot conversion (with reward assignment)  
- Stage 2: LeRobot → π₀ training batches (qc_ACT architecture)
"""

# Core data loading components
from .data import (
    # Stage 2: qc_ACT Data Loading
    ACRLPDDataLoader,
    create_acrlpd_data_loader,
    load_acrlpd_norm_stats,
    
    # Backward compatibility for Stage 1
    SamplingStrategy,
    SamplingConfig,
    RewardProcessingConfig,
)

# Optional Stage 1 components (may not exist)
try:
    from .data import (
        ACRLPDDataConverter,
        create_acrlpd_data_converter
    )
except ImportError:
    ACRLPDDataConverter = None
    create_acrlpd_data_converter = None

# Version info
__version__ = "2.0.0"

# Main exports
__all__ = [
    # Stage 2: Data Loading
    "ACRLPDDataLoader",
    "create_acrlpd_data_loader",
    "load_acrlpd_norm_stats",
    
    # Backward compatibility
    "SamplingStrategy",
    "SamplingConfig",
    "RewardProcessingConfig",
    
    # Stage 1 (optional)
    "ACRLPDDataConverter",
    "create_acrlpd_data_converter",
    
    # Utilities
    "create_aloha_folder_pipeline",
]

# Convenience function  
def create_aloha_folder_pipeline(
    positive_dir: str,
    negative_dir: str, 
    output_repo_id: str,
    batch_size: int = 128,
    **kwargs
):
    """
    Create qc_ACT data loader for ALOHA training using unified config system.
    
    Args:
        positive_dir: Positive episodes directory (not used - for compatibility)
        negative_dir: Negative episodes directory (not used - for compatibility) 
        output_repo_id: Existing LeRobot dataset repository ID (not used - uses config)
        batch_size: Training batch size
        **kwargs: Additional loader arguments
        
    Returns:
        (None, dataloader) tuple - dataset is None as Stage 1 not implemented
        
    Note:
        This function is deprecated. Use get_config() + create_acrlpd_data_loader() directly.
    """
    from .config import get_config
    
    # Use default ALOHA config
    rl_config = get_config("rl_aloha_fold")
    
    dataloader = create_acrlpd_data_loader(
        rl_config=rl_config,
        batch_size=batch_size,
        episodes_per_memory_pool=kwargs.get('episodes_per_memory_pool', 16),
        **{k: v for k, v in kwargs.items() if k != 'episodes_per_memory_pool'}
    )
    return None, dataloader