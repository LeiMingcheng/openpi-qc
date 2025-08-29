"""
Data loading infrastructure for ACRLPD training.

This module provides a clean two-stage data loading architecture:
- Stage 1: ACRLPDDataConverter (H5 → LeRobot + reward assignment)  
- Stage 2: ACRLPDDataLoader (LeRobot → π₀ training batches)
"""

from .acrlpd_data_converter import (
    DatasetConfig,
)

from .acrlpd_data_loader import (
    ACRLPDDataLoader,
    SamplingConfig,
    SamplingStrategy, 
    RewardProcessingConfig,
    create_acrlpd_data_loader,
    load_acrlpd_norm_stats
)

from .compute_acrlpd_norm_stats import (
    compute_acrlpd_norm_stats
)

__all__ = [
    # Stage 1: Data Conversion
    "ACRLPDDataConverter", 
    "DatasetConfig",
    "RewardStrategy",
    "RewardConfig",
    
    # Stage 2: Data Loading
    "ACRLPDDataLoader", 
    "SamplingConfig",
    "SamplingStrategy",
    "RewardProcessingConfig",
    "create_acrlpd_data_loader",
    
    # Utilities
    "compute_acrlpd_norm_stats",
    "load_acrlpd_norm_stats",
    
    # Factory Functions
    "create_acrlpd_data_converter"
]