"""
ACRLPD UnspecifiedValue handling utilities.

This module provides specialized functions for diagnosing and cleaning 
UnspecifiedValue instances that can occur during JAX FSDP initialization.

The actual FSDP sharding is handled by OpenPI's standard sharding module.
This module only provides the UnspecifiedValue preprocessing functions.
"""

import logging
import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


def diagnose_and_mark_unspecified(train_state_structure):
    """Diagnose and mark UnspecifiedValue instances in training state structure.
    
    Args:
        train_state_structure: Result from jax.eval_shape containing potential UnspecifiedValue
        
    Returns:
        Tuple of (unspecified_count, problematic_paths, field_analysis)
    """
    unspecified_count = 0
    problematic_paths = []
    field_analysis = {}
    
    def _analyze_field(path, field):
        nonlocal unspecified_count, problematic_paths
        path_str = jax.tree_util.keystr(path)
        field_type_str = str(type(field))
        
        # Multiple ways to detect UnspecifiedValue
        is_unspecified = (
            'UnspecifiedValue' in field_type_str or
            'UnspecifiedValue' in str(field.__class__.__name__) if hasattr(field, '__class__') else False or
            'unspecified' in field_type_str.lower() or
            str(field) == 'UnspecifiedValue' or
            (hasattr(field, '__module__') and hasattr(field, '__class__') and 
             'jax' in str(field.__module__) and 'Unspecified' in str(field.__class__.__name__))
        )
        
        if is_unspecified:
            unspecified_count += 1
            problematic_paths.append(path_str)
            field_analysis[path_str] = f'UnspecifiedValue({field_type_str})'
            logger.debug(f"🔍 发现UnspecifiedValue: {path_str} = {field} (类型: {field_type_str})")
        elif hasattr(field, 'dtype') and hasattr(field, 'shape'):
            # Valid array-like structure
            field_analysis[path_str] = f'Array[{field.shape}, {field.dtype}]'
        elif hasattr(field, '__call__'):
            # Function or callable (like transformations)
            field_analysis[path_str] = 'Callable'
        else:
            # Other types
            field_analysis[path_str] = field_type_str
    
    jax.tree_util.tree_map_with_path(_analyze_field, train_state_structure)
    
    return unspecified_count, problematic_paths, field_analysis


def clean_unspecified_values(train_state_structure):
    """Clean UnspecifiedValue instances from eval_shape results.
    
    This function replaces all UnspecifiedValue objects with concrete placeholders
    that can be properly handled by JAX sharding operations.
    
    Args:
        train_state_structure: Training state structure from eval_shape
        
    Returns:
        Cleaned training state structure with UnspecifiedValue replaced
    """
    
    def _replace_unspecified(field):
        """Replace UnspecifiedValue with appropriate placeholders."""
        field_type_str = str(type(field))
        
        # Multiple ways to detect UnspecifiedValue (same logic as diagnose function)
        is_unspecified = (
            'UnspecifiedValue' in field_type_str or
            'UnspecifiedValue' in str(field.__class__.__name__) if hasattr(field, '__class__') else False or
            'unspecified' in field_type_str.lower() or
            str(field) == 'UnspecifiedValue' or
            (hasattr(field, '__module__') and hasattr(field, '__class__') and 
             'jax' in str(field.__module__) and 'Unspecified' in str(field.__class__.__name__))
        )
        
        if is_unspecified:
            logger.debug(f"🔄 替换UnspecifiedValue: {field} (类型: {field_type_str})")
            # Create a minimal placeholder that won't be sharded
            # Use a simple scalar array as placeholder
            return jax.ShapeDtypeStruct((), jnp.float32)
        else:
            return field
    
    cleaned_structure = jax.tree_map(_replace_unspecified, train_state_structure)
    
    # Log cleanup results
    original_count, _, _ = diagnose_and_mark_unspecified(train_state_structure)
    cleaned_count, _, _ = diagnose_and_mark_unspecified(cleaned_structure)
    
    if original_count > 0:
        logger.info(f"🔧 清理UnspecifiedValue: {original_count} → {cleaned_count}")
    
    return cleaned_structure

