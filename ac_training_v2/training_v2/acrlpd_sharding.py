"""
ACRLPD-specific sharding utilities.

This module provides only the missing functions needed for ACRLPD training
that are not available in OpenPI's sharding module.
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


def create_acrlpd_train_state_sharding(
    train_state_structure,
    mesh: jax.sharding.Mesh,
    log: bool = False
):
    """Create precise FSDP sharding strategy for ACRLPD training state.
    
    Uses OpenPI's fsdp_sharding with different thresholds for different components.
    
    Args:
        train_state_structure: Training state structure from eval_shape
        mesh: JAX mesh for device distribution
        log: Whether to log sharding decisions
        
    Returns:
        Sharding strategy for the complete training state
    """
    # Import OpenPI's fsdp_sharding here
    import openpi.training.sharding as openpi_sharding
    
    def _create_field_sharding(path, field):
        """Create appropriate sharding for each field in the training state."""
        path_str = jax.tree_util.keystr(path)
        
        # Skip GraphDef fields (they are static and shouldn't be sharded)
        if 'model_def' in path_str or 'graphdef' in path_str.lower():
            if log:
                logger.info(f"Replicating GraphDef: {path_str}")
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        
        # Skip transformation objects (static, non-pytree)
        if '_tx' in path_str or 'transformation' in path_str.lower():
            if log:
                logger.info(f"Replicating Transformation: {path_str}")
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        
        # Handle UnspecifiedValue by replicating (using improved detection logic)
        field_type_str = str(type(field))
        is_unspecified = (
            'UnspecifiedValue' in field_type_str or
            'UnspecifiedValue' in str(field.__class__.__name__) if hasattr(field, '__class__') else False or
            'unspecified' in field_type_str.lower() or
            str(field) == 'UnspecifiedValue' or
            (hasattr(field, '__module__') and hasattr(field, '__class__') and 
             'jax' in str(field.__module__) and 'Unspecified' in str(field.__class__.__name__))
        )
        
        if is_unspecified:
            if log:
                logger.warning(f"🔄 分片中发现UnspecifiedValue，复制到所有设备: {path_str} (类型: {field_type_str})")
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        
        # For arrays, create direct sharding using OpenPI's logic
        if hasattr(field, 'shape') and hasattr(field, 'dtype'):
            # Use different thresholds based on component type
            if 'params' in path_str:
                min_size_mbytes = 1  # Aggressive for parameters
            elif 'opt_state' in path_str:
                min_size_mbytes = 1  # Aggressive for optimizer states
            else:
                min_size_mbytes = 4  # Standard for others
            
            # Apply OpenPI's sharding logic directly
            return _apply_openpi_sharding_logic(field, mesh, min_size_mbytes, path_str, log)
        
        # Replicate everything else
        else:
            if log:
                logger.info(f"Replicating other: {path_str} ({type(field)})")
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    
    # Create the sharding specification
    sharding_spec = jax.tree_util.tree_map_with_path(_create_field_sharding, train_state_structure)
    
    # 防御性检查：确保没有UnspecifiedValue残留在sharding中
    def _validate_sharding(path, sharding):
        path_str = jax.tree_util.keystr(path)
        sharding_type_str = str(type(sharding))
        
        # 检查是否是UnspecifiedValue
        is_unspecified = (
            'UnspecifiedValue' in sharding_type_str or
            'unspecified' in sharding_type_str.lower() or
            str(sharding) == 'UnspecifiedValue'
        )
        
        if is_unspecified:
            logger.error(f"❌ 分片规范中发现UnspecifiedValue: {path_str} = {sharding} (类型: {sharding_type_str})")
            logger.error(f"   将强制替换为复制分片")
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        else:
            return sharding
    
    # 应用验证和修复
    validated_sharding_spec = jax.tree_util.tree_map_with_path(_validate_sharding, sharding_spec)
    
    if log:
        logger.info(f"✅ 分片规范创建完成，已验证无UnspecifiedValue残留")
    
    return validated_sharding_spec


def _apply_openpi_sharding_logic(array, mesh, min_size_mbytes, path_str, log):
    """Apply OpenPI's sharding logic directly to avoid pytree issues."""
    import openpi.training.sharding as openpi_sharding
    import numpy as np
    
    min_size_bytes = min_size_mbytes * 2**20
    
    # If FSDP is not actually going to be used, replicate everything
    if mesh.shape[openpi_sharding.FSDP_AXIS] == 1:
        return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    
    # Replicate scalar and vector arrays
    if not hasattr(array, "shape") or len(array.shape) < 2:
        return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    
    # Replicate small arrays
    arr_size = np.prod(array.shape) * np.dtype(array.dtype).itemsize
    if arr_size < min_size_bytes:
        return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Shard matrices and larger tensors along the largest axis that is divisible by the fsdp dimension
    axes = np.argsort(array.shape)[::-1]
    spec = [None] * len(axes)
    for i in axes:
        if array.shape[i] % mesh.shape[openpi_sharding.FSDP_AXIS] == 0:
            if log:
                logger.info(
                    f"Sharding {path_str} of shape {array.shape} ({arr_size / 2**20:.2f} MiB) along axis {i}"
                )
            spec[i] = openpi_sharding.FSDP_AXIS
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec))

    # Replicate if no valid sharding was found
    if log:
        logger.warning(
            f"Could not find a valid sharding for {path_str} of shape {array.shape} with mesh of shape {mesh.shape}"
        )
    return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())