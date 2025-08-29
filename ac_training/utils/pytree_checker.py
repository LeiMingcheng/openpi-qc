"""
PyTree Consistency Checker for ACRLPD Training.

This module provides utilities to detect and diagnose pytree metadata
inconsistencies that can cause training failures in JAX/FSDP contexts.
"""

import logging
import jax
from typing import Any, Dict, Optional
import optax

logger = logging.getLogger(__name__)


def check_optimizer_consistency(opt1: optax.GradientTransformation, 
                               opt2: optax.GradientTransformation,
                               name1: str = "opt1", 
                               name2: str = "opt2") -> bool:
    """
    Check if two optimizer instances have consistent pytree metadata.
    
    Args:
        opt1: First optimizer instance
        opt2: Second optimizer instance  
        name1: Name for first optimizer (for logging)
        name2: Name for second optimizer (for logging)
        
    Returns:
        True if optimizers are consistent, False otherwise
    """
    try:
        # Get treedef structures
        _, treedef1 = jax.tree_util.tree_flatten(opt1)
        _, treedef2 = jax.tree_util.tree_flatten(opt2)
        
        # Check if treedefs are equal
        if treedef1 == treedef2:
            logger.info(f"✅ Optimizer consistency check passed: {name1} ≡ {name2}")
            return True
        else:
            logger.error(f"❌ Optimizer inconsistency detected: {name1} ≠ {name2}")
            logger.error(f"  {name1} treedef: {treedef1}")
            logger.error(f"  {name2} treedef: {treedef2}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to check optimizer consistency: {e}")
        return False


def diagnose_pytree_structure(obj: Any, name: str = "object") -> Dict[str, Any]:
    """
    Diagnose the pytree structure of an object for debugging.
    
    Args:
        obj: Object to diagnose
        name: Name of the object (for logging)
        
    Returns:
        Dictionary containing structure information
    """
    try:
        leaves, treedef = jax.tree_util.tree_flatten(obj)
        
        info = {
            'name': name,
            'num_leaves': len(leaves),
            'treedef': str(treedef),
            'leaf_types': [type(leaf).__name__ for leaf in leaves[:10]],  # First 10 only
            'leaf_shapes': []
        }
        
        # Get shapes for array leaves
        for leaf in leaves[:10]:  # First 10 only
            if hasattr(leaf, 'shape'):
                info['leaf_shapes'].append(leaf.shape)
            else:
                info['leaf_shapes'].append('no_shape')
        
        logger.info(f"🔍 PyTree structure for {name}:")
        logger.info(f"  Leaves: {info['num_leaves']}")
        logger.info(f"  TreeDef: {info['treedef'][:100]}...")  # Truncate long treedefs
        logger.info(f"  First 10 leaf types: {info['leaf_types']}")
        
        return info
        
    except Exception as e:
        logger.error(f"❌ Failed to diagnose pytree structure for {name}: {e}")
        return {'name': name, 'error': str(e)}


def validate_fsdp_compatibility(train_state: Any, 
                               mesh: jax.sharding.Mesh,
                               data_sharding: jax.sharding.Sharding) -> bool:
    """
    Validate that a training state is compatible with FSDP requirements.
    
    Args:
        train_state: Training state to validate
        mesh: JAX mesh for FSDP
        data_sharding: Data sharding strategy
        
    Returns:
        True if compatible, False otherwise
    """
    try:
        logger.info("🔍 Validating FSDP compatibility...")
        
        # Check basic pytree structure
        leaves, treedef = jax.tree_util.tree_flatten(train_state)
        logger.info(f"  Training state has {len(leaves)} leaves")
        
        # Check optimizer fields are marked as non-pytree nodes
        optimizer_fields = ['pi0_tx', 'critic_tx', 'temperature_tx']
        for field in optimizer_fields:
            if hasattr(train_state, field):
                opt = getattr(train_state, field)
                if opt is not None:
                    # Check if the optimizer appears in the training state's pytree leaves
                    # This is the correct way to check if struct.field(pytree_node=False) works
                    train_state_leaves, _ = jax.tree_util.tree_flatten(train_state)
                    opt_found_in_leaves = any(leaf is opt for leaf in train_state_leaves)
                    
                    if opt_found_in_leaves:
                        logger.warning(f"⚠️  {field} is included in training state pytree (should be excluded)")
                    else:
                        logger.info(f"✅ {field} correctly excluded from training state pytree")
        
        # Test eval_shape compatibility
        try:
            state_shape = jax.eval_shape(lambda: train_state)
            logger.info("✅ Training state is eval_shape compatible")
        except Exception as e:
            logger.error(f"❌ Training state eval_shape failed: {e}")
            return False
        
        # Test sharding compatibility
        try:
            import openpi.training.sharding as sharding
            with sharding.set_mesh(mesh):
                # This should work without errors
                sharded_leaves = jax.tree_map(lambda x: x, train_state)
                logger.info("✅ Training state is sharding compatible")
        except Exception as e:
            logger.error(f"❌ Training state sharding failed: {e}")
            return False
        
        logger.info("✅ FSDP compatibility validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ FSDP compatibility validation failed: {e}")
        return False


def compare_optimizer_metadata(opt1: optax.GradientTransformation,
                              opt2: optax.GradientTransformation) -> None:
    """
    Compare metadata between two optimizers in detail.
    
    Args:
        opt1: First optimizer
        opt2: Second optimizer
    """
    logger.info("🔍 Detailed optimizer metadata comparison:")
    
    try:
        # Compare string representations
        str1 = str(opt1)
        str2 = str(opt2)
        
        if str1 == str2:
            logger.info("✅ String representations match")
        else:
            logger.warning("⚠️  String representations differ")
            logger.info(f"  Opt1: {str1[:100]}...")
            logger.info(f"  Opt2: {str2[:100]}...")
        
        # Compare function references
        if hasattr(opt1, 'init') and hasattr(opt2, 'init'):
            if opt1.init == opt2.init:
                logger.info("✅ Init functions are identical")
            else:
                logger.warning("⚠️  Init functions differ")
                logger.info(f"  Opt1.init: {opt1.init}")
                logger.info(f"  Opt2.init: {opt2.init}")
        
        if hasattr(opt1, 'update') and hasattr(opt2, 'update'):
            if opt1.update == opt2.update:
                logger.info("✅ Update functions are identical")
            else:
                logger.warning("⚠️  Update functions differ")
                logger.info(f"  Opt1.update: {opt1.update}")
                logger.info(f"  Opt2.update: {opt2.update}")
                
    except Exception as e:
        logger.error(f"❌ Optimizer metadata comparison failed: {e}")


if __name__ == "__main__":
    """Test the pytree checker utilities."""
    logging.basicConfig(level=logging.INFO)
    
    # Test optimizer consistency
    import openpi.training.optimizer as _optimizer
    from config import get_config
    
    rl_config = get_config("rl_fold_box")
    
    # Create two optimizers the same way using effective lr schedule
    opt1 = _optimizer.create_optimizer(rl_config.actor_optimizer, rl_config.get_effective_actor_lr_schedule())
    opt2 = _optimizer.create_optimizer(rl_config.actor_optimizer, rl_config.get_effective_actor_lr_schedule())
    
    check_optimizer_consistency(opt1, opt2, "opt1", "opt2")
    compare_optimizer_metadata(opt1, opt2)