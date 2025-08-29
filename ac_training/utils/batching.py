"""
Batching utilities for ACRLPD training.

This module provides mask handling and bootstrap computation utilities
that were previously in the batching module.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MaskHandler:
    """Handles mask operations and sample weighting for training."""
    
    @staticmethod
    def compute_adaptive_sample_weights(
        batch: Dict[str, Any],
        use_reward_weighting: bool = True,
        use_rarity_weighting: bool = True,
        temperature: float = 1.0
    ) -> jnp.ndarray:
        """
        Compute adaptive sample weights based on reward and rarity.
        
        Args:
            batch: Training batch containing rewards and other data
            use_reward_weighting: Whether to weight by reward magnitude
            use_rarity_weighting: Whether to weight by data rarity
            temperature: Temperature for softmax weighting
            
        Returns:
            Sample weights array
        """
        batch_size = len(batch['rewards'])
        weights = jnp.ones(batch_size)
        
        if use_reward_weighting and 'rewards' in batch:
            # Weight by absolute reward magnitude
            reward_weights = jnp.abs(batch['rewards']) + 1e-8
            reward_weights = reward_weights / (jnp.mean(reward_weights) + 1e-8)
            weights = weights * reward_weights
            
        if use_rarity_weighting:
            # Simple rarity weighting based on data diversity
            # This is a simplified implementation - could be enhanced with more sophisticated methods
            if 'observations' in batch:
                obs = batch['observations']
                # Compute pairwise distances to estimate rarity
                obs_flat = obs.reshape(obs.shape[0], -1)
                distances = jnp.linalg.norm(obs_flat[:, None] - obs_flat[None, :], axis=-1)
                # Rarity is inversely proportional to average distance to other samples
                avg_distances = jnp.mean(distances, axis=1)
                rarity_weights = 1.0 / (avg_distances + 1e-8)
                rarity_weights = rarity_weights / (jnp.mean(rarity_weights) + 1e-8)
                weights = weights * rarity_weights
        
        # Apply temperature scaling
        if temperature != 1.0:
            weights = jnp.power(weights, 1.0 / temperature)
            
        # Normalize weights
        weights = weights / (jnp.mean(weights) + 1e-8)
        
        return weights
    
    @staticmethod
    def compute_temporal_consistency_mask(
        current_observations: jnp.ndarray,
        next_observations: jnp.ndarray,
        consistency_threshold: float = 0.1
    ) -> jnp.ndarray:
        """
        Compute mask for temporally consistent transitions.
        
        Args:
            current_observations: Current state observations
            next_observations: Next state observations
            consistency_threshold: Threshold for consistency check
            
        Returns:
            Boolean mask for consistent transitions
        """
        # Compute state differences
        obs_diff = jnp.linalg.norm(
            next_observations - current_observations, 
            axis=-1
        )
        
        # Mark transitions with reasonable state changes as consistent
        # Very small changes might indicate duplicate data
        # Very large changes might indicate discontinuities
        median_diff = jnp.median(obs_diff)
        consistency_mask = (obs_diff > consistency_threshold * median_diff) & \
                          (obs_diff < (1.0 / consistency_threshold) * median_diff)
        
        return consistency_mask
    
    @staticmethod
    def apply_loss_masking(
        loss: jnp.ndarray,
        mask: jnp.ndarray,
        sample_weights: Optional[jnp.ndarray] = None,
        normalization: str = "mean"
    ) -> jnp.ndarray:
        """
        Apply masking and weighting to loss.
        
        Args:
            loss: Raw loss values
            mask: Boolean mask for valid samples
            sample_weights: Optional sample weights
            normalization: How to normalize ("mean", "sum", "none")
            
        Returns:
            Masked and weighted loss
        """
        # Apply mask
        masked_loss = loss * mask
        
        # Apply sample weights if provided
        if sample_weights is not None:
            masked_loss = masked_loss * sample_weights
            
        # Normalize
        if normalization == "mean":
            # Normalize by valid samples
            valid_count = jnp.sum(mask) + 1e-8
            return jnp.sum(masked_loss) / valid_count
        elif normalization == "sum":
            return jnp.sum(masked_loss)
        else:  # "none"
            return masked_loss


class BootstrapHandler:
    """Handles bootstrap target computation for Q-learning."""
    
    def __init__(self):
        pass
    
    def compute_bootstrap_target(
        self,
        rewards: jnp.ndarray,
        next_q_values: jnp.ndarray,
        masks: jnp.ndarray,
        discount: float = 0.99,
        horizon_length: int = 1,
        adaptive_horizon: bool = False
    ) -> jnp.ndarray:
        """
        Compute bootstrap targets for Q-chunking.
        
        Args:
            rewards: Reward sequences [batch_size, horizon_length] or [batch_size]
            next_q_values: Q-values for next states [batch_size]
            masks: Episode termination masks [batch_size, horizon_length] or [batch_size]
            discount: Discount factor
            horizon_length: Action chunking horizon length
            adaptive_horizon: Whether to use adaptive horizon (simplified)
            
        Returns:
            Bootstrap target values [batch_size]
        """
        # Standard Q-chunking bootstrap: r[-1] + γ^H * mask[-1] * Q_next
        
        # Handle different input shapes
        if rewards.ndim > 1:
            # Multi-step rewards: use final step
            final_rewards = rewards[..., -1]  # [batch_size]
            final_masks = masks[..., -1] if masks.ndim > 1 else masks  # [batch_size]
        else:
            # Single-step rewards: use as-is
            final_rewards = rewards  # [batch_size]
            final_masks = masks  # [batch_size]
        
        # Q-chunking discount: γ^H instead of γ
        chunking_discount = discount ** horizon_length
        
        # Standard Q-chunking bootstrap formula
        bootstrap_targets = final_rewards + chunking_discount * next_q_values * final_masks
        
        if adaptive_horizon:
            # Simple adaptive discounting based on reward magnitude
            # Higher rewards get slightly reduced discounting
            adaptive_discount = chunking_discount * (1.0 - 0.1 * jnp.tanh(jnp.abs(final_rewards)))
            bootstrap_targets = final_rewards + adaptive_discount * next_q_values * final_masks
            
        return bootstrap_targets
    
    def compute_multi_step_bootstrap_targets(
        self,
        rewards: jnp.ndarray,  # Shape: [batch, n_steps]
        next_q_values: jnp.ndarray,
        masks: jnp.ndarray,  # Shape: [batch, n_steps]
        discount: float = 0.99,
        lambda_coeff: float = 0.95
    ) -> jnp.ndarray:
        """
        Compute multi-step bootstrap targets with TD(λ).
        
        Args:
            rewards: Multi-step rewards
            next_q_values: Q-values for final next states
            masks: Step-wise termination masks
            discount: Discount factor
            lambda_coeff: TD(λ) mixing coefficient
            
        Returns:
            Multi-step bootstrap targets
        """
        n_steps = rewards.shape[1]
        
        # Compute n-step returns
        discounted_rewards = rewards * (discount ** jnp.arange(n_steps)[None, :])
        
        # Apply masks to handle episode boundaries
        masked_rewards = discounted_rewards * masks
        
        # Sum discounted rewards over steps
        multi_step_return = jnp.sum(masked_rewards, axis=1)
        
        # Add final bootstrap term
        final_mask = jnp.prod(masks, axis=1)  # Only if all steps are valid
        final_discount = discount ** n_steps
        bootstrap_term = final_discount * next_q_values * final_mask
        
        # TD(λ) mixing (simplified)
        if lambda_coeff < 1.0:
            # Mix with single-step return
            single_step_target = rewards[:, 0] + discount * next_q_values * masks[:, 0]
            multi_step_target = multi_step_return + bootstrap_term
            
            targets = lambda_coeff * multi_step_target + (1 - lambda_coeff) * single_step_target
        else:
            targets = multi_step_return + bootstrap_term
            
        return targets