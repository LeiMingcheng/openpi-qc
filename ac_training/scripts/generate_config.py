#!/usr/bin/env python3
"""
Configuration generator for ACRLPD + π₀ training.

This script helps generate and customize training configurations for different
robot platforms and experimental settings.

Usage:
    # Generate default DROID config
    python generate_config.py --platform droid --output droid_config.json

    # Generate ALOHA config with custom parameters
    python generate_config.py --platform aloha --horizon_length 20 \
        --bc_alpha 0.001 --output aloha_custom.json

    # Generate config for hyperparameter sweep
    python generate_config.py --platform libero --sweep \
        --param_ranges horizon_length:5,10,15 bc_alpha:0.001,0.01,0.1
"""

import argparse
import json
import dataclasses
from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.acrlpd_pi0_agent import (
    ACRLPDPi0Config, get_droid_config, get_aloha_config, get_libero_config
)
from agents.loss_functions import LossWeights
from agents.critic_networks import CriticConfig
from training.training_loop import TrainingConfig


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate ACRLPD + π₀ training configurations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Platform selection
    parser.add_argument(
        "--platform", type=str, required=True,
        choices=["droid", "aloha", "libero", "custom"],
        help="Robot platform to generate config for"
    )
    
    # Output options
    parser.add_argument(
        "--output", type=str, default="config.json",
        help="Output file path for the configuration"
    )
    parser.add_argument(
        "--format", type=str, default="json",
        choices=["json", "yaml", "python"],
        help="Output format"
    )
    
    # Config overrides
    parser.add_argument(
        "--horizon_length", type=int, default=None,
        help="Override action chunking horizon"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--bc_alpha", type=float, default=None,
        help="Override BC loss weight"
    )
    parser.add_argument(
        "--pi0_lr", type=float, default=None,
        help="Override π₀ learning rate"
    )
    parser.add_argument(
        "--critic_lr", type=float, default=None,
        help="Override critic learning rate"
    )

    # Hyperparameter sweep
    parser.add_argument(
        "--sweep", action="store_true",
        help="Generate configs for hyperparameter sweep"
    )
    parser.add_argument(
        "--param_ranges", type=str, nargs="*",
        help="Parameter ranges for sweep (format: param:val1,val2,val3)"
    )
    parser.add_argument(
        "--sweep_dir", type=str, default="./sweep_configs",
        help="Directory for sweep configurations"
    )
    
    # Additional options
    parser.add_argument(
        "--include_comments", action="store_true",
        help="Include explanatory comments in output"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate generated configuration"
    )
    
    return parser


def apply_overrides(config: ACRLPDPi0Config, args: argparse.Namespace) -> ACRLPDPi0Config:
    """Apply command-line overrides to configuration."""
    updates = {}
    
    if args.horizon_length is not None:
        updates['horizon_length'] = args.horizon_length
    
    if args.batch_size is not None:
        updates['batch_size'] = args.batch_size
    
    if args.pi0_lr is not None:
        updates['pi0_learning_rate'] = args.pi0_lr
    
    if args.critic_lr is not None:
        updates['critic_learning_rate'] = args.critic_lr
    

    # Update loss weights
    loss_updates = {}
    if args.bc_alpha is not None:
        loss_updates['bc_weight'] = args.bc_alpha
    
    if loss_updates:
        updates['loss_weights'] = dataclasses.replace(config.loss_weights, **loss_updates)
    
    if updates:
        config = dataclasses.replace(config, **updates)
    
    return config


def config_to_dict(config: ACRLPDPi0Config) -> Dict[str, Any]:
    """Convert config to dictionary for serialization."""
    
    def serialize_value(value):
        if dataclasses.is_dataclass(value):
            return {f.name: serialize_value(getattr(value, f.name)) for f in dataclasses.fields(value)}
        elif hasattr(value, '__dict__'):
            return {k: serialize_value(v) for k, v in value.__dict__.items()}
        else:
            return value
    
    return serialize_value(config)


def generate_sweep_configs(
    base_config: ACRLPDPi0Config,
    param_ranges: List[str],
    output_dir: Path
) -> List[Path]:
    """Generate configurations for hyperparameter sweep."""
    import itertools
    
    # Parse parameter ranges
    param_dict = {}
    for param_range in param_ranges:
        param_name, values_str = param_range.split(':')
        values = []
        for val_str in values_str.split(','):
            # Try to parse as number, fallback to string
            try:
                if '.' in val_str:
                    values.append(float(val_str))
                else:
                    values.append(int(val_str))
            except ValueError:
                values.append(val_str)
        param_dict[param_name] = values
    
    # Generate all combinations
    param_names = list(param_dict.keys())
    param_values = list(param_dict.values())
    
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []
    
    for i, combination in enumerate(itertools.product(*param_values)):
        # Create config with current parameter combination
        config = base_config
        updates = {}
        
        for param_name, param_value in zip(param_names, combination):
            if param_name == 'bc_alpha':
                updates['loss_weights'] = dataclasses.replace(
                    config.loss_weights, bc_weight=param_value
                )
            else:
                updates[param_name] = param_value
        
        if updates:
            config = dataclasses.replace(config, **updates)
        
        # Generate filename
        param_str = "_".join(f"{name}_{value}" for name, value in zip(param_names, combination))
        filename = f"sweep_{i:03d}_{param_str}.json"
        filepath = output_dir / filename
        
        # Save configuration
        config_dict = config_to_dict(config)
        config_dict['_sweep_params'] = dict(zip(param_names, combination))
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        generated_files.append(filepath)
        print(f"Generated: {filepath}")
    
    return generated_files


def save_config(
    config: ACRLPDPi0Config,
    output_path: Path,
    format_type: str,
    include_comments: bool = False
):
    """Save configuration to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format_type == "json":
        config_dict = config_to_dict(config)
        
        if include_comments:
            config_dict['_comments'] = {
                'horizon_length': 'Number of time steps in each action chunk',
                'bc_alpha': 'Weight for behavior cloning loss regularization',
                'best_of_n_samples': 'Number of action candidates for Best-of-N sampling',
                'pi0_learning_rate': 'Learning rate for π₀ model parameters',
                'critic_learning_rate': 'Learning rate for critic networks'
            }
        
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    elif format_type == "yaml":
        try:
            import yaml
            config_dict = config_to_dict(config)
            with open(output_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        except ImportError:
            print("PyYAML not available, falling back to JSON")
            output_path = output_path.with_suffix('.json')
            save_config(config, output_path, "json", include_comments)
    
    elif format_type == "python":
        # Generate Python code for the configuration
        python_code = f"""# Generated ACRLPD + π₀ configuration
from agents.acrlpd_pi0_agent import ACRLPDPi0Config
from agents.loss_functions import LossWeights
from agents.critic_networks import CriticConfig
import openpi.models.pi0 as _pi0

# Create configuration
config = ACRLPDPi0Config(
    # Core ACRLPD parameters
    horizon_length={config.horizon_length},
    discount={config.discount},
    batch_size={config.batch_size},
    
    # Loss weighting
    loss_weights=LossWeights(
        critic_weight={config.loss_weights.critic_weight},
        actor_weight={config.loss_weights.actor_weight},
        bc_weight={config.loss_weights.bc_weight},
        alpha_weight={config.loss_weights.alpha_weight}
    ),
    
    # Learning rates
    pi0_learning_rate={config.pi0_learning_rate},
    critic_learning_rate={config.critic_learning_rate},
        
    # Sampling configuration
    best_of_n_samples={config.best_of_n_samples},
    diffusion_steps={config.diffusion_steps},
    use_best_of_n={config.use_best_of_n},
    
    # Temperature control
    use_adaptive_temperature={config.use_adaptive_temperature},
    initial_temperature={config.initial_temperature},
    target_entropy_multiplier={config.target_entropy_multiplier}
)
"""
        
        with open(output_path, 'w') as f:
            f.write(python_code)


def main():
    """Main function."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    print(f"Generating {args.platform} configuration...")
    
    # Create base configuration
    if args.platform == "droid":
        config = get_droid_config()
    elif args.platform == "aloha":
        config = get_aloha_config()
    elif args.platform == "libero":
        config = get_libero_config()
    else:  # custom
        config = ACRLPDPi0Config()
    
    # Apply overrides
    config = apply_overrides(config, args)
    
    # Validate if requested
    if args.validate:
        try:
            config.validate()
            print("✓ Configuration validation passed")
        except Exception as e:
            print(f"✗ Configuration validation failed: {e}")
            return 1
    
    # Generate sweep configs if requested
    if args.sweep:
        if not args.param_ranges:
            print("Error: --param_ranges required for sweep mode")
            return 1
        
        sweep_dir = Path(args.sweep_dir)
        generated_files = generate_sweep_configs(config, args.param_ranges, sweep_dir)
        print(f"Generated {len(generated_files)} sweep configurations in {sweep_dir}")
        
        # Also save the base config
        base_config_path = sweep_dir / "base_config.json"
        save_config(config, base_config_path, "json", args.include_comments)
        print(f"Base configuration saved to {base_config_path}")
    
    else:
        # Save single configuration
        output_path = Path(args.output)
        save_config(config, output_path, args.format, args.include_comments)
        print(f"Configuration saved to {output_path}")
    
    # Print summary
    print(f"\nConfiguration summary:")
    print(f"  Platform: {args.platform}")
    print(f"  Horizon length: {config.horizon_length}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  BC alpha: {config.loss_weights.bc_weight}")
    print(f"  Best-of-N samples: {config.best_of_n_samples}")
    
    return 0


if __name__ == "__main__":
    exit(main())