#!/usr/bin/env python3
"""
[DEPRECATED] Configuration generator for ACRLPD + π₀ training.

⚠️  This script is deprecated after the agents_v2 refactor.
    
    The ACRLPDPi0Config class has been replaced with parameterized constructors
    in agents_v2. Configuration is now handled through:
    - config.py: RLTrainConfig with nested dataclass structure
    - Direct agent creation via create_acrlpd_pi0_agent_from_rl_config()
    
    For configuration examples, see:
    - config.py: get_config() function with predefined configs
    - scripts/train_acrlpd_pi0.py: Usage of unified configuration system
    
Usage (legacy):
    This script was used to generate JSON configurations for the old v1 system.
    It is preserved for reference but should not be used in production.
"""

import sys
from pathlib import Path

def main():
    print("⚠️  DEPRECATED: This script is deprecated after agents_v2 refactor.")
    print("")
    print("📝 Please use the new configuration system instead:")
    print("   - Edit config.py to modify training parameters")
    print("   - Use scripts/train_acrlpd_pi0.py with --config parameter")
    print("")
    print("Examples:")
    print("   python scripts/train_acrlpd_pi0.py --config rl_fold_box")
    print("   python scripts/train_acrlpd_pi0.py --config aloha_real")
    print("")
    print("For custom configs, modify config.py and add your configuration.")
    sys.exit(1)

if __name__ == "__main__":
    main()