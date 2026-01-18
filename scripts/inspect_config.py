"""
Utility script to inspect and optionally modify the pipeline_config.pkl file.

Usage:
    python scripts/inspect_config.py                    # View config
    python scripts/inspect_config.py --fix              # Remove problematic keys
    python scripts/inspect_config.py --path /custom/path/pipeline_config.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load the pickle config file."""
    with open(config_path, "rb") as f:
        return pickle.load(f)


def save_config(config_path: str, config: dict) -> None:
    """Save the pickle config file."""
    with open(config_path, "wb") as f:
        pickle.dump(config, f)


def main():
    parser = argparse.ArgumentParser(description="Inspect or fix pipeline_config.pkl")
    parser.add_argument(
        "--path",
        default="./src/whisper_model_cache/pipeline_config.pkl",
        help="Path to the config file",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Remove device-specific keys (torch_dtype, device, batch_size)",
    )
    args = parser.parse_args()

    config_path = Path(args.path)
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    # Load and display current config
    config = load_config(config_path)
    print("Current configuration:")
    print("-" * 40)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("-" * 40)

    if args.fix:
        # Keys that should be determined at runtime, not stored in config
        keys_to_remove = ["torch_dtype", "device", "batch_size"]
        removed = []
        
        for key in keys_to_remove:
            if key in config:
                del config[key]
                removed.append(key)
        
        if removed:
            save_config(config_path, config)
            print(f"\nRemoved keys: {removed}")
            print("Updated configuration:")
            print("-" * 40)
            for key, value in config.items():
                print(f"  {key}: {value}")
            print("-" * 40)
        else:
            print("\nNo problematic keys found. Config is clean.")


if __name__ == "__main__":
    main()
