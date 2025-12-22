"""Main entry point that routes to the correct backend based on config."""

import os
import sys
from pathlib import Path

import yaml


def load_backend_config() -> str:
    """Load backend selection from config.yaml or environment."""
    # Check environment variable first
    backend = os.environ.get("TRANSCRIBE_BACKEND")
    if backend:
        return backend

    # Check config.yaml
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
            backend = config.get("backend")
            if backend:
                return backend

    # Default to m4
    return "m4"


def main() -> int:
    """Route to the correct backend."""
    backend = load_backend_config()
    backend = backend.lower()

    print(f"Launching backend: {backend}", file=sys.stderr)

    if backend == "m4":
        from transcribe_m4.__main__ import main as backend_main
    elif backend == "whisper":
        from transcribe_whisper.__main__ import main as backend_main
    elif backend == "sherpa":
        from transcribe_sherpa.__main__ import main as backend_main
    else:
        print(f"Unknown backend: {backend}", file=sys.stderr)
        print("Valid backends: m4, whisper, sherpa", file=sys.stderr)
        return 1

    return backend_main()


if __name__ == "__main__":
    sys.exit(main())
