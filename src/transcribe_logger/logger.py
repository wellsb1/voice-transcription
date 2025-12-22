"""Transcript logger with file rollover - passes through stdin to stdout."""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from transcribe_shared import load_yaml


def load_config() -> dict:
    """Load config from TRANSCRIBE_LOGGER_CONFIG or config-backend-m4.yaml."""
    env_path = os.environ.get("TRANSCRIBE_LOGGER_CONFIG")
    if env_path:
        config_path = Path(env_path)
    else:
        config_path = Path("config-backend-m4.yaml")

    return load_yaml(config_path)


class TranscriptLogger:
    """Logs JSONL transcripts to rolling files while passing through to stdout."""

    def __init__(
        self,
        output_dir: Path,
        device_name: str = "m4-mini",
        max_lines: int = 100,
    ):
        self.output_dir = output_dir
        self.device_name = device_name
        self.max_lines = max_lines

        self.current_file: Optional[Path] = None
        self.current_handle = None
        self.line_count = 0

        # Ensure output dir exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _new_file(self) -> Path:
        """Generate new filename with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return self.output_dir / f"{timestamp}-{self.device_name}.jsonl"

    def _open_new_file(self):
        """Close current file and open a new one."""
        if self.current_handle:
            self.current_handle.close()

        self.current_file = self._new_file()
        self.current_handle = open(self.current_file, "a")
        self.line_count = 0

    def write(self, line: str):
        """Write a line to the current log file and stdout."""
        # Pass through to stdout
        print(line, flush=True)

        # Rollover if needed
        if self.current_handle is None or self.line_count >= self.max_lines:
            self._open_new_file()

        # Write to file
        self.current_handle.write(line + "\n")
        self.current_handle.flush()
        self.line_count += 1

    def close(self):
        """Close the current file."""
        if self.current_handle:
            self.current_handle.close()
            self.current_handle = None


def main():
    """Read from stdin, log to files, pass through to stdout."""
    config = load_config()

    # Get settings from config
    device_name = config.get("device_name", "m4-mini")
    logs_config = config.get("logs", {})
    output_dir = Path(logs_config.get("output_dir", ".transcripts"))
    max_lines = logs_config.get("max_lines", 100)

    logger = TranscriptLogger(
        output_dir=output_dir,
        device_name=device_name,
        max_lines=max_lines,
    )

    try:
        for line in sys.stdin:
            line = line.rstrip("\n")
            if line:
                logger.write(line)
    except KeyboardInterrupt:
        pass
    finally:
        logger.close()


if __name__ == "__main__":
    main()
