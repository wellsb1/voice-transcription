"""Menu bar widget for transcription control and monitoring."""

import glob
import json
import os
import subprocess
import threading
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import rumps
import yaml

# PyObjC for SF Symbols
from AppKit import NSImage
from Foundation import NSLog


def sf_symbol(name: str, size: float = 18.0) -> NSImage:
    """Get an SF Symbol as NSImage."""
    from AppKit import NSFontWeightRegular, NSImageSymbolConfiguration

    img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(name, None)
    if img:
        # Configure size
        config = NSImageSymbolConfiguration.configurationWithPointSize_weight_(size, NSFontWeightRegular)
        img = img.imageWithSymbolConfiguration_(config)
        img.setTemplate_(True)  # Makes it adapt to light/dark menu bar
    return img


class TranscribeWidget(rumps.App):
    """Menu bar app for controlling transcription pipeline."""

    # SF Symbol names
    ICON_OFF = "waveform"
    ICON_ON = "waveform.circle.fill"

    def __init__(self):
        # Start with text, will set icon after app starts
        super().__init__("~", quit_button=None)
        self._icon_initialized = False

        # Get script directory (where transcribe lives)
        self.script_dir = Path(__file__).parent.parent.parent

        # Load widget config
        self.widget_config = self._load_widget_config()
        self.stats_window_minutes = self.widget_config.get("stats_window_minutes", 5)
        self.auto_start = self.widget_config.get("auto_start", None)

        # Current config being used
        self.current_config: Optional[str] = None

        # Process state
        self.process: Optional[subprocess.Popen] = None
        self.reader_thread: Optional[threading.Thread] = None
        self.running = False

        # Stats tracking: list of (timestamp, word_count) tuples
        self.word_counts: deque = deque()
        self.lock = threading.Lock()

        # Discover available configs
        self.configs = self._discover_configs()

        # Build menu
        self._build_menu()

        # Timer for updating stats display
        self.timer = rumps.Timer(self.update_stats_display, 1)
        self.timer.start()

        # Auto-start if configured (delayed to let UI initialize)
        if self.auto_start:
            self.auto_start_timer = rumps.Timer(self._do_auto_start, 1)
            self.auto_start_timer.start()

    def _discover_configs(self) -> list[str]:
        """Discover available config-backend-*.yaml files."""
        pattern = str(self.script_dir / "config-backend-*.yaml")
        configs = sorted(glob.glob(pattern))
        return [os.path.basename(c) for c in configs]

    def _resolve_config(self, name: str) -> Optional[str]:
        """
        Resolve a config name to full filename.

        Accepts:
            - Full name: "config-backend-m4.yaml"
            - Short name: "m4"

        Returns:
            Full config filename or None if not found
        """
        # Already a full name
        if name in self.configs:
            return name

        # Try as short name
        full_name = f"config-backend-{name}.yaml"
        if full_name in self.configs:
            return full_name

        return None

    def _do_auto_start(self, timer):
        """Auto-start with configured config (runs once)."""
        timer.stop()

        config = self._resolve_config(self.auto_start)
        if config:
            NSLog("TranscribeWidget: Auto-starting with %@", config)
            self.start(config)
        else:
            NSLog("TranscribeWidget: Auto-start config not found: %@", self.auto_start)

    def _build_menu(self):
        """Build the menu with discovered configs."""
        # Stats item (non-clickable)
        self.stats_item = rumps.MenuItem(f"Words ({self.stats_window_minutes}m): 0", callback=None)
        self.stats_item.set_callback(None)

        # Config submenu
        self.config_menu = rumps.MenuItem("Start")
        for config in self.configs:
            # Display name: strip "config-backend-" prefix and ".yaml" suffix
            display_name = config.replace("config-backend-", "").replace(".yaml", "")
            item = rumps.MenuItem(display_name, callback=self._make_start_callback(config))
            self.config_menu[display_name] = item

        # Stop item (hidden initially)
        self.stop_item = rumps.MenuItem("Stop", callback=self.stop)

        # Current config display
        self.current_config_item = rumps.MenuItem("Not running", callback=None)
        self.current_config_item.set_callback(None)

        self.menu = [
            self.config_menu,
            self.stop_item,
            None,  # Separator
            self.current_config_item,
            self.stats_item,
            None,  # Separator
            rumps.MenuItem("Refresh Configs", callback=self.refresh_configs),
            rumps.MenuItem("Quit", callback=self.quit_app),
        ]

        # Initially hide stop, show start
        self.stop_item.set_callback(None)  # Disable initially

    def _make_start_callback(self, config: str):
        """Create a callback for starting with a specific config."""
        def callback(_):
            self.start(config)
        return callback

    def refresh_configs(self, _):
        """Refresh the list of available configs."""
        self.configs = self._discover_configs()
        # Update submenu
        self.config_menu.clear()
        for config in self.configs:
            display_name = config.replace("config-backend-", "").replace(".yaml", "")
            item = rumps.MenuItem(display_name, callback=self._make_start_callback(config))
            self.config_menu[display_name] = item
        NSLog("TranscribeWidget: Refreshed configs, found %d", len(self.configs))

    def _set_icon(self, symbol_name: str):
        """Set menu bar icon using SF Symbol."""
        try:
            button = self._nsapp.nsstatusitem.button()
            button.setImage_(sf_symbol(symbol_name))
            self.title = ""  # Clear text when using icon
        except AttributeError:
            # App not fully initialized yet
            pass

    def _set_icon_off(self):
        """Set icon to inactive state (waveform)."""
        self._set_icon(self.ICON_OFF)

    def _set_icon_on(self):
        """Set icon to active/recording state (waveform with circle)."""
        self._set_icon(self.ICON_ON)

    def _load_widget_config(self) -> dict:
        """Load widget config from TRANSCRIBE_WIDGET_CONFIG or ./config-widget.yaml"""
        env_path = os.environ.get("TRANSCRIBE_WIDGET_CONFIG")
        if env_path:
            config_path = Path(env_path)
        else:
            config_path = self.script_dir / "config-widget.yaml"

        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f) or {}
        return {}

    def start(self, config: str):
        """Start the transcription pipeline with the given config."""
        if self.running:
            self.stop(None)

        self.current_config = config
        command = f"{self.script_dir}/transcribe --config={config}"

        NSLog("TranscribeWidget: Starting with config: %@", config)

        try:
            self.process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=str(self.script_dir),
            )
            self.running = True
            self._set_icon_on()

            # Update menu state
            self.config_menu.title = "Running..."
            self.stop_item.set_callback(self.stop)
            display_name = config.replace("config-backend-", "").replace(".yaml", "")
            self.current_config_item.title = f"Config: {display_name}"

            # Start reader threads
            self.reader_thread = threading.Thread(target=self._read_output, daemon=True)
            self.reader_thread.start()

            self.stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
            self.stderr_thread.start()

            NSLog("TranscribeWidget: Started successfully")

        except Exception as e:
            NSLog("TranscribeWidget: Failed to start: %@", str(e))
            rumps.alert("Error", f"Failed to start: {e}")

    def stop(self, _):
        """Stop the transcription pipeline."""
        if not self.running:
            return

        NSLog("TranscribeWidget: Stopping...")
        self.running = False

        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

        self._set_icon_off()
        self.config_menu.title = "Start"
        self.stop_item.set_callback(None)
        self.current_config_item.title = "Not running"
        self.current_config = None
        NSLog("TranscribeWidget: Stopped")

    def _read_stderr(self):
        """Read stderr from process for debugging."""
        if not self.process or not self.process.stderr:
            return

        for line in self.process.stderr:
            if not self.running:
                break
            NSLog("TranscribeWidget stderr: %@", line.strip())

    def _read_output(self):
        """Read stdout from process and parse JSONL for stats."""
        if not self.process or not self.process.stdout:
            return

        for line in self.process.stdout:
            if not self.running:
                break

            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                speaker = data.get("speaker", "")
                text = data.get("text", "")
                word_count = len(text.split())

                with self.lock:
                    self.word_counts.append((datetime.now(), word_count))

                # Output simplified format to stdout
                if text:
                    print(f"[{speaker}] {text}", flush=True)

            except json.JSONDecodeError:
                # Not valid JSON, skip
                pass

    def _check_process_alive(self):
        """Check if the subprocess is still running."""
        if self.running and self.process:
            exit_code = self.process.poll()
            if exit_code is not None:
                # Process died
                NSLog("TranscribeWidget: Process exited with code %d", exit_code)
                self.running = False
                self.process = None
                self._set_icon_off()
                self.config_menu.title = "Start"
                self.stop_item.set_callback(None)
                self.current_config_item.title = "Not running"

    def update_stats_display(self, _):
        """Update the stats display in the menu."""
        # Initialize icon on first timer tick (app is now fully loaded)
        if not self._icon_initialized:
            self._set_icon_off()
            self._icon_initialized = True

        # Check if process died
        self._check_process_alive()

        now = datetime.now()
        cutoff = now - timedelta(minutes=self.stats_window_minutes)

        with self.lock:
            # Remove old entries
            while self.word_counts and self.word_counts[0][0] < cutoff:
                self.word_counts.popleft()

            # Sum recent words
            total_words = sum(count for _, count in self.word_counts)

        self.stats_item.title = f"Words ({self.stats_window_minutes}m): {total_words}"

        # Show word count next to icon if running and has words
        if self.running and total_words > 0:
            self.title = str(total_words)
        elif self.running:
            self.title = ""
        else:
            self.title = ""

    def quit_app(self, _):
        """Clean shutdown."""
        self.stop(None)
        rumps.quit_application()


def main():
    # Hide dock icon (run as menu bar accessory)
    from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
    NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    TranscribeWidget().run()


if __name__ == "__main__":
    main()
