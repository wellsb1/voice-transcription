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

from transcribe_shared.config import load_yaml

from .jarvis import JarvisConfig, JarvisListener, JarvisMode

# PyObjC for SF Symbols
from AppKit import NSImage
from Foundation import NSLog


def sf_symbol(name: str, size: float = 18.0, color=None) -> NSImage:
    """Get an SF Symbol as NSImage, optionally tinted."""
    from AppKit import NSFontWeightRegular, NSImageSymbolConfiguration

    img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(name, None)
    if img:
        # Configure size
        config = NSImageSymbolConfiguration.configurationWithPointSize_weight_(size, NSFontWeightRegular)
        if color:
            color_config = NSImageSymbolConfiguration.configurationWithHierarchicalColor_(color)
            config = config.configurationByApplyingConfiguration_(color_config)
        img = img.imageWithSymbolConfiguration_(config)
        img.setTemplate_(color is None)  # Template only when no color
    return img


class TranscribeWidget(rumps.App):
    """Menu bar app for controlling transcription pipeline."""

    # SF Symbol names
    ICON_OFF = "waveform"
    ICON_ON = "waveform.circle.fill"
    ICON_CAPTURE = "mic.fill"

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

        # Set up Jarvis config (before menu build, which references jarvis_enabled)
        jarvis_cfg = self.widget_config.get("jarvis", {})
        self.jarvis_enabled = jarvis_cfg.get("enabled", False)
        self.jarvis: Optional[JarvisListener] = None

        if self.jarvis_enabled:
            self.jarvis_config = JarvisConfig(
                audio_device=jarvis_cfg.get("audio_device"),
                sample_rate=jarvis_cfg.get("sample_rate", 16000),
                wakeword_models=jarvis_cfg.get("wakeword_models", []),
                wakeword_threshold=jarvis_cfg.get("wakeword_threshold", 0.5),
                sound_trigger=jarvis_cfg.get("sound_trigger", "/System/Library/Sounds/Tink.aiff"),
                sound_done=jarvis_cfg.get("sound_done", "/System/Library/Sounds/Pop.aiff"),
                sound_cancel=jarvis_cfg.get("sound_cancel", "/System/Library/Sounds/Funk.aiff"),
                capture_max_duration=jarvis_cfg.get("capture_max_duration", 30.0),
                capture_silence_duration=jarvis_cfg.get("capture_silence_duration", 1.5),
                capture_silence_threshold=jarvis_cfg.get("capture_silence_threshold", 0.005),
                whisper_model=jarvis_cfg.get("whisper_model", "mlx-community/whisper-large-v3-turbo"),
                builtin_commands=jarvis_cfg.get("builtin_commands", {
                    "start_transcribing": ["start transcribing", "begin transcribing", "start recording"],
                    "stop_transcribing": ["stop transcribing", "stop recording"],
                }),
                default_transcription_config=jarvis_cfg.get("default_transcription_config", "m4"),
                catchall_command=jarvis_cfg.get("catchall_command"),
            )

        # Discover available configs
        self.configs = self._discover_configs()

        # Build menu
        self._build_menu()

        # Timer for updating stats display
        self.timer = rumps.Timer(self.update_stats_display, 1)
        self.timer.start()

        # Auto-start if configured (delayed to let UI initialize)
        self._start_timer = rumps.Timer(self._delayed_start, 2)
        self._start_timer.start()

    # Backend descriptions for menu display
    BACKEND_DESCRIPTIONS = {
        "m4": "Large model, Apple Silicon, speaker diarization",
        "sherpa": "Small model, lightweight, CPU",
        "whisper": "Medium model, general purpose",
    }

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

    def _delayed_start(self, timer):
        """Delayed startup for Jarvis and auto-start (runs once)."""
        timer.stop()

        # Start Jarvis if configured
        if self.jarvis_enabled and self.jarvis_config.wakeword_models:
            try:
                self.jarvis = JarvisListener(
                    config=self.jarvis_config,
                    on_builtin_command=self._handle_jarvis_command,
                    on_mode_change=self._handle_jarvis_mode_change,
                )
                self.jarvis.start()
                NSLog("TranscribeWidget: Jarvis started")
            except Exception as e:
                NSLog("TranscribeWidget: Failed to start Jarvis: %@", str(e))

        # Auto-start transcription if configured
        if self.auto_start:
            config = self._resolve_config(self.auto_start)
            if config:
                NSLog("TranscribeWidget: Auto-starting with %@", config)
                self.start(config)
            else:
                NSLog("TranscribeWidget: Auto-start config not found: %@", self.auto_start)

    def _handle_jarvis_command(self, command: str):
        """Handle built-in voice commands from Jarvis (called from background thread)."""
        from PyObjCTools import AppHelper

        if command == "start_transcribing":
            config_name = self.jarvis_config.default_transcription_config if self.jarvis_enabled else "m4"
            config = self._resolve_config(config_name)
            if config:
                AppHelper.callAfter(lambda: self.start(config))
        elif command == "stop_transcribing":
            AppHelper.callAfter(lambda: self.stop(None))

    def _handle_jarvis_mode_change(self, mode: JarvisMode):
        """Handle Jarvis mode changes (called from background thread)."""
        from PyObjCTools import AppHelper

        if mode == JarvisMode.CAPTURE:
            AppHelper.callAfter(self._set_icon_capture)
        elif mode == JarvisMode.LISTENING:
            # Restore appropriate icon based on transcription state
            if self.running:
                AppHelper.callAfter(self._set_icon_on)
            else:
                AppHelper.callAfter(self._set_icon_off)

    def _build_menu(self):
        """Build the menu with discovered configs."""
        # Stats item (non-clickable)
        self.stats_item = rumps.MenuItem(f"Words ({self.stats_window_minutes}m): 0", callback=None)
        self.stats_item.set_callback(None)

        # Config submenu
        self.config_menu = rumps.MenuItem("Start")
        for config in self.configs:
            # Display name: strip "config-backend-" prefix and ".yaml" suffix
            short_name = config.replace("config-backend-", "").replace(".yaml", "")
            desc = self.BACKEND_DESCRIPTIONS.get(short_name, "")
            display_name = f"{short_name} \u2014 {desc}" if desc else short_name
            item = rumps.MenuItem(display_name, callback=self._make_start_callback(config))
            self.config_menu[display_name] = item

        # Stop item (hidden initially)
        self.stop_item = rumps.MenuItem("Stop", callback=self.stop)

        # Current config display
        self.current_config_item = rumps.MenuItem("Not running", callback=None)
        self.current_config_item.set_callback(None)

        # Jarvis toggle
        jarvis_label = "Jarvis: On" if self.jarvis_enabled else "Jarvis: Off"
        self.jarvis_toggle = rumps.MenuItem(jarvis_label, callback=self._toggle_jarvis)

        self.menu = [
            self.config_menu,
            self.stop_item,
            None,  # Separator
            self.jarvis_toggle,
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
            short_name = config.replace("config-backend-", "").replace(".yaml", "")
            desc = self.BACKEND_DESCRIPTIONS.get(short_name, "")
            display_name = f"{short_name} \u2014 {desc}" if desc else short_name
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

    def _set_icon_capture(self):
        """Set icon to capture/trigger mode (blue tint)."""
        try:
            from AppKit import NSColor
            button = self._nsapp.nsstatusitem.button()
            button.setImage_(sf_symbol(self.ICON_CAPTURE, color=NSColor.whiteColor()))
            self.title = ""
        except AttributeError:
            pass

    def _load_widget_config(self) -> dict:
        """Load widget config from TRANSCRIBE_WIDGET_CONFIG or ./config-widget.yaml"""
        env_path = os.environ.get("TRANSCRIBE_WIDGET_CONFIG")
        if env_path:
            config_path = Path(env_path)
        else:
            config_path = self.script_dir / "config-widget.yaml"

        return load_yaml(config_path)

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
                start_new_session=True,
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
            # Send SIGTERM to entire process group so the pipeline
            # (backend | logger | plugins) all get the signal and can flush
            import signal
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.process = None

        self._set_icon_off()
        self.config_menu.title = "Start"
        self.stop_item.set_callback(None)
        self.current_config_item.title = "Not running"
        self.current_config = None
        NSLog("TranscribeWidget: Stopped")

    def _toggle_jarvis(self, sender):
        """Toggle Jarvis on/off."""
        if self.jarvis and self.jarvis.mode != JarvisMode.DISABLED:
            self.jarvis.stop()
            self.jarvis = None
            sender.title = "Jarvis: Off"
            NSLog("TranscribeWidget: Jarvis disabled")
        elif self.jarvis_enabled and self.jarvis_config.wakeword_models:
            try:
                self.jarvis = JarvisListener(
                    config=self.jarvis_config,
                    on_builtin_command=self._handle_jarvis_command,
                    on_mode_change=self._handle_jarvis_mode_change,
                )
                self.jarvis.start()
                sender.title = "Jarvis: On"
                NSLog("TranscribeWidget: Jarvis enabled")
            except Exception as e:
                NSLog("TranscribeWidget: Failed to start Jarvis: %@", str(e))

    def _read_stderr(self):
        """Read stderr from process for debugging."""
        if not self.process or not self.process.stderr:
            return

        for line in self.process.stderr:
            if not self.running:
                break
            NSLog("TranscribeWidget stderr: %@", line.strip())

    def _read_output(self):
        """Read stdout from process and parse JSONL batch envelopes for stats."""
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

                # Batch envelope format: {"id", "timestamp", "device", "utterances": [...]}
                utterances = data.get("utterances", [])

                # Backwards compatible: flat array of utterance objects
                if isinstance(data, list):
                    utterances = data

                for item in utterances:
                    speaker = item.get("speaker", "")
                    text = item.get("text", "")
                    word_count = len(text.split())

                    with self.lock:
                        self.word_counts.append((datetime.now(), word_count))

                    if text:
                        NSLog("TranscribeWidget: [%@] %d words", speaker, word_count)

            except json.JSONDecodeError:
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
                self._set_icon_off()  # Back to listening/off state
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
        if self.jarvis:
            self.jarvis.stop()
        self.stop(None)
        rumps.quit_application()


def main():
    # Hide dock icon (run as menu bar accessory)
    from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
    NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    TranscribeWidget().run()


if __name__ == "__main__":
    main()
