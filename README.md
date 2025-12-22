# Voice Transcription System

A modular, always-on voice transcription system with speaker diarization. Captures audio from a USB microphone, transcribes speech locally using Whisper, identifies speakers, and outputs JSONL for downstream processing.

## Features

- **Multiple backend support**: M4 (Apple Silicon), faster-whisper (CPU), sherpa-onnx (cross-platform)
- **Speaker diarization**: Identifies and tracks speakers across sessions
- **Menu bar widget**: macOS menu bar app for easy control
- **Transcript logging**: Rolling file storage with configurable retention
- **Unix pipeline design**: Composable components, pipe-friendly output
- **Privacy-focused**: Audio is never stored, only text

## Quick Start

```bash
# Clone and enter directory
git clone https://github.com/wellsb1/voice-transcription.git
cd voice-transcription

# Create .env with your settings
cat > .env << EOF
HUGGINGFACE_TOKEN=your_token_here
DEVICE_NAME=my_office
EOF

# Run the M4 backend (Apple Silicon)
./transcribe --config=config-backend-m4.yaml

# Or use the menu bar widget
./transcribe-widget
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRANSCRIBE SCRIPT                         │
│  ./transcribe --config=config-backend-{backend}.yaml            │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────┐   ┌─────────────────┐   ┌───────────────┐  │
│  │ transcribe-m4   │   │transcribe-whisper│  │transcribe-sherpa│ │
│  │ (Apple Silicon) │   │ (CPU/CUDA)      │   │ (ONNX Runtime) │  │
│  └────────┬────────┘   └────────┬────────┘   └───────┬───────┘  │
│           │                     │                     │          │
│           └─────────────────────┴─────────────────────┘          │
│                              │                                   │
│                              ▼                                   │
│                    ┌─────────────────┐                           │
│                    │ transcribe-logger│                          │
│                    │ (Rolling files)  │                          │
│                    └─────────────────┘                           │
│                              │                                   │
│                              ▼                                   │
│                         JSONL stdout                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       MENU BAR WIDGET                            │
│  ./transcribe-widget                                             │
│                                                                  │
│  • Discovers config-backend-*.yaml files                         │
│  • Start/stop transcription from menu bar                        │
│  • Shows word count statistics                                   │
│  • Runs selected backend piped through logger                    │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### Scripts

| Script | Description |
|--------|-------------|
| `transcribe` | Main launcher - runs backend + logger pipeline |
| `transcribe-backend` | Direct backend execution (no logger) |
| `transcribe-m4` | M4/Apple Silicon backend launcher |
| `transcribe-whisper` | faster-whisper backend launcher |
| `transcribe-sherpa` | sherpa-onnx backend launcher |
| `transcribe-logger` | Rolling file logger (stdin → files + stdout) |
| `transcribe-widget` | macOS menu bar widget |

### Usage Examples

```bash
# Run with default config (config-backend-m4.yaml)
./transcribe

# Run with specific config
./transcribe --config=config-backend-whisper.yaml

# List available configs
./transcribe --list-configs

# Run backend directly (no logging to files)
./transcribe-backend

# Run with debug audio playback (M4 only)
./transcribe-backend --debug

# List audio devices
./transcribe-backend --list-devices

# Run the menu bar widget
./transcribe-widget
```

---

## Backends

### M4 Backend (Apple Silicon)

**Best for**: Mac Mini M4, MacBook Pro M1/M2/M3/M4

Uses MLX-Whisper optimized for Apple Silicon and pyannote for speaker diarization.

**Architecture**:
```
┌──────────────────────────────────────────────────────────────┐
│                      M4 BACKEND                               │
│                                                               │
│  ┌─────────────┐      ┌─────────────┐      ┌──────────────┐  │
│  │   Audio     │      │   Batch     │      │   Pipeline   │  │
│  │  Capture    │─────▶│  Detector   │─────▶│              │  │
│  │ (100ms)     │      │ (Silero VAD)│      │ Diarize +    │  │
│  └─────────────┘      └─────────────┘      │ Transcribe   │  │
│                                            └──────────────┘  │
│                                                    │         │
│  Memory buffer (30-60s batches)                    ▼         │
│  No disk queue - processes in real-time      JSONL stdout   │
└──────────────────────────────────────────────────────────────┘
```

**Key characteristics**:
- In-memory audio buffering (no disk queue)
- Batch processing: 30-60 second chunks for better diarization context
- Uses pyannote for speaker diarization with embeddings
- Cross-batch speaker tracking via embedding similarity

**Models**:
- ASR: `mlx-community/whisper-large-v3-turbo` (MLX-optimized)
- Diarization: `pyannote/speaker-diarization-community-1`

**Dependencies**: `mlx-whisper`, `pyannote.audio`, `torch`, `sounddevice`

---

### Whisper Backend (CPU/CUDA)

**Best for**: Linux servers, Raspberry Pi 5, CUDA-enabled systems

Uses faster-whisper (CTranslate2) for efficient CPU inference.

**Architecture**:
```
┌──────────────────────────────────────────────────────────────┐
│                    WHISPER BACKEND                            │
│                                                               │
│  ┌─────────────┐      ┌─────────────┐      ┌──────────────┐  │
│  │   Audio     │      │    Disk     │      │  Processing  │  │
│  │  Capture    │─────▶│   Queue     │─────▶│   Thread     │  │
│  │ (VAD-aware) │      │  (.npy)     │      │              │  │
│  └─────────────┘      └─────────────┘      │ Whisper +    │  │
│                                            │ SpeechBrain  │  │
│  Chunks on silence boundaries              └──────────────┘  │
│  Disk queue survives crashes                      │          │
│                                                   ▼          │
│                                             JSONL stdout     │
└──────────────────────────────────────────────────────────────┘
```

**Key characteristics**:
- VAD-aware chunking (2-30 second variable chunks)
- Disk-backed queue for crash resilience
- SpeechBrain ECAPA-TDNN for speaker embeddings
- Separate capture and processing threads

**Models**:
- ASR: `medium.en` (faster-whisper, INT8 quantized)
- Speaker embeddings: SpeechBrain ECAPA-TDNN

**Dependencies**: `faster-whisper`, `speechbrain`, `torch`, `sounddevice`

---

### Sherpa Backend (Cross-platform ONNX)

**Best for**: Raspberry Pi, edge devices, Windows

Uses sherpa-onnx for lightweight ONNX Runtime inference.

**Architecture**:
```
┌──────────────────────────────────────────────────────────────┐
│                    SHERPA BACKEND                             │
│                                                               │
│  ┌─────────────┐      ┌─────────────┐      ┌──────────────┐  │
│  │   Audio     │      │    Disk     │      │  Processing  │  │
│  │  Capture    │─────▶│   Queue     │─────▶│   Thread     │  │
│  │ (VAD-aware) │      │  (.npy)     │      │              │  │
│  └─────────────┘      └─────────────┘      │ Sherpa-ONNX  │  │
│                                            │ Whisper      │  │
│  Uses Silero VAD (ONNX)                    └──────────────┘  │
│  Minimal dependencies                             │          │
│                                                   ▼          │
│                                             JSONL stdout     │
└──────────────────────────────────────────────────────────────┘
```

**Key characteristics**:
- Pure ONNX Runtime (no PyTorch required for inference)
- Smallest memory footprint
- Optional INT8 quantization
- Supports CPU, CUDA, CoreML providers

**Models**:
- ASR: `small.en` (sherpa-onnx Whisper format)
- VAD: Silero VAD (ONNX)
- Speaker embeddings: Optional (3D Speaker model)

**Dependencies**: `sherpa-onnx`, `numpy`, `sounddevice`

---

## Configuration

### Config Files

Each backend has its own config file: `config-backend-{backend}.yaml`

```
config-backend-m4.yaml      # Apple Silicon (MLX)
config-backend-whisper.yaml # faster-whisper (CPU/CUDA)
config-backend-sherpa.yaml  # sherpa-onnx (ONNX Runtime)
config-widget.yaml          # Menu bar widget settings
```

### Environment Variables (.env)

Create a `.env` file in the project root for secrets and machine-specific settings:

```bash
# Required for pyannote models
HUGGINGFACE_TOKEN=hf_your_token_here

# Device identifier in JSONL output
DEVICE_NAME=wells_office
```

### Config Override Hierarchy

1. **CLI arguments** (highest priority): `--config=path/to/config.yaml`
2. **Environment variables**: `DEVICE_NAME=kitchen`
3. **YAML interpolation**: `${DEVICE_NAME:-default}`
4. **Dataclass defaults** (lowest priority)

### M4 Backend Config (`config-backend-m4.yaml`)

```yaml
# ASR - mlx-whisper model
whisper_model: "mlx-community/whisper-large-v3-turbo"

# Transcript filtering - words to always ignore
ignore_words: []  # Example: ["acl", "ack", "hmm"]

# Diarization - pyannote model
diarization_model: "pyannote/speaker-diarization-community-1"
huggingface_token: ${HUGGINGFACE_TOKEN:-null}

# Batch parameters
min_batch_duration: 30.0   # Minimum batch for pyannote context (seconds)
max_batch_duration: 60.0   # Force batch during monologues (seconds)
silence_duration: 0.5      # Silence threshold for batch boundary (seconds)
vad_threshold: 0.6         # VAD sensitivity (0.0-1.0, higher = less sensitive)

# Speaker tracking
speaker_similarity_threshold: 0.75   # Cosine similarity for matching
speaker_inactivity_timeout: 1800.0   # Reset speaker IDs after 30 min

# Audio settings
audio_device: null              # null = system default input
audio_output_device: null       # null = system default output (for --debug)
sample_rate: 16000              # 16kHz required for Whisper

# Output
device_name: ${DEVICE_NAME:-default}
```

### Whisper/Sherpa Backend Configs

Similar structure with backend-specific model settings:

```yaml
# Whisper backend
whisper_model: "medium.en"
whisper_compute_type: "int8"  # int8, float16, float32

# Sherpa backend
whisper_model: "small.en"
sherpa_use_int8: false
sherpa_provider: "cpu"  # cpu, cuda, coreml
sherpa_num_threads: 4
models_dir: "./models/sherpa"

# Common settings
vad_threshold: 0.5
silence_duration: 0.5
min_chunk_duration: 2.0
max_chunk_duration: 30.0
min_audio_energy: 0.01
min_speech_duration: 0.4

queue_dir: "./data/queue-{backend}"
```

---

## JSONL Output Format

One JSON object per line:

```json
{"timestamp": "2024-01-15T10:30:45.123456", "device": "wells_office", "speaker": "SPEAKER_00", "confidence": 0.92, "start": 0.0, "end": 2.5, "text": "Hello, how are you?"}
{"timestamp": "2024-01-15T10:30:48.456789", "device": "wells_office", "speaker": "SPEAKER_01", "confidence": 0.87, "start": 2.8, "end": 5.1, "text": "I'm doing well, thanks."}
```

| Field | Description |
|-------|-------------|
| `timestamp` | ISO 8601 timestamp with microseconds |
| `device` | Device/location identifier from config |
| `speaker` | Speaker label (e.g., "SPEAKER_00") |
| `confidence` | Speaker match confidence (0.0-1.0) |
| `start` | Segment start time within batch (seconds) |
| `end` | Segment end time within batch (seconds) |
| `text` | Transcribed text |

---

## Transcript Logger

The logger sits in the pipeline between the backend and stdout. It:
1. Reads JSONL lines from stdin
2. Writes each line to a rolling log file
3. Passes through to stdout (for further piping)

### Rolling File Behavior

- Each file stores up to **100 lines** (configurable via `max_lines`)
- When the limit is reached, the current file is closed and a new timestamped file is created
- Files accumulate in the output directory - they are not automatically deleted
- The logger flushes after every line for crash resilience

### Config

Add a `logs` section to any backend config (`config-backend-*.yaml`):

```yaml
logs:
  output_dir: ".transcripts"  # Directory for log files (default: .transcripts)
  max_lines: 100              # Lines per file before rollover (default: 100)
```

Or set via environment variable:
```bash
TRANSCRIBE_LOGGER_CONFIG=/path/to/config.yaml
```

### File Naming

Files are named with the creation timestamp and device name:

```
.transcripts/
├── 20241222143052-wells_office.jsonl   # First 100 lines
├── 20241222143521-wells_office.jsonl   # Next 100 lines
└── 20241222144003-wells_office.jsonl   # And so on...
```

Format: `{YYYYMMDDHHMMSS}-{device_name}.jsonl`

### Standalone Usage

```bash
# Pipe any backend through the logger
./transcribe-m4 | ./transcribe-logger

# Or use the main transcribe script which does this automatically
./transcribe --config=config-backend-m4.yaml
```

---

## Menu Bar Widget

macOS menu bar application for controlling transcription.

### Features

- Dynamically discovers `config-backend-*.yaml` files
- Start/stop transcription with one click
- Shows real-time word count (configurable window)
- Runs selected backend piped through logger

### Config (`config-widget.yaml`)

```yaml
# Rolling window for word count stats (minutes)
stats_window_minutes: 5
```

### Usage

```bash
./transcribe-widget
```

The widget appears in your menu bar with a waveform icon:
- Click to see available backends
- Select a config to start transcription
- Click "Stop" to stop
- Shows word count while running

---

## Installation

### Prerequisites

**macOS**:
```bash
brew install portaudio ffmpeg
```

**Linux (Debian/Ubuntu)**:
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv libportaudio2 ffmpeg
```

### Setup

Each backend auto-creates its virtual environment on first run:

```bash
# Clone repository
git clone https://github.com/wellsb1/voice-transcription.git
cd voice-transcription

# Create .env
cp .env.example .env  # Edit with your settings

# Run any backend - venv created automatically
./transcribe --config=config-backend-m4.yaml
```

### Manual venv Setup

```bash
# M4 backend
python3 -m venv venv-m4
source venv-m4/bin/activate
pip install numpy torch torchaudio sounddevice pyyaml mlx-whisper pyannote.audio

# Whisper backend
python3 -m venv venv-whisper
source venv-whisper/bin/activate
pip install numpy torch torchaudio sounddevice pyyaml faster-whisper speechbrain

# Sherpa backend
python3 -m venv venv-sherpa
source venv-sherpa/bin/activate
pip install numpy sounddevice pyyaml sherpa-onnx
```

---

## Project Structure

```
voice-transcription/
├── transcribe              # Main launcher (backend + logger pipeline)
├── transcribe-backend      # Direct backend execution
├── transcribe-m4           # M4 backend launcher
├── transcribe-whisper      # Whisper backend launcher
├── transcribe-sherpa       # Sherpa backend launcher
├── transcribe-logger       # Rolling file logger
├── transcribe-widget       # macOS menu bar widget
│
├── config-backend-m4.yaml      # M4 backend config
├── config-backend-whisper.yaml # Whisper backend config
├── config-backend-sherpa.yaml  # Sherpa backend config
├── config-widget.yaml          # Widget config
├── .env                        # Secrets (gitignored)
│
├── src/
│   ├── transcribe_m4/          # M4 backend
│   │   ├── __main__.py         # Entry point
│   │   ├── asr.py              # MLX-Whisper transcription
│   │   ├── batch.py            # Silero VAD batch detection
│   │   ├── capture.py          # Audio capture + buffer
│   │   ├── config.py           # Config loading
│   │   ├── diarize.py          # Pyannote diarization
│   │   ├── output.py           # JSONL formatting
│   │   └── pipeline.py         # Orchestration + speaker registry
│   │
│   ├── transcribe_whisper/     # Whisper backend
│   │   ├── __main__.py
│   │   ├── asr.py              # faster-whisper transcription
│   │   ├── capture.py          # VAD-aware audio capture
│   │   ├── config.py
│   │   ├── embeddings.py       # SpeechBrain ECAPA-TDNN
│   │   ├── output.py
│   │   ├── queue.py            # Disk-backed audio queue
│   │   └── vad.py              # Silero VAD
│   │
│   ├── transcribe_sherpa/      # Sherpa backend
│   │   ├── __main__.py
│   │   ├── asr.py              # sherpa-onnx transcription
│   │   ├── capture.py
│   │   ├── config.py
│   │   ├── embeddings.py       # 3D Speaker (optional)
│   │   ├── output.py
│   │   ├── queue.py
│   │   └── vad.py              # Silero VAD (ONNX)
│   │
│   ├── transcribe_shared/      # Shared utilities
│   │   ├── config.py           # YAML loading, env interpolation
│   │   ├── speaker_registry.py # Cross-batch speaker tracking
│   │   └── transcript_filter.py # Garbage/hallucination filtering
│   │
│   ├── transcribe_logger/      # Transcript logger
│   │   ├── __main__.py
│   │   └── logger.py           # Rolling file writer
│   │
│   └── transcribe_widget/      # Menu bar widget
│       ├── __main__.py
│       └── app.py              # rumps-based menu bar app
│
├── .transcripts/               # Log files (gitignored)
├── data/                       # Runtime data (gitignored)
│   └── queue-*/                # Disk queues for whisper/sherpa
├── models/                     # Downloaded models (gitignored)
└── docs/                       # Legacy documentation
```

---

## Troubleshooting

### No audio being captured

```bash
# List available devices
./transcribe-backend --list-devices

# Check if audio is flowing (M4 backend)
./transcribe-backend --debug
```

### Speaker IDs keep changing

The speaker similarity threshold may be too high. Lower it in your config:

```yaml
speaker_similarity_threshold: 0.6  # Default is 0.75
```

### "No speaker embeddings returned"

For M4 backend, ensure you have a valid HuggingFace token in `.env`:

```bash
HUGGINGFACE_TOKEN=hf_your_token_here
```

### Whisper hallucinations

Add common hallucinated words to the ignore list:

```yaml
ignore_words: ["acl", "ack", "um", "uh"]
```

---

## License

MIT License - See LICENSE file for details.
