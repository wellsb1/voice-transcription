# Voice-Activated Transcription Device

> **DEPRECATED**: This document is outdated. See [README.md](../README.md) for current documentation.

---

## Project Overview

Build an always-on, open-source voice transcription system that:
- Runs on Raspberry Pi 5 (8GB)
- Continuously captures audio from a USB conference microphone (Jabra Speak2 75)
- Transcribes speech locally using Whisper
- Identifies speakers (diarization) using speaker embeddings
- Outputs JSONL to stdout for Unix-style piping
- **Does NOT store audio recordings** — only text is persisted

## Target Hardware

- **Compute**: Raspberry Pi 5 8GB
- **Microphone**: Jabra Speak2 75 USB conference speakerphone (4 beamforming mics, built-in noise cancellation, voice level normalization)
- **Storage**: Samsung PRO Endurance 128GB microSD (high endurance for continuous writes)

## Development Environment

- **Development**: macOS (any recent version)
- **Deployment**: Raspberry Pi OS 64-bit (Bookworm)
- **Language**: Python 3.11+
- The entire stack is cross-platform — develop on Mac, deploy to Pi with same code

---

## Architecture

The system uses a Unix pipeline approach with two separate programs:

1. **Transcriber** (`python -m transcribe`) - Captures audio, transcribes, outputs JSONL to stdout
2. **Uploader** (future) - Reads stdin or watches files, queues, uploads with retry

```
┌─────────────────────────────────────────────────────────────┐
│                      TRANSCRIBER PROCESS                     │
│                                                              │
│  ┌──────────────┐  Disk   ┌──────────────────────────────┐  │
│  │ Capture      │  Queue  │ Processing Thread            │  │
│  │ Thread       │────────▶│                              │  │
│  │              │ (.npy)  │ Whisper → Embedding →        │  │
│  │ VAD-aware    │         │ Speaker Match → JSONL stdout │  │
│  │ chunking     │         │                              │  │
│  └──────────────┘         └──────────────────────────────┘  │
│                                                              │
│  Capture: VAD-aware chunking (chunks on silence boundaries)  │
│  Queue: Disk-backed, unbounded, crash-resilient              │
└──────────────────────────────────────────────────────────────┘
          │
          │ stdout (JSONL)
          ▼
┌─────────────────────────────────────────────────────────────┐
│  Usage examples:                                             │
│                                                              │
│  python -m transcribe                    # Just transcribe   │
│  python -m transcribe | tee out.jsonl    # Save + display    │
│  python -m transcribe | ./uploader       # Pipe to uploader  │
└──────────────────────────────────────────────────────────────┘
```

### VAD-Aware Chunking

Instead of fixed 10-second chunks (which cut words in half), the capture thread
uses Voice Activity Detection to chunk on natural silence boundaries:

1. Capture audio in 100ms windows
2. Run VAD on each window to detect speech/silence
3. When 500ms+ of silence detected → emit chunk
4. If someone talks for 2 minutes straight → buffer grows, emits when they pause

**Benefits:**
- Words never get cut in half at chunk boundaries
- Natural sentence/thought boundaries
- Speaker changes often happen at pauses
- Variable chunk sizes (1s to 30s+) based on speech patterns

**Configurable parameters:**
- `silence_duration`: Seconds of silence to trigger chunk (default: 0.5s)
- `min_chunk_duration`: Minimum chunk size before checking silence (default: 1s)
- `max_chunk_duration`: Force chunk even without silence (default: 30s)

### Threading Model

Audio capture and transcription run in separate threads:

- **Capture Thread**: Records 100ms windows, runs VAD, chunks on silence, writes to disk queue
- **Processing Thread**: Reads from queue, runs Whisper → Speaker ID → stdout

**Why disk-backed queue?**
- Transcription may take 2-3s for 10s of audio
- Capture must never block — Pi audio drivers buffer ~1-2s max
- Disk queue = unbounded backlog without memory exhaustion
- Files persist across crashes — can resume processing on restart

---

## JSONL Output Format

One JSON object per line, self-contained:

```json
{"ts": "2024-01-15T10:30:45.123", "device": "office", "speaker": 0, "confidence": 0.92, "text": "Hello, how are you?"}
{"ts": "2024-01-15T10:30:48.456", "device": "office", "speaker": 1, "confidence": 0.87, "text": "I'm doing well, thanks."}
```

Fields:
- `ts` - ISO 8601 timestamp with milliseconds
- `device` - Device name from config (e.g., "office", "living_room")
- `speaker` - Integer speaker ID (0-indexed, consistent across session)
- `confidence` - Speaker match confidence (0.0-1.0)
- `text` - Transcribed text

---

## Configuration

### Config File: `config.yaml`

```yaml
# Device identification - appears in JSONL output
device_name: "office"

# Audio settings
audio_device: null  # null = system default, or device name/index
sample_rate: 16000

# VAD-aware chunking settings
vad_threshold: 0.5           # 0.0-1.0, higher = less sensitive to quiet speech
silence_duration: 0.5        # Seconds of silence to trigger chunk boundary
min_chunk_duration: 1.0      # Minimum chunk size before checking for silence
max_chunk_duration: 30.0     # Force chunk even if no silence (handles monologues)

# Model settings
whisper_model: "tiny.en"  # Options: tiny.en, base.en, small.en

# Speaker identification
speaker_similarity_threshold: 0.75  # 0.0-1.0, higher = stricter matching
speaker_registry_path: "./data/speakers.json"

# Queue settings (disk-backed audio buffer)
queue_dir: "./data/queue"
```

### Override Hierarchy

1. **CLI args** (highest priority): `--device-name=kitchen`
2. **Environment variables**: `TRANSCRIBE_DEVICE_NAME=kitchen`
3. **Config file** (lowest priority): `device_name: office`

---

## Project Structure

```
scriptaculus/
├── config.yaml              # User config (gitignored)
├── config.example.yaml      # Template config (committed)
├── pyproject.toml           # Package metadata and dependencies
├── src/
│   └── transcribe/
│       ├── __init__.py
│       ├── __main__.py      # Entry point: python -m transcribe
│       ├── cli.py           # Argument parsing, config loading
│       ├── config.py        # Config schema and loading
│       ├── queue.py         # Disk-backed audio queue
│       ├── capture.py       # Audio capture (sounddevice)
│       ├── vad.py           # Voice activity detection (silero)
│       ├── transcribe.py    # Whisper transcription
│       ├── embeddings.py    # SpeechBrain speaker embeddings
│       ├── diarize.py       # Speaker registry and matching
│       └── output.py        # JSONL formatting and stdout output
├── data/                    # Runtime data (gitignored)
│   ├── speakers.json        # Persisted speaker registry
│   └── queue/               # Disk-backed audio queue (.npy files)
└── models/                  # Downloaded ML models (gitignored)
```

---

## Core Components

### 1. Audio Capture (`capture.py`)

**Library**: `sounddevice`

- Captures 16kHz mono audio from USB microphone
- Runs in dedicated thread to prevent blocking
- Writes 10-second chunks to disk queue
- Handles device selection via config

### 2. Disk Queue (`queue.py`)

- Stores audio chunks as timestamped `.npy` files
- Unbounded queue that survives crashes
- Files processed in chronological order
- Deleted after successful processing

### 3. Voice Activity Detection (`vad.py`)

**Library**: `silero-vad`

- Filters out silence to avoid wasting compute
- ~95% of chunks are silence in typical environments
- Reduces unnecessary transcription

### 4. Transcription (`transcribe.py`)

**Library**: `faster-whisper`

- Uses CTranslate2 backend — faster than original Whisper
- Use `tiny.en` model on Pi (good balance of speed/accuracy)
- Can use `small.en` or `base.en` on Mac for development

### 5. Speaker Embedding (`embeddings.py`)

**Library**: `speechbrain`

- ECAPA-TDNN model pretrained on VoxCeleb
- Extracts 192-dimensional speaker embedding vector
- This is the "voice fingerprint" for each speaker

### 6. Speaker Registry (`diarize.py`)

- Maintains consistent speaker IDs across chunks
- Uses cosine similarity to match new embeddings
- Persists to JSON for crash recovery
- Sliding window of recent embeddings for accuracy

---

## CLI Usage

```bash
# Basic usage
python -m transcribe

# List available audio devices
python -m transcribe --list-devices

# With config overrides
python -m transcribe --device-name=kitchen --whisper-model=base.en

# Specify custom config file
python -m transcribe --config=/path/to/config.yaml

# Save output to file
python -m transcribe | tee transcripts.jsonl

# Future: Pipe to uploader
python -m transcribe | ./uploader
```

---

## Dependencies

Defined in `pyproject.toml`:

```
sounddevice>=0.4.6      # Audio capture
numpy>=1.24.0           # Array operations
torch>=2.0.0            # ML framework
torchaudio>=2.0.0       # Audio processing
faster-whisper>=0.10.0  # Speech-to-text
speechbrain>=0.5.15     # Speaker embeddings
pyyaml>=6.0             # Config file parsing
```

### System Dependencies (Pi)

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv libportaudio2 ffmpeg
```

### System Dependencies (macOS)

```bash
brew install portaudio ffmpeg
```

---

## Performance Expectations

### Raspberry Pi 5 (8GB)

| Component | Time per 10s chunk |
|-----------|-------------------|
| Audio capture | 10s (real-time) |
| VAD check | ~50ms |
| faster-whisper (tiny.en) | ~1-2s |
| ECAPA-TDNN embedding | ~200-500ms |
| Speaker matching | <10ms |
| **Total processing** | **~2-3s per 10s of audio** |

Processing runs faster than real-time, so the system keeps up with continuous audio.

### Memory Usage

| Component | RAM |
|-----------|-----|
| Base OS + Python | ~500MB |
| PyTorch | ~500MB |
| faster-whisper (tiny.en) | ~500MB |
| SpeechBrain ECAPA-TDNN | ~300MB |
| Audio buffers | ~50MB |
| **Total** | **~2GB** |

The 8GB Pi has plenty of headroom.

---

## Deployment

### Initial Setup on Pi

```bash
# Clone repo
git clone https://github.com/your-repo/scriptaculus.git
cd scriptaculus

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install package
pip install -e .

# Copy and edit config
cp config.example.yaml config.yaml
nano config.yaml

# Test run
python -m transcribe
```

### Run as Service (systemd)

Create `/etc/systemd/system/transcription.service`:

```ini
[Unit]
Description=Voice Transcription Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/scriptaculus
Environment=PATH=/home/pi/scriptaculus/venv/bin
ExecStart=/home/pi/scriptaculus/venv/bin/python -m transcribe
Restart=always
RestartSec=10
StandardOutput=append:/var/log/transcription.jsonl

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable transcription
sudo systemctl start transcription
sudo systemctl status transcription
```

---

## Future Enhancements

1. **Uploader service** — Separate process that queues JSONL to cloud with retry
2. **Noise suppression** — Add RNNoise if Jabra's built-in processing isn't sufficient
3. **Web dashboard** — Local web UI to view live transcripts and speaker stats
4. **Speaker naming** — Allow manual labeling of speaker IDs to names
5. **Keyword detection** — Trigger actions on specific phrases
6. **Multiple model sizes** — Auto-select model based on available resources

---

## Key Design Decisions

1. **VAD-aware chunking** — Chunks on silence boundaries instead of fixed intervals. Words never get cut in half. Variable chunk sizes (1-30s+) based on natural speech patterns.

2. **Unix pipeline architecture** — Transcriber outputs to stdout, separate uploader handles persistence/upload. Allows `tee`, `grep`, custom processing.

3. **Disk-backed queue** — Decouples capture from processing, survives crashes, never drops audio.

4. **Chunked processing, not true streaming** — Simpler, more reliable, and fast enough for real-time use. ~2-3s latency is acceptable for logging.

5. **Speaker registry with cosine similarity** — Maintains consistent speaker labels across chunks without complex clustering.

6. **No audio storage** — Privacy-friendly, only text leaves the device.

7. **faster-whisper over whisper.cpp** — Better Python integration, similar performance on Pi.

8. **SpeechBrain ECAPA-TDNN over pyannote** — Lightweight enough to run on Pi, just does embedding extraction.

9. **JSON speaker registry** — Human-readable, inspectable, avoids pickle security concerns.

10. **Config hierarchy** — YAML file for defaults, env vars for deployment, CLI for testing.

11. **Jabra hardware handles noise** — Offloads noise cancellation and voice normalization to dedicated hardware.
