import Foundation
import CoreAudio

// -- Argument parsing --

struct Options {
    var excludeBundleIDs: [String] = []
    var sampleRate: Int = 16000
    var micDevice: String? = nil  // nil = system default
    var noMic: Bool = false
}

func parseArgs() -> Options {
    var opts = Options()
    let args = Array(CommandLine.arguments.dropFirst())
    var i = 0

    while i < args.count {
        let arg = args[i]
        switch arg {
        case "--exclude":
            i += 1
            if i < args.count {
                opts.excludeBundleIDs = args[i].split(separator: ",").map(String.init)
            }
        case "--sample-rate":
            i += 1
            if i < args.count, let rate = Int(args[i]) {
                opts.sampleRate = rate
            }
        case "--mic":
            i += 1
            if i < args.count {
                opts.micDevice = args[i]
            }
        case "--no-mic":
            opts.noMic = true
        case "--help", "-h":
            fputs("""
            Usage: audio-tap [OPTIONS]

            Captures system audio (+ microphone) and outputs raw PCM float32 to stdout.

            Options:
              --exclude IDS    Comma-separated bundle IDs to exclude
              --sample-rate N  Output sample rate (default: 16000)
              --mic DEVICE     Microphone device name (default: system default)
              --no-mic         Capture system audio only, no microphone
              --help           Show this help

            Examples:
              audio-tap --exclude com.apple.Music,com.spotify.client
              audio-tap --exclude com.apple.Music --mic "MacBook Pro Microphone"
              audio-tap --no-mic --exclude com.apple.Music

            """, stderr)
            exit(0)
        default:
            fputs("audio-tap: Unknown argument: \(arg)\n", stderr)
            exit(1)
        }
        i += 1
    }
    return opts
}

// -- Main --

// Unbuffered stdout for real-time streaming
setvbuf(stdout, nil, _IONBF, 0)

let options = parseArgs()

fputs("audio-tap: Starting system audio capture\n", stderr)
fputs("audio-tap: Output sample rate: \(options.sampleRate) Hz\n", stderr)
if !options.excludeBundleIDs.isEmpty {
    fputs("audio-tap: Excluding: \(options.excludeBundleIDs.joined(separator: ", "))\n", stderr)
}

let resolver = ProcessResolver()

// Resolve excluded processes
let excludedObjectIDs = resolver.resolve(bundleIDs: options.excludeBundleIDs)

// Resolve microphone device
var micUID: String? = nil
if !options.noMic {
    if let micName = options.micDevice {
        micUID = resolver.findDeviceUID(matching: micName)
        if let uid = micUID {
            fputs("audio-tap: Mic device: \(micName) (uid=\(uid))\n", stderr)
        } else {
            fputs("audio-tap: Warning: mic device '\(micName)' not found, using default\n", stderr)
            micUID = resolver.getDefaultInputDeviceUID()
        }
    } else {
        micUID = resolver.getDefaultInputDeviceUID()
    }
    if let uid = micUID {
        fputs("audio-tap: Using mic: \(uid)\n", stderr)
    } else {
        fputs("audio-tap: Warning: no mic device found\n", stderr)
    }
} else {
    fputs("audio-tap: Mic disabled (--no-mic)\n", stderr)
}

if #available(macOS 14.2, *) {
    // Create and start the tap
    let tap = SystemTap(
        excludeProcesses: excludedObjectIDs,
        micDeviceUID: micUID,
        outputSampleRate: Double(options.sampleRate)
    )

    do {
        try tap.start()
    } catch {
        fputs("audio-tap: Failed to start: \(error)\n", stderr)
        exit(1)
    }

    // Signal handling for clean shutdown
    let sigintSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
    let sigtermSource = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
    signal(SIGINT, SIG_IGN)
    signal(SIGTERM, SIG_IGN)

    func shutdown() {
        fputs("\naudio-tap: Shutting down...\n", stderr)
        tap.stop()
        exit(0)
    }

    sigintSource.setEventHandler { shutdown() }
    sigtermSource.setEventHandler { shutdown() }
    sigintSource.resume()
    sigtermSource.resume()

    fputs("audio-tap: Capturing... (Ctrl-C to stop)\n", stderr)

    // Run loop keeps the process alive for Core Audio callbacks
    RunLoop.current.run()
} else {
    fputs("audio-tap: Requires macOS 14.2 or later\n", stderr)
    exit(1)
}
