import CoreAudio
import AVFoundation

/// Creates a Core Audio process tap (tap-only aggregate device) for system audio,
/// optionally captures mic via AVAudioEngine, mixes both, resamples, and writes
/// raw PCM float32 to stdout.
@available(macOS 14.2, *)
class SystemTap {
    private let excludeProcesses: [AudioObjectID]
    private let micDeviceUID: String?
    private let outputSampleRate: Double

    private var tapID: AudioObjectID = AudioObjectID(kAudioObjectUnknown)
    private var aggregateDeviceID: AudioObjectID = AudioObjectID(kAudioObjectUnknown)
    private var ioProcID: AudioDeviceIOProcID?
    private var resampler: Resampler?

    // Mic capture via AVAudioEngine
    private var micEngine: AVAudioEngine?
    private let micLock = NSLock()
    private var micBuffer: [Float32] = []

    init(excludeProcesses: [AudioObjectID], micDeviceUID: String?, outputSampleRate: Double) {
        self.excludeProcesses = excludeProcesses
        self.micDeviceUID = micDeviceUID
        self.outputSampleRate = outputSampleRate
    }

    func start() throws {
        // 1. Create tap description
        let tapDesc = CATapDescription(
            monoGlobalTapButExcludeProcesses: excludeProcesses
        )
        tapDesc.uuid = UUID()
        tapDesc.name = "audio-tap-transcription"

        // 2. Create the process tap
        var newTapID: AudioObjectID = AudioObjectID(kAudioObjectUnknown)
        let tapStatus = AudioHardwareCreateProcessTap(tapDesc, &newTapID)
        guard tapStatus == noErr else {
            throw TapError.createFailed(tapStatus)
        }
        self.tapID = newTapID
        fputs("audio-tap: Created process tap (id=\(tapID))\n", stderr)

        // 3. Read the tap's actual UID (may differ from tapDesc.uuid)
        let tapUID = try getTapUID(tapID)
        fputs("audio-tap: Tap UID: \(tapUID)\n", stderr)

        // 4. Read the tap's native format
        let tapFmt = try getTapFormat(tapID)
        let nativeSR = tapFmt.mSampleRate
        fputs("audio-tap: Tap native format: \(Int(nativeSR)) Hz, \(tapFmt.mChannelsPerFrame) ch\n", stderr)

        // 5. Create aggregate device with system output sub-device + tap
        self.aggregateDeviceID = try createAggregateDevice(tapUID: tapUID)
        fputs("audio-tap: Created aggregate device (id=\(aggregateDeviceID))\n", stderr)

        try waitForDeviceAlive(aggregateDeviceID)

        // 6. Create resampler if needed
        if nativeSR != outputSampleRate {
            self.resampler = try Resampler(
                inputSampleRate: nativeSR,
                outputSampleRate: outputSampleRate,
                channels: 1
            )
            fputs("audio-tap: Resampling \(Int(nativeSR)) -> \(Int(outputSampleRate)) Hz\n", stderr)
        }

        // 7. Start mic capture if requested
        if let micUID = micDeviceUID {
            try startMicCapture(deviceUID: micUID, tapSampleRate: nativeSR)
        }

        // 8. Install IO proc on aggregate device
        var newIOProcID: AudioDeviceIOProcID?
        let ioProcStatus = AudioDeviceCreateIOProcIDWithBlock(
            &newIOProcID,
            aggregateDeviceID,
            nil
        ) { [weak self] _, inInputData, _, _, _ in
            self?.handleAudio(inInputData)
        }
        guard ioProcStatus == noErr else {
            throw TapError.ioProcFailed(ioProcStatus)
        }
        self.ioProcID = newIOProcID

        // 9. Start IO
        let startStatus = AudioDeviceStart(aggregateDeviceID, ioProcID)
        guard startStatus == noErr else {
            throw TapError.startFailed(startStatus)
        }

        fputs("audio-tap: Audio capture started\n", stderr)
    }

    func stop() {
        // Stop mic engine
        micEngine?.stop()
        micEngine = nil

        if let ioProcID = ioProcID {
            AudioDeviceStop(aggregateDeviceID, ioProcID)
            AudioDeviceDestroyIOProcID(aggregateDeviceID, ioProcID)
            self.ioProcID = nil
        }

        if aggregateDeviceID != AudioObjectID(kAudioObjectUnknown) {
            AudioHardwareDestroyAggregateDevice(aggregateDeviceID)
            aggregateDeviceID = AudioObjectID(kAudioObjectUnknown)
        }

        if tapID != AudioObjectID(kAudioObjectUnknown) {
            AudioHardwareDestroyProcessTap(tapID)
            tapID = AudioObjectID(kAudioObjectUnknown)
        }

        fputs("audio-tap: Cleanup complete\n", stderr)
    }

    // MARK: - Mic capture via AVAudioEngine

    private func startMicCapture(deviceUID: String, tapSampleRate: Double) throws {
        let engine = AVAudioEngine()

        // Set the input device by UID
        let inputNode = engine.inputNode
        var deviceID: AudioDeviceID = kAudioObjectUnknown

        // Find device ID from UID
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var propSize: UInt32 = 0
        AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &propSize)
        let deviceCount = Int(propSize) / MemoryLayout<AudioObjectID>.size
        var devices = [AudioObjectID](repeating: 0, count: deviceCount)
        AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &propSize, &devices)

        for dev in devices {
            var uidAddr = AudioObjectPropertyAddress(
                mSelector: kAudioDevicePropertyDeviceUID,
                mScope: kAudioObjectPropertyScopeGlobal,
                mElement: kAudioObjectPropertyElementMain
            )
            var uid: CFString = "" as CFString
            var uidSize = UInt32(MemoryLayout<CFString>.size)
            if AudioObjectGetPropertyData(dev, &uidAddr, 0, nil, &uidSize, &uid) == noErr {
                if (uid as String) == deviceUID {
                    deviceID = dev
                    break
                }
            }
        }

        guard deviceID != kAudioObjectUnknown else {
            fputs("audio-tap: Warning: could not find mic device \(deviceUID) for AVAudioEngine\n", stderr)
            return
        }

        // Set the input device on the audio unit
        let audioUnit = inputNode.audioUnit!
        var devID = deviceID
        let setStatus = AudioUnitSetProperty(
            audioUnit,
            kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global,
            0,
            &devID,
            UInt32(MemoryLayout<AudioDeviceID>.size)
        )
        if setStatus != noErr {
            fputs("audio-tap: Warning: failed to set mic device (OSStatus \(setStatus))\n", stderr)
        }

        // Install tap on input node — convert to mono float32 at the tap's sample rate
        let micFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: tapSampleRate,
            channels: 1,
            interleaved: true
        )!

        inputNode.installTap(onBus: 0, bufferSize: 512, format: micFormat) { [weak self] buffer, _ in
            guard let self = self else { return }
            let frameCount = Int(buffer.frameLength)
            guard frameCount > 0, let channelData = buffer.floatChannelData else { return }

            self.micLock.lock()
            self.micBuffer.append(contentsOf: UnsafeBufferPointer(start: channelData[0], count: frameCount))
            self.micLock.unlock()
        }

        try engine.start()
        self.micEngine = engine
        fputs("audio-tap: Mic capture started via AVAudioEngine (device=\(deviceUID))\n", stderr)
    }

    // MARK: - Audio callback

    private func handleAudio(_ inputData: UnsafePointer<AudioBufferList>) {
        let abl = UnsafeMutableAudioBufferListPointer(
            UnsafeMutablePointer(mutating: inputData)
        )

        // Get the first buffer (tap audio)
        guard let firstBuffer = abl.first,
              let data = firstBuffer.mData,
              firstBuffer.mDataByteSize > 0 else { return }

        let frameCount = Int(firstBuffer.mDataByteSize) / MemoryLayout<Float32>.size
        guard frameCount > 0 else { return }

        let tapPtr = data.assumingMemoryBound(to: Float32.self)

        // If we have mic audio, mix it in
        if micEngine != nil {
            micLock.lock()
            let micCount = min(micBuffer.count, frameCount)
            if micCount > 0 {
                // Mix: average tap and mic for overlapping frames
                var mixed = [Float32](repeating: 0, count: frameCount)
                for i in 0..<frameCount {
                    let tapSample = tapPtr[i]
                    let micSample = i < micCount ? micBuffer[i] : 0
                    mixed[i] = tapSample * 0.5 + micSample * 0.5
                }
                // Remove consumed mic samples
                micBuffer.removeFirst(micCount)
                micLock.unlock()

                mixed.withUnsafeBufferPointer { ptr in
                    writeOutput(ptr)
                }
                return
            }
            micLock.unlock()
        }

        // No mic or no mic data available — just output tap audio
        let samples = UnsafeBufferPointer<Float32>(start: tapPtr, count: frameCount)
        writeOutput(samples)
    }

    private func writeOutput(_ samples: UnsafeBufferPointer<Float32>) {
        if let resampler = resampler {
            let resampled = resampler.convert(samples)
            guard !resampled.isEmpty else { return }
            _ = resampled.withUnsafeBufferPointer { ptr in
                fwrite(ptr.baseAddress!, MemoryLayout<Float32>.size, ptr.count, stdout)
            }
        } else {
            fwrite(samples.baseAddress!, MemoryLayout<Float32>.size, samples.count, stdout)
        }
    }

    // MARK: - Private helpers

    private func getTapFormat(_ tapID: AudioObjectID) throws -> AudioStreamBasicDescription {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioTapPropertyFormat,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        var format = AudioStreamBasicDescription()
        var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        let status = AudioObjectGetPropertyData(tapID, &address, 0, nil, &size, &format)
        guard status == noErr else {
            throw TapError.formatFailed(status)
        }
        return format
    }

    private func getTapUID(_ tapID: AudioObjectID) throws -> String {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioTapPropertyUID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var uid: CFString = "" as CFString
        var size = UInt32(MemoryLayout<CFString>.stride)
        let status = withUnsafeMutablePointer(to: &uid) { ptr in
            AudioObjectGetPropertyData(tapID, &address, 0, nil, &size, ptr)
        }
        guard status == noErr else {
            throw TapError.formatFailed(status)
        }
        return uid as String
    }

    private func getDefaultOutputDeviceUID() throws -> String {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultOutputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var outputID: AudioObjectID = kAudioObjectUnknown
        var size = UInt32(MemoryLayout<AudioObjectID>.size)
        let status = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &outputID
        )
        guard status == noErr, outputID != kAudioObjectUnknown else {
            throw TapError.aggregateFailed(status)
        }

        var uidAddr = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceUID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var uid: CFString = "" as CFString
        var uidSize = UInt32(MemoryLayout<CFString>.stride)
        let uidStatus = withUnsafeMutablePointer(to: &uid) { ptr in
            AudioObjectGetPropertyData(outputID, &uidAddr, 0, nil, &uidSize, ptr)
        }
        guard uidStatus == noErr else {
            throw TapError.aggregateFailed(uidStatus)
        }
        return uid as String
    }

    private func createAggregateDevice(tapUID: String) throws -> AudioObjectID {
        let outputUID = try getDefaultOutputDeviceUID()
        fputs("audio-tap: Using output device as sub-device: \(outputUID)\n", stderr)

        let description: [String: Any] = [
            kAudioAggregateDeviceUIDKey as String: "com.transcribe.audio-tap.\(UUID().uuidString)",
            kAudioAggregateDeviceNameKey as String: "audio-tap-aggregate",
            kAudioAggregateDeviceMainSubDeviceKey as String: outputUID,
            kAudioAggregateDeviceIsPrivateKey as String: true,
            kAudioAggregateDeviceIsStackedKey as String: false,
            kAudioAggregateDeviceSubDeviceListKey as String: [
                [kAudioSubDeviceUIDKey as String: outputUID]
            ],
            kAudioAggregateDeviceTapListKey as String: [
                [
                    kAudioSubTapUIDKey as String: tapUID,
                    kAudioSubTapDriftCompensationKey as String: true,
                ]
            ],
        ]

        var deviceID: AudioObjectID = AudioObjectID(kAudioObjectUnknown)
        let status = AudioHardwareCreateAggregateDevice(description as CFDictionary, &deviceID)
        guard status == noErr else {
            throw TapError.aggregateFailed(status)
        }
        return deviceID
    }

    private func waitForDeviceAlive(_ deviceID: AudioObjectID) throws {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceIsAlive,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        for _ in 0..<20 {
            var alive: UInt32 = 0
            var size = UInt32(MemoryLayout<UInt32>.size)
            let status = AudioObjectGetPropertyData(deviceID, &address, 0, nil, &size, &alive)
            if status == noErr && alive == 1 {
                return
            }
            Thread.sleep(forTimeInterval: 0.1)
        }
        fputs("audio-tap: Warning: device may not be fully alive\n", stderr)
    }
}

enum TapError: Error, CustomStringConvertible {
    case createFailed(OSStatus)
    case formatFailed(OSStatus)
    case aggregateFailed(OSStatus)
    case ioProcFailed(OSStatus)
    case startFailed(OSStatus)

    var description: String {
        switch self {
        case .createFailed(let s): return "Failed to create process tap (OSStatus \(s))"
        case .formatFailed(let s): return "Failed to read tap format (OSStatus \(s))"
        case .aggregateFailed(let s): return "Failed to create aggregate device (OSStatus \(s))"
        case .ioProcFailed(let s): return "Failed to create IO proc (OSStatus \(s))"
        case .startFailed(let s): return "Failed to start audio device (OSStatus \(s))"
        }
    }
}
