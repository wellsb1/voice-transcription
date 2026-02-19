import CoreAudio
import AVFoundation
import os

/// Captures system audio via Core Audio process tap and/or microphone via AVAudioEngine.
/// Delivers 16kHz mono Float32 audio chunks to a callback.
@available(macOS 14.2, *)
class AudioCaptureManager {
    typealias AudioCallback = ([Float], Date) -> Void

    private let onAudio: AudioCallback
    private let outputSampleRate: Double
    private let logger = Logger(subsystem: "com.transcribeme.app", category: "audio")

    // System tap state
    private var tapID: AudioObjectID = AudioObjectID(kAudioObjectUnknown)
    private var aggregateDeviceID: AudioObjectID = AudioObjectID(kAudioObjectUnknown)
    private var ioProcID: AudioDeviceIOProcID?
    private var resampler: Resampler?

    // Mic capture
    private var micEngine: AVAudioEngine?
    private let micLock = NSLock()
    private var micBuffer: [Float32] = []

    // Device-only capture (no system tap)
    private var deviceEngine: AVAudioEngine?

    private(set) var isRunning = false

    init(sampleRate: Double = 16000, onAudio: @escaping AudioCallback) {
        self.outputSampleRate = sampleRate
        self.onAudio = onAudio
    }

    // MARK: - Public API

    func startSystemTap(excludeBundleIDs: [String], micDevice: String?, noMic: Bool = false) throws {
        guard !isRunning else { return }

        let resolver = ProcessResolver()
        let excludeIDs = resolver.resolve(bundleIDs: excludeBundleIDs)

        var micUID: String? = nil
        if !noMic {
            if let micName = micDevice {
                micUID = resolver.findDeviceUID(matching: micName) ?? resolver.getDefaultInputDeviceUID()
            } else {
                micUID = resolver.getDefaultInputDeviceUID()
            }
        }

        // Create process tap
        let tapDesc = CATapDescription(monoGlobalTapButExcludeProcesses: excludeIDs)
        tapDesc.uuid = UUID()
        tapDesc.name = "transcribe-app-tap"

        var newTapID: AudioObjectID = AudioObjectID(kAudioObjectUnknown)
        let tapStatus = AudioHardwareCreateProcessTap(tapDesc, &newTapID)
        guard tapStatus == noErr else { throw TapError.createFailed(tapStatus) }
        self.tapID = newTapID

        let tapUID = try getTapUID(tapID)
        let tapFmt = try getTapFormat(tapID)
        let nativeSR = tapFmt.mSampleRate
        logger.info("Tap format: \(Int(nativeSR)) Hz, \(tapFmt.mChannelsPerFrame) ch")

        // Create aggregate device
        self.aggregateDeviceID = try createAggregateDevice(tapUID: tapUID)
        try waitForDeviceAlive(aggregateDeviceID)

        // Resampler if needed
        if nativeSR != outputSampleRate {
            self.resampler = try Resampler(inputSampleRate: nativeSR, outputSampleRate: outputSampleRate)
        }

        // Start mic if requested
        if let uid = micUID {
            try startMicCapture(deviceUID: uid, tapSampleRate: nativeSR)
        }

        // Install IO proc
        var newIOProcID: AudioDeviceIOProcID?
        let ioProcStatus = AudioDeviceCreateIOProcIDWithBlock(
            &newIOProcID, aggregateDeviceID, nil
        ) { [weak self] _, inInputData, _, _, _ in
            self?.handleTapAudio(inInputData)
        }
        guard ioProcStatus == noErr else { throw TapError.ioProcFailed(ioProcStatus) }
        self.ioProcID = newIOProcID

        let startStatus = AudioDeviceStart(aggregateDeviceID, ioProcID)
        guard startStatus == noErr else { throw TapError.startFailed(startStatus) }

        isRunning = true
        logger.info("System tap capture started")
    }

    func startDeviceOnly(deviceName: String? = nil) throws {
        guard !isRunning else { return }

        let engine = AVAudioEngine()
        let inputNode = engine.inputNode

        // Set specific device if requested
        if let name = deviceName {
            let resolver = ProcessResolver()
            if let uid = resolver.findDeviceUID(matching: name) {
                setAudioUnitDevice(inputNode.audioUnit!, uid: uid)
            }
        }

        let recordFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: outputSampleRate,
            channels: 1,
            interleaved: true
        )!

        inputNode.installTap(onBus: 0, bufferSize: 1600, format: recordFormat) { [weak self] buffer, _ in
            guard let self = self else { return }
            let frameCount = Int(buffer.frameLength)
            guard frameCount > 0, let data = buffer.floatChannelData else { return }
            let samples = Array(UnsafeBufferPointer(start: data[0], count: frameCount))
            self.onAudio(samples, Date())
        }

        try engine.start()
        self.deviceEngine = engine
        isRunning = true
        logger.info("Device-only capture started")
    }

    func stop() {
        // Stop mic engine
        micEngine?.stop()
        micEngine = nil

        // Stop device engine
        if let engine = deviceEngine {
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
            deviceEngine = nil
        }

        // Stop system tap
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

        resampler = nil
        micLock.lock()
        micBuffer.removeAll()
        micLock.unlock()

        isRunning = false
        logger.info("Audio capture stopped")
    }

    // MARK: - Mic capture

    private func startMicCapture(deviceUID: String, tapSampleRate: Double) throws {
        let engine = AVAudioEngine()
        let inputNode = engine.inputNode

        setAudioUnitDevice(inputNode.audioUnit!, uid: deviceUID)

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
        logger.info("Mic capture started (device=\(deviceUID))")
    }

    private func setAudioUnitDevice(_ audioUnit: AudioUnit, uid: String) {
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
            var devUID: CFString = "" as CFString
            var uidSize = UInt32(MemoryLayout<CFString>.size)
            if AudioObjectGetPropertyData(dev, &uidAddr, 0, nil, &uidSize, &devUID) == noErr {
                if (devUID as String) == uid {
                    var devID = dev
                    AudioUnitSetProperty(
                        audioUnit,
                        kAudioOutputUnitProperty_CurrentDevice,
                        kAudioUnitScope_Global,
                        0,
                        &devID,
                        UInt32(MemoryLayout<AudioDeviceID>.size)
                    )
                    break
                }
            }
        }
    }

    // MARK: - Audio callback

    private func handleTapAudio(_ inputData: UnsafePointer<AudioBufferList>) {
        let abl = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: inputData))

        guard let firstBuffer = abl.first,
              let data = firstBuffer.mData,
              firstBuffer.mDataByteSize > 0 else { return }

        let frameCount = Int(firstBuffer.mDataByteSize) / MemoryLayout<Float32>.size
        guard frameCount > 0 else { return }

        let tapPtr = data.assumingMemoryBound(to: Float32.self)
        var outputSamples: [Float]

        if micEngine != nil {
            micLock.lock()
            let micCount = min(micBuffer.count, frameCount)
            if micCount > 0 {
                var mixed = [Float32](repeating: 0, count: frameCount)
                for i in 0..<frameCount {
                    let tapSample = tapPtr[i]
                    let micSample = i < micCount ? micBuffer[i] : 0
                    mixed[i] = tapSample * 0.5 + micSample * 0.5
                }
                micBuffer.removeFirst(micCount)
                micLock.unlock()
                outputSamples = resampleIfNeeded(mixed)
            } else {
                micLock.unlock()
                let raw = Array(UnsafeBufferPointer(start: tapPtr, count: frameCount))
                outputSamples = resampleIfNeeded(raw)
            }
        } else {
            let raw = Array(UnsafeBufferPointer(start: tapPtr, count: frameCount))
            outputSamples = resampleIfNeeded(raw)
        }

        guard !outputSamples.isEmpty else { return }
        onAudio(outputSamples, Date())
    }

    private func resampleIfNeeded(_ samples: [Float]) -> [Float] {
        guard let resampler = resampler else { return samples }
        return samples.withUnsafeBufferPointer { ptr in
            resampler.convert(ptr)
        }
    }

    // MARK: - Core Audio helpers

    private func getTapFormat(_ tapID: AudioObjectID) throws -> AudioStreamBasicDescription {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioTapPropertyFormat,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var format = AudioStreamBasicDescription()
        var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        let status = AudioObjectGetPropertyData(tapID, &address, 0, nil, &size, &format)
        guard status == noErr else { throw TapError.formatFailed(status) }
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
        guard status == noErr else { throw TapError.formatFailed(status) }
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
        guard uidStatus == noErr else { throw TapError.aggregateFailed(uidStatus) }
        return uid as String
    }

    private func createAggregateDevice(tapUID: String) throws -> AudioObjectID {
        let outputUID = try getDefaultOutputDeviceUID()

        let description: [String: Any] = [
            kAudioAggregateDeviceUIDKey as String: "com.transcribeme.aggregate.\(UUID().uuidString)",
            kAudioAggregateDeviceNameKey as String: "transcribe-aggregate",
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
        guard status == noErr else { throw TapError.aggregateFailed(status) }
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
            if status == noErr && alive == 1 { return }
            Thread.sleep(forTimeInterval: 0.1)
        }
        logger.warning("Aggregate device may not be fully alive")
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
