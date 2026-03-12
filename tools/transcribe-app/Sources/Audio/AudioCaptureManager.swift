import CoreAudio
import AVFoundation
import os

/// Captures system audio via an in-process Core Audio process tap and/or
/// microphone via AVAudioEngine. Delivers 16kHz mono Float32 audio chunks to a callback.
@available(macOS 14.2, *)
class AudioCaptureManager {
    typealias AudioCallback = ([Float], Date) -> Void

    private let onAudio: AudioCallback
    private let outputSampleRate: Double
    private let logger = Logger(subsystem: "com.transcribeme.app", category: "audio")

    private var systemTap: SystemTap?
    private var deviceEngine: AVAudioEngine?

    private(set) var isRunning = false

    init(sampleRate: Double = 16000, onAudio: @escaping AudioCallback) {
        self.outputSampleRate = sampleRate
        self.onAudio = onAudio
    }

    // MARK: - Public API

    func startSystemTap(excludeBundleIDs: [String], micDevice: String?, noMic: Bool = false) throws {
        guard !isRunning else { return }

        // Resolve bundle IDs to Core Audio process IDs
        let resolver = ProcessResolver()
        let excludeProcesses = resolver.resolve(bundleIDs: excludeBundleIDs)

        // Determine mic device UID
        var micUID: String? = nil
        if !noMic {
            if let mic = micDevice {
                micUID = resolver.findDeviceUID(matching: mic)
            } else {
                micUID = resolver.getDefaultInputDeviceUID()
            }
        }

        let callback = onAudio
        let tap = SystemTap(
            excludeProcesses: excludeProcesses,
            micDeviceUID: micUID,
            outputSampleRate: outputSampleRate
        ) { samples in
            let array = Array(samples)
            callback(array, Date())
        }

        try tap.start()
        self.systemTap = tap
        isRunning = true
        logger.info("System tap capture started (in-process)")
    }

    func startDeviceOnly(deviceName: String? = nil) throws {
        guard !isRunning else { return }

        let engine = AVAudioEngine()
        let inputNode = engine.inputNode

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
        // Stop in-process system tap
        systemTap?.stop()
        systemTap = nil

        // Stop device engine
        if let engine = deviceEngine {
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
            deviceEngine = nil
        }

        isRunning = false
        logger.info("Audio capture stopped")
    }

    // MARK: - Helpers

    private func setAudioUnitDevice(_ audioUnit: AudioUnit, uid: String) {
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
}
