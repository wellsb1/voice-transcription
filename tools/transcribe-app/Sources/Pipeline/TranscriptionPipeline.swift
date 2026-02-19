import Foundation
import FluidAudio
import os

/// Orchestrates the transcription pipeline: audio → VAD → ASR → (diarization) → output.
@MainActor
class TranscriptionPipeline: ObservableObject {
    private let logger = Logger(subsystem: "com.transcribeme.app", category: "pipeline")

    @Published var totalWords = 0
    @Published var recentWords = 0
    @Published var isProcessing = false

    private var captureManager: AnyObject? // AudioCaptureManager, but stored as AnyObject for availability
    private let batchDetector: BatchDetector
    private var transcriptLogger: TranscriptLogger?
    private var pluginRunner: PluginRunner?
    private let transcriptSync = TranscriptSync()

    // Word count tracking
    private var wordCounts: [(Date, Int)] = []
    private let statsWindowMinutes = 5.0

    init() {
        let config = AppConfig.shared
        self.batchDetector = BatchDetector(
            sampleRate: 16000,
            minBatchDuration: config.minBatchDuration,
            maxBatchDuration: config.maxBatchDuration,
            silenceDuration: config.silenceDuration
        )
    }

    func start() async {
        guard !isProcessing else { return }
        guard ModelManager.shared.asrReady, ModelManager.shared.vadReady else {
            logger.error("Models not ready")
            return
        }

        let config = AppConfig.shared

        // Set up VAD in batch detector
        if let vad = ModelManager.shared.vadManager {
            await batchDetector.setVadManager(vad)
        }

        // Set up transcript logger
        try? FileManager.default.createDirectory(at: config.transcriptsDir, withIntermediateDirectories: true)
        transcriptLogger = TranscriptLogger(
            outputDir: config.transcriptsDir,
            deviceName: config.deviceName
        )

        // Set up plugin runner
        let pluginsURL: URL
        if let customDir = config.pluginsDir {
            pluginsURL = URL(fileURLWithPath: customDir)
        } else {
            pluginsURL = config.appSupportDir.appendingPathComponent("Plugins")
        }
        pluginRunner = PluginRunner(pluginsDir: pluginsURL)
        await pluginRunner?.runStartupHooks()

        // Start transcript sync
        if config.syncApiUrl != nil {
            await transcriptSync.startPeriodicSync()
        }

        // Start audio capture
        if #available(macOS 14.2, *) {
            let capture = AudioCaptureManager(sampleRate: 16000) { [weak self] samples, timestamp in
                guard let self = self else { return }
                Task { @MainActor in
                    await self.handleAudio(samples, timestamp: timestamp)
                }
            }

            do {
                switch config.audioSource {
                case .systemTap:
                    try capture.startSystemTap(
                        excludeBundleIDs: config.excludeApps,
                        micDevice: config.micDevice
                    )
                case .device:
                    try capture.startDeviceOnly(deviceName: config.micDevice)
                }
                self.captureManager = capture
                isProcessing = true
                logger.info("Pipeline started (source=\(config.audioSource.rawValue))")
            } catch {
                logger.error("Failed to start capture: \(error.localizedDescription)")
            }
        } else {
            logger.error("System tap requires macOS 14.2+")
        }
    }

    func stop() async {
        isProcessing = false
        if #available(macOS 14.2, *) {
            (captureManager as? AudioCaptureManager)?.stop()
        }
        captureManager = nil

        // Flush remaining audio
        if let batch = await batchDetector.flush() {
            await processBatch(batch)
        }

        await transcriptLogger?.close()
        await pluginRunner?.runShutdownHooks()
        await transcriptSync.stopPeriodicSync()

        logger.info("Pipeline stopped")
    }

    // MARK: - Audio handling

    private func handleAudio(_ samples: [Float], timestamp: Date) async {
        guard let batch = await batchDetector.addAudio(samples, timestamp: timestamp) else { return }
        await processBatch(batch)
    }

    private func processBatch(_ batch: BatchDetector.BatchResult) async {
        let config = AppConfig.shared
        let deviceName = config.deviceName
        let diarizationEnabled = config.diarizationEnabled

        guard let asr = ModelManager.shared.asrManager else { return }

        let duration = Double(batch.audio.count) / 16000.0
        logger.info("Processing batch: \(String(format: "%.1f", duration))s")

        do {
            let result = try await asr.transcribe(batch.audio, source: .system)
            let text = result.text.trimmingCharacters(in: .whitespacesAndNewlines)

            guard !text.isEmpty else { return }

            var utterances: [BatchUtterance]

            if diarizationEnabled, let diarizer = ModelManager.shared.diarizationManager {
                do {
                    let diarResult = try await diarizer.process(audio: batch.audio)
                    utterances = mergeAsrWithDiarization(text: text, diarization: diarResult, duration: duration)
                } catch {
                    logger.error("Diarization failed, falling back to single speaker: \(error.localizedDescription)")
                    utterances = [BatchUtterance(
                        speaker: deviceName,
                        confidence: Double(result.confidence),
                        start: 0,
                        end: duration,
                        text: text
                    )]
                }
            } else {
                utterances = [BatchUtterance(
                    speaker: deviceName,
                    confidence: Double(result.confidence),
                    start: 0,
                    end: duration,
                    text: text
                )]
            }

            let envelope = BatchEnvelope(
                device: deviceName,
                utterances: utterances,
                timestamp: batch.timestamp
            )

            // Log to file
            await transcriptLogger?.write(envelope)

            // Save audio clip if enabled (trimmed to speech boundaries + 300ms tail)
            if config.saveAudio {
                saveAudioClip(batch, envelope: envelope, config: config)
            }

            // Dispatch to plugins
            if let jsonLine = envelope.toJSONLine() {
                await pluginRunner?.dispatch(jsonLine: jsonLine)
            }

            // Update word counts
            let wordCount = text.split(separator: " ").count
            totalWords += wordCount
            wordCounts.append((Date(), wordCount))
            pruneOldCounts()

            logger.info("Transcribed \(wordCount) words from \(utterances.count) utterance(s)")
        } catch {
            logger.error("ASR failed: \(error.localizedDescription)")
        }
    }

    private func mergeAsrWithDiarization(text: String, diarization: DiarizationResult, duration: Double) -> [BatchUtterance] {
        let segments = diarization.segments
        guard !segments.isEmpty else {
            return [BatchUtterance(speaker: "SPEAKER_00", confidence: 1.0, start: 0, end: duration, text: text)]
        }

        // Group consecutive segments by speaker
        var utterances: [BatchUtterance] = []
        var currentSpeaker = ""
        var currentStart = 0.0
        var currentEnd = 0.0

        for segment in segments {
            let speaker = segment.speakerId
            if speaker != currentSpeaker && !currentSpeaker.isEmpty {
                utterances.append(BatchUtterance(
                    speaker: currentSpeaker,
                    confidence: 1.0,
                    start: currentStart,
                    end: currentEnd,
                    text: ""
                ))
                currentStart = Double(segment.startTimeSeconds)
            }
            if currentSpeaker.isEmpty { currentStart = Double(segment.startTimeSeconds) }
            currentSpeaker = speaker
            currentEnd = Double(segment.endTimeSeconds)
        }
        if !currentSpeaker.isEmpty {
            utterances.append(BatchUtterance(
                speaker: currentSpeaker,
                confidence: 1.0,
                start: currentStart,
                end: currentEnd,
                text: ""
            ))
        }

        // Distribute text: assign full text to first utterance (basic approach)
        if utterances.count == 1 || !utterances.isEmpty {
            var result = utterances
            result[0] = BatchUtterance(
                speaker: result[0].speaker,
                confidence: result[0].confidence,
                start: result[0].start,
                end: result[0].end,
                text: text
            )
            return result
        }

        return utterances
    }

    // MARK: - Stats

    private func pruneOldCounts() {
        let cutoff = Date().addingTimeInterval(-statsWindowMinutes * 60)
        wordCounts.removeAll { $0.0 < cutoff }
        recentWords = wordCounts.reduce(0) { $0 + $1.1 }
    }

    // MARK: - Audio saving

    private func saveAudioClip(_ batch: BatchDetector.BatchResult, envelope: BatchEnvelope, config: AppConfig) {
        let audioDir = config.audioDir
        let cal = Calendar.current
        let now = Date()
        let subdir = audioDir
            .appendingPathComponent(String(format: "%04d", cal.component(.year, from: now)))
            .appendingPathComponent(String(format: "%02d", cal.component(.month, from: now)))
            .appendingPathComponent(String(format: "%02d", cal.component(.day, from: now)))

        try? FileManager.default.createDirectory(at: subdir, withIntermediateDirectories: true)

        let filename = "\(envelope.id).wav"
        let filepath = subdir.appendingPathComponent(filename)

        // Trim to speech boundaries with 300ms tail
        let tailSamples = Int(0.3 * 16000) // 300ms at 16kHz
        let startSample = max(0, (batch.speechStartSample ?? 0))
        let endSample: Int
        if let speechEnd = batch.speechEndSample {
            endSample = min(batch.audio.count, speechEnd + tailSamples)
        } else {
            endSample = batch.audio.count
        }

        guard startSample < endSample else { return }
        let clipped = Array(batch.audio[startSample..<endSample])

        // Write 16kHz mono 16-bit PCM WAV
        writeWav(samples: clipped, to: filepath)
        let clipDuration = Double(clipped.count) / 16000.0
        logger.info("Saved audio clip: \(filename) (\(String(format: "%.1f", clipDuration))s)")
    }

    private func writeWav(samples: [Float], to filepath: URL) {
        let sampleRate: UInt32 = 16000
        let numChannels: UInt16 = 1
        let bitsPerSample: UInt16 = 16
        let dataSize = UInt32(samples.count * 2)
        let fileSize = 36 + dataSize

        var data = Data()
        data.append(contentsOf: "RIFF".utf8)
        data.append(contentsOf: withUnsafeBytes(of: fileSize.littleEndian) { Array($0) })
        data.append(contentsOf: "WAVE".utf8)
        data.append(contentsOf: "fmt ".utf8)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: numChannels.littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: sampleRate.littleEndian) { Array($0) })
        let byteRate = sampleRate * UInt32(numChannels) * UInt32(bitsPerSample / 8)
        data.append(contentsOf: withUnsafeBytes(of: byteRate.littleEndian) { Array($0) })
        let blockAlign = numChannels * (bitsPerSample / 8)
        data.append(contentsOf: withUnsafeBytes(of: blockAlign.littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: bitsPerSample.littleEndian) { Array($0) })
        data.append(contentsOf: "data".utf8)
        data.append(contentsOf: withUnsafeBytes(of: dataSize.littleEndian) { Array($0) })

        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            let int16 = Int16(clamped * 32767.0)
            data.append(contentsOf: withUnsafeBytes(of: int16.littleEndian) { Array($0) })
        }

        try? data.write(to: filepath)
    }
}
