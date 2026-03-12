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
    @Published var latestBatch: BatchEnvelope?
    @Published var startError: String?

    private var captureManager: AnyObject?
    private let batchDetector: BatchDetector
    private var transcriptLogger: TranscriptLogger?
    private var pluginRunner: PluginRunner?
    private let transcriptSync = TranscriptSync()
    private let batchProcessor: BatchProcessor
    private var lastBatchTime: Date?
    private let speakerTimeoutSeconds: TimeInterval = 1800

    // Word count tracking
    private var wordCounts: [(Date, Int)] = []
    private let statsWindowMinutes = 5.0
    private var statsTimer: Timer?

    init() {
        let config = AppConfig.shared
        self.batchDetector = BatchDetector(
            sampleRate: 16000,
            minBatchDuration: config.minBatchDuration,
            maxBatchDuration: config.maxBatchDuration,
            silenceDuration: config.silenceDuration
        )
        self.batchProcessor = BatchProcessor()
    }

    func startSync() async {
        await transcriptSync.startPeriodicSync()
    }

    func start() async {
        guard !isProcessing else {
            NSLog("[pipeline] start() skipped: already processing")
            return
        }
        NSLog("[pipeline] start() called: asrReady=%d vadReady=%d vadManager=%@",
              ModelManager.shared.asrReady ? 1 : 0,
              ModelManager.shared.vadReady ? 1 : 0,
              ModelManager.shared.vadManager == nil ? "nil" : "set")
        guard ModelManager.shared.asrReady, ModelManager.shared.vadReady else {
            NSLog("[pipeline] start() aborted: models not ready")
            return
        }

        let config = AppConfig.shared

        // Reset batch detector for new session
        await batchDetector.resume()

        // Set up VAD in batch detector
        if let vad = ModelManager.shared.vadManager {
            await batchDetector.setVadManager(vad)
        } else {
            NSLog("[pipeline] WARNING: vadManager is nil! vadReady=%d asrReady=%d",
                  ModelManager.shared.vadReady ? 1 : 0,
                  ModelManager.shared.asrReady ? 1 : 0)
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

        // Capture references for the audio callback (no main actor needed)
        let detector = batchDetector
        let processor = batchProcessor

        // Start audio capture
        if #available(macOS 14.2, *) {
            var audioChunkCount = 0
            let capture = AudioCaptureManager(sampleRate: 16000) { samples, timestamp in
                audioChunkCount += 1
                if audioChunkCount % 100 == 1 {
                    let sumSq = samples.reduce(Float(0)) { $0 + $1 * $1 }
                    let rms = sqrt(sumSq / Float(max(samples.count, 1)))
                    let peak = samples.map { abs($0) }.max() ?? 0
                    NSLog("[pipeline] chunk #%d, %d samples, rms=%.6f peak=%.4f", audioChunkCount, samples.count, rms, peak)
                }
                Task {
                    guard let batch = await detector.addAudio(samples, timestamp: timestamp) else { return }
                    NSLog("[pipeline] batch ready, %d samples (%.1fs)", batch.audio.count, Double(batch.audio.count) / 16000.0)
                    await processor.process(batch)
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
                startStatsTimer()
                NSLog("[pipeline] capture started successfully, source=%@", config.audioSource.rawValue)

                // Wire batch processor results back to main actor
                let transcriptLogger = self.transcriptLogger
                let pluginRunner = self.pluginRunner
                let transcriptSync = self.transcriptSync
                await processor.setOnBatchComplete { [weak self] result in
                    // Write to file, plugins, sync — all off main actor
                    await transcriptLogger?.write(result.envelope)

                    if result.saveAudio, let clipData = result.audioClipData {
                        try? clipData.data.write(to: clipData.filepath)
                    }

                    if let jsonLine = result.envelope.toJSONLine() {
                        await pluginRunner?.dispatch(jsonLine: jsonLine)
                    }

                    await transcriptSync.runSync()

                    // Only hop to main actor for UI updates
                    await MainActor.run { [weak self] in
                        guard let self else { return }
                        self.latestBatch = result.envelope
                        self.totalWords += result.wordCount
                        self.wordCounts.append((Date(), result.wordCount))
                        self.pruneOldCounts()
                    }
                }

                logger.info("Pipeline started (source=\(config.audioSource.rawValue))")
            } catch {
                NSLog("[pipeline] FAILED to start capture: %@", "\(error)")
                self.startError = "\(error)"
                logger.error("Failed to start capture: \(error.localizedDescription)")
            }
        } else {
            logger.error("System tap requires macOS 14.2+")
        }
    }

    func stop() async {
        isProcessing = false
        statsTimer?.invalidate()
        statsTimer = nil
        if #available(macOS 14.2, *) {
            (captureManager as? AudioCaptureManager)?.stop()
        }
        captureManager = nil

        // Flush remaining audio
        if let batch = await batchDetector.flush() {
            await batchProcessor.process(batch)
        }

        await transcriptLogger?.close()
        await pluginRunner?.runShutdownHooks()

        logger.info("Pipeline stopped")
    }

    // MARK: - Stats

    private func startStatsTimer() {
        statsTimer?.invalidate()
        statsTimer = Timer.scheduledTimer(withTimeInterval: 30, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.pruneOldCounts()
            }
        }
    }

    private func pruneOldCounts() {
        let cutoff = Date().addingTimeInterval(-statsWindowMinutes * 60)
        wordCounts.removeAll { $0.0 < cutoff }
        recentWords = wordCounts.reduce(0) { $0 + $1.1 }
    }
}

// MARK: - BatchProcessor (nonisolated — runs off main actor)

/// Handles the heavy work (ASR, diarization, audio clipping) entirely off the main actor.
actor BatchProcessor {
    private let logger = Logger(subsystem: "com.transcribeme.app", category: "processor")
    private let speakerManager = SpeakerManager()
    private var lastBatchTime: Date?
    private let speakerTimeoutSeconds: TimeInterval = 1800

    struct AudioClipData {
        let filepath: URL
        let data: Data
    }

    struct BatchResult {
        let envelope: BatchEnvelope
        let wordCount: Int
        let saveAudio: Bool
        let audioClipData: AudioClipData?
    }

    /// Called after each batch is processed (runs off main actor).
    private var onBatchComplete: ((BatchResult) async -> Void)?

    func setOnBatchComplete(_ handler: @escaping (BatchResult) async -> Void) {
        self.onBatchComplete = handler
    }

    func process(_ batch: BatchDetector.BatchResult) async {
        let config = await MainActor.run { AppConfig.shared }
        let deviceName = await MainActor.run { config.deviceName }
        let saveAudio = await MainActor.run { config.saveAudio }
        let audioDir = await MainActor.run { config.audioDir }

        // Reset speaker registry after inactivity gap
        let now = Date()
        if let last = lastBatchTime, now.timeIntervalSince(last) > speakerTimeoutSeconds {
            speakerManager.reset()
            logger.info("Speaker registry reset after \(Int(now.timeIntervalSince(last)))s inactivity")
        }
        lastBatchTime = now

        guard let asr = await MainActor.run(body: { ModelManager.shared.asrManager }) else {
            NSLog("[processor] no ASR manager available")
            return
        }

        let duration = Double(batch.audio.count) / 16000.0
        NSLog("[processor] starting ASR for %.1fs batch", duration)

        do {
            let result = try await asr.transcribe(batch.audio, source: .system)
            let text = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
            NSLog("[processor] ASR done: %d chars, conf=%.2f", text.count, result.confidence)

            guard !text.isEmpty else { return }

            var utterances: [BatchUtterance]

            if let diarizer = await MainActor.run(body: { ModelManager.shared.diarizationManager }) {
                do {
                    let diarResult = try await diarizer.process(audio: batch.audio)
                    logger.info("Diarization: \(diarResult.segments.count) segments, \(diarResult.speakerDatabase?.count ?? 0) speakers in DB")
                    utterances = mergeAsrWithDiarization(
                        tokenTimings: result.tokenTimings,
                        text: text,
                        diarization: diarResult,
                        duration: duration
                    )
                } catch {
                    logger.error("Diarization failed, falling back to single speaker: \(error.localizedDescription)")
                    utterances = [BatchUtterance(
                        speaker: "SPEAKER_00",
                        confidence: Double(result.confidence),
                        start: 0,
                        end: duration,
                        text: text
                    )]
                }
            } else {
                logger.info("Diarization skipped (models not loaded)")
                utterances = [BatchUtterance(
                    speaker: "SPEAKER_00",
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

            // Prepare audio clip data if needed
            var clipData: AudioClipData? = nil
            if saveAudio {
                clipData = buildAudioClip(batch, envelope: envelope, audioDir: audioDir)
            }

            let wordCount = text.split(separator: " ").count
            logger.info("Transcribed \(wordCount) words from \(utterances.count) utterance(s)")

            let batchResult = BatchResult(
                envelope: envelope,
                wordCount: wordCount,
                saveAudio: saveAudio,
                audioClipData: clipData
            )

            await onBatchComplete?(batchResult)
        } catch {
            logger.error("ASR failed: \(error.localizedDescription)")
        }
    }

    // MARK: - Audio clipping

    private func buildAudioClip(
        _ batch: BatchDetector.BatchResult,
        envelope: BatchEnvelope,
        audioDir: URL
    ) -> AudioClipData? {
        let cal = Calendar.current
        let now = Date()
        let subdir = audioDir
            .appendingPathComponent(String(format: "%04d", cal.component(.year, from: now)))
            .appendingPathComponent(String(format: "%02d", cal.component(.month, from: now)))
            .appendingPathComponent(String(format: "%02d", cal.component(.day, from: now)))

        try? FileManager.default.createDirectory(at: subdir, withIntermediateDirectories: true)

        let filename = "\(envelope.id).wav"
        let filepath = subdir.appendingPathComponent(filename)

        let tailSamples = Int(0.3 * 16000)
        let startSample = max(0, (batch.speechStartSample ?? 0))
        let endSample: Int
        if let speechEnd = batch.speechEndSample {
            endSample = min(batch.audio.count, speechEnd + tailSamples)
        } else {
            endSample = batch.audio.count
        }

        guard startSample < endSample else { return nil }
        let clipped = Array(batch.audio[startSample..<endSample])

        let data = buildWav(samples: clipped)
        let clipDuration = Double(clipped.count) / 16000.0
        logger.info("Prepared audio clip: \(filename) (\(String(format: "%.1f", clipDuration))s)")
        return AudioClipData(filepath: filepath, data: data)
    }

    private func buildWav(samples: [Float]) -> Data {
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

        return data
    }

    // MARK: - Diarization merge

    private struct SpeakerTurn {
        let speaker: String
        let start: Double
        let end: Double
    }

    private func formatSpeakerId(_ id: String) -> String {
        if let num = Int(id) {
            return String(format: "SPEAKER_%02d", num - 1)
        }
        return id
    }

    private func mergeAsrWithDiarization(
        tokenTimings: [TokenTiming]?,
        text: String,
        diarization: DiarizationResult,
        duration: Double
    ) -> [BatchUtterance] {
        let segments = diarization.segments
        guard !segments.isEmpty else {
            return [BatchUtterance(speaker: "SPEAKER_00", confidence: 1.0, start: 0, end: duration, text: text)]
        }

        // Build local → persistent speaker ID mapping via SpeakerManager
        var idMap: [String: String] = [:]
        if let db = diarization.speakerDatabase {
            for (localId, embedding) in db {
                let speakerDuration = segments
                    .filter { $0.speakerId == localId }
                    .reduce(Float(0)) { $0 + ($1.endTimeSeconds - $1.startTimeSeconds) }

                if let speaker = speakerManager.assignSpeaker(embedding, speechDuration: speakerDuration) {
                    idMap[localId] = formatSpeakerId(speaker.id)
                }
            }
        }

        // Group consecutive diarization segments by persistent speaker ID into speaker turns
        var turns: [SpeakerTurn] = []
        var currentSpeaker = ""
        var currentStart = 0.0
        var currentEnd = 0.0

        for segment in segments {
            let speaker = idMap[segment.speakerId] ?? formatSpeakerId(segment.speakerId)
            if speaker != currentSpeaker && !currentSpeaker.isEmpty {
                turns.append(SpeakerTurn(speaker: currentSpeaker, start: currentStart, end: currentEnd))
                currentStart = Double(segment.startTimeSeconds)
            }
            if currentSpeaker.isEmpty { currentStart = Double(segment.startTimeSeconds) }
            currentSpeaker = speaker
            currentEnd = Double(segment.endTimeSeconds)
        }
        if !currentSpeaker.isEmpty {
            turns.append(SpeakerTurn(speaker: currentSpeaker, start: currentStart, end: currentEnd))
        }

        guard !turns.isEmpty else {
            return [BatchUtterance(speaker: "SPEAKER_00", confidence: 1.0, start: 0, end: duration, text: text)]
        }

        if let timings = tokenTimings, !timings.isEmpty {
            return assignWordsToTurns(timings: timings, turns: turns, duration: duration)
        }

        let longestTurn = turns.max(by: { ($0.end - $0.start) < ($1.end - $1.start) })!
        return [BatchUtterance(
            speaker: longestTurn.speaker,
            confidence: 1.0,
            start: longestTurn.start,
            end: longestTurn.end,
            text: text
        )]
    }

    private struct WordTiming {
        let word: String
        let startTime: Double
        let endTime: Double
    }

    private func buildWordTimings(from tokenTimings: [TokenTiming]) -> [WordTiming] {
        var words: [WordTiming] = []
        var currentWord = ""
        var wordStart = 0.0
        var wordEnd = 0.0

        for timing in tokenTimings {
            let token = timing.token
            if token.isEmpty || token == "<blank>" || token == "<pad>" { continue }

            let startsNewWord = token.hasPrefix("▁") || token.hasPrefix(" ") || currentWord.isEmpty

            if startsNewWord && !currentWord.isEmpty {
                let trimmed = currentWord.trimmingCharacters(in: .whitespaces)
                if !trimmed.isEmpty {
                    words.append(WordTiming(word: trimmed, startTime: wordStart, endTime: wordEnd))
                }
                currentWord = ""
            }

            if startsNewWord {
                var stripped = token
                if stripped.hasPrefix("▁") || stripped.hasPrefix(" ") {
                    stripped = String(stripped.dropFirst())
                }
                currentWord = stripped
                wordStart = timing.startTime
            } else {
                currentWord += token
            }
            wordEnd = timing.endTime
        }

        let trimmed = currentWord.trimmingCharacters(in: .whitespaces)
        if !trimmed.isEmpty {
            words.append(WordTiming(word: trimmed, startTime: wordStart, endTime: wordEnd))
        }
        return words
    }

    private func assignWordsToTurns(
        timings: [TokenTiming],
        turns: [SpeakerTurn],
        duration: Double
    ) -> [BatchUtterance] {
        let words = buildWordTimings(from: timings)
        guard !words.isEmpty else {
            return [BatchUtterance(speaker: turns[0].speaker, confidence: 1.0, start: 0, end: duration, text: "")]
        }

        func speakerForWord(_ word: WordTiming) -> String {
            let wordMid = (word.startTime + word.endTime) / 2.0
            for turn in turns {
                if wordMid >= turn.start && wordMid <= turn.end {
                    return turn.speaker
                }
            }
            var bestTurn = turns[0]
            var bestDist = Double.greatestFiniteMagnitude
            for turn in turns {
                let dist = min(abs(wordMid - turn.start), abs(wordMid - turn.end))
                if dist < bestDist {
                    bestDist = dist
                    bestTurn = turn
                }
            }
            return bestTurn.speaker
        }

        var utterances: [BatchUtterance] = []
        var currentWords: [String] = []
        var currentSpeaker = ""
        var utteranceStart = 0.0
        var utteranceEnd = 0.0

        for word in words {
            let speaker = speakerForWord(word)

            if speaker != currentSpeaker && !currentWords.isEmpty {
                utterances.append(BatchUtterance(
                    speaker: currentSpeaker,
                    confidence: 1.0,
                    start: utteranceStart,
                    end: utteranceEnd,
                    text: currentWords.joined(separator: " ")
                ))
                currentWords = []
            }

            if currentWords.isEmpty {
                utteranceStart = word.startTime
            }
            currentSpeaker = speaker
            utteranceEnd = word.endTime
            currentWords.append(word.word)
        }

        if !currentWords.isEmpty {
            utterances.append(BatchUtterance(
                speaker: currentSpeaker,
                confidence: 1.0,
                start: utteranceStart,
                end: utteranceEnd,
                text: currentWords.joined(separator: " ")
            ))
        }

        return utterances
    }
}
