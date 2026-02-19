import Foundation
import FluidAudio
import os

/// Detects batch boundaries using streaming VAD.
/// Accumulates audio and signals when a batch is ready for transcription.
actor BatchDetector {
    private let sampleRate: Double
    private let minBatchDuration: Double
    private let maxBatchDuration: Double
    private let silenceDuration: Double

    private var vadManager: VadManager?
    private var vadState: VadStreamState = .initial()
    private var vadPending: [Float] = []
    private var vadProcessedSamples = 0 // total samples processed through VAD

    private var audioBuffer: [Float] = []
    private var batchTimestamp: Date?
    private var speechDetected = false
    private var silenceStart: Date?

    // Track speech boundaries (sample indices) for audio clipping
    private var firstSpeechSample: Int?
    private var lastSpeechEndSample: Int?

    private let logger = Logger(subsystem: "com.transcribeme.app", category: "batch")
    private let vadChunkSize = 4096

    struct BatchResult {
        let audio: [Float]
        let timestamp: Date
        let speechStartSample: Int? // first speech sample index
        let speechEndSample: Int?   // last speech end sample index
    }

    init(sampleRate: Double = 16000,
         minBatchDuration: Double = 30.0,
         maxBatchDuration: Double = 60.0,
         silenceDuration: Double = 0.5) {
        self.sampleRate = sampleRate
        self.minBatchDuration = minBatchDuration
        self.maxBatchDuration = maxBatchDuration
        self.silenceDuration = silenceDuration
    }

    func setVadManager(_ vad: VadManager) {
        self.vadManager = vad
        self.vadState = .initial()
    }

    func addAudio(_ samples: [Float], timestamp: Date) async -> BatchResult? {
        if batchTimestamp == nil {
            batchTimestamp = timestamp
        }
        audioBuffer.append(contentsOf: samples)
        vadPending.append(contentsOf: samples)

        let duration = Double(audioBuffer.count) / sampleRate

        if let vad = vadManager {
            while vadPending.count >= vadChunkSize {
                let chunk = Array(vadPending.prefix(vadChunkSize))
                vadPending.removeFirst(vadChunkSize)
                do {
                    let result = try await vad.processStreamingChunk(
                        chunk, state: vadState, config: .default,
                        returnSeconds: true, timeResolution: 2
                    )
                    vadState = result.state

                    if let event = result.event {
                        switch event.kind {
                        case .speechStart:
                            speechDetected = true
                            silenceStart = nil
                            if firstSpeechSample == nil {
                                firstSpeechSample = vadProcessedSamples + (event.sampleIndex > 0 ? event.sampleIndex : 0)
                            }
                            logger.info("Speech start at \(String(format: "%.1f", duration))s")
                        case .speechEnd:
                            if silenceStart == nil {
                                silenceStart = Date()
                            }
                            lastSpeechEndSample = vadProcessedSamples + (event.sampleIndex > 0 ? event.sampleIndex : vadChunkSize)
                            logger.info("Speech end at \(String(format: "%.1f", duration))s")
                        @unknown default:
                            break
                        }
                    }
                } catch {
                    logger.error("VAD error: \(error.localizedDescription)")
                }
                vadProcessedSamples += vadChunkSize
            }
        }

        let shouldFlush: Bool
        if duration >= maxBatchDuration {
            shouldFlush = true
        } else if speechDetected,
                  duration >= minBatchDuration,
                  let silStart = silenceStart,
                  Date().timeIntervalSince(silStart) >= silenceDuration {
            shouldFlush = true
        } else {
            shouldFlush = false
        }

        if shouldFlush {
            return flushBatch()
        }
        return nil
    }

    func flush() -> BatchResult? {
        guard !audioBuffer.isEmpty else { return nil }
        return flushBatch()
    }

    private func flushBatch() -> BatchResult {
        let audio = audioBuffer
        let ts = batchTimestamp ?? Date()
        let speechStart = firstSpeechSample
        let speechEnd = lastSpeechEndSample

        audioBuffer.removeAll(keepingCapacity: true)
        vadPending.removeAll(keepingCapacity: true)
        batchTimestamp = nil
        speechDetected = false
        silenceStart = nil
        firstSpeechSample = nil
        lastSpeechEndSample = nil
        vadProcessedSamples = 0

        return BatchResult(audio: audio, timestamp: ts,
                          speechStartSample: speechStart, speechEndSample: speechEnd)
    }
}
