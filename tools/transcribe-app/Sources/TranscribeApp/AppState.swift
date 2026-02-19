import SwiftUI
import Combine
import os

/// Central app state shared across all views.
@MainActor
class AppState: ObservableObject {
    @Published var isRunning = false
    @Published var showSetup = false
    @Published var initialized = false
    @Published var totalWords = 0
    @Published var recentWords = 0

    let config = AppConfig.shared
    let models = ModelManager.shared
    let pipeline = TranscriptionPipeline()

    private let logger = Logger(subsystem: "com.transcribeme.app", category: "app")
    private var pipelineSink: AnyCancellable?

    var icon: String {
        isRunning ? "waveform.circle.fill" : "waveform"
    }

    init() {
        // Forward pipeline changes to this object so SwiftUI updates
        pipelineSink = pipeline.objectWillChange.sink { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                self.totalWords = self.pipeline.totalWords
                self.recentWords = self.pipeline.recentWords
            }
        }

        Task { @MainActor in
            await self.initialize()
        }
    }

    func initialize() async {
        guard !initialized else { return }
        initialized = true

        if config.modelsReady {
            await models.loadModelsIfReady()
            if config.autoStart && models.asrReady {
                await startTranscription()
            }
        } else {
            showSetup = true
        }
    }

    func startTranscription() async {
        guard models.asrReady else {
            showSetup = true
            return
        }

        await pipeline.start()
        isRunning = pipeline.isProcessing
        logger.info("Transcription started")
    }

    func stopTranscription() async {
        await pipeline.stop()
        isRunning = false
        logger.info("Transcription stopped")
    }

    func toggleTranscription() async {
        if isRunning {
            await stopTranscription()
        } else {
            await startTranscription()
        }
    }
}
