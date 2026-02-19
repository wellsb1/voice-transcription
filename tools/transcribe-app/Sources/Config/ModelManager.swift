import Foundation
import FluidAudio
import os

/// Manages FluidAudio model downloading, caching, and initialization.
@MainActor
class ModelManager: ObservableObject {
    static let shared = ModelManager()

    private let logger = Logger(subsystem: "com.transcribeme.app", category: "models")

    @Published var asrReady = false
    @Published var vadReady = false
    @Published var diarizationReady = false
    @Published var downloadProgress: Double = 0
    @Published var downloadStatus: String = ""
    @Published var isDownloading = false
    @Published var downloadError: String?

    private(set) var asrManager: AsrManager?
    private(set) var vadManager: VadManager?
    private(set) var diarizationManager: OfflineDiarizerManager?

    private init() {}

    var modelsDir: URL { AppConfig.shared.modelsDir }

    /// Download and initialize ASR + VAD models.
    func downloadCoreModels() async {
        isDownloading = true
        downloadError = nil
        downloadProgress = 0

        do {
            try FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)

            downloadStatus = "Downloading speech recognition model..."
            downloadProgress = 0.1
            let asrModels = try await AsrModels.downloadAndLoad(version: .v3)
            downloadProgress = 0.5

            downloadStatus = "Initializing speech recognition..."
            let asr = AsrManager()
            try await asr.initialize(models: asrModels)
            self.asrManager = asr
            self.asrReady = true
            downloadProgress = 0.7

            downloadStatus = "Downloading voice activity detection model..."
            let vad = try await VadManager()
            self.vadManager = vad
            self.vadReady = true
            downloadProgress = 1.0

            downloadStatus = "Ready"
            AppConfig.shared.modelsReady = true
            logger.info("Core models ready")
        } catch {
            downloadError = "Failed to download models: \(error.localizedDescription)"
            downloadStatus = "Download failed"
            logger.error("Model download failed: \(error.localizedDescription)")
        }

        isDownloading = false
    }

    /// Download and initialize diarization models.
    func downloadDiarizationModels() async {
        do {
            downloadStatus = "Downloading speaker identification models..."
            let diarizer = OfflineDiarizerManager()
            try await diarizer.prepareModels(directory: modelsDir)
            self.diarizationManager = diarizer
            self.diarizationReady = true
            AppConfig.shared.diarizationModelsReady = true
            downloadStatus = "Speaker identification ready"
            logger.info("Diarization models ready")
        } catch {
            downloadError = "Failed to download diarization models: \(error.localizedDescription)"
            logger.error("Diarization download failed: \(error.localizedDescription)")
        }
    }

    /// Load already-downloaded models (for app restart).
    func loadModelsIfReady() async {
        guard AppConfig.shared.modelsReady else { return }

        do {
            let asrModels = try await AsrModels.downloadAndLoad(version: .v3)
            let asr = AsrManager()
            try await asr.initialize(models: asrModels)
            self.asrManager = asr
            self.asrReady = true

            let vad = try await VadManager()
            self.vadManager = vad
            self.vadReady = true

            logger.info("Core models loaded")
        } catch {
            logger.error("Failed to load models: \(error.localizedDescription)")
            AppConfig.shared.modelsReady = false
        }

        if AppConfig.shared.diarizationModelsReady {
            do {
                let diarizer = OfflineDiarizerManager()
                try await diarizer.prepareModels(directory: modelsDir)
                self.diarizationManager = diarizer
                self.diarizationReady = true
                logger.info("Diarization models loaded")
            } catch {
                logger.error("Failed to load diarization models: \(error.localizedDescription)")
                AppConfig.shared.diarizationModelsReady = false
            }
        }
    }

    func cleanup() {
        asrManager?.cleanup()
        asrManager = nil
        vadManager = nil
        diarizationManager = nil
        asrReady = false
        vadReady = false
        diarizationReady = false
    }
}
