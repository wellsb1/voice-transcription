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
    @Published var isLoading = false
    @Published var loadingStatus: String = ""
    @Published var downloadError: String?

    private(set) var asrManager: AsrManager?
    private(set) var vadManager: VadManager?
    private(set) var diarizationManager: OfflineDiarizerManager?

    private init() {}

    var modelsDir: URL { AppConfig.shared.modelsDir }

    /// Fast filesystem check: are ASR models cached on disk?
    func coreModelsExistOnDisk() -> Bool {
        return AsrModels.modelsExist(
            at: AsrModels.defaultCacheDirectory(for: .v3),
            version: .v3
        )
    }

    /// Delete cached model files and re-download everything.
    func forceRedownloadAll() async {
        cleanup()

        // Delete ASR + VAD caches (shared FluidAudio location)
        let asrDir = AsrModels.defaultCacheDirectory(for: .v3)
        try? FileManager.default.removeItem(at: asrDir)

        let vadDir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("FluidAudio/Models/silero-vad-coreml")
        try? FileManager.default.removeItem(at: vadDir)

        // Delete diarization cache (app-specific location)
        let diarDir = modelsDir.appendingPathComponent("speaker-diarization-coreml")
        try? FileManager.default.removeItem(at: diarDir)

        AppConfig.shared.modelsReady = false
        AppConfig.shared.diarizationModelsReady = false

        await downloadCoreModels()
        await downloadDiarizationModels()
    }

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

    /// Load already-cached models. No guard on UserDefaults — checks disk directly.
    func loadModelsIfReady() async {
        isLoading = true
        loadingStatus = "Loading speech recognition..."

        do {
            let asrModels = try await AsrModels.downloadAndLoad(version: .v3)
            let asr = AsrManager()
            try await asr.initialize(models: asrModels)
            self.asrManager = asr
            self.asrReady = true

            loadingStatus = "Loading voice detection..."
            let vad = try await VadManager()
            self.vadManager = vad
            self.vadReady = true

            AppConfig.shared.modelsReady = true
            logger.info("Core models loaded")
        } catch {
            logger.error("Failed to load models: \(error.localizedDescription)")
            AppConfig.shared.modelsReady = false
        }

        loadingStatus = "Loading speaker identification..."
        do {
            let diarizer = OfflineDiarizerManager()
            try await diarizer.prepareModels(directory: modelsDir)
            self.diarizationManager = diarizer
            self.diarizationReady = true
            AppConfig.shared.diarizationModelsReady = true
            logger.info("Diarization models loaded")
        } catch {
            logger.error("Failed to load diarization models: \(error.localizedDescription)")
            AppConfig.shared.diarizationModelsReady = false
        }

        loadingStatus = ""
        isLoading = false
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

/// Calculate total size of a directory on disk (nonisolated, safe for background threads).
func modelDirectorySize(_ url: URL) -> UInt64 {
    guard let enumerator = FileManager.default.enumerator(
        at: url,
        includingPropertiesForKeys: [.fileSizeKey],
        options: [.skipsHiddenFiles]
    ) else { return 0 }

    var total: UInt64 = 0
    for case let fileURL as URL in enumerator {
        if let size = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
            total += UInt64(size)
        }
    }
    return total
}
