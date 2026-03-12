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
    @Published var needsMicPermission = false
    @Published var needsScreenPermission = false

    let config = AppConfig.shared
    let models = ModelManager.shared
    let pipeline = TranscriptionPipeline()
    let transcriptStore: TranscriptStore

    private let logger = Logger(subsystem: "com.transcribeme.app", category: "app")
    private var pipelineSink: AnyCancellable?
    private var batchSink: AnyCancellable?
    private var wakeSink: AnyCancellable?

    var icon: String {
        if isRunning {
            return "waveform.circle.fill"
        } else if config.syncApiKey == nil {
            return "waveform.badge.exclamationmark"
        } else {
            return "waveform"
        }
    }

    init() {
        self.transcriptStore = TranscriptStore(transcriptsDir: AppConfig.shared.transcriptsDir)

        // Forward pipeline changes to this object so SwiftUI updates
        pipelineSink = pipeline.objectWillChange.sink { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                self.totalWords = self.pipeline.totalWords
                self.recentWords = self.pipeline.recentWords
            }
        }

        // Forward live batches to the transcript store
        batchSink = pipeline.$latestBatch.compactMap { $0 }.sink { [weak self] envelope in
            Task { @MainActor in
                self?.transcriptStore.addLive(envelope)
            }
        }

        // Restart audio capture after wake from sleep
        wakeSink = NotificationCenter.default.publisher(
            for: NSWorkspace.didWakeNotification
        ).sink { [weak self] _ in
            Task { @MainActor in
                guard let self, self.isRunning else { return }
                self.logger.info("System woke from sleep, restarting pipeline")
                await self.pipeline.stop()
                await self.pipeline.start()
                self.isRunning = self.pipeline.isProcessing
            }
        }

        Task { @MainActor in
            await self.initialize()
        }
    }

    func initialize() async {
        guard !initialized else { return }
        initialized = true

        // Validate stored API key and start sync on startup
        if config.syncApiKey != nil {
            if let email = await AuthService.shared.checkAuth() {
                config.userEmail = email
                await pipeline.startSync()
            } else {
                await AuthService.shared.logout()
            }
        }

        // Check permissions before proceeding
        if await !checkPermissions() {
            return // PermissionView will be shown
        }

        // Check if models exist on disk (fast filesystem check)
        if models.coreModelsExistOnDisk() {
            // Models cached — load them (shows loading spinner, not SetupView)
            await models.loadModelsIfReady()

            if models.asrReady {
                config.modelsReady = true
                if config.autoStart {
                    await startTranscription()
                }
            } else {
                // Models on disk but failed to load (corrupted?) — offer re-download
                showSetup = true
            }
        } else if config.modelsReady {
            // UserDefaults says ready but files missing — try loading anyway
            // (downloadAndLoad will re-download if needed)
            await models.loadModelsIfReady()
            if models.asrReady {
                if config.autoStart {
                    await startTranscription()
                }
            } else {
                showSetup = true
            }
        } else {
            // Genuine first launch — no models on disk
            showSetup = true
        }
    }

    /// Returns true if all required permissions are granted.
    private func checkPermissions() async -> Bool {
        // Microphone: request if not determined, flag if denied
        let micStatus = PermissionManager.micStatus()
        if micStatus == .notDetermined {
            let granted = await PermissionManager.requestMic()
            if !granted {
                needsMicPermission = true
                return false
            }
        } else if micStatus == .denied || micStatus == .restricted {
            needsMicPermission = true
            return false
        }

        // Screen Recording: only needed for system tap mode
        if config.audioSource == .systemTap {
            if !PermissionManager.hasScreenRecording() {
                needsScreenPermission = true
                return false
            }
        }

        return true
    }

    /// Re-check permissions after user grants them in System Settings.
    func recheckPermissions() async {
        needsMicPermission = false
        needsScreenPermission = false

        if await checkPermissions() {
            // Permissions now OK — continue initialization flow
            await continueAfterPermissions()
        }
    }

    private func continueAfterPermissions() async {
        if models.coreModelsExistOnDisk() {
            await models.loadModelsIfReady()
            if models.asrReady {
                config.modelsReady = true
                if config.autoStart {
                    await startTranscription()
                }
            } else {
                showSetup = true
            }
        } else if config.modelsReady {
            await models.loadModelsIfReady()
            if models.asrReady {
                if config.autoStart {
                    await startTranscription()
                }
            } else {
                showSetup = true
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
