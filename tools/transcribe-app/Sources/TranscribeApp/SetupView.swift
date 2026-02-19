import SwiftUI

struct SetupView: View {
    @EnvironmentObject var appState: AppState
    @ObservedObject private var models = ModelManager.shared
    @State private var enableDiarization = false

    var body: some View {
        VStack(spacing: 24) {
            // Branding
            WaveformIcon(size: 64)

            HStack(spacing: 0) {
                Text("transcribed")
                    .font(.system(size: 28, weight: .bold))
                    .foregroundColor(Color(red: 0x11/255, green: 0x18/255, blue: 0x27/255))
                    .tracking(-0.5)
                Text(".")
                    .font(.system(size: 28, weight: .bold))
                    .foregroundColor(Color(red: 0x11/255, green: 0x18/255, blue: 0x27/255))
                    .tracking(-0.5)
                Text("me")
                    .font(.system(size: 28, weight: .bold))
                    .foregroundColor(Color(red: 0x9c/255, green: 0xa3/255, blue: 0xaf/255))
                    .tracking(-0.5)
            }

            Text("transcribed.me needs to download speech recognition models before it can start. This is a one-time download of approximately 2 GB.")
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 400)

            // Diarization toggle
            Toggle(isOn: $enableDiarization) {
                VStack(alignment: .leading) {
                    Text("Enable speaker identification")
                        .font(.body)
                    Text("Downloads additional models (~1 GB) to identify who is speaking")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .toggleStyle(.checkbox)
            .frame(maxWidth: 400)

            // Progress
            if models.isDownloading {
                VStack(spacing: 8) {
                    ProgressView(value: models.downloadProgress)
                        .frame(width: 300)
                    Text(models.downloadStatus)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            // Error
            if let error = models.downloadError {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .frame(maxWidth: 400)
            }

            // Actions
            if models.asrReady {
                Button("Start Transcribing") {
                    appState.showSetup = false
                    Task { await appState.startTranscription() }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
            } else if !models.isDownloading {
                Button("Download Models") {
                    Task {
                        await models.downloadCoreModels()
                        if enableDiarization {
                            await models.downloadDiarizationModels()
                        }
                    }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
            }
        }
        .padding(40)
        .frame(width: 500, height: 480)
    }
}
