import SwiftUI
import AppKit

struct PermissionView: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        VStack(spacing: 24) {
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

            Text("transcribed.me needs your permission to capture audio.")
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 400)

            VStack(alignment: .leading, spacing: 16) {
                if appState.needsMicPermission {
                    PermissionRow(
                        icon: "mic.fill",
                        title: "Microphone",
                        description: "Required to capture speech for transcription.",
                        action: {
                            NSWorkspace.shared.open(
                                URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone")!
                            )
                        }
                    )
                }

                if appState.needsScreenPermission {
                    PermissionRow(
                        icon: "rectangle.on.rectangle",
                        title: "Screen & System Audio Recording",
                        description: "Required to capture system audio from other apps.",
                        action: {
                            NSWorkspace.shared.open(
                                URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture")!
                            )
                        }
                    )
                }
            }
            .frame(maxWidth: 400)

            Button("Check Again") {
                Task { await appState.recheckPermissions() }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
        }
        .padding(40)
        .frame(width: 500, height: 440)
    }
}

private struct PermissionRow: View {
    let icon: String
    let title: String
    let description: String
    let action: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundStyle(.orange)
                .frame(width: 32)

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.headline)
                Text(description)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            Button("Open Settings") { action() }
                .controlSize(.small)
        }
    }
}

struct LoadingView: View {
    @ObservedObject private var models = ModelManager.shared

    var body: some View {
        VStack(spacing: 20) {
            WaveformIcon(size: 48)

            HStack(spacing: 0) {
                Text("transcribed")
                    .font(.system(size: 22, weight: .bold))
                    .foregroundColor(Color(red: 0x11/255, green: 0x18/255, blue: 0x27/255))
                    .tracking(-0.4)
                Text(".")
                    .font(.system(size: 22, weight: .bold))
                    .foregroundColor(Color(red: 0x11/255, green: 0x18/255, blue: 0x27/255))
                    .tracking(-0.4)
                Text("me")
                    .font(.system(size: 22, weight: .bold))
                    .foregroundColor(Color(red: 0x9c/255, green: 0xa3/255, blue: 0xaf/255))
                    .tracking(-0.4)
            }

            VStack(spacing: 8) {
                ProgressView()
                    .controlSize(.regular)
                Text(models.loadingStatus)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(32)
        .frame(width: 320, height: 240)
    }
}

struct SetupView: View {
    @EnvironmentObject var appState: AppState
    @ObservedObject private var models = ModelManager.shared

    var body: some View {
        VStack(spacing: 24) {
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

            Text("transcribed.me needs to download speech recognition and speaker identification models before it can start. This is a one-time download of approximately 500 MB.")
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 400)

            // Progress
            if models.isDownloading {
                VStack(spacing: 8) {
                    ProgressView()
                        .controlSize(.regular)
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

            // Download button (only when not already downloading)
            if !models.isDownloading {
                Button(models.downloadError != nil ? "Retry Download" : "Download Models") {
                    Task {
                        await models.downloadCoreModels()
                        await models.downloadDiarizationModels()
                    }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
            }
        }
        .padding(40)
        .frame(width: 500, height: 440)
        .onChange(of: models.asrReady) { _, ready in
            if ready {
                appState.showSetup = false
                Task { await appState.startTranscription() }
            }
        }
    }
}
