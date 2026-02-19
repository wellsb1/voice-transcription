import SwiftUI

struct AudioSettingsTab: View {
    @ObservedObject private var config = AppConfig.shared

    var body: some View {
        Form {
            Section("Audio Source") {
                Picker("Source", selection: $config.audioSource) {
                    Text("Microphone only").tag(AudioSource.device)
                    Text("System audio + Microphone").tag(AudioSource.systemTap)
                }
                .pickerStyle(.radioGroup)

                if config.audioSource == .systemTap {
                    TextField("Mic device name (optional)", text: Binding(
                        get: { config.micDevice ?? "" },
                        set: { config.micDevice = $0.isEmpty ? nil : $0 }
                    ))
                    .textFieldStyle(.roundedBorder)

                    VStack(alignment: .leading) {
                        Text("Excluded apps")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        TextField("Bundle IDs (comma-separated)", text: Binding(
                            get: { config.excludeApps.joined(separator: ", ") },
                            set: { config.excludeApps = $0.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) } }
                        ))
                        .textFieldStyle(.roundedBorder)
                    }
                }
            }

            Section("Device") {
                TextField("Device name", text: $config.deviceName)
                    .textFieldStyle(.roundedBorder)
                Text("Identifies this device in transcripts")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
    }
}

struct TranscriptionSettingsTab: View {
    @EnvironmentObject var appState: AppState
    @ObservedObject private var config = AppConfig.shared

    var body: some View {
        Form {
            Section("Speaker Identification") {
                Toggle("Enable speaker diarization", isOn: $config.diarizationEnabled)

                if config.diarizationEnabled && !appState.models.diarizationReady {
                    Button("Download Speaker ID Models") {
                        Task { await appState.models.downloadDiarizationModels() }
                    }
                    if appState.models.isDownloading {
                        ProgressView(appState.models.downloadStatus)
                    }
                }
            }

            Section("Batch Settings") {
                HStack {
                    Text("Min duration")
                    Slider(value: $config.minBatchDuration, in: 10...60, step: 5)
                    Text("\(Int(config.minBatchDuration))s")
                        .monospacedDigit()
                        .frame(width: 30)
                }
                HStack {
                    Text("Max duration")
                    Slider(value: $config.maxBatchDuration, in: 30...120, step: 10)
                    Text("\(Int(config.maxBatchDuration))s")
                        .monospacedDigit()
                        .frame(width: 30)
                }
            }

            Section("Startup") {
                Toggle("Auto-start transcription on launch", isOn: $config.autoStart)
                Toggle("Save audio recordings", isOn: $config.saveAudio)
            }
        }
        .padding()
    }
}

struct SyncSettingsTab: View {
    @ObservedObject private var config = AppConfig.shared
    @State private var isLoggingIn = false
    @State private var loginError: String?
    @State private var loginTask: Task<Void, Never>?
    @State private var showAdvanced = false
    @State private var apiKeyInput = ""

    private var isLoggedIn: Bool {
        config.syncApiKey != nil
    }

    var body: some View {
        Form {
            Section("Account") {
                if isLoggedIn {
                    HStack {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                        Text("Signed in")
                            .font(.body)
                        Spacer()
                        if let key = config.syncApiKey {
                            Text(String(key.prefix(10)) + "...")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .monospaced()
                        }
                    }

                    Text("Transcripts sync every 30 minutes")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    Button("Log Out") {
                        loginTask?.cancel()
                        Task { await AuthService.shared.logout() }
                    }
                } else if isLoggingIn {
                    HStack {
                        ProgressView()
                            .controlSize(.small)
                        Text("Waiting for authorization...")
                            .font(.body)
                            .foregroundStyle(.secondary)
                    }

                    Text("Complete authorization in your browser")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    Button("Cancel") {
                        loginTask?.cancel()
                        loginTask = nil
                        isLoggingIn = false
                    }
                } else {
                    Text("Sign in to sync transcripts to transcribed.me")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    Button("Login") {
                        loginError = nil
                        isLoggingIn = true
                        loginTask = Task {
                            do {
                                try await AuthService.shared.login()
                                isLoggingIn = false
                            } catch is CancellationError {
                                isLoggingIn = false
                            } catch {
                                loginError = error.localizedDescription
                                isLoggingIn = false
                            }
                        }
                    }
                    .buttonStyle(.borderedProminent)
                }

                if let error = loginError {
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(.red)
                }
            }

            DisclosureGroup("Advanced", isExpanded: $showAdvanced) {
                TextField("API URL", text: Binding(
                    get: { config.syncApiUrl ?? "" },
                    set: { config.syncApiUrl = $0.isEmpty ? nil : $0 }
                ))
                .textFieldStyle(.roundedBorder)

                SecureField("API Key", text: $apiKeyInput)
                    .textFieldStyle(.roundedBorder)
                    .onChange(of: apiKeyInput) { _, newValue in
                        config.syncApiKey = newValue.isEmpty ? nil : newValue
                    }
            }
        }
        .padding()
        .onAppear {
            apiKeyInput = config.syncApiKey ?? ""
        }
    }
}

struct AdvancedSettingsTab: View {
    @ObservedObject private var config = AppConfig.shared

    var body: some View {
        Form {
            Section("VAD") {
                HStack {
                    Text("Threshold")
                    Slider(value: $config.vadThreshold, in: 0.3...0.9, step: 0.05)
                    Text(String(format: "%.2f", config.vadThreshold))
                        .monospacedDigit()
                        .frame(width: 40)
                }
                HStack {
                    Text("Silence duration")
                    Slider(value: $config.silenceDuration, in: 0.2...2.0, step: 0.1)
                    Text(String(format: "%.1fs", config.silenceDuration))
                        .monospacedDigit()
                        .frame(width: 40)
                }
            }

            Section("Plugins") {
                TextField("Plugins directory", text: Binding(
                    get: { config.pluginsDir ?? "" },
                    set: { config.pluginsDir = $0.isEmpty ? nil : $0 }
                ))
                .textFieldStyle(.roundedBorder)
            }

            Section("Storage") {
                LabeledContent("Transcripts") {
                    Text(config.transcriptsDir.path)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                LabeledContent("Models") {
                    Text(config.modelsDir.path)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }
        }
        .padding()
    }
}
