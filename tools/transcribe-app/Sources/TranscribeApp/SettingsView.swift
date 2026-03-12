import SwiftUI
import AppKit
import FluidAudio

struct GeneralSettingsTab: View {
    @EnvironmentObject var appState: AppState
    @ObservedObject private var config = AppConfig.shared
    @State private var isLoggingIn = false
    @State private var loginError: String?
    @State private var loginTask: Task<Void, Never>?

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
                        Text(config.userEmail ?? "Signed in")
                            .font(.body)
                        Spacer()
                        Button("Log Out") {
                            loginTask?.cancel()
                            Task { await AuthService.shared.logout() }
                        }
                    }
                } else if isLoggingIn {
                    HStack {
                        ProgressView()
                            .controlSize(.small)
                        Text("Waiting for authorization...")
                            .font(.body)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Button("Cancel") {
                            loginTask?.cancel()
                            loginTask = nil
                            isLoggingIn = false
                        }
                    }
                } else {
                    HStack {
                        Text("Sign in to sync transcripts")
                            .foregroundStyle(.secondary)
                        Spacer()
                        Button("Login") {
                            loginError = nil
                            isLoggingIn = true
                            loginTask = Task {
                                do {
                                    try await AuthService.shared.login()
                                    isLoggingIn = false
                                    await appState.pipeline.startSync()
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
                }

                if let error = loginError {
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(.red)
                }
            }

            Section("Audio Source") {
                Picker("Source", selection: $config.audioSource) {
                    Text("Microphone only").tag(AudioSource.device)
                    Text("System audio + Microphone").tag(AudioSource.systemTap)
                }
                .pickerStyle(.radioGroup)
            }

            Section("Startup") {
                Toggle("Launch at login", isOn: $config.launchAtLogin)
                Toggle("Start transcribing on launch", isOn: $config.autoStart)
            }

            Section("Device") {
                TextField("Device name", text: $config.deviceName)
                    .textFieldStyle(.roundedBorder)
                Text("Identifies this device in transcripts")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
    }
}

// MARK: - App Picker for Excluded Apps

struct AppInfo: Identifiable, Hashable {
    let id: String // bundle ID
    let name: String
    let icon: NSImage?
}

struct ExcludedAppsEditor: View {
    @Binding var excludedBundleIDs: [String]
    @State private var showingPicker = false
    @State private var installedApps: [AppInfo] = []

    private var excludedApps: [AppInfo] {
        excludedBundleIDs.map { bundleID in
            if let app = installedApps.first(where: { $0.id == bundleID }) {
                return app
            }
            // Fallback: try to find the app by bundle ID
            if let url = NSWorkspace.shared.urlForApplication(withBundleIdentifier: bundleID) {
                let name = FileManager.default.displayName(atPath: url.path)
                    .replacingOccurrences(of: ".app", with: "")
                let icon = NSWorkspace.shared.icon(forFile: url.path)
                return AppInfo(id: bundleID, name: name, icon: icon)
            }
            return AppInfo(id: bundleID, name: bundleID, icon: nil)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Excluded apps")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Button {
                    loadInstalledApps()
                    showingPicker = true
                } label: {
                    Image(systemName: "plus")
                }
                .buttonStyle(.borderless)
            }

            if excludedApps.isEmpty {
                Text("No apps excluded")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            } else {
                ForEach(excludedApps) { app in
                    HStack(spacing: 6) {
                        if let icon = app.icon {
                            Image(nsImage: icon)
                                .resizable()
                                .frame(width: 16, height: 16)
                        }
                        Text(app.name)
                            .font(.system(size: 12))
                            .lineLimit(1)
                        Spacer()
                        Button {
                            excludedBundleIDs.removeAll { $0 == app.id }
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.borderless)
                    }
                }
            }
        }
        .sheet(isPresented: $showingPicker) {
            AppPickerSheet(
                apps: installedApps,
                excluded: $excludedBundleIDs,
                isPresented: $showingPicker
            )
        }
    }

    private func loadInstalledApps() {
        let appDirs = [
            URL(fileURLWithPath: "/Applications"),
            URL(fileURLWithPath: "/System/Applications"),
            FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Applications"),
        ]

        var seen = Set<String>()
        var apps: [AppInfo] = []

        for dir in appDirs {
            guard let contents = try? FileManager.default.contentsOfDirectory(
                at: dir, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]
            ) else { continue }

            for url in contents where url.pathExtension == "app" {
                guard let bundle = Bundle(url: url),
                      let bundleID = bundle.bundleIdentifier,
                      !seen.contains(bundleID) else { continue }
                seen.insert(bundleID)

                let name = FileManager.default.displayName(atPath: url.path)
                    .replacingOccurrences(of: ".app", with: "")
                let icon = NSWorkspace.shared.icon(forFile: url.path)
                apps.append(AppInfo(id: bundleID, name: name, icon: icon))
            }
        }

        installedApps = apps.sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }
}

struct AppPickerSheet: View {
    let apps: [AppInfo]
    @Binding var excluded: [String]
    @Binding var isPresented: Bool
    @State private var search = ""

    private var filtered: [AppInfo] {
        if search.isEmpty { return apps }
        return apps.filter { $0.name.localizedCaseInsensitiveContains(search) }
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Add App to Exclude List")
                    .font(.headline)
                Spacer()
                Button("Done") { isPresented = false }
                    .keyboardShortcut(.defaultAction)
            }
            .padding()

            TextField("Search apps...", text: $search)
                .textFieldStyle(.roundedBorder)
                .padding(.horizontal)

            List(filtered) { app in
                HStack(spacing: 8) {
                    if let icon = app.icon {
                        Image(nsImage: icon)
                            .resizable()
                            .frame(width: 20, height: 20)
                    }
                    Text(app.name)
                    Spacer()
                    if excluded.contains(app.id) {
                        Image(systemName: "checkmark")
                            .foregroundStyle(.blue)
                    }
                }
                .contentShape(Rectangle())
                .onTapGesture {
                    if excluded.contains(app.id) {
                        excluded.removeAll { $0 == app.id }
                    } else {
                        excluded.append(app.id)
                    }
                }
            }
        }
        .frame(width: 350, height: 400)
    }
}

// MARK: - Models Settings

struct ModelsSettingsTab: View {
    @ObservedObject private var models = ModelManager.shared
    @State private var asrSize: String = ""
    @State private var vadSize: String = ""
    @State private var diarizationSize: String = ""

    var body: some View {
        Form {
            Section("Speech Recognition") {
                HStack {
                    Image(systemName: models.asrReady ? "checkmark.circle.fill" : "circle")
                        .foregroundStyle(models.asrReady ? .green : .secondary)
                    Text(models.asrReady ? "Loaded" : "Not loaded")
                    Spacer()
                    if !asrSize.isEmpty {
                        Text(asrSize)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            Section("Voice Activity Detection") {
                HStack {
                    Image(systemName: models.vadReady ? "checkmark.circle.fill" : "circle")
                        .foregroundStyle(models.vadReady ? .green : .secondary)
                    Text(models.vadReady ? "Loaded" : "Not loaded")
                    Spacer()
                    if !vadSize.isEmpty {
                        Text(vadSize)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            Section("Speaker Identification") {
                HStack {
                    Image(systemName: models.diarizationReady ? "checkmark.circle.fill" : "circle")
                        .foregroundStyle(models.diarizationReady ? .green : .secondary)
                    Text(models.diarizationReady ? "Loaded" : "Not loaded")
                    Spacer()
                    if !diarizationSize.isEmpty {
                        Text(diarizationSize)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            Section {
                if models.isDownloading {
                    HStack {
                        ProgressView()
                            .controlSize(.small)
                        Text(models.downloadStatus)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                } else {
                    Button("Re-download All Models") {
                        Task {
                            await models.forceRedownloadAll()
                            calculateSizes()
                        }
                    }
                }

                if let error = models.downloadError {
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(.red)
                }
            }
        }
        .formStyle(.grouped)
        .onAppear { calculateSizes() }
    }

    private func calculateSizes() {
        let asrDir = AsrModels.defaultCacheDirectory(for: .v3)
        let fluidBase = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("FluidAudio/Models/silero-vad-coreml")
        let diarDir = AppConfig.shared.modelsDir.appendingPathComponent("speaker-diarization-coreml")

        Task.detached {
            let formatter = ByteCountFormatter()
            formatter.countStyle = .file

            let asrBytes = modelDirectorySize(asrDir)
            let vadBytes = modelDirectorySize(fluidBase)
            let diarBytes = modelDirectorySize(diarDir)

            await MainActor.run {
                asrSize = asrBytes > 0 ? formatter.string(fromByteCount: Int64(asrBytes)) : ""
                vadSize = vadBytes > 0 ? formatter.string(fromByteCount: Int64(vadBytes)) : ""
                diarizationSize = diarBytes > 0 ? formatter.string(fromByteCount: Int64(diarBytes)) : ""
            }
        }
    }
}

// MARK: - Advanced Settings

struct AdvancedSettingsTab: View {
    @ObservedObject private var config = AppConfig.shared
    @State private var apiKeyInput = ""
    @State private var showApiOverrides = false
    @State private var inputDevices: [ProcessResolver.AudioDevice] = []

    var body: some View {
        Form {
            if config.audioSource == .systemTap {
                Section("Audio") {
                    Picker("Microphone", selection: Binding(
                        get: { config.micDevice ?? "" },
                        set: { config.micDevice = $0.isEmpty ? nil : $0 }
                    )) {
                        Text("Default").tag("")
                        ForEach(inputDevices, id: \.uid) { device in
                            Text(device.name).tag(device.name)
                        }
                    }

                    ExcludedAppsEditor(excludedBundleIDs: $config.excludeApps)
                }
            }

            Section("Voice Detection") {
                HStack {
                    Text("VAD threshold")
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
                HStack {
                    Text("Min batch")
                    Slider(value: $config.minBatchDuration, in: 10...60, step: 5)
                    Text("\(Int(config.minBatchDuration))s")
                        .monospacedDigit()
                        .frame(width: 30)
                }
                HStack {
                    Text("Max batch")
                    Slider(value: $config.maxBatchDuration, in: 30...120, step: 10)
                    Text("\(Int(config.maxBatchDuration))s")
                        .monospacedDigit()
                        .frame(width: 30)
                }
            }

            Section("Storage") {
                Toggle("Save audio recordings", isOn: $config.saveAudio)
                TextField("Plugins directory", text: Binding(
                    get: { config.pluginsDir ?? "" },
                    set: { config.pluginsDir = $0.isEmpty ? nil : $0 }
                ))
                .textFieldStyle(.roundedBorder)
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

            DisclosureGroup("API Overrides", isExpanded: $showApiOverrides) {
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
        .formStyle(.grouped)
        .onAppear {
            apiKeyInput = config.syncApiKey ?? ""
            inputDevices = ProcessResolver().getInputDevices()
        }
    }
}
