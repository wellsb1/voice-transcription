import SwiftUI

struct WaveformIcon: View {
    var size: CGFloat = 28

    // Exact SVG paths from Logo.jsx: viewBox 0 0 32 32
    // M9 16v0  M12 13v6  M16 10v12  M20 13v6  M23 16v0
    // stroke="#ffffff" strokeWidth="2.5" strokeLinecap="round"
    private static let lines: [(x: CGFloat, y1: CGFloat, y2: CGFloat)] = [
        (9,  16, 16),   // dot
        (12, 13, 19),   // short
        (16, 10, 22),   // tall
        (20, 13, 19),   // short
        (23, 16, 16),   // dot
    ]

    var body: some View {
        Canvas { context, canvasSize in
            let scale = canvasSize.width / 32.0

            // Background circle
            let bg = Path(ellipseIn: CGRect(origin: .zero, size: canvasSize))
            context.fill(bg, with: .color(Color(red: 0x11/255, green: 0x18/255, blue: 0x27/255)))

            // Waveform strokes
            for line in Self.lines {
                var path = Path()
                path.move(to: CGPoint(x: line.x * scale, y: line.y1 * scale))
                path.addLine(to: CGPoint(x: line.x * scale, y: line.y2 * scale))
                context.stroke(path, with: .color(.white),
                              style: StrokeStyle(lineWidth: 2.5 * scale, lineCap: .round))
            }
        }
        .frame(width: size, height: size)
    }
}

struct BrandName: View {
    var body: some View {
        HStack(spacing: 0) {
            Text("transcribed")
                .font(.system(size: 16, weight: .bold))
                .foregroundColor(Color(red: 0x11/255, green: 0x18/255, blue: 0x27/255))
                .tracking(-0.3)
            Text(".")
                .font(.system(size: 16, weight: .bold))
                .foregroundColor(Color(red: 0x11/255, green: 0x18/255, blue: 0x27/255))
                .tracking(-0.3)
            Text("me")
                .font(.system(size: 16, weight: .bold))
                .foregroundColor(Color(red: 0x6b/255, green: 0x72/255, blue: 0x80/255))
                .tracking(-0.3)
        }
    }
}

struct MenuRowLabel: View {
    let text: String
    init(_ text: String) { self.text = text }
    @State private var isHovered = false

    var body: some View {
        Text(text)
            .font(.system(size: 13))
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.vertical, 4)
            .padding(.horizontal, 8)
            .background(isHovered ? Color.primary.opacity(0.1) : Color.clear)
            .cornerRadius(4)
            .onHover { isHovered = $0 }
    }
}

struct MenuRow: View {
    let text: String
    let action: () -> Void
    init(_ text: String, action: @escaping () -> Void) {
        self.text = text
        self.action = action
    }

    var body: some View {
        Button(action: action) {
            MenuRowLabel(text)
        }
        .buttonStyle(.plain)
    }
}

struct MenuBarView: View {
    @EnvironmentObject var appState: AppState
    @State private var isLoggingIn = false
    @State private var loginTask: Task<Void, Never>?
    @State private var loginError: String?

    private var isLoggedIn: Bool {
        appState.config.syncApiKey != nil
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            HStack(spacing: 10) {
                WaveformIcon(size: 28)
                BrandName()
                Spacer()
                Text(appState.isRunning ? "Active" : "Idle")
                    .font(.caption)
                    .foregroundStyle(appState.isRunning ? .green : .secondary)
            }

            // Login prompt
            if !isLoggedIn && !isLoggingIn {
                Divider()

                Button {
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
                } label: {
                    HStack {
                        Image(systemName: "person.crop.circle.badge.plus")
                        Text("Login to transcribed.me")
                        Spacer()
                    }
                    .font(.system(size: 13, weight: .medium))
                    .padding(.vertical, 6)
                    .padding(.horizontal, 8)
                    .background(Color.accentColor.opacity(0.1))
                    .cornerRadius(6)
                }
                .buttonStyle(.plain)

                if let error = loginError {
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .padding(.horizontal, 8)
                }
            }

            if isLoggingIn {
                Divider()

                HStack {
                    ProgressView()
                        .controlSize(.small)
                    Text("Waiting for authorization...")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("Cancel") {
                        loginTask?.cancel()
                        loginTask = nil
                        isLoggingIn = false
                    }
                    .font(.caption)
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 8)
            }

            Divider()

            // Stats
            if appState.isRunning {
                HStack {
                    Label("\(appState.recentWords)", systemImage: "text.word.spacing")
                        .font(.caption)
                    Text("words (5m)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Label("\(appState.totalWords)", systemImage: "sum")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            // Audio source indicator
            HStack {
                Image(systemName: appState.config.audioSource == .systemTap ? "speaker.wave.2" : "mic")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(appState.config.audioSource == .systemTap ? "System + Mic" : "Microphone only")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if isLoggedIn {
                    Spacer()
                    Image(systemName: "checkmark.icloud")
                        .font(.caption)
                        .foregroundStyle(.green)
                }
            }

            Divider()

            // Controls
            MenuRow(appState.isRunning ? "Stop Transcribing" : "Start Transcribing") {
                Task { await appState.toggleTranscription() }
            }

            Divider()

            MenuRow("View Transcripts Online") {
                if let url = URL(string: AppConfig.shared.syncApiUrl ?? "https://transcribed.me") {
                    NSWorkspace.shared.open(url)
                }
            }

            MenuRow("Show Transcripts in Finder") {
                NSWorkspace.shared.open(AppConfig.shared.transcriptsDir)
            }

            MenuRow("Settings...") {
                SettingsWindowController.shared.open(appState: appState)
            }

            MenuRow("Quit Transcribed") {
                Task {
                    await appState.stopTranscription()
                    NSApplication.shared.terminate(nil)
                }
            }
        }
        .padding()
        .frame(width: 280)
        .background(.ultraThinMaterial)
    }
}
