import SwiftUI
import AppKit

@MainActor
final class SettingsWindowController {
    static let shared = SettingsWindowController()
    private var window: NSWindow?

    func open(appState: AppState) {
        if let w = window, w.isVisible {
            w.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }

        let tabVC = NSTabViewController()
        tabVC.tabStyle = .toolbar

        let tabs: [(String, String, AnyView)] = [
            ("Audio", "waveform", AnyView(AudioSettingsTab().environmentObject(appState))),
            ("Transcription", "text.bubble", AnyView(TranscriptionSettingsTab().environmentObject(appState))),
            ("Sync", "arrow.triangle.2.circlepath", AnyView(SyncSettingsTab().environmentObject(appState))),
            ("Advanced", "gearshape.2", AnyView(AdvancedSettingsTab().environmentObject(appState))),
        ]

        for (label, icon, view) in tabs {
            let hostingController = NSHostingController(rootView: view)
            hostingController.preferredContentSize = NSSize(width: 480, height: 300)
            let item = NSTabViewItem(viewController: hostingController)
            item.label = label
            item.image = NSImage(systemSymbolName: icon, accessibilityDescription: label)
            tabVC.addTabViewItem(item)
        }

        let w = NSWindow(contentViewController: tabVC)
        w.title = "transcribed.me Settings"
        w.styleMask = [.titled, .closable]
        w.center()
        w.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        self.window = w
    }
}

@main
struct TranscribeApp: App {
    @StateObject private var appState = AppState()

    var body: some Scene {
        MenuBarExtra {
            if appState.showSetup {
                SetupView()
                    .environmentObject(appState)
            } else {
                MenuBarView()
                    .environmentObject(appState)
            }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: appState.icon)
                if appState.isRunning {
                    Text("\(appState.recentWords)")
                        .monospacedDigit()
                }
            }
        }
        .menuBarExtraStyle(.window)
    }

    init() {
        NSApplication.shared.setActivationPolicy(.accessory)
    }
}
