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
            ("General", "gearshape", AnyView(GeneralSettingsTab().environmentObject(appState))),
            ("Models", "arrow.down.circle", AnyView(ModelsSettingsTab())),
            ("Advanced", "slider.horizontal.3", AnyView(AdvancedSettingsTab().environmentObject(appState))),
        ]

        for (label, icon, view) in tabs {
            let hostingController = NSHostingController(rootView: view)
            hostingController.preferredContentSize = NSSize(width: 500, height: 520)
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

@MainActor
final class TranscriptWindowController {
    static let shared = TranscriptWindowController()
    private var window: NSWindow?

    func open(store: TranscriptStore) {
        if let w = window, w.isVisible {
            w.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }

        let view = TranscriptHistoryView(store: store)
            .frame(minWidth: 400, idealWidth: 550, minHeight: 300, idealHeight: 600)
        let hostingController = NSHostingController(rootView: view)
        // Allow the hosting view to resize freely with the window
        hostingController.sizingOptions = [.minSize, .intrinsicContentSize]

        let w = NSWindow(contentViewController: hostingController)
        w.title = "Transcript History"
        w.styleMask = [.titled, .closable, .resizable, .miniaturizable]
        w.setContentSize(NSSize(width: 550, height: 600))
        w.minSize = NSSize(width: 400, height: 300)
        w.center()
        w.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        self.window = w
    }
}

@main
struct TranscribeMeApp: App {
    @StateObject private var appState = AppState()

    var body: some Scene {
        MenuBarExtra {
            if appState.needsMicPermission || appState.needsScreenPermission {
                PermissionView()
                    .environmentObject(appState)
            } else if appState.showSetup {
                SetupView()
                    .environmentObject(appState)
            } else if appState.models.isLoading {
                LoadingView()
            } else {
                MenuBarView()
                    .environmentObject(appState)
            }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: appState.icon)
                    .font(.system(size: 18))
                if appState.isRunning {
                    Text("\(appState.recentWords)")
                        .monospacedDigit()
                }
            }
        }
        .menuBarExtraStyle(.window)
    }

    static let lockFD: CInt = {
        let lockPath = FileManager.default.temporaryDirectory.appendingPathComponent("transcribeme.lock").path
        let fd = open(lockPath, O_CREAT | O_RDWR, 0o644)
        guard fd >= 0 else { return -1 }
        if flock(fd, LOCK_EX | LOCK_NB) != 0 {
            NSLog("Another instance is already running, exiting")
            exit(0)
        }
        return fd
    }()

    init() {
        _ = Self.lockFD
        NSApplication.shared.setActivationPolicy(.accessory)
        AppConfig.applyLaunchAtLogin(AppConfig.shared.launchAtLogin)
    }
}
