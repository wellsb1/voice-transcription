import Foundation
import os

/// Spawns plugin scripts with JSONL batch data on stdin.
actor PluginRunner {
    private let logger = Logger(subsystem: "com.transcribeme.app", category: "plugins")
    private let pluginsDir: URL

    init(pluginsDir: URL) {
        self.pluginsDir = pluginsDir
    }

    func runStartupHooks() {
        runHooks(suffix: ".startup")
    }

    func runShutdownHooks() {
        runHooks(suffix: ".shutdown")
    }

    func dispatch(jsonLine: String) {
        let fm = FileManager.default
        guard fm.fileExists(atPath: pluginsDir.path) else { return }

        guard let contents = try? fm.contentsOfDirectory(
            at: pluginsDir, includingPropertiesForKeys: [.isExecutableKey],
            options: .skipsHiddenFiles
        ) else { return }

        for pluginURL in contents {
            let name = pluginURL.lastPathComponent
            if name.hasSuffix(".startup") || name.hasSuffix(".shutdown") { continue }

            guard fm.isExecutableFile(atPath: pluginURL.path) else { continue }

            Task.detached { [logger] in
                do {
                    let process = Process()
                    process.executableURL = pluginURL
                    let pipe = Pipe()
                    process.standardInput = pipe
                    process.standardOutput = FileHandle.nullDevice
                    process.standardError = FileHandle.nullDevice

                    try process.run()
                    pipe.fileHandleForWriting.write(Data((jsonLine + "\n").utf8))
                    try pipe.fileHandleForWriting.close()
                    process.waitUntilExit()
                } catch {
                    logger.error("Plugin \(name) failed: \(error.localizedDescription)")
                }
            }
        }
    }

    // MARK: - Private

    private func runHooks(suffix: String) {
        let fm = FileManager.default
        guard fm.fileExists(atPath: pluginsDir.path) else { return }

        guard let contents = try? fm.contentsOfDirectory(
            at: pluginsDir, includingPropertiesForKeys: nil,
            options: .skipsHiddenFiles
        ) else { return }

        for hookURL in contents {
            guard hookURL.lastPathComponent.hasSuffix(suffix),
                  fm.isExecutableFile(atPath: hookURL.path) else { continue }

            Task.detached { [logger] in
                do {
                    let process = Process()
                    process.executableURL = hookURL
                    process.standardOutput = FileHandle.nullDevice
                    process.standardError = FileHandle.nullDevice
                    try process.run()
                    process.waitUntilExit()
                } catch {
                    logger.error("Hook \(hookURL.lastPathComponent) failed: \(error.localizedDescription)")
                }
            }
        }
    }
}
