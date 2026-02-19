import Foundation
import os

/// Logs JSONL batch envelopes to rolling files.
actor TranscriptLogger {
    private let outputDir: URL
    private let deviceName: String
    private let maxLines: Int

    private var currentFile: URL?
    private var currentHandle: FileHandle?
    private var lineCount = 0

    private let logger = Logger(subsystem: "com.transcribeme.app", category: "logger")

    init(outputDir: URL, deviceName: String, maxLines: Int = 100) {
        self.outputDir = outputDir
        self.deviceName = deviceName
        self.maxLines = maxLines
    }

    func write(_ envelope: BatchEnvelope) {
        guard let line = envelope.toJSONLine() else { return }

        if currentHandle == nil || lineCount >= maxLines {
            openNewFile()
        }

        guard let data = (line + "\n").data(using: .utf8) else { return }
        currentHandle?.write(data)
        try? currentHandle?.synchronize()
        lineCount += 1
    }

    func close() {
        try? currentHandle?.close()
        currentHandle = nil
        currentFile = nil
    }

    private func openNewFile() {
        try? currentHandle?.close()

        let now = Date()
        let cal = Calendar.current
        let subdir = outputDir
            .appendingPathComponent(String(format: "%04d", cal.component(.year, from: now)))
            .appendingPathComponent(String(format: "%02d", cal.component(.month, from: now)))
            .appendingPathComponent(String(format: "%02d", cal.component(.day, from: now)))

        try? FileManager.default.createDirectory(at: subdir, withIntermediateDirectories: true)

        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMddHHmmss"
        let filename = "\(formatter.string(from: now))-\(deviceName).jsonl"
        let filepath = subdir.appendingPathComponent(filename)

        FileManager.default.createFile(atPath: filepath.path, contents: nil)
        currentHandle = try? FileHandle(forWritingTo: filepath)
        currentHandle?.seekToEndOfFile()
        currentFile = filepath
        lineCount = 0

        logger.info("Opened transcript file: \(filepath.lastPathComponent)")
    }
}
