import Foundation
import os

/// Loads transcript batches from JSONL files on disk and receives live updates.
@MainActor
class TranscriptStore: ObservableObject {
    struct Entry: Identifiable {
        let id: String
        let timestamp: Date
        let speaker: String
        let text: String
        let confidence: Double
    }

    @Published var entries: [Entry] = []
    @Published var hasMore = true

    private let transcriptsDir: URL
    private var allFiles: [URL] = []
    private var nextFileIndex: Int = 0
    private var loadedFileCount = 0
    private let filesPerPage = 5
    private let logger = Logger(subsystem: "com.transcribeme.app", category: "store")
    private let isoFormatter: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()
    private let isoFormatterNoFrac: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime]
        return f
    }()

    init(transcriptsDir: URL) {
        self.transcriptsDir = transcriptsDir
        scanFiles()
        loadMore()
    }

    /// Scan transcript directory for all JSONL files, sorted newest first.
    private func scanFiles() {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(
            at: transcriptsDir,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            hasMore = false
            return
        }

        var files: [URL] = []
        for case let url as URL in enumerator {
            if url.pathExtension == "jsonl" {
                files.append(url)
            }
        }

        // Sort descending by filename (yyyyMMddHHmmss prefix = chronological order)
        allFiles = files.sorted { $0.lastPathComponent > $1.lastPathComponent }
        nextFileIndex = 0
        hasMore = !allFiles.isEmpty
    }

    /// Load the next page of older transcript files.
    func loadMore() {
        guard nextFileIndex < allFiles.count else {
            hasMore = false
            return
        }

        let endIndex = min(nextFileIndex + filesPerPage, allFiles.count)
        var newEntries: [Entry] = []

        for i in nextFileIndex..<endIndex {
            let entries = parseFile(allFiles[i])
            newEntries.append(contentsOf: entries)
        }

        nextFileIndex = endIndex
        hasMore = nextFileIndex < allFiles.count

        // Files are newest-first, so new entries go to the front (older content)
        // But within each file, entries are chronological. We want the final list
        // ordered oldest→newest so the newest is at the bottom.
        // Since we load newest files first, prepend older batches.
        newEntries.sort { $0.timestamp < $1.timestamp }
        entries.insert(contentsOf: newEntries, at: 0)
    }

    /// Add a live batch from the pipeline.
    func addLive(_ envelope: BatchEnvelope) {
        let ts = isoFormatter.date(from: envelope.timestamp)
            ?? isoFormatterNoFrac.date(from: envelope.timestamp)
            ?? Date()

        for utterance in envelope.utterances {
            let entry = Entry(
                id: "\(envelope.id)-\(utterance.speaker)-\(utterance.start)",
                timestamp: ts,
                speaker: utterance.speaker,
                text: utterance.text,
                confidence: utterance.confidence
            )
            entries.append(entry)
        }
    }

    /// Reload all files from disk.
    func refresh() {
        entries.removeAll()
        scanFiles()
        loadMore()
    }

    private func parseFile(_ url: URL) -> [Entry] {
        guard let data = try? String(contentsOf: url, encoding: .utf8) else { return [] }
        let decoder = JSONDecoder()
        var results: [Entry] = []

        for line in data.components(separatedBy: .newlines) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty, let lineData = trimmed.data(using: .utf8) else { continue }
            guard let envelope = try? decoder.decode(BatchEnvelope.self, from: lineData) else { continue }

            let ts = isoFormatter.date(from: envelope.timestamp)
                ?? isoFormatterNoFrac.date(from: envelope.timestamp)
                ?? Date()

            for utterance in envelope.utterances {
                results.append(Entry(
                    id: "\(envelope.id)-\(utterance.speaker)-\(utterance.start)",
                    timestamp: ts,
                    speaker: utterance.speaker,
                    text: utterance.text,
                    confidence: utterance.confidence
                ))
            }
        }
        return results
    }
}
