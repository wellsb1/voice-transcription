import Foundation
import os

/// Uploads transcript batches to the transcribed.me API.
actor TranscriptSync {
    private let logger = Logger(subsystem: "com.transcribeme.app", category: "sync")
    private let session = URLSession.shared
    private var syncTask: Task<Void, Never>?

    func startPeriodicSync(intervalSeconds: TimeInterval = 1800) {
        Task { await runSync() }
        guard syncTask == nil else { return }
        syncTask = Task {
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: UInt64(intervalSeconds * 1_000_000_000))
                guard !Task.isCancelled else { break }
                await runSync()
            }
        }
    }

    struct SyncCursor: Codable {
        var id: String?
        var timestamp: String?
        var file: String?
        var offset: Int?
    }

    private var cursorFileURL: URL {
        get async {
            await MainActor.run {
                AppConfig.shared.dataDir.appendingPathComponent("sync-cursor.json")
            }
        }
    }

    func runSync() async {
        let apiUrl = await MainActor.run { AppConfig.shared.syncApiUrl }
        let apiKey = await MainActor.run { AppConfig.shared.syncApiKey }
        let transcriptsDir = await MainActor.run { AppConfig.shared.transcriptsDir }

        guard let apiUrl, !apiUrl.isEmpty, let apiKey, !apiKey.isEmpty else {
            return
        }

        logger.info("Sync starting: \(apiUrl)")

        let cursorURL = await cursorFileURL
        var cursor = loadCursor(from: cursorURL)
        let files = getJsonlFiles(in: transcriptsDir)
        guard !files.isEmpty else { return }

        var uploaded = 0
        var skipped = 0

        for file in files {
            let relPath = file.relativePath

            if let cursorFile = cursor.file, relPath < cursorFile {
                continue
            }

            let isResumeFile = cursor.file != nil && relPath == cursor.file

            guard let contents = try? String(contentsOf: file.url, encoding: .utf8) else { continue }
            let lines = contents.components(separatedBy: "\n")

            for (offset, line) in lines.enumerated() {
                let lineNum = offset + 1
                guard !line.trimmingCharacters(in: .whitespaces).isEmpty else { continue }

                if isResumeFile, lineNum <= (cursor.offset ?? 0) {
                    continue
                }

                guard let data = line.data(using: .utf8),
                      let batch = try? JSONDecoder().decode(BatchEnvelope.self, from: data) else {
                    skipped += 1
                    continue
                }

                if cursor.file == nil, let cursorTs = cursor.timestamp,
                   batch.timestamp <= cursorTs {
                    skipped += 1
                    continue
                }

                do {
                    let result = try await uploadBatch(apiUrl: apiUrl, apiKey: apiKey, jsonLine: line)
                    if result.ok {
                        uploaded += 1
                        cursor = SyncCursor(id: batch.id, timestamp: batch.timestamp, file: relPath, offset: lineNum)
                        saveCursor(cursor, to: cursorURL)
                    } else if result.status == 401 {
                        logger.warning("API key rejected (401), logging out")
                        await AuthService.shared.logout()
                        return
                    } else if result.status >= 400 && result.status < 500 {
                        skipped += 1
                        cursor = SyncCursor(id: batch.id, timestamp: batch.timestamp, file: relPath, offset: lineNum)
                        saveCursor(cursor, to: cursorURL)
                    } else {
                        logger.error("Server error (HTTP \(result.status)), stopping sync")
                        return
                    }
                } catch {
                    logger.error("Network error: \(error.localizedDescription), stopping sync")
                    return
                }
            }
        }

        logger.info("Sync done: \(uploaded) uploaded, \(skipped) skipped")
    }

    // MARK: - Private

    private struct UploadResult {
        let ok: Bool
        let status: Int
    }

    private func uploadBatch(apiUrl: String, apiKey: String, jsonLine: String) async throws -> UploadResult {
        guard let url = URL(string: "\(apiUrl)/api/transcripts/batch"),
              let body = jsonLine.data(using: .utf8) else {
            return UploadResult(ok: false, status: 0)
        }

        var request = URLRequest(url: url, timeoutInterval: 10)
        request.httpMethod = "POST"
        request.setValue(apiKey, forHTTPHeaderField: "x-api-key")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = body

        let (_, response) = try await session.data(for: request)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        return UploadResult(ok: status >= 200 && status < 300, status: status)
    }

    private func loadCursor(from url: URL) -> SyncCursor {
        guard let data = try? Data(contentsOf: url),
              let cursor = try? JSONDecoder().decode(SyncCursor.self, from: data) else {
            return SyncCursor()
        }
        return cursor
    }

    private func saveCursor(_ cursor: SyncCursor, to url: URL) {
        let dir = url.deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        if let data = try? JSONEncoder().encode(cursor) {
            try? data.write(to: url)
        }
    }

    private struct JsonlFile {
        let url: URL
        let relativePath: String
    }

    private func getJsonlFiles(in dir: URL) -> [JsonlFile] {
        guard let enumerator = FileManager.default.enumerator(
            at: dir, includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) else { return [] }

        var files: [JsonlFile] = []
        while let url = enumerator.nextObject() as? URL {
            if url.pathExtension == "jsonl" {
                let rel = url.path.replacingOccurrences(of: dir.path + "/", with: "")
                files.append(JsonlFile(url: url, relativePath: rel))
            }
        }
        return files.sorted { $0.relativePath < $1.relativePath }
    }
}
