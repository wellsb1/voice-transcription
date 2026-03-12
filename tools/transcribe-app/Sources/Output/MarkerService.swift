import Foundation
import os

/// Creates transcript markers via the transcribed.me API.
actor MarkerService {
    static let shared = MarkerService()

    private let logger = Logger(subsystem: "com.transcribeme.app", category: "marker")
    private let session = URLSession.shared

    struct MarkerResponse: Codable {
        let id: String
        let markerTimestamp: String
    }

    enum MarkerError: Error, LocalizedError {
        case notLoggedIn
        case requestFailed(Int, String)
        case networkError(Error)

        var errorDescription: String? {
            switch self {
            case .notLoggedIn: return "Not logged in"
            case .requestFailed(let status, let msg): return "Server error \(status): \(msg)"
            case .networkError(let err): return err.localizedDescription
            }
        }
    }

    /// Create a marker at the current time, optionally with a note.
    func createMarker(note: String? = nil) async throws {
        let config = await MainActor.run { AppConfig.shared }
        let apiUrl = await MainActor.run { config.syncApiUrl }
        let apiKey = await MainActor.run { config.syncApiKey }
        let device = await MainActor.run { config.deviceName }

        guard let apiUrl, !apiUrl.isEmpty, let apiKey, !apiKey.isEmpty else {
            throw MarkerError.notLoggedIn
        }

        guard let url = URL(string: "\(apiUrl)/api/transcripts/markers") else {
            throw MarkerError.requestFailed(0, "Invalid URL")
        }

        let iso = ISO8601DateFormatter()
        iso.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let timestamp = iso.string(from: Date())

        var body: [String: String] = [
            "markerTimestamp": timestamp,
            "device": device,
        ]
        if let note, !note.isEmpty {
            body["note"] = note
        }

        var request = URLRequest(url: url, timeoutInterval: 10)
        request.httpMethod = "POST"
        request.setValue(apiKey, forHTTPHeaderField: "x-api-key")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(body)

        let (data, response) = try await session.data(for: request)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0

        if status >= 200 && status < 300 {
            logger.info("Marker created at \(timestamp)")
        } else {
            let msg = String(data: data, encoding: .utf8) ?? ""
            throw MarkerError.requestFailed(status, msg)
        }
    }
}
