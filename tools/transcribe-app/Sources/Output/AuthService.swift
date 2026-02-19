import Foundation
import AppKit
import os

actor AuthService {
    static let shared = AuthService()

    private let logger = Logger(subsystem: "com.transcribeme.app", category: "auth")
    private let session = URLSession.shared

    struct DeviceFlowResponse: Codable {
        let deviceCode: String
        let userCode: String
        let verificationUrl: String
        let expiresIn: Int
        let interval: Int
    }

    struct PollResponse: Codable {
        let status: String
        let apiKey: String?
    }

    enum AuthError: Error, LocalizedError {
        case requestFailed(String)
        case expired
        case denied
        case networkError(Error)

        var errorDescription: String? {
            switch self {
            case .requestFailed(let msg): return msg
            case .expired: return "Authorization request expired"
            case .denied: return "Authorization was denied"
            case .networkError(let err): return err.localizedDescription
            }
        }
    }

    func login() async throws {
        let config = await MainActor.run { AppConfig.shared }
        let baseUrl = await MainActor.run { config.syncApiUrl ?? "https://transcribed.me" }
        let deviceName = await MainActor.run { config.deviceName }

        // Step 1: Request device auth flow
        let flow = try await requestDeviceAuth(baseUrl: baseUrl, deviceName: deviceName)
        logger.info("Device auth initiated: \(flow.userCode)")

        // Step 2: Open browser
        if let url = URL(string: flow.verificationUrl) {
            _ = await MainActor.run {
                NSWorkspace.shared.open(url)
            }
        }

        // Step 3: Poll for approval
        let apiKey = try await pollForApproval(
            baseUrl: baseUrl,
            deviceCode: flow.deviceCode,
            interval: flow.interval,
            expiresIn: flow.expiresIn
        )

        // Step 4: Store credentials
        await MainActor.run {
            config.syncApiKey = apiKey
            config.userEmail = deviceName
        }

        logger.info("Device authorized, API key stored")
    }

    func logout() async {
        let config = await MainActor.run { AppConfig.shared }
        await MainActor.run {
            config.syncApiKey = nil
            config.userEmail = nil
        }
        logger.info("Logged out")
    }

    private func requestDeviceAuth(baseUrl: String, deviceName: String) async throws -> DeviceFlowResponse {
        guard let url = URL(string: "\(baseUrl)/api/device-auth/request") else {
            throw AuthError.requestFailed("Invalid URL")
        }

        var request = URLRequest(url: url, timeoutInterval: 10)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(["deviceName": deviceName])

        let (data, response) = try await session.data(for: request)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0

        guard status >= 200 && status < 300 else {
            let body = String(data: data, encoding: .utf8) ?? ""
            throw AuthError.requestFailed("Server error \(status): \(body)")
        }

        return try JSONDecoder().decode(DeviceFlowResponse.self, from: data)
    }

    private func pollForApproval(baseUrl: String, deviceCode: String, interval: Int, expiresIn: Int) async throws -> String {
        guard let url = URL(string: "\(baseUrl)/api/device-auth/poll") else {
            throw AuthError.requestFailed("Invalid URL")
        }

        let deadline = Date().addingTimeInterval(Double(expiresIn))
        let pollInterval = UInt64(interval) * 1_000_000_000

        while Date() < deadline {
            try await Task.sleep(nanoseconds: pollInterval)

            guard !Task.isCancelled else {
                throw CancellationError()
            }

            var request = URLRequest(url: url, timeoutInterval: 10)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try JSONEncoder().encode(["deviceCode": deviceCode])

            do {
                let (data, _) = try await session.data(for: request)
                let poll = try JSONDecoder().decode(PollResponse.self, from: data)

                switch poll.status {
                case "approved":
                    guard let key = poll.apiKey else {
                        throw AuthError.requestFailed("Approved but no API key returned")
                    }
                    return key
                case "denied":
                    throw AuthError.denied
                case "expired":
                    throw AuthError.expired
                case "pending":
                    continue
                default:
                    throw AuthError.requestFailed("Unknown status: \(poll.status)")
                }
            } catch is CancellationError {
                throw CancellationError()
            } catch let error as AuthError {
                throw error
            } catch {
                logger.warning("Poll error (will retry): \(error.localizedDescription)")
                continue
            }
        }

        throw AuthError.expired
    }
}
