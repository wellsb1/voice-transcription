import AVFoundation
import CoreGraphics

/// Checks and requests macOS permissions needed by the app.
enum PermissionManager {

    // MARK: - Microphone

    static func micStatus() -> AVAuthorizationStatus {
        AVCaptureDevice.authorizationStatus(for: .audio)
    }

    static func requestMic() async -> Bool {
        await AVCaptureDevice.requestAccess(for: .audio)
    }

    // MARK: - Screen & System Audio Recording

    static func hasScreenRecording() -> Bool {
        CGPreflightScreenCaptureAccess()
    }

    /// Opens System Settings to the Screen Recording pane.
    /// Returns true only if permission was already granted at call time.
    @discardableResult
    static func requestScreenRecording() -> Bool {
        CGRequestScreenCaptureAccess()
    }
}
