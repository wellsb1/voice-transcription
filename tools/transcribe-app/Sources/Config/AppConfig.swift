import Foundation
import Security

enum AudioSource: String, Codable, CaseIterable {
    case device = "device"
    case systemTap = "system-tap"
}

@MainActor
class AppConfig: ObservableObject {
    static let shared = AppConfig()

    private let defaults = UserDefaults.standard
    private static let keyPrefix = "transcribe."

    @Published var deviceName: String {
        didSet {
            let slugified = Self.slugify(deviceName)
            if slugified != deviceName {
                deviceName = slugified
                return
            }
            defaults.set(deviceName, forKey: Self.key("deviceName"))
        }
    }
    @Published var audioSource: AudioSource {
        didSet { defaults.set(audioSource.rawValue, forKey: Self.key("audioSource")) }
    }
    @Published var micDevice: String? {
        didSet { defaults.set(micDevice, forKey: Self.key("micDevice")) }
    }
    @Published var excludeApps: [String] {
        didSet { defaults.set(excludeApps, forKey: Self.key("excludeApps")) }
    }
    @Published var diarizationEnabled: Bool {
        didSet { defaults.set(diarizationEnabled, forKey: Self.key("diarizationEnabled")) }
    }
    @Published var autoStart: Bool {
        didSet { defaults.set(autoStart, forKey: Self.key("autoStart")) }
    }
    @Published var minBatchDuration: Double {
        didSet { defaults.set(minBatchDuration, forKey: Self.key("minBatchDuration")) }
    }
    @Published var maxBatchDuration: Double {
        didSet { defaults.set(maxBatchDuration, forKey: Self.key("maxBatchDuration")) }
    }
    @Published var silenceDuration: Double {
        didSet { defaults.set(silenceDuration, forKey: Self.key("silenceDuration")) }
    }
    @Published var vadThreshold: Double {
        didSet { defaults.set(vadThreshold, forKey: Self.key("vadThreshold")) }
    }
    @Published var syncApiUrl: String? {
        didSet { defaults.set(syncApiUrl, forKey: Self.key("syncApiUrl")) }
    }
    @Published var saveAudio: Bool {
        didSet { defaults.set(saveAudio, forKey: Self.key("saveAudio")) }
    }
    @Published var pluginsDir: String? {
        didSet { defaults.set(pluginsDir, forKey: Self.key("pluginsDir")) }
    }
    @Published var modelsReady: Bool {
        didSet { defaults.set(modelsReady, forKey: Self.key("modelsReady")) }
    }
    @Published var diarizationModelsReady: Bool {
        didSet { defaults.set(diarizationModelsReady, forKey: Self.key("diarizationModelsReady")) }
    }
    @Published var userEmail: String? {
        didSet { defaults.set(userEmail, forKey: Self.key("userEmail")) }
    }

    // API key stored in Keychain
    var syncApiKey: String? {
        get { readKeychain(account: "syncApiKey") }
        set {
            if let value = newValue {
                writeKeychain(account: "syncApiKey", value: value)
            } else {
                deleteKeychain(account: "syncApiKey")
            }
            objectWillChange.send()
        }
    }

    private init() {
        let d = UserDefaults.standard
        self.deviceName = Self.slugify(d.string(forKey: Self.key("deviceName")) ?? Host.current().localizedName ?? "mac")
        self.audioSource = AudioSource(rawValue: d.string(forKey: Self.key("audioSource")) ?? "") ?? .systemTap
        self.micDevice = d.string(forKey: Self.key("micDevice"))
        self.excludeApps = d.array(forKey: Self.key("excludeApps")) as? [String] ?? ["com.apple.Music", "com.spotify.client"]
        self.diarizationEnabled = d.bool(forKey: Self.key("diarizationEnabled"))
        self.autoStart = d.bool(forKey: Self.key("autoStart"))
        self.minBatchDuration = d.object(forKey: Self.key("minBatchDuration")) as? Double ?? 30.0
        self.maxBatchDuration = d.object(forKey: Self.key("maxBatchDuration")) as? Double ?? 60.0
        self.silenceDuration = d.object(forKey: Self.key("silenceDuration")) as? Double ?? 0.5
        self.vadThreshold = d.object(forKey: Self.key("vadThreshold")) as? Double ?? 0.6
        self.syncApiUrl = d.string(forKey: Self.key("syncApiUrl")) ?? "https://transcribed.me"
        self.saveAudio = d.bool(forKey: Self.key("saveAudio"))
        self.pluginsDir = d.string(forKey: Self.key("pluginsDir"))
        self.modelsReady = d.bool(forKey: Self.key("modelsReady"))
        self.diarizationModelsReady = d.bool(forKey: Self.key("diarizationModelsReady"))
        self.userEmail = d.string(forKey: Self.key("userEmail"))
    }

    private static func key(_ name: String) -> String { "\(keyPrefix)\(name)" }

    static func slugify(_ name: String) -> String {
        name
            .lowercased()
            .folding(options: .diacriticInsensitive, locale: .current)
            .replacingOccurrences(of: "\u{2019}", with: "")
            .replacingOccurrences(of: "'", with: "")
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
            .joined(separator: "-")
    }

    // MARK: - Paths

    var appSupportDir: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("com.transcribeme.app")
    }

    var transcriptsDir: URL { appSupportDir.appendingPathComponent("Transcripts") }
    var audioDir: URL { appSupportDir.appendingPathComponent("Audio") }
    var modelsDir: URL { appSupportDir.appendingPathComponent("Models") }
    var dataDir: URL { appSupportDir.appendingPathComponent("Data") }

    // MARK: - Keychain

    private let keychainService = "com.transcribeme.app"

    private func readKeychain(account: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: keychainService,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        guard status == errSecSuccess, let data = result as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    private func writeKeychain(account: String, value: String) {
        deleteKeychain(account: account)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: keychainService,
            kSecAttrAccount as String: account,
            kSecValueData as String: value.data(using: .utf8)!,
        ]
        SecItemAdd(query as CFDictionary, nil)
    }

    private func deleteKeychain(account: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: keychainService,
            kSecAttrAccount as String: account,
        ]
        SecItemDelete(query as CFDictionary)
    }
}
