import Foundation

struct BatchUtterance: Codable {
    let speaker: String
    let confidence: Double
    let start: Double
    let end: Double
    let text: String
}

struct BatchEnvelope: Codable {
    let id: String
    let timestamp: String
    let device: String
    let utterances: [BatchUtterance]

    init(device: String, utterances: [BatchUtterance], timestamp: Date = Date()) {
        self.id = UUID().uuidString
        self.timestamp = ISO8601DateFormatter().string(from: timestamp)
        self.device = device
        self.utterances = utterances
    }

    func toJSONLine() -> String? {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        guard let data = try? encoder.encode(self) else { return nil }
        return String(data: data, encoding: .utf8)
    }
}
