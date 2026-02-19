import AVFoundation

/// Resamples audio from one sample rate to another using AVAudioConverter.
class Resampler {
    private let converter: AVAudioConverter
    private let inputFormat: AVAudioFormat
    private let outputFormat: AVAudioFormat
    private let ratio: Double

    init(inputSampleRate: Double, outputSampleRate: Double, channels: UInt32 = 1) throws {
        guard let inputFmt = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: inputSampleRate,
            channels: channels,
            interleaved: true
        ) else {
            throw ResamplerError.formatCreationFailed("input")
        }

        guard let outputFmt = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: outputSampleRate,
            channels: channels,
            interleaved: true
        ) else {
            throw ResamplerError.formatCreationFailed("output")
        }

        guard let conv = AVAudioConverter(from: inputFmt, to: outputFmt) else {
            throw ResamplerError.converterCreationFailed
        }

        self.inputFormat = inputFmt
        self.outputFormat = outputFmt
        self.converter = conv
        self.ratio = outputSampleRate / inputSampleRate
    }

    func convert(_ inputSamples: UnsafeBufferPointer<Float32>) -> [Float32] {
        let frameCount = AVAudioFrameCount(inputSamples.count)
        guard frameCount > 0 else { return [] }

        guard let inputBuffer = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: frameCount) else {
            return []
        }
        inputBuffer.frameLength = frameCount
        memcpy(inputBuffer.floatChannelData![0], inputSamples.baseAddress!, inputSamples.count * MemoryLayout<Float32>.size)

        let outputFrameCount = AVAudioFrameCount(Double(frameCount) * ratio + 1)
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: outputFrameCount) else {
            return []
        }

        var error: NSError?
        let status = converter.convert(to: outputBuffer, error: &error) { _, outStatus in
            outStatus.pointee = .haveData
            return inputBuffer
        }

        guard status != .error, error == nil else { return [] }

        let outCount = Int(outputBuffer.frameLength)
        return Array(UnsafeBufferPointer(start: outputBuffer.floatChannelData![0], count: outCount))
    }
}

enum ResamplerError: Error, CustomStringConvertible {
    case formatCreationFailed(String)
    case converterCreationFailed

    var description: String {
        switch self {
        case .formatCreationFailed(let which): return "Failed to create \(which) audio format"
        case .converterCreationFailed: return "Failed to create AVAudioConverter"
        }
    }
}
