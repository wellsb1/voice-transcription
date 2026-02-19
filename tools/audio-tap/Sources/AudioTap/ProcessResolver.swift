import CoreAudio
import Foundation

/// Resolves bundle IDs (e.g. "com.apple.Music") to Core Audio process AudioObjectIDs.
class ProcessResolver {

    /// Resolve an array of bundle IDs to their AudioObjectIDs.
    /// Only returns IDs for currently running processes.
    func resolve(bundleIDs: [String]) -> [AudioObjectID] {
        guard !bundleIDs.isEmpty else { return [] }

        let allProcesses = getProcessList()
        var result: [AudioObjectID] = []
        var found = Set<String>()

        for processID in allProcesses {
            if let bundleID = getBundleID(for: processID), bundleIDs.contains(bundleID) {
                fputs("audio-tap: Excluding \(bundleID) (objectID=\(processID))\n", stderr)
                result.append(processID)
                found.insert(bundleID)
            }
        }

        for bid in bundleIDs where !found.contains(bid) {
            fputs("audio-tap: Warning: \(bid) not running, cannot exclude\n", stderr)
        }

        return result
    }

    /// Find the default input (microphone) device UID.
    func getDefaultInputDeviceUID() -> String? {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        var deviceID: AudioObjectID = kAudioObjectUnknown
        var size = UInt32(MemoryLayout<AudioObjectID>.size)
        let status = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &address, 0, nil, &size, &deviceID
        )
        guard status == noErr, deviceID != kAudioObjectUnknown else { return nil }

        return getDeviceUID(for: deviceID)
    }

    /// Find a device UID by name substring match.
    func findDeviceUID(matching name: String) -> String? {
        let devices = getDeviceList()
        for deviceID in devices {
            if let deviceName = getDeviceName(for: deviceID),
               deviceName.localizedCaseInsensitiveContains(name) {
                return getDeviceUID(for: deviceID)
            }
        }
        return nil
    }

    // MARK: - Private

    private func getProcessList() -> [AudioObjectID] {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyProcessObjectList,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(
            AudioObjectID(kAudioObjectSystemObject),
            &address, 0, nil, &size
        ) == noErr else { return [] }

        let count = Int(size) / MemoryLayout<AudioObjectID>.size
        var processes = [AudioObjectID](repeating: 0, count: count)
        guard AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &address, 0, nil, &size, &processes
        ) == noErr else { return [] }

        return processes
    }

    private func getDeviceList() -> [AudioObjectID] {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(
            AudioObjectID(kAudioObjectSystemObject),
            &address, 0, nil, &size
        ) == noErr else { return [] }

        let count = Int(size) / MemoryLayout<AudioObjectID>.size
        var devices = [AudioObjectID](repeating: 0, count: count)
        guard AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &address, 0, nil, &size, &devices
        ) == noErr else { return [] }

        return devices
    }

    private func getBundleID(for processID: AudioObjectID) -> String? {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioProcessPropertyBundleID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        var bundleID: CFString = "" as CFString
        var size = UInt32(MemoryLayout<CFString>.size)
        guard AudioObjectGetPropertyData(
            processID, &address, 0, nil, &size, &bundleID
        ) == noErr else { return nil }

        return bundleID as String
    }

    private func getDeviceUID(for deviceID: AudioObjectID) -> String? {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceUID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        var uid: CFString = "" as CFString
        var size = UInt32(MemoryLayout<CFString>.size)
        guard AudioObjectGetPropertyData(
            deviceID, &address, 0, nil, &size, &uid
        ) == noErr else { return nil }

        return uid as String
    }

    private func getDeviceName(for deviceID: AudioObjectID) -> String? {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceNameCFString,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        var name: CFString = "" as CFString
        var size = UInt32(MemoryLayout<CFString>.size)
        guard AudioObjectGetPropertyData(
            deviceID, &address, 0, nil, &size, &name
        ) == noErr else { return nil }

        return name as String
    }
}
