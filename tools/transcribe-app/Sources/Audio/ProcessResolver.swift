import CoreAudio
import Foundation

/// Resolves bundle IDs to Core Audio process AudioObjectIDs and performs device lookups.
class ProcessResolver {

    func resolve(bundleIDs: [String]) -> [AudioObjectID] {
        guard !bundleIDs.isEmpty else { return [] }

        let allProcesses = getProcessList()
        var result: [AudioObjectID] = []
        var found = Set<String>()

        for processID in allProcesses {
            if let bundleID = getBundleID(for: processID), bundleIDs.contains(bundleID) {
                result.append(processID)
                found.insert(bundleID)
            }
        }

        return result
    }

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

    struct AudioDevice {
        let name: String
        let uid: String
    }

    /// Returns all input-capable audio devices (microphones).
    func getInputDevices() -> [AudioDevice] {
        var result: [AudioDevice] = []
        for deviceID in getDeviceList() {
            // Check if device has input channels
            var address = AudioObjectPropertyAddress(
                mSelector: kAudioDevicePropertyStreamConfiguration,
                mScope: kAudioObjectPropertyScopeInput,
                mElement: kAudioObjectPropertyElementMain
            )
            var size: UInt32 = 0
            guard AudioObjectGetPropertyDataSize(deviceID, &address, 0, nil, &size) == noErr,
                  size > 0 else { continue }

            let bufferListPtr = UnsafeMutablePointer<AudioBufferList>.allocate(capacity: 1)
            defer { bufferListPtr.deallocate() }
            guard AudioObjectGetPropertyData(deviceID, &address, 0, nil, &size, bufferListPtr) == noErr else { continue }

            let channelCount = UnsafeMutableAudioBufferListPointer(bufferListPtr).reduce(0) { $0 + Int($1.mNumberChannels) }
            guard channelCount > 0 else { continue }

            if let name = getDeviceName(for: deviceID) {
                let uid = getDeviceUID(for: deviceID) ?? name
                result.append(AudioDevice(name: name, uid: uid))
            }
        }
        return result
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
