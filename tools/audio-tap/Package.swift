// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "audio-tap",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "audio-tap",
            path: "Sources/AudioTap",
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-sectcreate",
                    "-Xlinker", "__TEXT",
                    "-Xlinker", "__info_plist",
                    "-Xlinker", "Info.plist",
                ])
            ]
        ),
    ]
)
