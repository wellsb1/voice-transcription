# Package transcribed.me macOS App for Distribution

## Context

The Swift menu bar app works locally but can't be distributed to other users. It builds as a bare executable via SPM, has no .app bundle, no code signing, no installer. Also, the app name is wrong -- it's called "Transcribe" everywhere but should be "transcribed.me".

## Step 1: Rename app from "Transcribe" to "transcribed.me"

The name "Transcribe" appears in ~15 locations that need updating:

| File | What to change |
|------|---------------|
| `Package.swift` | Product/target name `Transcribe` -> `transcribed.me` |
| `Resources/Info.plist` | `CFBundleName`, `CFBundleDisplayName` |
| `Sources/TranscribeApp/TranscribeApp.swift` | `@main struct` name, Settings window title, lock file path |
| `Sources/Pipeline/TranscriptionPipeline.swift` | Logger subsystem label (already `com.transcribeme.app`, keep as-is for bundle ID) |
| All logger subsystem strings | Cosmetic -- these use bundle ID format, fine as-is |

The bundle identifier `com.transcribeme.app` stays the same -- that's correct and used for Application Support path, Keychain, and TCC permissions.

## Step 2: Add `CFBundleExecutable` to Info.plist

Currently missing. Required for macOS to know which binary to launch from the .app bundle.

## Step 3: Create a build script to produce .app bundle

The .app bundle structure needed:

```
transcribed.me.app/
  Contents/
    Info.plist
    MacOS/
      transcribed.me    (compiled binary)
    Resources/
      AppIcon.icns      (when available)
    _CodeSignature/     (created by codesign)
```

Create a `Makefile` that:
1. Runs `swift build -c release`
2. Creates the .app directory structure
3. Copies binary and Info.plist into place
4. Ad-hoc signs with entitlements and hardened runtime (`codesign --sign - --options runtime --entitlements ...`)

For now, ad-hoc signing is sufficient -- no Apple Developer account needed for local testing and small-scale sharing (users right-click > Open on first launch).

## Step 4: Full signing and notarization targets in Makefile

The Makefile will have both ad-hoc and Developer ID paths:
- `make` / `make build` -- build + ad-hoc sign (works now)
- `make sign IDENTITY="Developer ID Application: ..."` -- sign with Developer ID
- `make notarize` -- submit to Apple notarization, wait, staple
- `make dmg` -- create DMG with Applications shortcut for drag-install

All targets are present from the start. Ad-hoc works immediately; Developer ID targets work once you have the certificate.

### Notarization prerequisites

1. Apple Developer Program membership ($99/year)
2. Create "Developer ID Application" certificate at developer.apple.com
3. Store credentials: `xcrun notarytool store-credentials "transcribe-notary" --team-id TEAMID --apple-id email --password app-specific-password`

### Notarization pipeline

```
swift build -c release
  -> create .app bundle
  -> codesign --sign "Developer ID Application: ..." --options runtime --entitlements ... --timestamp
  -> ditto -c -k --keepParent .app .zip   (must use ditto, not zip)
  -> xcrun notarytool submit .zip --keychain-profile ... --wait
  -> xcrun stapler staple .app
  -> spctl -a -vvv -t execute .app        (verify: "accepted source=Notarized Developer ID")
```

## Step 5: Add `com.apple.security.network.client` to Entitlements.plist

Missing entitlement for outbound HTTPS calls. Not strictly required outside sandbox but good practice for hardened runtime.

### Current entitlements

- `com.apple.security.cs.allow-unsigned-executable-memory` -- needed for CoreML
- `com.apple.security.cs.disable-library-validation` -- needed for FluidAudio
- `com.apple.security.cs.allow-jit` -- needed for CoreML
- `com.apple.security.device.audio-input` -- microphone access

### Entitlements NOT needed (outside App Store)

- No sandbox entitlement (not using App Sandbox)
- No special ScreenCaptureKit entitlement (uses TCC at runtime via usage description in Info.plist)
- No special CoreML or Keychain entitlements outside sandbox

## Step 6: Auto-updates (future)

Add Sparkle framework for in-app auto-updates:
- Add `Sparkle` SPM dependency
- Add `SUFeedURL` and `SUPublicEDKey` to Info.plist
- Add "Check for Updates" menu item
- Host appcast.xml + DMGs on GitHub Releases

Requires a proper .app bundle (Step 3) before Sparkle can work.

## Files to modify

| File | Change |
|------|--------|
| `Package.swift` | Rename product/target |
| `Resources/Info.plist` | Fix names, add `CFBundleExecutable` |
| `Resources/Entitlements.plist` | Add network.client entitlement |
| `Sources/TranscribeApp/TranscribeApp.swift` | Rename struct, update lock file path |
| `Makefile` | **New file** -- build, bundle, sign, notarize, dmg targets |

## Verification

1. `make` produces `build/transcribed.me.app/`
2. Double-clicking the .app launches the menu bar icon
3. System Settings > Privacy shows "transcribed.me" (not "Transcribe")
4. Audio capture, sync, and diarization all work from the bundled .app
5. ScreenCaptureKit TCC prompt references the correct app name

## Distribution options summary

| Option | Cost | User experience | Requirements |
|--------|------|-----------------|-------------|
| Ad-hoc signed .app (ZIP) | $0 | Users must right-click > Open | Build script only |
| Notarized DMG | $99/yr | Clean install, no warnings | Developer ID + notarization |
| Homebrew cask | $99/yr | `brew install --cask ...` | Notarized DMG + tap repo |
| Mac App Store | $99/yr | App Store install | Sandbox required -- **not viable** (ScreenCaptureKit) |
