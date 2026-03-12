APP_NAME = transcribed.me
BUNDLE = build/$(APP_NAME).app
BINARY = .build/release/$(APP_NAME)
ENTITLEMENTS = Resources/Entitlements.plist
INFO_PLIST = Resources/Info.plist
IDENTITY ?= Developer ID Application: Wells Burke (RM3832QES4)
NOTARY_PROFILE ?= transcribe-notary
VERSION = $(shell /usr/libexec/PlistBuddy -c "Print CFBundleShortVersionString" $(INFO_PLIST))

.PHONY: all build bundle sign clean dmg notarize run

all: build bundle sign

build:
	swift build -c release

bundle:
	rm -rf "$(BUNDLE)"
	mkdir -p "$(BUNDLE)/Contents/MacOS"
	mkdir -p "$(BUNDLE)/Contents/Resources"
	cp "$(BINARY)" "$(BUNDLE)/Contents/MacOS/$(APP_NAME)"
	cp "$(INFO_PLIST)" "$(BUNDLE)/Contents/"
	cp Resources/AppIcon.icns "$(BUNDLE)/Contents/Resources/"

sign:
ifeq ($(IDENTITY),-)
	codesign --force --sign - \
		--entitlements "$(ENTITLEMENTS)" \
		"$(BUNDLE)"
else
	codesign --force --sign "$(IDENTITY)" \
		--options runtime \
		--timestamp \
		--entitlements "$(ENTITLEMENTS)" \
		"$(BUNDLE)"
endif
	@echo "Signed with identity: $(IDENTITY)"

notarize:
	@test "$(IDENTITY)" != "-" || (echo "Error: Set IDENTITY for notarization" && exit 1)
	ditto -c -k --keepParent "$(BUNDLE)" "build/$(APP_NAME).zip"
	xcrun notarytool submit "build/$(APP_NAME).zip" \
		--keychain-profile "$(NOTARY_PROFILE)" --wait
	xcrun stapler staple "$(BUNDLE)"
	@echo "Notarized and stapled"

dmg:
	rm -f "build/$(APP_NAME)-$(VERSION).dmg"
	mkdir -p build/dmg-staging
	cp -R "$(BUNDLE)" build/dmg-staging/
	ln -s /Applications build/dmg-staging/Applications
	hdiutil create -volname "$(APP_NAME)" \
		-srcfolder build/dmg-staging \
		-ov -format UDZO \
		"build/$(APP_NAME)-$(VERSION).dmg"
	rm -rf build/dmg-staging
	@if [ "$(IDENTITY)" != "-" ]; then \
		codesign --sign "$(IDENTITY)" --timestamp "build/$(APP_NAME)-$(VERSION).dmg"; \
	fi
	@echo "Created build/$(APP_NAME)-$(VERSION).dmg"

run: all
	open "$(BUNDLE)"

clean:
	rm -rf build
	swift package clean
