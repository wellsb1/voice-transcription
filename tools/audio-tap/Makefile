.PHONY: build install clean

BUILD_DIR = .build/release
BINARY = $(BUILD_DIR)/audio-tap
INSTALL_DIR = ../../bin

build:
	swift build -c release
	codesign -s - --force $(BINARY)
	@echo "Built: $(BINARY)"

install: build
	mkdir -p $(INSTALL_DIR)
	cp $(BINARY) $(INSTALL_DIR)/audio-tap
	codesign -s - --force $(INSTALL_DIR)/audio-tap
	@echo "Installed: $(INSTALL_DIR)/audio-tap"

clean:
	swift package clean
	rm -rf .build
