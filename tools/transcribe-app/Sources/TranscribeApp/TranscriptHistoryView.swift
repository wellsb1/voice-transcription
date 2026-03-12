import SwiftUI
import AppKit

struct TranscriptHistoryView: View {
    @ObservedObject var store: TranscriptStore

    private static let dateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "EEEE, MMMM d"
        return f
    }()

    private static let timeFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "h:mm a"
        return f
    }()

    var body: some View {
        VStack(spacing: 0) {
            // Toolbar
            HStack {
                Text("\(store.entries.count) utterances")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                if store.hasMore {
                    Button("Load Older") {
                        store.loadMore()
                    }
                    .font(.caption)
                    .buttonStyle(.borderless)
                }
                Button {
                    store.refresh()
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .buttonStyle(.borderless)
                .help("Refresh")
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(.bar)

            Divider()

            if store.entries.isEmpty {
                Spacer()
                VStack(spacing: 12) {
                    Image(systemName: "text.bubble")
                        .font(.system(size: 40))
                        .foregroundStyle(.tertiary)
                    Text("No transcripts yet")
                        .font(.title3)
                        .foregroundStyle(.secondary)
                    Text("Transcripts will appear here as speech is detected.")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                        .multilineTextAlignment(.center)
                }
                Spacer()
            } else {
                SelectableTranscriptView(
                    attributedString: buildAttributedString()
                )
            }
        }
    }

    private func buildAttributedString() -> NSAttributedString {
        let result = NSMutableAttributedString()
        let cal = Calendar.current
        let today = cal.startOfDay(for: Date())
        let yesterday = cal.date(byAdding: .day, value: -1, to: today)!

        let bodyFont = NSFont.systemFont(ofSize: 13)
        let timeFont = NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)
        let speakerFont = NSFont.systemFont(ofSize: 11, weight: .semibold)
        let dateFont = NSFont.systemFont(ofSize: 11, weight: .semibold)

        let timeColor = NSColor.tertiaryLabelColor
        let dateColor = NSColor.secondaryLabelColor
        let bodyColor = NSColor.labelColor

        let speakerNSColors: [NSColor] = [
            .systemBlue, .systemGreen, .systemOrange, .systemPurple,
            .systemPink, .systemCyan, .systemMint, .systemIndigo,
        ]

        var lastDateKey = ""
        let parStyle = NSMutableParagraphStyle()
        parStyle.lineSpacing = 3
        parStyle.paragraphSpacing = 2

        let datePar = NSMutableParagraphStyle()
        datePar.alignment = .center
        datePar.paragraphSpacingBefore = 12
        datePar.paragraphSpacing = 6

        for entry in store.entries {
            let dateKey = Self.dateFormatter.string(from: entry.timestamp)
            if dateKey != lastDateKey {
                lastDateKey = dateKey
                let entryDay = cal.startOfDay(for: entry.timestamp)
                let label: String
                if entryDay == today {
                    label = "Today"
                } else if entryDay == yesterday {
                    label = "Yesterday"
                } else {
                    label = dateKey
                }
                if result.length > 0 {
                    result.append(NSAttributedString(string: "\n"))
                }
                result.append(NSAttributedString(string: "— \(label) —\n", attributes: [
                    .font: dateFont,
                    .foregroundColor: dateColor,
                    .paragraphStyle: datePar,
                ]))
            }

            // Time
            let timeStr = Self.timeFormatter.string(from: entry.timestamp)
            result.append(NSAttributedString(string: timeStr, attributes: [
                .font: timeFont,
                .foregroundColor: timeColor,
                .paragraphStyle: parStyle,
            ]))

            // Speaker
            let speakerLabel: String
            let speakerColor: NSColor
            if let suffix = entry.speaker.split(separator: "_").last, let num = Int(suffix) {
                speakerLabel = "S\(num)"
                speakerColor = speakerNSColors[num % speakerNSColors.count]
            } else {
                speakerLabel = String(entry.speaker.prefix(2)).uppercased()
                speakerColor = .systemBlue
            }
            result.append(NSAttributedString(string: "  \(speakerLabel)  ", attributes: [
                .font: speakerFont,
                .foregroundColor: speakerColor,
            ]))

            // Text
            result.append(NSAttributedString(string: entry.text + "\n", attributes: [
                .font: bodyFont,
                .foregroundColor: bodyColor,
                .paragraphStyle: parStyle,
            ]))
        }

        return result
    }
}

/// NSTextView wrapper that supports native text selection, Cmd+C, and auto-scrolls to bottom.
struct SelectableTranscriptView: NSViewRepresentable {
    let attributedString: NSAttributedString

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSScrollView()
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = false
        scrollView.autohidesScrollers = true

        let textView = NSTextView()
        textView.isEditable = false
        textView.isSelectable = true
        textView.isRichText = true
        textView.drawsBackground = false
        textView.textContainerInset = NSSize(width: 12, height: 10)
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.autoresizingMask = [.width]
        textView.textContainer?.widthTracksTextView = true

        scrollView.documentView = textView

        textView.textStorage?.setAttributedString(attributedString)

        // Scroll to bottom on first load
        DispatchQueue.main.async {
            let endRange = NSRange(location: textView.string.count, length: 0)
            textView.scrollRangeToVisible(endRange)
        }

        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        guard let textView = scrollView.documentView as? NSTextView else { return }

        let wasAtBottom: Bool
        let clipView = scrollView.contentView
        let contentHeight = textView.frame.height
        let scrollOffset = clipView.bounds.origin.y + clipView.bounds.height
        wasAtBottom = (contentHeight - scrollOffset) < 50

        textView.textStorage?.setAttributedString(attributedString)

        if wasAtBottom {
            DispatchQueue.main.async {
                let endRange = NSRange(location: textView.string.count, length: 0)
                textView.scrollRangeToVisible(endRange)
            }
        }
    }
}
