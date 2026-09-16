//
//  VideoLibraryView.swift
//  WhamApp
//

import SwiftUI
import AVFoundation

// MARK: - Core Data Model
struct VideoModel: Identifiable, Hashable {
    let id: UUID
    let url: URL

    var gyroJsonURL: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("\(url.deletingPathExtension().lastPathComponent).json")
    }

    var whamOutputURL: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("\(url.deletingPathExtension().lastPathComponent)_wham_output.json")
    }

    var smplMeshURL: URL {
        SMPLMeshCache.outputURL(forJSONURL: whamOutputURL)
    }

    var hasGyro: Bool {
        return FileManager.default.fileExists(atPath: gyroJsonURL.path)
    }

    var isAnalyzed: Bool {
        return FileManager.default.fileExists(atPath: whamOutputURL.path)
    }
}

// MARK: - Library View
struct VideoLibraryView: View {
    @State private var videos: [VideoModel] = []
    @Environment(\.dismiss) var dismiss

    var body: some View {
        NavigationView {
            List(videos) { video in
                NavigationLink(destination: VideoDetailView(video: video)) {
                    HStack {
                        VideoThumbnailView(url: video.url)
                            .frame(width: 80, height: 60)
                            .cornerRadius(8)

                        VStack(alignment: .leading, spacing: 4) {
                            Text(video.url.lastPathComponent)
                                .font(.system(size: 14, weight: .bold, design: .monospaced))
                                .lineLimit(1)

                            HStack {
                                if video.hasGyro {
                                    Text("AR: ✅")
                                        .font(.caption2)
                                        .foregroundColor(.green)
                                } else {
                                    Text("AR: ❌")
                                        .font(.caption2)
                                        .foregroundColor(.red)
                                }

                                if video.isAnalyzed {
                                    Text("| WHAM: ✅")
                                        .font(.caption2)
                                        .foregroundColor(.blue)
                                }
                            }
                        }

                        Spacer()
                    }
                }
            }
            .navigationTitle("Thư viện WHAM")
            .toolbar {
                Button("Đóng") { dismiss() }
            }
            .onAppear(perform: loadVideos)
        }
    }

    func loadVideos() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        do {
            let files = try FileManager.default.contentsOfDirectory(at: docs, includingPropertiesForKeys: [.creationDateKey], options: .skipsHiddenFiles)

            let sortedVideoURLs = files.filter { $0.pathExtension == "mp4" }
                .sorted { (url1, url2) -> Bool in
                    let date1 = (try? url1.resourceValues(forKeys: [.creationDateKey]))?.creationDate ?? .distantPast
                    let date2 = (try? url2.resourceValues(forKeys: [.creationDateKey]))?.creationDate ?? .distantPast
                    return date1 > date2
                }

            self.videos = sortedVideoURLs.map { url in
                let idString = url.deletingPathExtension().lastPathComponent
                return VideoModel(id: UUID(uuidString: idString) ?? UUID(), url: url)
            }
        } catch {
            print("❌ Lỗi load thư viện: \(error)")
        }
    }
}

// MARK: - Thumbnail Generator
struct VideoThumbnailView: View {
    let url: URL
    @State private var thumbnail: UIImage? = nil

    var body: some View {
        Group {
            if let img = thumbnail {
                Image(uiImage: img)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } else {
                Color.black
            }
        }
        .onAppear(perform: generateThumbnail)
    }

    func generateThumbnail() {
        let asset = AVURLAsset(url: url)
        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true
        let time = CMTime(seconds: 1, preferredTimescale: 60)

        gen.generateCGImagesAsynchronously(forTimes: [NSValue(time: time)]) { _, cgImage, _, _, _ in
            if let cgImage = cgImage {
                DispatchQueue.main.async {
                    self.thumbnail = UIImage(cgImage: cgImage)
                }
            }
        }
    }
}
