//
//  VideoDetailView.swift
//  WhamApp
//

import SwiftUI
import AVKit
import SceneKit

struct VideoDetailView: View {
    let video: VideoModel
    @StateObject private var analyzer = WhamAnalyzer()

    @State private var isAnalyzedLocal: Bool

    init(video: VideoModel) {
        self.video = video
        _isAnalyzedLocal = State(initialValue: video.isAnalyzed)
    }

    var body: some View {
        VStack {
            if analyzer.isProcessing {
                ProcessingView(analyzer: analyzer)

            } else if isAnalyzedLocal {
                AnalyzedTabView(video: video)

            } else {
                NotAnalyzedView(video: video, analyzer: analyzer) {
                    self.isAnalyzedLocal = true
                }
            }
        }
        .navigationTitle(video.url.lastPathComponent)
        .navigationBarTitleDisplayMode(.inline)
    }
}

// MARK: - Subviews for the 3 States

struct ProcessingView: View {
    @ObservedObject var analyzer: WhamAnalyzer

    var body: some View {
        VStack(spacing: 20) {
            ProgressView(value: analyzer.progress, total: 1.0)
                .progressViewStyle(.linear)
                .tint(.blue)
                .padding(.horizontal, 40)

            Text(analyzer.statusMessage)
                .font(.system(size: 14, weight: .bold, design: .monospaced))
                .foregroundColor(.blue)

            Text("\(Int(analyzer.progress * 100))%")
                .font(.headline)

            if let debugImg = analyzer.debugImage {
                VStack(spacing: 8) {
                    Text("YOLO 2D DEBUGGER (Frame 15):")
                        .font(.system(size: 12, weight: .bold, design: .monospaced))
                        .foregroundColor(.red)

                    Image(uiImage: debugImg)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .background(Color.black)
                        .cornerRadius(12)
                        .padding(.horizontal, 20)

                    Text("If the box/dots don't align perfectly with the human, YOLO is broken.")
                        .font(.caption2)
                        .foregroundColor(.gray)
                }
                .padding(.top, 20)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.primary.opacity(0.05))
    }
}

struct NotAnalyzedView: View {
    var video: VideoModel
    @ObservedObject var analyzer: WhamAnalyzer
    var onComplete: () -> Void

    var body: some View {
        VStack {
            // Media Playback
            VideoPlayer(player: AVPlayer(url: video.url))
                .frame(maxHeight: 400)
                .cornerRadius(12)
                .padding()

            Spacer()

            // The Trigger
            Button(action: {
                Task {
                    await analyzer.analyze(
                        videoURL: video.url,
                        gyroJsonURL: video.gyroJsonURL,
                        outputURL: video.whamOutputURL
                    )

                    if !analyzer.statusMessage.contains("Lỗi") && !analyzer.statusMessage.contains("Failed") {
                        onComplete()
                    }
                }
            }) {
                HStack {
                    Image(systemName: "cpu")
                    Text("BẮT ĐẦU PHÂN TÍCH WHAM")
                        .fontWeight(.bold)
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(video.hasGyro ? Color.blue : Color.gray)
                .foregroundColor(.white)
                .cornerRadius(16)
                .padding()
            }
            .disabled(!video.hasGyro)

            if !video.hasGyro {
                Text("⚠️ Cần dữ liệu AR (Gyro) để phân tích.")
                    .font(.caption)
                    .foregroundColor(.red)
            }

            if analyzer.statusMessage.contains("Lỗi") {
                Text(analyzer.statusMessage)
                    .font(.caption)
                    .foregroundColor(.red)
                    .padding(.bottom)
            }
        }
    }
}


// MARK: - The Tab Router
struct AnalyzedTabView: View {
    let video: VideoModel
    @State private var whamData: [[String: Any]] = []
    @State private var meshReader: SMPLMeshCache.Reader?
    @State private var meshStatus = "Legacy result: no SMPL mesh cache"

    var body: some View {
        TabView {
            // TAB 1: Video with AR overlay
            VideoOverlayView(
                videoURL: video.url,
                whamData: whamData,
                meshReader: meshReader,
                meshStatus: meshStatus
            )
                .tabItem {
                    Image(systemName: "play.tv.fill")
                    Text("Video Overlay")
                }

            // TAB 2: Pure 3D space
            Wham3DView(
                whamData: whamData,
                meshReader: meshReader,
                meshStatus: meshStatus
            )
                .tabItem {
                    Image(systemName: "cube.transparent.fill")
                    Text("3D World")
                }
        }
        .onAppear {
            loadJSONData()
        }
    }

    private func loadJSONData() {
        if let data = try? Data(contentsOf: video.whamOutputURL),
           let json = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
            self.whamData = json
            do {
                let reader = try SMPLMeshCache.Reader(url: video.smplMeshURL)
                guard reader.frameCount == json.count else {
                    meshStatus = "Mesh frame count does not match this result"
                    meshReader = nil
                    return
                }
                meshReader = reader
                meshStatus = ""
            } catch {
                meshReader = nil
                meshStatus = "SMPL mesh unavailable: \(error.localizedDescription)"
            }
        }
    }
}

private struct BodyPresentationPicker<Engine: BodyPresentationControlling>: View {
    @ObservedObject var engine: Engine
    let meshCacheAvailable: Bool
    let onChange: () -> Void

    private var meshAvailable: Bool {
        meshCacheAvailable && engine.meshAvailable
    }

    var body: some View {
        Picker(
            "Body rendering",
            selection: Binding(
                get: { engine.presentationMode },
                set: { mode in
                    engine.setPresentationMode(mode)
                    onChange()
                }
            )
        ) {
            ForEach(BodyPresentationMode.allCases) { mode in
                Text(mode.rawValue)
                    .tag(mode)
                    .disabled(mode == .mesh && !meshAvailable)
            }
        }
        .pickerStyle(.segmented)
        .frame(maxWidth: 260)
    }
}

// MARK: - TAB 1: The AR Overlay (Video + 3D)
struct VideoOverlayView: View {
    let videoURL: URL
    let whamData: [[String: Any]]
    let meshReader: SMPLMeshCache.Reader?
    let meshStatus: String

    @StateObject private var engine = VideoOverlayEngine()
    @State private var player: AVPlayer
    @State private var lastRenderedFrame = -1
    @State private var timestamps: [Double] = []
    @State private var viewportSize: CGSize = .zero

    let timer = Timer.publish(every: 1.0 / 30.0, on: .main, in: .common).autoconnect()

    init(
        videoURL: URL,
        whamData: [[String: Any]],
        meshReader: SMPLMeshCache.Reader?,
        meshStatus: String
    ) {
        self.videoURL = videoURL
        self.whamData = whamData
        self.meshReader = meshReader
        self.meshStatus = meshStatus
        _player = State(initialValue: AVPlayer(url: videoURL))
    }

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .top) {
                VideoPlayer(player: player)

                TransparentSceneView(
                    scene: engine.scene,
                    pointOfView: engine.cameraNode
                )
                .allowsHitTesting(false)

                VStack(spacing: 6) {
                    BodyPresentationPicker(
                        engine: engine,
                        meshCacheAvailable: meshReader != nil,
                        onChange: refreshCurrentFrame
                    )
                    if !projectionMetadataAvailable {
                        Text("Re-analyze this video once to generate a camera-aligned overlay")
                            .font(.caption2)
                            .foregroundStyle(.white)
                            .lineLimit(2)
                    } else if engine.presentationMode == .skeleton && meshReader == nil {
                        Text(meshStatus)
                            .font(.caption2)
                            .foregroundStyle(.white)
                            .lineLimit(2)
                    } else if engine.presentationMode == .skeleton && !engine.meshAvailable {
                        Text("Generate and bundle SMPLFaces.bin to enable mesh rendering")
                            .font(.caption2)
                            .foregroundStyle(.white)
                            .lineLimit(2)
                    }
                }
                .padding(10)
                .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 12))
                .padding(.top, 8)
            }
            .onAppear {
                viewportSize = geometry.size
                synchronizeTimeline()
                selectDefaultPresentation()
                refreshCurrentFrame()
            }
            .onChange(of: geometry.size) {
                viewportSize = geometry.size
                refreshCurrentFrame()
            }
        }
        .ignoresSafeArea()
        .onChange(of: whamData.count) {
            synchronizeTimeline()
            refreshCurrentFrame()
        }
        .onChange(of: meshReader?.frameCount) {
            selectDefaultPresentation()
            refreshCurrentFrame()
        }
        .onReceive(timer) { _ in
            guard whamData.count > 0 else { return }

            let frameIndex = VideoOverlayTimeline.frameIndex(
                at: player.currentTime().seconds,
                timestamps: timestamps
            )
            updateBody(frameIndex: frameIndex)
        }
        .onDisappear {
            player.pause()
        }
    }

    private func selectDefaultPresentation() {
        engine.setPresentationMode(.preferred(
            meshCacheAvailable: meshReader != nil,
            topologyAvailable: engine.meshAvailable
        ))
    }

    private func refreshCurrentFrame() {
        lastRenderedFrame = -1
        let frameIndex = VideoOverlayTimeline.frameIndex(
            at: player.currentTime().seconds,
            timestamps: timestamps
        )
        updateBody(frameIndex: frameIndex)
    }

    private var projectionMetadataAvailable: Bool {
        whamData.contains { VideoOverlayMetadata(dictionary: $0) != nil }
    }

    private func synchronizeTimeline() {
        timestamps = whamData.enumerated().map { index, frame in
            if let number = frame["timestamp_seconds"] as? NSNumber {
                return number.doubleValue
            }
            if let value = frame["timestamp_seconds"] as? Double {
                return value
            }
            return Double(index) / 30
        }
    }

    private func updateBody(frameIndex: Int) {
        guard !whamData.isEmpty, viewportSize.width > 0, viewportSize.height > 0 else {
            return
        }
        let safeIndex = max(0, min(frameIndex, whamData.count - 1))
        guard safeIndex != lastRenderedFrame,
              let keypoints = whamFloatArray(whamData[safeIndex]["keypoints_3d"]),
              let metadata = VideoOverlayMetadata(
                dictionary: whamData[safeIndex]
              ) else {
            engine.clear()
            return
        }

        let vertices = engine.presentationMode == .mesh
            ? try? meshReader?.frame(at: safeIndex)
            : nil
        engine.applyFrameData(
            keypointsWorld: keypoints,
            meshVerticesWorld: vertices ?? nil,
            metadata: metadata,
            viewportSize: viewportSize
        )
        lastRenderedFrame = safeIndex
    }
}

// MARK: - TAB 2: The Pure 3D World
struct Wham3DView: View {
    let whamData: [[String: Any]]
    let meshReader: SMPLMeshCache.Reader?
    let meshStatus: String

    @StateObject private var engine = Skeleton3DEngine()
    @State private var frameIndex = 0

    var body: some View {
        ZStack {
            SceneView(
                scene: engine.scene,
                pointOfView: engine.cameraNode,
                options: [.allowsCameraControl, .autoenablesDefaultLighting]
            )
            .background(Color.black)
            .ignoresSafeArea()

            VStack {
                Text("WHAM 3D \(engine.presentationMode.rawValue.uppercased())")
                    .font(.system(.caption, design: .monospaced))
                    .padding(8)
                    .background(Color.black.opacity(0.7))
                    .foregroundColor(.green)
                    .cornerRadius(8)
                    .padding(.top)

                Spacer()

                VStack {
                    BodyPresentationPicker(
                        engine: engine,
                        meshCacheAvailable: meshReader != nil,
                        onChange: updateBody
                    )

                    if engine.presentationMode == .skeleton && meshReader == nil {
                        Text(meshStatus)
                            .font(.caption2)
                            .foregroundColor(.orange)
                    } else if engine.presentationMode == .skeleton && !engine.meshAvailable {
                        Text("SMPLFaces.bin is not bundled")
                            .font(.caption2)
                            .foregroundColor(.orange)
                    }

                    Text("Frame: \(frameIndex)")
                        .font(.caption)
                        .foregroundColor(.white)

                    Slider(value: Binding(
                        get: { Double(frameIndex) },
                        set: { newVal in
                            frameIndex = Int(newVal)
                            updateBody()
                        }
                    ), in: 0...Double(max(whamData.count - 1, 0)))
                }
                .padding()
                .background(Color.black.opacity(0.6))
            }
        }
        .onAppear {
            engine.setPresentationMode(.preferred(
                meshCacheAvailable: meshReader != nil,
                topologyAvailable: engine.meshAvailable
            ))
            updateBody()
        }
        .onChange(of: meshReader?.frameCount) {
            engine.setPresentationMode(.preferred(
                meshCacheAvailable: meshReader != nil,
                topologyAvailable: engine.meshAvailable
            ))
            updateBody()
        }
    }

    private func updateBody() {
        guard frameIndex < whamData.count else { return }
        guard let keypoints = whamFloatArray(
            whamData[frameIndex]["keypoints_3d"]
        ) else { return }
        let vertices = engine.presentationMode == .mesh
            ? try? meshReader?.frame(at: frameIndex)
            : nil
        engine.applyFrameData(
            keypoints3D: keypoints,
            meshVertices: vertices ?? nil
        )
    }
}

private func whamFloatArray(_ value: Any?) -> [Float]? {
    if let floats = value as? [Float] { return floats }
    if let numbers = value as? [NSNumber] {
        return numbers.map(\.floatValue)
    }
    if let doubles = value as? [Double] {
        return doubles.map(Float.init)
    }
    return nil
}
