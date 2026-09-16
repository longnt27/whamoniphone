//
//  ARWhamManager.swift
//  WhamApp
//
//  Created by admin on 20/3/26.
//

import ARKit
import AVFoundation
import CoreMotion
import SwiftUI

class ARWhamManager: NSObject, ARSessionDelegate, ObservableObject {
    static let recordingFramesPerSecond = 30
    static let targetRecordingResolution = CGSize(width: 1920, height: 1440)

    @Published var isRecording = false
    @Published var recordedSeconds: Int = 0
    @Published var lastThumbnail: UIImage?

    let session = ARSession()
    private let dataQueue = DispatchQueue(label: "com.wham.record", qos: .userInitiated)
    private let motionManager = CMMotionManager()
    private let gyroQueue: OperationQueue = {
        let queue = OperationQueue()
        queue.name = "com.wham.gyro"
        queue.maxConcurrentOperationCount = 1
        queue.qualityOfService = .userInitiated
        return queue
    }()
    private let gyroLock = NSLock()
    private var latestGyro = CMRotationRate()
    private var sensorData: [[String: Any]] = []
    private var currentID: UUID?
    private var timer: Timer?
    private var recordingResolution = ARWhamManager.targetRecordingResolution
    private var lastRecordedFrameTimestamp: TimeInterval?

    // --- AVFoundation Core ---
    private var assetWriter: AVAssetWriter?
    private var assetWriterInput: AVAssetWriterInput?
    private var pixelBufferAdaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var isReadyToRecord = false

    var formattedTime: String {
        let m = recordedSeconds / 60, s = recordedSeconds % 60
        return String(format: "%02d:%02d", m, s)
    }

    override init() {
        super.init()
        session.delegate = self
        resetTracking()
        loadLatestThumbnail()
    }

    func resetTracking() {
        let config = ARWorldTrackingConfiguration()
        config.planeDetection = [.horizontal]
        if let format = Self.preferredRecordingFormat(
            from: ARWorldTrackingConfiguration.supportedVideoFormats
        ) {
            config.videoFormat = format
            recordingResolution = format.imageResolution
        }
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
    }

    func toggleRecording() { isRecording ? stop() : start() }
    func pause() { session.pause() }
    func resume() { resetTracking() }

    private func start() {
        let id = UUID()
        currentID = id
        dataQueue.sync { sensorData.removeAll() }
        lastRecordedFrameTimestamp = nil
        recordedSeconds = 0
        isReadyToRecord = false

        setupWriter(id: id)
        startGyroscope()

        isRecording = true

        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
            self.recordedSeconds += 1
        }
    }

    private func stop() {
        isRecording = false
        motionManager.stopGyroUpdates()
        timer?.invalidate()

        assetWriterInput?.markAsFinished()
        assetWriter?.finishWriting { [weak self] in
            guard let self = self, let id = self.currentID else { return }
            self.dataQueue.async {
                self.saveSensorData(id: id)
                DispatchQueue.main.async { self.loadLatestThumbnail() }
            }
        }
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        autoreleasepool {
            guard isRecording else { return }

            let ts = frame.timestamp
            let minimumInterval = 1.0 / Double(Self.recordingFramesPerSecond)
            if let previous = lastRecordedFrameTimestamp,
               ts - previous < minimumInterval * 0.8 {
                return
            }
            lastRecordedFrameTimestamp = ts

            let mat = frame.camera.transform
            let buffer = frame.capturedImage
            let presentationTime = CMTime(seconds: ts, preferredTimescale: 1000000)
            let gyro = currentGyro()

            dataQueue.async { [weak self] in
                let pos = mat.columns.3
                let point: [String: Any] = [
                    "t": ts,
                    "x": gyro.x,
                    "y": gyro.y,
                    "z": gyro.z,
                    "pos": ["x": pos.x, "y": pos.y, "z": pos.z],
                    "m": [
                        mat.columns.0.x, mat.columns.0.y, mat.columns.0.z, mat.columns.0.w,
                        mat.columns.1.x, mat.columns.1.y, mat.columns.1.z, mat.columns.1.w,
                        mat.columns.2.x, mat.columns.2.y, mat.columns.2.z, mat.columns.2.w,
                        mat.columns.3.x, mat.columns.3.y, mat.columns.3.z, mat.columns.3.w
                    ]
                ]
                self?.sensorData.append(point)
            }

            guard let writer = assetWriter,
                  let input = assetWriterInput,
                  let adaptor = pixelBufferAdaptor else { return }

            if writer.status == .writing {
                if !isReadyToRecord {
                    writer.startSession(atSourceTime: presentationTime)
                    isReadyToRecord = true
                }

                if input.isReadyForMoreMediaData {
                    adaptor.append(buffer, withPresentationTime: presentationTime)
                }
            } else if writer.status == .failed {
                print("Writer Failed: \(writer.error?.localizedDescription ?? "Unknown")")
            }
        }
    }

    private func setupWriter(id: UUID) {
        let url = getURL(id: id, ext: "mp4")
        try? FileManager.default.removeItem(at: url)

        do {
            assetWriter = try AVAssetWriter(outputURL: url, fileType: .mp4)

            let settings: [String: Any] = [
                AVVideoCodecKey: AVVideoCodecType.h264,
                AVVideoWidthKey: Int(recordingResolution.width),
                AVVideoHeightKey: Int(recordingResolution.height),
                AVVideoCompressionPropertiesKey: [
                    AVVideoExpectedSourceFrameRateKey:
                        Self.recordingFramesPerSecond
                ]
            ]

            assetWriterInput = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
            assetWriterInput?.transform = CGAffineTransform(rotationAngle: .pi / 2)
            assetWriterInput?.expectsMediaDataInRealTime = true

            if let input = assetWriterInput {
                pixelBufferAdaptor = AVAssetWriterInputPixelBufferAdaptor(
                    assetWriterInput: input,
                    sourcePixelBufferAttributes: [
                        kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
                    ]
                )

                if assetWriter!.canAdd(input) {
                    assetWriter!.add(input)
                }
            }

            assetWriter?.startWriting()

        } catch {
            print("Setup Writer thất bại: \(error)")
        }
    }

    private func startGyroscope() {
        gyroLock.lock()
        latestGyro = CMRotationRate()
        gyroLock.unlock()
        guard motionManager.isGyroAvailable else { return }
        motionManager.gyroUpdateInterval = 1.0 / 60.0
        motionManager.startGyroUpdates(to: gyroQueue) { [weak self] data, _ in
            guard let self, let data else { return }
            self.gyroLock.lock()
            self.latestGyro = data.rotationRate
            self.gyroLock.unlock()
        }
    }

    private func currentGyro() -> CMRotationRate {
        gyroLock.lock()
        defer { gyroLock.unlock() }
        return latestGyro
    }

    private func saveSensorData(id: UUID) {
        let url = getURL(id: id, ext: "json")
        if let data = try? JSONSerialization.data(
            withJSONObject: sensorData,
            options: []
        ) {
            try? data.write(to: url)
            print(
                "✅ Saved \(sensorData.count) synchronized 30 FPS "
                    + "ARKit + gyro samples"
            )
        }
    }

    func loadLatestThumbnail() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let files = try? FileManager.default.contentsOfDirectory(at: docs, includingPropertiesForKeys: [.creationDateKey])
        if let url = files?.filter({ $0.pathExtension == "mp4" }).sorted(by: {
            (try? $0.resourceValues(forKeys: [.creationDateKey]))?.creationDate ?? .distantPast >
            (try? $1.resourceValues(forKeys: [.creationDateKey]))?.creationDate ?? .distantPast }).first {
            let gen = AVAssetImageGenerator(asset: AVURLAsset(url: url))
            gen.appliesPreferredTrackTransform = true
            gen.generateCGImagesAsynchronously(forTimes: [NSValue(time: .zero)]) { _, cg, _, _, _ in
                if let cg = cg { DispatchQueue.main.async { self.lastThumbnail = UIImage(cgImage: cg) } }
            }
        }
    }

    func getURL(id: UUID, ext: String) -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("\(id.uuidString).\(ext)")
    }

    private static func preferredRecordingFormat(
        from formats: [ARConfiguration.VideoFormat]
    ) -> ARConfiguration.VideoFormat? {
        let thirtyFPSFormats = formats.filter {
            $0.framesPerSecond == recordingFramesPerSecond
        }
        return thirtyFPSFormats.min { first, second in
            resolutionDistance(first.imageResolution)
                < resolutionDistance(second.imageResolution)
        }
    }

    private static func resolutionDistance(_ resolution: CGSize) -> CGFloat {
        abs(resolution.width - targetRecordingResolution.width)
            + abs(resolution.height - targetRecordingResolution.height)
    }
}
