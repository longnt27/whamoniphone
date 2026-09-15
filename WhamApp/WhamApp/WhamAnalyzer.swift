//
//  WhamAnalyzer.swift
//  WhamApp
//

import Foundation
import AVFoundation
import CoreML
import UIKit

private struct GyroSample: Codable {
    let t: Double
    let x: Double
    let y: Double
    let z: Double
}

@MainActor
class WhamAnalyzer: ObservableObject {
    @Published var progress: Double = 0
    @Published var isProcessing = false
    @Published var statusMessage: String = ""
    @Published var debugImage: UIImage?

    nonisolated private let ciContext = CIContext()

    init() {}

    func analyze(videoURL: URL, gyroJsonURL: URL, outputURL: URL) async {
        isProcessing = true
        progress = 0
        statusMessage = "Đang chuẩn bị pipeline đã kiểm định..."

        do {
            try await Task.detached(priority: .userInitiated) { [self] in
                let gyroSamples = (try? self.loadGyroData(from: gyroJsonURL)) ?? []
                let asset = AVURLAsset(url: videoURL)
                let reader = try AVAssetReader(asset: asset)
                let tracks = try await asset.loadTracks(withMediaType: .video)
                guard let track = tracks.first else { throw AnalyzerError.noVideoTrack }

                let transform = try await track.load(.preferredTransform)
                let duration = try await asset.load(.duration).seconds
                let nominalFPS = max(Double(try await track.load(.nominalFrameRate)), 1)
                let estimatedFrames = max(Int(duration * nominalFPS), 1)
                let readerOutput = AVAssetReaderTrackOutput(track: track, outputSettings: [
                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
                ])
                guard reader.canAdd(readerOutput) else { throw AnalyzerError.readerSetup }
                reader.add(readerOutput)
                guard reader.startReading() else { throw reader.error ?? AnalyzerError.readerSetup }

                var observations: [MobileWhamFrameObservation] = []
                var frameIndex = 0
                var firstPresentationTime: Double?

                await MainActor.run {
                    self.statusMessage = "Đang nạp YOLO26m + HMR2-S + adapter..."
                }
                let visionConfiguration = MLModelConfiguration()
                visionConfiguration.computeUnits = .all
                let preprocessor = MobileWhamPreprocessor(ciContext: self.ciContext)
                var poseDetector: PoseDetector? = try PoseDetector(
                    backend: .yolo26m,
                    configuration: visionConfiguration
                )
                var hmr2s: MLModel? = try MobileWhamModelCatalog.load(
                    MobileWhamModelCatalog.hmrFrontend,
                    configuration: visionConfiguration
                )
                var tokenAdapter: MLModel? = try MobileWhamModelCatalog.load(
                    MobileWhamModelCatalog.tokenAdapter,
                    configuration: visionConfiguration
                )
                var smplInitializer: MLModel? = try MobileWhamModelCatalog.load(
                    MobileWhamModelCatalog.smplInitializer,
                    configuration: visionConfiguration
                )

                await MainActor.run {
                    self.statusMessage = "Đang trích xuất keypoint, token và SMPL (1/2)..."
                }

                while reader.status == .reading {
                    guard let sampleBuffer = readerOutput.copyNextSampleBuffer() else { break }
                    let presentationTime = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
                    if firstPresentationTime == nil { firstPresentationTime = presentationTime }
                    let relativeTime = presentationTime - (firstPresentationTime ?? presentationTime)

                    let observation = try autoreleasepool { () throws -> MobileWhamFrameObservation in
                        guard let rawPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
                            return try MobileWhamPreprocessor.missingFrame(
                                videoTime: relativeTime
                            )
                        }

                        let orientation = self.getVideoOrientation(from: transform)
                        let orientedImage = CIImage(cvPixelBuffer: rawPixelBuffer).oriented(orientation)
                        let width = Int(orientedImage.extent.width)
                        let height = Int(orientedImage.extent.height)
                        guard let uprightPixelBuffer = preprocessor.makePixelBuffer(
                            width: width,
                            height: height
                        ) else {
                            return try MobileWhamPreprocessor.missingFrame(
                                videoTime: relativeTime
                            )
                        }
                        self.ciContext.render(orientedImage, to: uprightPixelBuffer)

                        guard let detection = poseDetector?.detect(pixelBuffer: uprightPixelBuffer).first else {
                            return try MobileWhamPreprocessor.missingFrame(
                                videoTime: relativeTime
                            )
                        }

                        if frameIndex == 15 {
                            let debugImage = self.drawYOLODebug(
                                pixelBuffer: uprightPixelBuffer,
                                detection: detection
                            )
                            Task { @MainActor in self.debugImage = debugImage }
                        }

                        let visual = try preprocessor.visualObservation(
                            detection: detection,
                            source: uprightPixelBuffer
                        )

                        guard let crop = visual.crop,
                              let hmr2s,
                              let tokenAdapter else {
                            return try MobileWhamPreprocessor.frame(
                                visual: visual,
                                frontend: nil,
                                adaptedFeature: nil,
                                videoTime: relativeTime
                            )
                        }
                        let hmr = try MobileWhamArrays.predict(hmr2s, [
                            "image_input": MLFeatureValue(pixelBuffer: crop)
                        ])
                        let frontend = MobileWhamFrontendOutput(
                            token: try MobileWhamArrays.required(hmr, "image_token"),
                            pose: try MobileWhamArrays.required(hmr, "pose_6d"),
                            betas: try MobileWhamArrays.required(hmr, "betas")
                        )
                        let adapted = try MobileWhamArrays.predict(tokenAdapter, [
                            "hmr2s_token": MLFeatureValue(
                                multiArray: frontend.token
                            )
                        ])

                        return try MobileWhamPreprocessor.frame(
                            visual: visual,
                            frontend: frontend,
                            adaptedFeature: try MobileWhamArrays.required(
                                adapted,
                                "wham_image_token"
                            ),
                            videoTime: relativeTime
                        )
                    }
                    observations.append(observation)

                    frameIndex += 1
                    let extractionProgress = min(Double(frameIndex) / Double(estimatedFrames), 1) * 0.5
                    await MainActor.run { self.progress = extractionProgress }
                }

                if reader.status == .failed { throw reader.error ?? AnalyzerError.readerFailed }
                guard observations.count >= 2 else { throw AnalyzerError.insufficientFrames }
                guard let firstPose = observations[0].hmrPose,
                      let firstBetas = observations[0].hmrBetas,
                      let smplInitializerModel = smplInitializer else {
                    throw AnalyzerError.initializationFailed
                }

                let smpl = try MobileWhamArrays.predict(smplInitializerModel, [
                    "pose_6d": MLFeatureValue(multiArray: firstPose),
                    "betas": MLFeatureValue(multiArray: firstBetas)
                ])
                let initial3D = try MobileWhamArrays.required(smpl, "init_kp3d")

                // Release the vision models before allocating the recurrent models.
                poseDetector = nil
                hmr2s = nil
                tokenAdapter = nil
                smplInitializer = nil

                await MainActor.run {
                    self.statusMessage = "Đang nạp WHAM init/step/world..."
                }
                let whamConfiguration = MLModelConfiguration()
                whamConfiguration.computeUnits = .all
                let initializer = try MobileWhamModelCatalog.load(
                    MobileWhamModelCatalog.whamInitializer,
                    configuration: whamConfiguration
                )
                let recurrentStep = try MobileWhamModelCatalog.load(
                    MobileWhamModelCatalog.whamStep,
                    configuration: whamConfiguration
                )
                let worldStep = try MobileWhamModelCatalog.load(
                    MobileWhamModelCatalog.worldStep,
                    configuration: whamConfiguration
                )
                let core = MobileWhamCore(
                    initializer: initializer,
                    recurrentStep: recurrentStep,
                    worldStep: worldStep
                )
                var coreState = try core.initialize(
                    firstPose: firstPose,
                    initialJoints3D: initial3D,
                    firstKeypoints: observations[0].keypoints
                )
                var results: [[String: Any]] = []

                await MainActor.run {
                    self.statusMessage = "Đang chạy WHAM + world + smoothing (2/2)..."
                }
                for index in 1..<observations.count {
                    let observation = observations[index]
                    let previousTime = observations[index - 1].videoTime
                    let cameraAngularVelocity = try self.cameraAngularVelocity(
                        gyroSamples: gyroSamples,
                        at: observation.videoTime,
                        frameDuration: max(observation.videoTime - previousTime, 1 / nominalFPS)
                    )

                    let step = try core.step(
                        observation: observation,
                        cameraAngularVelocity: cameraAngularVelocity,
                        state: &coreState
                    )

                    results.append([
                        "frame": index,
                        "timestamp_seconds": observation.videoTime,
                        "pose_6d": MobileWhamArrays.floats(step.pose),
                        "shape": MobileWhamArrays.floats(step.shape),
                        "root_orient": MobileWhamArrays.floats(step.refinedRoot),
                        "translation_world": MobileWhamArrays.floats(
                            step.refinedTranslation
                        ),
                        "keypoints_3d": MobileWhamArrays.floats(
                            step.jointsWorld,
                            limit: 51
                        ),
                        "keypoints_3d_root_relative": MobileWhamArrays.floats(
                            step.keypointsRootRelative
                        ),
                        "image_feature_valid": observation.imageFeatureValid[0].floatValue
                    ])

                    let inferenceProgress = 0.5 + Double(index) / Double(observations.count - 1) * 0.5
                    await MainActor.run { self.progress = inferenceProgress }
                }

                // Keep one output per source frame for the existing viewer. Frame
                // zero is a documented warm-start duplicate of frame one's result.
                if var warmStart = results.first {
                    warmStart["frame"] = 0
                    warmStart["timestamp_seconds"] = observations[0].videoTime
                    warmStart["warm_start_duplicate"] = true
                    warmStart["pipeline"] = "YOLO26m-pose -> HMR2-S -> selected token adapter -> WHAM init/step/world -> light causal smoothing"
                    warmStart["initializer"] = "HMR2-S SMPL and 3D joints -> WHAM_I"
                    warmStart["keypoints_3d_space"] = "world"
                    warmStart["smoothing_pose_alpha"] = TemporalOutputSmoother.selectedPoseAlpha
                    warmStart["smoothing_shape_alpha"] = TemporalOutputSmoother.selectedShapeAlpha
                    warmStart["camera_motion"] = gyroSamples.isEmpty ? "none" : "device_gyro_approximation"
                    results.insert(warmStart, at: 0)
                }

                let finalData = try JSONSerialization.data(
                    withJSONObject: results,
                    options: [.prettyPrinted, .sortedKeys]
                )
                try finalData.write(to: outputURL, options: .atomic)
            }.value

            statusMessage = "✅ Hoàn tất — pipeline YOLO26m/HMR2-S/adapter/WHAM/smoothing"
            progress = 1
        } catch {
            print("❌ Analysis Failed: \(error)")
            statusMessage = "❌ Lỗi: \(error.localizedDescription)"
        }
        isProcessing = false
    }

    // MARK: - Camera motion

    nonisolated private func cameraAngularVelocity(
        gyroSamples: [GyroSample],
        at videoTime: Double,
        frameDuration: Double
    ) throws -> MLMultiArray {
        let output = try MobileWhamArrays.zeros([1, 1, 6], dataType: .float16)
        guard let first = gyroSamples.first, !gyroSamples.isEmpty else { return output }

        let target = first.t + videoTime
        var lower = 0
        var upper = gyroSamples.count
        while lower < upper {
            let middle = (lower + upper) / 2
            if gyroSamples[middle].t < target { lower = middle + 1 } else { upper = middle }
        }
        let right = min(lower, gyroSamples.count - 1)
        let left = max(right - 1, 0)
        let sample = abs(gyroSamples[left].t - target) <= abs(gyroSamples[right].t - target)
            ? gyroSamples[left]
            : gyroSamples[right]

        // Convert the measured device angular rate to WHAM's normalized 6D
        // relative-rotation representation. Sign is inverted to approximate the
        // world-to-camera delta used by the official DPVO preprocessing.
        let dt = max(frameDuration, 1e-4)
        let rx = -sample.x * dt
        let ry = -sample.y * dt
        let rz = -sample.z * dt
        let theta = sqrt(rx * rx + ry * ry + rz * rz)
        let a = theta < 1e-8 ? 1.0 : sin(theta) / theta
        let b = theta < 1e-8 ? 0.5 : (1 - cos(theta)) / (theta * theta)
        let r00 = 1 - b * (ry * ry + rz * rz)
        let r01 = b * rx * ry - a * rz
        let r02 = b * rx * rz + a * ry
        let r10 = b * ry * rx + a * rz
        let r11 = 1 - b * (rx * rx + rz * rz)
        let r12 = b * ry * rz - a * rx
        let sixD = [r00 - 1, r01, r02, r10, r11 - 1, r12]
        for index in 0..<6 { output[index] = NSNumber(value: Float(sixD[index] / dt)) }
        return output
    }

    // MARK: - Image and data helpers

    nonisolated private func getVideoOrientation(from transform: CGAffineTransform) -> CGImagePropertyOrientation {
        if transform.a == 0 && transform.b == 1 && transform.c == -1 && transform.d == 0 { return .right }
        if transform.a == 0 && transform.b == -1 && transform.c == 1 && transform.d == 0 { return .left }
        if transform.a == 1 && transform.b == 0 && transform.c == 0 && transform.d == 1 { return .up }
        if transform.a == -1 && transform.b == 0 && transform.c == 0 && transform.d == -1 { return .down }
        return .right
    }

    nonisolated private func loadGyroData(from url: URL) throws -> [GyroSample] {
        try JSONDecoder().decode([GyroSample].self, from: Data(contentsOf: url)).sorted { $0.t < $1.t }
    }

    // MARK: - YOLO visual debugger

    nonisolated private func drawYOLODebug(
        pixelBuffer: CVPixelBuffer,
        detection: PoseDetector.Detection
    ) -> UIImage? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else { return nil }
        let width = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let height = CGFloat(CVPixelBufferGetHeight(pixelBuffer))

        UIGraphicsBeginImageContextWithOptions(CGSize(width: width, height: height), false, 1)
        guard let context = UIGraphicsGetCurrentContext() else { return nil }
        UIImage(cgImage: cgImage).draw(in: CGRect(x: 0, y: 0, width: width, height: height))
        let realBox = CGRect(
            x: detection.box.minX * width,
            y: detection.box.minY * height,
            width: detection.box.width * width,
            height: detection.box.height * height
        )
        context.setStrokeColor(UIColor.red.cgColor)
        context.setLineWidth(4)
        context.stroke(realBox)
        context.setFillColor(UIColor.green.cgColor)
        for point in detection.keypoints {
            let x = point.x * width
            let y = point.y * height
            context.fillEllipse(in: CGRect(x: x - 6, y: y - 6, width: 12, height: 12))
        }
        let image = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return image
    }

    private enum AnalyzerError: LocalizedError {
        case noVideoTrack
        case readerSetup
        case readerFailed
        case insufficientFrames
        case initializationFailed

        var errorDescription: String? {
            switch self {
            case .noVideoTrack: return "Video does not contain a readable video track"
            case .readerSetup: return "Could not configure the video reader"
            case .readerFailed: return "Video decoding failed"
            case .insufficientFrames: return "At least two decoded frames are required"
            case .initializationFailed:
                return "HMR2-S could not initialize WHAM from the first frame"
            }
        }
    }
}
