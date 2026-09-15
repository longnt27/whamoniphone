//
//  WhamAnalyzer.swift
//  WhamApp
//

import Foundation
import AVFoundation
import CoreML
import UIKit

private struct WHAMFrameObservation {
    let keypoints: MLMultiArray
    let keypointMask: MLMultiArray
    let imageFeature: MLMultiArray
    let imageFeatureValid: MLMultiArray
    let hmrPose: MLMultiArray?
    let hmrBetas: MLMultiArray?
    let videoTime: Double
}

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

                var observations: [WHAMFrameObservation] = []
                var frameIndex = 0
                var firstPresentationTime: Double?

                await MainActor.run {
                    self.statusMessage = "Đang nạp YOLO26m + HMR2-S + adapter..."
                }
                let visionConfiguration = MLModelConfiguration()
                visionConfiguration.computeUnits = .all
                var poseDetector: PoseDetector? = try PoseDetector(
                    backend: .yolo26m,
                    configuration: visionConfiguration
                )
                var hmr2s: MLModel? = try self.loadModel(
                    named: "HMR2SFrontend",
                    configuration: visionConfiguration
                )
                var tokenAdapter: MLModel? = try self.loadModel(
                    named: "HMR2STokenAdapter",
                    configuration: visionConfiguration
                )
                var smplInitializer: MLModel? = try self.loadModel(
                    named: "HMR2SSMPLInit",
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

                    let observation = try autoreleasepool { () throws -> WHAMFrameObservation in
                        guard let rawPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
                            return try self.missingObservation(videoTime: relativeTime)
                        }

                        let orientation = self.getVideoOrientation(from: transform)
                        let orientedImage = CIImage(cvPixelBuffer: rawPixelBuffer).oriented(orientation)
                        let width = Int(orientedImage.extent.width)
                        let height = Int(orientedImage.extent.height)
                        guard let uprightPixelBuffer = self.makePixelBuffer(width: width, height: height) else {
                            return try self.missingObservation(videoTime: relativeTime)
                        }
                        self.ciContext.render(orientedImage, to: uprightPixelBuffer)

                        guard let detection = poseDetector?.detect(pixelBuffer: uprightPixelBuffer).first else {
                            return try self.missingObservation(videoTime: relativeTime)
                        }

                        if frameIndex == 15 {
                            let debugImage = self.drawYOLODebug(
                                pixelBuffer: uprightPixelBuffer,
                                detection: detection
                            )
                            Task { @MainActor in self.debugImage = debugImage }
                        }

                        let squareBox = self.squarePersonBox(
                            detection,
                            imageWidth: CGFloat(width),
                            imageHeight: CGFloat(height)
                        )
                        let (keypoints, keypointMask) = try self.makeWHAMKeypoints(
                            detection: detection,
                            squareBox: squareBox,
                            imageWidth: CGFloat(width),
                            imageHeight: CGFloat(height)
                        )

                        guard let crop = self.cropPixelBuffer(uprightPixelBuffer, to: squareBox),
                              let hmr2s,
                              let tokenAdapter else {
                            return WHAMFrameObservation(
                                keypoints: keypoints,
                                keypointMask: keypointMask,
                                imageFeature: try self.zeros(
                                    [1, 1, 1024], dataType: .float16
                                ),
                                imageFeatureValid: try self.scalar(
                                    0, dataType: .float16
                                ),
                                hmrPose: nil,
                                hmrBetas: nil,
                                videoTime: relativeTime
                            )
                        }
                        let hmr = try self.predict(hmr2s, [
                            "image_input": MLFeatureValue(pixelBuffer: crop)
                        ])
                        let token = try self.requiredArray(hmr, "image_token")
                        let adapted = try self.predict(tokenAdapter, [
                            "hmr2s_token": MLFeatureValue(multiArray: token)
                        ])

                        return WHAMFrameObservation(
                            keypoints: keypoints,
                            keypointMask: keypointMask,
                            imageFeature: try self.requiredArray(
                                adapted, "wham_image_token"
                            ),
                            imageFeatureValid: try self.scalar(
                                1, dataType: .float16
                            ),
                            hmrPose: try self.requiredArray(hmr, "pose_6d"),
                            hmrBetas: try self.requiredArray(hmr, "betas"),
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

                let smpl = try self.predict(smplInitializerModel, [
                    "pose_6d": MLFeatureValue(multiArray: firstPose),
                    "betas": MLFeatureValue(multiArray: firstBetas)
                ])
                let initial3D = try self.requiredArray(smpl, "init_kp3d")

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
                let initializer = try self.loadModel(
                    named: "WHAM_I", configuration: whamConfiguration
                )
                let recurrentStep = try self.loadModel(
                    named: "WHAM_ImageStep", configuration: whamConfiguration
                )
                let worldStep = try self.loadModel(
                    named: "WHAM_WorldStep", configuration: whamConfiguration
                )

                let initialKeypoints = try self.concatenateInitializer(
                    joints3D: initial3D,
                    observation: observations[0].keypoints
                )
                let initialPose = try self.copyArray(
                    firstPose, shape: [1, 1, 144], dataType: .float32
                )
                let initialState = try self.predict(initializer, [
                    "init_kp": MLFeatureValue(multiArray: initialKeypoints),
                    "init_smpl": MLFeatureValue(multiArray: initialPose)
                ])

                var hEnc = try self.copyArray(
                    self.requiredArray(initialState, "h_enc"),
                    shape: [3, 1, 512], dataType: .float16
                )
                var cEnc = try self.copyArray(
                    self.requiredArray(initialState, "c_enc"),
                    shape: [3, 1, 512], dataType: .float16
                )
                var hTraj = try self.copyArray(
                    self.requiredArray(initialState, "h_traj"),
                    shape: [3, 1, 563], dataType: .float16
                )
                var cTraj = try self.copyArray(
                    self.requiredArray(initialState, "c_traj"),
                    shape: [3, 1, 563], dataType: .float16
                )
                var hDec = try self.copyArray(
                    self.requiredArray(initialState, "h_dec"),
                    shape: [3, 1, 563], dataType: .float16
                )
                var cDec = try self.copyArray(
                    self.requiredArray(initialState, "c_dec"),
                    shape: [3, 1, 563], dataType: .float16
                )
                var previousKeypoints = try self.copyArray(
                    initial3D, shape: [1, 1, 51], dataType: .float16
                )
                var previousRoot = try self.poseRoot(firstPose)
                var previousPose = try self.copyArray(
                    firstPose, shape: [1, 1, 144], dataType: .float16
                )
                var previousUnrefinedRoot = previousRoot
                var previousUnrefinedTranslation = try self.zeros(
                    [1, 1, 3], dataType: .float16
                )
                var previousBodyFeet = try self.zeros(
                    [1, 1, 4, 3], dataType: .float16
                )
                var previousWorldFeet = try self.zeros(
                    [1, 1, 4, 3], dataType: .float16
                )
                var previousRefinedRoot = previousRoot
                var previousRefinedTranslation = try self.zeros(
                    [1, 1, 3], dataType: .float16
                )
                var hRefiner = try self.zeros([2, 1, 512], dataType: .float16)
                var cRefiner = try self.zeros([2, 1, 512], dataType: .float16)
                var hasPreviousWorldFrame: Float = 0
                var smoother = TemporalOutputSmoother()
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

                    let output = try self.predict(recurrentStep, [
                        "x_step": MLFeatureValue(multiArray: observation.keypoints),
                        "keypoint_mask_step": MLFeatureValue(multiArray: observation.keypointMask),
                        "image_feature_step": MLFeatureValue(multiArray: observation.imageFeature),
                        "image_feature_valid_step": MLFeatureValue(multiArray: observation.imageFeatureValid),
                        "cam_a_step": MLFeatureValue(multiArray: cameraAngularVelocity),
                        "prev_kp3d": MLFeatureValue(multiArray: previousKeypoints),
                        "prev_root": MLFeatureValue(multiArray: previousRoot),
                        "prev_pose": MLFeatureValue(multiArray: previousPose),
                        "h_enc_in": MLFeatureValue(multiArray: hEnc),
                        "c_enc_in": MLFeatureValue(multiArray: cEnc),
                        "h_traj_in": MLFeatureValue(multiArray: hTraj),
                        "c_traj_in": MLFeatureValue(multiArray: cTraj),
                        "h_dec_in": MLFeatureValue(multiArray: hDec),
                        "c_dec_in": MLFeatureValue(multiArray: cDec)
                    ])
                    let rawPose = try self.requiredArray(output, "pred_pose")
                    let rawShape = try self.requiredArray(output, "pred_shape")
                    let smoothed = try smoother.smooth(pose: rawPose, shape: rawShape)
                    let world = try self.predict(worldStep, [
                        "pred_pose": MLFeatureValue(multiArray: smoothed.pose),
                        "pred_shape": MLFeatureValue(multiArray: smoothed.shape),
                        "pred_contact": MLFeatureValue(multiArray: try self.requiredArray(output, "pred_contact")),
                        "pred_root": MLFeatureValue(multiArray: try self.requiredArray(output, "pred_root")),
                        "pred_vel": MLFeatureValue(multiArray: try self.requiredArray(output, "pred_vel")),
                        "pred_kp3d": MLFeatureValue(multiArray: try self.requiredArray(output, "pred_kp3d")),
                        "h_enc_out": MLFeatureValue(multiArray: try self.requiredArray(output, "h_enc_out")),
                        "prev_unrefined_root": MLFeatureValue(multiArray: previousUnrefinedRoot),
                        "prev_unrefined_translation": MLFeatureValue(multiArray: previousUnrefinedTranslation),
                        "prev_body_feet": MLFeatureValue(multiArray: previousBodyFeet),
                        "prev_world_feet": MLFeatureValue(multiArray: previousWorldFeet),
                        "prev_refined_root": MLFeatureValue(multiArray: previousRefinedRoot),
                        "prev_refined_translation": MLFeatureValue(multiArray: previousRefinedTranslation),
                        "h_refiner_in": MLFeatureValue(multiArray: hRefiner),
                        "c_refiner_in": MLFeatureValue(multiArray: cRefiner),
                        "has_previous": MLFeatureValue(multiArray: try self.scalar(hasPreviousWorldFrame, dataType: .float16))
                    ])

                    hEnc = try self.requiredArray(output, "h_enc_out")
                    cEnc = try self.requiredArray(output, "c_enc_out")
                    hTraj = try self.requiredArray(output, "h_traj_out")
                    cTraj = try self.requiredArray(output, "c_traj_out")
                    hDec = try self.requiredArray(output, "h_dec_out")
                    cDec = try self.requiredArray(output, "c_dec_out")
                    previousKeypoints = try self.requiredArray(output, "pred_kp3d")
                    previousRoot = try self.requiredArray(output, "pred_root")
                    // Keep smoothing output-only, matching the locked 3DPW protocol.
                    previousPose = rawPose
                    previousUnrefinedRoot = try self.requiredArray(world, "unrefined_root")
                    previousUnrefinedTranslation = try self.requiredArray(world, "unrefined_translation")
                    previousBodyFeet = try self.requiredArray(world, "body_feet")
                    previousWorldFeet = try self.requiredArray(world, "world_feet")
                    previousRefinedRoot = try self.requiredArray(world, "refined_root")
                    previousRefinedTranslation = try self.requiredArray(world, "refined_translation")
                    hRefiner = try self.requiredArray(world, "h_refiner_out")
                    cRefiner = try self.requiredArray(world, "c_refiner_out")
                    hasPreviousWorldFrame = 1

                    results.append([
                        "frame": index,
                        "timestamp_seconds": observation.videoTime,
                        "pose_6d": self.toFloatArray(smoothed.pose),
                        "shape": self.toFloatArray(smoothed.shape),
                        "root_orient": self.toFloatArray(previousRefinedRoot),
                        "translation_world": self.toFloatArray(previousRefinedTranslation),
                        "keypoints_3d": self.toFloatArray(
                            try self.requiredArray(world, "joints_world"), limit: 51
                        ),
                        "keypoints_3d_root_relative": self.toFloatArray(previousKeypoints),
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

    // MARK: - WHAM inputs

    nonisolated private func makeWHAMKeypoints(
        detection: PoseDetector.Detection,
        squareBox: CGRect,
        imageWidth: CGFloat,
        imageHeight: CGFloat
    ) throws -> (MLMultiArray, MLMultiArray) {
        let keypoints = try zeros([1, 1, 37], dataType: .float16)
        let mask = try zeros([1, 1, 17], dataType: .float16)
        let side = max(squareBox.width, 1)

        for index in 0..<min(detection.keypoints.count, 17) {
            let point = detection.keypoints[index]
            let pixelX = point.x * imageWidth
            let pixelY = point.y * imageHeight
            keypoints[index * 2] = NSNumber(value: Float(2 * (pixelX - squareBox.midX) / side))
            keypoints[index * 2 + 1] = NSNumber(value: Float(2 * (pixelY - squareBox.midY) / side))
            let confidence = index < detection.keypointConfidences.count
                ? detection.keypointConfidences[index]
                : 0
            mask[index] = NSNumber(value: confidence < 0.3 ? 1 : 0)
        }

        let longestSide = max(imageWidth, imageHeight)
        keypoints[34] = NSNumber(value: Float(2 * squareBox.midX / longestSide - imageWidth / longestSide))
        keypoints[35] = NSNumber(value: Float(2 * squareBox.midY / longestSide - imageHeight / longestSide))
        keypoints[36] = NSNumber(value: Float(side / longestSide))
        return (keypoints, mask)
    }

    nonisolated private func missingObservation(videoTime: Double) throws -> WHAMFrameObservation {
        let mask = try zeros([1, 1, 17], dataType: .float16)
        for index in 0..<17 { mask[index] = 1 }
        return WHAMFrameObservation(
            keypoints: try zeros([1, 1, 37], dataType: .float16),
            keypointMask: mask,
            imageFeature: try zeros([1, 1, 1024], dataType: .float16),
            imageFeatureValid: try scalar(0, dataType: .float16),
            hmrPose: nil,
            hmrBetas: nil,
            videoTime: videoTime
        )
    }

    nonisolated private func cameraAngularVelocity(
        gyroSamples: [GyroSample],
        at videoTime: Double,
        frameDuration: Double
    ) throws -> MLMultiArray {
        let output = try zeros([1, 1, 6], dataType: .float16)
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

    nonisolated private func identityPose() throws -> MLMultiArray {
        let output = try zeros([1, 1, 144])
        for joint in 0..<24 {
            output[joint * 6] = 1
            output[joint * 6 + 4] = 1
        }
        return output
    }

    nonisolated private func identityRotation() throws -> MLMultiArray {
        let output = try zeros([1, 1, 6])
        output[0] = 1
        output[4] = 1
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

    nonisolated private func squarePersonBox(
        _ detection: PoseDetector.Detection,
        imageWidth: CGFloat,
        imageHeight: CGFloat
    ) -> CGRect {
        let candidates = detection.keypoints.enumerated().filter { index, _ in
            index < detection.keypointConfidences.count
                && detection.keypointConfidences[index] >= 0.3
        }.map(\.element)
        let centerX: CGFloat
        let centerY: CGFloat
        let side: CGFloat
        if candidates.count >= 7 {
            let minX = (candidates.map(\.x).min() ?? detection.box.minX) * imageWidth
            let maxX = (candidates.map(\.x).max() ?? detection.box.maxX) * imageWidth
            let minY = (candidates.map(\.y).min() ?? detection.box.minY) * imageHeight
            let maxY = (candidates.map(\.y).max() ?? detection.box.maxY) * imageHeight
            centerX = (minX + maxX) / 2
            centerY = (minY + maxY) / 2
            side = max(max(maxX - minX, maxY - minY) * 1.2, 1)
        } else {
            centerX = detection.box.midX * imageWidth
            centerY = detection.box.midY * imageHeight
            side = max(max(detection.box.width * imageWidth, detection.box.height * imageHeight) * 1.05, 1)
        }
        return CGRect(x: centerX - side / 2, y: centerY - side / 2, width: side, height: side)
    }

    nonisolated private func makePixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        guard width > 0, height > 0 else { return nil }
        var buffer: CVPixelBuffer?
        let attributes: CFDictionary = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue as Any,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue as Any
        ] as CFDictionary
        guard CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attributes,
            &buffer
        ) == kCVReturnSuccess else { return nil }
        return buffer
    }

    nonisolated private func cropPixelBuffer(_ buffer: CVPixelBuffer, to rect: CGRect) -> CVPixelBuffer? {
        let imageHeight = CGFloat(CVPixelBufferGetHeight(buffer))
        let coreImageRect = CGRect(
            x: rect.minX,
            y: imageHeight - rect.maxY,
            width: rect.width,
            height: rect.height
        )
        let source = CIImage(cvPixelBuffer: buffer).cropped(to: coreImageRect)
        let translated = source.transformed(by: CGAffineTransform(
            translationX: -coreImageRect.minX,
            y: -coreImageRect.minY
        ))
        let resized = translated.transformed(by: CGAffineTransform(
            scaleX: 256 / rect.width,
            y: 256 / rect.height
        ))
        let black = CIImage(color: .black).cropped(to: CGRect(x: 0, y: 0, width: 256, height: 256))
        let rendered = resized.composited(over: black)
        guard let output = makePixelBuffer(width: 256, height: 256) else { return nil }
        ciContext.render(rendered, to: output)
        return output
    }

    nonisolated private func loadModel(
        named name: String,
        configuration: MLModelConfiguration
    ) throws -> MLModel {
        guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {
            throw AnalyzerError.missingModel(name)
        }
        return try MLModel(contentsOf: url, configuration: configuration)
    }

    nonisolated private func predict(
        _ model: MLModel,
        _ values: [String: MLFeatureValue]
    ) throws -> MLFeatureProvider {
        try model.prediction(from: MLDictionaryFeatureProvider(dictionary: values))
    }

    nonisolated private func requiredArray(
        _ provider: MLFeatureProvider,
        _ name: String
    ) throws -> MLMultiArray {
        guard let array = provider.featureValue(for: name)?.multiArrayValue else {
            throw AnalyzerError.missingOutput(name)
        }
        return array
    }

    nonisolated private func copyArray(
        _ source: MLMultiArray,
        shape: [NSNumber],
        dataType: MLMultiArrayDataType
    ) throws -> MLMultiArray {
        let output = try zeros(shape, dataType: dataType)
        guard source.count == output.count else {
            throw AnalyzerError.arraySizeMismatch(source.count, output.count)
        }
        for index in 0..<source.count {
            output[index] = NSNumber(value: source[index].floatValue)
        }
        return output
    }

    nonisolated private func concatenateInitializer(
        joints3D: MLMultiArray,
        observation: MLMultiArray
    ) throws -> MLMultiArray {
        guard joints3D.count == 51, observation.count == 37 else {
            throw AnalyzerError.initializationFailed
        }
        let output = try zeros([1, 1, 88], dataType: .float32)
        for index in 0..<51 {
            output[index] = NSNumber(value: joints3D[index].floatValue)
        }
        for index in 0..<37 {
            output[51 + index] = NSNumber(value: observation[index].floatValue)
        }
        return output
    }

    nonisolated private func poseRoot(_ pose: MLMultiArray) throws -> MLMultiArray {
        guard pose.count >= 6 else {
            throw AnalyzerError.arraySizeMismatch(pose.count, 6)
        }
        let root = try zeros([1, 1, 6], dataType: .float16)
        for index in 0..<6 {
            root[index] = NSNumber(value: pose[index].floatValue)
        }
        return root
    }

    nonisolated private func toFloatArray(
        _ array: MLMultiArray,
        limit: Int? = nil
    ) -> [Float] {
        let count = min(limit ?? array.count, array.count)
        return (0..<count).map { array[$0].floatValue }
    }

    nonisolated private func reshapeTo1x1x1024(_ array: MLMultiArray) throws -> MLMultiArray {
        let output = try zeros([1, 1, 1024])
        for index in 0..<1024 { output[index] = array[index] }
        return output
    }

    nonisolated private func zeros(_ shape: [NSNumber]) throws -> MLMultiArray {
        try MLMultiArray(shape: shape, dataType: .float32)
    }

    nonisolated private func zeros(
        _ shape: [NSNumber],
        dataType: MLMultiArrayDataType
    ) throws -> MLMultiArray {
        try MLMultiArray(shape: shape, dataType: dataType)
    }

    nonisolated private func scalar(_ value: Float) throws -> MLMultiArray {
        let output = try zeros([1, 1, 1])
        output[0] = NSNumber(value: value)
        return output
    }

    nonisolated private func scalar(
        _ value: Float,
        dataType: MLMultiArrayDataType
    ) throws -> MLMultiArray {
        let output = try zeros([1, 1, 1], dataType: dataType)
        output[0] = NSNumber(value: value)
        return output
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
        case missingModel(String)
        case missingOutput(String)
        case arraySizeMismatch(Int, Int)

        var errorDescription: String? {
            switch self {
            case .noVideoTrack: return "Video does not contain a readable video track"
            case .readerSetup: return "Could not configure the video reader"
            case .readerFailed: return "Video decoding failed"
            case .insufficientFrames: return "At least two decoded frames are required"
            case .initializationFailed:
                return "HMR2-S could not initialize WHAM from the first frame"
            case .missingModel(let name):
                return "Bundled Core ML model \(name).mlmodelc was not found"
            case .missingOutput(let name):
                return "Required Core ML output \(name) was not found"
            case .arraySizeMismatch(let source, let destination):
                return "Core ML array size mismatch: \(source) → \(destination)"
            }
        }
    }
}
