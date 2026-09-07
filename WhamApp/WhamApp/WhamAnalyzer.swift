//
//  WhamAnalyzer.swift
//  WhamApp
//

import Foundation
import AVFoundation
import CoreML
import UIKit

@MainActor
class WhamAnalyzer: ObservableObject {
    @Published var progress: Double = 0
    @Published var isProcessing = false
    @Published var statusMessage: String = ""
    @Published var debugImage: UIImage?

    private let ciContext = CIContext()

    // One record per source frame keeps keypoints and image features aligned,
    // including frames whose detection, image conversion or crop failed.
    private struct FrameInput {
        let keypoints: MLMultiArray
        let features: MLMultiArray
        let sourceFrame: Int
        let timestamp: Double
        let hasImageFeatures: Bool
    }

    init() {}

    func analyze(videoURL: URL, gyroJsonURL: URL, outputURL: URL) async {
        self.isProcessing = true
        self.progress = 0
        self.statusMessage = "Đang chuẩn bị dữ liệu..."

        do {
            try await Task.detached(priority: .userInitiated) {
                let gyroData = (try? self.loadGyroData(from: gyroJsonURL)) ?? []
                let asset = AVURLAsset(url: videoURL)
                let estimatedFrames = gyroData.isEmpty ? 500 : gyroData.count
                let reader = try AVAssetReader(asset: asset)
                let tracks = try await asset.loadTracks(withMediaType: .video)
                guard let track = tracks.first else {
                    throw self.analysisError("Video has no video track.")
                }
                let transform = try await track.load(.preferredTransform)
                let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
                ])
                reader.add(output)
                guard reader.startReading() else {
                    throw reader.error ?? self.analysisError("Cannot read the video.")
                }

                var frames: [FrameInput] = []
                var frameIdx = 0

                await MainActor.run { self.statusMessage = "Đang nạp AI Thị giác..." }
                var poseDetector: PoseDetector? = PoseDetector()
                var fastViT: _FastViT? = try _FastViT(configuration: MLModelConfiguration())
                let metadata = fastViT?.model.modelDescription.metadata[.creatorDefinedKey] as? [String: String]
                guard metadata?["wham.rgb_normalization"] == "imagenet-rgb-0-255-v1" else {
                    throw self.analysisError("Re-export FastViT with utils/extractViT.py; the old package lacks RGB normalization.")
                }

                await MainActor.run { self.statusMessage = "Đang trích xuất (Stage 1/2)..." }

                // STAGE 1: paired keypoint/feature extraction.
                while reader.status == .reading {
                    guard let buffer = output.copyNextSampleBuffer() else { break }
                    let frame = try autoreleasepool { () throws -> FrameInput in
                        let kpArray = try self.zeros([1, 1, 37])
                        var features = try self.zeros([1, 1, 1024])
                        var hasImageFeatures = false
                        let timestamp = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(buffer))

                        if let rawPix = CMSampleBufferGetImageBuffer(buffer) {
                            let orientation = self.getVideoOrientation(from: transform)
                            let oriented = CIImage(cvPixelBuffer: rawPix).oriented(orientation)
                            let ciImage = oriented.transformed(by: CGAffineTransform(
                                translationX: -oriented.extent.minX, y: -oriented.extent.minY
                            ))
                            let realWidth = ciImage.extent.width
                            let realHeight = ciImage.extent.height
                            var uprightPix: CVPixelBuffer?
                            CVPixelBufferCreate(nil, Int(realWidth), Int(realHeight), kCVPixelFormatType_32BGRA, nil, &uprightPix)

                            if let pix = uprightPix {
                                self.ciContext.render(ciImage, to: pix)
                                if let person = poseDetector?.detect(pixelBuffer: pix).first {
                                    if frameIdx == 15 {
                                        let debugImg = self.drawYOLO_DEBUG(pixelBuffer: pix, detection: person)
                                        Task { @MainActor in self.debugImage = debugImg }
                                    }
                                    let realPixelBox = CGRect(
                                        x: person.box.origin.x * realWidth,
                                        y: person.box.origin.y * realHeight,
                                        width: person.box.width * realWidth,
                                        height: person.box.height * realHeight
                                    )
                                    if let cropped = self.cropPixelBuffer(pix, to: realPixelBox),
                                       let vitOutput = try fastViT?.prediction(input: _FastViTInput(image_input: cropped)) {
                                        features = try self.reshapeTo1x1x1024(vitOutput.features_1024)
                                        hasImageFeatures = true
                                    }

                                    // Existing keypoint normalization is unchanged in this PR.
                                    // Matching upstream bbox normalization remains a separate task.
                                    for (j, kp) in person.keypoints.prefix(17).enumerated() {
                                        kpArray[[0, 0, j*2] as [NSNumber]] = NSNumber(value: Float(kp.x * 2.0 - 1.0))
                                        kpArray[[0, 0, j*2+1] as [NSNumber]] = NSNumber(value: Float(kp.y * 2.0 - 1.0))
                                    }
                                    kpArray[[0, 0, 34] as [NSNumber]] = NSNumber(value: Float(person.box.midX * 2.0 - 1.0))
                                    kpArray[[0, 0, 35] as [NSNumber]] = NSNumber(value: Float(person.box.midY * 2.0 - 1.0))
                                    kpArray[[0, 0, 36] as [NSNumber]] = NSNumber(value: Float(person.box.width * 2.0))
                                }
                            }
                        }
                        return FrameInput(keypoints: kpArray, features: features,
                                          sourceFrame: frameIdx, timestamp: timestamp,
                                          hasImageFeatures: hasImageFeatures)
                    }
                    frames.append(frame)
                    frameIdx += 1
                    let currentProgress = min(0.5, (Double(frameIdx) / Double(max(estimatedFrames, 1))) * 0.5)
                    await MainActor.run { self.progress = currentProgress }
                }
                if reader.status == .failed {
                    throw reader.error ?? self.analysisError("Video decoding failed.")
                }
                guard !frames.isEmpty else { throw self.analysisError("Video contains no readable frames.") }

                await MainActor.run { self.statusMessage = "Đang dọn dẹp RAM..." }
                poseDetector = nil
                fastViT = nil
                try await Task.sleep(nanoseconds: 3_000_000_000)

                // STAGE 2: explicit feature input to the re-exported WHAM graph.
                await MainActor.run { self.statusMessage = "Đang nạp AI WHAM..." }
                let whamConfig = MLModelConfiguration()
                whamConfig.computeUnits = .all
                let initModel = try WHAM_I(configuration: whamConfig)
                let stepModel = try WHAM_S(configuration: whamConfig)
                let featureDescription = stepModel.model.modelDescription.inputDescriptionsByName["features_step"]
                guard featureDescription?.type == .multiArray,
                      featureDescription?.multiArrayConstraint?.shape.map({ $0.intValue }) == [1, 1, 1024] else {
                    throw self.analysisError("Re-export WHAM_S with utils/wham_coreml.py; features_step [1,1,1024] is required.")
                }

                await MainActor.run { self.statusMessage = "Đang nội suy 3D (Stage 2/2)..." }
                var whamResults: [[String: Any]] = []
                let totalFrames = frames.count
                let init_kp = try self.zeros([1, 1, 88])
                let init_smpl = try self.zeros([1, 1, 144])
                let initOutput = try initModel.prediction(init_kp: init_kp, init_smpl: init_smpl)
                var current_h_enc = initOutput.h_enc
                var current_c_enc = initOutput.c_enc
                var current_h_traj = initOutput.h_traj
                var current_c_traj = initOutput.c_traj
                var current_h_dec = initOutput.h_dec
                var current_c_dec = initOutput.c_dec
                var prev_kp3d = try self.zeros([1, 1, 51])
                var prev_root = try self.zeros([1, 1, 6])
                var prev_pose = init_smpl
                let cam_a_step = try self.zeros([1, 1, 6])

                for i in 0..<totalFrames {
                    try autoreleasepool {
                        let frame = frames[i]
                        // A stale generated WHAM_SInput class must not silently
                        // omit the feature tensor. The schema is checked above.
                        let stepInput = try MLDictionaryFeatureProvider(dictionary: [
                            "x_step": frame.keypoints,
                            "features_step": frame.features,
                            "cam_a_step": cam_a_step,
                            "prev_kp3d": prev_kp3d,
                            "prev_root": prev_root,
                            "prev_pose": prev_pose,
                            "h_enc_in": current_h_enc,
                            "c_enc_in": current_c_enc,
                            "h_traj_in": current_h_traj,
                            "c_traj_in": current_c_traj,
                            "h_dec_in": current_h_dec,
                            "c_dec_in": current_c_dec
                        ])
                        let stepOutput = try stepModel.model.prediction(from: stepInput)
                        current_h_enc = try self.tensor("h_enc_out", from: stepOutput)
                        current_c_enc = try self.tensor("c_enc_out", from: stepOutput)
                        current_h_traj = try self.tensor("h_traj_out", from: stepOutput)
                        current_c_traj = try self.tensor("c_traj_out", from: stepOutput)
                        current_h_dec = try self.tensor("h_dec_out", from: stepOutput)
                        current_c_dec = try self.tensor("c_dec_out", from: stepOutput)
                        prev_kp3d = try self.tensor("pred_kp3d", from: stepOutput)
                        prev_root = try self.tensor("pred_root", from: stepOutput)
                        prev_pose = try self.tensor("pred_pose", from: stepOutput)
                        let shape = try self.tensor("pred_shape", from: stepOutput)

                        var result: [String: Any] = [
                            "frame": frame.sourceFrame,
                            "has_image_features": frame.hasImageFeatures,
                            "pose_6d": self.toFloatArray(prev_pose),
                            "shape": self.toFloatArray(shape),
                            "root_orient": self.toFloatArray(prev_root),
                            "keypoints_3d": self.toFloatArray(prev_kp3d)
                        ]
                        if frame.timestamp.isFinite { result["timestamp_s"] = frame.timestamp }
                        whamResults.append(result)
                    }
                    let currentProgress = 0.5 + (Double(i + 1) / Double(totalFrames)) * 0.5
                    await MainActor.run { self.progress = currentProgress }
                }
                let finalData = try JSONSerialization.data(withJSONObject: whamResults)
                try finalData.write(to: outputURL)
            }.value

            self.statusMessage = "✅ PHÂN TÍCH THÀNH CÔNG!"
            self.isProcessing = false
        } catch {
            print("Analysis Failed: \(error)")
            self.statusMessage = "❌ Lỗi: \(error.localizedDescription)"
            self.isProcessing = false
        }
    }

    nonisolated private func analysisError(_ message: String) -> NSError {
        NSError(domain: "WhamAnalyzer", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }

    nonisolated private func tensor(_ name: String, from output: MLFeatureProvider) throws -> MLMultiArray {
        guard let value = output.featureValue(for: name)?.multiArrayValue else {
            throw analysisError("WHAM output is missing tensor \(name).")
        }
        return value
    }

    nonisolated private func getVideoOrientation(from transform: CGAffineTransform) -> CGImagePropertyOrientation {
        if transform.a == 0 && transform.b == 1.0 && transform.c == -1.0 && transform.d == 0 {
            return .right
        } else if transform.a == 0 && transform.b == -1.0 && transform.c == 1.0 && transform.d == 0 {
            return .left
        } else if transform.a == 1.0 && transform.b == 0 && transform.c == 0 && transform.d == 1.0 {
            return .up
        } else if transform.a == -1.0 && transform.b == 0 && transform.c == 0 && transform.d == -1.0 {
            return .down
        }
        return .right
    }

    nonisolated private func toFloatArray(_ arr: MLMultiArray) -> [Float] {
        var result: [Float] = []
        result.reserveCapacity(arr.count)
        for i in 0..<arr.count { result.append(arr[i].floatValue) }
        return result
    }

    nonisolated private func reshapeTo1x1x1024(_ arr: MLMultiArray) throws -> MLMultiArray {
        guard arr.count == 1024 else { throw analysisError("FastViT must output 1024 features, got \(arr.count).") }
        let newArr = try MLMultiArray(shape: [1, 1, 1024], dataType: .float32)
        for i in 0..<1024 {
            let value = arr[i].floatValue
            guard value.isFinite else { throw analysisError("FastViT returned a non-finite feature.") }
            newArr[i] = NSNumber(value: value)
        }
        return newArr
    }

    nonisolated private func zeros(_ shape: [NSNumber]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        // MLMultiArray allocation is not a promise of zero-filled storage.
        for i in 0..<array.count { array[i] = 0 }
        return array
    }

    nonisolated private func cropPixelBuffer(_ buffer: CVPixelBuffer, to rect: CGRect) -> CVPixelBuffer? {
        guard rect.minX.isFinite, rect.minY.isFinite, rect.width.isFinite,
              rect.height.isFinite, rect.width > 0, rect.height > 0 else { return nil }
        let image = CIImage(cvPixelBuffer: buffer)
        let bounds = CGRect(x: 0, y: 0, width: image.extent.width, height: image.extent.height)
        let clipped = rect.intersection(bounds)
        guard !clipped.isNull, clipped.width >= 1, clipped.height >= 1 else { return nil }
        // PoseDetector returns top-left image coordinates; Core Image uses bottom-left.
        let crop = CGRect(x: image.extent.minX + clipped.minX,
                          y: image.extent.maxY - clipped.maxY,
                          width: clipped.width, height: clipped.height)
        let transform = CGAffineTransform(translationX: -crop.minX, y: -crop.minY)
            .concatenating(CGAffineTransform(scaleX: 256.0 / crop.width, y: 256.0 / crop.height))
        let resized = image.cropped(to: crop).transformed(by: transform)
        var newBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(nil, 256, 256, kCVPixelFormatType_32BGRA, nil, &newBuffer)
        guard status == kCVReturnSuccess, let result = newBuffer else { return nil }
        ciContext.render(resized, to: result)
        return result
    }

    nonisolated private func loadGyroData(from url: URL) throws -> [[String: Any]] {
        let data = try Data(contentsOf: url)
        return try JSONSerialization.jsonObject(with: data) as? [[String: Any]] ?? []
    }

    // --- YOLO 2D VISUAL DEBUGGER ---
    nonisolated private func drawYOLO_DEBUG(pixelBuffer: CVPixelBuffer, detection: PoseDetector.Detection) -> UIImage? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else { return nil }
        let width = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let height = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        UIGraphicsBeginImageContextWithOptions(CGSize(width: width, height: height), false, 1.0)
        defer { UIGraphicsEndImageContext() }
        guard let ctx = UIGraphicsGetCurrentContext() else { return nil }
        UIImage(cgImage: cgImage).draw(in: CGRect(x: 0, y: 0, width: width, height: height))
        let realBox = CGRect(x: detection.box.origin.x * width, y: detection.box.origin.y * height,
                             width: detection.box.width * width, height: detection.box.height * height)
        ctx.setStrokeColor(UIColor.red.cgColor)
        ctx.setLineWidth(4.0)
        ctx.stroke(realBox)
        ctx.setFillColor(UIColor.green.cgColor)
        for kp in detection.keypoints {
            let realX = kp.x * width
            let realY = kp.y * height
            ctx.fillEllipse(in: CGRect(x: realX - 6, y: realY - 6, width: 12, height: 12))
        }
        return UIGraphicsGetImageFromCurrentImageContext()
    }
}
