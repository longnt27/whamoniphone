//
//  PoseDetector.swift
//  WhamApp
//

import Vision
import CoreML
import UIKit
import CoreImage

final class PoseDetector {
    enum Backend: String, CaseIterable, Codable, Identifiable {
        case yoloV8n = "YOLOv8n-pose"
        case yolo26n = "YOLO26n-pose"
        case yolo26m = "YOLO26m-pose"

        var id: String { rawValue }

        var resourceName: String {
            switch self {
            case .yoloV8n: return "yolov8n-pose"
            case .yolo26n: return "yolo26n-pose"
            case .yolo26m: return "yolo26m-pose"
            }
        }

        fileprivate var outputFeatureName: String {
            switch self {
            case .yoloV8n: return "var_1033"
            case .yolo26n, .yolo26m: return "var_1573"
            }
        }
    }

    let backend: Backend
    private var model: MLModel?
    private let ciContext = CIContext()

    init(backend: Backend = .yolo26m) {
        self.backend = backend
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        do {
            self.model = try Self.loadModel(backend: backend, configuration: configuration)
        } catch {
            print("❌ Could not load \(backend.rawValue): \(error)")
        }
    }

    init(
        backend: Backend = .yolo26m,
        configuration: MLModelConfiguration
    ) throws {
        self.backend = backend
        self.model = try Self.loadModel(backend: backend, configuration: configuration)
    }

    struct Detection {
        let box: CGRect
        let keypoints: [CGPoint]
        let keypointConfidences: [Float]
        let confidence: Float
    }

    func detect(pixelBuffer: CVPixelBuffer) -> [Detection] {
        guard let model else { return [] }

        guard let letterboxed = letterboxPixelBuffer(pixelBuffer, size: 640) else {
            print("❌ Could not resize pose-detector input")
            return []
        }

        do {
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "image": MLFeatureValue(pixelBuffer: letterboxed.buffer)
            ])
            let prediction = try model.prediction(from: input)
            let namedOutput = prediction.featureValue(
                for: backend.outputFeatureName
            )?.multiArrayValue
            let fallbackOutput = prediction.featureNames.lazy.compactMap {
                prediction.featureValue(for: $0)?.multiArrayValue
            }.first
            guard let output = namedOutput ?? fallbackOutput else {
                throw DetectorError.missingOutput(backend.outputFeatureName)
            }

            let detections: [Detection]
            switch backend {
            case .yoloV8n:
                detections = Self.parseYOLOv8Output(output)
            case .yolo26n, .yolo26m:
                detections = Self.parseYOLO26Output(output)
            }
            return detections.map { Self.removeLetterbox($0, letterboxed.transform) }
        } catch {
            print("❌ \(backend.rawValue) inference failed: \(error)")
            return []
        }
    }

    func detectAsync(pixelBuffer: CVPixelBuffer) async -> [Detection] {
        detect(pixelBuffer: pixelBuffer)
    }

    private static func loadModel(
        backend: Backend,
        configuration: MLModelConfiguration
    ) throws -> MLModel {
        guard let url = Bundle.main.url(
            forResource: backend.resourceName,
            withExtension: "mlmodelc"
        ) else {
            throw DetectorError.missingModel(backend.resourceName)
        }
        return try MLModel(contentsOf: url, configuration: configuration)
    }

    private struct LetterboxTransform {
        let sourceWidth: CGFloat
        let sourceHeight: CGFloat
        let scale: CGFloat
        let padX: CGFloat
        let padY: CGFloat
        let targetSize: CGFloat
    }

    private func letterboxPixelBuffer(
        _ buffer: CVPixelBuffer,
        size: Int
    ) -> (buffer: CVPixelBuffer, transform: LetterboxTransform)? {
        let ciImage = CIImage(cvPixelBuffer: buffer)
        let sourceWidth = CGFloat(CVPixelBufferGetWidth(buffer))
        let sourceHeight = CGFloat(CVPixelBufferGetHeight(buffer))
        let targetSize = CGFloat(size)
        let scale = min(targetSize / sourceWidth, targetSize / sourceHeight)
        let scaledWidth = sourceWidth * scale
        let scaledHeight = sourceHeight * scale
        let padX = (targetSize - scaledWidth) / 2
        let padY = (targetSize - scaledHeight) / 2
        let resized = ciImage
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            .transformed(by: CGAffineTransform(translationX: padX, y: padY))
        let gray = CGFloat(114.0 / 255.0)
        let background = CIImage(color: CIColor(red: gray, green: gray, blue: gray))
            .cropped(to: CGRect(x: 0, y: 0, width: targetSize, height: targetSize))
        let letterboxed = resized.composited(over: background)

        var newBuffer: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary

        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            size,
            size,
            kCVPixelFormatType_32BGRA,
            attrs,
            &newBuffer
        )

        if status == kCVReturnSuccess, let newBuffer {
            ciContext.render(letterboxed, to: newBuffer)
            return (
                newBuffer,
                LetterboxTransform(
                    sourceWidth: sourceWidth,
                    sourceHeight: sourceHeight,
                    scale: scale,
                    padX: padX,
                    padY: padY,
                    targetSize: targetSize
                )
            )
        }
        return nil
    }

    private static func removeLetterbox(
        _ detection: Detection,
        _ transform: LetterboxTransform
    ) -> Detection {
        func sourcePoint(_ point: CGPoint) -> CGPoint {
            let x = (point.x * transform.targetSize - transform.padX)
                / transform.scale / transform.sourceWidth
            let y = (point.y * transform.targetSize - transform.padY)
                / transform.scale / transform.sourceHeight
            return CGPoint(x: min(max(x, 0), 1), y: min(max(y, 0), 1))
        }

        let minimum = sourcePoint(CGPoint(x: detection.box.minX, y: detection.box.minY))
        let maximum = sourcePoint(CGPoint(x: detection.box.maxX, y: detection.box.maxY))
        return Detection(
            box: CGRect(
                x: minimum.x,
                y: minimum.y,
                width: max(maximum.x - minimum.x, 0),
                height: max(maximum.y - minimum.y, 0)
            ),
            keypoints: detection.keypoints.map(sourcePoint),
            keypointConfidences: detection.keypointConfidences,
            confidence: detection.confidence
        )
    }

    /// YOLOv8 pose export: [1, 56, 8400], center-x/center-y/width/height,
    /// confidence, then 17 x (x, y, confidence).
    static func parseYOLOv8Output(_ output: MLMultiArray) -> [Detection] {
        let numDetections = output.shape[2].intValue
        var bestScore: Float = 0
        var bestIndex = -1

        for index in 0..<numDetections {
            let score = output[[0, 4, index] as [NSNumber]].floatValue
            if score > 0.5 && score > bestScore {
                bestScore = score
                bestIndex = index
            }
        }
        guard bestIndex >= 0 else { return [] }

        let centerX = output[[0, 0, bestIndex] as [NSNumber]].doubleValue / 640
        let centerY = output[[0, 1, bestIndex] as [NSNumber]].doubleValue / 640
        let width = output[[0, 2, bestIndex] as [NSNumber]].doubleValue / 640
        let height = output[[0, 3, bestIndex] as [NSNumber]].doubleValue / 640
        let box = CGRect(
            x: centerX - width / 2,
            y: centerY - height / 2,
            width: width,
            height: height
        )

        var keypoints: [CGPoint] = []
        var keypointConfidences: [Float] = []
        for joint in 0..<17 {
            let offset = 5 + joint * 3
            let x = output[[0, offset, bestIndex] as [NSNumber]].doubleValue / 640
            let y = output[[0, offset + 1, bestIndex] as [NSNumber]].doubleValue / 640
            keypoints.append(CGPoint(x: x, y: y))
            keypointConfidences.append(
                output[[0, offset + 2, bestIndex] as [NSNumber]].floatValue
            )
        }
        return [Detection(
            box: box,
            keypoints: keypoints,
            keypointConfidences: keypointConfidences,
            confidence: bestScore
        )]
    }

    /// YOLO26 end-to-end pose export: [1, 300, 57], x1/y1/x2/y2,
    /// confidence, class, then 17 x (x, y, confidence). NMS is in the model.
    static func parseYOLO26Output(_ output: MLMultiArray) -> [Detection] {
        let numDetections = output.shape[1].intValue
        var bestScore: Float = 0
        var bestIndex = -1

        for index in 0..<numDetections {
            let score = output[[0, index, 4] as [NSNumber]].floatValue
            if score > 0.5 && score > bestScore {
                bestScore = score
                bestIndex = index
            }
        }
        guard bestIndex >= 0 else { return [] }

        let x1 = output[[0, bestIndex, 0] as [NSNumber]].doubleValue / 640
        let y1 = output[[0, bestIndex, 1] as [NSNumber]].doubleValue / 640
        let x2 = output[[0, bestIndex, 2] as [NSNumber]].doubleValue / 640
        let y2 = output[[0, bestIndex, 3] as [NSNumber]].doubleValue / 640
        let box = CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1)

        var keypoints: [CGPoint] = []
        var keypointConfidences: [Float] = []
        for joint in 0..<17 {
            let offset = 6 + joint * 3
            let x = output[[0, bestIndex, offset] as [NSNumber]].doubleValue / 640
            let y = output[[0, bestIndex, offset + 1] as [NSNumber]].doubleValue / 640
            keypoints.append(CGPoint(x: x, y: y))
            keypointConfidences.append(
                output[[0, bestIndex, offset + 2] as [NSNumber]].floatValue
            )
        }
        return [Detection(
            box: box,
            keypoints: keypoints,
            keypointConfidences: keypointConfidences,
            confidence: bestScore
        )]
    }

    private enum DetectorError: LocalizedError {
        case missingModel(String)
        case missingOutput(String)

        var errorDescription: String? {
            switch self {
            case .missingModel(let name):
                return "Bundled Core ML model \(name).mlmodelc was not found"
            case .missingOutput(let name):
                return "Core ML output \(name) was not found"
            }
        }
    }
}
