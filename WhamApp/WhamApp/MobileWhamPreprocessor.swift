import CoreImage
import CoreML
import Foundation

struct MobileWhamPreprocessor {
    static let keypointConfidenceThreshold: Float = 0.3
    static let minimumConfidentKeypoints = 7
    static let keypointCropScale: CGFloat = 1.2
    static let boxCropScale: CGFloat = 1.05
    static let hmrCropSize = 256

    let ciContext: CIContext

    init(ciContext: CIContext = CIContext()) {
        self.ciContext = ciContext
    }

    static func missingFrame(videoTime: Double) throws -> MobileWhamFrameObservation {
        let mask = try MobileWhamArrays.zeros([1, 1, 17], dataType: .float16)
        for index in 0..<17 { mask[index] = 1 }
        return MobileWhamFrameObservation(
            keypoints: try MobileWhamArrays.zeros([1, 1, 37], dataType: .float16),
            keypointMask: mask,
            imageFeature: try MobileWhamArrays.zeros(
                [1, 1, 1024], dataType: .float16
            ),
            imageFeatureValid: try MobileWhamArrays.scalar(0, dataType: .float16),
            hmrPose: nil,
            hmrBetas: nil,
            hmrCamera: nil,
            hmrCropBox: nil,
            sourceSize: .zero,
            videoTime: videoTime
        )
    }

    func visualObservation(
        detection: PoseDetector.Detection?,
        source: CVPixelBuffer
    ) throws -> MobileWhamVisualObservation {
        guard let detection else {
            let missing = try Self.missingFrame(videoTime: 0)
            return MobileWhamVisualObservation(
                keypoints: missing.keypoints,
                keypointMask: missing.keypointMask,
                crop: nil,
                cropBox: nil,
                sourceSize: CGSize(
                    width: CVPixelBufferGetWidth(source),
                    height: CVPixelBufferGetHeight(source)
                )
            )
        }
        let width = CGFloat(CVPixelBufferGetWidth(source))
        let height = CGFloat(CVPixelBufferGetHeight(source))
        let box = squarePersonBox(
            detection,
            imageWidth: width,
            imageHeight: height
        )
        let (keypoints, mask) = try whamKeypoints(
            detection: detection,
            squareBox: box,
            imageWidth: width,
            imageHeight: height
        )
        return MobileWhamVisualObservation(
            keypoints: keypoints,
            keypointMask: mask,
            crop: cropPixelBuffer(source, to: box),
            cropBox: box,
            sourceSize: CGSize(width: width, height: height)
        )
    }

    static func frame(
        visual: MobileWhamVisualObservation,
        frontend: MobileWhamFrontendOutput?,
        adaptedFeature: MLMultiArray?,
        videoTime: Double
    ) throws -> MobileWhamFrameObservation {
        MobileWhamFrameObservation(
            keypoints: visual.keypoints,
            keypointMask: visual.keypointMask,
            imageFeature: try adaptedFeature ?? MobileWhamArrays.zeros(
                [1, 1, 1024], dataType: .float16
            ),
            imageFeatureValid: try MobileWhamArrays.scalar(
                adaptedFeature == nil ? 0 : 1,
                dataType: .float16
            ),
            hmrPose: frontend?.pose,
            hmrBetas: frontend?.betas,
            hmrCamera: frontend?.camera,
            hmrCropBox: visual.cropBox,
            sourceSize: visual.sourceSize,
            videoTime: videoTime
        )
    }

    func whamKeypoints(
        detection: PoseDetector.Detection,
        squareBox: CGRect,
        imageWidth: CGFloat,
        imageHeight: CGFloat
    ) throws -> (MLMultiArray, MLMultiArray) {
        let keypoints = try MobileWhamArrays.zeros([1, 1, 37], dataType: .float16)
        let mask = try MobileWhamArrays.zeros([1, 1, 17], dataType: .float16)
        let side = max(squareBox.width, 1)

        for index in 0..<min(detection.keypoints.count, 17) {
            let point = detection.keypoints[index]
            let pixelX = point.x * imageWidth
            let pixelY = point.y * imageHeight
            keypoints[index * 2] = NSNumber(
                value: Float(2 * (pixelX - squareBox.midX) / side)
            )
            keypoints[index * 2 + 1] = NSNumber(
                value: Float(2 * (pixelY - squareBox.midY) / side)
            )
            let confidence = index < detection.keypointConfidences.count
                ? detection.keypointConfidences[index]
                : 0
            mask[index] = NSNumber(
                value: confidence < Self.keypointConfidenceThreshold ? 1 : 0
            )
        }

        let longestSide = max(imageWidth, imageHeight)
        keypoints[34] = NSNumber(
            value: Float(2 * squareBox.midX / longestSide - imageWidth / longestSide)
        )
        keypoints[35] = NSNumber(
            value: Float(2 * squareBox.midY / longestSide - imageHeight / longestSide)
        )
        keypoints[36] = NSNumber(value: Float(side / longestSide))
        return (keypoints, mask)
    }

    func squarePersonBox(
        _ detection: PoseDetector.Detection,
        imageWidth: CGFloat,
        imageHeight: CGFloat
    ) -> CGRect {
        let candidates = detection.keypoints.enumerated().filter { index, _ in
            index < detection.keypointConfidences.count
                && detection.keypointConfidences[index]
                    >= Self.keypointConfidenceThreshold
        }.map(\.element)
        let centerX: CGFloat
        let centerY: CGFloat
        let side: CGFloat
        if candidates.count >= Self.minimumConfidentKeypoints {
            let minX = (candidates.map(\.x).min() ?? detection.box.minX) * imageWidth
            let maxX = (candidates.map(\.x).max() ?? detection.box.maxX) * imageWidth
            let minY = (candidates.map(\.y).min() ?? detection.box.minY) * imageHeight
            let maxY = (candidates.map(\.y).max() ?? detection.box.maxY) * imageHeight
            centerX = (minX + maxX) / 2
            centerY = (minY + maxY) / 2
            side = max(
                max(maxX - minX, maxY - minY) * Self.keypointCropScale,
                1
            )
        } else {
            centerX = detection.box.midX * imageWidth
            centerY = detection.box.midY * imageHeight
            side = max(
                max(
                    detection.box.width * imageWidth,
                    detection.box.height * imageHeight
                ) * Self.boxCropScale,
                1
            )
        }
        return CGRect(
            x: centerX - side / 2,
            y: centerY - side / 2,
            width: side,
            height: side
        )
    }

    func makePixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
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

    private func cropPixelBuffer(
        _ buffer: CVPixelBuffer,
        to rect: CGRect
    ) -> CVPixelBuffer? {
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
            scaleX: CGFloat(Self.hmrCropSize) / rect.width,
            y: CGFloat(Self.hmrCropSize) / rect.height
        ))
        let black = CIImage(color: .black).cropped(
            to: CGRect(
                x: 0,
                y: 0,
                width: Self.hmrCropSize,
                height: Self.hmrCropSize
            )
        )
        guard let output = makePixelBuffer(
            width: Self.hmrCropSize,
            height: Self.hmrCropSize
        ) else { return nil }
        ciContext.render(resized.composited(over: black), to: output)
        return output
    }
}
