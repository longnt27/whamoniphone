import CoreGraphics
import Foundation
import simd

struct VideoOverlayMetadata {
    let camera: SIMD3<Float>
    let cropCenter: CGPoint
    let cropSize: CGFloat
    let sourceSize: CGSize
    let poseRoot6D: [Float]
    let refinedRoot6D: [Float]
    let worldTranslation: SIMD3<Float>

    init(
        camera: SIMD3<Float>,
        cropCenter: CGPoint,
        cropSize: CGFloat,
        sourceSize: CGSize,
        poseRoot6D: [Float],
        refinedRoot6D: [Float],
        worldTranslation: SIMD3<Float>
    ) {
        self.camera = camera
        self.cropCenter = cropCenter
        self.cropSize = cropSize
        self.sourceSize = sourceSize
        self.poseRoot6D = Array(poseRoot6D.prefix(6))
        self.refinedRoot6D = Array(refinedRoot6D.prefix(6))
        self.worldTranslation = worldTranslation
    }

    init?(dictionary: [String: Any]) {
        guard let cameraValues = Self.floatArray(dictionary["hmr_camera"]),
              cameraValues.count >= 3,
              let cropCenterValues = Self.floatArray(
                dictionary["hmr_crop_center_pixels"]
              ),
              cropCenterValues.count >= 2,
              let cropSizeValue = Self.floatValue(
                dictionary["hmr_crop_size_pixels"]
              ),
              let sourceSizeValues = Self.floatArray(
                dictionary["source_image_size_pixels"]
              ),
              sourceSizeValues.count >= 2,
              let poseValues = Self.floatArray(dictionary["pose_6d"]),
              poseValues.count >= 6,
              let refinedRootValues = Self.floatArray(
                dictionary["root_orient"]
              ),
              refinedRootValues.count >= 6,
              let translationValues = Self.floatArray(
                dictionary["translation_world"]
              ),
              translationValues.count >= 3 else {
            return nil
        }

        self.init(
            camera: SIMD3<Float>(
                cameraValues[0], cameraValues[1], cameraValues[2]
            ),
            cropCenter: CGPoint(
                x: CGFloat(cropCenterValues[0]),
                y: CGFloat(cropCenterValues[1])
            ),
            cropSize: CGFloat(cropSizeValue),
            sourceSize: CGSize(
                width: CGFloat(sourceSizeValues[0]),
                height: CGFloat(sourceSizeValues[1])
            ),
            poseRoot6D: poseValues,
            refinedRoot6D: refinedRootValues,
            worldTranslation: SIMD3<Float>(
                translationValues[0],
                translationValues[1],
                translationValues[2]
            )
        )
    }

    var isValid: Bool {
        camera.x.isFinite
            && camera.y.isFinite
            && camera.z.isFinite
            && camera.x > 0
            && cropCenter.x.isFinite
            && cropCenter.y.isFinite
            && cropSize.isFinite
            && cropSize > 0
            && sourceSize.width.isFinite
            && sourceSize.height.isFinite
            && sourceSize.width > 0
            && sourceSize.height > 0
            && poseRoot6D.count == 6
            && refinedRoot6D.count == 6
            && poseRoot6D.allSatisfy(\.isFinite)
            && refinedRoot6D.allSatisfy(\.isFinite)
            && worldTranslation.x.isFinite
            && worldTranslation.y.isFinite
            && worldTranslation.z.isFinite
    }

    private static func floatArray(_ value: Any?) -> [Float]? {
        if let values = value as? [Float] { return values }
        if let values = value as? [Double] { return values.map(Float.init) }
        if let values = value as? [NSNumber] {
            return values.map(\.floatValue)
        }
        return nil
    }

    private static func floatValue(_ value: Any?) -> Float? {
        if let value = value as? Float { return value }
        if let value = value as? Double { return Float(value) }
        if let value = value as? NSNumber { return value.floatValue }
        return nil
    }
}

enum SMPLVideoProjector {
    // HMR2's released renderer and training projection both use this focal
    // length for the 256-pixel person crop.
    static let focalLength: Float = 5_000

    static func worldToCamera(
        _ worldVertices: [SIMD3<Float>],
        metadata: VideoOverlayMetadata
    ) -> [SIMD3<Float>] {
        guard let worldRotation = worldBodyRotation(metadata: metadata) else {
            return []
        }
        let inverse = worldRotation.transpose
        return worldVertices.map {
            inverse * ($0 - metadata.worldTranslation)
        }
    }

    static func cameraToWorld(
        _ cameraVertex: SIMD3<Float>,
        metadata: VideoOverlayMetadata
    ) -> SIMD3<Float> {
        guard let worldRotation = worldBodyRotation(metadata: metadata) else {
            return SIMD3<Float>(repeating: .nan)
        }
        return worldRotation * cameraVertex + metadata.worldTranslation
    }

    static func projectWorldToViewport(
        _ worldVertices: [SIMD3<Float>],
        metadata: VideoOverlayMetadata,
        viewportSize: CGSize
    ) -> [SIMD3<Float>] {
        mapSourcePixelsToViewport(
            projectToSourcePixels(
                cameraVertices: worldToCamera(
                    worldVertices,
                    metadata: metadata
                ),
                metadata: metadata
            ),
            sourceSize: metadata.sourceSize,
            viewportSize: viewportSize
        )
    }

    static func projectToSourcePixels(
        cameraVertices: [SIMD3<Float>],
        metadata: VideoOverlayMetadata
    ) -> [SIMD3<Float>] {
        guard metadata.isValid else { return [] }
        let width = Float(metadata.sourceSize.width)
        let height = Float(metadata.sourceSize.height)
        let boxScale = Float(metadata.cropSize) * metadata.camera.x
        guard boxScale.isFinite, boxScale > 1e-6 else { return [] }

        // This is HMR2's cam_crop_to_full conversion. Its camera output is
        // [scale, tx, ty] in the square person crop, not in the full video.
        let depthTranslation = 2 * focalLength / boxScale
        let xTranslation = (
            2 * (Float(metadata.cropCenter.x) - width / 2) / boxScale
        ) + metadata.camera.y
        let yTranslation = (
            2 * (Float(metadata.cropCenter.y) - height / 2) / boxScale
        ) + metadata.camera.z

        return cameraVertices.map { vertex in
            let depth = max(vertex.z + depthTranslation, 1e-4)
            let x = focalLength * (vertex.x + xTranslation) / depth + width / 2
            let y = focalLength * (vertex.y + yTranslation) / depth + height / 2
            return SIMD3<Float>(x, y, depth)
        }
    }

    static func aspectFitRect(
        sourceSize: CGSize,
        viewportSize: CGSize
    ) -> CGRect {
        guard sourceSize.width > 0,
              sourceSize.height > 0,
              viewportSize.width > 0,
              viewportSize.height > 0 else {
            return .zero
        }
        let scale = min(
            viewportSize.width / sourceSize.width,
            viewportSize.height / sourceSize.height
        )
        let size = CGSize(
            width: sourceSize.width * scale,
            height: sourceSize.height * scale
        )
        return CGRect(
            x: (viewportSize.width - size.width) / 2,
            y: (viewportSize.height - size.height) / 2,
            width: size.width,
            height: size.height
        )
    }

    static func mapSourcePixelsToViewport(
        _ sourcePixels: [SIMD3<Float>],
        sourceSize: CGSize,
        viewportSize: CGSize
    ) -> [SIMD3<Float>] {
        let videoRect = aspectFitRect(
            sourceSize: sourceSize,
            viewportSize: viewportSize
        )
        guard !videoRect.isEmpty else { return [] }
        let scaleX = Float(videoRect.width / sourceSize.width)
        let scaleY = Float(videoRect.height / sourceSize.height)
        let originX = Float(videoRect.minX)
        let originY = Float(videoRect.minY)
        return sourcePixels.map {
            SIMD3<Float>(
                originX + $0.x * scaleX,
                originY + $0.y * scaleY,
                -$0.z
            )
        }
    }

    private static func worldBodyRotation(
        metadata: VideoOverlayMetadata
    ) -> simd_float3x3? {
        guard let refinedRoot = rotationMatrix6D(metadata.refinedRoot6D),
              let cameraRoot = rotationMatrix6D(metadata.poseRoot6D) else {
            return nil
        }
        return refinedRoot * cameraRoot.transpose
    }

    private static func rotationMatrix6D(
        _ values: [Float]
    ) -> simd_float3x3? {
        guard values.count >= 6 else { return nil }
        let firstRaw = SIMD3<Float>(values[0], values[1], values[2])
        let secondRaw = SIMD3<Float>(values[3], values[4], values[5])
        guard simd_length_squared(firstRaw) > 1e-12 else { return nil }
        let first = simd_normalize(firstRaw)
        let orthogonalSecond = secondRaw - first * simd_dot(first, secondRaw)
        guard simd_length_squared(orthogonalSecond) > 1e-12 else { return nil }
        let second = simd_normalize(orthogonalSecond)
        let third = simd_cross(first, second)

        // WHAM stores the six values as the first two matrix rows. simd's
        // matrix initializer receives columns, so transpose that row layout.
        return simd_float3x3(columns: (
            SIMD3<Float>(first.x, second.x, third.x),
            SIMD3<Float>(first.y, second.y, third.y),
            SIMD3<Float>(first.z, second.z, third.z)
        ))
    }
}

enum VideoOverlayTimeline {
    static func frameIndex(at time: Double, timestamps: [Double]) -> Int {
        guard !timestamps.isEmpty, time.isFinite else { return 0 }
        var lower = 0
        var upper = timestamps.count
        while lower < upper {
            let middle = (lower + upper) / 2
            if timestamps[middle] <= time {
                lower = middle + 1
            } else {
                upper = middle
            }
        }
        return max(0, min(lower - 1, timestamps.count - 1))
    }
}
