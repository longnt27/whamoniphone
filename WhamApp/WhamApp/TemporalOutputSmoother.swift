import CoreML
import Foundation
import simd

/// Validation-selected causal output filter. It is deliberately applied after
/// WHAM and never fed back into WHAM's recurrent state.
struct TemporalOutputSmoother {
    static let selectedPoseAlpha: Float = 0.75
    static let selectedShapeAlpha: Float = 0.35

    let poseAlpha: Float
    let shapeAlpha: Float

    private var previousRotations: [simd_quatf]?
    private var previousShape: [Float]?

    init(
        poseAlpha: Float = Self.selectedPoseAlpha,
        shapeAlpha: Float = Self.selectedShapeAlpha
    ) {
        self.poseAlpha = poseAlpha
        self.shapeAlpha = shapeAlpha
    }

    mutating func reset() {
        previousRotations = nil
        previousShape = nil
    }

    mutating func smooth(
        pose: MLMultiArray,
        shape: MLMultiArray
    ) throws -> (pose: MLMultiArray, shape: MLMultiArray) {
        guard pose.count == 24 * 6, shape.count == 10 else {
            throw SmoothingError.invalidShape(pose.count, shape.count)
        }
        let poseOutput = try MLMultiArray(shape: pose.shape, dataType: pose.dataType)
        let shapeOutput = try MLMultiArray(shape: shape.shape, dataType: shape.dataType)

        let currentRotations = (0..<24).map { rotation(from: pose, joint: $0) }
        let filteredRotations: [simd_quatf]
        if let previousRotations {
            filteredRotations = zip(previousRotations, currentRotations).map {
                simd_normalize(simd_slerp($0.0, $0.1, poseAlpha))
            }
        } else {
            filteredRotations = currentRotations
        }
        for (joint, quaternion) in filteredRotations.enumerated() {
            write(rotation: quaternion, joint: joint, to: poseOutput)
        }
        previousRotations = filteredRotations

        let currentShape = (0..<10).map { shape[$0].floatValue }
        let filteredShape: [Float]
        if let previousShape {
            filteredShape = zip(previousShape, currentShape).map {
                shapeAlpha * $0.1 + (1 - shapeAlpha) * $0.0
            }
        } else {
            filteredShape = currentShape
        }
        for index in 0..<10 {
            shapeOutput[index] = NSNumber(value: filteredShape[index])
        }
        previousShape = filteredShape
        return (poseOutput, shapeOutput)
    }

    private func rotation(from pose: MLMultiArray, joint: Int) -> simd_quatf {
        let offset = joint * 6
        let firstRaw = SIMD3<Float>(
            pose[offset].floatValue,
            pose[offset + 1].floatValue,
            pose[offset + 2].floatValue
        )
        let secondRaw = SIMD3<Float>(
            pose[offset + 3].floatValue,
            pose[offset + 4].floatValue,
            pose[offset + 5].floatValue
        )
        let first = safeNormalize(firstRaw, fallback: SIMD3<Float>(1, 0, 0))
        let projected = secondRaw - simd_dot(first, secondRaw) * first
        let second = safeNormalize(projected, fallback: SIMD3<Float>(0, 1, 0))
        let third = simd_cross(first, second)
        let matrix = simd_float3x3(columns: (
            SIMD3<Float>(first.x, second.x, third.x),
            SIMD3<Float>(first.y, second.y, third.y),
            SIMD3<Float>(first.z, second.z, third.z)
        ))
        return simd_normalize(simd_quatf(matrix))
    }

    private func safeNormalize(
        _ value: SIMD3<Float>,
        fallback: SIMD3<Float>
    ) -> SIMD3<Float> {
        let magnitude = simd_length(value)
        return magnitude > 1e-8 ? value / magnitude : fallback
    }

    private func write(
        rotation: simd_quatf,
        joint: Int,
        to pose: MLMultiArray
    ) {
        let matrix = simd_float3x3(rotation)
        let first = SIMD3<Float>(matrix[0][0], matrix[1][0], matrix[2][0])
        let second = SIMD3<Float>(matrix[0][1], matrix[1][1], matrix[2][1])
        let values = [first.x, first.y, first.z, second.x, second.y, second.z]
        for component in 0..<6 {
            pose[joint * 6 + component] = NSNumber(value: values[component])
        }
    }

    enum SmoothingError: LocalizedError {
        case invalidShape(Int, Int)

        var errorDescription: String? {
            switch self {
            case .invalidShape(let poseCount, let shapeCount):
                return "Expected 144 pose and 10 shape values; got \(poseCount) and \(shapeCount)"
            }
        }
    }
}
