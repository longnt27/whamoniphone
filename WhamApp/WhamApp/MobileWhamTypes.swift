import CoreML
import Foundation

struct MobileWhamVisualObservation {
    let keypoints: MLMultiArray
    let keypointMask: MLMultiArray
    let crop: CVPixelBuffer?
}

struct MobileWhamFrameObservation {
    let keypoints: MLMultiArray
    let keypointMask: MLMultiArray
    let imageFeature: MLMultiArray
    let imageFeatureValid: MLMultiArray
    let hmrPose: MLMultiArray?
    let hmrBetas: MLMultiArray?
    let videoTime: Double
}

struct MobileWhamFrontendOutput {
    let token: MLMultiArray
    let pose: MLMultiArray
    let betas: MLMultiArray
}

enum MobileWhamArrays {
    static func zeros(
        _ shape: [NSNumber],
        dataType: MLMultiArrayDataType = .float32
    ) throws -> MLMultiArray {
        try MLMultiArray(shape: shape, dataType: dataType)
    }

    static func scalar(
        _ value: Float,
        dataType: MLMultiArrayDataType = .float32
    ) throws -> MLMultiArray {
        let output = try zeros([1, 1, 1], dataType: dataType)
        output[0] = NSNumber(value: value)
        return output
    }

    static func copy(
        _ source: MLMultiArray,
        shape: [NSNumber],
        dataType: MLMultiArrayDataType
    ) throws -> MLMultiArray {
        let output = try zeros(shape, dataType: dataType)
        guard source.count == output.count else {
            throw MobileWhamError.arraySizeMismatch(source.count, output.count)
        }
        for index in 0..<source.count {
            output[index] = NSNumber(value: source[index].floatValue)
        }
        return output
    }

    static func required(
        _ provider: MLFeatureProvider,
        _ name: String
    ) throws -> MLMultiArray {
        guard let array = provider.featureValue(for: name)?.multiArrayValue else {
            throw MobileWhamError.missingOutput(name)
        }
        return array
    }

    static func predict(
        _ model: MLModel,
        _ values: [String: MLFeatureValue]
    ) throws -> MLFeatureProvider {
        try model.prediction(from: MLDictionaryFeatureProvider(dictionary: values))
    }

    static func poseRoot(_ pose: MLMultiArray) throws -> MLMultiArray {
        guard pose.count >= 6 else {
            throw MobileWhamError.arraySizeMismatch(pose.count, 6)
        }
        let root = try zeros([1, 1, 6], dataType: .float16)
        for index in 0..<6 {
            root[index] = NSNumber(value: pose[index].floatValue)
        }
        return root
    }

    static func initializerInput(
        joints3D: MLMultiArray,
        observation: MLMultiArray
    ) throws -> MLMultiArray {
        guard joints3D.count == 51, observation.count == 37 else {
            throw MobileWhamError.initializationFailed
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

    static func floats(_ array: MLMultiArray, limit: Int? = nil) -> [Float] {
        let count = min(limit ?? array.count, array.count)
        return (0..<count).map { array[$0].floatValue }
    }
}

enum MobileWhamError: LocalizedError {
    case pixelBufferAllocation
    case initializationFailed
    case missingModel(String)
    case missingOutput(String)
    case arraySizeMismatch(Int, Int)

    var errorDescription: String? {
        switch self {
        case .pixelBufferAllocation:
            return "Could not allocate a pipeline pixel buffer"
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
