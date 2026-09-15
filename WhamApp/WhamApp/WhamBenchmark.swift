import CoreImage
import CoreML
import Darwin
import Foundation
import UIKit

struct BenchmarkMetric: Codable, Identifiable {
    var id: String { stage }
    let stage: String
    let samples: Int
    let meanMilliseconds: Double
    let p50Milliseconds: Double
    let p95Milliseconds: Double
}

struct DeviceBenchmarkReport: Codable {
    let schemaVersion: Int
    let createdAt: Date
    let pipeline: String
    let trainingPerformed: Bool
    let device: String
    let operatingSystem: String
    let thermalStateBefore: String
    let thermalStateAfter: String
    let poseDetector: String
    let sampleSet: String
    let sampleFrames: Int
    let passes: Int
    let processedSourceFrames: Int
    let personDetections: Int
    let residentMemoryBeforeBytes: UInt64
    let peakResidentMemoryBytes: UInt64
    let residentMemoryAfterBytes: UInt64
    let selectedPipelineModelBytes: UInt64
    let bundledModelBytes: [String: UInt64]
    let metrics: [BenchmarkMetric]
    let imageFeaturePoseMaxDelta: Float
    let imageFeatureConnectionPassed: Bool
    let outputSmoothingPoseMaxDelta: Float
    let outputSmoothingConnectionPassed: Bool
    let smoothingPoseAlpha: Float
    let smoothingShapeAlpha: Float
    let approximateSmoothingDelayMillisecondsAt30FPS: Double
    let modelProvenance: [String: String]
    let scope: String
    let notes: [String]
}

private struct BenchmarkObservation {
    let keypoints: MLMultiArray
    let keypointMask: MLMultiArray
    let crop: CVPixelBuffer?
}

private struct FrontendOutput {
    let token: MLMultiArray
    let pose: MLMultiArray
    let betas: MLMultiArray
}

@MainActor
final class WhamBenchmark: ObservableObject {
    @Published var isRunning = false
    @Published var status = "Ready"
    @Published var report: DeviceBenchmarkReport?
    @Published var reportURL: URL?

    func run(passes: Int = 5) {
        guard !isRunning else { return }
        isRunning = true
        status = "Running the validated eight-frame pipeline…"

        Task.detached(priority: .userInitiated) {
            do {
                let result = try Self.perform(passes: max(passes, 1))
                let url = try Self.persist(result)
                await MainActor.run {
                    self.report = result
                    self.reportURL = url
                    self.status = "Complete — share the JSON below"
                    self.isRunning = false
                }
            } catch {
                await MainActor.run {
                    self.status = "Failed: \(error.localizedDescription)"
                    self.isRunning = false
                }
            }
        }
    }

    nonisolated private static func perform(passes: Int) throws -> DeviceBenchmarkReport {
        let thermalBefore = thermalStateName(ProcessInfo.processInfo.thermalState)
        let memoryBefore = residentMemoryBytes()
        var peakMemory = memoryBefore
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        let ciContext = CIContext()
        var samples: [String: [Double]] = [:]

        func measured<T>(_ stage: String, _ operation: () throws -> T) rethrows -> T {
            let started = DispatchTime.now().uptimeNanoseconds
            let result = try operation()
            let milliseconds = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
            samples[stage, default: []].append(milliseconds)
            peakMemory = max(peakMemory, residentMemoryBytes())
            return result
        }

        let detector = try measured("load_yolo26m_pose") {
            try PoseDetector(backend: .yolo26m, configuration: configuration)
        }
        let hmr2s = try measured("load_hmr2s_frontend") {
            try loadModel(named: "HMR2SFrontend", configuration: configuration)
        }
        let smplInitializer = try measured("load_hmr2s_smpl_init") {
            try loadModel(named: "HMR2SSMPLInit", configuration: configuration)
        }
        let tokenAdapter = try measured("load_hmr2s_token_adapter") {
            try loadModel(named: "HMR2STokenAdapter", configuration: configuration)
        }
        let whamInitializer = try measured("load_wham_init") {
            try loadModel(named: "WHAM_I", configuration: configuration)
        }
        let whamStep = try measured("load_wham_image_step") {
            try loadModel(named: "WHAM_ImageStep", configuration: configuration)
        }
        let worldStep = try measured("load_wham_world_step") {
            try loadModel(named: "WHAM_WorldStep", configuration: configuration)
        }
        let frames = try loadBenchmarkFrames(ciContext: ciContext)
        guard frames.count >= 2 else { throw BenchmarkError.missingSampleFrames }

        // Warm all execution providers using one untimed real-image track.
        _ = try runTrack(
            frames: frames,
            detector: detector,
            hmr2s: hmr2s,
            smplInitializer: smplInitializer,
            tokenAdapter: tokenAdapter,
            whamInitializer: whamInitializer,
            whamStep: whamStep,
            worldStep: worldStep,
            ciContext: ciContext,
            measured: nil
        )

        var detections = 0
        var comparisonInput: [String: MLFeatureValue]?
        var comparisonToken: MLMultiArray?
        var maximumSmoothingPoseDelta: Float = 0
        for _ in 0..<passes {
            let clipStarted = DispatchTime.now().uptimeNanoseconds
            let track = try runTrack(
                frames: frames,
                detector: detector,
                hmr2s: hmr2s,
                smplInitializer: smplInitializer,
                tokenAdapter: tokenAdapter,
                whamInitializer: whamInitializer,
                whamStep: whamStep,
                worldStep: worldStep,
                ciContext: ciContext,
                measured: { stage, operation in
                    try measured(stage, operation)
                }
            )
            detections += track.detections
            comparisonInput = track.lastWHAMInput
            comparisonToken = track.lastValidToken
            maximumSmoothingPoseDelta = max(
                maximumSmoothingPoseDelta,
                track.maximumSmoothingPoseDelta
            )
            let milliseconds = Double(
                DispatchTime.now().uptimeNanoseconds - clipStarted
            ) / 1_000_000
            samples["end_to_end_8_frame_track", default: []].append(milliseconds)
            samples["amortized_end_to_end_source_frame", default: []].append(
                milliseconds / Double(frames.count)
            )
            peakMemory = max(peakMemory, residentMemoryBytes())
        }

        guard var connectedDictionary = comparisonInput,
              let connectedToken = comparisonToken else {
            throw BenchmarkError.initializationFailed
        }
        connectedDictionary["image_feature_step"] = MLFeatureValue(multiArray: connectedToken)
        connectedDictionary["image_feature_valid_step"] = MLFeatureValue(
            multiArray: try scalar(1, dataType: .float16)
        )
        let connected = try predict(whamStep, connectedDictionary)
        var ablatedDictionary = connectedDictionary
        ablatedDictionary["image_feature_step"] = MLFeatureValue(
            multiArray: try zeros([1, 1, 1024], dataType: .float16)
        )
        ablatedDictionary["image_feature_valid_step"] = MLFeatureValue(
            multiArray: try scalar(0, dataType: .float16)
        )
        let ablated = try predict(whamStep, ablatedDictionary)
        let connectedPose = try requiredArray(connected, "pred_pose")
        let ablatedPose = try requiredArray(ablated, "pred_pose")
        var maxPoseDelta: Float = 0
        for index in 0..<connectedPose.count {
            maxPoseDelta = max(
                maxPoseDelta,
                abs(connectedPose[index].floatValue - ablatedPose[index].floatValue)
            )
        }

        let orderedStages = [
            "load_yolo26m_pose", "load_hmr2s_frontend",
            "load_hmr2s_smpl_init", "load_hmr2s_token_adapter",
            "load_wham_init", "load_wham_image_step",
            "load_wham_world_step",
            "yolo26m_pose_and_parse", "crop_and_keypoint_normalization",
            "hmr2s_frontend", "hmr2s_token_adapter",
            "hmr2s_first_frame_smpl_init", "wham_init",
            "wham_recurrent_step", "temporal_output_smoothing",
            "wham_smpl_world_refiner_step",
            "end_to_end_8_frame_track",
            "amortized_end_to_end_source_frame"
        ]
        let metrics = orderedStages.compactMap { stage -> BenchmarkMetric? in
            guard let values = samples[stage], !values.isEmpty else { return nil }
            let sorted = values.sorted()
            return BenchmarkMetric(
                stage: stage,
                samples: values.count,
                meanMilliseconds: values.reduce(0, +) / Double(values.count),
                p50Milliseconds: quantile(sorted, 0.50),
                p95Milliseconds: quantile(sorted, 0.95)
            )
        }
        let modelNames = [
            "yolo26m-pose", "HMR2SFrontend", "HMR2SSMPLInit",
            "HMR2STokenAdapter", "WHAM_I", "WHAM_ImageStep", "WHAM_WorldStep"
        ]
        let modelBytes = Dictionary(uniqueKeysWithValues: modelNames.compactMap { name in
            bundledModelSizeBytes(name).map { (name, $0) }
        })

        return DeviceBenchmarkReport(
            schemaVersion: 8,
            createdAt: Date(),
            pipeline: "YOLO26m-pose → released HMR2.0-S → selected token adapter → released split WHAM → light causal smoothing",
            trainingPerformed: false,
            device: "iOS device (" + deviceIdentifier() + ")",
            operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
            thermalStateBefore: thermalBefore,
            thermalStateAfter: thermalStateName(ProcessInfo.processInfo.thermalState),
            poseDetector: PoseDetector.Backend.yolo26m.rawValue,
            sampleSet: "8 frames sampled from the official WHAM IMG_9730 example",
            sampleFrames: frames.count,
            passes: passes,
            processedSourceFrames: frames.count * passes,
            personDetections: detections,
            residentMemoryBeforeBytes: memoryBefore,
            peakResidentMemoryBytes: peakMemory,
            residentMemoryAfterBytes: residentMemoryBytes(),
            selectedPipelineModelBytes: modelBytes.values.reduce(0, +),
            bundledModelBytes: modelBytes,
            metrics: metrics,
            imageFeaturePoseMaxDelta: maxPoseDelta,
            imageFeatureConnectionPassed: maxPoseDelta > 0.00001,
            outputSmoothingPoseMaxDelta: maximumSmoothingPoseDelta,
            outputSmoothingConnectionPassed: maximumSmoothingPoseDelta > 0.00001,
            smoothingPoseAlpha: TemporalOutputSmoother.selectedPoseAlpha,
            smoothingShapeAlpha: TemporalOutputSmoother.selectedShapeAlpha,
            approximateSmoothingDelayMillisecondsAt30FPS: 11.11111111111111,
            modelProvenance: pipelineProvenance(hmr2s, tokenAdapter),
            scope: "Validation-selected end-to-end learned pipeline through SMPL mesh generation, contact-aware trajectory refinement, world-space rollout, and causal output smoothing. Static test images use zero camera angular velocity; accuracy is measured separately on 3DPW.",
            notes: [
                "Run a Release build on a physical iPhone with Low Power Mode off.",
                "One untimed full-track warm-up precedes the measured passes.",
                "HMR2.0-S supplies both the real 1024-D feature and real first-frame SMPL/3D-joint initialization; there is no neutral or zero initializer fallback.",
                "The 3DPW-validation-selected residual adapter maps every recurrent HMR2-S token into WHAM's expected HMR2a token space.",
                "The image-feature A/B guard holds motion and recurrent state fixed and must change WHAM pose output.",
                "Light causal geodesic pose smoothing (alpha 0.75) and shape EMA (alpha 0.35) are applied after WHAM and never fed back into its recurrent state.",
                "The bundled still frames have no gyroscope stream, so cam_a_step is zero. Its tensor path still runs with the same shape.",
                "WHAM_WorldStep emits the 6,890-vertex world-space SMPL mesh and keeps trajectory-refiner state explicit, preventing Core ML from unrolling a full clip.",
                "The Kaggle report supplies camera-relative accuracy; this report supplies physical-iPhone latency, model bytes, memory, and thermal state."
            ]
        )
    }

    private struct TrackResult {
        let detections: Int
        let lastWHAMInput: [String: MLFeatureValue]?
        let lastValidToken: MLMultiArray?
        let maximumSmoothingPoseDelta: Float
    }

    private typealias Measurement = (
        _ stage: String,
        _ operation: () throws -> Any
    ) throws -> Any

    nonisolated private static func runTrack(
        frames: [CVPixelBuffer],
        detector: PoseDetector,
        hmr2s: MLModel,
        smplInitializer: MLModel,
        tokenAdapter: MLModel,
        whamInitializer: MLModel,
        whamStep: MLModel,
        worldStep: MLModel,
        ciContext: CIContext,
        measured: Measurement?
    ) throws -> TrackResult {
        func run<T>(_ stage: String, _ operation: () throws -> T) throws -> T {
            guard let measured else { return try operation() }
            guard let value = try measured(stage, operation) as? T else {
                throw BenchmarkError.measurementTypeMismatch(stage)
            }
            return value
        }

        var detections = 0
        var previousKeypoints: MLMultiArray?
        var previousRoot: MLMultiArray?
        var previousPose: MLMultiArray?
        var hEnc: MLMultiArray?
        var cEnc: MLMultiArray?
        var hTraj: MLMultiArray?
        var cTraj: MLMultiArray?
        var hDec: MLMultiArray?
        var cDec: MLMultiArray?
        var lastInput: [String: MLFeatureValue]?
        var lastValidToken: MLMultiArray?
        var maximumSmoothingPoseDelta: Float = 0
        var smoother = TemporalOutputSmoother()
        var previousUnrefinedRoot: MLMultiArray?
        var previousUnrefinedTranslation = try zeros([1, 1, 3], dataType: .float16)
        var previousBodyFeet = try zeros([1, 1, 4, 3], dataType: .float16)
        var previousWorldFeet = try zeros([1, 1, 4, 3], dataType: .float16)
        var previousRefinedRoot: MLMultiArray?
        var previousRefinedTranslation = try zeros([1, 1, 3], dataType: .float16)
        var hRefiner = try zeros([2, 1, 512], dataType: .float16)
        var cRefiner = try zeros([2, 1, 512], dataType: .float16)
        var hasPreviousWorldFrame: Float = 0

        for (frameIndex, frame) in frames.enumerated() {
            let detection: PoseDetector.Detection? = try run("yolo26m_pose_and_parse") {
                detector.detect(pixelBuffer: frame).first
            }
            if detection != nil { detections += 1 }
            if frameIndex == 0, detection == nil {
                throw BenchmarkError.noPersonInWarmupFrame
            }
            let observation: BenchmarkObservation = try run(
                "crop_and_keypoint_normalization"
            ) {
                try makeObservation(
                    detection: detection,
                    source: frame,
                    ciContext: ciContext
                )
            }

            var frontend: FrontendOutput?
            if let crop = observation.crop {
                let provider: MLFeatureProvider = try run("hmr2s_frontend") {
                    try predict(hmr2s, [
                        "image_input": MLFeatureValue(pixelBuffer: crop)
                    ])
                }
                frontend = FrontendOutput(
                    token: try requiredArray(provider, "image_token"),
                    pose: try requiredArray(provider, "pose_6d"),
                    betas: try requiredArray(provider, "betas")
                )
            }

            if frameIndex == 0 {
                guard let frontend else { throw BenchmarkError.initializationFailed }
                let smpl: MLFeatureProvider = try run("hmr2s_first_frame_smpl_init") {
                    try predict(smplInitializer, [
                        "pose_6d": MLFeatureValue(multiArray: frontend.pose),
                        "betas": MLFeatureValue(multiArray: frontend.betas)
                    ])
                }
                let initial3D = try requiredArray(smpl, "init_kp3d")
                let initialKeypoints = try concatenateInitializer(
                    joints3D: initial3D,
                    observation: observation.keypoints
                )
                let initialPose = try copyArray(
                    frontend.pose,
                    shape: [1, 1, 144],
                    dataType: .float32
                )
                let initial: MLFeatureProvider = try run("wham_init") {
                    try predict(whamInitializer, [
                        "init_kp": MLFeatureValue(multiArray: initialKeypoints),
                        "init_smpl": MLFeatureValue(multiArray: initialPose)
                    ])
                }
                previousKeypoints = try copyArray(
                    initial3D, shape: [1, 1, 51], dataType: .float16
                )
                previousRoot = try poseRoot(frontend.pose)
                previousUnrefinedRoot = previousRoot
                previousRefinedRoot = previousRoot
                previousPose = try copyArray(
                    frontend.pose, shape: [1, 1, 144], dataType: .float16
                )
                hEnc = try copyArray(
                    requiredArray(initial, "h_enc"),
                    shape: [3, 1, 512],
                    dataType: .float16
                )
                cEnc = try copyArray(
                    requiredArray(initial, "c_enc"),
                    shape: [3, 1, 512],
                    dataType: .float16
                )
                hTraj = try copyArray(
                    requiredArray(initial, "h_traj"),
                    shape: [3, 1, 563],
                    dataType: .float16
                )
                cTraj = try copyArray(
                    requiredArray(initial, "c_traj"),
                    shape: [3, 1, 563],
                    dataType: .float16
                )
                hDec = try copyArray(
                    requiredArray(initial, "h_dec"),
                    shape: [3, 1, 563],
                    dataType: .float16
                )
                cDec = try copyArray(
                    requiredArray(initial, "c_dec"),
                    shape: [3, 1, 563],
                    dataType: .float16
                )
                continue
            }

            guard let previousKeypointsValue = previousKeypoints,
                  let previousRootValue = previousRoot,
                  let previousPoseValue = previousPose,
                  let hEncValue = hEnc,
                  let cEncValue = cEnc,
                  let hTrajValue = hTraj,
                  let cTrajValue = cTraj,
                  let hDecValue = hDec,
                  let cDecValue = cDec else {
                throw BenchmarkError.initializationFailed
            }
            let feature: MLMultiArray
            if let token = frontend?.token {
                let adapted: MLFeatureProvider = try run("hmr2s_token_adapter") {
                    try predict(tokenAdapter, [
                        "hmr2s_token": MLFeatureValue(multiArray: token)
                    ])
                }
                feature = try requiredArray(adapted, "wham_image_token")
                lastValidToken = feature
            } else {
                feature = try zeros([1, 1, 1024], dataType: .float16)
            }
            let dictionary: [String: MLFeatureValue] = [
                "x_step": MLFeatureValue(multiArray: observation.keypoints),
                "keypoint_mask_step": MLFeatureValue(multiArray: observation.keypointMask),
                "image_feature_step": MLFeatureValue(multiArray: feature),
                "image_feature_valid_step": MLFeatureValue(
                    multiArray: try scalar(frontend == nil ? 0 : 1, dataType: .float16)
                ),
                "cam_a_step": MLFeatureValue(
                    multiArray: try zeros([1, 1, 6], dataType: .float16)
                ),
                "prev_kp3d": MLFeatureValue(multiArray: previousKeypointsValue),
                "prev_root": MLFeatureValue(multiArray: previousRootValue),
                "prev_pose": MLFeatureValue(multiArray: previousPoseValue),
                "h_enc_in": MLFeatureValue(multiArray: hEncValue),
                "c_enc_in": MLFeatureValue(multiArray: cEncValue),
                "h_traj_in": MLFeatureValue(multiArray: hTrajValue),
                "c_traj_in": MLFeatureValue(multiArray: cTrajValue),
                "h_dec_in": MLFeatureValue(multiArray: hDecValue),
                "c_dec_in": MLFeatureValue(multiArray: cDecValue)
            ]
            if frontend != nil {
                lastInput = dictionary
            }
            let output: MLFeatureProvider = try run("wham_recurrent_step") {
                try predict(whamStep, dictionary)
            }
            let rawPose = try requiredArray(output, "pred_pose")
            let rawShape = try requiredArray(output, "pred_shape")
            let smoothed: (pose: MLMultiArray, shape: MLMultiArray) = try run(
                "temporal_output_smoothing"
            ) {
                try smoother.smooth(pose: rawPose, shape: rawShape)
            }
            for valueIndex in 0..<rawPose.count {
                maximumSmoothingPoseDelta = max(
                    maximumSmoothingPoseDelta,
                    abs(rawPose[valueIndex].floatValue - smoothed.pose[valueIndex].floatValue)
                )
            }
            guard let previousUnrefinedRootValue = previousUnrefinedRoot,
                  let previousRefinedRootValue = previousRefinedRoot else {
                throw BenchmarkError.initializationFailed
            }
            let world: MLFeatureProvider = try run("wham_smpl_world_refiner_step") {
                try predict(worldStep, [
                    "pred_pose": MLFeatureValue(
                        multiArray: smoothed.pose
                    ),
                    "pred_shape": MLFeatureValue(
                        multiArray: smoothed.shape
                    ),
                    "pred_contact": MLFeatureValue(
                        multiArray: try requiredArray(output, "pred_contact")
                    ),
                    "pred_root": MLFeatureValue(
                        multiArray: try requiredArray(output, "pred_root")
                    ),
                    "pred_vel": MLFeatureValue(
                        multiArray: try requiredArray(output, "pred_vel")
                    ),
                    "pred_kp3d": MLFeatureValue(
                        multiArray: try requiredArray(output, "pred_kp3d")
                    ),
                    "h_enc_out": MLFeatureValue(
                        multiArray: try requiredArray(output, "h_enc_out")
                    ),
                    "prev_unrefined_root": MLFeatureValue(
                        multiArray: previousUnrefinedRootValue
                    ),
                    "prev_unrefined_translation": MLFeatureValue(
                        multiArray: previousUnrefinedTranslation
                    ),
                    "prev_body_feet": MLFeatureValue(multiArray: previousBodyFeet),
                    "prev_world_feet": MLFeatureValue(multiArray: previousWorldFeet),
                    "prev_refined_root": MLFeatureValue(
                        multiArray: previousRefinedRootValue
                    ),
                    "prev_refined_translation": MLFeatureValue(
                        multiArray: previousRefinedTranslation
                    ),
                    "h_refiner_in": MLFeatureValue(multiArray: hRefiner),
                    "c_refiner_in": MLFeatureValue(multiArray: cRefiner),
                    "has_previous": MLFeatureValue(
                        multiArray: try scalar(hasPreviousWorldFrame, dataType: .float16)
                    )
                ])
            }
            // Touch the final mesh output so this benchmark validates the public contract.
            _ = try requiredArray(world, "vertices_world")[0].floatValue
            previousUnrefinedRoot = try requiredArray(world, "unrefined_root")
            previousUnrefinedTranslation = try requiredArray(
                world, "unrefined_translation"
            )
            previousBodyFeet = try requiredArray(world, "body_feet")
            previousWorldFeet = try requiredArray(world, "world_feet")
            previousRefinedRoot = try requiredArray(world, "refined_root")
            previousRefinedTranslation = try requiredArray(
                world, "refined_translation"
            )
            hRefiner = try requiredArray(world, "h_refiner_out")
            cRefiner = try requiredArray(world, "c_refiner_out")
            hasPreviousWorldFrame = 1
            previousKeypoints = try requiredArray(output, "pred_kp3d")
            previousRoot = try requiredArray(output, "pred_root")
            // Preserve the validated protocol: smoothing is output-only.
            previousPose = rawPose
            hEnc = try requiredArray(output, "h_enc_out")
            cEnc = try requiredArray(output, "c_enc_out")
            hTraj = try requiredArray(output, "h_traj_out")
            cTraj = try requiredArray(output, "c_traj_out")
            hDec = try requiredArray(output, "h_dec_out")
            cDec = try requiredArray(output, "c_dec_out")
        }
        return TrackResult(
            detections: detections,
            lastWHAMInput: lastInput,
            lastValidToken: lastValidToken,
            maximumSmoothingPoseDelta: maximumSmoothingPoseDelta
        )
    }

    nonisolated private static func loadModel(
        named name: String,
        configuration: MLModelConfiguration
    ) throws -> MLModel {
        guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {
            throw BenchmarkError.missingModel(name)
        }
        return try MLModel(contentsOf: url, configuration: configuration)
    }

    nonisolated private static func predict(
        _ model: MLModel,
        _ values: [String: MLFeatureValue]
    ) throws -> MLFeatureProvider {
        try model.prediction(from: MLDictionaryFeatureProvider(dictionary: values))
    }

    nonisolated private static func requiredArray(
        _ provider: MLFeatureProvider,
        _ name: String
    ) throws -> MLMultiArray {
        guard let array = provider.featureValue(for: name)?.multiArrayValue else {
            throw BenchmarkError.missingOutput(name)
        }
        return array
    }

    nonisolated private static func copyArray(
        _ source: MLMultiArray,
        shape: [NSNumber],
        dataType: MLMultiArrayDataType
    ) throws -> MLMultiArray {
        let output = try zeros(shape, dataType: dataType)
        guard output.count == source.count else {
            throw BenchmarkError.arraySizeMismatch(source.count, output.count)
        }
        for index in 0..<source.count {
            output[index] = NSNumber(value: source[index].floatValue)
        }
        return output
    }

    nonisolated private static func poseRoot(_ pose: MLMultiArray) throws -> MLMultiArray {
        guard pose.count >= 6 else {
            throw BenchmarkError.arraySizeMismatch(pose.count, 6)
        }
        let root = try zeros([1, 1, 6], dataType: .float16)
        for index in 0..<6 { root[index] = NSNumber(value: pose[index].floatValue) }
        return root
    }

    nonisolated private static func concatenateInitializer(
        joints3D: MLMultiArray,
        observation: MLMultiArray
    ) throws -> MLMultiArray {
        guard joints3D.count == 51, observation.count == 37 else {
            throw BenchmarkError.initializationFailed
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

    nonisolated private static func pipelineProvenance(
        _ hmr2sModel: MLModel,
        _ adapterModel: MLModel
    ) -> [String: String] {
        let metadata = hmr2sModel.modelDescription.metadata[
            MLModelMetadataKey.creatorDefinedKey
        ]
            as? [String: String]
        let adapterMetadata = adapterModel.modelDescription.metadata[
            MLModelMetadataKey.creatorDefinedKey
        ] as? [String: String]
        return [
            "yolo26_weights_sha256": "2fbf16367022256a226035695c5c389384c6706e8bb8ab8fcd0e7976f05443c4",
            "yolo26_variant": "yolo26m-pose",
            "hmr2s_repository": metadata?["source_repository"] ?? "missing",
            "hmr2s_commit": metadata?["source_commit"] ?? "missing",
            "hmr2s_checkpoint_sha256": metadata?["checkpoint_sha256"] ?? "missing",
            "token_adapter_checkpoint_sha256": adapterMetadata?["checkpoint_sha256"] ?? "missing",
            "wham_repository": "https://github.com/yohanshin/WHAM.git",
            "wham_commit": "2b54f7797391c94876848b905ed875b154c4a295",
            "wham_checkpoint_sha256": "2ba0cb6a7dd597023a6b2ad6056e7a8b6b33144a35fabea570bfd00842cd4eaf",
            "training_performed_by_this_project": metadata?["training_performed_by_this_project"] ?? "missing"
        ]
    }

    nonisolated private static func loadBenchmarkFrames(
        ciContext: CIContext
    ) throws -> [CVPixelBuffer] {
        var urls = Bundle.main.urls(
            forResourcesWithExtension: "jpg",
            subdirectory: "BenchmarkFrames"
        ) ?? []
        if urls.isEmpty {
            urls = (Bundle.main.urls(forResourcesWithExtension: "jpg", subdirectory: nil) ?? [])
                .filter { $0.lastPathComponent.hasPrefix("frame_") }
        }
        return try urls.sorted { $0.lastPathComponent < $1.lastPathComponent }.map { url in
            guard let image = UIImage(contentsOfFile: url.path),
                  let cgImage = image.cgImage,
                  let buffer = makePixelBuffer(width: cgImage.width, height: cgImage.height) else {
                throw BenchmarkError.invalidSampleFrame(url.lastPathComponent)
            }
            ciContext.render(CIImage(cgImage: cgImage), to: buffer)
            return buffer
        }
    }

    nonisolated private static func makeObservation(
        detection: PoseDetector.Detection?,
        source: CVPixelBuffer,
        ciContext: CIContext
    ) throws -> BenchmarkObservation {
        guard let detection else {
            let mask = try zeros([1, 1, 17], dataType: .float16)
            for index in 0..<17 { mask[index] = 1 }
            return BenchmarkObservation(
                keypoints: try zeros([1, 1, 37], dataType: .float16),
                keypointMask: mask,
                crop: nil
            )
        }

        let width = CGFloat(CVPixelBufferGetWidth(source))
        let height = CGFloat(CVPixelBufferGetHeight(source))
        let box = squarePersonBox(detection, imageWidth: width, imageHeight: height)
        let keypoints = try zeros([1, 1, 37], dataType: .float16)
        let mask = try zeros([1, 1, 17], dataType: .float16)
        for index in 0..<min(detection.keypoints.count, 17) {
            let point = detection.keypoints[index]
            keypoints[index * 2] = NSNumber(
                value: Float(2 * (point.x * width - box.midX) / box.width)
            )
            keypoints[index * 2 + 1] = NSNumber(
                value: Float(2 * (point.y * height - box.midY) / box.height)
            )
            let confidence = index < detection.keypointConfidences.count
                ? detection.keypointConfidences[index]
                : 0
            mask[index] = NSNumber(value: confidence < 0.3 ? 1 : 0)
        }
        let longest = max(width, height)
        keypoints[34] = NSNumber(value: Float(2 * box.midX / longest - width / longest))
        keypoints[35] = NSNumber(value: Float(2 * box.midY / longest - height / longest))
        keypoints[36] = NSNumber(value: Float(box.width / longest))

        let coreImageBox = CGRect(
            x: box.minX,
            y: height - box.maxY,
            width: box.width,
            height: box.height
        )
        let cropped = CIImage(cvPixelBuffer: source).cropped(to: coreImageBox)
        let translated = cropped.transformed(by: CGAffineTransform(
            translationX: -coreImageBox.minX,
            y: -coreImageBox.minY
        ))
        let resized = translated.transformed(by: CGAffineTransform(
            scaleX: 256 / box.width,
            y: 256 / box.height
        ))
        let black = CIImage(color: .black).cropped(
            to: CGRect(x: 0, y: 0, width: 256, height: 256)
        )
        guard let crop = makePixelBuffer(width: 256, height: 256) else {
            throw BenchmarkError.pixelBufferAllocation
        }
        ciContext.render(resized.composited(over: black), to: crop)
        return BenchmarkObservation(
            keypoints: keypoints,
            keypointMask: mask,
            crop: crop
        )
    }

    nonisolated private static func squarePersonBox(
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
            side = max(
                max(
                    detection.box.width * imageWidth,
                    detection.box.height * imageHeight
                ) * 1.05,
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

    nonisolated private static func persist(_ report: DeviceBenchmarkReport) throws -> URL {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(report)
        let documents = FileManager.default.urls(
            for: .documentDirectory,
            in: .userDomainMask
        )[0]
        let url = documents.appendingPathComponent(
            "selected_mobile_pipeline_device_benchmark.json"
        )
        try data.write(to: url, options: .atomic)
        return url
    }

    nonisolated private static func zeros(
        _ shape: [NSNumber],
        dataType: MLMultiArrayDataType
    ) throws -> MLMultiArray {
        try MLMultiArray(shape: shape, dataType: dataType)
    }

    nonisolated private static func scalar(
        _ value: Float,
        dataType: MLMultiArrayDataType
    ) throws -> MLMultiArray {
        let output = try zeros([1, 1, 1], dataType: dataType)
        output[0] = NSNumber(value: value)
        return output
    }

    nonisolated private static func makePixelBuffer(
        width: Int,
        height: Int
    ) -> CVPixelBuffer? {
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

    nonisolated private static func quantile(
        _ sorted: [Double],
        _ q: Double
    ) -> Double {
        guard sorted.count > 1 else { return sorted.first ?? 0 }
        let position = q * Double(sorted.count - 1)
        let lower = Int(position.rounded(.down))
        let upper = Int(position.rounded(.up))
        let fraction = position - Double(lower)
        return sorted[lower] * (1 - fraction) + sorted[upper] * fraction
    }

    nonisolated private static func residentMemoryBytes() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(
                    mach_task_self_,
                    task_flavor_t(MACH_TASK_BASIC_INFO),
                    $0,
                    &count
                )
            }
        }
        return result == KERN_SUCCESS ? UInt64(info.resident_size) : 0
    }

    nonisolated private static func bundledModelSizeBytes(_ name: String) -> UInt64? {
        guard let modelURL = Bundle.main.url(
            forResource: name,
            withExtension: "mlmodelc"
        ) else { return nil }
        let keys: Set<URLResourceKey> = [.isRegularFileKey, .fileSizeKey]
        guard let enumerator = FileManager.default.enumerator(
            at: modelURL,
            includingPropertiesForKeys: Array(keys),
            options: [.skipsHiddenFiles]
        ) else { return nil }
        var bytes: UInt64 = 0
        for case let fileURL as URL in enumerator {
            guard let values = try? fileURL.resourceValues(forKeys: keys),
                  values.isRegularFile == true else { continue }
            bytes += UInt64(values.fileSize ?? 0)
        }
        return bytes
    }

    nonisolated private static func deviceIdentifier() -> String {
        var system = utsname()
        uname(&system)
        return withUnsafePointer(to: &system.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) {
                String(cString: $0)
            }
        }
    }

    nonisolated private static func thermalStateName(
        _ state: ProcessInfo.ThermalState
    ) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    private enum BenchmarkError: LocalizedError {
        case pixelBufferAllocation
        case missingSampleFrames
        case invalidSampleFrame(String)
        case noPersonInWarmupFrame
        case initializationFailed
        case missingModel(String)
        case missingOutput(String)
        case arraySizeMismatch(Int, Int)
        case measurementTypeMismatch(String)

        var errorDescription: String? {
            switch self {
            case .pixelBufferAllocation:
                return "Could not allocate benchmark pixel buffers"
            case .missingSampleFrames:
                return "Bundled BenchmarkFrames were not found"
            case .invalidSampleFrame(let name):
                return "Could not decode benchmark frame \(name)"
            case .noPersonInWarmupFrame:
                return "YOLO26 did not detect a person in the initializer frame"
            case .initializationFailed:
                return "The HMR2.0-S/WHAM recurrent state was not initialized"
            case .missingModel(let name):
                return "Bundled Core ML model \(name).mlmodelc was not found"
            case .missingOutput(let name):
                return "Required Core ML output \(name) was not found"
            case .arraySizeMismatch(let source, let destination):
                return "Core ML array size mismatch: \(source) → \(destination)"
            case .measurementTypeMismatch(let stage):
                return "Internal benchmark measurement mismatch at \(stage)"
            }
        }
    }
}
