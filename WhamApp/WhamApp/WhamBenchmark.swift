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
            try MobileWhamModelCatalog.load(
                MobileWhamModelCatalog.hmrFrontend,
                configuration: configuration
            )
        }
        let smplInitializer = try measured("load_hmr2s_smpl_init") {
            try MobileWhamModelCatalog.load(
                MobileWhamModelCatalog.smplInitializer,
                configuration: configuration
            )
        }
        let tokenAdapter = try measured("load_hmr2s_token_adapter") {
            try MobileWhamModelCatalog.load(
                MobileWhamModelCatalog.tokenAdapter,
                configuration: configuration
            )
        }
        let whamInitializer = try measured("load_wham_init") {
            try MobileWhamModelCatalog.load(
                MobileWhamModelCatalog.whamInitializer,
                configuration: configuration
            )
        }
        let whamStep = try measured("load_wham_image_step") {
            try MobileWhamModelCatalog.load(
                MobileWhamModelCatalog.whamStep,
                configuration: configuration
            )
        }
        let worldStep = try measured("load_wham_world_step") {
            try MobileWhamModelCatalog.load(
                MobileWhamModelCatalog.worldStep,
                configuration: configuration
            )
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
            multiArray: try MobileWhamArrays.scalar(1, dataType: .float16)
        )
        let connected = try MobileWhamArrays.predict(whamStep, connectedDictionary)
        var ablatedDictionary = connectedDictionary
        ablatedDictionary["image_feature_step"] = MLFeatureValue(
            multiArray: try MobileWhamArrays.zeros(
                [1, 1, 1024],
                dataType: .float16
            )
        )
        ablatedDictionary["image_feature_valid_step"] = MLFeatureValue(
            multiArray: try MobileWhamArrays.scalar(0, dataType: .float16)
        )
        let ablated = try MobileWhamArrays.predict(whamStep, ablatedDictionary)
        let connectedPose = try MobileWhamArrays.required(connected, "pred_pose")
        let ablatedPose = try MobileWhamArrays.required(ablated, "pred_pose")
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
        let modelBytes = Dictionary(
            uniqueKeysWithValues: MobileWhamModelCatalog.resourceNames.compactMap { name in
                MobileWhamModelCatalog.bundledModelSizeBytes(name).map { (name, $0) }
            }
        )

        return DeviceBenchmarkReport(
            schemaVersion: 8,
            createdAt: Date(),
            pipeline: MobileWhamModelCatalog.pipelineDescription,
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
            modelProvenance: MobileWhamModelCatalog.provenance(
                hmrFrontendModel: hmr2s,
                adapterModel: tokenAdapter
            ),
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
        var coreState: MobileWhamCore.State?
        var lastInput: [String: MLFeatureValue]?
        var lastValidToken: MLMultiArray?
        var maximumSmoothingPoseDelta: Float = 0
        let preprocessor = MobileWhamPreprocessor(ciContext: ciContext)
        let core = MobileWhamCore(
            initializer: whamInitializer,
            recurrentStep: whamStep,
            worldStep: worldStep
        )

        for (frameIndex, frame) in frames.enumerated() {
            let detection: PoseDetector.Detection? = try run("yolo26m_pose_and_parse") {
                detector.detect(pixelBuffer: frame).first
            }
            if detection != nil { detections += 1 }
            if frameIndex == 0, detection == nil {
                throw BenchmarkError.noPersonInWarmupFrame
            }
            let visual: MobileWhamVisualObservation = try run(
                "crop_and_keypoint_normalization"
            ) {
                try preprocessor.visualObservation(
                    detection: detection,
                    source: frame
                )
            }

            var frontend: MobileWhamFrontendOutput?
            if let crop = visual.crop {
                let provider: MLFeatureProvider = try run("hmr2s_frontend") {
                    try MobileWhamArrays.predict(hmr2s, [
                        "image_input": MLFeatureValue(pixelBuffer: crop)
                    ])
                }
                frontend = MobileWhamFrontendOutput(
                    token: try MobileWhamArrays.required(provider, "image_token"),
                    pose: try MobileWhamArrays.required(provider, "pose_6d"),
                    betas: try MobileWhamArrays.required(provider, "betas")
                )
            }

            if frameIndex == 0 {
                guard let frontend else { throw BenchmarkError.initializationFailed }
                let smpl: MLFeatureProvider = try run("hmr2s_first_frame_smpl_init") {
                    try MobileWhamArrays.predict(smplInitializer, [
                        "pose_6d": MLFeatureValue(multiArray: frontend.pose),
                        "betas": MLFeatureValue(multiArray: frontend.betas)
                    ])
                }
                let initial3D = try MobileWhamArrays.required(smpl, "init_kp3d")
                coreState = try core.initialize(
                    firstPose: frontend.pose,
                    initialJoints3D: initial3D,
                    firstKeypoints: visual.keypoints,
                    predict: { stage, model, values in
                        try run(stage) {
                            try MobileWhamArrays.predict(model, values)
                        }
                    }
                )
                continue
            }

            guard var state = coreState else {
                throw BenchmarkError.initializationFailed
            }
            let feature: MLMultiArray
            if let token = frontend?.token {
                let adapted: MLFeatureProvider = try run("hmr2s_token_adapter") {
                    try MobileWhamArrays.predict(tokenAdapter, [
                        "hmr2s_token": MLFeatureValue(multiArray: token)
                    ])
                }
                feature = try MobileWhamArrays.required(
                    adapted,
                    "wham_image_token"
                )
                lastValidToken = feature
            } else {
                feature = try MobileWhamArrays.zeros(
                    [1, 1, 1024],
                    dataType: .float16
                )
            }
            let observation = try MobileWhamPreprocessor.frame(
                visual: visual,
                frontend: frontend,
                adaptedFeature: frontend == nil ? nil : feature,
                videoTime: Double(frameIndex)
            )
            let step = try core.step(
                observation: observation,
                cameraAngularVelocity: try MobileWhamArrays.zeros(
                    [1, 1, 6],
                    dataType: .float16
                ),
                state: &state,
                predict: { stage, model, values in
                    try run(stage) {
                        try MobileWhamArrays.predict(model, values)
                    }
                },
                smooth: { stage, smoother, pose, shape in
                    try run(stage) {
                        try smoother.smooth(pose: pose, shape: shape)
                    }
                }
            )
            _ = step.verticesWorld[0].floatValue
            coreState = state
            maximumSmoothingPoseDelta = max(
                maximumSmoothingPoseDelta,
                step.maximumSmoothingPoseDelta
            )
            if frontend != nil {
                lastInput = step.whamInput
            }
        }
        return TrackResult(
            detections: detections,
            lastWHAMInput: lastInput,
            lastValidToken: lastValidToken,
            maximumSmoothingPoseDelta: maximumSmoothingPoseDelta
        )
    }

    nonisolated private static func loadBenchmarkFrames(
        ciContext: CIContext
    ) throws -> [CVPixelBuffer] {
        let preprocessor = MobileWhamPreprocessor(ciContext: ciContext)
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
                  let buffer = preprocessor.makePixelBuffer(
                    width: cgImage.width,
                    height: cgImage.height
                  ) else {
                throw BenchmarkError.invalidSampleFrame(url.lastPathComponent)
            }
            ciContext.render(CIImage(cgImage: cgImage), to: buffer)
            return buffer
        }
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
        case missingSampleFrames
        case invalidSampleFrame(String)
        case noPersonInWarmupFrame
        case initializationFailed
        case measurementTypeMismatch(String)

        var errorDescription: String? {
            switch self {
            case .missingSampleFrames:
                return "Bundled BenchmarkFrames were not found"
            case .invalidSampleFrame(let name):
                return "Could not decode benchmark frame \(name)"
            case .noPersonInWarmupFrame:
                return "YOLO26 did not detect a person in the initializer frame"
            case .initializationFailed:
                return "The HMR2.0-S/WHAM recurrent state was not initialized"
            case .measurementTypeMismatch(let stage):
                return "Internal benchmark measurement mismatch at \(stage)"
            }
        }
    }
}
