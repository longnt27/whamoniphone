import CoreML
import Foundation

struct MobileWhamPoseRouting {
    let recurrentPose: MLMultiArray
    let outputPose: MLMultiArray
}

struct MobileWhamCore {
    typealias Prediction = (
        _ stage: String,
        _ model: MLModel,
        _ values: [String: MLFeatureValue]
    ) throws -> MLFeatureProvider

    typealias Smoothing = (
        _ stage: String,
        _ smoother: inout TemporalOutputSmoother,
        _ pose: MLMultiArray,
        _ shape: MLMultiArray
    ) throws -> (pose: MLMultiArray, shape: MLMultiArray)

    struct State {
        var previousKeypoints: MLMultiArray
        var previousRoot: MLMultiArray
        var previousPose: MLMultiArray
        var hEnc: MLMultiArray
        var cEnc: MLMultiArray
        var hTraj: MLMultiArray
        var cTraj: MLMultiArray
        var hDec: MLMultiArray
        var cDec: MLMultiArray
        var previousUnrefinedRoot: MLMultiArray
        var previousUnrefinedTranslation: MLMultiArray
        var previousBodyFeet: MLMultiArray
        var previousWorldFeet: MLMultiArray
        var previousRefinedRoot: MLMultiArray
        var previousRefinedTranslation: MLMultiArray
        var hRefiner: MLMultiArray
        var cRefiner: MLMultiArray
        var hasPreviousWorldFrame: Float
        var smoother: TemporalOutputSmoother
    }

    struct StepResult {
        let rawPose: MLMultiArray
        let pose: MLMultiArray
        let shape: MLMultiArray
        let keypointsRootRelative: MLMultiArray
        let jointsWorld: MLMultiArray
        let verticesWorld: MLMultiArray
        let refinedRoot: MLMultiArray
        let refinedTranslation: MLMultiArray
        let whamInput: [String: MLFeatureValue]
        let maximumSmoothingPoseDelta: Float
    }

    let initializer: MLModel
    let recurrentStep: MLModel
    let worldStep: MLModel

    static func outputOnlyPoseRouting(
        rawPose: MLMultiArray,
        smoothedPose: MLMultiArray
    ) -> MobileWhamPoseRouting {
        MobileWhamPoseRouting(
            recurrentPose: rawPose,
            outputPose: smoothedPose
        )
    }

    func initialize(
        firstPose: MLMultiArray,
        initialJoints3D: MLMultiArray,
        firstKeypoints: MLMultiArray,
        predict: Prediction = defaultPrediction
    ) throws -> State {
        let initialKeypoints = try MobileWhamArrays.initializerInput(
            joints3D: initialJoints3D,
            observation: firstKeypoints
        )
        let initialPose = try MobileWhamArrays.copy(
            firstPose,
            shape: [1, 1, 144],
            dataType: .float32
        )
        let initial = try predict("wham_init", initializer, [
            "init_kp": MLFeatureValue(multiArray: initialKeypoints),
            "init_smpl": MLFeatureValue(multiArray: initialPose)
        ])
        let root = try MobileWhamArrays.poseRoot(firstPose)
        return State(
            previousKeypoints: try MobileWhamArrays.copy(
                initialJoints3D,
                shape: [1, 1, 51],
                dataType: .float16
            ),
            previousRoot: root,
            previousPose: try MobileWhamArrays.copy(
                firstPose,
                shape: [1, 1, 144],
                dataType: .float16
            ),
            hEnc: try Self.stateArray(initial, "h_enc", [3, 1, 512]),
            cEnc: try Self.stateArray(initial, "c_enc", [3, 1, 512]),
            hTraj: try Self.stateArray(initial, "h_traj", [3, 1, 563]),
            cTraj: try Self.stateArray(initial, "c_traj", [3, 1, 563]),
            hDec: try Self.stateArray(initial, "h_dec", [3, 1, 563]),
            cDec: try Self.stateArray(initial, "c_dec", [3, 1, 563]),
            previousUnrefinedRoot: root,
            previousUnrefinedTranslation: try MobileWhamArrays.zeros(
                [1, 1, 3], dataType: .float16
            ),
            previousBodyFeet: try MobileWhamArrays.zeros(
                [1, 1, 4, 3], dataType: .float16
            ),
            previousWorldFeet: try MobileWhamArrays.zeros(
                [1, 1, 4, 3], dataType: .float16
            ),
            previousRefinedRoot: root,
            previousRefinedTranslation: try MobileWhamArrays.zeros(
                [1, 1, 3], dataType: .float16
            ),
            hRefiner: try MobileWhamArrays.zeros(
                [2, 1, 512], dataType: .float16
            ),
            cRefiner: try MobileWhamArrays.zeros(
                [2, 1, 512], dataType: .float16
            ),
            hasPreviousWorldFrame: 0,
            smoother: TemporalOutputSmoother()
        )
    }

    func step(
        observation: MobileWhamFrameObservation,
        cameraAngularVelocity: MLMultiArray,
        state: inout State,
        predict: Prediction = defaultPrediction,
        smooth: Smoothing = defaultSmoothing
    ) throws -> StepResult {
        let whamInput: [String: MLFeatureValue] = [
            "x_step": MLFeatureValue(multiArray: observation.keypoints),
            "keypoint_mask_step": MLFeatureValue(
                multiArray: observation.keypointMask
            ),
            "image_feature_step": MLFeatureValue(
                multiArray: observation.imageFeature
            ),
            "image_feature_valid_step": MLFeatureValue(
                multiArray: observation.imageFeatureValid
            ),
            "cam_a_step": MLFeatureValue(multiArray: cameraAngularVelocity),
            "prev_kp3d": MLFeatureValue(multiArray: state.previousKeypoints),
            "prev_root": MLFeatureValue(multiArray: state.previousRoot),
            "prev_pose": MLFeatureValue(multiArray: state.previousPose),
            "h_enc_in": MLFeatureValue(multiArray: state.hEnc),
            "c_enc_in": MLFeatureValue(multiArray: state.cEnc),
            "h_traj_in": MLFeatureValue(multiArray: state.hTraj),
            "c_traj_in": MLFeatureValue(multiArray: state.cTraj),
            "h_dec_in": MLFeatureValue(multiArray: state.hDec),
            "c_dec_in": MLFeatureValue(multiArray: state.cDec)
        ]
        let output = try predict(
            "wham_recurrent_step",
            recurrentStep,
            whamInput
        )
        let rawPose = try MobileWhamArrays.required(output, "pred_pose")
        let rawShape = try MobileWhamArrays.required(output, "pred_shape")
        let smoothed = try smooth(
            "temporal_output_smoothing",
            &state.smoother,
            rawPose,
            rawShape
        )
        let poseRouting = Self.outputOnlyPoseRouting(
            rawPose: rawPose,
            smoothedPose: smoothed.pose
        )
        let predictedKeypoints = try MobileWhamArrays.required(output, "pred_kp3d")
        let predictedRoot = try MobileWhamArrays.required(output, "pred_root")
        let hEncOut = try MobileWhamArrays.required(output, "h_enc_out")

        let world = try predict(
            "wham_smpl_world_refiner_step",
            worldStep,
            [
                "pred_pose": MLFeatureValue(multiArray: poseRouting.outputPose),
                "pred_shape": MLFeatureValue(multiArray: smoothed.shape),
                "pred_contact": MLFeatureValue(
                    multiArray: try MobileWhamArrays.required(output, "pred_contact")
                ),
                "pred_root": MLFeatureValue(multiArray: predictedRoot),
                "pred_vel": MLFeatureValue(
                    multiArray: try MobileWhamArrays.required(output, "pred_vel")
                ),
                "pred_kp3d": MLFeatureValue(multiArray: predictedKeypoints),
                "h_enc_out": MLFeatureValue(multiArray: hEncOut),
                "prev_unrefined_root": MLFeatureValue(
                    multiArray: state.previousUnrefinedRoot
                ),
                "prev_unrefined_translation": MLFeatureValue(
                    multiArray: state.previousUnrefinedTranslation
                ),
                "prev_body_feet": MLFeatureValue(
                    multiArray: state.previousBodyFeet
                ),
                "prev_world_feet": MLFeatureValue(
                    multiArray: state.previousWorldFeet
                ),
                "prev_refined_root": MLFeatureValue(
                    multiArray: state.previousRefinedRoot
                ),
                "prev_refined_translation": MLFeatureValue(
                    multiArray: state.previousRefinedTranslation
                ),
                "h_refiner_in": MLFeatureValue(multiArray: state.hRefiner),
                "c_refiner_in": MLFeatureValue(multiArray: state.cRefiner),
                "has_previous": MLFeatureValue(
                    multiArray: try MobileWhamArrays.scalar(
                        state.hasPreviousWorldFrame,
                        dataType: .float16
                    )
                )
            ]
        )

        var maximumDelta: Float = 0
        for index in 0..<rawPose.count {
            maximumDelta = max(
                maximumDelta,
                abs(rawPose[index].floatValue - smoothed.pose[index].floatValue)
            )
        }

        state.previousKeypoints = predictedKeypoints
        state.previousRoot = predictedRoot
        state.previousPose = poseRouting.recurrentPose
        state.hEnc = hEncOut
        state.cEnc = try MobileWhamArrays.required(output, "c_enc_out")
        state.hTraj = try MobileWhamArrays.required(output, "h_traj_out")
        state.cTraj = try MobileWhamArrays.required(output, "c_traj_out")
        state.hDec = try MobileWhamArrays.required(output, "h_dec_out")
        state.cDec = try MobileWhamArrays.required(output, "c_dec_out")
        state.previousUnrefinedRoot = try MobileWhamArrays.required(
            world,
            "unrefined_root"
        )
        state.previousUnrefinedTranslation = try MobileWhamArrays.required(
            world,
            "unrefined_translation"
        )
        state.previousBodyFeet = try MobileWhamArrays.required(world, "body_feet")
        state.previousWorldFeet = try MobileWhamArrays.required(world, "world_feet")
        state.previousRefinedRoot = try MobileWhamArrays.required(
            world,
            "refined_root"
        )
        state.previousRefinedTranslation = try MobileWhamArrays.required(
            world,
            "refined_translation"
        )
        state.hRefiner = try MobileWhamArrays.required(world, "h_refiner_out")
        state.cRefiner = try MobileWhamArrays.required(world, "c_refiner_out")
        state.hasPreviousWorldFrame = 1

        return StepResult(
            rawPose: rawPose,
            pose: poseRouting.outputPose,
            shape: smoothed.shape,
            keypointsRootRelative: predictedKeypoints,
            jointsWorld: try MobileWhamArrays.required(world, "joints_world"),
            verticesWorld: try MobileWhamArrays.required(world, "vertices_world"),
            refinedRoot: state.previousRefinedRoot,
            refinedTranslation: state.previousRefinedTranslation,
            whamInput: whamInput,
            maximumSmoothingPoseDelta: maximumDelta
        )
    }

    private static func stateArray(
        _ provider: MLFeatureProvider,
        _ name: String,
        _ shape: [NSNumber]
    ) throws -> MLMultiArray {
        try MobileWhamArrays.copy(
            MobileWhamArrays.required(provider, name),
            shape: shape,
            dataType: .float16
        )
    }

    private static func defaultPrediction(
        _ stage: String,
        _ model: MLModel,
        _ values: [String: MLFeatureValue]
    ) throws -> MLFeatureProvider {
        try MobileWhamArrays.predict(model, values)
    }

    private static func defaultSmoothing(
        _ stage: String,
        _ smoother: inout TemporalOutputSmoother,
        _ pose: MLMultiArray,
        _ shape: MLMultiArray
    ) throws -> (pose: MLMultiArray, shape: MLMultiArray) {
        try smoother.smooth(pose: pose, shape: shape)
    }
}
