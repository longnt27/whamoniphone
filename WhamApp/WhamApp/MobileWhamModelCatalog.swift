import CoreML
import Foundation

enum MobileWhamModelCatalog {
    static let detector = "yolo26m-pose"
    static let hmrFrontend = "HMR2SFrontend"
    static let smplInitializer = "HMR2SSMPLInit"
    static let tokenAdapter = "HMR2STokenAdapter"
    static let whamInitializer = "WHAM_I"
    static let whamStep = "WHAM_ImageStep"
    static let worldStep = "WHAM_WorldStep"

    static let resourceNames = [
        detector,
        hmrFrontend,
        smplInitializer,
        tokenAdapter,
        whamInitializer,
        whamStep,
        worldStep
    ]

    static let pipelineDescription =
        "YOLO26m-pose → released HMR2.0-S → selected token adapter → "
        + "released split WHAM → light causal smoothing"

    static let yoloWeightsSHA256 =
        "2fbf16367022256a226035695c5c389384c6706e8bb8ab8fcd0e7976f05443c4"
    static let adapterCheckpointSHA256 =
        "4fd581b2b7f2d0cac8bda7597692f7e77ca082435c5e352c69964d64da10f526"
    static let whamCheckpointSHA256 =
        "2ba0cb6a7dd597023a6b2ad6056e7a8b6b33144a35fabea570bfd00842cd4eaf"

    static func load(
        _ resourceName: String,
        configuration: MLModelConfiguration
    ) throws -> MLModel {
        guard let url = Bundle.main.url(
            forResource: resourceName,
            withExtension: "mlmodelc"
        ) else {
            throw MobileWhamError.missingModel(resourceName)
        }
        return try MLModel(contentsOf: url, configuration: configuration)
    }

    static func provenance(
        hmrFrontendModel: MLModel,
        adapterModel: MLModel
    ) -> [String: String] {
        let hmrMetadata = hmrFrontendModel.modelDescription.metadata[
            MLModelMetadataKey.creatorDefinedKey
        ] as? [String: String]
        let adapterMetadata = adapterModel.modelDescription.metadata[
            MLModelMetadataKey.creatorDefinedKey
        ] as? [String: String]
        return [
            "yolo26_weights_sha256": yoloWeightsSHA256,
            "yolo26_variant": detector,
            "hmr2s_repository": hmrMetadata?["source_repository"] ?? "missing",
            "hmr2s_commit": hmrMetadata?["source_commit"] ?? "missing",
            "hmr2s_checkpoint_sha256": hmrMetadata?["checkpoint_sha256"] ?? "missing",
            "token_adapter_checkpoint_sha256":
                adapterMetadata?["checkpoint_sha256"] ?? adapterCheckpointSHA256,
            "wham_repository": "https://github.com/yohanshin/WHAM.git",
            "wham_commit": "2b54f7797391c94876848b905ed875b154c4a295",
            "wham_checkpoint_sha256": whamCheckpointSHA256,
            "training_performed_by_this_project":
                hmrMetadata?["training_performed_by_this_project"] ?? "missing"
        ]
    }

    static func bundledModelSizeBytes(_ resourceName: String) -> UInt64? {
        guard let modelURL = Bundle.main.url(
            forResource: resourceName,
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
}
