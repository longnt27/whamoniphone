//
//  WhamAppTests.swift
//  WhamAppTests
//
//  Created by admin on 20/3/26.
//

import Testing
import CoreML
import Foundation
@testable import WhamApp

struct WhamAppTests {

    @Test func smplMeshCacheRoundTripsFloat16Frames() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("fixture.whammesh")

        let writer = try SMPLMeshCache.Writer(outputURL: url, vertexCount: 3)
        try writer.append([
            SIMD3<Float>(1.25, -2.5, 3.75),
            SIMD3<Float>(4.5, 5.25, -6.0),
            SIMD3<Float>(-7.5, 8.0, 9.5),
        ])
        try writer.append([
            SIMD3<Float>(0.125, 0.25, 0.5),
            SIMD3<Float>(1.0, 2.0, 4.0),
            SIMD3<Float>(8.0, 16.0, 32.0),
        ])
        try writer.finalize()

        let reader = try SMPLMeshCache.Reader(url: url, expectedVertexCount: 3)
        #expect(reader.vertexCount == 3)
        #expect(reader.frameCount == 2)
        let first = try reader.frame(at: 0)
        let second = try reader.frame(at: 1)
        #expect(first[0] == SIMD3<Float>(1.25, -2.5, 3.75))
        #expect(first[2] == SIMD3<Float>(-7.5, 8.0, 9.5))
        #expect(second[0] == SIMD3<Float>(0.125, 0.25, 0.5))
        #expect(second[2] == SIMD3<Float>(8.0, 16.0, 32.0))
    }

    @Test func smplMeshCacheDerivesSiblingURL() {
        let json = URL(fileURLWithPath: "/tmp/walk_wham_output.json")
        #expect(
            SMPLMeshCache.outputURL(forJSONURL: json).path
                == "/tmp/walk_wham_output.whammesh"
        )
    }

    @Test func smplMeshCacheRejectsUnexpectedVertexCount() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("fixture.whammesh")
        let writer = try SMPLMeshCache.Writer(outputURL: url, vertexCount: 3)
        try writer.append(Array(repeating: SIMD3<Float>(0, 0, 0), count: 3))
        try writer.finalize()

        #expect(throws: SMPLMeshCache.CacheError.self) {
            try SMPLMeshCache.Reader(url: url, expectedVertexCount: 4)
        }
    }

    @Test func smplMeshCacheRejectsMalformedAndOutOfRangeData() throws {
        #expect(throws: SMPLMeshCache.CacheError.self) {
            try SMPLMeshCache.Reader(
                data: Data(repeating: 0, count: 24),
                expectedVertexCount: 3
            )
        }

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("fixture.whammesh")
        let writer = try SMPLMeshCache.Writer(outputURL: url, vertexCount: 3)
        try writer.append(Array(repeating: SIMD3<Float>(1, 2, 3), count: 3))
        try writer.finalize()

        #expect(throws: SMPLMeshCache.CacheError.self) {
            try SMPLMeshCache.Reader(url: url, expectedVertexCount: 3).frame(at: 1)
        }
        var truncated = try Data(contentsOf: url)
        truncated.removeLast()
        #expect(throws: SMPLMeshCache.CacheError.self) {
            try SMPLMeshCache.Reader(data: truncated, expectedVertexCount: 3)
        }
    }

    @Test func selectedMobilePipelineCatalogIsLocked() {
        #expect(MobileWhamModelCatalog.resourceNames == [
            "yolo26m-pose",
            "HMR2SFrontend",
            "HMR2SSMPLInit",
            "HMR2STokenAdapter",
            "WHAM_I",
            "WHAM_ImageStep",
            "WHAM_WorldStep"
        ])
        #expect(
            MobileWhamModelCatalog.yoloWeightsSHA256
                == "2fbf16367022256a226035695c5c389384c6706e8bb8ab8fcd0e7976f05443c4"
        )
        #expect(
            MobileWhamModelCatalog.adapterCheckpointSHA256
                == "4fd581b2b7f2d0cac8bda7597692f7e77ca082435c5e352c69964d64da10f526"
        )
    }

    @Test func missingMobileObservationUsesMaskedFloat16Tensors() throws {
        let observation = try MobileWhamPreprocessor.missingFrame(videoTime: 1.25)

        #expect(observation.keypoints.shape.map(\.intValue) == [1, 1, 37])
        #expect(observation.keypoints.dataType == .float16)
        #expect(observation.keypointMask.shape.map(\.intValue) == [1, 1, 17])
        #expect(observation.keypointMask.dataType == .float16)
        #expect((0..<17).allSatisfy {
            observation.keypointMask[$0].floatValue == 1
        })
        #expect(observation.imageFeature.shape.map(\.intValue) == [1, 1, 1024])
        #expect(observation.imageFeature.dataType == .float16)
        #expect(observation.imageFeatureValid[0].floatValue == 0)
        #expect(observation.hmrPose == nil)
        #expect(observation.hmrBetas == nil)
        #expect(observation.videoTime == 1.25)
    }

    @Test func smoothingIsOutputOnlyForRecurrentFeedback() throws {
        let rawPose = try MLMultiArray(shape: [1, 1, 144], dataType: .float16)
        let smoothedPose = try MLMultiArray(
            shape: [1, 1, 144],
            dataType: .float16
        )
        rawPose[0] = 0.25
        smoothedPose[0] = 0.75

        let routing = MobileWhamCore.outputOnlyPoseRouting(
            rawPose: rawPose,
            smoothedPose: smoothedPose
        )

        #expect(routing.recurrentPose === rawPose)
        #expect(routing.outputPose === smoothedPose)
        #expect(routing.recurrentPose[0].floatValue == 0.25)
        #expect(routing.outputPose[0].floatValue == 0.75)
    }

    @Test func parsesYOLOv8PoseLayout() throws {
        let output = try MLMultiArray(shape: [1, 56, 2], dataType: .float32)
        output[[0, 4, 1] as [NSNumber]] = NSNumber(value: 0.9)
        output[[0, 0, 1] as [NSNumber]] = NSNumber(value: 320)
        output[[0, 1, 1] as [NSNumber]] = NSNumber(value: 320)
        output[[0, 2, 1] as [NSNumber]] = NSNumber(value: 128)
        output[[0, 3, 1] as [NSNumber]] = NSNumber(value: 256)
        for joint in 0..<17 {
            let offset = 5 + joint * 3
            output[[0, offset, 1] as [NSNumber]] = NSNumber(value: 10 + joint)
            output[[0, offset + 1, 1] as [NSNumber]] = NSNumber(value: 20 + joint)
            output[[0, offset + 2, 1] as [NSNumber]] = NSNumber(value: 0.8)
        }

        let detection = try #require(PoseDetector.parseYOLOv8Output(output).first)
        #expect(abs(detection.box.minX - 0.4) < 0.0001)
        #expect(abs(detection.box.minY - 0.3) < 0.0001)
        #expect(abs(detection.box.width - 0.2) < 0.0001)
        #expect(abs(detection.box.height - 0.4) < 0.0001)
        #expect(abs(detection.keypoints[0].x - 10.0 / 640.0) < 0.0001)
        #expect(detection.keypointConfidences.count == 17)
    }

    @Test func parsesYOLO26EndToEndPoseLayout() throws {
        let output = try MLMultiArray(shape: [1, 2, 57], dataType: .float32)
        output[[0, 1, 4] as [NSNumber]] = NSNumber(value: 0.95)
        output[[0, 1, 0] as [NSNumber]] = NSNumber(value: 64)
        output[[0, 1, 1] as [NSNumber]] = NSNumber(value: 128)
        output[[0, 1, 2] as [NSNumber]] = NSNumber(value: 384)
        output[[0, 1, 3] as [NSNumber]] = NSNumber(value: 512)
        output[[0, 1, 5] as [NSNumber]] = NSNumber(value: 0)
        for joint in 0..<17 {
            let offset = 6 + joint * 3
            output[[0, 1, offset] as [NSNumber]] = NSNumber(value: 30 + joint)
            output[[0, 1, offset + 1] as [NSNumber]] = NSNumber(value: 40 + joint)
            output[[0, 1, offset + 2] as [NSNumber]] = NSNumber(value: 0.85)
        }

        let detection = try #require(PoseDetector.parseYOLO26Output(output).first)
        #expect(abs(detection.box.minX - 0.1) < 0.0001)
        #expect(abs(detection.box.minY - 0.2) < 0.0001)
        #expect(abs(detection.box.width - 0.5) < 0.0001)
        #expect(abs(detection.box.height - 0.6) < 0.0001)
        #expect(abs(detection.keypoints[0].y - 40.0 / 640.0) < 0.0001)
        #expect(detection.keypointConfidences.count == 17)
    }

    @Test func selectedTemporalSmootherUsesGeodesicPoseAndShapeEMA() throws {
        func identityPose() throws -> MLMultiArray {
            let pose = try MLMultiArray(shape: [1, 1, 144], dataType: .float32)
            for joint in 0..<24 {
                pose[joint * 6] = 1
                pose[joint * 6 + 4] = 1
            }
            return pose
        }

        var smoother = TemporalOutputSmoother()
        let initialShape = try MLMultiArray(shape: [1, 1, 10], dataType: .float32)
        _ = try smoother.smooth(pose: identityPose(), shape: initialShape)

        let quarterTurn = try identityPose()
        for joint in 0..<24 {
            let offset = joint * 6
            quarterTurn[offset] = 0
            quarterTurn[offset + 1] = -1
            quarterTurn[offset + 3] = 1
            quarterTurn[offset + 4] = 0
        }
        let currentShape = try MLMultiArray(shape: [1, 1, 10], dataType: .float32)
        for index in 0..<10 { currentShape[index] = 10 }

        let filtered = try smoother.smooth(pose: quarterTurn, shape: currentShape)
        let expectedCosine = cos(Float.pi * 0.75 / 2)
        let expectedSine = sin(Float.pi * 0.75 / 2)
        #expect(abs(filtered.pose[0].floatValue - expectedCosine) < 0.0001)
        #expect(abs(filtered.pose[1].floatValue + expectedSine) < 0.0001)
        #expect(abs(filtered.pose[3].floatValue - expectedSine) < 0.0001)
        #expect(abs(filtered.shape[0].floatValue - 3.5) < 0.0001)
    }

}
