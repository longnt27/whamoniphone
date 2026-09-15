//
//  WhamAppTests.swift
//  WhamAppTests
//
//  Created by admin on 20/3/26.
//

import Testing
import CoreML
@testable import WhamApp

struct WhamAppTests {

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
