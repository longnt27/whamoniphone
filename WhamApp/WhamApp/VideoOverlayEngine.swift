import Combine
import SceneKit
import UIKit
import simd

final class VideoOverlayEngine: ObservableObject, BodyPresentationControlling {
    let scene = SCNScene()
    let cameraNode = SCNNode()

    @Published private(set) var presentationMode: BodyPresentationMode
    var meshAvailable: Bool { topology != nil }

    private let topology: SMPLTopology?
    private let meshNode = SCNNode()
    private var jointNodes: [SCNNode] = []
    private var boneNodes: [SCNNode] = []

    private let boneConnections: [(Int, Int)] = [
        (0, 1), (1, 3), (0, 2), (2, 4),
        (5, 6), (11, 12), (5, 11), (6, 12), (0, 5), (0, 6),
        (5, 7), (7, 9), (6, 8), (8, 10),
        (11, 13), (13, 15), (12, 14), (14, 16),
    ]

    init(topology: SMPLTopology? = SMPLTopology.loadFromBundle()) {
        self.topology = topology
        presentationMode = topology == nil ? .skeleton : .mesh
        setupScene()
    }

    func setPresentationMode(_ requestedMode: BodyPresentationMode) {
        presentationMode = requestedMode == .mesh && !meshAvailable
            ? .skeleton
            : requestedMode
        updateVisibility()
    }

    func clear() {
        meshNode.geometry = nil
        jointNodes.forEach { $0.isHidden = true }
        boneNodes.forEach { $0.isHidden = true }
    }

    func applyFrameData(
        keypointsWorld: [Float],
        meshVerticesWorld: [SIMD3<Float>]?,
        metadata: VideoOverlayMetadata,
        viewportSize: CGSize,
        registration: VideoOverlayRegistration = .identity
    ) {
        guard keypointsWorld.count == 51,
              viewportSize.width > 0,
              viewportSize.height > 0 else {
            clear()
            return
        }

        // SceneKit's orthographicScale is the half-height of the visible
        // camera volume. The projected coordinates below are expressed in
        // UIKit points around the viewport centre, so using the full height
        // compresses both position and size by exactly 2x toward the centre.
        cameraNode.camera?.orthographicScale = Double(viewportSize.height / 2)
        let worldJoints = stride(from: 0, to: 51, by: 3).map {
            SIMD3<Float>(
                keypointsWorld[$0],
                keypointsWorld[$0 + 1],
                keypointsWorld[$0 + 2]
            )
        }
        let projectedJoints = SMPLVideoProjector.projectWorldToViewport(
            worldJoints,
            metadata: metadata,
            viewportSize: viewportSize,
            registration: registration
        )
        guard projectedJoints.count == 17 else {
            clear()
            return
        }

        let positions = projectedJoints.map {
            scenePosition($0, viewportSize: viewportSize)
        }
        updateSkeleton(positions)

        if presentationMode == .mesh {
            if let topology, let meshVerticesWorld {
                let cameraVertices = SMPLVideoProjector.worldToCamera(
                    meshVerticesWorld,
                    metadata: metadata
                )
                let sourcePixels = registration.applying(
                    to: SMPLVideoProjector.projectToSourcePixels(
                        cameraVertices: cameraVertices,
                        metadata: metadata
                    )
                )
                let projectedMesh = SMPLVideoProjector.mapSourcePixelsToViewport(
                    sourcePixels,
                    sourceSize: metadata.sourceSize,
                    viewportSize: viewportSize
                )
                meshNode.geometry = SMPLMeshGeometry.makeVideoOverlay(
                    projectedVertices: projectedMesh,
                    cameraVertices: cameraVertices,
                    viewportSize: viewportSize,
                    topology: topology
                )
            } else {
                meshNode.geometry = nil
            }
        }
        updateVisibility()
    }

    private func setupScene() {
        scene.background.contents = UIColor.clear

        let camera = SCNCamera()
        camera.usesOrthographicProjection = true
        camera.orthographicScale = 1
        camera.zNear = 0.001
        camera.zFar = 10_000
        cameraNode.camera = camera
        cameraNode.position = SCNVector3(0, 0, 1)
        scene.rootNode.addChildNode(cameraNode)

        let ambient = SCNNode()
        ambient.light = SCNLight()
        ambient.light?.type = .ambient
        ambient.light?.intensity = 350
        scene.rootNode.addChildNode(ambient)

        let directional = SCNNode()
        directional.light = SCNLight()
        directional.light?.type = .directional
        directional.light?.intensity = 700
        directional.eulerAngles = SCNVector3(-0.35, 0.4, 0)
        scene.rootNode.addChildNode(directional)

        meshNode.name = "WHAM camera-aligned SMPL mesh"
        scene.rootNode.addChildNode(meshNode)

        for _ in 0..<17 {
            let sphere = SCNSphere(radius: 4)
            sphere.firstMaterial?.diffuse.contents = UIColor.systemGreen
            sphere.firstMaterial?.lightingModel = .constant
            let node = SCNNode(geometry: sphere)
            jointNodes.append(node)
            scene.rootNode.addChildNode(node)
        }

        for _ in boneConnections {
            let cylinder = SCNCylinder(radius: 2, height: 1)
            cylinder.firstMaterial?.diffuse.contents = UIColor.white
            cylinder.firstMaterial?.lightingModel = .constant
            let node = SCNNode(geometry: cylinder)
            boneNodes.append(node)
            scene.rootNode.addChildNode(node)
        }
        updateVisibility()
    }

    private func updateSkeleton(_ positions: [SCNVector3]) {
        SCNTransaction.begin()
        SCNTransaction.animationDuration = 0
        for index in jointNodes.indices {
            jointNodes[index].position = positions[index]
        }
        for (index, connection) in boneConnections.enumerated() {
            let start = positions[connection.0]
            let end = positions[connection.1]
            let direction = SIMD3<Float>(
                end.x - start.x,
                end.y - start.y,
                end.z - start.z
            )
            let height = simd_length(direction)
            let node = boneNodes[index]
            guard height > 0.001 else {
                node.isHidden = true
                continue
            }
            (node.geometry as? SCNCylinder)?.height = CGFloat(height)
            node.position = SCNVector3(
                (start.x + end.x) / 2,
                (start.y + end.y) / 2,
                (start.z + end.z) / 2
            )
            let up = SIMD3<Float>(0, 1, 0)
            let normalizedDirection = simd_normalize(direction)
            let axis = simd_cross(up, normalizedDirection)
            if simd_length_squared(axis) > 1e-8 {
                node.simdOrientation = simd_quatf(
                    angle: acos(max(-1, min(1, simd_dot(up, normalizedDirection)))),
                    axis: simd_normalize(axis)
                )
            }
        }
        SCNTransaction.commit()
    }

    private func scenePosition(
        _ projected: SIMD3<Float>,
        viewportSize: CGSize
    ) -> SCNVector3 {
        SCNVector3(
            projected.x - Float(viewportSize.width / 2),
            Float(viewportSize.height / 2) - projected.y,
            projected.z
        )
    }

    private func updateVisibility() {
        let showMesh = presentationMode == .mesh && meshAvailable
        meshNode.isHidden = !showMesh
        jointNodes.forEach { $0.isHidden = showMesh }
        boneNodes.forEach { $0.isHidden = showMesh }
    }
}
