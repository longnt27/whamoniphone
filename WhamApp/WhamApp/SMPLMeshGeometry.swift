import SceneKit
import UIKit
import simd

enum SMPLMeshGeometry {
    static func vertexNormals(
        vertices: [SIMD3<Float>],
        faces: [UInt16]
    ) -> [SIMD3<Float>] {
        var normals = Array(
            repeating: SIMD3<Float>(repeating: 0),
            count: vertices.count
        )
        guard faces.count.isMultiple(of: 3) else { return normals }

        for offset in stride(from: 0, to: faces.count, by: 3) {
            let first = Int(faces[offset])
            let second = Int(faces[offset + 1])
            let third = Int(faces[offset + 2])
            guard first < vertices.count,
                  second < vertices.count,
                  third < vertices.count else {
                continue
            }
            let normal = simd_cross(
                vertices[second] - vertices[first],
                vertices[third] - vertices[first]
            )
            guard simd_length_squared(normal) > 1e-12 else { continue }
            normals[first] += normal
            normals[second] += normal
            normals[third] += normal
        }

        for index in normals.indices {
            let magnitudeSquared = simd_length_squared(normals[index])
            normals[index] = magnitudeSquared > 1e-12
                ? simd_normalize(normals[index])
                : SIMD3<Float>(0, 1, 0)
        }
        return normals
    }

    static func make(
        vertices worldVertices: [SIMD3<Float>],
        topology: SMPLTopology
    ) -> SCNGeometry? {
        guard worldVertices.count == topology.vertexCount else { return nil }

        let vertices = worldVertices.map { vertex in
            SIMD3<Float>(vertex.x, -vertex.y, -vertex.z)
        }
        return makeSceneGeometry(vertices: vertices, topology: topology)
    }

    static func makeVideoOverlay(
        projectedVertices: [SIMD3<Float>],
        cameraVertices: [SIMD3<Float>],
        viewportSize: CGSize,
        topology: SMPLTopology
    ) -> SCNGeometry? {
        guard projectedVertices.count == topology.vertexCount,
              cameraVertices.count == topology.vertexCount,
              viewportSize.width > 0,
              viewportSize.height > 0 else {
            return nil
        }
        let halfWidth = Float(viewportSize.width / 2)
        let halfHeight = Float(viewportSize.height / 2)
        let vertices = projectedVertices.map {
            SIMD3<Float>($0.x - halfWidth, halfHeight - $0.y, $0.z)
        }
        let lightingVertices = cameraVertices.map {
            SIMD3<Float>($0.x, -$0.y, -$0.z)
        }
        let normals = vertexNormals(
            vertices: lightingVertices,
            faces: topology.indices
        )
        return makeSceneGeometry(
            vertices: vertices,
            normals: normals,
            topology: topology
        )
    }

    private static func makeSceneGeometry(
        vertices: [SIMD3<Float>],
        topology: SMPLTopology
    ) -> SCNGeometry {
        let normals = vertexNormals(vertices: vertices, faces: topology.indices)
        return makeSceneGeometry(
            vertices: vertices,
            normals: normals,
            topology: topology
        )
    }

    private static func makeSceneGeometry(
        vertices: [SIMD3<Float>],
        normals: [SIMD3<Float>],
        topology: SMPLTopology
    ) -> SCNGeometry {
        let sceneVertices = vertices.map { SCNVector3($0.x, $0.y, $0.z) }
        let sceneNormals = normals.map { SCNVector3($0.x, $0.y, $0.z) }
        let indexData = topology.indices.withUnsafeBytes { Data($0) }
        let element = SCNGeometryElement(
            data: indexData,
            primitiveType: .triangles,
            primitiveCount: topology.triangleCount,
            bytesPerIndex: MemoryLayout<UInt16>.size
        )
        let geometry = SCNGeometry(
            sources: [
                SCNGeometrySource(vertices: sceneVertices),
                SCNGeometrySource(normals: sceneNormals),
            ],
            elements: [element]
        )
        geometry.materials = [bodyMaterial()]
        return geometry
    }

    private static func bodyMaterial() -> SCNMaterial {
        let material = SCNMaterial()
        material.name = "WHAM SMPL body"
        material.lightingModel = .physicallyBased
        material.diffuse.contents = UIColor.systemBlue
        material.metalness.contents = 0.12
        material.roughness.contents = 0.52
        material.transparency = 0.82
        material.isDoubleSided = true
        return material
    }
}
