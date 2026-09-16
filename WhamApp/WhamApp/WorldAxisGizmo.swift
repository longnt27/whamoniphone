import SceneKit
import UIKit
import simd

enum WorldAxisGizmo {
    static let rootName = "WHAM world axes"

    static func make(length: Float = 1) -> SCNNode {
        let root = SCNNode()
        root.name = rootName

        let origin = SCNSphere(radius: 0.018)
        origin.firstMaterial = constantMaterial(color: .white)
        let originNode = SCNNode(geometry: origin)
        originNode.name = "world-axis-origin"
        root.addChildNode(originNode)

        addAxis(
            name: "x",
            label: "X",
            direction: SIMD3<Float>(1, 0, 0),
            length: length,
            color: .systemRed,
            to: root
        )
        addAxis(
            name: "y",
            label: "Y",
            direction: SIMD3<Float>(0, 1, 0),
            length: length,
            color: .systemGreen,
            to: root
        )
        addAxis(
            name: "z",
            label: "Z",
            direction: SIMD3<Float>(0, 0, 1),
            length: length,
            color: .systemBlue,
            to: root
        )
        return root
    }

    private static func addAxis(
        name: String,
        label: String,
        direction: SIMD3<Float>,
        length: Float,
        color: UIColor,
        to root: SCNNode
    ) {
        let safeLength = max(length, 0.1)
        let tipHeight = min(0.08, safeLength * 0.2)
        let shaftEnd = direction * (safeLength - tipHeight)
        let endpoint = direction * safeLength

        let shaft = segment(
            from: .zero,
            to: shaftEnd,
            radius: 0.008,
            color: color
        )
        shaft.name = "world-axis-\(name)-shaft"
        root.addChildNode(shaft)

        let tipContainer = SCNNode()
        tipContainer.name = "world-axis-\(name)-tip"
        tipContainer.simdPosition = endpoint
        let cone = SCNCone(
            topRadius: 0,
            bottomRadius: 0.026,
            height: CGFloat(tipHeight)
        )
        cone.firstMaterial = constantMaterial(color: color)
        let coneNode = SCNNode(geometry: cone)
        coneNode.simdPosition = -direction * (tipHeight / 2)
        coneNode.simdOrientation = orientation(along: direction)
        tipContainer.addChildNode(coneNode)
        root.addChildNode(tipContainer)

        let text = SCNText(string: label, extrusionDepth: 0.002)
        text.font = .boldSystemFont(ofSize: 14)
        text.flatness = 0.2
        text.firstMaterial = constantMaterial(color: color)
        let labelNode = SCNNode(geometry: text)
        labelNode.name = "world-axis-\(name)-label"
        labelNode.simdPosition = endpoint + direction * 0.06
        labelNode.simdScale = SIMD3<Float>(repeating: 0.006)
        labelNode.constraints = [SCNBillboardConstraint()]
        root.addChildNode(labelNode)
    }

    private static func segment(
        from start: SIMD3<Float>,
        to end: SIMD3<Float>,
        radius: CGFloat,
        color: UIColor
    ) -> SCNNode {
        let direction = end - start
        let height = max(simd_length(direction), 0.0001)
        let cylinder = SCNCylinder(radius: radius, height: CGFloat(height))
        cylinder.firstMaterial = constantMaterial(color: color)
        let node = SCNNode(geometry: cylinder)
        node.simdPosition = (start + end) / 2
        node.simdOrientation = orientation(along: direction)
        return node
    }

    private static func orientation(
        along direction: SIMD3<Float>
    ) -> simd_quatf {
        let normalized = simd_normalize(direction)
        let up = SIMD3<Float>(0, 1, 0)
        let dot = max(-1, min(1, simd_dot(up, normalized)))
        let axis = simd_cross(up, normalized)
        guard simd_length_squared(axis) > 1e-8 else {
            return dot >= 0
                ? simd_quatf(angle: 0, axis: up)
                : simd_quatf(angle: .pi, axis: SIMD3<Float>(1, 0, 0))
        }
        return simd_quatf(
            angle: acos(dot),
            axis: simd_normalize(axis)
        )
    }

    private static func constantMaterial(color: UIColor) -> SCNMaterial {
        let material = SCNMaterial()
        material.diffuse.contents = color
        material.emission.contents = color.withAlphaComponent(0.18)
        material.lightingModel = .constant
        material.isDoubleSided = true
        return material
    }
}
