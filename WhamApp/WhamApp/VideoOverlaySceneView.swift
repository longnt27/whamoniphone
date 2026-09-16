import SceneKit
import SwiftUI
import UIKit

final class VideoOverlaySceneView: SCNView {
    override init(frame: CGRect, options: [String: Any]? = nil) {
        super.init(frame: frame, options: options)
        configureTransparency()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        configureTransparency()
    }

    private func configureTransparency() {
        isOpaque = false
        backgroundColor = .clear
        scene = SCNScene()
        scene?.background.contents = UIColor.clear
        antialiasingMode = .multisampling4X
    }
}

struct TransparentSceneView: UIViewRepresentable {
    let scene: SCNScene
    let pointOfView: SCNNode

    func makeUIView(context: Context) -> VideoOverlaySceneView {
        let view = VideoOverlaySceneView(frame: .zero)
        view.scene = scene
        view.scene?.background.contents = UIColor.clear
        view.pointOfView = pointOfView
        view.isPlaying = true
        view.rendersContinuously = false
        return view
    }

    func updateUIView(_ view: VideoOverlaySceneView, context: Context) {
        view.scene = scene
        view.scene?.background.contents = UIColor.clear
        view.pointOfView = pointOfView
        view.setNeedsDisplay()
    }
}
