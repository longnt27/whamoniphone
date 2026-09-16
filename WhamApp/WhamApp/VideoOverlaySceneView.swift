import AVKit
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
        preferredFramesPerSecond = 30
        rendersContinuously = true
    }
}

final class VideoOverlayPlayerViewController: UIViewController {
    let playerController = AVPlayerViewController()
    let overlayView = VideoOverlaySceneView(frame: .zero)

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black

        addChild(playerController)
        playerController.view.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(playerController.view)
        NSLayoutConstraint.activate([
            playerController.view.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            playerController.view.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            playerController.view.topAnchor.constraint(equalTo: view.topAnchor),
            playerController.view.bottomAnchor.constraint(equalTo: view.bottomAnchor),
        ])
        playerController.didMove(toParent: self)

        overlayView.translatesAutoresizingMaskIntoConstraints = false
        overlayView.isUserInteractionEnabled = false
        view.addSubview(overlayView)
        NSLayoutConstraint.activate([
            overlayView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            overlayView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            overlayView.topAnchor.constraint(equalTo: view.topAnchor),
            overlayView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
        ])
    }

    func configure(
        player: AVPlayer,
        scene: SCNScene,
        pointOfView: SCNNode
    ) {
        loadViewIfNeeded()
        playerController.player = player
        overlayView.scene = scene
        overlayView.scene?.background.contents = UIColor.clear
        overlayView.pointOfView = pointOfView
        overlayView.setNeedsDisplay()
    }
}

struct VideoOverlayPlayerView: UIViewControllerRepresentable {
    let player: AVPlayer
    let scene: SCNScene
    let pointOfView: SCNNode

    func makeUIViewController(context: Context) -> VideoOverlayPlayerViewController {
        let controller = VideoOverlayPlayerViewController()
        controller.configure(
            player: player,
            scene: scene,
            pointOfView: pointOfView
        )
        return controller
    }

    func updateUIViewController(
        _ controller: VideoOverlayPlayerViewController,
        context: Context
    ) {
        controller.configure(
            player: player,
            scene: scene,
            pointOfView: pointOfView
        )
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
