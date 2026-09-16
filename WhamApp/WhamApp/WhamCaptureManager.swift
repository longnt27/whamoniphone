//
//  WhamCaptureManager.swift
//  WhamApp
//
//  Created by admin on 20/3/26.
//

import AVFoundation
import CoreMotion
import UIKit

class WhamCaptureManager: NSObject, ObservableObject, AVCaptureFileOutputRecordingDelegate {
    let session = AVCaptureSession()
    private let movieOutput = AVCaptureMovieFileOutput()
    private let motionManager = CMMotionManager()

    @Published var isRecording = false
    @Published var lastThumbnail: UIImage?
    @Published var recordedSeconds: Int = 0

    private var timer: Timer?
    private var currentID: UUID?
    private var gyroData: [[String: Any]] = []

    override init() {
        super.init()
        setupSession()
        loadLatestThumbnail()
    }

    private func setupSession() {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device) else {
            print("❌ Không tìm thấy camera")
            return
        }

        session.beginConfiguration()
        if session.canAddInput(input) { session.addInput(input) }
        if session.canAddOutput(movieOutput) { session.addOutput(movieOutput) }
        do {
            try device.lockForConfiguration()
            let frameDuration = CMTime(value: 1, timescale: 30)
            device.activeVideoMinFrameDuration = frameDuration
            device.activeVideoMaxFrameDuration = frameDuration
            device.unlockForConfiguration()
        } catch {
            print("⚠️ Could not lock camera to 30 FPS: \(error)")
        }
        session.commitConfiguration()

        DispatchQueue.global(qos: .userInitiated).async {
            self.session.startRunning()
        }
    }

    private func loadLatestThumbnail() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        do {
            let files = try FileManager.default.contentsOfDirectory(at: docs, includingPropertiesForKeys: [.creationDateKey])
            let sortedVideos = files.filter { $0.pathExtension == "mp4" }
                .sorted { (url1, url2) -> Bool in
                    let date1 = (try? url1.resourceValues(forKeys: [.creationDateKey]))?.creationDate ?? Date.distantPast
                    let date2 = (try? url2.resourceValues(forKeys: [.creationDateKey]))?.creationDate ?? Date.distantPast
                    return date1 > date2
                }

            if let latestVideo = sortedVideos.first {
                generateThumbnail(for: latestVideo)
            }
        } catch {
            print("⚠️ Lỗi load video cũ: \(error)")
        }
    }

    func toggleRecording() {
        if isRecording { stop() } else { start() }
    }

    private func start() {
        let id = UUID()
        currentID = id
        gyroData.removeAll()
        recordedSeconds = 0
        isRecording = true

        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
            self.recordedSeconds += 1
        }

        if motionManager.isGyroAvailable {
            motionManager.gyroUpdateInterval = 1.0 / 60.0
            motionManager.startGyroUpdates(to: .main) { data, _ in
                guard let data = data else { return }
                let frame: [String: Any] = [
                    "t": data.timestamp,
                    "x": data.rotationRate.x,
                    "y": data.rotationRate.y,
                    "z": data.rotationRate.z
                ]
                self.gyroData.append(frame)
            }
        }

        let url = getURL(id: id, ext: "mp4")
        movieOutput.startRecording(to: url, recordingDelegate: self)
    }

    private func stop() {
        movieOutput.stopRecording()
        motionManager.stopGyroUpdates()
        timer?.invalidate()
        timer = nil
        if let id = currentID { saveJSON(id: id) }
        isRecording = false
    }

    func formattedTime() -> String {
        let minutes = recordedSeconds / 60
        let seconds = recordedSeconds % 60
        return String(format: "%02d:%02d", minutes, seconds)
    }

    private func saveJSON(id: UUID) {
        let url = getURL(id: id, ext: "json")
        if let data = try? JSONSerialization.data(withJSONObject: gyroData, options: .prettyPrinted) {
            try? data.write(to: url)
            print("✅ Đã lưu \(gyroData.count) frames Gyro cho \(id.uuidString)")
        }
    }

    func getURL(id: UUID, ext: String) -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("\(id.uuidString).\(ext)")
    }

    func fileOutput(_ output: AVCaptureFileOutput, didFinishRecordingTo url: URL, from connections: [AVCaptureConnection], error: Error?) {
        generateThumbnail(for: url)
    }

    private func generateThumbnail(for url: URL) {
        let asset = AVURLAsset(url: url)
        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true
        let time = CMTime(seconds: 0, preferredTimescale: 600)

        gen.generateCGImagesAsynchronously(forTimes: [NSValue(time: time)]) { _, cgImage, _, _, _ in
            if let cgImage = cgImage {
                DispatchQueue.main.async {
                    self.lastThumbnail = UIImage(cgImage: cgImage)
                }
            }
        }
    }
}
