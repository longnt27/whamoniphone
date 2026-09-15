import SwiftUI

struct BenchmarkView: View {
    @StateObject private var benchmark = WhamBenchmark()
    @State private var passes = 5

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Text("The validation-selected pipeline over eight real frames: YOLO26m-pose, HMR2.0-S, the learned token adapter, WHAM_I, recurrent WHAM, light causal smoothing, SMPL mesh generation, and world-trajectory refinement. One complete untimed warm-up runs first. Accuracy comes from the separate 3DPW evaluation.")
                        .font(.callout)
                } header: {
                    Text("On-device workload")
                }

                Section {
                    Picker("Measured passes", selection: $passes) {
                        Text("1 × 8 frames").tag(1)
                        Text("5 × 8 frames").tag(5)
                        Text("10 × 8 frames").tag(10)
                    }

                    Button {
                        benchmark.run(passes: passes)
                    } label: {
                        Label(
                            benchmark.isRunning
                                ? "Running…"
                                : "Run validated pipeline test",
                            systemImage: "gauge.with.dots.needle.67percent"
                        )
                    }
                    .disabled(benchmark.isRunning)

                    if benchmark.isRunning {
                        ProgressView()
                    }
                    Text(benchmark.status)
                        .font(.caption)
                        .foregroundStyle(benchmark.status.hasPrefix("Failed") ? .red : .secondary)
                }

                if let report = benchmark.report {
                    Section("Device") {
                        LabeledContent("Hardware", value: report.device)
                        LabeledContent("OS", value: report.operatingSystem)
                        LabeledContent(
                            "Thermal",
                            value: "\(report.thermalStateBefore) → \(report.thermalStateAfter)"
                        )
                        LabeledContent("Detector", value: report.poseDetector)
                        LabeledContent("Test set", value: "\(report.sampleFrames) frames × \(report.passes)")
                        LabeledContent("Detections", value: "\(report.personDetections)/\(report.processedSourceFrames)")
                        LabeledContent(
                            "Selected models",
                            value: ByteCountFormatter.string(
                                fromByteCount: Int64(report.selectedPipelineModelBytes),
                                countStyle: .file
                            )
                        )
                        LabeledContent(
                            "Peak memory",
                            value: ByteCountFormatter.string(
                                fromByteCount: Int64(report.peakResidentMemoryBytes),
                                countStyle: .memory
                            )
                        )
                        LabeledContent(
                            "Image connected",
                            value: report.imageFeatureConnectionPassed ? "PASS" : "FAIL"
                        )
                        LabeledContent(
                            "Smoothing active",
                            value: report.outputSmoothingConnectionPassed ? "PASS" : "FAIL"
                        )
                        LabeledContent(
                            "Smoothing",
                            value: String(
                                format: "pose %.2f · shape %.2f · ≈%.1f ms delay",
                                report.smoothingPoseAlpha,
                                report.smoothingShapeAlpha,
                                report.approximateSmoothingDelayMillisecondsAt30FPS
                            )
                        )
                    }

                    Section("Latency") {
                        ForEach(report.metrics) { metric in
                            VStack(alignment: .leading, spacing: 3) {
                                Text(metric.stage)
                                    .font(.system(.body, design: .monospaced))
                                Text(String(format: "n=%d  ·  mean %.2f ms  ·  p50 %.2f  ·  p95 %.2f", metric.samples, metric.meanMilliseconds, metric.p50Milliseconds, metric.p95Milliseconds))
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }

                    Section("Scope") {
                        Text(report.scope)
                            .font(.caption)
                    }

                    if let url = benchmark.reportURL {
                        Section {
                            ShareLink(item: url) {
                                Label("Share benchmark JSON", systemImage: "square.and.arrow.up")
                            }
                        }
                    }
                }
            }
            .navigationTitle("WHAM Benchmark")
        }
    }
}
