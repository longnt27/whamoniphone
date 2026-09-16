//
//  WhamApp.swift
//  WhamApp
//
//  Created by admin on 20/3/26.
//

import SwiftUI

@main
struct WhamApp: App {
    var body: some Scene {
        WindowGroup {
            AppRootView()
        }
    }
}

private enum AppTab {
    case camera
    case benchmark
}

private struct AppRootView: View {
    @State private var selection: AppTab = .camera

    var body: some View {
        ZStack(alignment: .bottom) {
            Group {
                switch selection {
                case .camera:
                    MainCameraView()
                case .benchmark:
                    BenchmarkView()
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            PersistentAppTabBar(selection: $selection)
                .padding(.horizontal, 22)
                .padding(.bottom, 8)
        }
    }
}

private struct PersistentAppTabBar: View {
    @Binding var selection: AppTab

    var body: some View {
        HStack(spacing: 0) {
            tabButton(
                title: "AR Camera",
                systemImage: "camera.viewfinder",
                tab: .camera
            )
            tabButton(
                title: "Benchmark",
                systemImage: "gauge.with.dots.needle.67percent",
                tab: .benchmark
            )
        }
        .padding(6)
        .frame(maxWidth: 370)
        .background(.ultraThinMaterial, in: Capsule())
        .overlay {
            Capsule()
                .stroke(Color.white.opacity(0.16), lineWidth: 0.75)
        }
        .shadow(color: .black.opacity(0.34), radius: 14, y: 6)
    }

    private func tabButton(
        title: String,
        systemImage: String,
        tab: AppTab
    ) -> some View {
        Button {
            selection = tab
        } label: {
            HStack(spacing: 7) {
                Image(systemName: systemImage)
                    .font(.system(size: 17, weight: .semibold))
                Text(title)
                    .font(.subheadline.weight(.semibold))
            }
            .foregroundStyle(
                selection == tab ? Color.white : Color.white.opacity(0.76)
            )
            .padding(.vertical, 10)
            .frame(maxWidth: .infinity)
            .background {
                if selection == tab {
                    Capsule().fill(Color.blue)
                }
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(selection == tab ? .isSelected : [])
    }
}
