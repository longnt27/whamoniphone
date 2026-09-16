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
        .padding(.top, 9)
        .padding(.bottom, 4)
        .background(.ultraThinMaterial)
        .overlay(alignment: .top) {
            Divider().opacity(0.55)
        }
    }

    private func tabButton(
        title: String,
        systemImage: String,
        tab: AppTab
    ) -> some View {
        Button {
            selection = tab
        } label: {
            VStack(spacing: 3) {
                Image(systemName: systemImage)
                    .font(.system(size: 21, weight: .semibold))
                Text(title)
                    .font(.caption2.weight(.semibold))
            }
            .foregroundStyle(selection == tab ? Color.blue : Color.white)
            .frame(maxWidth: .infinity)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(selection == tab ? .isSelected : [])
    }
}
