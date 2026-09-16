//
//  ContentView.swift
//  WhamApp
//
//  Created by admin on 20/3/26.
//

import SwiftUI

struct MainCameraView: View {
    @StateObject var manager = ARWhamManager()
    @State private var isShowingLibrary = false

    var body: some View {
        ZStack {
            ARViewContainer(session: manager.session)
                .ignoresSafeArea()

            if manager.isRecording {
                VStack {
                    Text(manager.formattedTime)
                        .font(.system(size: 24, weight: .bold, design: .monospaced))
                        .foregroundColor(.white)
                        .padding(.horizontal, 15)
                        .padding(.vertical, 5)
                        .background(Color.black.opacity(0.6))
                        .cornerRadius(10)
                        .padding(.top, 60)
                    Spacer()
                }
            }

            VStack {
                Spacer()

                if !manager.isRecording {
                    Text("Di chuyển iPhone để calibrate SLAM...")
                        .font(.caption)
                        .foregroundColor(.white)
                        .padding(5)
                        .background(Color.black.opacity(0.4))
                        .cornerRadius(5)
                        .padding(.bottom, 10)
                }

                HStack(spacing: 40) {
                    Button(action: { isShowingLibrary = true }) {
                        Group {
                            if let thumb = manager.lastThumbnail {
                                Image(uiImage: thumb)
                                    .resizable()
                                    .aspectRatio(contentMode: .fill)
                            } else {
                                Color.black.opacity(0.5)
                            }
                        }
                        .frame(width: 65, height: 65)
                        .cornerRadius(12)
                        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.white, lineWidth: 2))
                        .clipped()
                    }

                    Button(action: {
                        manager.toggleRecording()
                    }) {
                        ZStack {
                            Circle()
                                .stroke(Color.white, lineWidth: 4)
                                .frame(width: 80, height: 80)

                            if manager.isRecording {
                                RoundedRectangle(cornerRadius: 6)
                                    .fill(Color.red)
                                    .frame(width: 35, height: 35)
                            } else {
                                Circle()
                                    .fill(Color.red)
                                    .frame(width: 68, height: 68)
                            }
                        }
                    }

                    Button(action: { manager.resetTracking() }) {
                        Image(systemName: "arrow.counterclockwise.circle")
                            .font(.system(size: 30))
                            .foregroundColor(.white)
                    }
                }
                .padding(.bottom, 40)
            }
        }
        .sheet(isPresented: $isShowingLibrary, onDismiss: { manager.resume() }) {
            VideoLibraryView()
                .onAppear{ manager.pause() }
        }
    }
}

struct ContentView: View {
    var body: some View {
        MainCameraView()
    }
}
