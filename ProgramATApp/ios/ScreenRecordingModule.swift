//
//  ScreenRecordingModule.swift
//  ProgramATApp
//
//  Native bridge for recording a usage session's screen with ReplayKit and
//  saving the finished clip to the user's Photos library. Used by the Tools
//  screen when the user arms "Record this usage session" before running a
//  tool.
//

import Foundation
import React
import ReplayKit
import Photos

@objc(ScreenRecordingModule)
class ScreenRecordingModule: NSObject {

    private var isRecording = false
    private var outputURL: URL?

    @objc
    static func requiresMainQueueSetup() -> Bool {
        return false
    }

    @objc
    func startScreenRecording(
        _ resolve: @escaping RCTPromiseResolveBlock,
        rejecter reject: @escaping RCTPromiseRejectBlock
    ) {

        guard RPScreenRecorder.shared().isAvailable else {
            reject(
                "recorder_unavailable",
                "Screen recording is not available on this device.",
                nil
            )
            return
        }

        guard !self.isRecording else {
            reject(
                "already_recording",
                "A screen recording is already in progress.",
                nil
            )
            return
        }

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("ProgramAT-\(UUID().uuidString).mp4")
        try? FileManager.default.removeItem(at: url)
        self.outputURL = url

        RPScreenRecorder.shared().startRecording(withMicrophoneEnabled: false) { error in

            if let error = error {
                let nsError = error as NSError
                if nsError.domain == RPRecordingErrorDomain,
                   nsError.code == RPRecordingErrorCode.userDeclined.rawValue {
                    reject(
                        "recording_permission_denied",
                        "Screen recording permission was declined.",
                        error
                    )
                } else {
                    reject(
                        "start_recording_failed",
                        error.localizedDescription,
                        error
                    )
                }
                return
            }

            self.isRecording = true
            resolve(true)
        }
    }

    @objc
    func stopScreenRecordingAndSave(
        _ resolve: @escaping RCTPromiseResolveBlock,
        rejecter reject: @escaping RCTPromiseRejectBlock
    ) {

        guard self.isRecording, let url = self.outputURL else {
            reject(
                "not_recording",
                "No screen recording is in progress.",
                nil
            )
            return
        }

        RPScreenRecorder.shared().stopRecording(withOutput: url) { error in

            self.isRecording = false

            if let error = error {
                reject(
                    "stop_recording_failed",
                    error.localizedDescription,
                    error
                )
                return
            }

            PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in

                guard status == .authorized || status == .limited else {
                    try? FileManager.default.removeItem(at: url)
                    reject(
                        "photos_permission_denied",
                        "Photos access was denied; the recording could not be saved.",
                        nil
                    )
                    return
                }

                PHPhotoLibrary.shared().performChanges({
                    PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: url)
                }) { success, saveError in
                    try? FileManager.default.removeItem(at: url)

                    if success {
                        resolve(true)
                    } else {
                        reject(
                            "save_to_photos_failed",
                            saveError?.localizedDescription ?? "Unknown error saving recording.",
                            saveError
                        )
                    }
                }
            }
        }
    }

    @objc
    func isScreenRecordingActive(
        _ resolve: @escaping RCTPromiseResolveBlock,
        rejecter reject: @escaping RCTPromiseRejectBlock
    ) {
        resolve(self.isRecording)
    }
}
