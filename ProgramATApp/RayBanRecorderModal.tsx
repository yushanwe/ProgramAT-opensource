/**
 * RayBanRecorderModal
 *
 * Displays a live Meta Ray-Ban camera preview fullscreen and records it —
 * along with microphone audio — via iOS screen recording (ReplayKit). Because
 * ReplayKit captures whatever is on screen, showing the Ray-Ban preview while
 * recording produces a video that contains the glasses POV + the user's
 * narration, with no custom frame-stitching or audio mixing required.
 *
 * Flow:
 *  1. Modal opens → start Ray-Ban stream → poll frames → show live preview
 *  2. User taps Start Recording → iOS system prompt → recording begins
 *  3. User taps Stop → stopScreenRecordingAndSave → fetchMostRecentVideoPath
 *  4. onVideoRecorded(path) is called → modal closes
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  Image,
  Modal,
  ActivityIndicator,
  AccessibilityInfo,
  NativeModules,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useTheme } from './ThemeContext';

const { MetaWearablesModule, ScreenRecordingModule: SRM } = NativeModules as {
  MetaWearablesModule?: {
    startRayBanStream?: () => Promise<boolean>;
    stopRayBanStream?: () => Promise<boolean>;
    captureRayBanFrame?: () => Promise<{ base64: string; width: number; height: number }>;
  };
  ScreenRecordingModule?: {
    startScreenRecording: () => Promise<boolean>;
    stopScreenRecordingAndSave: () => Promise<boolean>;
    fetchMostRecentVideoPath: () => Promise<string>;
  };
};

interface RayBanRecorderModalProps {
  visible: boolean;
  onVideoRecorded: (filePath: string) => void;
  onCancel: () => void;
}

type RecorderState = 'starting' | 'ready' | 'recording' | 'stopping' | 'error';

export default function RayBanRecorderModal({
  visible,
  onVideoRecorded,
  onCancel,
}: RayBanRecorderModalProps) {
  const { theme } = useTheme();
  const [state, setState] = useState<RecorderState>('starting');
  const [previewUri, setPreviewUri] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isRecordingRef = useRef(false);
  const frameInFlightRef = useRef(false);

  const stopPolling = useCallback(() => {
    if (pollIntervalRef.current !== null) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  }, []);

  const stopRayBanCleanup = useCallback(async () => {
    stopPolling();
    setPreviewUri(null);
    try {
      await MetaWearablesModule?.stopRayBanStream?.();
    } catch {
      // best-effort
    }
  }, [stopPolling]);

  // Start/stop the stream and frame polling whenever visibility changes.
  useEffect(() => {
    if (!visible) {
      setState('starting');
      setErrorMessage(null);
      return;
    }

    let cancelled = false;

    const start = async () => {
      setState('starting');
      setErrorMessage(null);

      try {
        await MetaWearablesModule?.startRayBanStream?.();
      } catch (err: any) {
        if (!cancelled) {
          setErrorMessage(
            'Could not start the Ray-Ban stream. Make sure your glasses are connected and try again.',
          );
          setState('error');
        }
        return;
      }

      if (cancelled) return;

      pollIntervalRef.current = setInterval(async () => {
        if (cancelled || frameInFlightRef.current) return;
        frameInFlightRef.current = true;
        try {
          const frame = await MetaWearablesModule?.captureRayBanFrame?.();
          if (frame?.base64 && !cancelled) {
            setPreviewUri(frame.base64);
          }
        } catch {
          // individual frame failures are non-fatal
        } finally {
          frameInFlightRef.current = false;
        }
      }, 200);

      if (!cancelled) {
        setState('ready');
        AccessibilityInfo.announceForAccessibility(
          'Ray-Ban camera preview ready. Double tap Start Recording to begin.',
        );
      }
    };

    start();

    return () => {
      cancelled = true;
      frameInFlightRef.current = false;
      stopPolling();
    };
  }, [visible, stopPolling]);

  const handleStartRecording = async () => {
    try {
      await SRM?.startScreenRecording?.();
      isRecordingRef.current = true;
      setState('recording');
      AccessibilityInfo.announceForAccessibility(
        'Recording started. Look through your glasses and narrate. Double tap Stop when finished.',
      );
    } catch (err: any) {
      const msg =
        err?.code === 'recording_permission_denied'
          ? 'Screen recording permission was declined.'
          : 'Could not start screen recording. Please try again.';
      setErrorMessage(msg);
      setState('error');
    }
  };

  const handleStopRecording = async () => {
    setState('stopping');
    try {
      await SRM?.stopScreenRecordingAndSave?.();
      isRecordingRef.current = false;
      const path = await SRM?.fetchMostRecentVideoPath?.();
      if (!path) throw new Error('No path returned after saving.');
      await stopRayBanCleanup();
      onVideoRecorded(path);
    } catch {
      setErrorMessage('Failed to save the recording. Please try again.');
      setState('error');
    }
  };

  const handleCancel = useCallback(async () => {
    if (isRecordingRef.current) {
      try {
        await SRM?.stopScreenRecordingAndSave?.();
      } catch {
        // ignore — user cancelled, we don't care about the saved file
      }
      isRecordingRef.current = false;
    }
    await stopRayBanCleanup();
    onCancel();
  }, [stopRayBanCleanup, onCancel]);

  if (!visible) return null;

  return (
    <Modal visible={visible} animationType="slide" statusBarTranslucent>
      <SafeAreaView style={styles.container} edges={['top', 'bottom']}>

        {/* Preview area */}
        {previewUri ? (
          <Image
            source={{ uri: previewUri }}
            style={styles.preview}
            resizeMode="cover"
            accessible={true}
            accessibilityLabel="Meta Ray-Ban camera preview"
            accessibilityHint="Live view from your Ray-Ban glasses"
          />
        ) : (
          <View style={styles.previewPlaceholder}>
            {state === 'starting' && (
              <>
                <ActivityIndicator size="large" color="#fff" />
                <Text style={styles.placeholderText}>Connecting to Ray-Ban…</Text>
              </>
            )}
            {state === 'error' && (
              <Text style={[styles.placeholderText, { color: '#ff6b6b' }]}>
                {errorMessage ?? 'Something went wrong.'}
              </Text>
            )}
          </View>
        )}

        {/* REC badge overlaid on preview */}
        {state === 'recording' && (
          <View
            style={styles.recordingBadge}
            accessible={true}
            accessibilityLabel="Recording in progress">
            <View style={styles.recordingDot} />
            <Text style={styles.recordingBadgeText}>REC</Text>
          </View>
        )}

        {/* Bottom controls */}
        <View style={styles.controls}>
          {state === 'ready' && (
            <TouchableOpacity
              style={[styles.primaryButton, styles.recordButton]}
              onPress={handleStartRecording}
              accessible={true}
              accessibilityRole="button"
              accessibilityLabel="Start recording"
              accessibilityHint="Starts screen recording with your microphone. Look through your glasses and narrate.">
              <Text style={styles.primaryButtonText}>⏺  Start Recording</Text>
            </TouchableOpacity>
          )}

          {state === 'recording' && (
            <TouchableOpacity
              style={[styles.primaryButton, styles.stopButton]}
              onPress={handleStopRecording}
              accessible={true}
              accessibilityRole="button"
              accessibilityLabel="Stop recording"
              accessibilityHint="Stops recording and attaches the video to your message">
              <Text style={[styles.primaryButtonText, { color: '#111' }]}>⏹  Stop Recording</Text>
            </TouchableOpacity>
          )}

          {state === 'stopping' && (
            <View
              style={styles.stoppingRow}
              accessible={true}
              accessibilityLabel="Saving recording, please wait">
              <ActivityIndicator size="small" color="#fff" />
              <Text style={styles.placeholderText}>Saving recording…</Text>
            </View>
          )}

          {state !== 'stopping' && (
            <TouchableOpacity
              style={styles.cancelButton}
              onPress={handleCancel}
              accessible={true}
              accessibilityRole="button"
              accessibilityLabel="Cancel"
              accessibilityHint="Discards any recording and returns to the message composer">
              <Text style={styles.cancelButtonText}>Cancel</Text>
            </TouchableOpacity>
          )}
        </View>

      </SafeAreaView>
    </Modal>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  preview: {
    flex: 1,
  },
  previewPlaceholder: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 16,
  },
  placeholderText: {
    color: '#fff',
    fontSize: 16,
    textAlign: 'center',
    paddingHorizontal: 24,
  },
  recordingBadge: {
    position: 'absolute',
    top: 64,
    left: 16,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: 'rgba(0,0,0,0.55)',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 6,
  },
  recordingDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: '#e53e3e',
  },
  recordingBadgeText: {
    color: '#fff',
    fontWeight: '700',
    fontSize: 13,
    letterSpacing: 1,
  },
  controls: {
    paddingHorizontal: 24,
    paddingVertical: 24,
    alignItems: 'center',
    gap: 12,
  },
  primaryButton: {
    paddingVertical: 16,
    paddingHorizontal: 32,
    borderRadius: 30,
    minWidth: 220,
    alignItems: 'center',
  },
  recordButton: {
    backgroundColor: '#e53e3e',
  },
  stopButton: {
    backgroundColor: '#fff',
  },
  primaryButtonText: {
    color: '#fff',
    fontSize: 17,
    fontWeight: '700',
  },
  stoppingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  cancelButton: {
    paddingVertical: 12,
    paddingHorizontal: 24,
  },
  cancelButtonText: {
    color: 'rgba(255,255,255,0.65)',
    fontSize: 16,
  },
});
