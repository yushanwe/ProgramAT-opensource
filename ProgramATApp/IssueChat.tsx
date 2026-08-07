/**
 * IssueChat Component
 * Chat-style interface for creating or updating a GitHub issue describing a
 * visual assistive tool. Replaces the single-text-field TextInput screen.
 *
 * Every turn (text, video, ideation answers, brainstorm choices) goes through
 * the existing HTTP endpoints (/submit-creation, /submit-update,
 * /brainstorm-next-question) rather than the WebSocket text path, since the
 * WS-side server state (selected_issue, pending_ideation, incomplete_issue)
 * is global across all connected clients, not scoped per conversation.
 */

import React, { useState, useRef, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  ScrollView,
  TextInput as RNTextInput,
  Platform,
  Keyboard,
  KeyboardAvoidingView,
  ActivityIndicator,
  AccessibilityInfo,
  Modal,
  NativeModules,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { launchImageLibrary } from 'react-native-image-picker';
import { useTheme } from './ThemeContext';
import WebSocketService from './WebSocketService';
import VideoRecorderModal from './VideoRecorderModal';
import { isBrainstormingEnabled, isBasicModeEnabled } from './Settings';
import { ClaudeProgressResponse, IssueChatItem, RetryDescriptor } from './IssueChatTypes';
import { fetchClaudeProgress, submitCreation, submitUpdate, nextBrainstormQuestion, askBrainstormAgent } from './IssueSubmissionService';
import {
  buildClaudeAccessibilitySections,
  buildClaudeAccessibilityLabel,
  getChangedClaudeAnnouncement,
  sanitizeClaudeCommentBody,
} from './claudeCommentSanitizer';
import TextToSpeechService from './TextToSpeechService';
import BeepService from './BeepService';
import RayBanRecorderModal from './RayBanRecorderModal';

interface IssueChatProps {
  serverFeedback?: string;
  selectedIssue?: {number: number; title: string} | null;
  onNewIssue?: () => void;
  onBack?: () => void;
  showBackButton?: boolean;
}

type Awaiting = 'answer' | 'choice' | 'clarification' | null;
type ProgressTarget = {
  mode: 'create' | 'update';
  issueNumber?: number | null;
  prNumber?: number | null;
  commentId?: number | null;
  afterCommentId?: number | null;
  afterTimestamp?: string | null;
};
const CLAUDE_POLL_INTERVAL_MS = 5000;
const CLAUDE_LOADING_AUDIO_INTERVAL_MS = 6000;
export interface AutoScrollState {
  itemCount: number;
  lastItemId: string | null;
}

export function shouldAutoScrollForNewItems(previousState: AutoScrollState, nextState: AutoScrollState): boolean {
  return nextState.itemCount > previousState.itemCount
    || (nextState.lastItemId !== null && nextState.lastItemId !== previousState.lastItemId);
}

function buildProgressAnnouncement(status: string, label: string): string {
  if (status === 'completed') return `Completed: ${label}`;
  if (status === 'in_progress') return `In progress: ${label}`;
  if (status === 'failed') return `Failed: ${label}`;
  return `Pending: ${label}`;
}

function summarizeClaudeAnnouncement(progress: ClaudeProgressResponse): string {
  if (progress.status === 'waiting_for_comment') return 'Claude has not posted a progress comment yet.';
  if (progress.status === 'failed') return 'Claude posted a failed status update.';
  if (progress.status === 'completed') return 'Claude posted a completed status update.';
  return 'Claude updated progress.';
}

function hasClaudeCompletionLikeText(progress: ClaudeProgressResponse | null): boolean {
  if (!progress) return false;
  const body = sanitizeClaudeCommentBody(progress.body).toLowerCase();
  const message = (progress.message || '').toLowerCase();
  const combined = `${body}\n${message}`;
  return combined.includes('implementation summary')
    || combined.includes('summary')
    || combined.includes('finished')
    || combined.includes('completed')
    || combined.includes('done');
}

function formatClaudeBody(body: string | undefined): string {
  const value = sanitizeClaudeCommentBody(body);
  return value || 'Claude has not posted a progress comment yet.';
}

function extractCreateToolName(summary: string | null): string | null {
  if (!summary) return null;
  const normalized = summary.trim();
  const patterns = [
    /tool name\s*:\s*([^\n.]+)/i,
    /title\s*:\s*([^\n.]+)/i,
    /name\s*:\s*([^\n.]+)/i,
  ];
  for (const pattern of patterns) {
    const candidate = normalized.match(pattern)?.[1]?.trim();
    if (candidate) {
      return candidate;
    }
  }
  return null;
} 
  
const { ScreenRecordingModule } = NativeModules as {
  ScreenRecordingModule?: {
    fetchMostRecentVideoPath?: () => Promise<string>;
  };
};

const hasRayBan = !!(NativeModules as any).MetaWearablesModule?.startRayBanStream;

/**
 * Build the short VoiceOver announcement for the "What I understand" card.
 * On first arrival (no integration note) we extract just the first sentence.
 * On updates we use the integration note — it is already a single sentence
 * describing only what changed.
 * Always prefixed with the card label so the user knows what they're hearing.
 */
function buildUnderstandingAnnouncement(summary: string, integrationNote?: string | null): string {
  const label = 'What I understand: ';
  if (integrationNote) {
    return label + integrationNote;
  }
  // Extract first sentence only.
  const match = summary.match(/^[^.!?]+[.!?]/);
  const firstSentence = match ? match[0] : summary;
  return label + firstSentence;
}

export default function IssueChat({
  selectedIssue,
  onBack,
  showBackButton = false,
}: IssueChatProps) {
  const { theme } = useTheme();
  const isCreateMode = !selectedIssue;

  const [items, setItems] = useState<IssueChatItem[]>([]);
  const [composeText, setComposeText] = useState('');
  const [stagedVideoUri, setStagedVideoUri] = useState<string | null>(null);
  const [stagedVideoSource, setStagedVideoSource] = useState<'phone' | 'rayban' | 'library' | null>(null);
  const [isVideoRecorderOpen, setIsVideoRecorderOpen] = useState(false);
  const [isRayBanRecorderOpen, setIsRayBanRecorderOpen] = useState(false);
  const [isPickingFromLibrary, setIsPickingFromLibrary] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [progressText, setProgressText] = useState<string | null>(null);
  const [claudeProgress, setClaudeProgress] = useState<ClaudeProgressResponse | null>(null);
  const [progressTarget, setProgressTarget] = useState<ProgressTarget | null>(null);

  const [activeToken, setActiveToken] = useState<string | null>(null);
  const [awaiting, setAwaiting] = useState<Awaiting>(null);
  const [understandingSummary, setUnderstandingSummary] = useState<string | null>(null);
  const [lastIntegrated, setLastIntegrated] = useState<string | null>(null);
  const [isSummaryModalOpen, setIsSummaryModalOpen] = useState(false);

  const [brainstormingEnabled, setBrainstormingEnabled] = useState(true);
  const [basicMode, setBasicMode] = useState(false);
  const brainstormingActive = brainstormingEnabled && !basicMode;

  const brainstormHistoryRef = useRef<Array<{question: string; answer: string}>>([]);
  const idCounterRef = useRef(0);
  const inFlightRef = useRef(false);
  const scrollViewRef = useRef<ScrollView>(null);
  const composeInputRef = useRef<RNTextInput>(null);
  const lastAnnouncedIdRef = useRef<string | null>(null);
  const lastAnnouncedProgressStepRef = useRef<string | null>(null);
  const progressPollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const claudeLoadingTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const claudeLoadingLoopTokenRef = useRef(0);
  const claudeProgressItemIdRef = useRef<string | null>(null);
  const previousClaudeBodyRef = useRef<string>('');
  const latestClaudePollRequestIdRef = useRef(0);
  const latestClaudeAppliedRequestIdRef = useRef(0);
  const latestClaudeAppliedUpdatedAtRef = useRef<number | null>(null);
  const hasClaudeMessageRef = useRef(false);
  const allowNextClaudeAutoFollowRef = useRef(false);
  const lastAutoScrollStateRef = useRef<{ itemCount: number; lastItemId: string | null }>({
    itemCount: 0,
    lastItemId: null,
  });
  const createToolName = extractCreateToolName(understandingSummary);
  const modeBannerText = isCreateMode
    ? createToolName
      ? `Mode: Creating ${createToolName}`
      : null
    : selectedIssue?.title
    ? `Mode: Updating ${selectedIssue.title}`
    : null;
  const isClaudeWaitingActive = !!claudeProgress
    && claudeProgress.status !== 'waiting_for_comment'
    && claudeProgress.status !== 'unavailable'
    && claudeProgress.status !== 'completed'
    && claudeProgress.status !== 'failed'
    && claudeProgress.status !== 'cancelled'
    && !hasClaudeCompletionLikeText(claudeProgress);

  const stopClaudeLoadingSound = () => {
    claudeLoadingLoopTokenRef.current += 1;
    if (claudeLoadingTimeoutRef.current) {
      clearTimeout(claudeLoadingTimeoutRef.current);
      claudeLoadingTimeoutRef.current = null;
    }
    BeepService.stopLoadingSound();
  };

  useEffect(() => {
    isBrainstormingEnabled().then(setBrainstormingEnabled).catch(() => setBrainstormingEnabled(true));
    isBasicModeEnabled().then(setBasicMode).catch(() => setBasicMode(false));
  }, []);

  // Announce the screen title on mount, matching ToolSelector's entry pattern.
  useEffect(() => {
    const timeout = setTimeout(() => {
      AccessibilityInfo.announceForAccessibility(
        isCreateMode ? 'Create issue' : `Updating ${selectedIssue?.title}`,
      );
    }, 100);
    return () => clearTimeout(timeout);
  }, [isCreateMode, selectedIssue?.title]);

  // Auto-scroll to the newest message only when a new chat item is appended.
  // In-place Claude progress updates intentionally preserve the user's position.
  useEffect(() => {
    const lastItem = items[items.length - 1];
    const nextState = {
      itemCount: items.length,
      lastItemId: lastItem?.id || null,
    };
    const previousState = lastAutoScrollStateRef.current;
    const appendedMessage = shouldAutoScrollForNewItems(previousState, nextState);

    const shouldAutoFollow = appendedMessage
      && scrollViewRef.current
      && (!hasClaudeMessageRef.current || allowNextClaudeAutoFollowRef.current);

    if (shouldAutoFollow) {
      setTimeout(() => {
        scrollViewRef.current?.scrollToEnd({ animated: true });
      }, 100);
      allowNextClaudeAutoFollowRef.current = false;
    }
    lastAutoScrollStateRef.current = nextState;
  }, [items, progressText, isSending]);

  // Loading sound while a request is in flight — video uploads can take
  // 15+ seconds to summarize, and the "Thinking…" bubble alone isn't audible.
  // Starts immediately (no delay) since requests here are never near-instant
  // and the user should hear confirmation from the moment they hit send.
  useEffect(() => {
    if (isSending) {
      BeepService.startLoadingSound();
    } else {
      BeepService.stopLoadingSound();
    }

    return () => {
      BeepService.stopLoadingSound();
    };
  }, [isSending]);

  // Announce new assistant-originated items. assistant-updated is skipped
  // because App.tsx's WS listener already speaks "Update sent to issue" for
  // the issue_updated broadcast that /submit-update still triggers.
  useEffect(() => {
    const last = items[items.length - 1];
    if (!last || last.id === lastAnnouncedIdRef.current) return;
    if (last.kind === 'assistant-question') {
      lastAnnouncedIdRef.current = last.id;
      // Announce the question and play an earcon. All question announcements
      // are handled here regardless of how the question arrived.
      AccessibilityInfo.announceForAccessibility(last.question);
      BeepService.playBeep(880, 120);
    } else if (last.kind === 'assistant-clarification-answer') {
      lastAnnouncedIdRef.current = last.id;
      // Same earcon as a structured question — this is equally new
      // assistant-originated spoken content the user should notice.
      AccessibilityInfo.announceForAccessibility(last.answer);
      BeepService.playBeep(880, 120);
    } else if (last.kind === 'assistant-choice-prompt') {
      lastAnnouncedIdRef.current = last.id;
      // Announce the choice prompt. Summary card is silently updated and
      // readable by VoiceOver navigation — no competing announcement needed.
      AccessibilityInfo.announceForAccessibility(last.text);
    } else if (last.kind === 'assistant-created') {
      lastAnnouncedIdRef.current = last.id;
      const note = last.videoSummarySkipped ? ' Video summarization was skipped.' : '';
      TextToSpeechService.speakWithInterrupt(`Issue ${last.issueNumber} created.${note}`);
      AccessibilityInfo.announceForAccessibility(`Issue ${last.issueNumber} created.${note}`);
    } else if (last.kind === 'assistant-error') {
      lastAnnouncedIdRef.current = last.id;
      TextToSpeechService.speakWithInterrupt(last.text);
      AccessibilityInfo.announceForAccessibility(last.text);
    }
  }, [items]);

  // Focus the compose field when the server asks a question — the user
  // should be able to answer immediately without hunting for the input.
  useEffect(() => {
    if (awaiting === 'answer') {
      const timeout = setTimeout(() => composeInputRef.current?.focus(), 150);
      return () => clearTimeout(timeout);
    } else if (awaiting === 'clarification') {
      // No new chat bubble carries this mode switch, so announce it
      // explicitly — otherwise a screen-reader user won't know the compose
      // bar's purpose just changed.
      AccessibilityInfo.announceForAccessibility('Ask the agent a question. Type your question and press send.');
      const timeout = setTimeout(() => composeInputRef.current?.focus(), 150);
      return () => clearTimeout(timeout);
    }
  }, [awaiting]);

  // Announce progress text updates to VoiceOver as they arrive.
  useEffect(() => {
    if (progressText) {
      AccessibilityInfo.announceForAccessibility(progressText);
    }
  }, [progressText]);

  useEffect(() => {
    if (!claudeProgress) return;
    const changedSectionAnnouncement = getChangedClaudeAnnouncement(previousClaudeBodyRef.current, claudeProgress.body);
    const activeStep = claudeProgress.steps.find((step) => step.status === 'in_progress' || step.status === 'failed');
    const announcement = changedSectionAnnouncement
      || (activeStep ? buildProgressAnnouncement(activeStep.status, activeStep.label) : summarizeClaudeAnnouncement(claudeProgress));
    const announcementKey = `${claudeProgress.comment_id || 'none'}:${claudeProgress.updated_at || claudeProgress.status}:${announcement}`;
    if (lastAnnouncedProgressStepRef.current === announcementKey) return;
    lastAnnouncedProgressStepRef.current = announcementKey;
    previousClaudeBodyRef.current = claudeProgress.body || '';
    AccessibilityInfo.announceForAccessibility(announcement);
  }, [claudeProgress]);

  useEffect(() => {
    if (!isClaudeWaitingActive) {
      stopClaudeLoadingSound();
      return;
    }

    const loopToken = claudeLoadingLoopTokenRef.current + 1;
    claudeLoadingLoopTokenRef.current = loopToken;

    const playAndScheduleNext = async () => {
      await BeepService.playLoadingSound();
      if (claudeLoadingLoopTokenRef.current !== loopToken) return;
      claudeLoadingTimeoutRef.current = setTimeout(() => {
        claudeLoadingTimeoutRef.current = null;
        if (claudeLoadingLoopTokenRef.current !== loopToken) return;
        void playAndScheduleNext();
      }, CLAUDE_LOADING_AUDIO_INTERVAL_MS);
    };

    void playAndScheduleNext();
    return () => {
      if (claudeLoadingLoopTokenRef.current === loopToken) {
        stopClaudeLoadingSound();
      }
    };
  }, [isClaudeWaitingActive]);

  useEffect(() => () => {
    stopClaudeLoadingSound();
  }, []);

  useEffect(() => {
    let cancelled = false;

    const clearPoll = () => {
      if (progressPollIntervalRef.current) {
        clearInterval(progressPollIntervalRef.current);
        progressPollIntervalRef.current = null;
      }
    };

    const poll = async () => {
      if (!progressTarget || cancelled) return;
      latestClaudePollRequestIdRef.current += 1;
      const requestId = latestClaudePollRequestIdRef.current;
      const result = await fetchClaudeProgress(progressTarget);
      if (cancelled || requestId < latestClaudeAppliedRequestIdRef.current) return;
      if (!result) return;
      const resultUpdatedAt = result.updated_at ? new Date(result.updated_at).getTime() : null;
      if (
        resultUpdatedAt !== null
        && latestClaudeAppliedUpdatedAtRef.current !== null
        && resultUpdatedAt < latestClaudeAppliedUpdatedAtRef.current
      ) {
        return;
      }
      latestClaudeAppliedRequestIdRef.current = requestId;
      if (resultUpdatedAt !== null) {
        latestClaudeAppliedUpdatedAtRef.current = resultUpdatedAt;
      }
      setClaudeProgress((prev) => {
        if (
          prev
          && prev.updated_at
          && result.updated_at
          && new Date(result.updated_at).getTime() < new Date(prev.updated_at).getTime()
        ) {
          return prev;
        }
        if (result.status === 'unavailable' && prev && prev.comment_id) {
          return {
            ...prev,
            message: result.message || prev.message,
            error: result.error || prev.error,
          };
        }
        return result;
      });
    };

    clearPoll();
    setClaudeProgress(null);
    lastAnnouncedProgressStepRef.current = null;
    latestClaudePollRequestIdRef.current = 0;
    latestClaudeAppliedRequestIdRef.current = 0;
    latestClaudeAppliedUpdatedAtRef.current = null;

    if (progressTarget) {
      poll();
      progressPollIntervalRef.current = setInterval(() => {
        poll();
      }, CLAUDE_POLL_INTERVAL_MS);
    }

    return () => {
      cancelled = true;
      clearPoll();
      console.log('[Claude Progress] polling=stopped reason=page_exit');
    };
  }, [progressTarget]);

  useEffect(() => {
    if (!claudeProgress || !progressTarget) return;
    upsertClaudeProgressItem(claudeProgress);
  }, [claudeProgress, progressTarget]);

  // Listen for WS 'progress' broadcasts fired by the backend while an HTTP
  // request from this screen is in flight (e.g. "Summarizing video…"). This
  // is the only channel carrying them; _broadcast_ws has no per-request id,
  // so inFlightRef scopes display to "a request from this screen is pending."
  useEffect(() => {
    const handleWsMessage = (event: any) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'progress' && msg.message && inFlightRef.current) {
          setProgressText((prev) => (prev === msg.message ? prev : msg.message));
        }
      } catch {
        // Not JSON, or not a message this screen cares about.
      }
    };
    WebSocketService.addMessageListener(handleWsMessage);
    return () => WebSocketService.removeMessageListener(handleWsMessage);
  }, []);

  const nextId = (kind: string) => {
    idCounterRef.current += 1;
    return `${Date.now()}_${kind}_${idCounterRef.current}`;
  };

  const append = (item: IssueChatItem) => setItems((prev) => [...prev, item]);

  const upsertClaudeProgressItem = (progress: ClaudeProgressResponse) => {
    const body = formatClaudeBody(progress.body);
    setItems((prev) => {
      const existingId = claudeProgressItemIdRef.current
        || prev.find((item) => item.kind === 'assistant-claude-progress')?.id
        || null;
      if (!existingId) {
        hasClaudeMessageRef.current = true;
        allowNextClaudeAutoFollowRef.current = true;
        const id = nextId('assistant-claude-progress');
        claudeProgressItemIdRef.current = id;
        return [...prev, {
          kind: 'assistant-claude-progress',
          id,
          ts: new Date(),
          status: progress.status,
          body,
          commentId: progress.comment_id,
          updatedAt: progress.updated_at,
          message: progress.message,
        }];
      }
      return prev.map((item) => {
        if (item.kind !== 'assistant-claude-progress' || item.id !== existingId) return item;
        claudeProgressItemIdRef.current = existingId;
        return {
          ...item,
          ts: new Date(),
          status: progress.status,
          body,
          commentId: progress.comment_id,
          updatedAt: progress.updated_at,
          message: progress.message,
        };
      });
    });
  };

  const resolveLatestChoicePrompt = () => {
    setItems((prev) => {
      const idx = [...prev].reverse().findIndex((i) => i.kind === 'assistant-choice-prompt');
      if (idx === -1) return prev;
      const realIdx = prev.length - 1 - idx;
      const target = prev[realIdx];
      if (target.kind !== 'assistant-choice-prompt') return prev;
      const updated = [...prev];
      updated[realIdx] = { ...target, resolved: true };
      return updated;
    });
  };

  const resetConversation = () => {
    setActiveToken(null);
    setAwaiting(null);
    setUnderstandingSummary(null);
    setLastIntegrated(null);
    setIsSummaryModalOpen(false);
    brainstormHistoryRef.current = [];
  };

  const handlePickFromLibrary = async () => {
    setIsPickingFromLibrary(true);
    try {
      const result = await launchImageLibrary({
        mediaType: 'video',
        videoQuality: 'high',
      });
      if (result.didCancel || !result.assets?.length) return;
      const asset = result.assets[0];
      if (asset.uri) {
        setStagedVideoUri(asset.uri);
        setStagedVideoSource('library');
      }
    } finally {
      setIsPickingFromLibrary(false);
    }
  };

  const handleAttachRecentSession = async () => {
    try {
      const path = await ScreenRecordingModule?.fetchMostRecentVideoPath?.();
      if (path) {
        setStagedVideoUri(path);
        setStagedVideoSource('library');
      }
    } catch (error: any) {
      console.error('[IssueChat] Failed to attach recent session:', error);
      const code = error?.code;
      const message =
        code === 'no_recent_video'
          ? 'No recent recording was found.'
          : code === 'photos_read_permission_denied'
          ? 'Photos access was denied. Could not attach a recent recording.'
          : 'Could not attach a recent recording.';
      AccessibilityInfo.announceForAccessibility(message);
    }
  };

  const handleAttachVideoMenu = () => {
    const actions: any[] = [
      { text: 'Record a new video', onPress: () => setIsVideoRecorderOpen(true) },
    ];
    if (hasRayBan) {
      actions.push({ text: 'Record with Ray-Ban glasses', onPress: () => setIsRayBanRecorderOpen(true) });
    }
    actions.push(
      { text: 'Choose from photo library', onPress: handlePickFromLibrary },
      { text: 'Attach most recent recording', onPress: handleAttachRecentSession },
      { text: 'Cancel', style: 'cancel' },
    );
    Alert.alert('Attach a video', 'Choose how to attach a video to this message.', actions);
  };

  // --- Turn dispatchers ---

  const sendNewIssueText = async (text: string) => {
    append({ kind: 'user-text', id: nextId('user-text'), ts: new Date(), text });
    setComposeText('');
    Keyboard.dismiss();
    setIsSending(true);
    inFlightRef.current = true;
    try {
      const result = await submitCreation({ text, brainstormingEnabled: brainstormingActive });
      handleCreationResponse(result, { op: 'create-text', text });
    } finally {
      inFlightRef.current = false;
      setProgressText(null);
      setIsSending(false);
    }
  };

  const sendNewIssueVideo = async (text: string, videoUri: string) => {
    append({ kind: 'user-video', id: nextId('user-video'), ts: new Date(), videoUri, caption: text });
    setComposeText('');
    setStagedVideoUri(null);
    setStagedVideoSource(null);
    Keyboard.dismiss();
    setIsSending(true);
    inFlightRef.current = true;
    try {
      const result = await submitCreation({ text, videoUri, brainstormingEnabled: brainstormingActive });
      handleCreationResponse(result, { op: 'create-video', text, videoUri });
    } finally {
      inFlightRef.current = false;
      setProgressText(null);
      setIsSending(false);
    }
  };

  const sendIdeationAnswer = async (answer: string, token: string) => {
    append({ kind: 'user-text', id: nextId('user-text'), ts: new Date(), text: answer });
    setComposeText('');
    Keyboard.dismiss();
    setIsSending(true);
    inFlightRef.current = true;
    try {
      const result = await submitCreation({ text: answer, ideationAnswer: answer, token });
      if (result.status === 'error' && result.error === 'Ideation session expired or not found') {
        resetConversation();
        append({
          kind: 'assistant-error',
          id: nextId('assistant-error'),
          ts: new Date(),
          text: 'That brainstorming session expired. Send your description again to start over.',
        });
        return;
      }
      handleCreationResponse(result, { op: 'ideation-answer', text: answer, token });
    } finally {
      inFlightRef.current = false;
      setProgressText(null);
      setIsSending(false);
    }
  };

  const sendUpdateIdeationAnswer = async (answer: string, token: string) => {
    if (!selectedIssue) return;
    append({ kind: 'user-text', id: nextId('user-text'), ts: new Date(), text: answer });
    setComposeText('');
    Keyboard.dismiss();
    setIsSending(true);
    inFlightRef.current = true;
    try {
      const result = await submitUpdate({
        text: answer,
        issueNumber: selectedIssue.number,
        ideationAnswer: answer,
        token,
      });
      if (result.status === 'error' && result.error === 'Ideation session expired or not found') {
        resetConversation();
        append({
          kind: 'assistant-error',
          id: nextId('assistant-error'),
          ts: new Date(),
          text: 'That brainstorming session expired. Send your update again to start over.',
        });
        return;
      }
      handleUpdateResponse(result, { op: 'update-answer', text: answer, token, issueNumber: selectedIssue.number });
    } finally {
      inFlightRef.current = false;
      setProgressText(null);
      setIsSending(false);
    }
  };

  const handleCreationResponse = (
    result: Awaited<ReturnType<typeof submitCreation>>,
    retry: RetryDescriptor,
  ) => {
    if (result.status === 'created') {
      const videoSummarySkipped = !result.video_summary;
      append({
        kind: 'assistant-created',
        id: nextId('assistant-created'),
        ts: new Date(),
        issueNumber: result.issue_number,
        issueUrl: result.issue_url,
        videoSummarySkipped,
      });
      setProgressTarget({ mode: 'create', issueNumber: result.issue_number, prNumber: result.pr_number, commentId: result.comment_id });
      resetConversation();
      return;
    }

    if (result.status === 'ideation' && brainstormingActive) {
      setActiveToken(result.token);
      setAwaiting('answer');
      if (result.summary) setUnderstandingSummary(result.summary);
      if (result.integration_note) setLastIntegrated(result.integration_note);
      append({
        kind: 'assistant-question',
        id: nextId('assistant-question'),
        ts: new Date(),
        question: result.question,
        token: result.token,
      });
      return;
    }

    if (result.status === 'brainstorm_choice' && brainstormingActive) {
      setActiveToken(result.token);
      setAwaiting('choice');
      if (result.summary) setUnderstandingSummary(result.summary);
      if (result.integration_note) setLastIntegrated(result.integration_note);
      brainstormHistoryRef.current = result.brainstorm_history || [];
      append({
        kind: 'assistant-choice-prompt',
        id: nextId('assistant-choice-prompt'),
        ts: new Date(),
        text: 'Thanks for that context! You can keep brainstorming or start building.',
        token: result.token,
        resolved: false,
      });
      return;
    }

    // Either an explicit error, or a question/choice arriving while
    // brainstorming is off — treat both as an error since the compose bar
    // won't offer a way to answer it.
    const message = result.status === 'error' ? result.error : 'Something went wrong. Please try again.';
    append({ kind: 'assistant-error', id: nextId('assistant-error'), ts: new Date(), text: message, retry });
  };

  const handleUpdateResponse = (
    result: Awaited<ReturnType<typeof submitUpdate>>,
    retry: RetryDescriptor,
  ) => {
    if (result.status === 'updated') {
      const videoSummarySkipped = !result.video_summary;
      append({
        kind: 'assistant-updated',
        id: nextId('assistant-updated'),
        ts: new Date(),
        issueNumber: result.issue_number,
        issueUrl: result.issue_url,
        videoSummarySkipped,
      });
      setProgressTarget({
        mode: 'update',
        prNumber: result.pr_number ?? selectedIssue?.number,
        commentId: null,
        afterCommentId: result.comment_id,
        afterTimestamp: result.comment_created_at ?? null,
      });
      resetConversation();
      return;
    }

    if (result.status === 'ideation' && brainstormingActive) {
      setActiveToken(result.token);
      setAwaiting('answer');
      if (result.summary) setUnderstandingSummary(result.summary);
      if (result.integration_note) setLastIntegrated(result.integration_note);
      append({
        kind: 'assistant-question',
        id: nextId('assistant-question'),
        ts: new Date(),
        question: result.question,
        token: result.token,
      });
      return;
    }

    if (result.status === 'brainstorm_choice' && brainstormingActive) {
      setActiveToken(result.token);
      setAwaiting('choice');
      if (result.summary) setUnderstandingSummary(result.summary);
      if (result.integration_note) setLastIntegrated(result.integration_note);
      brainstormHistoryRef.current = result.brainstorm_history || [];
      append({
        kind: 'assistant-choice-prompt',
        id: nextId('assistant-choice-prompt'),
        ts: new Date(),
        text: 'Thanks for that context! You can keep brainstorming or start building.',
        token: result.token,
        resolved: false,
      });
      return;
    }

    const message = result.status === 'error' ? result.error : 'Something went wrong. Please try again.';
    append({ kind: 'assistant-error', id: nextId('assistant-error'), ts: new Date(), text: message, retry });
  };

  const handleKeepBrainstorming = async () => {
    if (!activeToken) return;
    resolveLatestChoicePrompt();
    append({
      kind: 'user-choice',
      id: nextId('user-choice'),
      ts: new Date(),
      choice: 'keep_brainstorming',
      label: '🧠 Keep Brainstorming',
    });
    setIsSending(true);
    inFlightRef.current = true;
    try {
      const result = await nextBrainstormQuestion(activeToken);
      if (result.status === 'ideation') {
        brainstormHistoryRef.current = result.brainstorm_history || [];
        setAwaiting('answer');
        if (result.summary) setUnderstandingSummary(result.summary);
        if (result.integration_note) setLastIntegrated(result.integration_note);
        append({
          kind: 'assistant-question',
          id: nextId('assistant-question'),
          ts: new Date(),
          question: result.question,
          token: result.token,
        });
      } else {
        append({
          kind: 'assistant-error',
          id: nextId('assistant-error'),
          ts: new Date(),
          text: result.error || 'Failed to get next question',
          retry: { op: 'next-question', token: activeToken },
        });
      }
    } finally {
      inFlightRef.current = false;
      setProgressText(null);
      setIsSending(false);
    }
  };

  const handleStartBuilding = async () => {
    if (!activeToken) return;
    if (!isCreateMode && !selectedIssue) return;
    resolveLatestChoicePrompt();
    append({
      kind: 'user-choice',
      id: nextId('user-choice'),
      ts: new Date(),
      choice: 'start_building',
      label: '🚀 Start Building',
    });

    const history = brainstormHistoryRef.current;
    const lastAnswer = history.length > 0 ? history[history.length - 1].answer : 'Ready to build';

    setIsSending(true);
    inFlightRef.current = true;
    try {
      if (isCreateMode) {
        const result = await submitCreation({
          text: lastAnswer,
          ideationAnswer: lastAnswer,
          token: activeToken,
          choice: 'start_building',
        });
        if (result.status === 'created') {
          const videoSummarySkipped = !result.video_summary;
          append({
            kind: 'assistant-created',
            id: nextId('assistant-created'),
            ts: new Date(),
            issueNumber: result.issue_number,
            issueUrl: result.issue_url,
            videoSummarySkipped,
          });
          setProgressTarget({ mode: 'create', issueNumber: result.issue_number, prNumber: result.pr_number, commentId: result.comment_id });
          resetConversation();
        } else {
          const message = result.status === 'error' ? result.error : 'Failed to create issue';
          append({
            kind: 'assistant-error',
            id: nextId('assistant-error'),
            ts: new Date(),
            text: message,
            retry: { op: 'start-building', token: activeToken, mode: 'create' },
          });
        }
      } else {
        const result = await submitUpdate({
          text: lastAnswer,
          issueNumber: selectedIssue.number,
          ideationAnswer: lastAnswer,
          token: activeToken,
          choice: 'start_building',
        });
        if (result.status === 'updated') {
          const videoSummarySkipped = !result.video_summary;
          append({
            kind: 'assistant-updated',
            id: nextId('assistant-updated'),
            ts: new Date(),
            issueNumber: result.issue_number,
            issueUrl: result.issue_url,
            videoSummarySkipped,
          });
          setProgressTarget({
            mode: 'update',
            prNumber: result.pr_number ?? selectedIssue.number,
            commentId: null,
            afterCommentId: result.comment_id,
            afterTimestamp: result.comment_created_at ?? null,
          });
          resetConversation();
        } else {
          const message = result.status === 'error' ? result.error : 'Failed to send update';
          append({
            kind: 'assistant-error',
            id: nextId('assistant-error'),
            ts: new Date(),
            text: message,
            retry: { op: 'start-building', token: activeToken, mode: 'update', issueNumber: selectedIssue.number },
          });
        }
      }
    } finally {
      inFlightRef.current = false;
      setProgressText(null);
      setIsSending(false);
    }
  };

  const handleTalkToAgent = () => {
    if (!activeToken) return;
    resolveLatestChoicePrompt();
    append({
      kind: 'user-choice',
      id: nextId('user-choice'),
      ts: new Date(),
      choice: 'ask_agent',
      label: '💬 Talk to Agent',
    });
    setAwaiting('clarification');
  };

  const sendAgentQuestion = async (question: string, token: string) => {
    const trimmed = question.trim();
    if (!trimmed) return;
    append({ kind: 'user-text', id: nextId('user-text'), ts: new Date(), text: trimmed });
    setComposeText('');
    Keyboard.dismiss();
    setAwaiting(null);
    setIsSending(true);
    inFlightRef.current = true;
    try {
      const result = await askBrainstormAgent(token, trimmed);
      if (result.status === 'clarification') {
        brainstormHistoryRef.current = result.brainstorm_history || [];
        if (result.summary) setUnderstandingSummary(result.summary);
        if (result.integration_note) setLastIntegrated(result.integration_note);
        append({
          kind: 'assistant-clarification-answer',
          id: nextId('assistant-clarification-answer'),
          ts: new Date(),
          question: trimmed,
          answer: result.answer,
          token,
        });
        setActiveToken(token);
        append({
          kind: 'assistant-choice-prompt',
          id: nextId('assistant-choice-prompt'),
          ts: new Date(),
          text: 'Anything else? You can keep brainstorming, start building, or ask another question.',
          token,
          resolved: false,
        });
      } else {
        append({
          kind: 'assistant-error',
          id: nextId('assistant-error'),
          ts: new Date(),
          text: result.error || 'Failed to get an answer',
          retry: { op: 'ask-agent', token, question: trimmed },
        });
      }
    } finally {
      inFlightRef.current = false;
      setProgressText(null);
      setIsSending(false);
    }
  };

  const handleTalkToAgent = () => {
    if (!activeToken) return;
    resolveLatestChoicePrompt();
    append({
      kind: 'user-choice',
      id: nextId('user-choice'),
      ts: new Date(),
      choice: 'ask_agent',
      label: '💬 Talk to Agent',
    });
    setAwaiting('clarification');
  };

  const sendAgentQuestion = async (question: string, token: string) => {
    const trimmed = question.trim();
    if (!trimmed) return;
    append({ kind: 'user-text', id: nextId('user-text'), ts: new Date(), text: trimmed });
    setComposeText('');
    Keyboard.dismiss();
    setAwaiting(null);
    setIsSending(true);
    inFlightRef.current = true;
    try {
      const result = await submitUpdate({ text, issueNumber: selectedIssue.number, videoUri, brainstormingEnabled: brainstormingActive });
      if (result.status === 'updated') {
        const videoSummarySkipped = !!videoUri && !result.video_summary;
        append({
          kind: 'assistant-updated',
          id: nextId('assistant-updated'),
          ts: new Date(),
          issueNumber: result.issue_number,
          issueUrl: result.issue_url,
          videoSummarySkipped,
        });
      } else {
        append({
          kind: 'assistant-error',
          id: nextId('assistant-error'),
          ts: new Date(),
          text: result.error,
          retry: { op: 'update', text, videoUri, issueNumber: selectedIssue.number },
        });
      }
      handleUpdateResponse(result, { op: 'update', text, videoUri, issueNumber: selectedIssue.number });
    } finally {
      inFlightRef.current = false;
      setProgressText(null);
      setIsSending(false);
    }
  };

  const sendUpdate = async (text: string, videoUri: string | null) => {
    if (!selectedIssue) return;
    if (videoUri) {
      append({ kind: 'user-video', id: nextId('user-video'), ts: new Date(), videoUri, caption: text });
    } else {
      append({ kind: 'user-text', id: nextId('user-text'), ts: new Date(), text });
    }
    setComposeText('');
    setStagedVideoUri(null);
    Keyboard.dismiss();
    setIsSending(true);
    inFlightRef.current = true;
    try {
      const result = await submitUpdate({ text, issueNumber: selectedIssue.number, videoUri, brainstormingEnabled: brainstormingActive });
      handleUpdateResponse(result, { op: 'update', text, videoUri, issueNumber: selectedIssue.number });
    } finally {
      inFlightRef.current = false;
      setProgressText(null);
      setIsSending(false);
    }
  };

  const handleRetry = (retry: RetryDescriptor) => {
    switch (retry.op) {
      case 'create-text':
        sendNewIssueText(retry.text);
        return;
      case 'create-video':
        sendNewIssueVideo(retry.text, retry.videoUri);
        return;
      case 'update':
        sendUpdate(retry.text, retry.videoUri);
        return;
      case 'update-answer':
        sendUpdateIdeationAnswer(retry.text, retry.token);
        return;
      case 'ideation-answer':
        sendIdeationAnswer(retry.text, retry.token);
        return;
      case 'next-question':
        handleKeepBrainstorming();
        return;
      case 'start-building':
        handleStartBuilding();
        return;
      case 'ask-agent':
        sendAgentQuestion(retry.question, retry.token);
        return;
    }
  };

  const handleSend = () => {
    const text = composeText.trim();
    if (!text && !stagedVideoUri) return;

    if (awaiting === 'clarification' && activeToken) {
      sendAgentQuestion(text, activeToken);
      return;
    }

    if (awaiting === 'answer' && activeToken) {
      if (isCreateMode) {
        sendIdeationAnswer(text, activeToken);
      } else {
        sendUpdateIdeationAnswer(text, activeToken);
      }
      return;
    }

    if (isCreateMode) {
      if (stagedVideoUri) {
        sendNewIssueVideo(text, stagedVideoUri);
      } else {
        sendNewIssueText(text);
      }
      return;
    }

    sendUpdate(text, stagedVideoUri);
  };

  const handleNewConversation = () => {
    setItems([]);
    setComposeText('');
    setStagedVideoUri(null);
    setStagedVideoSource(null);
    setClaudeProgress(null);
    setProgressTarget(null);
    claudeProgressItemIdRef.current = null;
    previousClaudeBodyRef.current = '';
    latestClaudePollRequestIdRef.current = 0;
    latestClaudeAppliedRequestIdRef.current = 0;
    latestClaudeAppliedUpdatedAtRef.current = null;
    hasClaudeMessageRef.current = false;
    allowNextClaudeAutoFollowRef.current = false;
    lastAutoScrollStateRef.current = { itemCount: 0, lastItemId: null };
    resetConversation();
    Keyboard.dismiss();
  };

  useEffect(() => {
    hasClaudeMessageRef.current = false;
    allowNextClaudeAutoFollowRef.current = false;
    claudeProgressItemIdRef.current = null;
    previousClaudeBodyRef.current = '';
    latestClaudePollRequestIdRef.current = 0;
    latestClaudeAppliedRequestIdRef.current = 0;
    latestClaudeAppliedUpdatedAtRef.current = null;
    lastAutoScrollStateRef.current = {
      itemCount: items.length,
      lastItemId: items[items.length - 1]?.id || null,
    };
  }, [isCreateMode, selectedIssue?.number]);

  // A staged video needs a caption — an empty/placeholder caption feeds
  // parse_transcript_with_ai garbage to build the issue title from.
  const sendDisabled =
    isSending ||
    awaiting === 'choice' ||
    (stagedVideoUri ? !composeText.trim() : !composeText.trim());

  const placeholder =
    awaiting === 'clarification'
      ? 'Ask the agent a question…'
      : awaiting === 'answer'
      ? 'Type your answer…'
      : stagedVideoUri
      ? 'Add a short description of the video…'
      : isCreateMode
      ? "Describe the visual assistive technology you'd like…"
      : 'Describe your update…';

  const sendAccessibilityLabel = isSending
    ? 'Submitting…'
    : stagedVideoUri
    ? 'Submit with video'
    : awaiting === 'answer'
    ? 'Send answer'
    : 'Send text';

  const renderItem = (item: IssueChatItem) => {
    switch (item.kind) {
      case 'user-text':
        return (
          <View
            key={item.id}
            style={[styles.messageContainer, styles.userAlign, { backgroundColor: theme.primary }]}
            accessible={true}
            accessibilityLabel={`You said: ${item.text}`}
            accessibilityRole="text"
            accessibilityHint="Long press to copy text">
            <Text style={[styles.messageText, styles.userMessageText]} selectable={true} accessible={false}>
              {item.text}
            </Text>
            <Text style={[styles.timestamp, styles.userTimestamp]} accessible={false}>
              {item.ts.toLocaleTimeString()}
            </Text>
          </View>
        );
      case 'user-video':
        return (
          <View
            key={item.id}
            style={[styles.messageContainer, styles.userAlign, { backgroundColor: theme.primary }]}
            accessible={true}
            accessibilityLabel={`You attached a video with the description: ${item.caption}`}
            accessibilityRole="text">
            <Text style={[styles.messageText, styles.userMessageText]} accessible={false}>
              📹 Video attached
            </Text>
            {item.caption !== '' && (
              <Text style={[styles.messageText, styles.userMessageText, styles.videoCaption]} accessible={false}>
                {item.caption}
              </Text>
            )}
            <Text style={[styles.timestamp, styles.userTimestamp]} accessible={false}>
              {item.ts.toLocaleTimeString()}
            </Text>
          </View>
        );
      case 'user-choice':
        return (
          <View
            key={item.id}
            style={[styles.messageContainer, styles.userAlign, { backgroundColor: theme.primary }]}
            accessible={true}
            accessibilityLabel={`You said: ${item.label}`}
            accessibilityRole="text">
            <Text style={[styles.messageText, styles.userMessageText]} accessible={false}>
              {item.label}
            </Text>
            <Text style={[styles.timestamp, styles.userTimestamp]} accessible={false}>
              {item.ts.toLocaleTimeString()}
            </Text>
          </View>
        );
      case 'assistant-question':
        return (
          <View
            key={item.id}
            style={[styles.messageContainer, styles.assistantAlign, { backgroundColor: theme.card, borderColor: theme.border }]}>
            <Text
              style={[styles.messageText, { color: theme.text }]}
              selectable={true}
              accessible={true}
              accessibilityRole="text"
              accessibilityLabel={`Assistant said: ${item.question}`}
              accessibilityHint="Long press to copy text">
              {item.question}
            </Text>
            <Text style={[styles.timestamp, { color: theme.textTertiary }]} accessible={false}>
              {item.ts.toLocaleTimeString()}
            </Text>
          </View>
        );
      case 'assistant-clarification-answer':
        return (
          <View
            key={item.id}
            style={[styles.messageContainer, styles.assistantAlign, { backgroundColor: theme.card, borderColor: theme.border }]}>
            <Text
              style={[styles.messageText, { color: theme.text }]}
              selectable={true}
              accessible={true}
              accessibilityRole="text"
              accessibilityLabel={`Assistant answered: ${item.answer}`}
              accessibilityHint="Long press to copy text">
              {item.answer}
            </Text>
            <Text style={[styles.timestamp, { color: theme.textTertiary }]} accessible={false}>
              {item.ts.toLocaleTimeString()}
            </Text>
          </View>
        );
      case 'assistant-choice-prompt':
        return (
          <View
            key={item.id}
            style={[styles.messageContainer, styles.assistantAlign, { backgroundColor: theme.card, borderColor: theme.border }]}>
            <Text
              style={[styles.messageText, { color: theme.text }]}
              accessible={true}
              accessibilityRole="text"
              accessibilityLabel={`Assistant said: ${item.text}`}>
              {item.text}
            </Text>
            {!item.resolved && (
              <View style={styles.brainstormChoiceButtons}>
                <TouchableOpacity
                  style={[styles.brainstormButton, { backgroundColor: theme.primary }]}
                  onPress={handleKeepBrainstorming}
                  disabled={isSending}
                  accessible={true}
                  accessibilityLabel="Keep brainstorming"
                  accessibilityHint="Continue asking more questions about your tool design"
                  accessibilityRole="button"
                  accessibilityState={{ disabled: isSending }}>
                  {isSending ? (
                    <ActivityIndicator size="small" color="#fff" />
                  ) : (
                    <Text style={styles.brainstormButtonText}>🧠 Keep Brainstorming</Text>
                  )}
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.brainstormButton, { backgroundColor: theme.success }]}
                  onPress={handleStartBuilding}
                  disabled={isSending}
                  accessible={true}
                  accessibilityLabel="Start building"
                  accessibilityHint="Create the tool with all the brainstorming information"
                  accessibilityRole="button"
                  accessibilityState={{ disabled: isSending }}>
                  {isSending ? (
                    <ActivityIndicator size="small" color="#fff" />
                  ) : (
                    <Text style={styles.brainstormButtonText}>🚀 Start Building</Text>
                  )}
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.brainstormButton, { backgroundColor: theme.info }]}
                  onPress={handleTalkToAgent}
                  disabled={isSending}
                  accessible={true}
                  accessibilityLabel="Talk to agent"
                  accessibilityHint="Ask a free-form question about your tool and get an answer, then return to this menu"
                  accessibilityRole="button"
                  accessibilityState={{ disabled: isSending }}>
                  {isSending ? (
                    <ActivityIndicator size="small" color="#fff" />
                  ) : (
                    <Text style={styles.brainstormButtonText}>💬 Talk to Agent</Text>
                  )}
                </TouchableOpacity>
              </View>
            )}
            <Text style={[styles.timestamp, { color: theme.textTertiary }]} accessible={false}>
              {item.ts.toLocaleTimeString()}
            </Text>
          </View>
        );
      case 'assistant-created':
      case 'assistant-updated': {
        const verb = item.kind === 'assistant-created' ? 'created' : 'updated';
        const note = item.videoSummarySkipped ? ' Video summarization was skipped.' : '';
        return (
          <View
            key={item.id}
            style={[styles.messageContainer, styles.assistantAlign, { backgroundColor: theme.card, borderColor: theme.border }]}
            accessible={true}
            accessibilityLabel={`Issue ${item.issueNumber} ${verb}.${note}`}
            accessibilityRole="text">
            <Text style={[styles.messageText, { color: theme.text }]} selectable={true} accessible={false}>
              {`Issue #${item.issueNumber} ${verb}.${note}`}
            </Text>
            <Text style={[styles.timestamp, { color: theme.textTertiary }]} accessible={false}>
              {item.ts.toLocaleTimeString()}
            </Text>
          </View>
        );
      }
      case 'assistant-claude-progress': {
        const accessibilityLabel = buildClaudeAccessibilityLabel(item.body, item.message);
        const accessibilitySections = buildClaudeAccessibilitySections(item.body);
        return (
          <View
            key={item.id}
            testID={`claude-progress-${item.id}`}
            style={[styles.messageContainer, styles.assistantAlign, { backgroundColor: theme.card, borderColor: theme.border }]}
            accessible={false}>
            <Text
              style={[styles.progressMessageLabel, { color: theme.primary }]}
              accessible={true}
              accessibilityRole="header"
              accessibilityLabel="Claude">
              Claude
            </Text>
            {accessibilitySections.map((section, sectionIndex) => (
              <View
                key={`${item.id}_section_${sectionIndex}`}
                accessible={true}
                accessibilityRole="text"
                accessibilityLabel={section.label}>
                {section.lines.map((line, lineIndex) => {
                  if (line.kind === 'blank') {
                    return <View key={`${item.id}_blank_${sectionIndex}_${lineIndex}`} style={styles.claudeLineSpacer} />;
                  }
                  return (
                    <Text
                      key={`${item.id}_${sectionIndex}_${lineIndex}`}
                      style={[
                        line.kind === 'heading'
                          ? [styles.claudeHeadingText, { color: theme.text }]
                          : [styles.messageText, { color: theme.text }],
                      ]}
                      selectable={true}
                      accessible={false}
                      accessibilityRole="text"
                      accessibilityLabel={accessibilityLabel}>
                      {line.text}
                    </Text>
                  );
                })}
              </View>
            ))}
            <Text style={[styles.timestamp, { color: theme.textTertiary }]} accessible={false}>
              {item.ts.toLocaleTimeString()}
            </Text>
          </View>
        );
      }
      case 'assistant-error':
        return (
          <View
            key={item.id}
            style={[styles.messageContainer, styles.assistantAlign, styles.errorMessage, { borderColor: theme.error }]}
            accessible={true}
            accessibilityRole="alert"
            accessibilityLiveRegion="assertive"
            accessibilityLabel={item.text}>
            <Text style={[styles.messageText, { color: theme.error }]} accessible={false}>
              {item.text}
            </Text>
            {item.retry && (
              <TouchableOpacity
                style={[styles.retryButton, { borderColor: theme.error }]}
                onPress={() => handleRetry(item.retry!)}
                accessible={true}
                accessibilityRole="button"
                accessibilityLabel="Try again"
                accessibilityHint="Resends the same message">
                <Text style={[styles.retryButtonText, { color: theme.error }]}>Try again</Text>
              </TouchableOpacity>
            )}
            <Text style={[styles.timestamp, { color: theme.textTertiary }]} accessible={false}>
              {item.ts.toLocaleTimeString()}
            </Text>
          </View>
        );
    }
  };

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.background }]} edges={['bottom']}>
      <KeyboardAvoidingView
        style={styles.container}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 64 : 0}>
      <View style={[styles.header, { borderBottomColor: theme.border }]}>
        {modeBannerText && (
          <View
            style={[
              styles.modeBanner,
              {
                backgroundColor: (isCreateMode ? theme.success : theme.primary) + '18',
              },
            ]}
            accessible={false}>
            <Text
              style={[styles.modeBannerText, { color: theme.text }]}
              accessible={true}
              accessibilityRole="text"
              accessibilityLabel={modeBannerText}
              numberOfLines={1}
              ellipsizeMode="tail">
              {modeBannerText}
            </Text>
          </View>
        )}
        <View style={styles.headerTopRow}>
          {showBackButton && onBack && (
            <TouchableOpacity
              style={[styles.backButton, { backgroundColor: theme.backgroundSecondary }]}
              onPress={onBack}
              accessible={true}
              accessibilityRole="button"
              accessibilityLabel="Back to PR list"
              accessibilityHint="Double tap to return to pull request list">
              <Text style={[styles.backButtonText, { color: theme.primary }]}>← Back to PRs</Text>
            </TouchableOpacity>
          )}
          <View style={styles.headerTopRowSpacer} />
          <TouchableOpacity
            style={[styles.newConversationButton, { borderColor: theme.border }]}
            onPress={handleNewConversation}
            accessible={true}
            accessibilityRole="button"
            accessibilityLabel="Start a new conversation"
            accessibilityHint="Clears the conversation and starts over">
            <Text style={[styles.newConversationButtonText, { color: theme.textSecondary }]}>New</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.headerTitleRow}>
          <View
            style={[
              styles.modeBadge,
              { backgroundColor: isCreateMode ? theme.success : theme.primary },
            ]}
            accessible={true}
            accessibilityRole="text"
            accessibilityLabel={isCreateMode ? 'Create new mode' : `Update mode. ${selectedIssue?.title}`}>
            <Text style={styles.modeBadgeText} accessible={false} numberOfLines={1} ellipsizeMode="tail">
              {isCreateMode ? '✨ Create New' : `🔄 ${selectedIssue?.title}`}
            </Text>
          </View>
          <Text
            style={[styles.headerText, { color: theme.text }]}
            accessible={true}
            accessibilityRole="header"
            accessibilityLabel={isCreateMode ? 'New Visual AT Tool' : `Update ${selectedIssue?.title}`}
            numberOfLines={2}
            ellipsizeMode="tail">
            {isCreateMode ? 'New Visual AT Tool' : `Update ${selectedIssue?.title}`}
          </Text>
        </View>
      </View>

      <ScrollView
        ref={scrollViewRef}
        style={styles.messagesContainer}
        contentContainerStyle={styles.messagesContent}
        keyboardShouldPersistTaps="handled"
        accessibilityLiveRegion="polite">
        {items.map(renderItem)}
        {isSending && (
          <View
            style={[styles.messageContainer, styles.assistantAlign, styles.thinkingMessage, { backgroundColor: theme.card, borderColor: theme.border }]}
            accessible={true}
            accessibilityLiveRegion="polite"
            accessibilityLabel={progressText ?? 'Thinking…'}>
            <Text style={[styles.thinkingText, { color: theme.textSecondary }]} accessible={false}>
              {progressText ?? 'Thinking…'}
            </Text>
          </View>
        )}
      </ScrollView>

      {understandingSummary && (
        <TouchableOpacity
          style={[styles.summaryCard, { backgroundColor: theme.card, borderColor: theme.border }]}
          onPress={() => setIsSummaryModalOpen(true)}
          activeOpacity={0.7}
          accessible={true}
          accessibilityRole="button"
          accessibilityLabel={`What I understand. ${lastIntegrated ? `Just integrated: ${lastIntegrated}. ` : ''}${understandingSummary}`}
          accessibilityHint="Double tap to view the full text">
          <View style={styles.summaryCardHeader}>
            <Text style={[styles.summaryCardLabel, { color: theme.textSecondary }]} accessible={false}>
              What I understand
            </Text>
            <Text style={[styles.summaryCardExpandHint, { color: theme.primary }]} accessible={false}>
              View full ›
            </Text>
          </View>
          {lastIntegrated && (
            <Text
              style={[styles.summaryCardLatest, { color: theme.textSecondary }]}
              accessible={false}
              numberOfLines={1}
              ellipsizeMode="tail">
              Just integrated: {lastIntegrated}
            </Text>
          )}
          <Text
            style={[styles.summaryCardText, { color: theme.text }]}
            accessible={false}
            numberOfLines={3}
            ellipsizeMode="tail">
            {understandingSummary}
          </Text>
        </TouchableOpacity>
      )}

      <Modal
        visible={isSummaryModalOpen}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setIsSummaryModalOpen(false)}>
        <View style={styles.summaryModalOverlay}>
          <SafeAreaView style={[styles.summaryModalSheet, { backgroundColor: theme.card }]} edges={['bottom']}>
            <View style={[styles.summaryModalHeader, { borderBottomColor: theme.border }]}>
              <Text
                style={[styles.summaryModalTitle, { color: theme.text }]}
                accessible={true}
                accessibilityRole="header">
                What I understand
              </Text>
              <TouchableOpacity
                onPress={() => setIsSummaryModalOpen(false)}
                accessible={true}
                accessibilityRole="button"
                accessibilityLabel="Close"
                accessibilityHint="Closes this dialog and returns to the conversation">
                <Text style={[styles.summaryModalCloseText, { color: theme.primary }]}>Done</Text>
              </TouchableOpacity>
            </View>
            <ScrollView style={styles.summaryModalScroll} contentContainerStyle={styles.summaryModalScrollContent}>
              {lastIntegrated && (
                <Text
                  style={[styles.summaryModalLatest, { color: theme.textSecondary }]}
                  accessible={true}
                  accessibilityRole="text"
                  accessibilityLabel={`Just integrated: ${lastIntegrated}`}>
                  Just integrated: {lastIntegrated}
                </Text>
              )}
              <Text
                style={[styles.summaryModalText, { color: theme.text }]}
                selectable={true}
                accessible={true}
                accessibilityRole="text"
                accessibilityLabel={understandingSummary ?? ''}
                accessibilityHint="Long press to copy text">
                {understandingSummary}
              </Text>
            </ScrollView>
          </SafeAreaView>
        </View>
      </Modal>

      <View style={[styles.composeBar, { backgroundColor: theme.background, borderTopColor: theme.border }]}>
        {!basicMode && stagedVideoUri && (
          <View
            style={[styles.videoChip, { backgroundColor: theme.backgroundSecondary, borderColor: theme.border }]}
            accessible={true}
            accessibilityLiveRegion="polite"
            accessibilityLabel="Video attached, ready to send">
            <Text style={[styles.videoChipText, { color: theme.text }]}>📹 Video attached</Text>
            <TouchableOpacity
              onPress={() => {
                if (stagedVideoSource === 'rayban') {
                  setIsRayBanRecorderOpen(true);
                } else if (stagedVideoSource === 'library') {
                  handlePickFromLibrary();
                } else {
                  setIsVideoRecorderOpen(true);
                }
              }}
              accessible={true}
              accessibilityRole="button"
              accessibilityLabel={stagedVideoSource === 'library' ? 'Pick a different video from library' : 'Re-record video'}>
              <Text style={[styles.videoActionText, { color: theme.primary }]}>
                {stagedVideoSource === 'library' ? 'Replace' : 'Re-record'}
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              onPress={handlePickFromLibrary}
              accessible={true}
              accessibilityRole="button"
              accessibilityLabel="Pick different video from library">
              <Text style={[styles.videoActionText, { color: theme.primary }]}>Library</Text>
            </TouchableOpacity>
            <TouchableOpacity
              onPress={() => { setStagedVideoUri(null); setStagedVideoSource(null); }}
              accessible={true}
              accessibilityRole="button"
              accessibilityLabel="Remove video attachment">
              <Text style={[styles.videoActionText, { color: theme.error }]}>Remove</Text>
            </TouchableOpacity>
          </View>
        )}

        {!basicMode && !stagedVideoUri && isPickingFromLibrary && (
          <View style={styles.pickingRow}>
            <ActivityIndicator size="small" color={theme.primary} />
            <Text style={[styles.pickingText, { color: theme.textSecondary }]}>
              Attaching video from library…
            </Text>
          </View>
        )}

        <View style={styles.composeRow}>
          {!basicMode && !stagedVideoUri && !isPickingFromLibrary && awaiting !== 'choice' && (
            <View style={styles.attachButtons}>
              <TouchableOpacity
                style={[styles.attachButton, { backgroundColor: theme.backgroundSecondary, borderColor: theme.border }]}
                onPress={handleAttachVideoMenu}
                accessible={true}
                accessibilityRole="button"
                accessibilityLabel="Attach a video"
                accessibilityHint="Opens a menu to record a new video, choose one from your photo library, or attach your most recent recording">
                <Text style={styles.attachButtonText}>📹</Text>
              </TouchableOpacity>
            </View>
          )}

          <RNTextInput
            ref={composeInputRef}
            style={[styles.textInput, {
              backgroundColor: theme.inputBackground,
              borderColor: theme.inputBorder,
              color: theme.text,
            }]}
            placeholder={placeholder}
            placeholderTextColor={theme.inputPlaceholder}
            multiline
            value={composeText}
            onChangeText={setComposeText}
            autoCorrect={true}
            autoCapitalize="sentences"
            editable={awaiting !== 'choice' && !isSending}
            accessible={true}
            accessibilityLabel="Message input"
            accessibilityHint="Type your message here. Use context menu to copy or paste"
            contextMenuHidden={false}
            selectTextOnFocus={false}
          />

          {awaiting !== 'choice' && (
            <TouchableOpacity
              style={[
                styles.sendButton,
                { backgroundColor: theme.primary },
                sendDisabled && styles.sendButtonDisabled,
              ]}
              onPress={handleSend}
              disabled={sendDisabled}
              accessible={true}
              accessibilityLabel={sendAccessibilityLabel}
              accessibilityHint="Sends the message to the server"
              accessibilityRole="button"
              accessibilityState={{ disabled: sendDisabled }}>
              {isSending ? (
                <ActivityIndicator size="small" color="#fff" />
              ) : (
                <Text style={styles.sendButtonText}>Send</Text>
              )}
            </TouchableOpacity>
          )}
        </View>
      </View>

      {!basicMode && (
        <VideoRecorderModal
          visible={isVideoRecorderOpen}
          onVideoRecorded={(path) => {
            setStagedVideoUri(path);
            setStagedVideoSource('phone');
            setIsVideoRecorderOpen(false);
          }}
          onCancel={() => setIsVideoRecorderOpen(false)}
        />
      )}
      {!basicMode && hasRayBan && (
        <RayBanRecorderModal
          visible={isRayBanRecorderOpen}
          onVideoRecorded={(path) => {
            setStagedVideoUri(path);
            setStagedVideoSource('rayban');
            setIsRayBanRecorderOpen(false);
          }}
          onCancel={() => setIsRayBanRecorderOpen(false)}
        />
      )}
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
  },
  modeBanner: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    marginBottom: 10,
  },
  modeBannerText: {
    fontSize: 14,
    fontWeight: '600',
    textAlign: 'center',
  },
  headerTopRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  headerTopRowSpacer: {
    flex: 1,
  },
  backButton: {
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 8,
  },
  backButtonText: {
    fontSize: 14,
    fontWeight: '600',
  },
  headerTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: 8,
    marginTop: 10,
  },
  modeBadge: {
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 12,
    flexShrink: 0,
  },
  modeBadgeText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#fff',
  },
  headerText: {
    flexShrink: 1,
    fontSize: 16,
    fontWeight: 'bold',
  },
  newConversationButton: {
    paddingVertical: 6,
    paddingHorizontal: 10,
    borderRadius: 8,
    borderWidth: 1,
  },
  newConversationButtonText: {
    fontSize: 13,
    fontWeight: '600',
  },
  progressCard: {
    marginTop: 12,
    borderWidth: 1,
    borderRadius: 10,
    padding: 10,
    maxHeight: 180,
  },
  progressTitle: {
    fontSize: 14,
    fontWeight: '700',
  },
  progressMessage: {
    fontSize: 12,
    marginTop: 2,
    marginBottom: 8,
  },
  progressSteps: {
    maxHeight: 120,
  },
  progressStepRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 8,
    gap: 8,
  },
  progressMarker: {
    width: 16,
    fontSize: 14,
    fontWeight: '700',
  },
  progressStepLabel: {
    flex: 1,
    fontSize: 14,
    lineHeight: 18,
  },
  progressEmpty: {
    fontSize: 13,
    fontStyle: 'italic',
  },
  messagesContainer: {
    flex: 1,
  },
  messagesContent: {
    padding: 15,
  },
  messageContainer: {
    marginBottom: 15,
    padding: 10,
    borderRadius: 10,
    maxWidth: '85%',
  },
  userAlign: {
    alignSelf: 'flex-end',
  },
  assistantAlign: {
    alignSelf: 'flex-start',
    borderWidth: 1,
  },
  errorMessage: {
    borderWidth: 1,
  },
  thinkingMessage: {
    marginBottom: 15,
  },
  thinkingText: {
    fontStyle: 'italic',
  },
  messageText: {
    fontSize: 16,
    lineHeight: 20,
  },
  userMessageText: {
    color: '#fff',
  },
  videoCaption: {
    marginTop: 4,
  },
  timestamp: {
    fontSize: 12,
    marginTop: 5,
  },
  userTimestamp: {
    color: 'rgba(255,255,255,0.75)',
  },
  brainstormChoiceButtons: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 12,
  },
  brainstormButton: {
    flex: 1,
    paddingVertical: 16,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 56,
  },
  brainstormButtonText: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '600',
    textAlign: 'center',
  },
  retryButton: {
    marginTop: 10,
    alignSelf: 'flex-start',
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 6,
    borderWidth: 1,
  },
  retryButtonText: {
    fontSize: 13,
    fontWeight: '600',
  },
  progressMessageLabel: {
    fontSize: 12,
    fontWeight: '700',
    textTransform: 'uppercase',
    marginBottom: 6,
  },
  claudeAccessibilityOnly: {
    position: 'absolute',
    width: 1,
    height: 1,
    opacity: 0,
  },
  claudeHeadingText: {
    fontSize: 15,
    fontWeight: '700',
    marginTop: 6,
    marginBottom: 4,
  },
  claudeLineSpacer: {
    height: 8,
  },
  summaryCard: {
    marginHorizontal: 12,
    marginBottom: 8,
    padding: 10,
    borderRadius: 10,
    borderWidth: 1,
  },
  summaryCardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  summaryCardLabel: {
    fontSize: 12,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  summaryCardExpandHint: {
    fontSize: 12,
    fontWeight: '600',
  },
  summaryCardLatest: {
    fontSize: 13,
    fontStyle: 'italic',
    marginBottom: 4,
  },
  summaryCardText: {
    fontSize: 15,
    lineHeight: 20,
  },
  summaryModalLatest: {
    fontSize: 14,
    fontStyle: 'italic',
    marginBottom: 12,
  },
  summaryModalOverlay: {
    flex: 1,
    justifyContent: 'flex-end',
    backgroundColor: 'rgba(0,0,0,0.4)',
  },
  summaryModalSheet: {
    maxHeight: '70%',
    borderTopLeftRadius: 16,
    borderTopRightRadius: 16,
  },
  summaryModalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 14,
    borderBottomWidth: 1,
  },
  summaryModalTitle: {
    fontSize: 17,
    fontWeight: '700',
  },
  summaryModalCloseText: {
    fontSize: 16,
    fontWeight: '600',
  },
  summaryModalScroll: {
    flexGrow: 0,
  },
  summaryModalScrollContent: {
    padding: 16,
  },
  summaryModalText: {
    fontSize: 16,
    lineHeight: 22,
  },
  composeBar: {
    padding: 12,
    borderTopWidth: 1,
  },
  videoChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 8,
    borderWidth: 1,
    marginBottom: 8,
  },
  videoChipText: {
    fontSize: 14,
    fontWeight: '600',
    flex: 1,
  },
  videoActionText: {
    fontSize: 13,
    fontWeight: '600',
  },
  pickingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingVertical: 6,
    marginBottom: 8,
  },
  pickingText: {
    fontSize: 14,
    fontStyle: 'italic',
  },
  composeRow: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: 8,
  },
  attachButtons: {
    flexDirection: 'row',
    gap: 6,
  },
  attachButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  attachButtonText: {
    fontSize: 18,
  },
  textInput: {
    flex: 1,
    borderWidth: 1,
    borderRadius: 20,
    paddingHorizontal: 15,
    paddingVertical: 10,
    maxHeight: 100,
    fontSize: 16,
  },
  sendButton: {
    borderRadius: 20,
    paddingHorizontal: 18,
    paddingVertical: 10,
    minHeight: 40,
    alignItems: 'center',
    justifyContent: 'center',
  },
  sendButtonDisabled: {
    opacity: 0.5,
  },
  sendButtonText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 15,
  },
});
