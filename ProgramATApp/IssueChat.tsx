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
  findNodeHandle,
  Modal,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { launchImageLibrary } from 'react-native-image-picker';
import { useTheme } from './ThemeContext';
import WebSocketService from './WebSocketService';
import VideoRecorderModal from './VideoRecorderModal';
import { isBrainstormingEnabled, isBasicModeEnabled } from './Settings';
import { IssueChatItem, RetryDescriptor } from './IssueChatTypes';
import { submitCreation, submitUpdate, nextBrainstormQuestion } from './IssueSubmissionService';
import TextToSpeechService from './TextToSpeechService';

interface IssueChatProps {
  serverFeedback?: string;
  selectedIssue?: {number: number; title: string} | null;
  onNewIssue?: () => void;
  onBack?: () => void;
  showBackButton?: boolean;
}

type Awaiting = 'answer' | 'choice' | null;

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
  const [isVideoRecorderOpen, setIsVideoRecorderOpen] = useState(false);
  const [isPickingFromLibrary, setIsPickingFromLibrary] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [progressText, setProgressText] = useState<string | null>(null);

  const [activeToken, setActiveToken] = useState<string | null>(null);
  const [awaiting, setAwaiting] = useState<Awaiting>(null);
  const [understandingSummary, setUnderstandingSummary] = useState<string | null>(null);
  const [isSummaryModalOpen, setIsSummaryModalOpen] = useState(false);

  const [brainstormingEnabled, setBrainstormingEnabled] = useState(true);
  const [basicMode, setBasicMode] = useState(false);
  const brainstormingActive = brainstormingEnabled && !basicMode;

  const brainstormHistoryRef = useRef<Array<{question: string; answer: string}>>([]);
  const idCounterRef = useRef(0);
  const inFlightRef = useRef(false);
  const scrollViewRef = useRef<ScrollView>(null);
  const composeInputRef = useRef<RNTextInput>(null);
  const headerRef = useRef<Text>(null);
  const lastAnnouncedIdRef = useRef<string | null>(null);

  useEffect(() => {
    isBrainstormingEnabled().then(setBrainstormingEnabled).catch(() => setBrainstormingEnabled(true));
    isBasicModeEnabled().then(setBasicMode).catch(() => setBasicMode(false));
  }, []);

  // Focus the header on mount, matching ToolSelector's screen-entry pattern.
  useEffect(() => {
    const timeout = setTimeout(() => {
      if (headerRef.current) {
        const reactTag = findNodeHandle(headerRef.current);
        if (reactTag) {
          AccessibilityInfo.setAccessibilityFocus(reactTag);
        }
      }
    }, 100);
    return () => clearTimeout(timeout);
  }, []);

  // Auto-scroll to the newest message.
  useEffect(() => {
    if (scrollViewRef.current) {
      setTimeout(() => {
        scrollViewRef.current?.scrollToEnd({ animated: true });
      }, 100);
    }
  }, [items, progressText, isSending]);

  // Announce new assistant-originated items. assistant-updated is skipped
  // because App.tsx's WS listener already speaks "Update sent to issue" for
  // the issue_updated broadcast that /submit-update still triggers.
  useEffect(() => {
    const last = items[items.length - 1];
    if (!last || last.id === lastAnnouncedIdRef.current) return;
    if (last.kind === 'assistant-question') {
      lastAnnouncedIdRef.current = last.id;
      TextToSpeechService.speakWithInterrupt(last.question);
      AccessibilityInfo.announceForAccessibility(last.question);
    } else if (last.kind === 'assistant-choice-prompt') {
      lastAnnouncedIdRef.current = last.id;
      TextToSpeechService.speakWithInterrupt(last.text);
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
    }
  }, [awaiting]);

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
      }
    } finally {
      setIsPickingFromLibrary(false);
    }
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
      resetConversation();
      return;
    }

    if (result.status === 'ideation' && brainstormingActive) {
      setActiveToken(result.token);
      setAwaiting('answer');
      if (result.summary) setUnderstandingSummary(result.summary);
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
        resetConversation();
      } else {
        const message = result.status === 'error' ? result.error : 'Failed to create issue';
        append({
          kind: 'assistant-error',
          id: nextId('assistant-error'),
          ts: new Date(),
          text: message,
          retry: { op: 'start-building', token: activeToken },
        });
      }
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
      const result = await submitUpdate({ text, issueNumber: selectedIssue.number, videoUri });
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
      case 'ideation-answer':
        sendIdeationAnswer(retry.text, retry.token);
        return;
      case 'next-question':
        handleKeepBrainstorming();
        return;
      case 'start-building':
        handleStartBuilding();
        return;
    }
  };

  const handleSend = () => {
    const text = composeText.trim();
    if (!text && !stagedVideoUri) return;

    if (awaiting === 'answer' && activeToken) {
      sendIdeationAnswer(text, activeToken);
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
    resetConversation();
    Keyboard.dismiss();
  };

  // A staged video needs a caption — an empty/placeholder caption feeds
  // parse_transcript_with_ai garbage to build the issue title from.
  const sendDisabled =
    isSending ||
    awaiting === 'choice' ||
    (stagedVideoUri ? !composeText.trim() : !composeText.trim());

  const placeholder =
    awaiting === 'answer'
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
            style={[styles.messageContainer, styles.assistantAlign, { backgroundColor: theme.card, borderColor: theme.border }]}
            accessible={true}
            accessibilityLabel={`Assistant said: ${item.question}`}
            accessibilityRole="text"
            accessibilityHint="Long press to copy text">
            <Text style={[styles.messageText, { color: theme.text }]} selectable={true} accessible={false}>
              {item.question}
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
            style={[styles.messageContainer, styles.assistantAlign, { backgroundColor: theme.card, borderColor: theme.border }]}
            accessible={true}
            accessibilityLabel={`Assistant said: ${item.text}`}
            accessibilityRole="text">
            <Text style={[styles.messageText, { color: theme.text }]} accessible={false}>
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
    <KeyboardAvoidingView
      style={[styles.container, { backgroundColor: theme.background }]}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      keyboardVerticalOffset={Platform.OS === 'ios' ? 64 : 0}>
      <View style={[styles.header, { borderBottomColor: theme.border }]}>
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
            ref={headerRef}
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
        accessible={true}
        accessibilityLabel="Conversation messages"
        accessibilityLiveRegion="polite">
        {items.map(renderItem)}
        {isSending && (
          <View
            style={[styles.messageContainer, styles.assistantAlign, styles.thinkingMessage, { backgroundColor: theme.card, borderColor: theme.border }]}
            accessible={false}>
            <Text style={[styles.thinkingText, { color: theme.textSecondary }]}>
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
          accessibilityLiveRegion="polite"
          accessibilityLabel={`What I understand: ${understandingSummary}`}
          accessibilityHint="Double tap to view the full text">
          <View style={styles.summaryCardHeader}>
            <Text style={[styles.summaryCardLabel, { color: theme.textSecondary }]} accessible={false}>
              What I understand
            </Text>
            <Text style={[styles.summaryCardExpandHint, { color: theme.primary }]} accessible={false}>
              View full ›
            </Text>
          </View>
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
              onPress={() => setIsVideoRecorderOpen(true)}
              accessible={true}
              accessibilityRole="button"
              accessibilityLabel="Re-record video">
              <Text style={[styles.videoActionText, { color: theme.primary }]}>Re-record</Text>
            </TouchableOpacity>
            <TouchableOpacity
              onPress={handlePickFromLibrary}
              accessible={true}
              accessibilityRole="button"
              accessibilityLabel="Pick different video from library">
              <Text style={[styles.videoActionText, { color: theme.primary }]}>Library</Text>
            </TouchableOpacity>
            <TouchableOpacity
              onPress={() => setStagedVideoUri(null)}
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
                onPress={() => setIsVideoRecorderOpen(true)}
                accessible={true}
                accessibilityRole="button"
                accessibilityLabel="Record a new video">
                <Text style={styles.attachButtonText}>📹</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.attachButton, { backgroundColor: theme.backgroundSecondary, borderColor: theme.border }]}
                onPress={handlePickFromLibrary}
                accessible={true}
                accessibilityRole="button"
                accessibilityLabel="Choose a video from your photo library">
                <Text style={styles.attachButtonText}>📁</Text>
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
            setIsVideoRecorderOpen(false);
          }}
          onCancel={() => setIsVideoRecorderOpen(false)}
        />
      )}
    </KeyboardAvoidingView>
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
  summaryCardText: {
    fontSize: 15,
    lineHeight: 20,
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
