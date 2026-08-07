import React from 'react';
import ReactTestRenderer, { act } from 'react-test-renderer';
import { AccessibilityInfo, ScrollView, Text, TextInput, TouchableOpacity, View } from 'react-native';
import IssueChat, { shouldAutoScrollForNewItems } from '../IssueChat';
import PRsAndText from '../PRsAndText';
import { ThemeProvider } from '../ThemeContext';
import {
  buildClaudeAccessibilityLabel,
  buildClaudeAccessibilitySections,
  getChangedClaudeAnnouncement,
  parseClaudeRenderLines,
  sanitizeClaudeCommentBody,
} from '../claudeCommentSanitizer';

jest.useFakeTimers();

jest.mock('@react-native-async-storage/async-storage', () => ({
  getItem: jest.fn(() => Promise.resolve(null)),
  setItem: jest.fn(() => Promise.resolve()),
}));

jest.mock('react-native-safe-area-context', () => ({
  SafeAreaView: ({ children }: any) => children,
}));

jest.mock('react-native-image-picker', () => ({
  launchImageLibrary: jest.fn(),
}), { virtual: true });

jest.mock('../Settings', () => ({
  isBrainstormingEnabled: jest.fn(() => Promise.resolve(true)),
  isBasicModeEnabled: jest.fn(() => Promise.resolve(false)),
}));

jest.mock('../WebSocketService', () => ({
  __esModule: true,
  default: {
    addMessageListener: jest.fn(),
    removeMessageListener: jest.fn(),
    getServerUrl: jest.fn(() => 'http://localhost:8081'),
  },
}));

jest.mock('../TextToSpeechService', () => ({
  __esModule: true,
  default: { speakWithInterrupt: jest.fn() },
}));

const mockPlayBeep = jest.fn();
const mockPlayLoadingSound = jest.fn();
const mockStopLoadingSound = jest.fn();

jest.mock('../BeepService', () => ({
  __esModule: true,
  default: {
    playBeep: (...args: any[]) => mockPlayBeep(...args),
    playLoadingSound: (...args: any[]) => mockPlayLoadingSound(...args),
    stopLoadingSound: (...args: any[]) => mockStopLoadingSound(...args),
  },
}));

jest.mock('../VideoRecorderModal', () => 'VideoRecorderModal');
jest.mock('../IssueSelector', () => 'IssueSelector');
jest.mock('../ReviewPane', () => 'ReviewPane');

const mockSubmitUpdate = jest.fn();
const mockSubmitCreation = jest.fn();
const mockNextBrainstormQuestion = jest.fn();
const mockFetchClaudeProgress = jest.fn();

jest.mock('../IssueSubmissionService', () => ({
  submitUpdate: (...args: any[]) => mockSubmitUpdate(...args),
  submitCreation: (...args: any[]) => mockSubmitCreation(...args),
  nextBrainstormQuestion: (...args: any[]) => mockNextBrainstormQuestion(...args),
  fetchClaudeProgress: (...args: any[]) => mockFetchClaudeProgress(...args),
}));

const announceSpy = jest.spyOn(AccessibilityInfo, 'announceForAccessibility').mockImplementation(jest.fn());

const setTimeoutSpy = jest.spyOn(global, 'setTimeout');
const mountedRenderers: ReactTestRenderer.ReactTestRenderer[] = [];

function renderWithTheme(element: React.ReactElement) {
  const renderer = ReactTestRenderer.create(<ThemeProvider>{element}</ThemeProvider>, {
    createNodeMock: (node) => {
      if (node.type === TextInput) {
        return {
          focus: jest.fn(),
        };
      }
      return {};
    },
  });
  mountedRenderers.push(renderer);
  return renderer;
}

describe('IssueChat progress', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockSubmitCreation.mockReset();
    mockSubmitUpdate.mockReset();
    mockNextBrainstormQuestion.mockReset();
    mockFetchClaudeProgress.mockReset();
    mockPlayBeep.mockReset();
    mockPlayLoadingSound.mockReset();
    mockStopLoadingSound.mockReset();
  });

  afterEach(async () => {
    while (mountedRenderers.length > 0) {
      const renderer = mountedRenderers.pop();
      if (renderer) {
        await act(async () => {
          renderer.unmount();
        });
      }
    }
    jest.clearAllTimers();
  });

  const count100msTimeouts = () => setTimeoutSpy.mock.calls.filter((call) => call[1] === 100).length;
  const count6000msTimeouts = () => setTimeoutSpy.mock.calls.filter((call) => call[1] === 6000).length;

  test('polls for Claude progress every 5 seconds and cleans up on unmount', async () => {
    mockSubmitUpdate.mockResolvedValue({
      status: 'updated',
      issue_number: 42,
      issue_url: 'https://github.test/issues/42',
      video_summary: '',
      pr_number: 42,
    });
    mockFetchClaudeProgress.mockResolvedValue({
      status: 'available',
      title: 'Creating Tool',
      issue_number: 42,
      comment_id: 100,
      body: 'Working on the PR update now.',
      message: 'Claude comment available.',
      steps: [],
    });

    let renderer: ReactTestRenderer.ReactTestRenderer;
    await act(async () => {
      renderer = renderWithTheme(<IssueChat selectedIssue={{ number: 42, title: 'PR 42' }} />);
    });

    const input = renderer!.root.findByType(TextInput);
    await act(async () => {
      input.props.onChangeText('Please update the tool');
    });
    const sendButton = renderer!.root.findAllByType(TouchableOpacity).find((node) => node.props.accessibilityLabel === 'Send text');
    expect(sendButton).toBeTruthy();

    await act(async () => {
      sendButton!.props.onPress();
    });

    const initialCallCount = mockFetchClaudeProgress.mock.calls.length;
    expect(initialCallCount).toBeGreaterThanOrEqual(1);

    await act(async () => {
      jest.advanceTimersByTime(4999);
    });
    expect(mockFetchClaudeProgress).toHaveBeenCalledTimes(initialCallCount);

    await act(async () => {
      jest.advanceTimersByTime(1);
      await Promise.resolve();
    });
    const callCountBeforeUnmount = mockFetchClaudeProgress.mock.calls.length;
    expect(callCountBeforeUnmount).toBe(initialCallCount + 1);

    await act(async () => {
      renderer!.unmount();
    });
    await act(async () => {
      jest.advanceTimersByTime(5000);
    });

    expect(mockFetchClaudeProgress).toHaveBeenCalledTimes(callCountBeforeUnmount);
  });

  test('update polling passes the current update request boundary and waits quietly before a new Claude comment exists', async () => {
    mockSubmitUpdate.mockResolvedValue({
      status: 'updated',
      issue_number: 142,
      issue_url: 'https://github.test/issues/142',
      video_summary: '',
      pr_number: 242,
      comment_id: 9001,
      comment_created_at: '2026-08-07T12:00:00Z',
    });
    mockFetchClaudeProgress
      .mockResolvedValueOnce({
        status: 'waiting_for_comment',
        issue_number: 242,
        comment_id: null,
        body: '',
        message: 'Claude progress comment has not been created yet.',
        steps: [],
        updated_at: null,
      })
      .mockResolvedValueOnce({
        status: 'available',
        issue_number: 242,
        comment_id: 9002,
        updated_at: '2026-08-07T12:00:10Z',
        body: 'New Claude update after this request',
        message: 'Claude comment available.',
        steps: [],
      });

    let renderer: ReactTestRenderer.ReactTestRenderer;
    await act(async () => {
      renderer = renderWithTheme(<IssueChat selectedIssue={{ number: 142, title: 'PR 142' }} />);
    });

    const input = renderer!.root.findByType(TextInput);
    await act(async () => {
      input.props.onChangeText('Please update the tool');
    });
    const sendButton = renderer!.root.findAllByType(TouchableOpacity).find((node) => node.props.accessibilityLabel === 'Send text');

    await act(async () => {
      sendButton!.props.onPress();
    });

    expect(mockFetchClaudeProgress).toHaveBeenCalledWith({
      mode: 'update',
      prNumber: 242,
      commentId: null,
      afterCommentId: 9001,
      afterTimestamp: '2026-08-07T12:00:00Z',
    });
    expect(JSON.stringify(renderer!.toJSON())).not.toContain('New Claude update after this request');

    await act(async () => {
      jest.advanceTimersByTime(5000);
      await Promise.resolve();
    });

    const rendered = JSON.stringify(renderer!.toJSON());
    expect(rendered).toContain('New Claude update after this request');
    expect(rendered).not.toContain('Claude progress comment has not been created yet.');
  });

  test('renders section-level VoiceOver nodes for Claude progress and announces active step once', async () => {
    mockSubmitUpdate.mockResolvedValue({
      status: 'updated',
      issue_number: 7,
      issue_url: 'https://github.test/issues/7',
      video_summary: '',
      pr_number: 7,
    });
    mockFetchClaudeProgress
      .mockResolvedValueOnce({
        status: 'available',
        title: 'Creating Tool',
        issue_number: 7,
        comment_id: 101,
        body: '### Progress\n\n- [x] Read issue and repository CLAUDE.md\n- [ ] Implement tool.py\n\n### Recent work\nInspected runtime input handling.',
        message: 'Claude comment available.',
        steps: [
          { id: 'step_1', label: 'Reading your tool requirements', raw_label: 'Read issue and repository CLAUDE.md', status: 'completed' },
          { id: 'step_2', label: 'Building the tool', raw_label: 'Implement tool.py', status: 'in_progress' },
        ],
      })
      .mockResolvedValueOnce({
        status: 'available',
        title: 'Creating Tool',
        issue_number: 7,
        comment_id: 101,
        body: '### Progress\n\n- [x] Read issue and repository CLAUDE.md\n- [ ] Implement tool.py\n\n### Recent work\nInspected runtime input handling.',
        message: 'Claude comment available.',
        steps: [
          { id: 'step_1', label: 'Reading your tool requirements', raw_label: 'Read issue and repository CLAUDE.md', status: 'completed' },
          { id: 'step_2', label: 'Building the tool', raw_label: 'Implement tool.py', status: 'in_progress' },
        ],
      });

    let renderer: ReactTestRenderer.ReactTestRenderer;
    await act(async () => {
      renderer = renderWithTheme(<IssueChat selectedIssue={{ number: 7, title: 'PR 7' }} />);
    });

    const input = renderer!.root.findByType(TextInput);
    await act(async () => {
      input.props.onChangeText('Update it');
    });
    const sendButton = renderer!.root.findAllByType(TouchableOpacity).find((node) => node.props.accessibilityLabel === 'Send text');

    await act(async () => {
      sendButton!.props.onPress();
    });

    const progressSection = renderer!.root.findAll((node) =>
      node.type === View
      && node.props?.accessible === true
      && node.props?.accessibilityLabel === 'Progress. Finished: Read issue and repository CLAUDE.md. Ongoing: Implement tool.py'
    );
    const recentWorkSection = renderer!.root.findAll((node) =>
      node.type === View
      && node.props?.accessible === true
      && node.props?.accessibilityLabel === 'Recent work. Inspected runtime input handling.'
    );
    expect(progressSection).toHaveLength(1);
    expect(recentWorkSection).toHaveLength(1);
    expect(announceSpy).toHaveBeenCalledWith('Recent Work: Inspected runtime input handling.');

    await act(async () => {
      jest.advanceTimersByTime(5000);
    });
    expect(announceSpy.mock.calls.filter((call) => call[0] === 'In progress: Building the tool')).toHaveLength(1);
  });

  test('does not show progress UI outside Create and Update pages', async () => {
    let renderer: ReactTestRenderer.ReactTestRenderer;
    await act(async () => {
      renderer = renderWithTheme(
        <PRsAndText
          selectedIssue={null}
          onIssueSelect={jest.fn()}
          onNewIssue={jest.fn()}
          prList={[]}
          appMode="development"
        />,
      );
    });

    expect(renderer!.root.findAll((node) => node.props?.testID === 'claude-progress-card')).toHaveLength(0);
  });

  test('updates one in-flow Claude message without duplicates while continuing after completed-looking content', async () => {
    mockSubmitUpdate.mockResolvedValue({
      status: 'updated',
      issue_number: 9,
      issue_url: 'https://github.test/issues/9',
      video_summary: '',
      pr_number: 9,
    });
    mockFetchClaudeProgress
      .mockResolvedValueOnce({
        status: 'available',
        issue_number: 9,
        comment_id: 201,
        updated_at: '2026-08-05T12:00:00Z',
        body: 'First Claude update',
        message: 'Claude comment available.',
        steps: [],
      })
      .mockResolvedValueOnce({
        status: 'available',
        issue_number: 9,
        comment_id: 201,
        updated_at: '2026-08-05T12:00:05Z',
        body: '### Progress\n- [x] First task\n- [x] Second task\n\n### Implementation summary\nCompleted summary text, still reviewing.',
        message: 'Claude comment available.',
        steps: [
          { id: 'step_1', label: 'First task', raw_label: 'First task', status: 'completed' },
          { id: 'step_2', label: 'Second task', raw_label: 'Second task', status: 'completed' },
        ],
      })
      .mockResolvedValueOnce({
        status: 'available',
        issue_number: 9,
        comment_id: 201,
        updated_at: '2026-08-05T12:00:15Z',
        body: 'Final Claude update after review',
        message: 'Claude comment available.',
        steps: [],
      });

    let renderer: ReactTestRenderer.ReactTestRenderer;
    await act(async () => {
      renderer = renderWithTheme(<IssueChat selectedIssue={{ number: 9, title: 'PR 9' }} />);
    });
    const input = renderer!.root.findByType(TextInput);
    await act(async () => {
      input.props.onChangeText('Update it');
    });
    const sendButton = renderer!.root.findAllByType(TouchableOpacity).find((node) => node.props.accessibilityLabel === 'Send text');
    await act(async () => {
      sendButton!.props.onPress();
    });
    await act(async () => {
      jest.advanceTimersByTime(5000);
      await Promise.resolve();
    });
    await act(async () => {
      jest.advanceTimersByTime(5000);
      await Promise.resolve();
    });

    const claudeTexts = renderer!.root.findAll((node) => node.type === 'Text' && node.props?.children === 'Claude');
    expect(claudeTexts).toHaveLength(1);
    expect(mockFetchClaudeProgress.mock.calls.length).toBeGreaterThanOrEqual(3);
    expect(JSON.stringify(renderer!.toJSON())).toContain('Final Claude update after review');
  });

  test('appending a normal new message still triggers the auto-scroll decision', () => {
    expect(
      shouldAutoScrollForNewItems(
        { itemCount: 2, lastItemId: 'assistant-created-1' },
        { itemCount: 3, lastItemId: 'user-text-2' },
      ),
    ).toBe(true);
  });

  test('first Claude insertion is allowed to auto-follow once and later polling never scrolls again', async () => {
    mockSubmitUpdate.mockResolvedValue({
      status: 'updated',
      issue_number: 55,
      issue_url: 'https://github.test/issues/55',
      video_summary: '',
      pr_number: 55,
    });
    mockFetchClaudeProgress
      .mockResolvedValueOnce({
        status: 'available',
        issue_number: 55,
        comment_id: 601,
        updated_at: '2026-08-07T10:00:00Z',
        body: '### Progress\n- [ ] Build tool',
        message: 'Claude comment available.',
        steps: [{ id: 'step_1', label: 'Build tool', raw_label: 'Build tool', status: 'in_progress' }],
      })
      .mockResolvedValueOnce({
        status: 'available',
        issue_number: 55,
        comment_id: 601,
        updated_at: '2026-08-07T10:00:10Z',
        body: '### Progress\n- [x] Build tool\n\n### Recent work\nValidated output.',
        message: 'Claude comment available.',
        steps: [{ id: 'step_1', label: 'Build tool', raw_label: 'Build tool', status: 'completed' }],
      });

    let renderer: ReactTestRenderer.ReactTestRenderer;
    await act(async () => {
      renderer = renderWithTheme(<IssueChat selectedIssue={{ number: 55, title: 'PR 55' }} />);
    });
    const input = renderer!.root.findByType(TextInput);

    await act(async () => {
      input.props.onChangeText('Initial user message');
    });
    const sendButton = renderer!.root.findAllByType(TouchableOpacity).find((node) => node.props.accessibilityLabel === 'Send text');
    const baseline100msTimeouts = count100msTimeouts();

    await act(async () => {
      sendButton!.props.onPress();
      jest.advanceTimersByTime(150);
      await Promise.resolve();
    });

    const timeoutCountBeforeClaudePolling = count100msTimeouts();
    expect(timeoutCountBeforeClaudePolling).toBeGreaterThan(baseline100msTimeouts);

    const timeoutCountAfterFirstClaudeInsert = timeoutCountBeforeClaudePolling;

    await act(async () => {
      jest.advanceTimersByTime(5000);
      await Promise.resolve();
    });

    expect(count100msTimeouts()).toBe(timeoutCountAfterFirstClaudeInsert);
  });

  test('updating the existing Claude progress message does not trigger the auto-scroll decision and keeps a stable message id', async () => {
    expect(
      shouldAutoScrollForNewItems(
        { itemCount: 4, lastItemId: 'claude-progress-stable' },
        { itemCount: 4, lastItemId: 'claude-progress-stable' },
      ),
    ).toBe(false);

    mockSubmitUpdate.mockResolvedValue({
      status: 'updated',
      issue_number: 55,
      issue_url: 'https://github.test/issues/55',
      video_summary: '',
      pr_number: 55,
    });
    mockFetchClaudeProgress
      .mockResolvedValueOnce({
        status: 'available',
        issue_number: 55,
        comment_id: 601,
        updated_at: '2026-08-07T10:00:00Z',
        body: '### Progress\n- [ ] Build tool',
        message: 'Claude comment available.',
        steps: [{ id: 'step_1', label: 'Build tool', raw_label: 'Build tool', status: 'in_progress' }],
      })
      .mockResolvedValueOnce({
        status: 'available',
        issue_number: 55,
        comment_id: 601,
        updated_at: '2026-08-07T10:00:10Z',
        body: '### Progress\n- [x] Build tool\n\n### Recent work\nValidated output.',
        message: 'Claude comment available.',
        steps: [{ id: 'step_1', label: 'Build tool', raw_label: 'Build tool', status: 'completed' }],
      });

    let renderer: ReactTestRenderer.ReactTestRenderer;
    await act(async () => {
      renderer = renderWithTheme(<IssueChat selectedIssue={{ number: 55, title: 'PR 55' }} />);
    });
    const input = renderer!.root.findByType(TextInput);
    await act(async () => {
      input.props.onChangeText('Update it');
    });
    const sendButton = renderer!.root.findAllByType(TouchableOpacity).find((node) => node.props.accessibilityLabel === 'Send text');
    await act(async () => {
      sendButton!.props.onPress();
    });
    await act(async () => {
      jest.advanceTimersByTime(150);
      await Promise.resolve();
    });

    const firstContainer = renderer!.root.findAll((node) => typeof node.props?.testID === 'string' && node.props.testID.startsWith('claude-progress-'))[0];
    const stableTestId = firstContainer.props.testID;

    await act(async () => {
      jest.advanceTimersByTime(5000);
      await Promise.resolve();
    });

    const secondContainer = renderer!.root.findAll((node) => typeof node.props?.testID === 'string' && node.props.testID.startsWith('claude-progress-'))[0];
    expect(secondContainer.props.testID).toBe(stableTestId);
  });

  test('starting a new conversation resets the Claude auto-follow session flag', async () => {
    mockSubmitUpdate.mockResolvedValue({
      status: 'updated',
      issue_number: 77,
      issue_url: 'https://github.test/issues/77',
      video_summary: '',
      pr_number: 77,
    });
    mockFetchClaudeProgress.mockResolvedValue({
      status: 'available',
      issue_number: 77,
      comment_id: 701,
      updated_at: '2026-08-07T11:00:00Z',
      body: '### Progress\n- [ ] Build tool',
      message: 'Claude comment available.',
      steps: [{ id: 'step_1', label: 'Build tool', raw_label: 'Build tool', status: 'in_progress' }],
    });

    let renderer: ReactTestRenderer.ReactTestRenderer;
    await act(async () => {
      renderer = renderWithTheme(<IssueChat selectedIssue={{ number: 77, title: 'PR 77' }} />);
    });

    const input = renderer!.root.findByType(TextInput);
    await act(async () => {
      input.props.onChangeText('First request');
    });
    const sendButton = () => renderer!.root.findAllByType(TouchableOpacity).find((node) => node.props.accessibilityLabel === 'Send text')!;
    const baseline100msTimeouts = count100msTimeouts();

    await act(async () => {
      sendButton().props.onPress();
      jest.advanceTimersByTime(150);
      await Promise.resolve();
    });
    const firstSessionTimeoutCount = count100msTimeouts();
    expect(firstSessionTimeoutCount).toBeGreaterThan(baseline100msTimeouts);

    const newConversationButton = renderer!.root.findAllByType(TouchableOpacity).find((node) => node.props.accessibilityLabel === 'Start a new conversation');
    await act(async () => {
      newConversationButton!.props.onPress();
    });

    await act(async () => {
      input.props.onChangeText('Second request');
    });
    await act(async () => {
      sendButton().props.onPress();
      jest.advanceTimersByTime(150);
      await Promise.resolve();
    });

    expect(count100msTimeouts()).toBeGreaterThan(firstSessionTimeoutCount);
  });

  test('manual ScrollView scrolling remains enabled after Claude appears', async () => {
    mockSubmitUpdate.mockResolvedValue({
      status: 'updated',
      issue_number: 91,
      issue_url: 'https://github.test/issues/91',
      video_summary: '',
      pr_number: 91,
    });
    mockFetchClaudeProgress.mockResolvedValue({
      status: 'available',
      issue_number: 91,
      comment_id: 991,
      body: '### Progress\n- [ ] Build tool',
      message: 'Claude comment available.',
      steps: [{ id: 'step_1', label: 'Build tool', raw_label: 'Build tool', status: 'in_progress' }],
    });

    let renderer: ReactTestRenderer.ReactTestRenderer;
    await act(async () => {
      renderer = renderWithTheme(<IssueChat selectedIssue={{ number: 91, title: 'PR 91' }} />);
    });
    const input = renderer!.root.findByType(TextInput);
    await act(async () => {
      input.props.onChangeText('Update it');
    });
    const sendButton = renderer!.root.findAllByType(TouchableOpacity).find((node) => node.props.accessibilityLabel === 'Send text');
    await act(async () => {
      sendButton!.props.onPress();
    });

    const messageScrollView = renderer!.root.findAllByType(ScrollView).find((node) => node.props?.keyboardShouldPersistTaps === 'handled');
    expect(messageScrollView).toBeTruthy();
    expect(messageScrollView!.props.scrollEnabled).not.toBe(false);
    expect(messageScrollView!.props.accessibilityElementsHidden).toBeUndefined();
    expect(messageScrollView!.props.importantForAccessibility).toBeUndefined();
  });

  test('keeps polling after completion-like Claude text but stops loading sound', async () => {
    mockSubmitUpdate.mockResolvedValue({
      status: 'updated',
      issue_number: 21,
      issue_url: 'https://github.test/issues/21',
      video_summary: '',
      pr_number: 21,
    });
    mockFetchClaudeProgress
      .mockResolvedValueOnce({
        status: 'available',
        issue_number: 21,
        comment_id: 401,
        body: '### Progress\n- [x] Build tool\n- [ ] Run validation\n\n### Recent work\nValidation is still running.',
        message: 'Claude comment available.',
        steps: [
          { id: 'step_1', label: 'Build tool', raw_label: 'Build tool', status: 'completed' },
          { id: 'step_2', label: 'Run validation', raw_label: 'Run validation', status: 'in_progress' },
        ],
      })
      .mockResolvedValueOnce({
        status: 'available',
        issue_number: 21,
        comment_id: 401,
        body: 'Done\n- [x] Build tool',
        message: 'Claude finished all checklist steps.',
        steps: [{ id: 'step_1', label: 'Build tool', raw_label: 'Build tool', status: 'completed' }],
      });

    let renderer: ReactTestRenderer.ReactTestRenderer;
    await act(async () => {
      renderer = renderWithTheme(<IssueChat selectedIssue={{ number: 21, title: 'PR 21' }} />);
    });
    const input = renderer!.root.findByType(TextInput);
    await act(async () => {
      input.props.onChangeText('Update it');
    });
    const sendButton = renderer!.root.findAllByType(TouchableOpacity).find((node) => node.props.accessibilityLabel === 'Send text');
    await act(async () => {
      sendButton!.props.onPress();
    });

    expect(mockPlayLoadingSound).toHaveBeenCalled();

    await act(async () => {
      jest.advanceTimersByTime(5000);
      await Promise.resolve();
    });

    expect(mockStopLoadingSound).toHaveBeenCalled();

    await act(async () => {
      await Promise.resolve();
    });
    await act(async () => {
      jest.advanceTimersByTime(5000);
      await Promise.resolve();
    });
    expect(mockFetchClaudeProgress.mock.calls.length).toBeGreaterThanOrEqual(2);
    await act(async () => {
      renderer!.unmount();
    });
  });

  test('prevents stale Claude responses from overwriting a newer comment', async () => {
    mockSubmitUpdate.mockResolvedValue({
      status: 'updated',
      issue_number: 44,
      issue_url: 'https://github.test/issues/44',
      video_summary: '',
      pr_number: 44,
    });
    mockFetchClaudeProgress
      .mockResolvedValueOnce({
        status: 'available',
        issue_number: 44,
        comment_id: 501,
        updated_at: '2026-08-06T15:10:10Z',
        body: 'Newer Claude update',
        message: 'Claude comment available.',
        steps: [],
      })
      .mockResolvedValueOnce({
        status: 'available',
        issue_number: 44,
        comment_id: 501,
        updated_at: '2026-08-06T15:10:00Z',
        body: 'Older Claude update',
        message: 'Claude comment available.',
        steps: [],
      });

    let renderer: ReactTestRenderer.ReactTestRenderer;
    await act(async () => {
      renderer = renderWithTheme(<IssueChat selectedIssue={{ number: 44, title: 'PR 44' }} />);
    });
    const input = renderer!.root.findByType(TextInput);
    await act(async () => {
      input.props.onChangeText('Update it');
    });
    const sendButton = renderer!.root.findAllByType(TouchableOpacity).find((node) => node.props.accessibilityLabel === 'Send text');
    await act(async () => {
      sendButton!.props.onPress();
    });

    await act(async () => {
      jest.advanceTimersByTime(5000);
      await Promise.resolve();
    });

    const rendered = JSON.stringify(renderer!.toJSON());
    expect(rendered).toContain('Newer Claude update');
    expect(rendered).not.toContain('Older Claude update');
  });

  test('renders Claude progress inside the normal message list for create and update', async () => {
    mockSubmitCreation.mockResolvedValue({
      status: 'created',
      issue_number: 12,
      issue_url: 'https://github.test/issues/12',
      video_summary: '',
    });
    mockFetchClaudeProgress.mockResolvedValue({
      status: 'available',
      issue_number: 12,
      comment_id: 300,
      body: 'Claude create progress',
      message: 'Claude comment available.',
      steps: [],
    });

    let createRenderer: ReactTestRenderer.ReactTestRenderer;
    await act(async () => {
      createRenderer = renderWithTheme(<IssueChat />);
    });
    const createInput = createRenderer!.root.findByType(TextInput);
    await act(async () => {
      createInput.props.onChangeText('Create it');
    });
    const createSend = createRenderer!.root.findAllByType(TouchableOpacity).find((node) => node.props.accessibilityLabel === 'Send text');
    await act(async () => {
      createSend!.props.onPress();
    });

    const createJson = JSON.stringify(createRenderer!.toJSON());
    expect(createJson).toContain('Issue #12 created.');
    expect(createJson).toContain('Claude create progress');

    mockSubmitUpdate.mockResolvedValue({
      status: 'updated',
      issue_number: 13,
      issue_url: 'https://github.test/issues/13',
      video_summary: '',
      pr_number: 13,
    });
    mockFetchClaudeProgress.mockResolvedValue({
      status: 'available',
      issue_number: 13,
      comment_id: 301,
      body: 'Claude update progress',
      message: 'Claude comment available.',
      steps: [],
    });

    let updateRenderer: ReactTestRenderer.ReactTestRenderer;
    await act(async () => {
      updateRenderer = renderWithTheme(<IssueChat selectedIssue={{ number: 13, title: 'PR 13' }} />);
    });
    const updateInput = updateRenderer!.root.findByType(TextInput);
    await act(async () => {
      updateInput.props.onChangeText('Update it');
    });
    const updateSend = updateRenderer!.root.findAllByType(TouchableOpacity).find((node) => node.props.accessibilityLabel === 'Send text');
    await act(async () => {
      updateSend!.props.onPress();
    });

    const updateJson = JSON.stringify(updateRenderer!.toJSON());
    expect(updateJson).toContain('Issue #13 updated.');
    expect(updateJson).toContain('Claude update progress');
  });

  test('update continues using one in-flow Claude message while newer poll responses replace older waiting state', async () => {
    mockSubmitUpdate.mockResolvedValue({
      status: 'updated',
      issue_number: 88,
      issue_url: 'https://github.test/issues/88',
      video_summary: '',
      pr_number: 188,
      comment_id: 5001,
      comment_created_at: '2026-08-07T13:00:00Z',
    });
    mockFetchClaudeProgress
      .mockResolvedValueOnce({
        status: 'waiting_for_comment',
        issue_number: 188,
        comment_id: null,
        body: '',
        message: 'Claude progress comment has not been created yet.',
        steps: [],
        updated_at: null,
      })
      .mockResolvedValueOnce({
        status: 'available',
        issue_number: 188,
        comment_id: 5002,
        updated_at: '2026-08-07T13:00:05Z',
        body: '### Progress\n- [ ] Build tool',
        message: 'Claude comment available.',
        steps: [{ id: 'step_1', label: 'Build tool', raw_label: 'Build tool', status: 'in_progress' }],
      })
      .mockResolvedValueOnce({
        status: 'available',
        issue_number: 188,
        comment_id: 5002,
        updated_at: '2026-08-07T13:00:15Z',
        body: '### Progress\n- [x] Build tool\n\n### Recent work\nValidated output.',
        message: 'Claude comment available.',
        steps: [{ id: 'step_1', label: 'Build tool', raw_label: 'Build tool', status: 'completed' }],
      });

    let renderer: ReactTestRenderer.ReactTestRenderer;
    await act(async () => {
      renderer = renderWithTheme(<IssueChat selectedIssue={{ number: 88, title: 'PR 88' }} />);
    });
    const input = renderer!.root.findByType(TextInput);
    await act(async () => {
      input.props.onChangeText('Update it');
    });
    const sendButton = renderer!.root.findAllByType(TouchableOpacity).find((node) => node.props.accessibilityLabel === 'Send text');
    await act(async () => {
      sendButton!.props.onPress();
    });
    await act(async () => {
      jest.advanceTimersByTime(5000);
      await Promise.resolve();
    });
    await act(async () => {
      jest.advanceTimersByTime(5000);
      await Promise.resolve();
    });

    const claudeLabels = renderer!.root.findAll((node) => node.type === 'Text' && node.props?.children === 'Claude');
    expect(claudeLabels).toHaveLength(1);
    expect(JSON.stringify(renderer!.toJSON())).toContain('Validated output.');
  });

  test('does not overlap loading audio while the previous audio is still playing', async () => {
    mockSubmitUpdate.mockResolvedValue({
      status: 'updated',
      issue_number: 66,
      issue_url: 'https://github.test/issues/66',
      video_summary: '',
      pr_number: 66,
    });
    let resolveLoadingSound: (() => void) | null = null;
    mockPlayLoadingSound.mockImplementation(() => new Promise<void>((resolve) => {
      resolveLoadingSound = resolve;
    }));
    mockFetchClaudeProgress
      .mockResolvedValueOnce({
        status: 'available',
        issue_number: 66,
        comment_id: 901,
        updated_at: '2026-08-07T12:00:00Z',
        body: '### Progress\n- [ ] Build tool',
        message: 'Claude comment available.',
        steps: [{ id: 'step_1', label: 'Build tool', raw_label: 'Build tool', status: 'in_progress' }],
      })
      .mockResolvedValueOnce({
        status: 'available',
        issue_number: 66,
        comment_id: 901,
        updated_at: '2026-08-07T12:00:06Z',
        body: '### Progress\n- [ ] Build tool\n\n### Recent work\nStill running.',
        message: 'Claude comment available.',
        steps: [{ id: 'step_1', label: 'Build tool', raw_label: 'Build tool', status: 'in_progress' }],
      });

    let renderer: ReactTestRenderer.ReactTestRenderer;
    await act(async () => {
      renderer = renderWithTheme(<IssueChat selectedIssue={{ number: 66, title: 'PR 66' }} />);
    });
    const input = renderer!.root.findByType(TextInput);
    await act(async () => {
      input.props.onChangeText('Update it');
    });
    const sendButton = renderer!.root.findAllByType(TouchableOpacity).find((node) => node.props.accessibilityLabel === 'Send text');
    await act(async () => {
      sendButton!.props.onPress();
    });

    expect(mockPlayLoadingSound).toHaveBeenCalledTimes(1);
    expect(count6000msTimeouts()).toBe(0);

    await act(async () => {
      jest.advanceTimersByTime(6000);
      await Promise.resolve();
    });

    expect(mockPlayLoadingSound).toHaveBeenCalledTimes(1);
    expect(count6000msTimeouts()).toBe(0);

    await act(async () => {
      resolveLoadingSound?.();
      await Promise.resolve();
    });

    expect(count6000msTimeouts()).toBeGreaterThan(0);

    await act(async () => {
      jest.advanceTimersByTime(5999);
      await Promise.resolve();
    });

    expect(mockPlayLoadingSound).toHaveBeenCalledTimes(1);

    await act(async () => {
      jest.advanceTimersByTime(1);
      await Promise.resolve();
    });

    expect(mockPlayLoadingSound).toHaveBeenCalledTimes(2);
  });
});

describe('sanitizeClaudeCommentBody', () => {
  test('removes img tags and image markdown', () => {
    const input = `Before\n<img src="https://example.com/spinner.gif" width="14px" />\n![spinner](https://example.com/spinner.gif)\nAfter`;
    expect(sanitizeClaudeCommentBody(input)).toBe('Before\n\nAfter');
  });

  test('preserves meaningful text and checklist content', () => {
    const input = `### Progress\n<div>Working on validation</div>\n- [x] Read issue\n- [ ] Validate tool\nPR: https://github.com/example/repo/pull/5\nSee [the pull request](https://github.com/example/repo/pull/5)`;
    expect(sanitizeClaudeCommentBody(input)).toContain('### Progress');
    expect(sanitizeClaudeCommentBody(input)).toContain('- [x] Read issue');
    expect(sanitizeClaudeCommentBody(input)).toContain('Working on validation');
    expect(sanitizeClaudeCommentBody(input)).toContain('See the pull request');
    expect(sanitizeClaudeCommentBody(input)).not.toContain('https://github.com/example/repo/pull/5');
  });

  test('removes raw media attachment urls and unsupported html', () => {
    const input = `<div><img src="https://github.com/user-attachments/files/spinner.gif" /></div>\nhttps://github.com/user-attachments/files/spinner.gif\nError: validation failed`;
    const output = sanitizeClaudeCommentBody(input);
    expect(output).not.toContain('<img');
    expect(output).not.toContain('spinner.gif');
    expect(output).toContain('Error: validation failed');
  });

  test('removes manual create-pr link and compare url', () => {
    const input = `Create a PR\nhttps://github.com/org/repo/compare/main...very-long-branch?expand=1\nUseful summary\nhttps://github.com/org/repo/pull/5`;
    const output = sanitizeClaudeCommentBody(input);
    expect(output).not.toContain('Create a PR');
    expect(output).not.toContain('/compare/main...very-long-branch');
    expect(output).toContain('Useful summary');
    expect(output).not.toContain('/pull/5');
  });

  test('preserves anchor text while removing html links and raw urls', () => {
    const input = `Review <a href="https://github.com/example/repo/pull/5">this fix</a> next.\nMore context: https://example.com/details`;
    const output = sanitizeClaudeCommentBody(input);
    expect(output).toContain('Review this fix next.');
    expect(output).toContain('More context:');
    expect(output).not.toContain('https://github.com/example/repo/pull/5');
    expect(output).not.toContain('https://example.com/details');
  });
});

describe('Claude accessibility helpers', () => {
  test('normalizes checklist items for VoiceOver', () => {
    const output = buildClaudeAccessibilityLabel(
      '- [x] Read issue and repository `CLAUDE.md`\n- [x] Implement `tools/exit_finder.py`\n- [ ] Add focused tests\n- [] Commit and push',
      'Claude comment available.',
    );
    expect(output).toContain('Finished: Read issue and repository CLAUDE.md');
    expect(output).toContain('Finished: Implement tools/exit_finder.py');
    expect(output).toContain('Ongoing: Add focused tests');
    expect(output).toContain('To do: Commit and push');
    expect(output).not.toContain('[x]');
    expect(output).not.toContain('`');
  });

  test('extracts changed section announcements', () => {
    const previous = '### Recent work\nChecked tool patterns.\n\n### Next step\nImplement the tool.';
    const next = '### Recent work\nValidated the generated tool.\n\n### Next step\nCommit and push.';
    expect(getChangedClaudeAnnouncement(previous, next)).toBe('Recent Work: Validated the generated tool.');
  });

  test('parses render lines with separate headings and bullets', () => {
    const lines = parseClaudeRenderLines('### Current analysis\n- [x] Read `CLAUDE.md`\nParagraph text');
    expect(lines[0].kind).toBe('heading');
    expect(lines[1].accessibilityLabel).toBe('Finished: Read CLAUDE.md');
    expect(lines[2].kind).toBe('paragraph');
  });

  test('groups major markdown sections into one accessibility node each', () => {
    const sections = buildClaudeAccessibilitySections(`### Progress

- [x] Read issue and repository \`CLAUDE.md\`
- [ ] Add focused tests

### Current analysis

Ordinary text for current analysis.

### Implementation decisions

- Model choice: \`Gemini 3.1 Flash Lite\`
- Frame strategy: latest-frame streaming
- Runtime input: expected vehicle details

### Recent work

Validated the generated tool.

### Next step

Commit and push.

### Implementation summary

Validation passed and PR prepared.
`);
    expect(sections[0].label).toBe('Progress. Finished: Read issue and repository CLAUDE.md. Ongoing: Add focused tests');
    expect(sections[1].label).toBe(
      'Current analysis. Ordinary text for current analysis.'
    );
    expect(sections[2].label).toBe(
      'Implementation decisions. Model choice: Gemini 3.1 Flash Lite. Frame strategy: latest-frame streaming. Runtime input: expected vehicle details'
    );
    expect(sections[3].label).toBe('Recent work. Validated the generated tool.');
    expect(sections[4].label).toBe('Next step. Commit and push.');
    expect(sections[5].label).toBe('Implementation summary. Validation passed and PR prepared.');
  });
});

describe('IssueChat Claude accessibility tree', () => {
  test('renders one accessible node per Claude major section without whole-message hiding', async () => {
    const claudeBody = `### Progress

- [x] Inspected the existing tool
- [x] Selected Gemini
- [ ] Updating streaming behavior
- [ ] Run validation

### Current analysis

Working through the rendered output.

### Implementation decisions

**Model:** Gemini 3.1 Flash Lite
**Strategy:** single-frame VLM
**Runtime input:** search criteria

### Recent work

Adjusted session-level auto-follow.

### Next step

Run focused validation.

### Implementation summary

No visual markdown changes.
`;
    mockSubmitUpdate.mockResolvedValue({
      status: 'updated',
      issue_number: 88,
      issue_url: 'https://github.test/issues/88',
      video_summary: '',
      pr_number: 88,
    });
    mockFetchClaudeProgress.mockResolvedValue({
      status: 'available',
      issue_number: 88,
      comment_id: 801,
      updated_at: '2026-08-07T12:00:00Z',
      body: claudeBody,
      message: 'Claude comment available.',
      steps: [],
    });

    let renderer: ReactTestRenderer.ReactTestRenderer;
    await act(async () => {
      renderer = renderWithTheme(<IssueChat selectedIssue={{ number: 88, title: 'PR 88' }} />);
    });
    const input = renderer!.root.findByType(TextInput);
    await act(async () => {
      input.props.onChangeText('Update it');
    });
    const sendButton = renderer!.root.findAllByType(TouchableOpacity).find((node) => node.props.accessibilityLabel === 'Send text');
    await act(async () => {
      sendButton!.props.onPress();
    });

    const expectedSectionLabels = buildClaudeAccessibilitySections(claudeBody).map((section) => section.label);
    const sectionNodes = renderer!.root.findAll((node) =>
      node.type === View
      && node.props?.accessible === true
      && typeof node.props?.accessibilityLabel === 'string'
      && expectedSectionLabels.includes(node.props.accessibilityLabel)
    );

    expect(sectionNodes).toHaveLength(6);
    sectionNodes.forEach((node) => {
      expect(node.props.accessibilityElementsHidden).toBeUndefined();
      expect(node.props.importantForAccessibility).toBeUndefined();
    });

    const claudeHeader = renderer!.root.findAll((node) => node.type === Text && node.props?.accessibilityLabel === 'Claude');
    expect(claudeHeader).toHaveLength(1);

    const claudeContainer = renderer!.root.findAll((node) =>
      node.type === View
      && typeof node.props?.testID === 'string'
      && node.props.testID.startsWith('claude-progress-')
    )[0];

    expect(claudeContainer.props.accessibilityElementsHidden).toBeUndefined();
    expect(claudeContainer.props.importantForAccessibility).toBeUndefined();

    const giantClaudeMessageNode = renderer!.root.findAll((node) =>
      node.type === View
      && node.props?.accessible === true
      && typeof node.props?.accessibilityLabel === 'string'
      && node.props.accessibilityLabel.includes('Progress.')
      && node.props.accessibilityLabel.includes('Implementation summary.')
    );
    expect(giantClaudeMessageNode).toHaveLength(0);
  });
});
