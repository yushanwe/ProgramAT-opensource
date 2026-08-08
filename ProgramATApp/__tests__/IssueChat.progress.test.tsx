import React from 'react';
import ReactTestRenderer, { act } from 'react-test-renderer';
import { AccessibilityInfo, ScrollView, Text, TextInput, TouchableOpacity, View } from 'react-native';
import IssueChat from '../IssueChat';
import { ThemeProvider } from '../ThemeContext';
import {
  buildClaudeAccessibilitySections,
  getChangedClaudeAnnouncement,
  parseClaudeMessage,
  parseClaudeRenderLines,
  sanitizeClaudeCommentBody,
} from '../claudeCommentSanitizer';

jest.useFakeTimers();

const mockScrollToEnd = jest.fn();
const mockFocus = jest.fn();

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

const mockPlayLoadingSound = jest.fn(() => Promise.resolve());
const mockStopLoadingSound = jest.fn();

jest.mock('../BeepService', () => ({
  __esModule: true,
  default: {
    playBeep: jest.fn(),
    playLoadingSound: (...args: any[]) => mockPlayLoadingSound(...args),
    stopLoadingSound: (...args: any[]) => mockStopLoadingSound(...args),
  },
}));

jest.mock('../VideoRecorderModal', () => 'VideoRecorderModal');
jest.mock('../RayBanRecorderModal', () => 'RayBanRecorderModal');

const mockSubmitUpdate = jest.fn();
const mockSubmitCreation = jest.fn();
const mockNextBrainstormQuestion = jest.fn();
const mockAskBrainstormAgent = jest.fn();
const mockFetchClaudeProgress = jest.fn();

jest.mock('../IssueSubmissionService', () => ({
  submitUpdate: (...args: any[]) => mockSubmitUpdate(...args),
  submitCreation: (...args: any[]) => mockSubmitCreation(...args),
  nextBrainstormQuestion: (...args: any[]) => mockNextBrainstormQuestion(...args),
  askBrainstormAgent: (...args: any[]) => mockAskBrainstormAgent(...args),
  fetchClaudeProgress: (...args: any[]) => mockFetchClaudeProgress(...args),
}));

const announceSpy = jest.spyOn(AccessibilityInfo, 'announceForAccessibility').mockImplementation(jest.fn());
const setTimeoutSpy = jest.spyOn(global, 'setTimeout');

describe('IssueChat Claude progress', () => {
  const mountedRenderers: ReactTestRenderer.ReactTestRenderer[] = [];
  const findClaudeCards = (renderer: ReactTestRenderer.ReactTestRenderer) => renderer.root.findAll((node) =>
    node.type === View
    && typeof node.props?.testID === 'string'
    && node.props.testID.startsWith('claude-progress-')
  );

  const renderWithTheme = async (element: React.ReactElement) => {
    let renderer!: ReactTestRenderer.ReactTestRenderer;
    await act(async () => {
      renderer = ReactTestRenderer.create(<ThemeProvider>{element}</ThemeProvider>, {
        createNodeMock: (node) => {
          if (node.type === TextInput) {
            return { focus: mockFocus };
          }
          if (node.type === ScrollView) {
            return { scrollToEnd: mockScrollToEnd };
          }
          if (node.type === View) {
            return { _nativeTag: 101 };
          }
          return {};
        },
      });
    });
    mountedRenderers.push(renderer);
    return renderer;
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockSubmitUpdate.mockReset();
    mockSubmitCreation.mockReset();
    mockNextBrainstormQuestion.mockReset();
    mockAskBrainstormAgent.mockReset();
    mockFetchClaudeProgress.mockReset();
    (AccessibilityInfo as any).setAccessibilityFocus = jest.fn();
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

  async function sendUpdateAndStartPolling(progressResponses: any[]) {
    mockSubmitUpdate.mockResolvedValue({
      status: 'updated',
      issue_number: 55,
      issue_url: 'https://github.test/issues/55',
      video_summary: '',
      pr_number: 55,
    });
    progressResponses.forEach((response) => mockFetchClaudeProgress.mockResolvedValueOnce(response));

    const renderer = await renderWithTheme(<IssueChat selectedIssue={{ number: 55, title: 'PR 55' }} />);
    const input = renderer.root.findByType(TextInput);

    await act(async () => {
      input.props.onChangeText('Update it');
    });

    const sendButton = renderer.root.findAllByType(TouchableOpacity).find((node) => node.props.accessibilityLabel === 'Send text');
    await act(async () => {
      sendButton!.props.onPress();
      await Promise.resolve();
    });

    return renderer;
  }

  test('polls every 6 seconds and passes update boundary metadata', async () => {
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
        steps: [],
        updated_at: null,
      })
      .mockResolvedValueOnce({
        status: 'available',
        issue_number: 242,
        comment_id: 9002,
        updated_at: '2026-08-07T12:00:10Z',
        body: '25% Inspecting the request',
        status_line: { percent: 25, label: 'Inspecting the request', text: '25% Inspecting the request' },
        steps: [],
      });

    const renderer = await renderWithTheme(<IssueChat selectedIssue={{ number: 142, title: 'PR 142' }} />);
    const input = renderer.root.findByType(TextInput);
    await act(async () => {
      input.props.onChangeText('Please update the tool');
    });
    const sendButton = renderer.root.findAllByType(TouchableOpacity).find((node) => node.props.accessibilityLabel === 'Send text');
    await act(async () => {
      sendButton!.props.onPress();
      await Promise.resolve();
    });

    expect(mockFetchClaudeProgress).toHaveBeenCalledWith({
      mode: 'update',
      prNumber: 242,
      commentId: null,
      afterCommentId: 9001,
      afterTimestamp: '2026-08-07T12:00:00Z',
    });

    await act(async () => {
      jest.advanceTimersByTime(5999);
      await Promise.resolve();
    });
    expect(mockFetchClaudeProgress).toHaveBeenCalledTimes(1);

    await act(async () => {
      jest.advanceTimersByTime(1);
      await Promise.resolve();
    });
    expect(mockFetchClaudeProgress).toHaveBeenCalledTimes(2);
  });

  test('appends the initial Claude waiting message before the first real Claude update', async () => {
    const renderer = await sendUpdateAndStartPolling([
      {
        status: 'waiting_for_comment',
        issue_number: 55,
        comment_id: null,
        body: '',
        steps: [],
        updated_at: null,
      },
      {
        status: 'available',
        issue_number: 55,
        comment_id: 601,
        updated_at: '2026-08-08T10:00:10Z',
        body: '25% Inspecting existing implementation\n\nLooking at the polling flow now.\n\n### Progress\n- [ ] Validate behavior',
        status_line: { percent: 25, label: 'Inspecting existing implementation', text: '25% Inspecting existing implementation' },
        steps: [],
      },
    ]);

    expect(JSON.stringify(renderer.toJSON())).toContain("Claude hasn't posted any comments yet.");

    await act(async () => {
      jest.advanceTimersByTime(6000);
      await Promise.resolve();
    });

    const claudeCards = findClaudeCards(renderer);
    expect(claudeCards).toHaveLength(2);
    expect(JSON.stringify(renderer.toJSON())).toContain('25% Inspecting existing implementation');
  });

  test('identical polls do not append duplicate Claude messages', async () => {
    const response = {
      status: 'available',
      issue_number: 55,
      comment_id: 601,
      updated_at: '2026-08-07T10:00:00Z',
      body: '25% Inspecting existing implementation\n\nLooking at the polling flow now.',
      status_line: { percent: 25, label: 'Inspecting existing implementation', text: '25% Inspecting existing implementation' },
      steps: [],
    };
    const renderer = await sendUpdateAndStartPolling([response, { ...response, updated_at: '2026-08-07T10:00:10Z' }]);

    expect(findClaudeCards(renderer)).toHaveLength(1);

    await act(async () => {
      jest.advanceTimersByTime(6000);
      await Promise.resolve();
    });

    expect(findClaudeCards(renderer)).toHaveLength(1);
  });

  test('changed Claude body appends a new message and keeps history', async () => {
    const renderer = await sendUpdateAndStartPolling([
      {
        status: 'available',
        issue_number: 55,
        comment_id: 601,
        updated_at: '2026-08-07T10:00:00Z',
        body: '<!-- USER_SUMMARY_START -->\n25% Inspecting existing implementation\n\nLooking at the polling flow now.\n<!-- USER_SUMMARY_END -->\n\n<!-- EXPERT_DETAIL_START -->\n### Current analysis\nComparing the polling flow now.\n<!-- EXPERT_DETAIL_END -->',
        summary_text: '25% Inspecting existing implementation\n\nLooking at the polling flow now.',
        expert_markdown: '### Current analysis\nComparing the polling flow now.',
        status_line: { percent: 25, label: 'Inspecting existing implementation', text: '25% Inspecting existing implementation' },
        steps: [],
      },
      {
        status: 'available',
        issue_number: 55,
        comment_id: 601,
        updated_at: '2026-08-07T10:00:10Z',
        body: '<!-- USER_SUMMARY_START -->\n50% Implementing changes\n\nThe append-only update path is in place.\n<!-- USER_SUMMARY_END -->\n\n<!-- EXPERT_DETAIL_START -->\n### Recent work\nAdded the split summary and expert rendering path.\n<!-- EXPERT_DETAIL_END -->',
        summary_text: '50% Implementing changes\n\nThe append-only update path is in place.',
        expert_markdown: '### Recent work\nAdded the split summary and expert rendering path.',
        status_line: { percent: 50, label: 'Implementing changes', text: '50% Implementing changes' },
        steps: [],
      },
    ]);

    await act(async () => {
      jest.advanceTimersByTime(6000);
      await Promise.resolve();
    });

    const claudeCards = findClaudeCards(renderer);
    expect(claudeCards).toHaveLength(2);
    expect(JSON.stringify(renderer.toJSON())).toContain('25% Inspecting existing implementation');
    expect(JSON.stringify(renderer.toJSON())).toContain('50% Implementing changes');
    expect(JSON.stringify(renderer.toJSON())).toContain('Looking at the polling flow now.');
    expect(JSON.stringify(renderer.toJSON())).toContain('The append-only update path is in place.');
    expect(JSON.stringify(renderer.toJSON())).not.toContain('Comparing the polling flow now.');
    expect(JSON.stringify(renderer.toJSON())).not.toContain('Added the split summary and expert rendering path.');
  });

  test('expert details are collapsed by default and expand per Claude message', async () => {
    const renderer = await sendUpdateAndStartPolling([
      {
        status: 'available',
        issue_number: 55,
        comment_id: 601,
        updated_at: '2026-08-07T10:00:00Z',
        body: '<!-- USER_SUMMARY_START -->\n25% Inspecting existing implementation\n\nLooking at the polling flow now.\n<!-- USER_SUMMARY_END -->\n\n<!-- EXPERT_DETAIL_START -->\n### Current analysis\nComparing the polling flow now.\n<!-- EXPERT_DETAIL_END -->',
        summary_text: '25% Inspecting existing implementation\n\nLooking at the polling flow now.',
        expert_markdown: '### Current analysis\nComparing the polling flow now.',
        status_line: { percent: 25, label: 'Inspecting existing implementation', text: '25% Inspecting existing implementation' },
        steps: [],
      },
      {
        status: 'available',
        issue_number: 55,
        comment_id: 601,
        updated_at: '2026-08-07T10:00:10Z',
        body: '<!-- USER_SUMMARY_START -->\n50% Implementing changes\n\nThe append-only update path is in place.\n<!-- USER_SUMMARY_END -->\n\n<!-- EXPERT_DETAIL_START -->\n### Recent work\nAdded the split summary and expert rendering path.\n<!-- EXPERT_DETAIL_END -->',
        summary_text: '50% Implementing changes\n\nThe append-only update path is in place.',
        expert_markdown: '### Recent work\nAdded the split summary and expert rendering path.',
        status_line: { percent: 50, label: 'Implementing changes', text: '50% Implementing changes' },
        steps: [],
      },
    ]);

    await act(async () => {
      jest.advanceTimersByTime(6000);
      await Promise.resolve();
    });

    expect(JSON.stringify(renderer.toJSON())).not.toContain('Comparing the polling flow now.');
    expect(JSON.stringify(renderer.toJSON())).not.toContain('Added the split summary and expert rendering path.');

    const toggleButtons = renderer.root.findAllByType(TouchableOpacity)
      .filter((node) => node.props.accessibilityRole === 'button' && /expert details/i.test(node.props.accessibilityLabel || ''));
    expect(toggleButtons).toHaveLength(2);
    expect(toggleButtons[0].props.accessibilityLabel).toBe('Expand expert details');
    expect(toggleButtons[0].props.accessibilityState).toEqual({ expanded: false });

    await act(async () => {
      toggleButtons[0].props.onPress();
    });

    const expandedButtons = renderer.root.findAllByType(TouchableOpacity)
      .filter((node) => node.props.accessibilityRole === 'button' && /expert details/i.test(node.props.accessibilityLabel || ''));
    expect(expandedButtons[0].props.accessibilityLabel).toBe('Collapse expert details');
    expect(JSON.stringify(renderer.toJSON())).toContain('Comparing the polling flow now.');
    expect(JSON.stringify(renderer.toJSON())).not.toContain('Added the split summary and expert rendering path.');
  });

  test('each new Claude message gets one scroll and focus transition, unchanged polls do not', async () => {
    const renderer = await sendUpdateAndStartPolling([
      {
        status: 'available',
        issue_number: 55,
        comment_id: 601,
        updated_at: '2026-08-07T10:00:00Z',
        body: '25% Inspecting existing implementation\n\nLooking at the polling flow now.',
        status_line: { percent: 25, label: 'Inspecting existing implementation', text: '25% Inspecting existing implementation' },
        steps: [],
      },
      {
        status: 'available',
        issue_number: 55,
        comment_id: 601,
        updated_at: '2026-08-07T10:00:10Z',
        body: '25% Inspecting existing implementation\n\nLooking at the polling flow now.',
        status_line: { percent: 25, label: 'Inspecting existing implementation', text: '25% Inspecting existing implementation' },
        steps: [],
      },
      {
        status: 'available',
        issue_number: 55,
        comment_id: 601,
        updated_at: '2026-08-07T10:00:20Z',
        body: '75% Validating behavior\n\nFocused checks are running now.',
        status_line: { percent: 75, label: 'Validating behavior', text: '75% Validating behavior' },
        steps: [],
      },
    ]);

    const count100msTimeouts = () => setTimeoutSpy.mock.calls.filter((call) => call[1] === 100).length;
    expect(count100msTimeouts()).toBeGreaterThanOrEqual(1);

    await act(async () => {
      jest.advanceTimersByTime(6000);
      await Promise.resolve();
    });
    const timeoutCountAfterUnchangedPoll = count100msTimeouts();

    await act(async () => {
      jest.advanceTimersByTime(6000);
      await Promise.resolve();
    });
    expect(count100msTimeouts()).toBeGreaterThan(timeoutCountAfterUnchangedPoll);

    const messageScrollView = renderer.root.findAllByType(ScrollView).find((node) => node.props?.keyboardShouldPersistTaps === 'handled');
    expect(messageScrollView?.props.scrollEnabled).not.toBe(false);
  });

  test('duplicate final summary is not appended again', async () => {
    const renderer = await sendUpdateAndStartPolling([
      {
        status: 'available',
        issue_number: 55,
        comment_id: 601,
        updated_at: '2026-08-07T10:00:00Z',
        body: '100% Complete\n\n### Implementation summary\nValidation passed and PR prepared.',
        status_line: { percent: 100, label: 'Complete', text: '100% Complete' },
        steps: [],
      },
      {
        status: 'available',
        issue_number: 55,
        comment_id: 601,
        updated_at: '2026-08-07T10:00:15Z',
        body: '100% Complete\n\n### Implementation summary\nValidation passed and PR prepared.',
        status_line: { percent: 100, label: 'Complete', text: '100% Complete' },
        steps: [],
      },
    ]);

    await act(async () => {
      jest.advanceTimersByTime(6000);
      await Promise.resolve();
    });

    expect(findClaudeCards(renderer)).toHaveLength(1);
  });

  test('waiting audio remains separate from polling and uses a 6 second gap', async () => {
    mockPlayLoadingSound.mockImplementationOnce(() => new Promise<void>((resolve) => setTimeout(resolve, 1)));
    await sendUpdateAndStartPolling([
      {
        status: 'available',
        issue_number: 55,
        comment_id: 601,
        updated_at: '2026-08-07T10:00:00Z',
        body: '50% Implementing changes\n\nStill working.',
        status_line: { percent: 50, label: 'Implementing changes', text: '50% Implementing changes' },
        steps: [],
      },
    ]);

    expect(mockPlayLoadingSound).toHaveBeenCalledTimes(1);

    await act(async () => {
      jest.advanceTimersByTime(1);
      await Promise.resolve();
    });
    await act(async () => {
      jest.advanceTimersByTime(6000);
      await Promise.resolve();
    });
    expect(mockPlayLoadingSound).toHaveBeenCalledTimes(2);
  });
});

describe('Claude progress sanitizing and accessibility', () => {
  test('preserves status lines and removes unsupported media', () => {
    const input = '<!-- USER_SUMMARY_START -->\n25% Inspecting implementation\n<!-- USER_SUMMARY_END -->\n<img src="https://example.com/spinner.gif" />\nWorking now.';
    expect(sanitizeClaudeCommentBody(input)).toBe('25% Inspecting implementation\n\nWorking now.');
  });

  test('extracts changed section announcements', () => {
    const previous = '50% Implementing changes\n\n### Recent work\nUpdated polling.';
    const next = '75% Validating behavior\n\n### Recent work\nRan focused validation.';
    expect(getChangedClaudeAnnouncement(previous, next)).toBe('Recent Work: Ran focused validation.');
  });

  test('renders percentage status lines as normal readable lines', () => {
    const lines = parseClaudeRenderLines('25% Inspecting existing implementation\nNext paragraph');
    expect(lines[0].text).toBe('25% Inspecting existing implementation');
    expect(lines[1].kind).toBe('paragraph');
  });

  test('parses Claude messages at the Progress heading boundary', () => {
    const parsed = parseClaudeMessage(`25% Processing CLAUDE.md

I have identified the runtime path.
I'm checking the streaming handler.
Next I will validate both modes.

### Progress
- [x] Read requirements

### Current analysis
Comparing the old and new paths.`);
    expect(parsed.summaryMarkdown).toBe(`25% Processing CLAUDE.md

I have identified the runtime path.
I'm checking the streaming handler.
Next I will validate both modes.`);
    expect(parsed.expertMarkdown).toContain('### Progress');
    expect(parsed.expertMarkdown).toContain('### Current analysis');
  });

  test('groups the status line and prose together for VoiceOver sections', () => {
    const sections = buildClaudeAccessibilitySections(`25% Inspecting existing implementation

Looking at the polling flow now.

### Current analysis

Comparing old and new message identity.

### Implementation summary

Validation passed.
`);
    expect(sections[0].label).toBe('25% Inspecting existing implementation. Looking at the polling flow now.');
    expect(sections[1].label).toBe('Current analysis. Comparing old and new message identity.');
    expect(sections[2].label).toBe('Implementation summary. Validation passed.');
  });
});
