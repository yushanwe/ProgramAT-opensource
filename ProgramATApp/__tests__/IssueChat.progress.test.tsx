import React from 'react';
import ReactTestRenderer, { act } from 'react-test-renderer';
import { AccessibilityInfo, TextInput, TouchableOpacity } from 'react-native';
import IssueChat from '../IssueChat';
import PRsAndText from '../PRsAndText';
import { ThemeProvider } from '../ThemeContext';

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

jest.mock('../BeepService', () => ({
  __esModule: true,
  default: { playBeep: jest.fn() },
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

function renderWithTheme(element: React.ReactElement) {
  return ReactTestRenderer.create(<ThemeProvider>{element}</ThemeProvider>);
}

describe('IssueChat progress', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockSubmitCreation.mockReset();
    mockSubmitUpdate.mockReset();
    mockNextBrainstormQuestion.mockReset();
    mockFetchClaudeProgress.mockReset();
  });

  test('polls for Claude progress and cleans up on unmount', async () => {
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

    const callCountBeforeUnmount = mockFetchClaudeProgress.mock.calls.length;
    expect(callCountBeforeUnmount).toBeGreaterThanOrEqual(1);

    await act(async () => {
      renderer!.unmount();
    });
    await act(async () => {
      jest.advanceTimersByTime(3000);
    });

    expect(mockFetchClaudeProgress).toHaveBeenCalledTimes(callCountBeforeUnmount);
  });

  test('renders separate VoiceOver labels for progress steps and announces active step once', async () => {
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
        body: '### Progress\n\n- [x] Read issue and repository CLAUDE.md\n- [ ] Implement tool.py',
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
        body: '### Progress\n\n- [x] Read issue and repository CLAUDE.md\n- [ ] Implement tool.py',
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

    const claudeLabel = renderer!.root.findAll((node) => node.props?.accessibilityLabel === 'Claude comment available. ### Progress\n\n- [x] Read issue and repository CLAUDE.md\n- [ ] Implement tool.py');
    expect(claudeLabel.length).toBeGreaterThan(0);
    expect(announceSpy).toHaveBeenCalledWith('In progress: Building the tool');

    await act(async () => {
      jest.advanceTimersByTime(2500);
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

  test('updates one in-flow Claude message without duplicates', async () => {
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
        status: 'completed',
        issue_number: 9,
        comment_id: 201,
        updated_at: '2026-08-05T12:00:05Z',
        body: 'Final Claude update',
        message: 'Claude finished all checklist steps.',
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
      jest.advanceTimersByTime(2500);
      await Promise.resolve();
    });

    const claudeTexts = renderer!.root.findAll((node) => node.type === 'Text' && node.props?.children === 'Claude');
    expect(claudeTexts).toHaveLength(1);
    expect(mockFetchClaudeProgress.mock.calls.length).toBeGreaterThanOrEqual(2);
  });
});
