import React from 'react';
import ReactTestRenderer from 'react-test-renderer';
import ToolRunner from '../ToolRunner';

jest.mock('../CameraView', () => 'CameraView');
jest.mock('../WebSocketService', () => ({
  __esModule: true,
  default: {
    getActiveSocket: jest.fn(() => null),
    addMessageListener: jest.fn(),
    removeMessageListener: jest.fn(),
  },
}));
jest.mock('../AudioOutputService', () => ({
  __esModule: true,
  default: {
    play: jest.fn(),
  },
}));
jest.mock('../BeepService', () => ({
  __esModule: true,
  default: {
    stopLoadingSound: jest.fn(),
    playBeep: jest.fn(() => Promise.resolve()),
  },
}));
jest.mock('../TextToSpeechService', () => ({
  __esModule: true,
  default: {
    stop: jest.fn(),
  },
}));
jest.mock('@react-native-voice/voice', () => ({
  __esModule: true,
  default: {
    cancel: jest.fn(() => Promise.resolve()),
    destroy: jest.fn(() => Promise.resolve()),
    removeAllListeners: jest.fn(),
    start: jest.fn(() => Promise.resolve()),
    stop: jest.fn(() => Promise.resolve()),
  },
}));
jest.mock('@react-native-async-storage/async-storage', () => ({
  __esModule: true,
  default: {
    getItem: jest.fn(() => Promise.resolve(null)),
    setItem: jest.fn(() => Promise.resolve()),
    removeItem: jest.fn(() => Promise.resolve()),
  },
}));

describe('ToolRunner runtime input rendering', () => {
  let setTimeoutSpy: jest.SpyInstance;

  beforeEach(() => {
    setTimeoutSpy = jest
      .spyOn(global, 'setTimeout')
      .mockImplementation(() => 0 as unknown as ReturnType<typeof setTimeout>);
  });

  afterEach(() => {
    setTimeoutSpy.mockRestore();
  });

  test('renders runtime input controls when selected tool includes runtime_input', async () => {
    let tree: ReactTestRenderer.ReactTestRenderer;
    await ReactTestRenderer.act(() => {
      tree = ReactTestRenderer.create(
        <ToolRunner
          selectedTool={{
            name: 'object_recognition',
            path: 'tools/object_recognition.py',
            language: 'python',
            runtime_input: {
              key: 'target_object',
              label: 'Object to find',
              placeholder: 'Enter an object, such as a water cup',
              prompt_instruction: 'Focus on {value}.',
            },
          }}
          showBackButton={true}
          onBack={jest.fn()}
        />,
      );
    });

    const inputs = tree!.root.findAll(
      node => node.props?.placeholder === 'Enter an object, such as a water cup',
    );
    expect(tree!.root.findAllByType('CameraView').length).toBeGreaterThan(0);
    const lowerScrollWrappers = tree!.root.findAll(
      node => node.props?.keyboardShouldPersistTaps === 'always',
    );
    expect(lowerScrollWrappers.length).toBeGreaterThan(0);
    expect(lowerScrollWrappers[0].props.accessible).toBe(false);
    expect(lowerScrollWrappers[0].props.keyboardDismissMode).toBe('none');
    expect(inputs.length).toBeGreaterThan(0);
    expect(
      tree!.root.findAll(node => node.props?.accessibilityLabel === 'Enter').length,
    ).toBeGreaterThan(0);
    expect(
      tree!.root.findAll(node => node.props?.accessibilityLabel === 'Clear').length,
    ).toBeGreaterThan(0);
    expect(tree!.root.findAllByProps({children: 'Active target: none'}).length).toBeGreaterThan(0);
    expect(
      tree!.root.findAll(node => node.props?.children === '← Back to Tools').length,
    ).toBeGreaterThan(0);
    const headerTitles = tree!.root.findAll(
      node =>
        node.props?.accessibilityLabel === 'Tool: object_recognition' &&
        node.props?.ellipsizeMode === 'tail',
    );
    expect(headerTitles.length).toBeGreaterThan(0);
    expect(headerTitles[0].props.numberOfLines).toBe(1);
    expect(headerTitles[0].props.ellipsizeMode).toBe('tail');
    await ReactTestRenderer.act(() => {
      tree!.unmount();
    });
  });

  test('does not render runtime input controls when selected tool lacks runtime_input', async () => {
    let tree: ReactTestRenderer.ReactTestRenderer;
    await ReactTestRenderer.act(() => {
      tree = ReactTestRenderer.create(
        <ToolRunner
          selectedTool={{
            name: 'scene_description',
            path: 'tools/scene_description.py',
            language: 'python',
          }}
        />,
      );
    });

    const inputs = tree!.root.findAll(
      node => node.props?.placeholder === 'Enter an object, such as a water cup',
    );
    expect(tree!.root.findAllByType('CameraView').length).toBeGreaterThan(0);
    const lowerScrollWrappers = tree!.root.findAll(
      node => node.props?.keyboardShouldPersistTaps === 'always',
    );
    expect(lowerScrollWrappers.length).toBeGreaterThan(0);
    expect(lowerScrollWrappers[0].props.accessible).toBe(false);
    expect(lowerScrollWrappers[0].props.keyboardDismissMode).toBe('none');
    expect(inputs).toHaveLength(0);
    expect(tree!.root.findAllByProps({children: 'Active target: none'})).toHaveLength(0);
    expect(
      tree!.root.findAll(node => node.props?.children === 'scene_description').length,
    ).toBe(0);
    await ReactTestRenderer.act(() => {
      tree!.unmount();
    });
  });
});
