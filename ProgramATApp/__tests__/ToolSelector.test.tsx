import React from 'react';
import ReactTestRenderer from 'react-test-renderer';
import ToolSelector from '../ToolSelector';

jest.mock('../ThemeContext', () => ({
  useTheme: () => ({
    theme: {
      background: '#000',
      backgroundSecondary: '#111',
      border: '#222',
      card: '#333',
      primary: '#444',
      success: '#0f0',
      text: '#fff',
      textSecondary: '#ccc',
      textTertiary: '#999',
    },
  }),
}));

jest.mock('../config', () => ({
  __esModule: true,
  default: {
    PRODUCTION_BRANCH: 'main',
  },
}));

const mockRequestProductionTools = jest.fn(() => true);

jest.mock('../WebSocketService', () => ({
  __esModule: true,
  default: {
    requestProductionTools: () => mockRequestProductionTools(),
  },
}));

jest.mock('../BeepService', () => ({
  __esModule: true,
  default: {
    startLoadingSound: jest.fn(),
    stopLoadingSound: jest.fn(),
  },
}));

jest.mock('react-native-safe-area-context', () => ({
  SafeAreaView: 'SafeAreaView',
}));

describe('ToolSelector', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    mockRequestProductionTools.mockClear();
  });

  afterEach(() => {
    jest.clearAllTimers();
    jest.useRealTimers();
  });

  test('clears stale tools while waiting for a new selection to load', async () => {
    const onToolSelect = jest.fn();
    const oldTools = [
      { name: 'old_tool', path: 'tools/old_tool.py', description: 'Old tool' },
    ];
    const newTools = [
      { name: 'new_tool', path: 'tools/new_tool.py', description: 'New tool' },
    ];

    let tree: ReactTestRenderer.ReactTestRenderer;
    await ReactTestRenderer.act(() => {
      tree = ReactTestRenderer.create(
        <ToolSelector
          onToolSelect={onToolSelect}
          selectedTool={null}
          issueTools={oldTools}
          selectedIssue={{ number: 1, title: 'Issue 1' }}
        />,
      );
    });

    expect(tree!.root.findAll(node => node.props?.children === 'old_tool')).toHaveLength(0);
    expect(
      tree!.root.findAllByProps({ children: 'Loading tools...' }).length,
    ).toBeGreaterThan(0);

    await ReactTestRenderer.act(() => {
      tree!.update(
        <ToolSelector
          onToolSelect={onToolSelect}
          selectedTool={null}
          issueTools={oldTools}
          selectedIssue={{ number: 2, title: 'Issue 2' }}
        />,
      );
    });

    expect(tree!.root.findAll(node => node.props?.children === 'old_tool')).toHaveLength(0);
    expect(
      tree!.root.findAllByProps({ children: 'Loading tools...' }).length,
    ).toBeGreaterThan(0);

    await ReactTestRenderer.act(() => {
      tree!.update(
        <ToolSelector
          onToolSelect={onToolSelect}
          selectedTool={null}
          issueTools={newTools}
          selectedIssue={{ number: 2, title: 'Issue 2' }}
        />,
      );
    });

    expect(tree!.root.findAll(node => node.props?.children === 'old_tool')).toHaveLength(0);
    expect(
      tree!.root.findAll(node => node.props?.children === 'new_tool').length,
    ).toBeGreaterThan(0);
    expect(
      tree!.root.findAllByProps({ children: 'Loading tools...' }),
    ).toHaveLength(0);

    await ReactTestRenderer.act(() => {
      tree!.unmount();
    });
  });

  test('finishes loading when the fresh response contains no tools', async () => {
    let tree: ReactTestRenderer.ReactTestRenderer;
    const staleTools = [{ name: 'stale_tool', path: 'tools/stale_tool.py' }];

    await ReactTestRenderer.act(() => {
      tree = ReactTestRenderer.create(
        <ToolSelector
          onToolSelect={jest.fn()}
          selectedTool={null}
          issueTools={staleTools}
          selectedIssue={{ number: 1, title: 'Issue 1' }}
        />,
      );
    });

    await ReactTestRenderer.act(() => {
      tree!.update(
        <ToolSelector
          onToolSelect={jest.fn()}
          selectedTool={null}
          issueTools={[]}
          selectedIssue={{ number: 1, title: 'Issue 1' }}
        />,
      );
    });

    expect(tree!.root.findAllByProps({ children: 'Loading tools...' })).toHaveLength(0);
    expect(tree!.root.findAll(node => node.props?.children === 'stale_tool')).toHaveLength(0);

    await ReactTestRenderer.act(() => {
      tree!.unmount();
    });
  });
});
