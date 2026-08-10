/**
 * ToolSelector Component
 * Displays available tools from the tools folder and allows selection
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  ScrollView,
  AccessibilityInfo,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import Config from './config';
import WebSocketService from './WebSocketService';
import BeepService from './BeepService';
import { useTheme } from './ThemeContext';
import { RuntimeInputDefinition } from './runtimeInput';

interface Tool {
  name: string;
  path: string;
  description?: string;
  code?: string;
  language?: string;
  pr_number?: number;
  pr_title?: string;
  branch_name?: string;
  custom_gpt?: boolean;
  gpt_query?: string;
  system_instruction?: string;
  query_interval?: number;
  source?: string;
  runtime_input?: RuntimeInputDefinition;
}

interface ToolSelectorProps {
  onToolSelect: (tool: Tool) => void;
  selectedTool: Tool | null;
  issueTools?: Tool[];
  productionMode?: boolean;
  selectedIssue?: {number: number; title: string} | null;
}

const TOOL_LOAD_TIMEOUT_MS = 60000;

export default function ToolSelector({ onToolSelect, selectedTool, issueTools = [], productionMode = false, selectedIssue = null }: ToolSelectorProps) {
  const { theme } = useTheme();
  const selectedIssueNumber = selectedIssue?.number;
  const [tools, setTools] = useState<Tool[]>([]);
  const [loading, setLoading] = useState(true);
  const expectingNewToolsRef = useRef(false); // ref avoids spurious effect triggers on state change
  const latestIssueToolsRef = useRef(issueTools);
  latestIssueToolsRef.current = issueTools;
  const toolsAtRequestStartRef = useRef<Tool[] | null>(null);
  const loadTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  console.log('[ToolSelector] Rendered - productionMode:', productionMode, 'issueTools:', issueTools.length);

  // Announce the screen title when entering the tool selector.
  useEffect(() => {
    const timeout = setTimeout(() => {
      AccessibilityInfo.announceForAccessibility('Tools');
    }, 100); // Small delay to ensure component is rendered
    
    return () => clearTimeout(timeout);
  }, []);

  const loadTools = useCallback(async () => {
    if (loadTimeoutRef.current) {
      clearTimeout(loadTimeoutRef.current);
      loadTimeoutRef.current = null;
    }

    // Remember the exact array present when the request began. The parent may
    // still hold tools from the previous screen/PR, and those must not be
    // mistaken for the response to this request.
    toolsAtRequestStartRef.current = latestIssueToolsRef.current;
    setTools([]);
    setLoading(true);
    expectingNewToolsRef.current = true;

    // Production mode: request tools from main branch only
    if (productionMode) {
      console.log('[ToolSelector] Production mode - requesting tools from main branch');
      const success = WebSocketService.requestProductionTools();

      if (!success) {
        console.error('[ToolSelector] Failed to request production tools - WebSocket not connected');
        setLoading(false);
        expectingNewToolsRef.current = false;
        return;
      }

      // Safety timeout in case tools never arrive
      loadTimeoutRef.current = setTimeout(() => {
        if (expectingNewToolsRef.current) {
          console.warn('[ToolSelector] Timeout - no tools received');
          expectingNewToolsRef.current = false;
          setLoading(false);
        }
        loadTimeoutRef.current = null;
      }, TOOL_LOAD_TIMEOUT_MS);

      return;
    }

    // Development mode: the PR request is sent by IssueSelector.
    console.log('[ToolSelector] Development mode - checking if PR is selected');

    if (!selectedIssueNumber) {
      console.log('[ToolSelector] No PR selected in development mode');
      setLoading(false);
      expectingNewToolsRef.current = false;
      return;
    }

    loadTimeoutRef.current = setTimeout(() => {
      if (expectingNewToolsRef.current) {
        console.warn('[ToolSelector] Timeout - no tools received for selected PR');
        expectingNewToolsRef.current = false;
        setLoading(false);
      }
      loadTimeoutRef.current = null;
    }, TOOL_LOAD_TIMEOUT_MS);
  }, [productionMode, selectedIssueNumber]);

  useEffect(() => {
    console.log('[ToolSelector] useEffect triggered - calling loadTools(), productionMode:', productionMode, 'selectedIssue:', selectedIssueNumber);
    loadTools();
    // A PR-tools response can update selectedIssue metadata in the same render
    // that delivers the tools. Reloading for that metadata update would consume
    // the successful response and leave the selector waiting forever. New PR
    // selections clear the parent list and remount this screen through the
    // normal navigation flow, so loading here is scoped to mount/mode changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [productionMode]);

  // Update tools when issueTools changes (both development and production modes).
  // Uses a ref guard (not state) so the effect only fires when issueTools changes,
  // never when loading starts (which would fire with stale old-PR tools).
  useEffect(() => {
    if (
      !expectingNewToolsRef.current ||
      issueTools === toolsAtRequestStartRef.current
    ) return;

    if (loadTimeoutRef.current) {
      clearTimeout(loadTimeoutRef.current);
      loadTimeoutRef.current = null;
    }

    console.log('[ToolSelector] Received tools:', issueTools.length);
    console.log('[ToolSelector] Tool names:', issueTools.map(t => t.name));
    issueTools.forEach(tool => {
      console.log(
        `[Runtime Input] tool=${tool.path || tool.name} enabled=${tool.runtime_input ? 'true' : 'false'}`,
      );
    });

    expectingNewToolsRef.current = false;
    setTools(issueTools);
    setLoading(false);
    console.log('[ToolSelector] Fresh sorted tools arrived, loading complete');
  }, [issueTools]);

  useEffect(() => () => {
    if (loadTimeoutRef.current) {
      clearTimeout(loadTimeoutRef.current);
      loadTimeoutRef.current = null;
    }
  }, []);

  // Loading sound effect for tool fetching
  useEffect(() => {
    let beepTimer: ReturnType<typeof setTimeout> | null = null;
    
    if (loading) {
      console.log('[ToolSelector] Loading tools, will beep after 3 seconds if still loading');
      // Wait 3 seconds before starting beep
      beepTimer = setTimeout(() => {
        console.log('[ToolSelector] 3 seconds elapsed, starting loading sound');
        BeepService.startLoadingSound();
      }, 3000);
    } else {
      console.log('[ToolSelector] Stopping loading sound');
      BeepService.stopLoadingSound();
    }

    // Cleanup on unmount or when loading state changes
    return () => {
      if (beepTimer) {
        clearTimeout(beepTimer);
      }
      BeepService.stopLoadingSound();
    };
  }, [loading]);

  const handleToolPress = (tool: Tool) => {
    onToolSelect(tool);
  };

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.backgroundSecondary }]} edges={[]} accessible={false}>
      <View style={[styles.header, { backgroundColor: theme.background, borderBottomColor: theme.border }]} accessible={false}>
        <Text 
          style={[styles.headerText, { color: theme.text }]}
          accessible={true}
          accessibilityRole="header"
          accessibilityLabel="Select a Tool">
          Select a Tool
        </Text>
        <Text style={[styles.headerSubtext, { color: theme.textSecondary }]} accessible={false}>
          {productionMode 
            ? `Production tools from ${Config.PRODUCTION_BRANCH} branch`
            : 'Choose a tool to run or create a new one'}
        </Text>
        {productionMode && (
          <View style={[styles.productionBadge, { backgroundColor: theme.success }]} accessible={false}>
            <Text style={styles.productionBadgeText} accessible={false}>🚀 Production Mode</Text>
          </View>
        )}
      </View>

      {loading ? (
        <View style={styles.loadingContainer}>
          <Text style={[styles.loadingText, { color: theme.textSecondary }]}>Loading tools...</Text>
        </View>
      ) : tools.length > 0 ? (
        <ScrollView style={styles.toolList}>
          {tools.map((tool) => {
            // Build comprehensive accessibility label including all metadata
            const accessibilityParts = [tool.name];
            if (tool.description) accessibilityParts.push(tool.description);
            if (tool.pr_title) accessibilityParts.push(`Pull request: ${tool.pr_title}`);
            if (tool.branch_name) accessibilityParts.push(`Branch: ${tool.branch_name}`);
            if (tool.language) accessibilityParts.push(`Language: ${tool.language}`);
            if (selectedTool?.name === tool.name) accessibilityParts.push('Selected');
            
            return (
              <TouchableOpacity
                key={`${tool.path}:${tool.name}`}
                style={[
                  styles.toolCard,
                  { 
                    backgroundColor: theme.card, 
                    borderColor: theme.border 
                  },
                  selectedTool?.name === tool.name && { 
                    backgroundColor: theme.backgroundSecondary, 
                    borderColor: theme.primary 
                  }
                ]}
                onPress={() => handleToolPress(tool)}
                accessibilityRole="button"
                accessibilityLabel={accessibilityParts.join('. ')}
                accessibilityHint="Double tap to select this tool"
                accessibilityState={{ selected: selectedTool?.name === tool.name }}>
                <View style={styles.toolHeader}>
                  <Text 
                    style={[
                      styles.toolName,
                      { color: theme.text },
                      selectedTool?.name === tool.name && { color: theme.primary }
                    ]}
                    accessible={false}>
                    {tool.name}
                  </Text>
                  {selectedTool?.name === tool.name && (
                    <View style={[styles.selectedBadge, { backgroundColor: theme.success }]} accessible={false}>
                      <Text style={styles.selectedBadgeText} accessible={false}>✓</Text>
                    </View>
                  )}
                </View>
                {tool.description && (
                  <Text style={[styles.toolDescription, { color: theme.textSecondary }]} accessible={false}>{tool.description}</Text>
                )}
                {tool.pr_title && (
                  <Text style={[styles.toolMeta, { color: theme.textTertiary }]} accessible={false}>PR: {tool.pr_title}</Text>
                )}
                {tool.branch_name && (
                  <Text style={[styles.toolMeta, { color: theme.textTertiary }]} accessible={false}>Branch: {tool.branch_name}</Text>
                )}
                {tool.language && (
                  <Text style={[styles.toolMeta, { color: theme.textTertiary }]} accessible={false}>Language: {tool.language}</Text>
                )}
              </TouchableOpacity>
            );
          })}
        </ScrollView>
      ) : (
        <View style={styles.emptyContainer}>
          {!productionMode && !selectedIssue ? (
            <>
              <Text style={[styles.emptyText, { color: theme.textTertiary }]}>No Branch Selected</Text>
              <Text style={[styles.emptySubtext, { color: theme.textTertiary }]}>
                Go to the PRs tab to select a pull request
              </Text>
            </>
          ) : (
            <>
              <Text style={[styles.emptyText, { color: theme.textTertiary }]}>No tools found</Text>
              <Text style={[styles.emptySubtext, { color: theme.textTertiary }]}>
                Request a tool to be created in your issue description
              </Text>
            </>
          )}
        </View>
      )}

      <View style={[styles.footer, { backgroundColor: theme.background, borderTopColor: theme.border }]}>
        <Text style={[styles.footerText, { color: theme.textTertiary }]}>
          Request new tools in your issue description
        </Text>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    paddingHorizontal: 16,
    paddingVertical: 16,
    borderBottomWidth: 1,
  },
  headerText: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  headerSubtext: {
    fontSize: 14,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    fontSize: 16,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 32,
  },
  emptyText: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 8,
  },
  emptySubtext: {
    fontSize: 14,
    textAlign: 'center',
  },
  toolList: {
    flex: 1,
    paddingHorizontal: 16,
    paddingTop: 16,
  },
  toolCard: {
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderWidth: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 2,
  },
  toolHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  toolName: {
    fontSize: 18,
    fontWeight: '600',
    flex: 1,
  },
  selectedBadge: {
    borderRadius: 12,
    width: 24,
    height: 24,
    alignItems: 'center',
    justifyContent: 'center',
  },
  selectedBadgeText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  toolDescription: {
    fontSize: 14,
    lineHeight: 20,
  },
  toolMeta: {
    fontSize: 11,
    marginTop: 4,
    fontStyle: 'italic',
  },
  productionBadge: {
    marginTop: 8,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
    alignSelf: 'flex-start',
  },
  productionBadgeText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },
  footer: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderTopWidth: 1,
  },
  footerText: {
    fontSize: 12,
    textAlign: 'center',
    fontStyle: 'italic',
  },
});
