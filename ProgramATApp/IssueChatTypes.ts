/**
 * IssueChatTypes
 * Message/turn data model for the chat-style issue creation/update UI.
 */

export type RetryDescriptor =
  | { op: 'create-text'; text: string }
  | { op: 'create-video'; text: string; videoUri: string }
  | { op: 'update'; text: string; videoUri: string | null; issueNumber: number }
  | { op: 'update-answer'; text: string; token: string; issueNumber: number }
  | { op: 'ideation-answer'; text: string; token: string }
  | { op: 'next-question'; token: string }
  | { op: 'start-building'; token: string; mode?: 'create' | 'update'; issueNumber?: number }
  | { op: 'ask-agent'; token: string; question: string };

export type IssueChatItem =
  | { kind: 'user-text'; id: string; ts: Date; text: string }
  | { kind: 'user-video'; id: string; ts: Date; videoUri: string; caption: string }
  | { kind: 'user-choice'; id: string; ts: Date; choice: 'keep_brainstorming' | 'start_building' | 'ask_agent'; label: string }
  | { kind: 'assistant-question'; id: string; ts: Date; question: string; token: string }
  | { kind: 'assistant-choice-prompt'; id: string; ts: Date; text: string; token: string; resolved: boolean }
  | { kind: 'assistant-clarification-answer'; id: string; ts: Date; question: string; answer: string; token: string }
  | { kind: 'assistant-created'; id: string; ts: Date; issueNumber: number; issueUrl: string; videoSummarySkipped: boolean }
  | { kind: 'assistant-updated'; id: string; ts: Date; issueNumber: number; issueUrl: string; videoSummarySkipped: boolean }
  | { kind: 'assistant-claude-progress'; id: string; ts: Date; status: string; body: string; commentId?: number | null; updatedAt?: string | null; message?: string }
  | { kind: 'assistant-error'; id: string; ts: Date; text: string; retry?: RetryDescriptor };

export type CreationResponse =
  | { status: 'created'; issue_number: number; issue_url: string; video_summary: string; pr_number?: number | null; comment_id?: number | null }
  | { status: 'ideation'; question: string; token: string; summary?: string; integration_note?: string }
  | { status: 'brainstorm_choice'; token: string; brainstorm_history: Array<{question: string; answer: string}>; summary?: string; integration_note?: string }
  | { status: 'error'; error: string; video_failed?: boolean };

export type UpdateResponse =
  | { status: 'updated'; issue_number: number; issue_url: string; video_summary: string; pr_number?: number | null; comment_id?: number | null; comment_created_at?: string | null }
  | { status: 'ideation'; question: string; token: string; summary?: string; integration_note?: string }
  | { status: 'brainstorm_choice'; token: string; brainstorm_history: Array<{question: string; answer: string}>; summary?: string; integration_note?: string }
  | { status: 'error'; error: string };

export type NextQuestionResponse =
  | { status: 'ideation'; question: string; token: string; brainstorm_history: Array<{question: string; answer: string}>; summary?: string; integration_note?: string }
  | { status: 'error'; error: string };

export type ClaudeProgressStepStatus = 'completed' | 'in_progress' | 'pending' | 'failed';

export interface ClaudeProgressStep {
  id: string;
  label: string;
  raw_label: string;
  status: ClaudeProgressStepStatus;
}

export interface ClaudeProgressResponse {
  status: 'waiting_for_comment' | 'available' | 'completed' | 'failed' | 'cancelled' | 'unavailable';
  title?: string | null;
  issue_number?: number | null;
  comment_id?: number | null;
  body?: string;
  steps: ClaudeProgressStep[];
  updated_at?: string | null;
  message?: string;
  error?: string;
}

export type AskAgentResponse =
  | { status: 'clarification'; token: string; answer: string; brainstorm_history: Array<{question: string; answer: string}>; summary?: string; integration_note?: string }
  | { status: 'error'; error: string };
