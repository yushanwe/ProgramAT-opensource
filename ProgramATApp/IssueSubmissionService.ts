/**
 * IssueSubmissionService
 * HTTP layer for the chat-style issue creation/update flow. Every call in
 * this module hits the existing /submit-creation, /submit-update, and
 * /brainstorm-next-question REST endpoints — no backend changes.
 */

import WebSocketService from './WebSocketService';
import { ClaudeProgressResponse, CreationResponse, NextQuestionResponse, UpdateResponse } from './IssueChatTypes';

const REQUEST_TIMEOUT_MS = 120_000; // text-only requests: brainstorm question, no video
const VIDEO_REQUEST_TIMEOUT_MS = 300_000; // video upload + summarization + two Gemini calls; uploads over a tunnel (e.g. ngrok) can be slow

/** Convert the WebSocket URL to an HTTP base URL for REST endpoints. */
export function getHttpBaseUrl(): string {
  const wsUrl = WebSocketService.getServerUrl(); // e.g. 'ws://1.2.3.4:8081'
  return wsUrl.replace(/^ws/, 'http'); // ws:// → http://, wss:// → https://
}

/**
 * Build the multipart 'video' part from a file:// URI, using the real file
 * extension for the name and MIME type. Hardcoding .mp4 for a .mov file makes
 * iOS reject the upload with "Network request failed", so derive both.
 */
export function buildVideoPart(uri: string) {
  const ext = (uri.split('.').pop() || 'mp4').toLowerCase().split('?')[0];
  const mimeByExt: Record<string, string> = {
    mp4: 'video/mp4',
    m4v: 'video/mp4',
    mov: 'video/quicktime',
    qt: 'video/quicktime',
    webm: 'video/webm',
  };
  const type = mimeByExt[ext] || 'video/mp4';
  return { uri, type, name: `recording.${ext}` } as any;
}

async function postWithTimeout(
  url: string,
  body: FormData | string,
  headers?: Record<string, string>,
  timeoutMs: number = REQUEST_TIMEOUT_MS,
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, {
      method: 'POST',
      body,
      headers,
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timeoutId);
  }
}

export interface SubmitCreationArgs {
  text: string;
  videoUri?: string | null;
  brainstormingEnabled?: boolean;
  ideationAnswer?: string;
  token?: string;
  choice?: 'keep_brainstorming' | 'start_building';
}

export async function submitCreation(args: SubmitCreationArgs): Promise<CreationResponse> {
  const baseUrl = getHttpBaseUrl();
  if (!baseUrl) {
    return { status: 'error', error: 'Server URL not configured' };
  }

  const metadata: Record<string, unknown> = { text: args.text };
  if (args.ideationAnswer !== undefined) metadata.ideation_answer = args.ideationAnswer;
  if (args.token !== undefined) metadata.token = args.token;
  if (args.choice !== undefined) metadata.choice = args.choice;
  if (args.brainstormingEnabled !== undefined) metadata.brainstormingEnabled = args.brainstormingEnabled;

  const formData = new FormData();
  formData.append('metadata', JSON.stringify(metadata));
  if (args.videoUri) {
    formData.append('video', buildVideoPart(args.videoUri));
  }

  try {
    const response = await postWithTimeout(
      `${baseUrl}/submit-creation`,
      formData,
      undefined,
      args.videoUri ? VIDEO_REQUEST_TIMEOUT_MS : REQUEST_TIMEOUT_MS,
    );
    const result = await response.json();
    console.log('[IssueSubmissionService] submit-creation response status=', response.status, 'body=', JSON.stringify(result));
    return result as CreationResponse;
  } catch (e) {
    console.log('[IssueSubmissionService] submit-creation threw:', String(e));
    return { status: 'error', error: 'Network error. Please try again.' };
  }
}

export interface SubmitUpdateArgs {
  text: string;
  issueNumber: number;
  videoUri?: string | null;
  brainstormingEnabled?: boolean;
  ideationAnswer?: string;
  token?: string;
  choice?: 'keep_brainstorming' | 'start_building';
}

export async function submitUpdate(args: SubmitUpdateArgs): Promise<UpdateResponse> {
  const baseUrl = getHttpBaseUrl();
  if (!baseUrl) {
    return { status: 'error', error: 'Server URL not configured' };
  }

  const formData = new FormData();
  formData.append('metadata', JSON.stringify({
    text: args.text,
    issue_number: args.issueNumber,
    ideation_answer: args.ideationAnswer,
    token: args.token,
    choice: args.choice,
    brainstormingEnabled: args.brainstormingEnabled,
  }));
  if (args.videoUri) {
    formData.append('video', buildVideoPart(args.videoUri));
  }

  try {
    const response = await postWithTimeout(
      `${baseUrl}/submit-update`,
      formData,
      undefined,
      args.videoUri ? VIDEO_REQUEST_TIMEOUT_MS : REQUEST_TIMEOUT_MS,
    );
    const result = await response.json();
    console.log('[IssueSubmissionService] submit-update response status=', response.status, 'body=', JSON.stringify(result));
    return result as UpdateResponse;
  } catch (e) {
    console.log('[IssueSubmissionService] submit-update threw:', String(e));
    return { status: 'error', error: 'Could not reach the server. Check your connection.' };
  }
}

export async function nextBrainstormQuestion(token: string): Promise<NextQuestionResponse> {
  const baseUrl = getHttpBaseUrl();
  if (!baseUrl) {
    return { status: 'error', error: 'Server URL not configured' };
  }

  try {
    const response = await postWithTimeout(
      `${baseUrl}/brainstorm-next-question`,
      JSON.stringify({ token }),
      { 'Content-Type': 'application/json' },
    );
    const result = await response.json();
    console.log('[IssueSubmissionService] brainstorm-next-question response:', JSON.stringify(result));
    return result as NextQuestionResponse;
  } catch (e) {
    console.log('[IssueSubmissionService] brainstorm-next-question threw:', String(e));
    return { status: 'error', error: 'Network error. Please try again.' };
  }
}

export interface FetchClaudeProgressArgs {
  mode: 'create' | 'update';
  issueNumber?: number | null;
  prNumber?: number | null;
  commentId?: number | null;
}

export async function fetchClaudeProgress(args: FetchClaudeProgressArgs): Promise<ClaudeProgressResponse> {
  const baseUrl = getHttpBaseUrl();
  if (!baseUrl) {
    return { status: 'unavailable', steps: [], error: 'Server URL not configured' };
  }
  const issueNumber = args.mode === 'update' ? args.prNumber : args.issueNumber;
  if (!issueNumber) {
    return { status: 'unavailable', steps: [], error: 'No issue number available for progress lookup' };
  }

  const query = new URLSearchParams();
  query.append(args.mode === 'update' ? 'pr_number' : 'issue_number', String(issueNumber));
  query.append('mode', args.mode);
  if (args.commentId) query.append('comment_id', String(args.commentId));

  try {
    const response = await fetch(`${baseUrl}/claude-progress?${query.toString()}`);
    const result = await response.json();
    return result as ClaudeProgressResponse;
  } catch (e) {
    console.log('[IssueSubmissionService] claude-progress threw:', String(e));
    return { status: 'unavailable', steps: [], message: 'Could not load Claude progress. Retrying…' };
  }
}
