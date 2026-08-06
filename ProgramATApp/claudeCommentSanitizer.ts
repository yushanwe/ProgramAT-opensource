const IMAGE_MARKDOWN_RE = /!\[[^\]]*]\(([^)]+)\)/gi;
const IMG_TAG_RE = /<img\b[^>]*>/gi;
const HTML_TAG_RE = /<\/?(?:div|span|p|br|details|summary|picture|source)\b[^>]*>/gi;
const GENERIC_HTML_RE = /<\/?[^>]+>/g;
const RAW_URL_RE = /https?:\/\/\S+/gi;
const CHECKLIST_LINE_RE = /^\s*[-*]\s*\[(x| |)\]\s+(.+?)\s*$/i;
const CHECKLIST_LINE_COMPACT_RE = /^\s*[-*]\s*\[\]\s+(.+?)\s*$/i;

function decodeHtmlEntities(text: string): string {
  return text
    .replace(/&nbsp;/gi, ' ')
    .replace(/&amp;/gi, '&')
    .replace(/&lt;/gi, '<')
    .replace(/&gt;/gi, '>')
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'");
}

function isMediaUrl(url: string): boolean {
  return /(?:githubusercontent\.com|user-images\.githubusercontent\.com|github\.com\/user-attachments\/files\/|\.gif\b|\.png\b|\.jpe?g\b|\.webp\b|spinner|badge)/i.test(url);
}

export function sanitizeClaudeCommentBody(body: string | undefined | null): string {
  if (!body) return '';

  let cleaned = body;

  const removedMediaUrls = new Set<string>();

  cleaned = cleaned.replace(IMAGE_MARKDOWN_RE, (_match, url: string) => {
    if (url && isMediaUrl(url)) {
      removedMediaUrls.add(url);
      return '';
    }
    return '';
  });

  cleaned = cleaned.replace(/<a\b[^>]*href=["']([^"']+)["'][^>]*>\s*<img\b[^>]*>\s*<\/a>/gi, (_match, url: string) => {
    if (url && isMediaUrl(url)) removedMediaUrls.add(url);
    return '';
  });

  cleaned = cleaned.replace(/^\s*Create a PR\s*$/gim, '');
  cleaned = cleaned.replace(/^\s*https:\/\/github\.com\/[^\s]+\/compare\/[^\s]+\s*$/gim, '');

  cleaned = cleaned.replace(IMG_TAG_RE, (match) => {
    const src = match.match(/\bsrc=["']([^"']+)["']/i)?.[1];
    if (src && isMediaUrl(src)) removedMediaUrls.add(src);
    return '';
  });

  cleaned = cleaned.replace(/<(?:video|svg|figure)\b[\s\S]*?<\/(?:video|svg|figure)>/gi, '');
  cleaned = cleaned.replace(HTML_TAG_RE, '\n');
  cleaned = cleaned.replace(GENERIC_HTML_RE, '');
  cleaned = decodeHtmlEntities(cleaned);

  cleaned = cleaned.replace(RAW_URL_RE, (url) => (removedMediaUrls.has(url) || isMediaUrl(url) ? '' : url));
  cleaned = cleaned.replace(/[ \t]+\n/g, '\n');
  cleaned = cleaned.replace(/\n{3,}/g, '\n\n');

  return cleaned.trim();
}

function stripInlineMarkdownPunctuation(text: string): string {
  return text
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/\[([^\]]+)]\(([^)]+)\)/g, '$1')
    .replace(/\s+/g, ' ')
    .trim();
}

export function buildClaudeAccessibilityLabel(body: string | undefined | null, message?: string): string {
  const sanitized = sanitizeClaudeCommentBody(body);
  if (!sanitized) {
    return message || 'Claude has not posted a progress comment yet.';
  }

  const lines = sanitized.split('\n');
  const checklistStates: Array<'checked' | 'unchecked'> = [];
  const normalizedLines = lines.map((line) => {
    const compactMatch = line.match(CHECKLIST_LINE_COMPACT_RE);
    const standardMatch = line.match(CHECKLIST_LINE_RE);
    const task = compactMatch?.[1] || standardMatch?.[2];
    if (!task) return stripInlineMarkdownPunctuation(line);

    const checked = compactMatch ? false : (standardMatch?.[1] || '').toLowerCase() === 'x';
    checklistStates.push(checked ? 'checked' : 'unchecked');
    const uncheckedBefore = checklistStates.filter((state) => state === 'unchecked').length;
    const prefix = checked
      ? 'Finished'
      : uncheckedBefore === 1
      ? 'Ongoing'
      : 'To do';
    return `${prefix}: ${stripInlineMarkdownPunctuation(task)}`;
  });

  return [message, normalizedLines.join('\n').trim()].filter(Boolean).join(' ').trim();
}

export function isTerminalClaudeProgress(progress: { status?: string; body?: string; steps?: Array<{ status: string }> } | null | undefined): boolean {
  if (!progress) return false;
  if (progress.status === 'completed' || progress.status === 'failed' || progress.status === 'cancelled') {
    return true;
  }
  const steps = progress.steps || [];
  if (steps.length > 0 && steps.every((step) => step.status === 'completed')) {
    return true;
  }
  const body = sanitizeClaudeCommentBody(progress.body).toLowerCase();
  return /\b(completed|finished|done|successfully opened pr|opened pr)\b/.test(body);
}
