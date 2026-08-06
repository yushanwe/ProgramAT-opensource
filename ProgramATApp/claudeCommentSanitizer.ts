const IMAGE_MARKDOWN_RE = /!\[[^\]]*]\(([^)]+)\)/gi;
const IMG_TAG_RE = /<img\b[^>]*>/gi;
const HTML_TAG_RE = /<\/?(?:div|span|p|br|details|summary|picture|source)\b[^>]*>/gi;
const GENERIC_HTML_RE = /<\/?[^>]+>/g;
const RAW_URL_RE = /https?:\/\/\S+/gi;
const CHECKLIST_LINE_RE = /^\s*[-*]\s*\[(x| |)\]\s+(.+?)\s*$/i;
const CHECKLIST_LINE_COMPACT_RE = /^\s*[-*]\s*\[\]\s+(.+?)\s*$/i;
const HEADING_RE = /^#{1,6}\s+(.+?)\s*$/;

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

export interface ClaudeRenderLine {
  kind: 'heading' | 'bullet' | 'paragraph' | 'blank';
  text: string;
  accessibilityLabel?: string;
}

export interface ClaudeAccessibilityBlock {
  kind: 'section' | 'checklist_item';
  label: string;
}

const SECTION_HEADINGS = new Set([
  'current analysis',
  'implementation decisions',
  'recent work',
  'next step',
  'implementation summary',
]);

export function normalizeChecklistLineForVoiceOver(line: string): string | null {
  const compactMatch = line.match(CHECKLIST_LINE_COMPACT_RE);
  const standardMatch = line.match(CHECKLIST_LINE_RE);
  const task = compactMatch?.[1] || standardMatch?.[2];
  if (!task) return null;
  return stripInlineMarkdownPunctuation(task);
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

export function parseClaudeRenderLines(body: string | undefined | null): ClaudeRenderLine[] {
  const sanitized = sanitizeClaudeCommentBody(body);
  if (!sanitized) return [];

  const lines = sanitized.split('\n');
  const renderLines: ClaudeRenderLine[] = [];
  let uncheckedSeen = 0;

  for (const rawLine of lines) {
    const line = rawLine.trimEnd();
    if (!line.trim()) {
      renderLines.push({ kind: 'blank', text: '' });
      continue;
    }
    const headingMatch = line.match(HEADING_RE);
    if (headingMatch) {
      renderLines.push({
        kind: 'heading',
        text: headingMatch[1],
        accessibilityLabel: stripInlineMarkdownPunctuation(headingMatch[1]),
      });
      continue;
    }
    const compactMatch = line.match(CHECKLIST_LINE_COMPACT_RE);
    const standardMatch = line.match(CHECKLIST_LINE_RE);
    const task = compactMatch?.[1] || standardMatch?.[2];
    if (task) {
      const checked = compactMatch ? false : (standardMatch?.[1] || '').toLowerCase() === 'x';
      if (!checked) uncheckedSeen += 1;
      const prefix = checked ? 'Finished' : uncheckedSeen === 1 ? 'Ongoing' : 'To do';
      renderLines.push({
        kind: 'bullet',
        text: line,
        accessibilityLabel: `${prefix}: ${stripInlineMarkdownPunctuation(task)}`,
      });
      continue;
    }
    renderLines.push({
      kind: line.startsWith('- ') || line.startsWith('* ') ? 'bullet' : 'paragraph',
      text: line,
      accessibilityLabel: stripInlineMarkdownPunctuation(line),
    });
  }

  return renderLines;
}

function normalizeSectionText(line: string): string {
  return stripInlineMarkdownPunctuation(line)
    .replace(/^\s*[-*]\s*/, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function flushSectionBlock(
  blocks: ClaudeAccessibilityBlock[],
  heading: string | null,
  lines: string[],
): void {
  const normalizedHeading = heading ? stripInlineMarkdownPunctuation(heading) : '';
  const normalizedLines = lines
    .map(normalizeSectionText)
    .filter(Boolean);
  if (!normalizedHeading && normalizedLines.length === 0) return;
  const parts = [normalizedHeading, ...normalizedLines].filter(Boolean);
  blocks.push({
    kind: 'section',
    label: parts.join('. ').trim(),
  });
}

export function buildClaudeAccessibilityBlocks(body: string | undefined | null): ClaudeAccessibilityBlock[] {
  const sanitized = sanitizeClaudeCommentBody(body);
  if (!sanitized) return [];

  const lines = sanitized.split('\n');
  const blocks: ClaudeAccessibilityBlock[] = [];
  let uncheckedSeen = 0;
  let currentHeading: string | null = null;
  let currentLines: string[] = [];

  const flush = () => {
    flushSectionBlock(blocks, currentHeading, currentLines);
    currentHeading = null;
    currentLines = [];
  };

  for (const rawLine of lines) {
    const line = rawLine.trimEnd();
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }

    const headingMatch = trimmed.match(HEADING_RE);
    if (headingMatch) {
      const nextHeading = headingMatch[1];
      if (currentHeading || currentLines.length > 0) {
        flush();
      }
      if (SECTION_HEADINGS.has(nextHeading.toLowerCase())) {
        currentHeading = nextHeading;
      } else {
        currentLines.push(nextHeading);
      }
      continue;
    }

    const compactMatch = trimmed.match(CHECKLIST_LINE_COMPACT_RE);
    const standardMatch = trimmed.match(CHECKLIST_LINE_RE);
    const task = compactMatch?.[1] || standardMatch?.[2];
    if (task) {
      flush();
      const checked = compactMatch ? false : (standardMatch?.[1] || '').toLowerCase() === 'x';
      if (!checked) uncheckedSeen += 1;
      const prefix = checked ? 'Finished' : uncheckedSeen === 1 ? 'Ongoing' : 'To do';
      blocks.push({
        kind: 'checklist_item',
        label: `${prefix}: ${stripInlineMarkdownPunctuation(task)}`,
      });
      continue;
    }

    currentLines.push(trimmed);
  }

  flush();
  return blocks;
}

function sectionBodyMap(body: string | undefined | null): Record<string, string> {
  const lines = sanitizeClaudeCommentBody(body).split('\n');
  const sections: Record<string, string[]> = {};
  let current = '';
  for (const rawLine of lines) {
    const line = rawLine.trim();
    const headingMatch = line.match(HEADING_RE);
    if (headingMatch) {
      current = headingMatch[1].toLowerCase();
      sections[current] = [];
      continue;
    }
    if (current) {
      sections[current].push(line);
    }
  }
  return Object.fromEntries(
    Object.entries(sections).map(([key, value]) => [key, value.join(' ').replace(/\s+/g, ' ').trim()])
  );
}

export function getChangedClaudeAnnouncement(
  previousBody: string | undefined | null,
  nextBody: string | undefined | null,
): string | null {
  const previous = sectionBodyMap(previousBody);
  const next = sectionBodyMap(nextBody);
  const orderedSections = ['recent work', 'current analysis', 'next step'];
  for (const section of orderedSections) {
    if (next[section] && next[section] !== previous[section]) {
      const prefix = section.replace(/\b\w/g, (char) => char.toUpperCase());
      return `${prefix}: ${next[section]}`;
    }
  }
  return null;
}
