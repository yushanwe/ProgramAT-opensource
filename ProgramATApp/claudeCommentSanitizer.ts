const IMAGE_MARKDOWN_RE = /!\[[^\]]*]\(([^)]+)\)/gi;
const IMG_TAG_RE = /<img\b[^>]*>/gi;
const HTML_TAG_RE = /<\/?(?:div|span|p|br|details|summary|picture|source)\b[^>]*>/gi;
const GENERIC_HTML_RE = /<\/?[^>]+>/g;
const RAW_URL_RE = /https?:\/\/\S+/gi;

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
