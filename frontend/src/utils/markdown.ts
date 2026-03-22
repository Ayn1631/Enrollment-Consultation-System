function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

function applyInlineMarkdown(text: string): string {
  let html = text
  html = html.replace(/`([^`\n]+)`/g, '<code>$1</code>')
  html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>')
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
  html = html.replace(/(^|[^\*])\*([^*\n]+)\*(?!\*)/g, '$1<em>$2</em>')
  return html
}

function renderParagraph(block: string): string {
  const lines = block.split('\n')
  if (lines.every((line) => /^\s*[-*]\s+/.test(line))) {
    const items = lines
      .map((line) => line.replace(/^\s*[-*]\s+/, '').trim())
      .filter(Boolean)
      .map((line) => `<li>${applyInlineMarkdown(line)}</li>`)
      .join('')
    return `<ul>${items}</ul>`
  }

  const html = lines
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => applyInlineMarkdown(line))
    .join('<br />')
  return `<p>${html}</p>`
}

export function renderMarkdown(source: string): string {
  if (!source.trim()) {
    return '<p></p>'
  }

  const escaped = escapeHtml(source).replace(/\r\n/g, '\n')
  const fencePattern = /```([\w-]*)\n([\s\S]*?)```/g
  const segments: string[] = []
  let lastIndex = 0
  let match: RegExpExecArray | null

  while ((match = fencePattern.exec(escaped)) !== null) {
    const [fullMatch, language, code] = match
    const plainText = escaped.slice(lastIndex, match.index)
    if (plainText.trim()) {
      const blocks = plainText.split(/\n{2,}/).map((block) => block.trim()).filter(Boolean)
      segments.push(...blocks.map((block) => renderParagraph(block)))
    }
    const languageClass = language ? ` class="language-${language}"` : ''
    segments.push(`<pre><code${languageClass}>${code.trimEnd()}</code></pre>`)
    lastIndex = match.index + fullMatch.length
  }

  const tail = escaped.slice(lastIndex)
  if (tail.trim()) {
    const blocks = tail.split(/\n{2,}/).map((block) => block.trim()).filter(Boolean)
    segments.push(...blocks.map((block) => renderParagraph(block)))
  }

  return segments.join('')
}
