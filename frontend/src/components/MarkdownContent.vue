<script setup lang="ts">
import MarkdownIt from 'markdown-it'
import mermaid from 'mermaid'
import { computed, nextTick, onMounted, ref, watch } from 'vue'

const props = defineProps<{
  content: string
}>()

const containerRef = ref<HTMLElement | null>(null)
type FenceToken = { info: string; content: string }

const markdown = new MarkdownIt({
  html: false,
  linkify: true,
  breaks: true
})

markdown.renderer.rules.fence = (tokens: FenceToken[], idx: number) => {
  const token = tokens[idx]
  const info = token.info.trim().split(/\s+/)[0] ?? ''
  const content = markdown.utils.escapeHtml(token.content)
  if (info === 'mermaid') {
    return `<div class="mermaid">${content}</div>`
  }
  const languageClass = info ? ` class="language-${markdown.utils.escapeHtml(info)}"` : ''
  return `<pre><code${languageClass}>${content}</code></pre>`
}

let mermaidInitialized = false

function ensureMermaidInitialized() {
  if (mermaidInitialized) return
  mermaid.initialize({
    startOnLoad: false,
    securityLevel: 'loose',
    theme: 'neutral'
  })
  mermaidInitialized = true
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

const renderedHtml = computed(() => markdown.render(props.content))

async function renderMermaidBlocks() {
  await nextTick()
  const container = containerRef.value
  if (!container) return

  const blocks = Array.from(container.querySelectorAll<HTMLElement>('.mermaid'))
  if (!blocks.length) return

  ensureMermaidInitialized()

  await Promise.all(
    blocks.map(async (block, index) => {
      const source = block.textContent?.trim() ?? ''
      if (!source) return
      try {
        const renderId = `mermaid-${Date.now()}-${index}`
        const { svg } = await mermaid.render(renderId, source)
        block.innerHTML = svg
      } catch {
        block.innerHTML =
          `<pre><code>${escapeHtml(source)}</code></pre>` +
          '<div class="mermaid-error-note">Mermaid 渲染失败，请检查图表语法。</div>'
        block.classList.add('mermaid-error')
      }
    })
  )
}

watch(renderedHtml, () => {
  void renderMermaidBlocks()
})

onMounted(() => {
  void renderMermaidBlocks()
})
</script>

<template>
  <div ref="containerRef" class="markdown-body" v-html="renderedHtml"></div>
</template>

<style scoped>
.markdown-body {
  line-height: 1.7;
  color: var(--ink-0);
}

.markdown-body :deep(p) {
  margin: 0;
}

.markdown-body :deep(p + p),
.markdown-body :deep(p + ul),
.markdown-body :deep(ul + p),
.markdown-body :deep(pre + p),
.markdown-body :deep(p + pre),
.markdown-body :deep(p + .mermaid),
.markdown-body :deep(.mermaid + p) {
  margin-top: 10px;
}

.markdown-body :deep(ul) {
  margin: 0;
  padding-left: 20px;
}

.markdown-body :deep(li + li) {
  margin-top: 4px;
}

.markdown-body :deep(code) {
  font-family: 'Consolas', 'SFMono-Regular', monospace;
  font-size: 0.92em;
  padding: 0.14em 0.38em;
  border-radius: 6px;
  background: rgba(127, 21, 27, 0.08);
}

.markdown-body :deep(pre) {
  margin: 0;
  padding: 12px 14px;
  border-radius: 12px;
  background: rgba(80, 14, 18, 0.92);
  color: #fff8f7;
  overflow-x: auto;
}

.markdown-body :deep(pre code) {
  padding: 0;
  background: transparent;
  color: inherit;
}

.markdown-body :deep(a) {
  color: var(--accent);
  text-decoration: none;
}

.markdown-body :deep(strong) {
  font-weight: 700;
}

.markdown-body :deep(.mermaid) {
  margin: 0;
  padding: 12px;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.92);
  border: 1px solid rgba(166, 30, 36, 0.16);
  overflow-x: auto;
}

.markdown-body :deep(.mermaid svg) {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 0 auto;
}

.markdown-body :deep(.mermaid-error-note) {
  margin-top: 8px;
  font-size: 12px;
  color: #8b1d1d;
}
</style>
