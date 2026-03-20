<script setup lang="ts">
defineProps<{ mode: string }>()
</script>

<template>
  <header class="topbar">
    <div class="brand">
      <div class="crest">ZUT</div>
      <div>
        <div class="title">中原工学院招生咨询系统</div>
        <div class="subtitle">Admissions Intelligence Console</div>
      </div>
    </div>
    <div class="mode-chip">
      <span class="dot"></span>
      {{ mode }}
    </div>
  </header>
</template>

<style scoped>
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  background: linear-gradient(120deg, rgba(255, 255, 255, 0.95), rgba(245, 242, 236, 0.9));
  border: 1px solid var(--line-soft);
  border-radius: 20px;
  box-shadow: var(--shadow-strong);
  backdrop-filter: blur(12px);
}

.brand {
  display: flex;
  align-items: center;
  gap: 14px;
}

.crest {
  width: 46px;
  height: 46px;
  border-radius: 14px;
  display: grid;
  place-items: center;
  font-family: var(--font-display);
  font-size: 18px;
  color: var(--ink-0);
  background: radial-gradient(circle at 30% 30%, rgba(198, 40, 50, 0.9), rgba(246, 224, 224, 0.96));
  border: 1px solid rgba(166, 30, 36, 0.34);
  letter-spacing: 0.06em;
}

.title {
  font-size: 20px;
  font-weight: 600;
  letter-spacing: 0.04em;
}

.subtitle {
  font-size: 12px;
  color: var(--ink-2);
  letter-spacing: 0.3em;
  text-transform: uppercase;
}

.mode-chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 14px;
  border-radius: 999px;
  border: 1px solid rgba(166, 30, 36, 0.28);
  background: rgba(166, 30, 36, 0.1);
  color: var(--accent);
  font-size: 13px;
}

.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 10px rgba(166, 30, 36, 0.45);
}
</style>
