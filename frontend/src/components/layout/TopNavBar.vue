<script setup lang="ts">
import { computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useStreamManagerStore } from '../../stores/streamManager'
import { useTheme } from '../../composables/useTheme'

const emit = defineEmits<{ 'toggle-stream-manager': [] }>()
const { theme, toggle: toggleTheme } = useTheme()

const router = useRouter()
const route = useRoute()
const streamManager = useStreamManagerStore()

const activeStreamCount = computed(() => streamManager.streams.filter(s => s.running).length)

const tabs = [
  { label: 'Control', route: '/' },
  { label: 'Data', route: '/data' },
  { label: 'ML', route: '/ml' },
]
</script>

<template>
  <header class="flex items-center justify-between px-4 h-12 bg-bg-panel border-b border-border shrink-0">
    <!-- Left: App title -->
    <div class="flex items-center gap-3">
      <span class="text-lg font-semibold text-text-main">Dendrite</span>
      <span class="text-xs text-text-muted">v0.10</span>
    </div>

    <!-- Center: Navigation tabs -->
    <nav class="flex gap-1">
      <button
        v-for="tab in tabs"
        :key="tab.route"
        @click="router.push(tab.route)"
        class="px-4 py-1.5 rounded text-sm transition-colors"
        :class="route.path === tab.route
          ? 'bg-bg-hover text-text-main'
          : 'text-text-muted hover:text-text-main hover:bg-bg-elevated'"
      >
        {{ tab.label }}
      </button>
    </nav>

    <!-- Right actions -->
    <div class="flex items-center gap-3">
      <button
        @click="toggleTheme"
        class="p-1.5 rounded text-text-muted hover:text-text-main hover:bg-bg-elevated transition-colors"
        :title="theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'"
      >
        <svg v-if="theme === 'dark'" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
        <svg v-else xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
      </button>
      <button
        @click="emit('toggle-stream-manager')"
        class="flex items-center gap-1.5 px-3 py-1.5 rounded text-sm transition-colors
               text-text-muted hover:text-text-main hover:bg-bg-elevated"
        title="Stream Manager"
      >
        <i class="pi pi-play-circle" :class="activeStreamCount > 0 ? 'text-status-ok' : ''" />
        <span class="text-xs">Stream Manager</span>
        <span v-if="activeStreamCount > 0" class="text-xs font-mono text-status-ok">{{ activeStreamCount }}</span>
      </button>
    </div>
  </header>
</template>
