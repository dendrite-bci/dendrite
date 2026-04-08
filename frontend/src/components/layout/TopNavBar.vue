<script setup lang="ts">
import { computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useStreamManagerStore } from '../../stores/streamManager'

const emit = defineEmits<{ 'toggle-stream-manager': [] }>()

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

    <!-- Stream Manager toggle -->
    <div class="flex items-center gap-3">
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
