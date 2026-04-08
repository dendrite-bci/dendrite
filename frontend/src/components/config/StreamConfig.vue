<script setup lang="ts">
import { ref } from 'vue'
import { useStreamsStore } from '../../stores/streams'
import { typeBadgeClass, typeBadgeBase } from '../../utils/streamBadge'
import StreamSetupDialog from './StreamSetupDialog.vue'

const streams = useStreamsStore()
const showSetupDialog = ref(false)

function openSetup() {
  showSetupDialog.value = true
}

function onApplied() {
  streams.fetchConfigured()
}

</script>

<template>
  <div>
    <button
      @click="openSetup"
      class="w-full px-3 py-1.5 text-xs rounded transition-colors mb-3"
      :class="streams.hasStreams
        ? 'bg-bg-input border border-border text-text-label hover:text-text-main hover:border-accent'
        : 'bg-accent text-white hover:bg-accent-hover'"
    >
      <i class="pi pi-wifi mr-1.5" />
      {{ streams.hasStreams ? 'Stream Setup' : 'Configure Streams' }}
    </button>

    <!-- Configured streams summary -->
    <div v-if="streams.hasStreams" class="space-y-2">
      <div
        v-for="(stream, uid) in streams.configuredStreams"
        :key="uid"
        @click="openSetup"
        class="px-3 py-3.5 bg-bg-elevated rounded-lg border border-border
               cursor-pointer hover:border-accent transition-colors"
      >
        <!-- Row 1: type badge + name + specs + status -->
        <div class="flex items-center gap-3">
          <span
            class="shrink-0"
            :class="[typeBadgeBase, typeBadgeClass(stream.type)]"
          >{{ stream.type }}</span>
          <div class="flex-1 min-w-0">
            <div class="text-sm text-text-muted truncate">{{ stream.name }}</div>
          </div>
          <span class="text-xs text-text-muted font-mono shrink-0">
            {{ stream.channel_count }}ch @ {{ stream.sample_rate }}Hz
          </span>
          <div class="w-2 h-2 rounded-full shrink-0"
               :class="streams.liveness[uid] === false ? 'bg-status-error' : 'bg-status-ok'" />
        </div>
      </div>
    </div>

    <div v-else class="px-3 py-6 text-center rounded border border-dashed border-border">
      <i class="pi pi-wifi text-xl text-text-disabled block mb-2" />
      <p class="text-xs text-text-muted">No streams configured</p>
    </div>

    <!-- Stream Setup Dialog -->
    <StreamSetupDialog
      v-if="showSetupDialog"
      @close="showSetupDialog = false"
      @applied="onApplied"
    />
  </div>
</template>
