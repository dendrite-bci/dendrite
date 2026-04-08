<script setup lang="ts">
import type { H5FileInfo } from '../../types/api'

const props = defineProps<{ info: H5FileInfo }>()
</script>

<template>
  <div class="bg-bg-elevated border border-border rounded-lg p-4 space-y-4">
    <!-- Error state -->
    <div v-if="props.info.error" class="text-xs text-status-error">{{ props.info.error }}</div>

    <template v-else>
      <!-- File size -->
      <div class="flex items-center justify-between">
        <span class="text-xs font-semibold text-text-label">H5 File Structure</span>
        <span class="text-xs text-text-disabled">{{ props.info.file_size_mb?.toFixed(2) }} MB</span>
      </div>

      <!-- Root attributes -->
      <div v-if="Object.keys(props.info.root_attributes || {}).length > 0">
        <span class="text-xs font-semibold text-text-muted block mb-2">Root Attributes</span>
        <div class="grid grid-cols-2 gap-2 text-xs">
          <template v-for="(val, key) in props.info.root_attributes" :key="key">
            <div class="text-text-muted truncate">{{ key }}</div>
            <div class="text-text-main truncate">{{ val }}</div>
          </template>
        </div>
      </div>

      <!-- Datasets -->
      <div v-if="Object.keys(props.info.datasets || {}).length > 0">
        <span class="text-xs font-semibold text-text-muted block mb-2"><i class="pi pi-table mr-1" />Datasets</span>
        <div class="space-y-1.5">
          <div
            v-for="(ds, name) in props.info.datasets"
            :key="name"
            class="bg-bg-input rounded px-3 py-2.5"
          >
            <div class="flex items-center justify-between">
              <span class="text-xs font-semibold text-text-main">{{ name }}</span>
              <span class="text-xs text-text-disabled">{{ ds.dtype }}</span>
            </div>
            <div class="text-xs text-text-muted">
              Shape: {{ JSON.stringify(ds.shape) }}
            </div>
            <div v-if="Object.keys(ds.attributes || {}).length > 0" class="mt-1 text-xs text-text-disabled">
              Attrs: {{ Object.keys(ds.attributes).join(', ') }}
            </div>
          </div>
        </div>
      </div>

      <!-- Groups -->
      <div v-if="Object.keys(props.info.groups || {}).length > 0">
        <span class="text-xs font-semibold text-text-muted block mb-2"><i class="pi pi-folder mr-1" />Groups</span>
        <div class="space-y-1.5">
          <div
            v-for="(grp, name) in props.info.groups"
            :key="name"
            class="bg-bg-input rounded px-3 py-2.5"
          >
            <span class="text-xs font-semibold text-text-main">{{ name }}/</span>
            <div v-if="grp.children.length > 0" class="text-xs text-text-muted mt-0.5">
              Children: {{ grp.children.join(', ') }}
            </div>
          </div>
        </div>
      </div>
    </template>
  </div>
</template>
