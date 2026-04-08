<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useConfigStore } from '../../stores/config'
import { useToast } from '../../composables/useToast'

const emit = defineEmits<{ close: [] }>()
const config = useConfigStore()
const toast = useToast()

function onKey(e: KeyboardEvent) { if (e.key === 'Escape') emit('close') }
onMounted(() => window.addEventListener('keydown', onKey))
onUnmounted(() => window.removeEventListener('keydown', onKey))

const fileName = ref('config.json')
const isSaving = ref(false)
const savedPath = ref<string | null>(null)

const studyName = computed(() => config.general.study_name)
const displayPath = computed(() => `studies/${studyName.value}/config/${fileName.value}`)

async function save() {
  isSaving.value = true
  try {
    const name = fileName.value.endsWith('.json') ? fileName.value : `${fileName.value}.json`
    const path = await config.saveConfig(name === 'config.json' ? undefined : name)
    if (path) {
      savedPath.value = path
      toast.success('Configuration saved')
      setTimeout(() => emit('close'), 1200)
    } else {
      toast.error('Failed to save configuration')
    }
  } finally {
    isSaving.value = false
  }
}
</script>

<template>
  <Teleport to="body">
    <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" @click.self="emit('close')">
      <div class="bg-bg-panel border border-border rounded-lg shadow-2xl shadow-black/40 w-[420px] flex flex-col">

        <!-- Header -->
        <div class="flex items-center justify-between px-5 py-3 border-b border-border shrink-0">
          <h2 class="text-base font-semibold text-text-main">Save Configuration</h2>
          <button @click="emit('close')" class="text-text-disabled hover:text-text-main transition-colors p-1">
            <i class="pi pi-times" />
          </button>
        </div>

        <!-- Content -->
        <div class="px-5 py-4 space-y-3">
          <!-- Success state -->
          <div v-if="savedPath" class="py-4 text-center">
            <i class="pi pi-check-circle text-2xl text-status-ok block mb-2" />
            <p class="text-sm text-text-main">Configuration saved</p>
            <p class="text-xs text-text-disabled mt-1 font-mono truncate">{{ savedPath }}</p>
          </div>

          <!-- Form -->
          <template v-else>
            <div>
              <span class="text-sm text-text-muted block mb-1">Study</span>
              <div class="text-sm text-text-main font-mono px-3 py-1.5 bg-bg-elevated rounded">
                {{ studyName }}
              </div>
            </div>
            <div>
              <span class="text-sm text-text-muted block mb-1">File name</span>
              <input
                v-model="fileName"
                class="w-full font-mono"
              />
            </div>
            <div class="text-xs text-text-disabled font-mono truncate">
              {{ displayPath }}
            </div>
          </template>
        </div>

        <!-- Footer -->
        <div v-if="!savedPath" class="flex justify-end gap-2 px-5 py-3 border-t border-border shrink-0">
          <button
            @click="emit('close')"
            :disabled="isSaving"
            class="px-5 py-2 text-sm rounded border border-border text-text-muted
                   hover:text-text-main hover:border-text-muted transition-colors disabled:opacity-30"
          >Cancel</button>
          <button
            @click="save"
            :disabled="!fileName.trim() || isSaving"
            class="px-5 py-2 text-sm rounded bg-accent text-white hover:bg-accent-hover
                   disabled:opacity-30 transition-colors"
          >
            <i v-if="isSaving" class="pi pi-spin pi-spinner mr-1" />
            {{ isSaving ? 'Saving...' : 'Save' }}
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>
