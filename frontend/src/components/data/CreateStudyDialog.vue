<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { useDataStore } from '../../stores/data'
import { useToast } from '../../composables/useToast'

const emit = defineEmits<{ close: [] }>()

const data = useDataStore()
const toast = useToast()

const studyName = ref('')
const description = ref('')
const folderPath = ref('')
const loading = ref(false)
const browsing = ref(false)
const result = ref<{ imported_count: number; skipped: number; errors: string[]; total_found: number } | null>(null)
const errorMsg = ref('')

async function browseFolder() {
  browsing.value = true
  try {
    const path = await data.pickFolder()
    if (path) folderPath.value = path
  } finally {
    browsing.value = false
  }
}

async function handleSubmit() {
  const name = studyName.value.trim()
  if (!name) return

  loading.value = true
  errorMsg.value = ''
  result.value = null

  try {
    if (folderPath.value.trim()) {
      const res = await data.importStudyFolder({
        folder_path: folderPath.value.trim(),
        study_name: name,
        description: description.value.trim() || undefined,
      })
      result.value = res
      toast.success(`Study "${name}" created — ${res.imported_count} recordings imported`)
    } else {
      await data.createStudy(name, description.value.trim() || undefined)
      toast.success(`Study "${name}" created`)
      emit('close')
    }
  } catch (e: any) {
    errorMsg.value = e.message || 'Failed'
    toast.error(e.message || 'Study creation failed')
  } finally {
    loading.value = false
  }
}

function onKey(e: KeyboardEvent) {
  if (e.key === 'Escape') emit('close')
}
onMounted(() => window.addEventListener('keydown', onKey))
onUnmounted(() => window.removeEventListener('keydown', onKey))
</script>

<template>
  <Teleport to="body">
    <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" @click.self="emit('close')">
      <div class="bg-bg-panel border border-border rounded-lg shadow-2xl p-6 w-[440px] space-y-4">
        <h3 class="text-base font-semibold text-text-main">New Study</h3>

        <!-- Name -->
        <label class="block">
          <span class="text-xs text-text-muted block mb-1">Study Name</span>
          <input
            v-model="studyName"
            placeholder="e.g. motor-imagery-pilot"
            class="w-full"
            @keydown.enter="handleSubmit"
          />
        </label>

        <!-- Description -->
        <label class="block">
          <span class="text-xs text-text-muted block mb-1">Description</span>
          <input
            v-model="description"
            placeholder="Optional"
            class="w-full"
          />
        </label>

        <!-- Folder path -->
        <div>
          <span class="text-xs text-text-muted block mb-1">Import recordings from folder</span>
          <div class="flex gap-2">
            <input
              v-model="folderPath"
              placeholder="Folder path (optional) — scans for .h5 files"
              class="flex-1 font-mono text-xs"
            />
            <button
              @click="browseFolder"
              :disabled="browsing"
              class="px-3 py-1.5 text-xs rounded border border-border text-text-muted
                     hover:text-text-main hover:border-text-muted transition-colors
                     disabled:opacity-30 shrink-0"
            >
              <i v-if="browsing" class="pi pi-spin pi-spinner mr-1" />
              Browse
            </button>
          </div>
          <span class="text-xs text-text-disabled mt-0.5 block">Leave empty to create an empty study</span>
        </div>

        <!-- Error -->
        <div v-if="errorMsg" class="px-3 py-2 rounded border border-status-error/30 bg-status-error/5">
          <p class="text-xs text-status-error">{{ errorMsg }}</p>
        </div>

        <!-- Import result -->
        <div v-if="result" class="px-3 py-2 rounded border border-status-ok/30 bg-status-ok/5 space-y-1">
          <p class="text-xs text-status-ok font-medium">
            {{ result.imported_count }} imported, {{ result.skipped }} skipped
            <span class="text-text-disabled">({{ result.total_found }} files found)</span>
          </p>
          <div v-if="result.errors.length" class="mt-1">
            <p class="text-xs text-status-error" v-for="err in result.errors.slice(0, 5)" :key="err">{{ err }}</p>
            <p v-if="result.errors.length > 5" class="text-xs text-text-disabled">
              ...and {{ result.errors.length - 5 }} more
            </p>
          </div>
        </div>

        <!-- Actions -->
        <div class="flex justify-end gap-3 pt-2">
          <button
            @click="emit('close')"
            class="px-4 py-2 text-xs rounded border border-border text-text-muted
                   hover:text-text-main hover:border-text-muted transition-colors"
          >{{ result ? 'Done' : 'Cancel' }}</button>
          <button
            v-if="!result"
            @click="handleSubmit"
            :disabled="loading || !studyName.trim()"
            class="px-4 py-2 text-xs rounded bg-accent text-white
                   hover:bg-accent/80 transition-colors disabled:opacity-30"
          >
            <i v-if="loading" class="pi pi-spin pi-spinner mr-1.5" />
            {{ folderPath.trim() ? 'Create & Import' : 'Create' }}
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>
