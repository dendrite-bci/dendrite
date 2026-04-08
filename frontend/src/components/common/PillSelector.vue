<script setup lang="ts">
defineProps<{
  modelValue: string | boolean
  options: Array<{ label: string; value: string | boolean }>
}>()

defineEmits<{
  'update:modelValue': [value: string | boolean]
}>()
</script>

<template>
  <div class="flex bg-bg-input rounded-lg p-0.5">
    <button v-for="opt in options" :key="String(opt.value)"
      @click="$emit('update:modelValue', opt.value)"
      class="flex-1 px-3 py-1.5 text-xs rounded transition-colors"
      :class="modelValue === opt.value
        ? 'bg-bg-hover text-text-main font-semibold'
        : 'text-text-muted hover:text-text-main'">
      {{ opt.label }}
    </button>
  </div>
</template>
