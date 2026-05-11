<script setup lang="ts">
withDefaults(defineProps<{
  modelValue: boolean
  compact?: boolean
  disabled?: boolean
}>(), {
  compact: false,
  disabled: false,
})

defineEmits<{
  'update:modelValue': [value: boolean]
}>()
</script>

<template>
  <button
    type="button"
    @click="!disabled && $emit('update:modelValue', !modelValue)"
    class="rounded-full transition-colors relative"
    :class="[
      modelValue ? 'bg-accent' : 'bg-border',
      compact ? 'w-7 h-[14px]' : 'w-10 h-[22px]',
      disabled ? 'opacity-40 cursor-not-allowed' : '',
    ]"
  >
    <div
      class="rounded-full bg-text-main absolute transition-transform"
      :class="[
        compact ? 'w-2.5 h-2.5 top-[2px]' : 'w-4 h-4 top-[3px]',
        modelValue
          ? (compact ? 'translate-x-[12px]' : 'translate-x-[22px]')
          : 'translate-x-[2px]',
      ]"
    />
  </button>
</template>
