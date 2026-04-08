<script setup lang="ts">
defineProps<{
  label: string
  value: string
  highlight?: boolean
}>()
</script>

<template>
  <span class="text-xs">
    <span class="text-text-disabled">{{ label }}</span>
    <span class="ml-1 font-semibold" :class="highlight ? 'text-accent' : 'text-text-main'">{{ value }}</span>
  </span>
</template>
