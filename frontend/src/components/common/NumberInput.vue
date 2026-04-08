<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  modelValue: number | null
  min?: number
  max?: number
  step?: number
  placeholder?: string
  disabled?: boolean
  compact?: boolean
}>(), {
  min: -Infinity,
  max: Infinity,
  step: 1,
  placeholder: '',
  disabled: false,
  compact: false,
})

const emit = defineEmits<{
  'update:modelValue': [value: number | null]
}>()

const displayValue = computed(() => props.modelValue ?? '')

function onInput(e: Event) {
  const raw = (e.target as HTMLInputElement).value
  if (raw === '') { emit('update:modelValue', null); return }
  const n = Number(raw)
  if (!isNaN(n)) emit('update:modelValue', n)
}

function clamp(v: number): number {
  return Math.min(props.max, Math.max(props.min, v))
}

function round(v: number): number {
  const decimals = (props.step.toString().split('.')[1] || '').length
  return Number(v.toFixed(decimals))
}

function increment() {
  const base = props.modelValue ?? 0
  emit('update:modelValue', round(clamp(base + props.step)))
}

function decrement() {
  const base = props.modelValue ?? 0
  emit('update:modelValue', round(clamp(base - props.step)))
}

let holdTimer: ReturnType<typeof setInterval> | null = null

function startHold(fn: () => void) {
  fn()
  holdTimer = setInterval(fn, 120)
}

function stopHold() {
  if (holdTimer) { clearInterval(holdTimer); holdTimer = null }
}
</script>

<template>
  <div
    class="inline-flex items-stretch rounded border border-border overflow-hidden"
    :class="[
      compact ? 'h-6' : 'h-8',
      disabled ? 'opacity-50 pointer-events-none' : '',
    ]"
  >
    <button
      type="button"
      tabindex="-1"
      class="shrink-0 flex items-center justify-center bg-bg-elevated/50 hover:bg-bg-hover text-text-disabled hover:text-text-muted transition-colors select-none"
      :class="compact ? 'w-4 text-[10px]' : 'w-5 text-xs'"
      @mousedown.prevent="startHold(decrement)"
      @mouseup="stopHold"
      @mouseleave="stopHold"
    >−</button>
    <input
      type="number"
      :value="displayValue"
      :min="min"
      :max="max"
      :step="step"
      :placeholder="placeholder"
      :disabled="disabled"
      class="min-w-0 flex-1 border-x border-border bg-bg-input text-center focus:z-10"
      :class="compact ? 'text-[11px] px-0.5' : 'text-sm px-1.5'"
      @input="onInput"
    />
    <button
      type="button"
      tabindex="-1"
      class="shrink-0 flex items-center justify-center bg-bg-elevated/50 hover:bg-bg-hover text-text-disabled hover:text-text-muted transition-colors select-none"
      :class="compact ? 'w-4 text-[10px]' : 'w-5 text-xs'"
      @mousedown.prevent="startHold(increment)"
      @mouseup="stopHold"
      @mouseleave="stopHold"
    >+</button>
  </div>
</template>
