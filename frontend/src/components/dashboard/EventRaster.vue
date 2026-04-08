<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useVisualizationStore, vizDirty } from '../../stores/visualization'
import { useModesStore } from '../../stores/modes'

const EVENT_COLORS = [
  '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
  '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
  '#F8C471', '#82E0AA', '#F1948A', '#85929E', '#AED6F1',
  '#D2B4DE', '#A3E4D7', '#F9E79F', '#FADBD8', '#D5F5E3',
]

const viz = useVisualizationStore()
const modes = useModesStore()
const canvas = ref<HTMLCanvasElement>()
let ctx: CanvasRenderingContext2D | null = null
let animFrame = 0
let observer: ResizeObserver | null = null

// Merge event_mapping from all mode instances: code → name
const eventNameMap = computed<Record<number, string>>(() => {
  const map: Record<number, string> = {}
  for (const inst of Object.values(modes.instances)) {
    if (inst.event_mapping) {
      for (const [code, name] of Object.entries(inst.event_mapping)) {
        map[Number(code)] = name
      }
    }
  }
  return map
})

function eventLabel(code: number): string {
  const name = eventNameMap.value[code]
  return name ? `${name}: ${code}` : String(code)
}

function sizeCanvas() {
  const el = canvas.value
  if (!el) return
  const parent = el.parentElement!
  const dpr = window.devicePixelRatio || 1
  el.width = parent.clientWidth * dpr
  el.height = parent.clientHeight * dpr
  el.style.width = parent.clientWidth + 'px'
  el.style.height = parent.clientHeight + 'px'
  ctx = el.getContext('2d')
  if (ctx) ctx.scale(dpr, dpr)
}

function render() {
  if (!ctx || !canvas.value) {
    animFrame = requestAnimationFrame(render)
    return
  }

  const w = canvas.value.clientWidth
  const h = canvas.value.clientHeight

  // Clear
  ctx.clearRect(0, 0, w, h)

  // Time window
  const windowSize = viz.timeAxis.length > 0
    ? viz.timeAxis[viz.timeAxis.length - 1]!
    : 10
  const currentTime = vizDirty.sampleCounter / viz.sampleRate
  const windowStart = Math.max(0, currentTime - windowSize)

  // Draw time axis
  const axisY = h - 14
  ctx.strokeStyle = '#808080'
  ctx.lineWidth = 1
  ctx.beginPath()
  ctx.moveTo(0, axisY)
  ctx.lineTo(w, axisY)
  ctx.stroke()

  // Tick marks every 1s
  ctx.fillStyle = '#808080'
  ctx.font = '10px Inter, sans-serif'
  ctx.textAlign = 'center'
  const tickInterval = windowSize <= 5 ? 0.5 : (windowSize <= 20 ? 1 : 2)
  const firstTick = Math.ceil(windowStart / tickInterval) * tickInterval
  for (let t = firstTick; t <= windowStart + windowSize; t += tickInterval) {
    const x = ((t - windowStart) / windowSize) * w
    ctx.beginPath()
    ctx.moveTo(x, axisY)
    ctx.lineTo(x, axisY + 4)
    ctx.stroke()
    ctx.fillText(t.toFixed(0) + 's', x, h - 2)
  }

  // Filter visible events
  const visible = viz.eventHistory
    .filter(e => e.timeAxisPos >= windowStart && e.timeAxisPos <= windowStart + windowSize)
    .slice(-50)

  if (visible.length === 0) {
    // "No events in window" hint
    ctx.fillStyle = '#808080'
    ctx.font = '11px Inter, sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText('No events in window', w / 2, h / 2)
    animFrame = requestAnimationFrame(render)
    return
  }

  // Draw stems
  const stemTop = 16
  const stemBottom = axisY - 2

  for (const evt of visible) {
    const x = ((evt.timeAxisPos - windowStart) / windowSize) * w
    const color = EVENT_COLORS[Math.abs(Math.round(evt.value)) % EVENT_COLORS.length]!

    // Vertical stem line
    ctx.strokeStyle = color
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(x, stemBottom)
    ctx.lineTo(x, stemTop)
    ctx.stroke()

    // Small diamond marker at top
    ctx.fillStyle = color!
    ctx.beginPath()
    ctx.moveTo(x, stemTop - 4)
    ctx.lineTo(x + 4, stemTop)
    ctx.lineTo(x, stemTop + 4)
    ctx.lineTo(x - 4, stemTop)
    ctx.closePath()
    ctx.fill()

    // Value label above diamond
    ctx.fillStyle = '#ccc'
    ctx.font = 'bold 9px sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText(eventLabel(evt.value), x, stemTop - 7)
  }

  animFrame = requestAnimationFrame(render)
}

onMounted(() => {
  sizeCanvas()
  observer = new ResizeObserver(() => sizeCanvas())
  if (canvas.value?.parentElement) {
    observer.observe(canvas.value.parentElement)
  }
  animFrame = requestAnimationFrame(render)
})

onUnmounted(() => {
  cancelAnimationFrame(animFrame)
  observer?.disconnect()
})
</script>

<template>
  <div class="h-[80px] bg-bg-panel rounded border border-border/50">
    <canvas ref="canvas" class="w-full h-full" />
  </div>
</template>
