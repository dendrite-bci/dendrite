<script setup lang="ts">
import { onMounted } from 'vue'
import { useConfigStore } from '../../stores/config'
import NumberInput from '../common/NumberInput.vue'

const config = useConfigStore()

onMounted(() => {
  config.fetchOutput()
})

interface FieldDef {
  key: string
  label: string
  type: 'text' | 'number' | 'select'
  placeholder?: string
  min?: number
  max?: number
  options?: string[]
}

const protocols: Array<{ key: string; label: string; description: string }> = [
  { key: 'lsl', label: 'LSL', description: 'Lab Streaming Layer' },
  { key: 'socket', label: 'Socket', description: 'TCP/UDP socket' },
  { key: 'zmq', label: 'ZMQ', description: 'ZeroMQ messaging' },
  { key: 'ros2', label: 'ROS2', description: 'Robot Operating System' },
]

const protocolFields: Record<string, FieldDef[]> = {
  lsl: [
    { key: 'stream_name', label: 'Stream Name', type: 'text', placeholder: 'PredictionStream' },
    { key: 'stream_type', label: 'Stream Type', type: 'text', placeholder: 'PredictionStream' },
    { key: 'source_id', label: 'Source ID', type: 'text', placeholder: 'dendrite_default' },
  ],
  socket: [
    { key: 'protocol', label: 'Protocol', type: 'select', options: ['TCP', 'UDP'] },
    { key: 'ip', label: 'IP Address', type: 'text', placeholder: '127.0.0.1' },
    { key: 'port', label: 'Port', type: 'number', min: 1, max: 65535 },
  ],
  zmq: [
    { key: 'ip', label: 'IP Address', type: 'text', placeholder: '127.0.0.1' },
    { key: 'port', label: 'Port', type: 'number', min: 1, max: 65535 },
    { key: 'message_format', label: 'Format', type: 'select', options: ['JSON'] },
  ],
  ros2: [
    { key: 'topic_name', label: 'Topic Name', type: 'text', placeholder: 'bmi_predictions' },
    { key: 'node_name', label: 'Node Name', type: 'text', placeholder: 'bmi_prediction_node' },
  ],
}

const unavailableMessages: Record<string, string> = {
  zmq: 'pyzmq not installed',
  ros2: 'rclpy not installed',
}

function isAvailable(key: string): boolean {
  return config.outputAvailability[key] !== false
}

function isEnabled(key: string): boolean {
  return config.output[key]?.enabled || false
}

function getFieldValue(protoKey: string, fieldKey: string): any {
  return config.output[protoKey]?.config?.[fieldKey] ?? ''
}

function fieldError(protoKey: string, fieldKey: string): string | undefined {
  return config.outputErrors[protoKey]?.find(e => e.field === fieldKey)?.msg
}

function toggleProtocol(key: string) {
  if (!isAvailable(key)) return
  const current = config.output[key] || {}
  const wasEnabled = current.enabled || false
  const updated = { ...config.output }

  if (!wasEnabled && !current.config) {
    // First enable — populate from defaults
    updated[key] = {
      enabled: true,
      config: { ...(config.outputDefaults[key] || {}) },
    }
  } else {
    updated[key] = { ...current, enabled: !wasEnabled }
  }
  config.updateOutput(updated)
}

function updateField(protoKey: string, fieldKey: string, value: any) {
  const current = config.output[protoKey] || { enabled: true, config: {} }
  const updated = { ...config.output }
  updated[protoKey] = {
    ...current,
    config: { ...current.config, [fieldKey]: value },
  }
  config.updateOutput(updated)
}

function onTextChange(protoKey: string, fieldKey: string, event: Event) {
  const value = (event.target as HTMLInputElement).value
  updateField(protoKey, fieldKey, value)
}

function onSelectChange(protoKey: string, fieldKey: string, event: Event) {
  const value = (event.target as HTMLSelectElement).value
  updateField(protoKey, fieldKey, value)
}
</script>

<template>
  <div>
    <div class="space-y-2">
      <div
        v-for="proto in protocols"
        :key="proto.key"
        class="bg-bg-elevated rounded border border-border"
      >
        <!-- Header row -->
        <div class="flex items-center justify-between px-3 py-2.5">
          <div>
            <div class="text-sm text-text-main">{{ proto.label }}</div>
            <div class="text-xs text-text-muted">{{ proto.description }}</div>
            <div
              v-if="!isAvailable(proto.key)"
              class="text-xs text-status-warn mt-0.5"
            >
              {{ unavailableMessages[proto.key] || 'Not available' }}
            </div>
          </div>
          <button
            @click="toggleProtocol(proto.key)"
            :disabled="!isAvailable(proto.key)"
            class="w-10 h-5 rounded-full transition-colors relative"
            :class="[
              isEnabled(proto.key) ? 'bg-accent' : 'bg-border',
              !isAvailable(proto.key) ? 'opacity-40 cursor-not-allowed' : '',
            ]"
          >
            <div
              class="w-4 h-4 rounded-full bg-white absolute top-0.5 transition-transform"
              :class="isEnabled(proto.key) ? 'translate-x-5' : 'translate-x-0.5'"
            />
          </button>
        </div>

        <!-- Config fields (shown when enabled) -->
        <div
          v-if="isEnabled(proto.key)"
          class="px-3 pb-3 pt-1 border-t border-border space-y-2"
        >
          <div v-for="field in protocolFields[proto.key]" :key="field.key">
            <label class="block text-xs text-text-muted mb-0.5">{{ field.label }}</label>

            <input
              v-if="field.type === 'text'"
              type="text"
              :value="getFieldValue(proto.key, field.key)"
              :placeholder="field.placeholder"
              @change="onTextChange(proto.key, field.key, $event)"
              class="w-full text-sm"
              :class="fieldError(proto.key, field.key) ? 'border-status-error' : ''"
            />

            <NumberInput
              v-else-if="field.type === 'number'"
              :modelValue="getFieldValue(proto.key, field.key) || null"
              :min="field.min"
              :max="field.max"
              @update:modelValue="updateField(proto.key, field.key, $event)"
            />

            <select
              v-else-if="field.type === 'select'"
              :value="getFieldValue(proto.key, field.key)"
              @change="onSelectChange(proto.key, field.key, $event)"
              class="w-full text-sm"
            >
              <option v-for="opt in field.options" :key="opt" :value="opt">{{ opt }}</option>
            </select>

            <p
              v-if="fieldError(proto.key, field.key)"
              class="text-xs text-status-error mt-1"
            >
              {{ fieldError(proto.key, field.key) }}
            </p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
