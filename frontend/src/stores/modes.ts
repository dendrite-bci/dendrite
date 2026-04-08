import { defineStore } from 'pinia'
import { ref, computed, watch } from 'vue'
import type { ModeInstance } from '../types/api'
import { usePipelineStore } from './pipeline'

export const useModesStore = defineStore('modes', () => {
  const instances = ref<Record<string, ModeInstance>>({})
  const modeActionLoading = ref<Record<string, boolean>>({})

  const instanceNames = computed(() => Object.keys(instances.value))
  const instanceCount = computed(() => instanceNames.value.length)

  const pipeline = usePipelineStore()

  const modeStates = computed(() => {
    const states: Record<string, string> = {}
    for (const name of instanceNames.value) {
      states[name] = pipeline.status.component_states?.[`mode:${name}`] ?? 'idle'
    }
    return states
  })

  async function fetchAll() {
    const res = await fetch('/api/modes')
    if (!res.ok) return
    const data = await res.json()
    instances.value = data.instances
  }

  async function addInstance(mode: string, config: Record<string, any> = {}, name?: string) {
    const res = await fetch('/api/modes', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, mode, config }),
    })
    if (res.ok) {
      const saved = await res.json()
      if (saved.instances) {
        instances.value = saved.instances
      } else {
        await fetchAll()
      }
    }
    return res.ok
  }

  async function updateInstance(name: string, mode: string, config: Record<string, any>): Promise<{ data?: any; error?: string }> {
    const res = await fetch(`/api/modes/${encodeURIComponent(name)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, mode, config }),
    })
    if (res.ok) {
      const saved = await res.json()
      instances.value[name] = { ...instances.value[name], ...saved, ...config }
      instances.value = { ...instances.value }
      return { data: saved }
    }
    const err = await res.json().catch(() => ({}))
    return { error: typeof err.detail === 'string' ? err.detail : `Save failed (${res.status})` }
  }

  async function removeInstance(name: string) {
    const res = await fetch(`/api/modes/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    })
    if (res.ok) {
      const { [name]: _, ...rest } = instances.value
      instances.value = rest
    }
    return res.ok
  }

  async function renameInstance(oldName: string, newName: string) {
    const res = await fetch(`/api/modes/${encodeURIComponent(oldName)}/rename`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ new_name: newName }),
    })
    if (res.ok) {
      const inst = instances.value[oldName]
      if (!inst) return
      const { [oldName]: _, ...rest } = instances.value
      instances.value = { ...rest, [newName]: { ...inst, name: newName } }
    }
    return res.ok
  }

  async function toggleInstance(name: string) {
    const inst = instances.value[name]
    if (!inst) return false
    const newEnabled = !(inst.enabled ?? true)
    return updateInstance(name, inst.mode, { ...inst, enabled: newEnabled })
  }

  async function cloneInstance(name: string) {
    const inst = instances.value[name]
    if (!inst) return false
    let cloneName = `${name}_copy`
    let i = 2
    while (instances.value[cloneName]) {
      cloneName = `${name}_copy${i++}`
    }
    const { mode, name: _oldName, ...config } = JSON.parse(JSON.stringify(inst))
    return addInstance(mode, config, cloneName)
  }

  async function _extractError(res: Response, fallback: string): Promise<string> {
    const data = await res.json()
    return typeof data.detail === 'string' ? data.detail : fallback
  }

  async function _modeAction(name: string, endpoint: string, errorMsg: string, ...targets: string[]) {
    modeActionLoading.value[name] = true
    pipeline.error = null
    try {
      const res = await fetch(`/api/modes/${encodeURIComponent(name)}/${endpoint}`, { method: 'POST' })
      if (!res.ok) {
        pipeline.error = await _extractError(res, errorMsg)
        modeActionLoading.value[name] = false
        return false
      }
      await pipeline.fetchStatus()
      _clearLoadingOnTransition(name, ...targets)
      return true
    } catch (e: any) {
      pipeline.error = e.message
      modeActionLoading.value[name] = false
      return false
    }
  }

  const startMode = (name: string) => _modeAction(name, 'start', 'Failed to start mode', 'running')
  const stopMode = (name: string) => _modeAction(name, 'stop', 'Failed to stop mode', 'idle', 'stopped')

  function _clearLoadingOnTransition(name: string, ...targets: string[]) {
    const current = modeStates.value[name] ?? 'idle'
    if (targets.includes(current) || current === 'error') {
      modeActionLoading.value[name] = false
      return
    }

    const timeout = setTimeout(() => {
      modeActionLoading.value[name] = false
      stop()
    }, 10_000)

    const stop = watch(modeStates, (states) => {
      const state = states[name] ?? 'idle'
      if (targets.includes(state) || state === 'error') {
        modeActionLoading.value[name] = false
        clearTimeout(timeout)
        stop()
      }
    })
  }

  fetchAll()

  return {
    instances, instanceNames, instanceCount, modeStates, modeActionLoading,
    fetchAll, addInstance, updateInstance, removeInstance, renameInstance,
    toggleInstance, cloneInstance, startMode, stopMode,
  }
})
