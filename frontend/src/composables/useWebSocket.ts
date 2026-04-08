import { ref } from 'vue'
import { decode } from 'msgpackr'

export function useWebSocket(path: string, options: {
  binary?: boolean
  onMessage?: (data: any) => void
  onClose?: () => void
  reconnectInterval?: number
} = {}) {
  const connected = ref(false)
  const error = ref<string | null>(null)

  let ws: WebSocket | null = null

  const reconnectMs = options.reconnectInterval ?? 3000

  function connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    const url = `${protocol}//${host}${path}`

    ws = new WebSocket(url)

    ws.binaryType = 'arraybuffer'

    ws.onopen = () => {
      connected.value = true
      error.value = null
    }

    ws.onmessage = (event: MessageEvent) => {
      if (!options.onMessage) return

      // queueMicrotask breaks the synchronous call stack during bursts
      // (e.g. 200 history messages on connect) so the browser can paint.
      if (options.binary && event.data instanceof ArrayBuffer) {
        const bytes = new Uint8Array(event.data)
        queueMicrotask(() => options.onMessage!(decode(bytes)))
      } else if (typeof event.data === 'string') {
        const text = event.data
        queueMicrotask(() => options.onMessage!(JSON.parse(text)))
      }
    }

    ws.onclose = () => {
      connected.value = false
      options.onClose?.()
      setTimeout(connect, reconnectMs)
    }

    ws.onerror = () => {
      error.value = 'WebSocket connection error'
      ws?.close()
    }
  }

  connect()

  return { connected, error }
}
