import { onUnmounted, watch, type Ref } from 'vue'
import uPlot from 'uplot'
import { useTheme } from './useTheme'

type OptsFactory = (dims: { width: number; height: number }) => uPlot.Options

/**
 * Wraps uPlot's create/resize/destroy lifecycle.
 *
 * Rebuilds on theme change: uPlot caches series/axis colors at construction,
 * so when CSS color tokens flip we recreate the chart with the last data.
 */
export function useUPlot(target: Ref<HTMLDivElement | null>) {
  let plot: uPlot | null = null
  let resizeObs: ResizeObserver | null = null
  let currentFactory: OptsFactory | null = null

  function create(optsFactory: OptsFactory, data: uPlot.AlignedData) {
    if (!target.value) return
    destroy()
    currentFactory = optsFactory
    const width = target.value.clientWidth || 400
    const height = target.value.clientHeight || 300
    const opts = optsFactory({ width, height })
    plot = new uPlot(opts, data, target.value)
    resizeObs = new ResizeObserver(() => {
      if (!target.value || !plot || !currentFactory) return
      const w = target.value.clientWidth
      const h = target.value.clientHeight || 300
      const newOpts = currentFactory({ width: w, height: h })
      plot.setSize({ width: w, height: newOpts.height as number })
    })
    resizeObs.observe(target.value)
  }

  function setData(data: uPlot.AlignedData) {
    plot?.setData(data)
  }

  function destroy() {
    plot?.destroy()
    plot = null
    resizeObs?.disconnect()
    resizeObs = null
    currentFactory = null
  }

  function getPlot() { return plot }

  const { theme } = useTheme()
  watch(theme, () => {
    if (!plot || !currentFactory) return
    const factory = currentFactory
    const data = plot.data as uPlot.AlignedData
    create(factory, data)
  })

  onUnmounted(destroy)

  return { create, setData, destroy, getPlot }
}
