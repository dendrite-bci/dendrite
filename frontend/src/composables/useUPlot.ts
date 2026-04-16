import { onUnmounted, type Ref } from 'vue'
import uPlot from 'uplot'

type OptsFactory = (dims: { width: number; height: number }) => uPlot.Options

/**
 * Wraps uPlot's create/resize/destroy lifecycle.
 *
 * Pass the template ref (declared by the caller with `ref()`) as `target`.
 * The caller owns the ref so template-ref usage is visible to vue-tsc's
 * unused-variable check.
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

  onUnmounted(destroy)

  return { create, setData, destroy, getPlot }
}
