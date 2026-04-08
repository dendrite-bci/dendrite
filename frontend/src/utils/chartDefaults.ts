import type uPlot from 'uplot'

// --- Theme-matched chart constants ---
export const CHART_FONT = "10px 'Inter', sans-serif"
export const GRID_STROKE = '#1e1e1e'
export const AXIS_STROKE = '#808080'

// --- Axis factory ---
export function makeAxis(overrides?: Partial<uPlot.Axis>): uPlot.Axis {
  return {
    stroke: AXIS_STROKE,
    grid: { stroke: GRID_STROKE, width: 1 },
    font: CHART_FONT,
    size: 30,
    ...overrides,
  }
}

// --- Cursor presets ---
export const CURSOR_HIDDEN: uPlot.Cursor = { show: false }
export const CURSOR_INTERACTIVE: uPlot.Cursor = { show: true, points: { show: false } }

// --- Legend presets ---
export const LEGEND_HIDDEN = { show: false } as const
export const LEGEND_SHOWN = { show: true, live: false } as const

// --- Percent axis value formatter ---
export const pctValues: uPlot.Axis.Values = (_, ticks) =>
  ticks.map(v => (v * 100).toFixed(0) + '%')
