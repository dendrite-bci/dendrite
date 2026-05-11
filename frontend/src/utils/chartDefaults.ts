import type uPlot from 'uplot'

export const CHART_FONT = "10px 'Inter', sans-serif"

export function cssVar(name: string, fallback: string): string {
  if (typeof document === 'undefined') return fallback
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback
}

export const getMutedStroke = () => cssVar('--color-text-muted', '#a8a8a8')
export const getDisabledStroke = () => cssVar('--color-text-disabled', '#787878')
export const getBorderStroke = () => cssVar('--color-border', '#2a2a2a')

export function makeAxis(overrides?: Partial<uPlot.Axis>): uPlot.Axis {
  return {
    stroke: cssVar('--color-text-disabled', '#808080'),
    grid: { stroke: cssVar('--color-border', '#1e1e1e'), width: 1 },
    font: CHART_FONT,
    size: 30,
    ...overrides,
  }
}

export const CURSOR_HIDDEN: uPlot.Cursor = { show: false }
export const CURSOR_INTERACTIVE: uPlot.Cursor = { show: true, points: { show: false } }

export const LEGEND_HIDDEN = { show: false } as const
export const LEGEND_SHOWN = { show: true, live: false } as const

export const pctValues: uPlot.Axis.Values = (_, ticks) =>
  ticks.map(v => (v * 100).toFixed(0) + '%')
