export interface PlotSeries {
  label: string
  time: number[]
  values: number[]
  color?: string
}

const UNIT_SUFFIXES: [RegExp, string][] = [
  [/_ms$/, ' (ms)'],
  [/_kbps$/, ' (kbps)'],
  [/_mb$/, ' (MB)'],
  [/_hz$/, ' (Hz)'],
]

export function prettifyLabel(raw: string): string {
  let label = raw

  // Extract unit suffix
  let unit = ''
  for (const [re, u] of UNIT_SUFFIXES) {
    if (re.test(label)) {
      label = label.replace(re, '')
      unit = u
      break
    }
  }

  // Replace underscores with spaces, capitalize each word
  label = label
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase())
    .trim()

  return label + unit
}

export function metricsToPlotSeries(
  metrics: Record<string, { time: number[]; values: number[] }>
): PlotSeries[] {
  return Object.entries(metrics).map(([key, m]) => ({
    label: prettifyLabel(key),
    time: m.time,
    values: m.values,
  }))
}
