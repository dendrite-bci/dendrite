export const CHART_COLORS = [
  '#40E8C0', '#6366f1', '#f59e0b', '#ef4444',
  '#22c55e', '#8b5cf6', '#ec4899', '#14b8a6',
]

const MODALITY_COLORS: Record<string, string> = {
  eeg: '#40E8C0',
  emg: '#F5A623',
  eog: '#38BDF8',
  ecg: '#F472B6',
  misc: '#A78BFA',
}

const FALLBACK_POOL = ['#14b8a6', '#f97316', '#7c3aed', '#ef4444', '#06b6d4', '#84cc16']

/** Deterministic color for any modality — known ones get fixed colors, unknowns hash into a pool. */
export function getModalityColor(modality: string): string {
  if (modality in MODALITY_COLORS) return MODALITY_COLORS[modality]!
  const hash = Array.from(modality).reduce((h, c) => h + c.charCodeAt(0), 0)
  return FALLBACK_POOL[hash % FALLBACK_POOL.length]!
}

export const MODE_COLORS: Record<string, string> = {
  synchronous: '#5b9cf0',
  asynchronous: '#f0715b',
  neurofeedback: '#a78bfa',
}

export function getModeColor(mode: string): string {
  return MODE_COLORS[mode] ?? '#5b9cf0'
}

const PRED_CLASS_COLORS = [
  '#34d399', '#fbbf24', '#60a5fa', '#f472b6', '#a78bfa',
  '#fb923c', '#38bdf8', '#e879f9', '#4ade80', '#f87171',
]

/** Deterministic color for a prediction class name — same class always gets same color. */
export function predClassColor(className: string): string {
  const hash = Array.from(className).reduce((h, c) => h + c.charCodeAt(0), 0)
  return PRED_CLASS_COLORS[hash % PRED_CLASS_COLORS.length]!
}