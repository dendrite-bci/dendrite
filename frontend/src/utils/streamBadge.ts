const BADGE_MAP: Record<string, string> = {
  EEG: 'text-accent/70',
  EMG: 'text-status-ok/70',
  MARKERS: 'text-status-error/70',
  EVENTS: 'text-status-error/70',
}

const AUTO_PALETTE: string[] = [
  'text-text-muted',
  'text-level-ok/70',
  'text-level-warn/70',
  'text-data-eval/70',
]

function hashIndex(s: string): number {
  let h = 0
  for (let i = 0; i < s.length; i++) h = ((h << 5) - h + s.charCodeAt(i)) | 0
  return ((h % AUTO_PALETTE.length) + AUTO_PALETTE.length) % AUTO_PALETTE.length
}

/** Badge class for stream/source type labels. Pair with `typeBadgeBase` for layout. */
export function typeBadgeClass(t: string): string {
  const key = t?.toUpperCase() ?? ''
  return BADGE_MAP[key] ?? AUTO_PALETTE[hashIndex(key)]!
}

/** Shared structural classes for type badge layout: right divider + spacing. */
export const typeBadgeBase = 'pr-2.5 mr-2.5 border-r border-border text-xs font-medium uppercase tracking-wider'
