export function formatDate(raw: string | null | undefined): string {
  if (!raw) return ''
  // Handle YYYYMMDD_HHMMSS format from session timestamps
  const m = raw.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/)
  const d = m ? new Date(+m[1]!, +m[2]! - 1, +m[3]!, +m[4]!, +m[5]!, +m[6]!) : new Date(raw)
  if (isNaN(d.getTime())) return raw
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
}

export function fileName(path: string): string {
  return path.split(/[/\\]/).pop() || path
}

export function formatPercent(v: number | null): string {
  if (v == null) return '-'
  return (v * 100).toFixed(1) + '%'
}

export function relativeTime(ts: string | null): string {
  if (!ts) return ''
  const diff = Date.now() - new Date(ts).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  return `${Math.floor(hrs / 24)}d ago`
}
