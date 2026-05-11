import { useToast } from '../composables/useToast'

export class ApiError extends Error {
  status: number
  detail: any

  constructor(status: number, detail: any) {
    const msg = typeof detail === 'string' ? detail : detail?.message ?? `HTTP ${status}`
    super(msg)
    this.status = status
    this.detail = detail
  }
}

type ApiFetchOptions = Omit<RequestInit, 'body'> & {
  silent?: boolean
  json?: unknown
  body?: BodyInit | null
  fallbackMessage?: string
}

/**
 * Thin wrapper around fetch() for API calls.
 *
 * - Pass `json` to auto-serialize a JSON body and set Content-Type.
 * - Checks res.ok and parses error detail from the response body.
 * - Shows a toast on failure (unless `silent: true`); `fallbackMessage`
 *   overrides the default "Request failed" text when the server didn't
 *   return a usable `detail`.
 * - Returns the parsed JSON body (or null for 204 No Content).
 * - Throws ApiError on failure so callers can catch for custom handling.
 */
export async function apiFetch<T = any>(url: string, options?: ApiFetchOptions): Promise<T> {
  const { silent, json, headers, body, fallbackMessage, ...rest } = options ?? {}

  const init: RequestInit = {
    ...rest,
    headers: json !== undefined
      ? { 'Content-Type': 'application/json', ...headers }
      : headers,
    body: json !== undefined ? JSON.stringify(json) : body,
  }

  let res: Response
  try {
    res = await fetch(url, init)
  } catch (e) {
    const netMsg = e instanceof Error && e.message ? e.message : 'Network error'
    if (!silent) useToast().error(fallbackMessage ?? netMsg)
    throw new ApiError(0, netMsg)
  }

  if (!res.ok) {
    let detail: any = `HTTP ${res.status}`
    try {
      const parsed = await res.json()
      detail = parsed.detail ?? parsed
    } catch {
      // Response body not JSON — use status text
    }
    if (!silent) {
      const fromDetail =
        typeof detail === 'string'
          ? detail
          : Array.isArray(detail)
            ? detail.map((e: any) => `${(e.loc ?? []).join('.')}: ${e.msg}`).join('; ')
            : detail?.message
      useToast().error(fromDetail ?? fallbackMessage ?? 'Request failed')
    }
    throw new ApiError(res.status, detail)
  }

  if (res.status === 204) return null as T
  return res.json()
}

/**
 * Non-throwing variant: returns null on failure, never shows a toast.
 * Use for background fetches and init paths where a silent no-op is the
 * desired failure mode.
 */
export async function apiFetchOrNull<T = any>(
  url: string,
  options?: Omit<ApiFetchOptions, 'silent'>,
): Promise<T | null> {
  try {
    return await apiFetch<T>(url, { ...options, silent: true })
  } catch {
    return null
  }
}
