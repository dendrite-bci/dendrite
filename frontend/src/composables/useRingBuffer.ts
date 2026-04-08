/**
 * Ring buffer with double-write trick for O(1) chronological reads.
 * Port of Python NumpyRingBuffer from v1 dashboard.
 */
export class RingBuffer {
  private data: Float64Array
  private capacity: number
  private writePos = 0
  private count = 0

  constructor(capacity: number, fillValue = 0.0) {
    this.capacity = capacity
    this.data = new Float64Array(2 * capacity)
    if (fillValue !== 0.0) {
      this.data.fill(fillValue)
    }
  }

  append(value: number): void {
    this.data[this.writePos] = value
    this.data[this.writePos + this.capacity] = value
    this.writePos = (this.writePos + 1) % this.capacity
    if (this.count < this.capacity) this.count++
  }

  /** Append multiple values at once (avoids per-sample overhead). */
  appendBatch(values: ArrayLike<number>, count?: number): void {
    const n = count ?? values.length
    for (let j = 0; j < n; j++) {
      this.data[this.writePos] = values[j]!
      this.data[this.writePos + this.capacity] = values[j]!
      this.writePos = (this.writePos + 1) % this.capacity
      if (this.count < this.capacity) this.count++
    }
  }

  /** Returns a contiguous view of the buffer in chronological order. */
  asArray(): Float64Array {
    if (this.count < this.capacity) {
      return this.data.subarray(0, this.count)
    }
    return this.data.subarray(this.writePos, this.writePos + this.capacity)
  }

  /** Copy to a plain number[] for uPlot (which needs regular arrays). */
  toArray(): number[] {
    const view = this.asArray()
    const out = new Array(view.length)
    for (let i = 0; i < view.length; i++) out[i] = view[i]
    return out
  }

  get length(): number {
    return this.count
  }

  get isFull(): boolean {
    return this.count >= this.capacity
  }

  clear(): void {
    this.writePos = 0
    this.count = 0
    this.data.fill(0)
  }
}
