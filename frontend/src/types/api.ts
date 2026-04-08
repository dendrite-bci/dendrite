export interface PipelineStatus {
  recording: boolean
  recording_id: number | null
  elapsed_seconds: number
  log_file: string | null
  mode_pids: Record<string, number>
  system_pids: Record<string, number>
  component_states: Record<string, string>
}

export interface GeneralConfig {
  study_name: string
  subject_id: string
  session_id: string
  recording_name: string
}

export interface StreamMetadata {
  uid: string
  name: string
  type: string
  channel_count: number
  sample_rate: number
  channel_format: string
  source_id: string
  labels: string[]
  channel_types: string[]
  channel_units: string[]
  stream_key: string
  has_metadata_issues: boolean
  metadata_issues: Record<string, any>
}

export interface ModalityChannel {
  label: string
  local_index: number
}

export interface StreamModalities {
  stream_name: string
  stream_type: string
  stream_key: string
  sample_rate: number
  modalities: Record<string, ModalityChannel[]>
}

export interface ModeInstance {
  name: string
  mode: string
  channel_selection?: Record<string, number[]>
  source_stream?: string
  decoder_config?: Record<string, any>
  event_mapping?: Record<number, string>
  mode_preprocessing?: Record<string, Record<string, any>>
  [key: string]: any
}

export interface ConfigFile {
  file_path: string
  study_name: string
  file_name: string
  modified: number
  size: number
}

export interface PreflightCheck {
  id: string
  label: string
  passed: boolean
  required: boolean
  detail: string | null
}

export interface PreflightResult {
  ready: boolean
  checks: PreflightCheck[]
}

// --- Data Explorer ---

export interface Study {
  study_id: number
  study_name: string
  description: string | null
  created_at: string
  recording_count?: number
  decoder_count?: number
}

export interface StudyDetail extends Study {
  recording_count: number
  decoder_count: number
}

export interface Recording {
  recording_id: number
  study_id: number
  study_name: string
  recording_name: string
  subject_id: string
  session_id: string
  run_number: number
  session_timestamp: string
  hdf5_file_path: string
  created_at: string
}

export interface Decoder {
  decoder_id: number
  study_id: number
  study_name: string
  decoder_name: string
  decoder_path: string
  model_type: string
  description: string | null
  num_classes: number | null
  training_accuracy: number | null
  validation_accuracy: number | null
  training_recording_ids: string | null
  created_at: string
}

export interface DecoderMetadata {
  model_type: string
  num_classes: number
  // Mapping
  event_mapping?: Record<string, string>
  label_mapping?: Record<string, number>
  // Input spec
  sample_rate?: number
  input_shapes?: Record<string, number[]>
  channel_labels?: Record<string, string[]>
  modality?: string
  epoch_tmin?: number
  epoch_tmax?: number
  // Training
  epochs?: number
  batch_size?: number
  learning_rate?: number
  optimizer_type?: string
  weight_decay?: number
  validation_split?: number
  seed?: number
  loss_type?: string
  label_smoothing_factor?: number
  // Early stopping
  use_early_stopping?: boolean
  early_stopping_patience?: number
  // Augmentation
  use_augmentation?: boolean
  aug_strategy?: string
  use_class_weights?: boolean
  mixup_alpha?: number
  // LR scheduling
  use_lr_scheduler?: boolean
  lr_scheduler_type?: string
  // SWA
  use_swa?: boolean
  swa_start_epoch?: number
  swa_lr?: number
  // Architecture
  model_params?: Record<string, any>
  pipeline_steps?: string[]
  // Preprocessing
  preprocessing_config?: {
    modality_preprocessing?: Record<string, {
      lowcut?: number
      highcut?: number
      apply_rereferencing?: boolean
    }>
  }
  // Provenance
  training_recording_ids?: number[]
  training_file_identifier?: string
}

export interface H5FileInfo {
  datasets: Record<string, { shape: number[]; dtype: string; attributes: Record<string, any> }>
  groups: Record<string, { attributes: Record<string, any>; children: string[] }>
  root_attributes: Record<string, any>
  file_size_mb: number
  error?: string
}

export interface ChannelInfo {
  labels: string[]
  count: number
  n_samples: number
  sample_rate?: number
}

// --- Signal Preview & Events ---

export interface SignalChannel {
  label: string
  data: number[]
}

export interface ModalitySignalPreview {
  time: number[]
  channels: SignalChannel[]
  sample_rate: number
  total_samples: number
  display_samples: number
}

export type SignalPreview = Record<string, ModalitySignalPreview>

export interface EventSummary {
  total_count: number
  event_types: Record<string, number>
  event_ids: Record<string, number>
  events: Record<string, any>[]
}

// --- ERP Preview ---

export interface ERPEventData {
  channels: number[][]
  labels: string[]
  count: number
}

export interface ERPPreview {
  erp_by_event: Record<string, ERPEventData>
  time_axis: number[]
  sample_rate: number
  n_epochs: number
  event_counts: Record<string, number>
  epoch_tmin: number
  epoch_tmax: number
}

// --- QC Preview ---

export interface QCChannelData {
  label: string
  data: number[]
  is_bad: boolean
}

export interface QCSignalGroup {
  time: number[]
  channels: QCChannelData[]
}

export interface QCChannelQuality {
  index: number
  label: string
  status: 'good' | 'warning' | 'bad'
  variance: number
  std: number
  max_deriv: number
}

export interface QCPreview {
  raw: QCSignalGroup
  preprocessed: QCSignalGroup
  quality: {
    channels: QCChannelQuality[]
    bad_channels: number[]
  }
  sample_rate: number
  total_samples: number
  total_channels: number
  display_samples: number
  channel_indices: number[]
  preprocessing: {
    lowcut: number
    highcut: number
    apply_rereferencing: boolean
    bad_channel_mode: string
  }
}

// --- ML Workbench ---

export interface ModelInfo {
  model_type: string
  description: string
  modalities: string[]
  default_steps: string[]
  step_types: Record<string, string>  // step_name → 'preprocessing' | 'features' | 'classifier'
}

export interface TrainingJob {
  job_id: number
  study_id: number | null
  dataset_id: number | null
  model_type: string
  job_type: 'training' | 'evaluation' | 'benchmark'
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  config_json: string
  result_json: string | null
  decoder_id: number | null
  error_message: string | null
  started_at: string | null
  completed_at: string | null
  created_at: string
  progress?: TrainingProgress
}

export interface TrainingProgress {
  type: 'epoch' | 'complete' | 'failed' | 'started' | 'cancelled'
    | 'optuna_trial' | 'optuna_complete' | 'eval_step' | 'eval_metrics'
    | 'bench_model_complete' | 'data_loading_step'
  job_id: number
  epoch?: number
  total_epochs?: number
  train_loss?: number
  train_acc?: number
  val_loss?: number
  val_acc?: number
  elapsed_seconds?: number
  result?: Record<string, any>
  error?: string
  // Optuna
  trial?: number
  total_trials?: number
  model_type?: string
  val_accuracy?: number
  accuracy?: number
  best_trial?: number
  best_accuracy?: number
  // Evaluation
  step?: number
  total_steps?: number
  prediction?: number
  true_label?: number
  confidence?: number
  correct?: boolean
  metrics?: Record<string, any>
}

// --- ML Data Loading ---

export interface MoabbDataset {
  code: string
  name: string
  paradigm: string
  n_subjects: number
  subjects: number[]
  events: Record<string, number>
  interval: number[] | null
  paradigm_bandpass: number[] | null
}

export interface LoadedDataInfo {
  source: 'moabb' | 'recording'
  source_id: string
  sample_rate: number
  channel_names: string[]
  channel_types: string[]
  n_samples: number
  n_channels: number
  n_times: number
  shape: number[]
  metadata: {
    paradigm: string
    class_names: (string | number)[]
    class_counts: Record<string, number>
    label_map: Record<string, number>
    n_channels: number
    n_times: number
    subject?: number
    dataset_code?: string
    dataset_id?: number
    dataset_name?: string
    event_id?: Record<string, number>
    [key: string]: any
  }
}

// --- Recording Metrics Views ---

export interface SessionSummary {
  duration_seconds: number
  sample_rate: number
  channels: number
  datasets: string[]
  modes: string[]
  has_metrics: boolean
}

export interface MetricSeries {
  time: number[]
  values: number[]
}

export interface RecordingTelemetry {
  latencies: Record<string, MetricSeries>
  mode_metrics: Record<string, MetricSeries>
  bandwidth: Record<string, MetricSeries>
}

export type ModePerformance = Record<string, Record<string, MetricSeries>>

export interface TelemetryData {
  type: string
  timestamp: number
  elapsed_s: number
  streams: Array<{
    type: string
    latency_ms: number
    last_update: number | null
  }>
  modes: Array<{
    name: string
    accuracy?: number
    confidence?: number
    kappa?: number
    internal_ms?: number
    inference_ms?: number
  }>
  system: {
    cpu_percent: number
    memory_percent: number
    memory_used_gb: number
    memory_total_gb: number
    processes: Array<{
      name: string
      pid: number
      cpu_percent: number
      memory_mb: number
    }>
  }
  visualization: {
    bandwidth_kbps?: number
    consumers?: number
  }
  channel_quality?: {
    channels: Array<{ index: number; status: string; variance: number }>
    bad_channels: Record<string, number[]>
    manual_flags?: Record<string, number[]>
    manual_unflagged?: Record<string, number[]>
    effective_bad?: Record<string, number[]>
    interp_version?: number
  }
}

// --- Output Protocols ---

export type ProtocolAvailability = Record<string, boolean>

export interface ProtocolFieldError {
  field: string
  msg: string
}
