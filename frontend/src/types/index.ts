export interface HardwareProfile {
  id: string;
  name: string;
  category: string;
  vram_gb: number;
  memory_bandwidth_gbps: number;
  pcie_gen: number;
  pcie_lanes: number;
  pcie_bandwidth_gbps: number;
  system_ram_gb: number;
  disk_read_mbps: number;
  compute_tflops_fp16: number;
  created_at: string;
  updated_at: string;
}

export interface ModelSpec {
  id: string;
  name: string;
  family: string;
  parameter_count_billion: number;
  hidden_dim: number;
  num_layers: number;
  num_heads: number;
  num_kv_heads: number;
  vocab_size: number;
  max_seq_length: number;
  intermediate_dim: number;
  default_dtype: string;
  created_at: string;
}

export interface InferencePlan {
  id: string;
  model_id: string;
  hardware_id: string;
  quantization: string;
  batch_size: number;
  seq_len: number;
  layers_per_chunk: number;
  total_chunks: number;
  param_vram_mb: number;
  activation_vram_mb: number;
  kv_cache_vram_mb: number;
  total_vram_mb: number;
  vram_utilization_percent: number;
  est_prefill_sec: number;
  est_generation_sec: number;
  est_tokens_per_sec: number;
  created_at: string;
}

export interface LayerStreamStep {
  layer_idx: number;
  chunk_idx: number;
  stage: string;
  disk_read_ms: number;
  pcie_transfer_ms: number;
  compute_ms: number;
  vram_allocated_mb: number;
  description: string;
}

export interface BenchmarkSimulation {
  benchmark_id: string;
  total_layers: number;
  chunks_count: number;
  steps: LayerStreamStep[];
  final_tokens_per_sec: number;
  peak_vram_mb: number;
  numerical_check_passed: boolean;
}

export interface BenchmarkRun {
  id: string;
  model_id: string;
  hardware_id: string;
  plan_id?: string;
  quantization: string;
  layers_per_chunk: number;
  prompt_tokens: number;
  output_tokens: number;
  total_time_ms: number;
  tokens_per_sec: number;
  peak_vram_mb: number;
  disk_read_time_ms: number;
  pcie_transfer_time_ms: number;
  compute_time_ms: number;
  status: string;
  notes: string;
  created_at: string;
}

export interface CompatibilityAudit {
  id: string;
  target_model: string;
  transformers_version: string;
  torch_version: string;
  has_stateful_attribute: boolean;
  dynamic_cache_compatible: boolean;
  rope_dimension_aligned: boolean;
  device_compatible: boolean;
  issues: string[];
  recommendations: string[];
  generated_patch: string;
  created_at?: string;
}

export interface AdvisorAnalysis {
  provider: string;
  model: string;
  is_grounded: boolean;
  advisory_label: string;
  recommendations: string[];
  analysis_text: string;
  hardware_assessment: string;
}

export interface AuditLog {
  id: string;
  user_id: string;
  role: string;
  action: string;
  resource: string;
  details: string;
  timestamp: string;
}

export interface UserInfo {
  sub: string;
  email: string;
  roles: string[];
  is_demo: boolean;
}

export interface PatchVerificationTestItem {
  name: string;
  passed: boolean;
  category: string;
  detail: string;
}

export interface PatchVerificationResponse {
  id: string;
  target_model: string;
  all_passed: boolean;
  total_tests: number;
  passed_tests: number;
  failed_tests: number;
  results: PatchVerificationTestItem[];
  verified_at: string;
}

export interface PatchVerificationRequest {
  patch_code?: string;
  model_name: string;
  transformers_version: string;
  head_dim: number;
  rotary_dim: number;
  target_device: string;
}

