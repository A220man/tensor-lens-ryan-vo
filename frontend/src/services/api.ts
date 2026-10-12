import {
  HardwareProfile,
  ModelSpec,
  InferencePlan,
  BenchmarkSimulation,
  BenchmarkRun,
  CompatibilityAudit,
  AdvisorAnalysis,
  AuditLog,
  UserInfo,
} from '../types';

let currentAccessToken: string | null = null;

export const setAuthToken = (token: string | null): void => {
  currentAccessToken = token;
  if (token) {
    sessionStorage.setItem('tensor_lens_session_token', token);
  } else {
    sessionStorage.removeItem('tensor_lens_session_token');
  }
};

export const getStoredToken = (): string | null => {
  if (!currentAccessToken) {
    currentAccessToken = sessionStorage.getItem('tensor_lens_session_token');
  }
  return currentAccessToken;
};

const request = async <T>(endpoint: string, options: RequestInit = {}): Promise<T> => {
  const token = getStoredToken();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string>),
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const response = await fetch(endpoint, {
    ...options,
    headers,
  });

  if (!response.ok) {
    let errorDetail = `Request failed with status ${response.status}`;
    try {
      const errorJson = await response.json();
      if (errorJson.detail) {
        errorDetail = errorJson.detail;
      }
    } catch {
      // Use status code string if json parsing fails
    }
    throw new Error(errorDetail);
  }

  return response.json();
};

export const api = {
  // Health
  getHealth: () => request<{ status: string; environment: string; demo_mode: boolean }>('/api/health'),

  // Auth
  getDemoToken: (role: string = 'analyst') =>
    request<{ access_token: string; token_type: string; role: string }>('/api/auth/demo-token', {
      method: 'POST',
      body: JSON.stringify({ role }),
    }),

  getUserInfo: () => request<UserInfo>('/api/auth/userinfo'),

  // Hardware Profiles
  getHardwareProfiles: () => request<HardwareProfile[]>('/api/hardware'),
  createHardwareProfile: (data: Partial<HardwareProfile>) =>
    request<HardwareProfile>('/api/hardware', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  // Model Specs
  getModels: () => request<ModelSpec[]>('/api/models'),
  createModel: (data: Partial<ModelSpec>) =>
    request<ModelSpec>('/api/models', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  // Memory Planner
  calculatePlan: (params: {
    model_id: string;
    hardware_id: string;
    quantization: string;
    batch_size: number;
    seq_len: number;
    manual_layers_per_chunk?: number;
  }) =>
    request<InferencePlan>('/api/planner/calculate', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  getPlans: () => request<InferencePlan[]>('/api/planner/plans'),

  // Benchmarking & Simulation
  simulateStream: (params: {
    model_id: string;
    hardware_id: string;
    quantization: string;
    layers_per_chunk: number;
    prompt_tokens: number;
    output_tokens: number;
    run_numerical_simulation?: boolean;
  }) =>
    request<BenchmarkSimulation>('/api/benchmark/simulate', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  recordBenchmarkRun: (params: {
    model_id: string;
    hardware_id: string;
    quantization: string;
    layers_per_chunk: number;
    prompt_tokens: number;
    output_tokens: number;
    run_numerical_simulation?: boolean;
  }) =>
    request<BenchmarkRun>('/api/benchmark/run', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  getBenchmarkHistory: () => request<BenchmarkRun[]>('/api/benchmark/history'),

  // Compatibility Diagnostics
  auditCompatibility: (params: {
    model_name: string;
    transformers_version: string;
    torch_version: string;
    target_device: string;
    head_dim: number;
    rotary_dim: number;
  }) =>
    request<CompatibilityAudit>('/api/compatibility/audit', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  // LLM Advisory
  requestAdvisory: (params: { benchmark_id?: string; plan_id?: string; user_query?: string }) =>
    request<AdvisorAnalysis>('/api/advisor/analyze', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  // Audit Logs
  getAuditLogs: () => request<AuditLog[]>('/api/audit-logs'),
};
