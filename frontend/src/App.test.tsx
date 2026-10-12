import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import App from './App';
import { AuthProvider } from './context/AuthContext';

// Mock the API client
vi.mock('./services/api', () => {
  return {
    setAuthToken: vi.fn(),
    getStoredToken: () => 'mock-jwt-token',
    api: {
      getHealth: vi.fn().mockResolvedValue({ status: 'healthy', environment: 'development', demo_mode: true }),
      getUserInfo: vi.fn().mockResolvedValue({
        sub: 'demo-user-analyst',
        email: 'demo-analyst@tensorlens.local',
        roles: ['analyst'],
        is_demo: true,
      }),
      getDemoToken: vi.fn().mockResolvedValue({
        access_token: 'mock-token',
        token_type: 'bearer',
        role: 'analyst',
      }),
      getHardwareProfiles: vi.fn().mockResolvedValue([
        {
          id: 'gpu-gtx-1650-4gb',
          name: 'NVIDIA GeForce GTX 1650 (4GB)',
          category: 'gpu',
          vram_gb: 4.0,
          memory_bandwidth_gbps: 128.0,
          pcie_gen: 3,
          pcie_lanes: 16,
          pcie_bandwidth_gbps: 15.75,
          system_ram_gb: 16.0,
          disk_read_mbps: 550.0,
          compute_tflops_fp16: 5.9,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        },
      ]),
      getModels: vi.fn().mockResolvedValue([
        {
          id: 'model-llama3-70b',
          name: 'Meta Llama-3-70B-Instruct',
          family: 'llama',
          parameter_count_billion: 70.6,
          hidden_dim: 8192,
          num_layers: 80,
          num_heads: 64,
          num_kv_heads: 8,
          vocab_size: 128256,
          max_seq_length: 8192,
          intermediate_dim: 28672,
          default_dtype: 'bfloat16',
          created_at: new Date().toISOString(),
        },
      ]),
      calculatePlan: vi.fn().mockResolvedValue({
        id: 'plan-12345',
        model_id: 'model-llama3-70b',
        hardware_id: 'gpu-gtx-1650-4gb',
        quantization: 'int4',
        batch_size: 1,
        seq_len: 512,
        layers_per_chunk: 2,
        total_chunks: 40,
        param_vram_mb: 980.5,
        activation_vram_mb: 128.0,
        kv_cache_vram_mb: 160.0,
        total_vram_mb: 1780.5,
        vram_utilization_percent: 43.5,
        est_prefill_sec: 24.2,
        est_generation_sec: 1.15,
        est_tokens_per_sec: 0.87,
        created_at: new Date().toISOString(),
      }),
      simulateStream: vi.fn().mockResolvedValue({
        benchmark_id: 'sim-99',
        total_layers: 80,
        chunks_count: 40,
        steps: [
          {
            layer_idx: 0,
            chunk_idx: 0,
            stage: 'Layers 0..1',
            disk_read_ms: 85.0,
            pcie_transfer_ms: 42.0,
            compute_ms: 12.0,
            vram_allocated_mb: 1780.5,
            description: 'Prefetched 2 layers, executed forward pass',
          },
        ],
        final_tokens_per_sec: 0.87,
        peak_vram_mb: 1780.5,
        numerical_check_passed: true,
      }),
      getBenchmarkHistory: vi.fn().mockResolvedValue([]),
      auditCompatibility: vi.fn().mockResolvedValue({
        id: 'audit-1',
        target_model: 'Meta-Llama-3-70B-Instruct',
        transformers_version: '4.45.2',
        torch_version: '2.4.1',
        has_stateful_attribute: false,
        dynamic_cache_compatible: false,
        rope_dimension_aligned: true,
        device_compatible: true,
        issues: ['Transformers >= 4.40 requires _is_stateful'],
        recommendations: ['Inject _is_stateful = False'],
        generated_patch: '# Patch: _is_stateful = False',
      }),
      getAuditLogs: vi.fn().mockResolvedValue([]),
    },
  };
});

describe('TensorLens Application Workflows', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders application brand, header, and default memory planner view', async () => {
    render(
      <AuthProvider>
        <App />
      </AuthProvider>
    );

    expect(screen.getByText('TensorLens')).toBeInTheDocument();
    expect(screen.getByText(/Layer-Streaming LLM Inference/i)).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText(/1. Layer Chunking & VRAM Memory Planner/i)).toBeInTheDocument();
      expect(screen.getByText(/Target Model Architecture/i)).toBeInTheDocument();
      expect(screen.getByText(/Target Hardware Constraint/i)).toBeInTheDocument();
    });
  });

  it('executes layer chunk calculation workflow and displays memory watermarks', async () => {
    render(
      <AuthProvider>
        <App />
      </AuthProvider>
    );

    await waitFor(() => {
      expect(screen.getByText(/Calculate VRAM & Layer Plan/i)).toBeInTheDocument();
    });

    const calculateBtn = screen.getByText(/Calculate VRAM & Layer Plan/i);
    fireEvent.click(calculateBtn);

    await waitFor(() => {
      expect(screen.getByText(/Execution Plan & Memory Watermark/i)).toBeInTheDocument();
      expect(screen.getByText(/2 layers/i)).toBeInTheDocument();
      expect(screen.getByText(/0.87 tok\/s/i)).toBeInTheDocument();
    });
  });

  it('navigates between workflow tabs including layer streaming and diagnostics', async () => {
    render(
      <AuthProvider>
        <App />
      </AuthProvider>
    );

    await waitFor(() => {
      expect(screen.getByText('2. Layer Stream Visualizer')).toBeInTheDocument();
    });

    // Switch to visualizer tab
    fireEvent.click(screen.getByText('2. Layer Stream Visualizer'));
    expect(screen.getByText(/Live Layer-Streaming Pipeline Visualizer/i)).toBeInTheDocument();

    // Switch to diagnostics tab
    fireEvent.click(screen.getByText('4. HF Compatibility & Patches'));
    expect(screen.getByText(/Hugging Face Compatibility & Patch Generator/i)).toBeInTheDocument();
  });
});
