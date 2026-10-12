import React, { useState, useEffect } from 'react';
import { HardwareProfile, ModelSpec, BenchmarkRun } from '../types';
import { api } from '../services/api';
import { useAuth } from '../context/AuthContext';

interface BenchmarkRunnerProps {
  hardwareList: HardwareProfile[];
  modelsList: ModelSpec[];
  onRequestAdvisor: (benchmarkId: string) => void;
}

export const BenchmarkRunner: React.FC<BenchmarkRunnerProps> = ({
  hardwareList,
  modelsList,
  onRequestAdvisor,
}) => {
  const { hasRole } = useAuth();
  const [runs, setRuns] = useState<BenchmarkRun[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [selectedModel, setSelectedModel] = useState<string>(
    modelsList[0]?.id || 'model-llama3-70b'
  );
  const [selectedHardware, setSelectedHardware] = useState<string>(
    hardwareList[0]?.id || 'gpu-gtx-1650-4gb'
  );
  const [quantization, setQuantization] = useState<string>('int4');
  const [layersPerChunk, setLayersPerChunk] = useState<number>(2);
  const [promptTokens, setPromptTokens] = useState<number>(32);
  const [outputTokens, setOutputTokens] = useState<number>(16);
  const [error, setError] = useState<string | null>(null);

  const loadHistory = async () => {
    try {
      const data = await api.getBenchmarkHistory();
      setRuns(data);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadHistory();
  }, []);

  const handleRecordRun = async () => {
    setSubmitting(true);
    setError(null);
    try {
      await api.recordBenchmarkRun({
        model_id: selectedModel,
        hardware_id: selectedHardware,
        quantization,
        layers_per_chunk: layersPerChunk,
        prompt_tokens: promptTokens,
        output_tokens: outputTokens,
        run_numerical_simulation: true,
      });
      await loadHistory();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  const handleExport = (format: 'json' | 'csv') => {
    window.open(`/api/benchmark/export?format=${format}`, '_blank');
  };

  return (
    <div className="benchmarks-container">
      <div className="card">
        <h2 className="card-title">3. Benchmark Execution &amp; Telemetry Archive</h2>
        <p className="card-desc">
          Execute and persist layer-streaming benchmarks across hardware targets and quantizations.
          Compare multi-layer chunk efficiency and export telemetry.
        </p>

        {error && <div className="alert alert-error">{error}</div>}

        <div className="form-grid">
          <div className="form-group">
            <label className="form-label">Model Spec</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="form-control"
            >
              {modelsList.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} ({m.parameter_count_billion}B params)
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Hardware Device</label>
            <select
              value={selectedHardware}
              onChange={(e) => setSelectedHardware(e.target.value)}
              className="form-control"
            >
              {hardwareList.map((h) => (
                <option key={h.id} value={h.id}>
                  {h.name} ({h.vram_gb} GB VRAM)
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Layers Per Chunk (N)</label>
            <input
              type="number"
              value={layersPerChunk}
              min={1}
              max={32}
              onChange={(e) => setLayersPerChunk(parseInt(e.target.value, 10) || 1)}
              className="form-control"
            />
          </div>

          <div className="form-group">
            <label className="form-label">Quantization</label>
            <select
              value={quantization}
              onChange={(e) => setQuantization(e.target.value)}
              className="form-control"
            >
              <option value="int4">4-bit (NF4 block compression)</option>
              <option value="int8">8-bit Quantization</option>
              <option value="fp16">16-bit Full Precision</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Prompt Tokens</label>
            <input
              type="number"
              value={promptTokens}
              min={1}
              max={2048}
              onChange={(e) => setPromptTokens(parseInt(e.target.value, 10) || 32)}
              className="form-control"
            />
          </div>

          <div className="form-group">
            <label className="form-label">Output Tokens</label>
            <input
              type="number"
              value={outputTokens}
              min={1}
              max={512}
              onChange={(e) => setOutputTokens(parseInt(e.target.value, 10) || 16)}
              className="form-control"
            />
          </div>
        </div>

        <div className="form-actions">
          <button
            onClick={handleRecordRun}
            disabled={submitting || !hasRole('analyst')}
            className="btn btn-primary"
            title={!hasRole('analyst') ? 'Requires Analyst or Admin role' : ''}
          >
            {submitting ? 'Recording Benchmark...' : 'Execute & Record Benchmark Run'}
          </button>
          {!hasRole('analyst') && (
            <span className="role-hint">Requires Analyst or Admin role to record persistent runs.</span>
          )}
        </div>
      </div>

      <div className="card">
        <div className="table-header-row">
          <div>
            <h3 className="section-title">Recorded Benchmark Runs</h3>
            <p className="section-subtitle">Real telemetry across configurations and devices</p>
          </div>
          <div className="export-btn-group">
            <button onClick={() => handleExport('csv')} className="btn btn-sm btn-outline">
              Export CSV
            </button>
            <button onClick={() => handleExport('json')} className="btn btn-sm btn-outline">
              Export JSON
            </button>
          </div>
        </div>

        {loading ? (
          <div className="empty-state">Loading benchmarks...</div>
        ) : runs.length === 0 ? (
          <div className="empty-state">No benchmark runs recorded yet. Execute a run above.</div>
        ) : (
          <div className="table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Hardware</th>
                  <th>Quant</th>
                  <th>Chunk Size</th>
                  <th>Throughput</th>
                  <th>Peak VRAM</th>
                  <th>Disk I/O</th>
                  <th>PCIe Transfer</th>
                  <th>GPU Compute</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => (
                  <tr key={r.id}>
                    <td><strong>{r.model_id.replace('model-', '')}</strong></td>
                    <td>{r.hardware_id.replace('gpu-', '')}</td>
                    <td><span className="badge badge-info">{r.quantization.toUpperCase()}</span></td>
                    <td>{r.layers_per_chunk} layer(s)</td>
                    <td className="highlight">{r.tokens_per_sec} tok/s</td>
                    <td>{Math.round(r.peak_vram_mb)} MB</td>
                    <td>{Math.round(r.disk_read_time_ms)} ms</td>
                    <td>{Math.round(r.pcie_transfer_time_ms)} ms</td>
                    <td>{Math.round(r.compute_time_ms)} ms</td>
                    <td>
                      <button
                        onClick={() => onRequestAdvisor(r.id)}
                        className="btn btn-xs btn-secondary"
                      >
                        Advisor
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
