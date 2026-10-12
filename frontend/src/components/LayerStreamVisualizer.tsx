import React, { useState } from 'react';
import { HardwareProfile, ModelSpec, BenchmarkSimulation } from '../types';
import { api } from '../services/api';

interface LayerStreamVisualizerProps {
  hardwareList: HardwareProfile[];
  modelsList: ModelSpec[];
}

export const LayerStreamVisualizer: React.FC<LayerStreamVisualizerProps> = ({
  hardwareList,
  modelsList,
}) => {
  const [selectedModel, setSelectedModel] = useState<string>(
    modelsList[0]?.id || 'model-llama3-70b'
  );
  const [selectedHardware, setSelectedHardware] = useState<string>(
    hardwareList[0]?.id || 'gpu-gtx-1650-4gb'
  );
  const [layersPerChunk, setLayersPerChunk] = useState<number>(1);
  const [quantization, setQuantization] = useState<string>('int4');
  const [running, setRunning] = useState<boolean>(false);
  const [simulation, setSimulation] = useState<BenchmarkSimulation | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleRunSimulation = async () => {
    setRunning(true);
    setError(null);
    try {
      const res = await api.simulateStream({
        model_id: selectedModel,
        hardware_id: selectedHardware,
        quantization,
        layers_per_chunk: layersPerChunk,
        prompt_tokens: 32,
        output_tokens: 16,
        run_numerical_simulation: true,
      });
      setSimulation(res);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="visualizer-container">
      <div className="card">
        <h2 className="card-title">2. Live Layer-Streaming Pipeline Visualizer</h2>
        <p className="card-desc">
          Executes and visualizes sequential layer offloading (Disk &rarr; RAM &rarr; PCIe &rarr; Compute &rarr; Evict).
          Tracks execution timings and numerical tensor integrity in real-time.
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
                  {m.name} ({m.num_layers} layers)
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
        </div>

        <div className="form-actions">
          <button
            onClick={handleRunSimulation}
            disabled={running}
            className="btn btn-primary"
          >
            {running ? 'Streaming Layers...' : 'Run Layer Stream Pipeline'}
          </button>
        </div>
      </div>

      {simulation && (
        <div className="card stream-results-card">
          <div className="stream-summary-bar">
            <div>
              <span className="summary-stat">Total Layers: <strong>{simulation.total_layers}</strong></span>
              <span className="summary-stat">Pipeline Chunks: <strong>{simulation.chunks_count}</strong></span>
              <span className="summary-stat">Peak VRAM: <strong>{Math.round(simulation.peak_vram_mb)} MB</strong></span>
              <span className="summary-stat">Throughput: <strong>{simulation.final_tokens_per_sec} tok/s</strong></span>
            </div>
            <div>
              <span className={`badge ${simulation.numerical_check_passed ? 'badge-success' : 'badge-warning'}`}>
                {simulation.numerical_check_passed ? '✓ Numerical Tensor Ops Validated' : 'Numerical Check Warning'}
              </span>
            </div>
          </div>

          <div className="stream-timeline">
            <h3 className="section-title">Sequential Stage Trace</h3>
            <div className="steps-list">
              {simulation.steps.map((step) => (
                <div key={step.chunk_idx} className="step-row">
                  <div className="step-chunk-badge">Chunk {step.chunk_idx + 1}</div>
                  <div className="step-info">
                    <div className="step-title">{step.stage}</div>
                    <div className="step-desc">{step.description}</div>
                  </div>
                  <div className="step-timings">
                    <span className="time-pill disk">Disk: {step.disk_read_ms}ms</span>
                    <span className="time-pill pcie">PCIe: {step.pcie_transfer_ms}ms</span>
                    <span className="time-pill compute">GPU Compute: {step.compute_ms}ms</span>
                    <span className="vram-pill">{Math.round(step.vram_allocated_mb)} MB</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
