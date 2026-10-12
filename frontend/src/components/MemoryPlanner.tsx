import React, { useState } from 'react';
import { HardwareProfile, ModelSpec, InferencePlan } from '../types';
import { api } from '../services/api';

interface MemoryPlannerProps {
  hardwareList: HardwareProfile[];
  modelsList: ModelSpec[];
  currentPlan: InferencePlan | null;
  onPlanCalculated: (plan: InferencePlan) => void;
  onRequestAdvisor: (planId: string) => void;
}

export const MemoryPlanner: React.FC<MemoryPlannerProps> = ({
  hardwareList,
  modelsList,
  currentPlan,
  onPlanCalculated,
  onRequestAdvisor,
}) => {
  const [selectedModel, setSelectedModel] = useState<string>(
    modelsList[0]?.id || 'model-llama3-70b'
  );
  const [selectedHardware, setSelectedHardware] = useState<string>(
    hardwareList[0]?.id || 'gpu-gtx-1650-4gb'
  );
  const [quantization, setQuantization] = useState<string>('int4');
  const [batchSize, setBatchSize] = useState<number>(1);
  const [seqLen, setSeqLen] = useState<number>(512);
  const [manualChunk, setManualChunk] = useState<string>('');
  const [calculating, setCalculating] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const activeModel = modelsList.find((m) => m.id === selectedModel);
  const activeHardware = hardwareList.find((h) => h.id === selectedHardware);

  const handleCalculate = async () => {
    setCalculating(true);
    setError(null);
    try {
      const plan = await api.calculatePlan({
        model_id: selectedModel,
        hardware_id: selectedHardware,
        quantization,
        batch_size: batchSize,
        seq_len: seqLen,
        manual_layers_per_chunk: manualChunk ? parseInt(manualChunk, 10) : undefined,
      });
      onPlanCalculated(plan);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setCalculating(false);
    }
  };

  return (
    <div className="planner-container">
      <div className="card">
        <h2 className="card-title">1. Layer Chunking &amp; VRAM Memory Planner</h2>
        <p className="card-desc">
          Calculates layer chunking (N-layer batching) to maximize GPU memory utilization
          without exceeding hardware limits. Solves AirLLM 70B inference on 4GB GPUs.
        </p>

        {error && <div className="alert alert-error">{error}</div>}

        <div className="form-grid">
          <div className="form-group">
            <label className="form-label">Target Model Architecture</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="form-control"
            >
              {modelsList.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} ({m.parameter_count_billion}B params, {m.num_layers} layers)
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Target Hardware Constraint</label>
            <select
              value={selectedHardware}
              onChange={(e) => setSelectedHardware(e.target.value)}
              className="form-control"
            >
              {hardwareList.map((h) => (
                <option key={h.id} value={h.id}>
                  {h.name} — {h.vram_gb} GB VRAM ({h.category.toUpperCase()})
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Quantization / Compression</label>
            <select
              value={quantization}
              onChange={(e) => setQuantization(e.target.value)}
              className="form-control"
            >
              <option value="int4">4-bit Block-wise Compression (NF4/INT4 ~0.56 B/param)</option>
              <option value="int8">8-bit Quantization (INT8 ~1.05 B/param)</option>
              <option value="fp16">16-bit Full Precision (FP16/BF16 2.0 B/param)</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Context Sequence Length</label>
            <input
              type="number"
              value={seqLen}
              min={64}
              max={32768}
              step={64}
              onChange={(e) => setSeqLen(parseInt(e.target.value, 10) || 512)}
              className="form-control"
            />
          </div>

          <div className="form-group">
            <label className="form-label">Batch Size</label>
            <input
              type="number"
              value={batchSize}
              min={1}
              max={32}
              onChange={(e) => setBatchSize(parseInt(e.target.value, 10) || 1)}
              className="form-control"
            />
          </div>

          <div className="form-group">
            <label className="form-label">Manual Layer Chunk Override (Optional)</label>
            <input
              type="number"
              value={manualChunk}
              placeholder="Auto-solve optimal N"
              min={1}
              max={activeModel?.num_layers || 80}
              onChange={(e) => setManualChunk(e.target.value)}
              className="form-control"
            />
          </div>
        </div>

        <div className="form-actions">
          <button
            onClick={handleCalculate}
            disabled={calculating}
            className="btn btn-primary"
          >
            {calculating ? 'Solving Optimal Plan...' : 'Calculate VRAM & Layer Plan'}
          </button>
        </div>
      </div>

      {currentPlan && activeHardware && (
        <div className="card plan-results-card">
          <div className="plan-header">
            <div>
              <h3 className="section-title">Execution Plan &amp; Memory Watermark</h3>
              <p className="section-subtitle">
                Model: <strong>{activeModel?.name}</strong> | Hardware: <strong>{activeHardware.name}</strong>
              </p>
            </div>
            <button
              onClick={() => onRequestAdvisor(currentPlan.id)}
              className="btn btn-secondary"
            >
              Ask AI Systems Advisor
            </button>
          </div>

          <div className="metrics-grid">
            <div className="metric-box">
              <span className="metric-label">Optimal Chunk Size</span>
              <span className="metric-value highlight">{currentPlan.layers_per_chunk} layers</span>
              <span className="metric-sub">{currentPlan.total_chunks} total chunks</span>
            </div>

            <div className="metric-box">
              <span className="metric-label">Peak VRAM Allocated</span>
              <span className="metric-value">{Math.round(currentPlan.total_vram_mb)} MB</span>
              <span className="metric-sub">
                Limit: {Math.round(activeHardware.vram_gb * 1024)} MB ({currentPlan.vram_utilization_percent}%)
              </span>
            </div>

            <div className="metric-box">
              <span className="metric-label">Estimated Throughput</span>
              <span className="metric-value">{currentPlan.est_tokens_per_sec} tok/s</span>
              <span className="metric-sub">{currentPlan.est_generation_sec}s per token</span>
            </div>

            <div className="metric-box">
              <span className="metric-label">Time to First Token</span>
              <span className="metric-value">{currentPlan.est_prefill_sec}s</span>
              <span className="metric-sub">Prefill for {currentPlan.seq_len} tokens</span>
            </div>
          </div>

          <div className="vram-meter-section">
            <div className="vram-meter-label">
              <span>Memory Allocation Watermark:</span>
              <span>{Math.round(currentPlan.total_vram_mb)} MB / {Math.round(activeHardware.vram_gb * 1024)} MB</span>
            </div>
            <div className="vram-meter-bar">
              <div
                className="vram-fill driver"
                style={{ width: `${(512 / (activeHardware.vram_gb * 1024)) * 100}%` }}
                title="CUDA System Reserve: 512 MB"
              ></div>
              <div
                className="vram-fill kv"
                style={{ width: `${(currentPlan.kv_cache_vram_mb / (activeHardware.vram_gb * 1024)) * 100}%` }}
                title={`KV Cache: ${currentPlan.kv_cache_vram_mb} MB`}
              ></div>
              <div
                className="vram-fill weights"
                style={{ width: `${(currentPlan.param_vram_mb / (activeHardware.vram_gb * 1024)) * 100}%` }}
                title={`Active Layer Weights: ${currentPlan.param_vram_mb} MB`}
              ></div>
              <div
                className="vram-fill activation"
                style={{ width: `${(currentPlan.activation_vram_mb / (activeHardware.vram_gb * 1024)) * 100}%` }}
                title={`Activations: ${currentPlan.activation_vram_mb} MB`}
              ></div>
            </div>
            <div className="vram-legend">
              <span className="legend-item"><span className="legend-dot driver"></span> Driver/System (512 MB)</span>
              <span className="legend-item"><span className="legend-dot kv"></span> KV Cache ({currentPlan.kv_cache_vram_mb} MB)</span>
              <span className="legend-item"><span className="legend-dot weights"></span> Layer Chunk Weights ({currentPlan.param_vram_mb} MB)</span>
              <span className="legend-item"><span className="legend-dot activation"></span> Forward Activations ({currentPlan.activation_vram_mb} MB)</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
