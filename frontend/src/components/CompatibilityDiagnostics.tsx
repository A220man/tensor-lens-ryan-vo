import React, { useState } from 'react';
import { CompatibilityAudit } from '../types';
import { api } from '../services/api';

export const CompatibilityDiagnostics: React.FC = () => {
  const [modelName, setModelName] = useState<string>('Meta-Llama-3-70B-Instruct');
  const [transformersVersion, setTransformersVersion] = useState<string>('4.45.2');
  const [torchVersion, setTorchVersion] = useState<string>('2.4.1');
  const [targetDevice, setTargetDevice] = useState<string>('cuda');
  const [headDim, setHeadDim] = useState<number>(128);
  const [rotaryDim, setRotaryDim] = useState<number>(128);
  const [auditing, setAuditing] = useState<boolean>(false);
  const [auditResult, setAuditResult] = useState<CompatibilityAudit | null>(null);
  const [copied, setCopied] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleAudit = async () => {
    setAuditing(true);
    setError(null);
    try {
      const res = await api.auditCompatibility({
        model_name: modelName,
        transformers_version: transformersVersion,
        torch_version: torchVersion,
        target_device: targetDevice,
        head_dim: headDim,
        rotary_dim: rotaryDim,
      });
      setAuditResult(res);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setAuditing(false);
    }
  };

  const copyPatch = () => {
    if (auditResult?.generated_patch) {
      navigator.clipboard.writeText(auditResult.generated_patch);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="diagnostics-container">
      <div className="card">
        <h2 className="card-title">4. Hugging Face Compatibility &amp; Patch Generator</h2>
        <p className="card-desc">
          Diagnoses breaking version drifts in modern Hugging Face Transformers and PyTorch
          (such as `_is_stateful`, `DynamicCache` subscripting, and RoPE dimension alignment from issues #258, #301).
        </p>

        {error && <div className="alert alert-error">{error}</div>}

        <div className="form-grid">
          <div className="form-group">
            <label className="form-label">Target Model</label>
            <input
              type="text"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              className="form-control"
            />
          </div>

          <div className="form-group">
            <label className="form-label">Transformers Version</label>
            <input
              type="text"
              value={transformersVersion}
              onChange={(e) => setTransformersVersion(e.target.value)}
              className="form-control"
            />
          </div>

          <div className="form-group">
            <label className="form-label">PyTorch Version</label>
            <input
              type="text"
              value={torchVersion}
              onChange={(e) => setTorchVersion(e.target.value)}
              className="form-control"
            />
          </div>

          <div className="form-group">
            <label className="form-label">Device Backend</label>
            <select
              value={targetDevice}
              onChange={(e) => setTargetDevice(e.target.value)}
              className="form-control"
            >
              <option value="cuda">NVIDIA CUDA</option>
              <option value="rocm">AMD ROCm</option>
              <option value="directml">Windows DirectML</option>
              <option value="cpu">Host CPU</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Attention Head Dim</label>
            <input
              type="number"
              value={headDim}
              onChange={(e) => setHeadDim(parseInt(e.target.value, 10) || 128)}
              className="form-control"
            />
          </div>

          <div className="form-group">
            <label className="form-label">Rotary Embedding Dim</label>
            <input
              type="number"
              value={rotaryDim}
              onChange={(e) => setRotaryDim(parseInt(e.target.value, 10) || 128)}
              className="form-control"
            />
          </div>
        </div>

        <div className="form-actions">
          <button
            onClick={handleAudit}
            disabled={auditing}
            className="btn btn-primary"
          >
            {auditing ? 'Auditing Runtime...' : 'Audit Environment & Generate Patches'}
          </button>
        </div>
      </div>

      {auditResult && (
        <div className="card diagnostics-results-card">
          <h3 className="section-title">Diagnostic Findings</h3>

          <div className="checks-grid">
            <div className={`check-card ${auditResult.has_stateful_attribute ? 'pass' : 'fail'}`}>
              <span className="check-title">GenerationMixin `_is_stateful`</span>
              <span className="check-status">
                {auditResult.has_stateful_attribute ? 'Compatible' : 'Attribute Missing in >= 4.40'}
              </span>
            </div>

            <div className={`check-card ${auditResult.dynamic_cache_compatible ? 'pass' : 'fail'}`}>
              <span className="check-title">KV Cache `DynamicCache`</span>
              <span className="check-status">
                {auditResult.dynamic_cache_compatible ? 'Compatible' : 'Subscripting Unsupported in >= 4.36'}
              </span>
            </div>

            <div className={`check-card ${auditResult.rope_dimension_aligned ? 'pass' : 'fail'}`}>
              <span className="check-title">RoPE Rotary Alignment</span>
              <span className="check-status">
                {auditResult.rope_dimension_aligned ? 'Aligned' : 'Head Dim Mismatch'}
              </span>
            </div>

            <div className={`check-card ${auditResult.device_compatible ? 'pass' : 'warn'}`}>
              <span className="check-title">Device &amp; Backend Support</span>
              <span className="check-status">
                {auditResult.device_compatible ? 'Configured' : 'ROCm/DirectML Tuning Recommended'}
              </span>
            </div>
          </div>

          {auditResult.issues.length > 0 && (
            <div className="issues-box">
              <h4 className="box-title">Identified Issues:</h4>
              <ul className="issues-list">
                {auditResult.issues.map((issue, idx) => (
                  <li key={idx}>{issue}</li>
                ))}
              </ul>
            </div>
          )}

          <div className="patch-box">
            <div className="patch-header">
              <h4 className="box-title">Automated Runtime Patch Shim</h4>
              <button onClick={copyPatch} className="btn btn-xs btn-outline">
                {copied ? 'Copied!' : 'Copy Code'}
              </button>
            </div>
            <pre className="code-block">
              <code>{auditResult.generated_patch}</code>
            </pre>
          </div>
        </div>
      )}
    </div>
  );
};
