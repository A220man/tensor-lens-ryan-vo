import React, { useState, useEffect } from 'react';
import { AdvisorAnalysis } from '../types';
import { api } from '../services/api';

interface LlmAdvisorModalProps {
  planId?: string;
  benchmarkId?: string;
  onClose: () => void;
}

export const LlmAdvisorModal: React.FC<LlmAdvisorModalProps> = ({
  planId,
  benchmarkId,
  onClose,
}) => {
  const [loading, setLoading] = useState<boolean>(true);
  const [analysis, setAnalysis] = useState<AdvisorAnalysis | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [customQuery, setCustomQuery] = useState<string>('');

  const fetchAdvice = async (query?: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.requestAdvisory({
        plan_id: planId,
        benchmark_id: benchmarkId,
        user_query: query || undefined,
      });
      setAnalysis(res);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAdvice();
  }, [planId, benchmarkId]);

  return (
    <div className="modal-backdrop">
      <div className="modal-content">
        <div className="modal-header">
          <div>
            <h3 className="modal-title">AI Systems Optimization Advisor</h3>
            <p className="modal-subtitle">Grounded Analysis &amp; Tuning Recommendations</p>
          </div>
          <button onClick={onClose} className="modal-close-btn">&times;</button>
        </div>

        <div className="modal-body">
          <div className="advisory-disclaimer">
            <span className="disclaimer-badge">OPT-IN ADVISORY ONLY</span>
            <span>
              All recommendations are grounded in verified telemetry. Model operator configuration
              governs external provider selection.
            </span>
          </div>

          {error && <div className="alert alert-error">{error}</div>}

          {loading ? (
            <div className="loading-spinner">Generating grounded systems analysis...</div>
          ) : analysis ? (
            <div className="analysis-results">
              <div className="provider-info-box">
                <span>Provider: <strong>{analysis.provider}</strong></span>
                <span>Model: <strong>{analysis.model}</strong></span>
                <span>Grounded: <strong>{analysis.is_grounded ? 'Yes' : 'No'}</strong></span>
              </div>

              {analysis.recommendations.length > 0 && (
                <div className="recommendations-box">
                  <h4 className="box-title">Key Tuning Recommendations:</h4>
                  <ul className="recs-list">
                    {analysis.recommendations.map((rec, i) => (
                      <li key={i}>{rec}</li>
                    ))}
                  </ul>
                </div>
              )}

              <div className="analysis-text-box">
                <h4 className="box-title">Detailed Evaluation:</h4>
                <p className="analysis-body">{analysis.analysis_text}</p>
              </div>

              <div className="query-box">
                <input
                  type="text"
                  placeholder="Ask a specific question about these telemetry metrics..."
                  value={customQuery}
                  onChange={(e) => setCustomQuery(e.target.value)}
                  className="form-control"
                />
                <button
                  onClick={() => fetchAdvice(customQuery)}
                  className="btn btn-primary btn-sm"
                  disabled={loading}
                >
                  Ask
                </button>
              </div>
            </div>
          ) : null}
        </div>

        <div className="modal-footer">
          <button onClick={onClose} className="btn btn-secondary">Close</button>
        </div>
      </div>
    </div>
  );
};
