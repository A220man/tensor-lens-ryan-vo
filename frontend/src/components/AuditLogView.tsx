import React, { useState, useEffect } from 'react';
import { AuditLog } from '../types';
import { api } from '../services/api';
import { useAuth } from '../context/AuthContext';

export const AuditLogView: React.FC = () => {
  const { hasRole } = useAuth();
  const [logs, setLogs] = useState<AuditLog[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchLogs = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getAuditLogs();
      setLogs(data);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (hasRole('admin')) {
      fetchLogs();
    } else {
      setLoading(false);
    }
  }, [hasRole('admin')]);

  if (!hasRole('admin')) {
    return (
      <div className="card">
        <h2 className="card-title">5. System Audit Trail</h2>
        <div className="alert alert-warning">
          Access Denied: Viewing audit logs requires the <strong>Administrator</strong> role.
          Use the role switcher in the top navigation to select Administrator in demo mode.
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="table-header-row">
        <div>
          <h2 className="card-title">5. System Audit Trail</h2>
          <p className="card-desc">Immutable audit record of all model creations, planning runs, and benchmarks.</p>
        </div>
        <button onClick={fetchLogs} className="btn btn-outline btn-sm">Refresh</button>
      </div>

      {error && <div className="alert alert-error">{error}</div>}

      {loading ? (
        <div className="empty-state">Loading audit trail...</div>
      ) : logs.length === 0 ? (
        <div className="empty-state">No audit logs recorded yet.</div>
      ) : (
        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>User / Subject</th>
                <th>Role</th>
                <th>Action</th>
                <th>Resource ID</th>
                <th>Details</th>
              </tr>
            </thead>
            <tbody>
              {logs.map((log) => (
                <tr key={log.id}>
                  <td>{new Date(log.timestamp).toLocaleString()}</td>
                  <td><code>{log.user_id}</code></td>
                  <td><span className="badge badge-info">{log.role}</span></td>
                  <td><strong>{log.action}</strong></td>
                  <td><code>{log.resource}</code></td>
                  <td>{log.details}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
