import React from 'react';
import { useAuth } from '../context/AuthContext';

interface HeaderProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

export const Header: React.FC<HeaderProps> = ({ activeTab, setActiveTab }) => {
  const { user, loginAsDemo, logout } = useAuth();

  return (
    <header className="app-header">
      <div className="header-top">
        <div className="brand">
          <div className="brand-badge">TL</div>
          <div>
            <h1 className="brand-title">TensorLens</h1>
            <p className="brand-subtitle">
              Layer-Streaming LLM Inference &amp; VRAM Optimization Workbench
            </p>
          </div>
        </div>

        <div className="header-controls">
          <div className="demo-pill">
            <span className="pulse-indicator"></span>
            <span>Demo Mode (Localhost)</span>
          </div>

          <div className="role-switcher">
            <span className="role-label">Role:</span>
            <select
              value={user?.roles[0] || 'analyst'}
              onChange={(e) => loginAsDemo(e.target.value as 'viewer' | 'analyst' | 'admin')}
              className="role-select"
            >
              <option value="viewer">Viewer (Read-Only)</option>
              <option value="analyst">Analyst / Operator</option>
              <option value="admin">Administrator</option>
            </select>
          </div>

          <button onClick={logout} className="btn-logout" title="Log out">
            Reset Auth
          </button>
        </div>
      </div>

      <nav className="header-nav">
        <button
          className={`nav-tab ${activeTab === 'planner' ? 'active' : ''}`}
          onClick={() => setActiveTab('planner')}
        >
          1. Memory &amp; Chunk Planner
        </button>
        <button
          className={`nav-tab ${activeTab === 'visualizer' ? 'active' : ''}`}
          onClick={() => setActiveTab('visualizer')}
        >
          2. Layer Stream Visualizer
        </button>
        <button
          className={`nav-tab ${activeTab === 'benchmarks' ? 'active' : ''}`}
          onClick={() => setActiveTab('benchmarks')}
        >
          3. Benchmarks &amp; Export
        </button>
        <button
          className={`nav-tab ${activeTab === 'diagnostics' ? 'active' : ''}`}
          onClick={() => setActiveTab('diagnostics')}
        >
          4. HF Compatibility &amp; Patches
        </button>
        <button
          className={`nav-tab ${activeTab === 'audit' ? 'active' : ''}`}
          onClick={() => setActiveTab('audit')}
        >
          5. Audit Log
        </button>
      </nav>
    </header>
  );
};
