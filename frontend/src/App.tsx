import React, { useState, useEffect } from 'react';
import { HardwareProfile, ModelSpec, InferencePlan } from './types';
import { api } from './services/api';
import { Header } from './components/Header';
import { MemoryPlanner } from './components/MemoryPlanner';
import { LayerStreamVisualizer } from './components/LayerStreamVisualizer';
import { BenchmarkRunner } from './components/BenchmarkRunner';
import { CompatibilityDiagnostics } from './components/CompatibilityDiagnostics';
import { AuditLogView } from './components/AuditLogView';
import { LlmAdvisorModal } from './components/LlmAdvisorModal';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('planner');
  const [hardwareList, setHardwareList] = useState<HardwareProfile[]>([]);
  const [modelsList, setModelsList] = useState<ModelSpec[]>([]);
  const [currentPlan, setCurrentPlan] = useState<InferencePlan | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Advisor Modal State
  const [advisorTarget, setAdvisorTarget] = useState<{
    planId?: string;
    benchmarkId?: string;
  } | null>(null);

  useEffect(() => {
    const fetchPrerequisites = async () => {
      setLoading(true);
      try {
        const [hw, models] = await Promise.all([
          api.getHardwareProfiles(),
          api.getModels(),
        ]);
        setHardwareList(hw);
        setModelsList(models);
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setLoading(false);
      }
    };
    fetchPrerequisites();
  }, []);

  return (
    <div className="app-shell">
      <Header activeTab={activeTab} setActiveTab={setActiveTab} />

      <main className="app-main">
        {error && (
          <div className="alert alert-error">
            <span>Failed to connect to TensorLens backend: {error}</span>
          </div>
        )}

        {loading ? (
          <div className="empty-state">
            <div className="loading-spinner">Initializing TensorLens workspace...</div>
          </div>
        ) : (
          <>
            {activeTab === 'planner' && (
              <MemoryPlanner
                hardwareList={hardwareList}
                modelsList={modelsList}
                currentPlan={currentPlan}
                onPlanCalculated={(plan) => setCurrentPlan(plan)}
                onRequestAdvisor={(planId) => setAdvisorTarget({ planId })}
              />
            )}

            {activeTab === 'visualizer' && (
              <LayerStreamVisualizer
                hardwareList={hardwareList}
                modelsList={modelsList}
              />
            )}

            {activeTab === 'benchmarks' && (
              <BenchmarkRunner
                hardwareList={hardwareList}
                modelsList={modelsList}
                onRequestAdvisor={(benchmarkId) => setAdvisorTarget({ benchmarkId })}
              />
            )}

            {activeTab === 'diagnostics' && (
              <CompatibilityDiagnostics />
            )}

            {activeTab === 'audit' && (
              <AuditLogView />
            )}
          </>
        )}
      </main>

      {advisorTarget && (
        <LlmAdvisorModal
          planId={advisorTarget.planId}
          benchmarkId={advisorTarget.benchmarkId}
          onClose={() => setAdvisorTarget(null)}
        />
      )}

      <footer className="app-footer">
        <p>
          TensorLens &bull; Ryan Vo &lt;ryandtvo@gmail.com&gt; &bull; AI &amp; Machine Learning Inference Workbench
        </p>
      </footer>
    </div>
  );
};

export default App;
