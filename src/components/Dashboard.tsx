import React, { useState, useEffect } from 'react';
import { BarChart3, Users, AlertTriangle, Brain, Cpu, TrendingUp, Activity, Database } from 'lucide-react';
import BatchAnalyzer from './BatchAnalyzer';
import { getStats } from '../utils/mockApi';

interface DashboardProps {
  refreshTrigger: number;
  onTabChange: (tab: 'dashboard' | 'predict' | 'history') => void;
}

const Dashboard: React.FC<DashboardProps> = ({ refreshTrigger, onTabChange }) => {
  const [stats, setStats] = useState(getStats());

  useEffect(() => {
    setStats(getStats());
  }, [refreshTrigger]);

  const statCards = [
    {
      icon: <Activity size={24} />,
      label: 'Total Predictions',
      value: stats.totalPredictions,
      color: 'var(--color-primary)',
      bg: 'var(--color-primary-light)',
    },
    {
      icon: <AlertTriangle size={24} />,
      label: 'High Risk Employees',
      value: stats.highRisk,
      color: 'var(--color-danger)',
      bg: 'var(--color-danger-light)',
    },
    {
      icon: <TrendingUp size={24} />,
      label: 'Avg. Attrition Prob.',
      value: `${stats.avgProb}%`,
      color: 'var(--color-warning)',
      bg: 'var(--color-warning-light)',
    },
    {
      icon: <Database size={24} />,
      label: 'Batch Processes',
      value: stats.totalBatches,
      color: 'var(--color-info)',
      bg: 'var(--color-info-light)',
    },
    {
      icon: <Users size={24} />,
      label: 'Records Analyzed',
      value: stats.totalRecords,
      color: 'var(--color-success)',
      bg: 'var(--color-success-light)',
    },
    {
      icon: <Brain size={24} />,
      label: 'AI Engine',
      value: 'Dual-Brain',
      color: '#8b5cf6',
      bg: '#ede9fe',
    },
  ];

  return (
    <div className="dashboard">
      {/* Stats Overview */}
      <div className="stats-grid">
        {statCards.map((card, i) => (
          <div
            key={i}
            className="stat-card"
            style={{ '--card-color': card.color, '--card-bg': card.bg } as React.CSSProperties}
          >
            <div className="stat-card-icon">{card.icon}</div>
            <div className="stat-card-info">
              <span className="stat-card-value">{card.value}</span>
              <span className="stat-card-label">{card.label}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Model Architecture Info */}
      <div className="model-info-banner">
        <div className="model-info-left">
          <div className="model-info-icon">
            <Cpu size={24} />
          </div>
          <div>
            <h3>Dual-Brain VotingClassifier Ensemble</h3>
            <p>5 Architectures: Random Forest • Gradient Boosting • XGBoost • Extra Trees • MLP</p>
          </div>
        </div>
        <div className="model-models">
          <div className="model-chip ibm">
            <Brain size={14} /> model_ibm.pkl
          </div>
          <div className="model-chip synthetic">
            <Cpu size={14} /> model_syn.pkl
          </div>
        </div>
      </div>

      {/* Batch Analyzer - Center Stage */}
      <div className="dashboard-section">
        <BatchAnalyzer onBatchComplete={() => setStats(getStats())} />
      </div>

      {/* Quick Actions */}
      <div className="quick-actions">
        <button className="quick-action-card" onClick={() => onTabChange('predict')}>
          <div className="qa-icon" style={{ background: '#ede9fe', color: '#7c3aed' }}>
            <Brain size={28} />
          </div>
          <h4>Individual Prediction</h4>
          <p>Analyze a single employee with the dual-pipeline form</p>
        </button>
        <button className="quick-action-card" onClick={() => onTabChange('history')}>
          <div className="qa-icon" style={{ background: '#fef3c7', color: '#d97706' }}>
            <BarChart3 size={28} />
          </div>
          <h4>View History</h4>
          <p>Cross-compare past runs and batch analyses</p>
        </button>
        <button className="quick-action-card" onClick={() => onTabChange('predict')}>
          <div className="qa-icon" style={{ background: '#dcfce7', color: '#16a34a' }}>
            <Users size={28} />
          </div>
          <h4>IBM HR Pipeline</h4>
          <p>35-feature authentic IBM schema analysis</p>
        </button>
      </div>
    </div>
  );
};

export default Dashboard;
