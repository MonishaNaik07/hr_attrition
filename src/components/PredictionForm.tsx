import React, { useState, useMemo } from 'react';
import { Brain, Cpu, Zap, AlertTriangle, CheckCircle, ChevronDown, Sparkles, RotateCcw } from 'lucide-react';
import { predictAttrition, saveRecord, generateId } from '../utils/mockApi';
import { PipelineType, PredictionResult, IBM_SCHEMA, SYNTHETIC_SCHEMA, SchemaField } from '../types';

interface PredictionFormProps {
  onPrediction: () => void;
}

const PredictionForm: React.FC<PredictionFormProps> = ({ onPrediction }) => {
  const [pipeline, setPipeline] = useState<PipelineType>('ibm');
  const [showResult, setShowResult] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [formData, setFormData] = useState<Record<string, string | number>>(() => {
    const initial: Record<string, string | number> = {};
    IBM_SCHEMA.forEach(f => { initial[f.name] = f.default ?? 0; });
    return initial;
  });

  const schema = useMemo(() => pipeline === 'ibm' ? IBM_SCHEMA : SYNTHETIC_SCHEMA, [pipeline]);

  const handlePipelineChange = (newPipeline: PipelineType) => {
    setPipeline(newPipeline);
    setShowResult(false);
    setResult(null);
    const newSchema = newPipeline === 'ibm' ? IBM_SCHEMA : SYNTHETIC_SCHEMA;
    const initial: Record<string, string | number> = {};
    newSchema.forEach(f => { initial[f.name] = f.default ?? 0; });
    setFormData(initial);
  };

  const handleFieldChange = (name: string, value: string | number) => {
    setFormData(prev => ({ ...prev, [name]: value }));
    if (showResult) setShowResult(false);
  };

  const handlePredict = () => {
    const prediction = predictAttrition(formData, pipeline);
    setResult(prediction);
    setShowResult(true);

    const record = {
      id: generateId(),
      timestamp: new Date().toISOString(),
      type: 'individual' as const,
      pipeline,
      input: { ...formData },
      result: prediction,
    };
    saveRecord(record);
    onPrediction();
  };

  const handleReset = () => {
    const initial: Record<string, string | number> = {};
    schema.forEach(f => { initial[f.name] = f.default ?? 0; });
    setFormData(initial);
    setShowResult(false);
    setResult(null);
  };

  const riskColorMap = {
    Low: 'var(--color-success)',
    Medium: 'var(--color-warning)',
    High: 'var(--color-danger)',
    Critical: 'var(--color-critical)',
  };

  const riskIconMap = {
    Low: <CheckCircle size={20} />,
    Medium: <AlertTriangle size={20} />,
    High: <AlertTriangle size={20} />,
    Critical: <Zap size={20} />,
  };

  const renderField = (field: SchemaField, index: number) => {
    const value = formData[field.name] ?? field.default;
    const animDelay = `${index * 20}ms`;

    if (field.type === 'categorical' && field.options) {
      return (
        <div className="form-field" key={field.name} style={{ animationDelay: animDelay }}>
          <label htmlFor={field.name}>{field.label}</label>
          <div className="select-wrapper">
            <select
              id={field.name}
              value={String(value)}
              onChange={e => handleFieldChange(field.name, e.target.value)}
            >
              {field.options.map(opt => (
                <option key={opt} value={opt}>{opt}</option>
              ))}
            </select>
            <ChevronDown size={16} className="select-icon" />
          </div>
        </div>
      );
    }

    return (
      <div className="form-field" key={field.name} style={{ animationDelay: animDelay }}>
        <label htmlFor={field.name}>
          {field.label}
          {field.min !== undefined && field.max !== undefined && (
            <span className="field-range">({field.min} – {field.max})</span>
          )}
        </label>
        <input
          id={field.name}
          type="number"
          min={field.min}
          max={field.max}
          value={Number(value)}
          onChange={e => handleFieldChange(field.name, Number(e.target.value))}
        />
      </div>
    );
  };

  return (
    <div className="prediction-form-container">
      {/* Pipeline Selector */}
      <div className="pipeline-selector">
        <div className="pipeline-selector-header">
          <Cpu size={20} />
          <h3>Select Pipeline Architecture</h3>
        </div>
        <div className="pipeline-toggle">
          <button
            className={`pipeline-toggle-btn ${pipeline === 'ibm' ? 'active' : ''}`}
            onClick={() => handlePipelineChange('ibm')}
          >
            <Brain size={18} />
            <div>
              <strong>Authentic IBM Schema</strong>
              <span>35 features • ColumnTransformer • 1,470 rows trained</span>
            </div>
          </button>
          <button
            className={`pipeline-toggle-btn ${pipeline === 'synthetic' ? 'active' : ''}`}
            onClick={() => handlePipelineChange('synthetic')}
          >
            <Cpu size={18} />
            <div>
              <strong>Synthetic Data Schema</strong>
              <span>20 numerical features • StandardScaler • 6,000 rows trained</span>
            </div>
          </button>
        </div>
        <div className="pipeline-info">
          <span className="pipeline-badge">
            {pipeline === 'ibm' ? 'model_ibm.pkl' : 'model_syn.pkl'}
          </span>
          <span className="pipeline-badge secondary">
            VotingClassifier • Soft Voting • 5 Estimators
          </span>
        </div>
      </div>

      {/* Dynamic Form Fields */}
      <div className="form-grid-container">
        <div className="form-grid-header">
          <Sparkles size={18} />
          <h4>{pipeline === 'ibm' ? 'IBM HR' : 'Synthetic'} Feature Input ({schema.length} fields)</h4>
        </div>
        <div className="form-grid">
          {schema.map((field, i) => renderField(field, i))}
        </div>
      </div>

      {/* Action Buttons */}
      <div className="form-actions">
        <button className="btn-predict" onClick={handlePredict}>
          <Zap size={18} /> Run AI Prediction
        </button>
        <button className="btn-secondary" onClick={handleReset}>
          <RotateCcw size={16} /> Reset
        </button>
      </div>

      {/* Result Card */}
      {showResult && result && (
        <div className="prediction-result-card">
          <div className="result-header">
            <h3>🎯 Prediction Result</h3>
            <div className="result-pipeline-tag">
              {pipeline === 'ibm' ? 'IBM Pipeline' : 'Synthetic Pipeline'}
            </div>
          </div>

          <div className="result-main-display">
            <div className="result-probability-ring" style={{ '--prob-color': riskColorMap[result.riskLevel] } as React.CSSProperties}>
              <svg viewBox="0 0 120 120">
                <circle cx="60" cy="60" r="54" fill="none" stroke="#e5e7eb" strokeWidth="8" />
                <circle
                  cx="60" cy="60" r="54" fill="none"
                  stroke={riskColorMap[result.riskLevel]}
                  strokeWidth="8"
                  strokeDasharray={`${result.attritionProbability * 3.39} 339`}
                  strokeLinecap="round"
                  transform="rotate(-90 60 60)"
                  className="result-ring-circle"
                />
              </svg>
              <div className="result-probability-text">
                <span className="prob-value">{result.attritionProbability}%</span>
                <span className="prob-label">Attrition Risk</span>
              </div>
            </div>
            <div className="result-risk-badge" style={{ color: riskColorMap[result.riskLevel] }}>
              {riskIconMap[result.riskLevel]}
              <span>{result.riskLevel} Risk</span>
            </div>
          </div>

          <div className="result-sections">
            <div className="result-section">
              <h4>🔍 Why They Are At Risk</h4>
              <ul className="risk-factors-list">
                {result.riskFactors.map((f, i) => (
                  <li key={i}>{f}</li>
                ))}
              </ul>
            </div>
            <div className="result-section">
              <h4>💡 Actionable HR Retention Solutions</h4>
              <ul className="retention-actions-list">
                {result.retentionActions.map((a, i) => (
                  <li key={i}>{a}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictionForm;
