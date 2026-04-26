import React, { useState, useRef, useCallback } from 'react';
import Papa from 'papaparse';
import { Upload, FileSpreadsheet, AlertCircle, CheckCircle2, Brain, Loader2, Download } from 'lucide-react';
import { predictAttrition, detectPipeline, saveRecord, saveBatch, generateId } from '../utils/mockApi';
import { PredictionRecord, BatchSummary } from '../types';

interface BatchAnalyzerProps {
  onBatchComplete: () => void;
}

const BatchAnalyzer: React.FC<BatchAnalyzerProps> = ({ onBatchComplete }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [batchResult, setBatchResult] = useState<BatchSummary | null>(null);
  const [error, setError] = useState('');
  const [previewData, setPreviewData] = useState<{ columns: string[]; rowCount: number; pipeline: string } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((f: File) => {
    if (!f.name.endsWith('.csv')) {
      setError('Please upload a CSV file.');
      return;
    }
    setError('');
    setFile(f);
    setBatchResult(null);

    // Preview the file
    Papa.parse(f, {
      header: true,
      preview: 5,
      complete: (results) => {
        const columns = results.meta.fields || [];
        const pipeline = detectPipeline(columns);
        setPreviewData({
          columns,
          rowCount: 0,
          pipeline: pipeline === 'ibm' ? 'IBM HR Pipeline' : pipeline === 'synthetic' ? 'Synthetic Pipeline' : 'Unknown',
        });
      },
    });

    // Get row count
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result as string;
      const lines = text.split('\n').filter(l => l.trim());
      setPreviewData(prev => prev ? { ...prev, rowCount: lines.length - 1 } : null);
    };
    reader.readAsText(f);
  }, []);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  };

  const processBatch = async () => {
    if (!file || !previewData) return;
    setProcessing(true);
    setProgress(0);
    setError('');

    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        const data = results.data as Record<string, string>[];
        const pipeline = detectPipeline(Object.keys(data[0] || {}));
        const batchId = generateId();
        let highRiskCount = 0;
        let mediumRiskCount = 0;
        let lowRiskCount = 0;
        let criticalCount = 0;
        let totalProb = 0;

        // Process each row with simulated async progress
        let processed = 0;
        const processNext = () => {
          const batchSize = Math.min(50, data.length - processed);
          for (let i = 0; i < batchSize; i++) {
            const row = data[processed + i];
            const input: Record<string, string | number> = {};
            for (const [key, val] of Object.entries(row)) {
              const num = Number(val);
              input[key] = isNaN(num) ? val : num;
            }

            const result = predictAttrition(input, pipeline as 'ibm' | 'synthetic');
            totalProb += result.attritionProbability;

            if (result.riskLevel === 'Critical') criticalCount++;
            else if (result.riskLevel === 'High') highRiskCount++;
            else if (result.riskLevel === 'Medium') mediumRiskCount++;
            else lowRiskCount++;

            const record: PredictionRecord = {
              id: generateId(),
              timestamp: new Date().toISOString(),
              type: 'batch',
              pipeline: pipeline as 'ibm' | 'synthetic',
              input,
              result,
              batchId,
            };
            saveRecord(record);
          }
          processed += batchSize;
          setProgress(Math.round((processed / data.length) * 100));

          if (processed < data.length) {
            requestAnimationFrame(processNext);
          } else {
            const summary: BatchSummary = {
              id: batchId,
              timestamp: new Date().toISOString(),
              pipeline: pipeline as 'ibm' | 'synthetic',
              totalRecords: data.length,
              highRiskCount,
              mediumRiskCount,
              lowRiskCount,
              criticalCount,
              avgAttritionProb: Math.round((totalProb / data.length) * 10) / 10,
              fileName: file.name,
            };
            saveBatch(summary);
            setBatchResult(summary);
            setProcessing(false);
            onBatchComplete();
          }
        };
        processNext();
      },
      error: () => {
        setError('Failed to parse CSV file. Please check the format.');
        setProcessing(false);
      },
    });
  };

  const exportResults = () => {
    // This would export the batch results as CSV
    if (!batchResult) return;
    const csvContent = `AttritionIQ Batch Report
========================
File: ${batchResult.fileName}
Pipeline: ${batchResult.pipeline}
Date: ${new Date(batchResult.timestamp).toLocaleString()}
Total Records: ${batchResult.totalRecords}
Critical Risk: ${batchResult.criticalCount}
High Risk: ${batchResult.highRiskCount}
Medium Risk: ${batchResult.mediumRiskCount}
Low Risk: ${batchResult.lowRiskCount}
Average Attrition Probability: ${batchResult.avgAttritionProb}%
`;
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `attrition_report_${batchResult.id}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const reset = () => {
    setFile(null);
    setPreviewData(null);
    setBatchResult(null);
    setError('');
    setProgress(0);
  };

  return (
    <div className="batch-analyzer">
      <div className="batch-header">
        <div className="batch-header-icon">
          <Upload size={24} />
        </div>
        <div>
          <h3>Drag & Drop CSV Batch Analyzer</h3>
          <p>Upload an entire department CSV for instant AI-powered attrition analysis</p>
        </div>
      </div>

      <div
        className={`drop-zone ${isDragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !file && fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          style={{ display: 'none' }}
          onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])}
        />

        {!file ? (
          <div className="drop-zone-content">
            <div className="drop-zone-icon">
              <FileSpreadsheet size={48} />
            </div>
            <h4>Drop your CSV file here</h4>
            <p>or click to browse files</p>
            <span className="drop-zone-hint">Supports IBM HR & Synthetic pipeline formats • Auto-detection enabled</span>
          </div>
        ) : (
          <div className="drop-zone-file-info">
            <FileSpreadsheet size={32} className="text-indigo-500" />
            <div className="file-details">
              <strong>{file.name}</strong>
              <span>{(file.size / 1024).toFixed(1)} KB</span>
            </div>
            {previewData && (
              <div className="file-preview">
                <div className="preview-tag">
                  <Brain size={14} /> {previewData.pipeline}
                </div>
                <div className="preview-tag">
                  {previewData.columns.length} columns
                </div>
                <div className="preview-tag">
                  {previewData.rowCount} records
                </div>
              </div>
            )}
            {!processing && !batchResult && (
              <div className="file-actions">
                <button className="btn-primary" onClick={(e) => { e.stopPropagation(); processBatch(); }}>
                  <Brain size={16} /> Analyze with AI
                </button>
                <button className="btn-secondary" onClick={(e) => { e.stopPropagation(); reset(); }}>
                  Remove
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {processing && (
        <div className="batch-progress">
          <div className="progress-header">
            <Loader2 size={18} className="animate-spin" />
            <span>Processing {progress}%</span>
          </div>
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${progress}%` }}></div>
          </div>
        </div>
      )}

      {error && (
        <div className="batch-alert error">
          <AlertCircle size={18} /> {error}
        </div>
      )}

      {batchResult && (
        <div className="batch-results">
          <div className="batch-results-header">
            <CheckCircle2 size={20} className="text-emerald-500" />
            <h4>Analysis Complete</h4>
          </div>
          <div className="batch-stats-grid">
            <div className="batch-stat-card">
              <span className="stat-value">{batchResult.totalRecords}</span>
              <span className="stat-label">Total Records</span>
            </div>
            <div className="batch-stat-card critical">
              <span className="stat-value">{batchResult.criticalCount}</span>
              <span className="stat-label">Critical</span>
            </div>
            <div className="batch-stat-card high">
              <span className="stat-value">{batchResult.highRiskCount}</span>
              <span className="stat-label">High Risk</span>
            </div>
            <div className="batch-stat-card medium">
              <span className="stat-value">{batchResult.mediumRiskCount}</span>
              <span className="stat-label">Medium</span>
            </div>
            <div className="batch-stat-card low">
              <span className="stat-value">{batchResult.lowRiskCount}</span>
              <span className="stat-label">Low Risk</span>
            </div>
            <div className="batch-stat-card avg">
              <span className="stat-value">{batchResult.avgAttritionProb}%</span>
              <span className="stat-label">Avg. Probability</span>
            </div>
          </div>
          <div className="batch-results-actions">
            <button className="btn-primary" onClick={exportResults}>
              <Download size={16} /> Export Report
            </button>
            <button className="btn-secondary" onClick={reset}>
              Upload Another
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default BatchAnalyzer;
