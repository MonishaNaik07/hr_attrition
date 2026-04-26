import React, { useState, useMemo } from 'react';
import { History as HistoryIcon, Clock, Brain, Cpu, Trash2, ChevronDown, ChevronUp, Search, Download, Filter } from 'lucide-react';
import { getHistory, getBatches, clearAllHistory } from '../utils/mockApi';
interface HistoryProps {
  refreshTrigger: number;
}

const HistoryView: React.FC<HistoryProps> = ({ refreshTrigger }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState<'all' | 'individual' | 'batch'>('all');
  const [filterRisk, setFilterRisk] = useState<string>('all');
  const [expandedRecord, setExpandedRecord] = useState<string | null>(null);
  const [showBatches, setShowBatches] = useState(false);
  const [confirmClear, setConfirmClear] = useState(false);

  const history = useMemo(() => getHistory(), [refreshTrigger]);
  const batches = useMemo(() => getBatches(), [refreshTrigger]);

  const filteredHistory = useMemo(() => {
    return history.filter(record => {
      if (filterType !== 'all' && record.type !== filterType) return false;
      if (filterRisk !== 'all' && record.result.riskLevel !== filterRisk) return false;
      if (searchTerm) {
        const search = searchTerm.toLowerCase();
        const inputStr = JSON.stringify(record.input).toLowerCase();
        if (!inputStr.includes(search)) return false;
      }
      return true;
    });
  }, [history, filterType, filterRisk, searchTerm]);

  const handleClearAll = () => {
    if (confirmClear) {
      clearAllHistory();
      setConfirmClear(false);
    } else {
      setConfirmClear(true);
      setTimeout(() => setConfirmClear(false), 3000);
    }
  };

  const exportHistory = () => {
    const csvRows = ['ID,Timestamp,Type,Pipeline,Risk Level,Attrition Probability %'];
    filteredHistory.forEach(r => {
      csvRows.push(`${r.id},${r.timestamp},${r.type},${r.pipeline},${r.result.riskLevel},${r.result.attritionProbability}`);
    });
    const blob = new Blob([csvRows.join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'attrition_history.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  const riskBadgeClass = (risk: string) => {
    switch (risk) {
      case 'Critical': return 'badge-critical';
      case 'High': return 'badge-high';
      case 'Medium': return 'badge-medium';
      default: return 'badge-low';
    }
  };

  return (
    <div className="history-container">
      <div className="history-header">
        <div className="history-title">
          <HistoryIcon size={24} />
          <div>
            <h3>Records & History</h3>
            <p>Cross-compare past predictions and batch runs</p>
          </div>
        </div>
        <div className="history-actions">
          <button className="btn-secondary" onClick={exportHistory}>
            <Download size={16} /> Export
          </button>
          <button
            className={`btn-secondary ${confirmClear ? 'btn-danger' : ''}`}
            onClick={handleClearAll}
          >
            <Trash2 size={16} /> {confirmClear ? 'Confirm Clear' : 'Clear All'}
          </button>
        </div>
      </div>

      {/* Batch Summaries */}
      {batches.length > 0 && (
        <div className="batch-section">
          <button className="batch-toggle" onClick={() => setShowBatches(!showBatches)}>
            <div className="batch-toggle-left">
              <Brain size={18} />
              <span>Batch Processing History ({batches.length} batches)</span>
            </div>
            {showBatches ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
          </button>
          {showBatches && (
            <div className="batch-list">
              {batches.map(batch => (
                <div key={batch.id} className="batch-card">
                  <div className="batch-card-header">
                    <span className="batch-filename">{batch.fileName}</span>
                    <span className="batch-date">
                      <Clock size={14} /> {new Date(batch.timestamp).toLocaleString()}
                    </span>
                  </div>
                  <div className="batch-card-stats">
                    <div className="batch-mini-stat">
                      <span className="batch-mini-value">{batch.totalRecords}</span>
                      <span className="batch-mini-label">Records</span>
                    </div>
                    <div className="batch-mini-stat">
                      <span className={`batch-mini-value ${batch.pipeline}`}>{batch.pipeline.toUpperCase()}</span>
                      <span className="batch-mini-label">Pipeline</span>
                    </div>
                    <div className="batch-mini-stat critical">
                      <span className="batch-mini-value">{batch.criticalCount}</span>
                      <span className="batch-mini-label">Critical</span>
                    </div>
                    <div className="batch-mini-stat high">
                      <span className="batch-mini-value">{batch.highRiskCount}</span>
                      <span className="batch-mini-label">High</span>
                    </div>
                    <div className="batch-mini-stat medium">
                      <span className="batch-mini-value">{batch.mediumRiskCount}</span>
                      <span className="batch-mini-label">Medium</span>
                    </div>
                    <div className="batch-mini-stat low">
                      <span className="batch-mini-value">{batch.lowRiskCount}</span>
                      <span className="batch-mini-label">Low</span>
                    </div>
                    <div className="batch-mini-stat avg">
                      <span className="batch-mini-value">{batch.avgAttritionProb}%</span>
                      <span className="batch-mini-label">Avg Risk</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Filters */}
      <div className="history-filters">
        <div className="filter-search">
          <Search size={16} />
          <input
            type="text"
            placeholder="Search records..."
            value={searchTerm}
            onChange={e => setSearchTerm(e.target.value)}
          />
        </div>
        <div className="filter-group">
          <Filter size={16} />
          <select value={filterType} onChange={e => setFilterType(e.target.value as 'all' | 'individual' | 'batch')}>
            <option value="all">All Types</option>
            <option value="individual">Individual</option>
            <option value="batch">Batch</option>
          </select>
          <select value={filterRisk} onChange={e => setFilterRisk(e.target.value)}>
            <option value="all">All Risk Levels</option>
            <option value="Critical">Critical</option>
            <option value="High">High</option>
            <option value="Medium">Medium</option>
            <option value="Low">Low</option>
          </select>
        </div>
      </div>

      {/* Records List */}
      <div className="history-list">
        {filteredHistory.length === 0 ? (
          <div className="history-empty">
            <HistoryIcon size={48} />
            <h4>No records found</h4>
            <p>Run predictions or batch analyses to see results here</p>
          </div>
        ) : (
          filteredHistory.map(record => (
            <div
              key={record.id}
              className={`history-record ${expandedRecord === record.id ? 'expanded' : ''}`}
            >
              <div
                className="history-record-summary"
                onClick={() => setExpandedRecord(expandedRecord === record.id ? null : record.id)}
              >
                <div className="record-left">
                  <span className={`risk-badge ${riskBadgeClass(record.result.riskLevel)}`}>
                    {record.result.riskLevel}
                  </span>
                  <span className={`record-type ${record.type}`}>
                    {record.type === 'individual' ? <Cpu size={14} /> : <Brain size={14} />}
                    {record.type}
                  </span>
                  <span className={`record-pipeline ${record.pipeline}`}>
                    {record.pipeline.toUpperCase()}
                  </span>
                  <span className="record-prob">{record.result.attritionProbability}%</span>
                </div>
                <div className="record-right">
                  <span className="record-time">
                    <Clock size={14} />
                    {new Date(record.timestamp).toLocaleString()}
                  </span>
                  {expandedRecord === record.id ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                </div>
              </div>
              {expandedRecord === record.id && (
                <div className="history-record-details">
                  <div className="detail-section">
                    <h5>Input Features</h5>
                    <div className="detail-grid">
                      {Object.entries(record.input).map(([key, val]) => (
                        <div key={key} className="detail-item">
                          <span className="detail-key">{key}</span>
                          <span className="detail-value">{String(val)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="detail-section">
                    <h5>🔍 Risk Factors</h5>
                    <ul className="detail-list">
                      {record.result.riskFactors.map((f, i) => (
                        <li key={i}>{f}</li>
                      ))}
                    </ul>
                  </div>
                  <div className="detail-section">
                    <h5>💡 Retention Actions</h5>
                    <ul className="detail-list actions">
                      {record.result.retentionActions.map((a, i) => (
                        <li key={i}>{a}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default HistoryView;
