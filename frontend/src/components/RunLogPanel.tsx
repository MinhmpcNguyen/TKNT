import React from 'react';
import { useEditorStore } from '../store';

export const RunLogPanel: React.FC = () => {
  const { runLogs, runStatus, runIterations, additionRequests, caseId, pipelineStage } = useEditorStore();

  return (
    <aside className="panel run-log-panel">
      <h3>Run Log</h3>
      {caseId && <div className="muted">Case: {caseId}</div>}
      {pipelineStage && <div className="muted">Stage: {pipelineStage}</div>}
      {runStatus && (
        <div className="muted">
          Status: {runStatus}
          {runIterations !== null ? ` | Iterations: ${runIterations}` : ''}
        </div>
      )}
      {additionRequests.length > 0 && (
        <div className="addition-requests">
          <div className="muted">Addition requests</div>
          <ul>
            {additionRequests.map((req, idx) => (
              <li key={`${idx}-${String(req['target_category'] ?? 'item')}`}>
                <strong>{String(req['target_category'] ?? 'item')}</strong>
                {req['count'] ? ` x${String(req['count'])}` : ''}
                {req['target_asset_id'] ? ` (${String(req['target_asset_id'])})` : ''}
                {req['reason'] ? ` — ${String(req['reason'])}` : ''}
                {req['priority'] ? ` [${String(req['priority'])}]` : ''}
              </li>
            ))}
          </ul>
        </div>
      )}
      {runLogs.length === 0 ? (
        <p className="muted">No logs yet.</p>
      ) : (
        <div className="log-list">
          {runLogs.map((line, idx) => (
            <div key={`${idx}-${line.slice(0, 12)}`} className="log-line">
              {line}
            </div>
          ))}
        </div>
      )}
    </aside>
  );
};
