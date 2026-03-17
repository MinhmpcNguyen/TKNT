import React from 'react';
import { useEditorStore } from '../store';

export const ValidationPanel: React.FC = () => {
  const violations = useEditorStore((s) => s.violations);

  return (
    <aside className="panel validation-panel">
      <h3>Validation</h3>
      {violations.length === 0 && <p>No issues</p>}
      <ul>
        {violations.map((v, idx) => (
          <li key={`${v.code}-${idx}`} className={`severity-${v.severity}`}>
            <strong>{v.code}</strong>
            <div>{v.message}</div>
            {v.suggested_fixes.length > 0 && (
              <div className="fixes">
                Suggested: {v.suggested_fixes.map((f) => f.action_type).join(', ')}
              </div>
            )}
          </li>
        ))}
      </ul>
    </aside>
  );
};
