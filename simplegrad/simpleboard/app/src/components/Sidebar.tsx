import React, { useState } from 'react';
import type { RunInfo } from '../api';
import { Activity, Clock, CheckCircle, XCircle, PlayCircle, Database } from 'lucide-react';

interface SidebarProps {
  runs: RunInfo[];
  selectedRunId: number | null;
  onSelectRun: (runId: number) => void;
  isOpen: boolean;
  databases: string[];
  selectedDatabase: string | null;
  onSelectDatabase: (dbName: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ 
  runs, 
  selectedRunId, 
  onSelectRun, 
  isOpen,
  databases,
  selectedDatabase,
  onSelectDatabase
}) => {
  const [showDatabaseMenu, setShowDatabaseMenu] = useState(false);

  if (!isOpen) return null;

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle size={16} color="#3EC291" />;
      case 'failed': return <XCircle size={16} color="#F7733C" />;
      case 'running': return <PlayCircle size={16} color="#FFC515" />;
      default: return <Activity size={16} />;
    }
  };

  return (
    <div className="nb-box" style={{ 
      width: '300px', 
      height: '100%', 
      overflowY: 'auto',
      display: 'flex',
      flexDirection: 'column',
      borderRight: 'none',
      zIndex: 10
    }}>
      {/* Database Section */}
      <div style={{ padding: '16px', borderBottom: '2px solid #1B1E20', background: '#FAFAFA' }}>
        <h2 style={{ margin: 0, marginBottom: '8px', fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Database size={18} /> EXPERIMENT
        </h2>
        {databases.length > 0 ? (
          <div style={{ position: 'relative' }}>
            <button
              onClick={() => setShowDatabaseMenu(!showDatabaseMenu)}
              style={{
                width: '100%',
                padding: '8px 12px',
                border: '2px solid #1B1E20',
                borderRadius: '4px',
                background: 'white',
                cursor: 'pointer',
                fontWeight: 'bold',
                textAlign: 'left',
                fontSize: '0.9rem'
              }}
            >
              {selectedDatabase || 'Select...'}
            </button>
            {showDatabaseMenu && (
              <div style={{
                position: 'absolute',
                top: '100%',
                left: 0,
                right: 0,
                background: 'white',
                border: '2px solid #1B1E20',
                borderRadius: '4px',
                marginTop: '4px',
                zIndex: 20,
                boxShadow: '2px 2px 8px rgba(27,30,32,0.15)'
              }}>
                {databases.map(dbName => (
                  <button
                    key={dbName}
                    onClick={() => {
                      onSelectDatabase(dbName);
                      setShowDatabaseMenu(false);
                    }}
                    style={{
                      display: 'block',
                      width: '100%',
                      padding: '8px 12px',
                      textAlign: 'left',
                      border: 'none',
                      background: selectedDatabase === dbName ? '#F7B7CF' : 'white',
                      cursor: 'pointer',
                      borderBottom: '1px solid #eee',
                      fontSize: '0.85rem',
                      fontWeight: selectedDatabase === dbName ? 'bold' : 'normal'
                    }}
                  >
                    {dbName}
                  </button>
                ))}
              </div>
            )}
          </div>
        ) : (
          <div style={{ 
            padding: '8px 12px', 
            color: '#666', 
            fontSize: '0.85rem',
            fontStyle: 'italic'
          }}>
            No databases found
          </div>
        )}
      </div>

      <div style={{ padding: '16px', borderBottom: '3px solid #1B1E20', background: '#FAFAFA' }}>
        <h2 style={{ margin: 0, fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Activity /> RUNS
        </h2>
      </div>
      <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
        {runs.map(run => (
          <div 
            key={run.run_id}
            onClick={() => onSelectRun(run.run_id)}
            className="nb-box"
            style={{ 
              padding: '12px', 
              cursor: 'pointer',
              background: selectedRunId === run.run_id ? '#F7B7CF' : 'white',
              borderColor: '#1B1E20',
              boxShadow: selectedRunId === run.run_id ? '2px 2px 0px #1B1E20' : '4px 4px 0px #1B1E20',
              transform: selectedRunId === run.run_id ? 'translate(2px, 2px)' : 'none'
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <span style={{ fontWeight: 'bold' }}>#{run.run_id}</span>
              {getStatusIcon(run.status)}
            </div>
            <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>{run.name}</div>
            <div style={{ fontSize: '0.8rem', color: '#666', display: 'flex', alignItems: 'center', gap: '4px' }}>
              <Clock size={12} />
              {run.created_at}
            </div>
          </div>
        ))}
        {runs.length === 0 && (
          <div style={{ textAlign: 'center', padding: '20px', color: '#666' }}>
            No runs found.
          </div>
        )}
      </div>
    </div>
  );
};
