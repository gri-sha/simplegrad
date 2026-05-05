/**
 * Settings modal component
 */

import { useState, useEffect } from 'react';
import { X } from 'lucide-react';
import { getApiUrl, setApiUrl, clearApiUrl } from '../api';
import { api } from '../api';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: () => void;
}

export function SettingsModal({ isOpen, onClose, onSave }: SettingsModalProps) {
  const [apiUrlInput, setApiUrlInput] = useState('');
  const [expDirInput, setExpDirInput] = useState('');
  const [originalExpDir, setOriginalExpDir] = useState('');
  const [expDirError, setExpDirError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      setApiUrlInput(getApiUrl());
      setExpDirError(null);
      api.getConfig()
        .then((cfg) => {
          setExpDirInput(cfg.exp_dir);
          setOriginalExpDir(cfg.exp_dir);
        })
        .catch(() => {});
    }
  }, [isOpen]);

  const handleSave = async () => {
    if (apiUrlInput.trim() === '') {
      clearApiUrl();
    } else {
      setApiUrl(apiUrlInput);
    }
    if (expDirInput.trim() && expDirInput.trim() !== originalExpDir) {
      try {
        await api.updateExpDir(expDirInput.trim());
      } catch (err) {
        setExpDirError('Failed to update experiments directory.');
        return;
      }
    }
    onSave();
    onClose();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose} onKeyDown={handleKeyDown}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Settings</h2>
          <button className="modal-close" onClick={onClose}>
            <X size={20} />
          </button>
        </div>

        <div className="modal-body">
          <div className="form-group">
            <label htmlFor="exp-dir">Experiments directory</label>
            <input
              id="exp-dir"
              type="text"
              className="form-input"
              value={expDirInput}
              onChange={(e) => { setExpDirInput(e.target.value); setExpDirError(null); }}
              placeholder="/path/to/experiments"
            />
            {expDirError && (
              <p className="form-help" style={{ color: 'var(--color-orange)' }}>{expDirError}</p>
            )}
            <p className="form-help">
              Directory where experiment databases are stored. Takes effect immediately without a restart.
            </p>
          </div>

          <div className="form-group">
            <label htmlFor="api-url">API base URL</label>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <input
                id="api-url"
                type="text"
                className="form-input"
                style={{ flex: 1 }}
                value={apiUrlInput}
                onChange={(e) => setApiUrlInput(e.target.value)}
                placeholder="Leave empty to use same-origin (default)"
              />
              <button
                className="btn btn-secondary"
                type="button"
                onClick={() => setApiUrlInput('')}
                title="Reset to default (same-origin)"
              >
                Reset
              </button>
            </div>
            <p className="form-help">
              Leave empty to connect to the server that served this page (default).
              Set an explicit URL only when connecting to a remote simpleboard instance.
            </p>
          </div>
        </div>

        <div className="modal-footer">
          <button className="btn btn-secondary" onClick={onClose}>
            Cancel
          </button>
          <button className="btn btn-primary" onClick={handleSave}>
            Save Changes
          </button>
        </div>
      </div>
    </div>
  );
}
