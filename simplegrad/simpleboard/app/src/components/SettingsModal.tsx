/**
 * Settings modal component
 */

import { useState, useEffect } from 'react';
import { X } from 'lucide-react';
import { getApiUrl, setApiUrl } from '../api';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: () => void;
}

export function SettingsModal({ isOpen, onClose, onSave }: SettingsModalProps) {
  const [apiUrlInput, setApiUrlInput] = useState('');

  useEffect(() => {
    if (isOpen) {
      setApiUrlInput(getApiUrl());
    }
  }, [isOpen]);

  const handleSave = () => {
    setApiUrl(apiUrlInput);
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
            <label htmlFor="api-url">API Base URL</label>
            <input
              id="api-url"
              type="text"
              className="form-input"
              value={apiUrlInput}
              onChange={(e) => setApiUrlInput(e.target.value)}
              placeholder="http://localhost:8000"
            />
            <p className="form-help">The URL where the simpleboard server is running.</p>
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
