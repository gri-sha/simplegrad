import React, { useState, useEffect } from 'react';
import { getApiUrl, setApiUrl } from '../api';
import { X } from 'lucide-react';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: () => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onClose, onSave }) => {
  const [url, setUrl] = useState('');

  useEffect(() => {
    if (isOpen) {
      setUrl(getApiUrl());
    }
  }, [isOpen]);

  const handleSave = () => {
    setApiUrl(url);
    onSave();
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0,0,0,0.5)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000
    }}>
      <div className="nb-box" style={{ width: '400px', padding: '24px', position: 'relative' }}>
        <button 
          onClick={onClose}
          className="nb-button ghost"
          style={{ position: 'absolute', top: '8px', right: '8px', padding: '4px' }}
        >
          <X size={20} />
        </button>
        
        <h2 style={{ marginTop: 0, marginBottom: '24px' }}>SETTINGS</h2>
        
        <div style={{ marginBottom: '24px' }}>
          <label style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold' }}>
            API Base URL
          </label>
          <input 
            type="text" 
            className="nb-input"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="http://localhost:8000"
          />
          <p style={{ fontSize: '0.8rem', color: '#666', marginTop: '8px' }}>
            Point this to your running SimpleGrad server.
          </p>
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px' }}>
          <button onClick={onClose} className="nb-button ghost">Cancel</button>
          <button onClick={handleSave} className="nb-button">Save Changes</button>
        </div>
      </div>
    </div>
  );
};
