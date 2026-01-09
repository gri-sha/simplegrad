/**
 * Top navigation bar component
 */

import { RefreshCw, Settings } from 'lucide-react';

interface TopBarProps {
  onRefresh: () => void;
  onOpenSettings: () => void;
  isLoading: boolean;
}

export function TopBar({ onRefresh, onOpenSettings, isLoading }: TopBarProps) {
  return (
    <header className="topbar">
      <div className="topbar-left">
        <img src="/simpleboard_v1.svg" alt="SimpleGrad" className="topbar-logo" />
      </div>

      <div className="topbar-right">
        {/* Refresh button */}
        <button
          className="topbar-button"
          onClick={onRefresh}
          disabled={isLoading}
          title="Refresh data"
        >
          <RefreshCw size={20} className={isLoading ? 'spinning' : ''} />
        </button>

        {/* Settings button */}
        <button className="topbar-button" onClick={onOpenSettings} title="Settings">
          <Settings size={20} />
        </button>
      </div>
    </header>
  );
}
