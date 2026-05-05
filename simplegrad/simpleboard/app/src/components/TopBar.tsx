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
        <img
          src="/simpleboard_hor.svg"
          alt="SimpleBoard"
          className="topbar-logo"
          style={{ height: '26px', width: 'auto' }}
        />
      </div>

      <div className="topbar-right">
        <button
          className="topbar-button"
          onClick={onRefresh}
          disabled={isLoading}
          title="Refresh data"
          aria-label="Refresh data"
        >
          <RefreshCw size={16} className={isLoading ? 'spinning' : ''} />
        </button>

        <button
          className="topbar-button"
          onClick={onOpenSettings}
          title="Settings"
          aria-label="Settings"
        >
          <Settings size={16} />
        </button>
      </div>
    </header>
  );
}
