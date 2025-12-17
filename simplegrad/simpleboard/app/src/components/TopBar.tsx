/**
 * Top navigation bar component
 */

import { RefreshCw, Settings } from 'lucide-react';

interface TopBarProps {
  onRefresh: () => void;
  onOpenSettings: () => void;
  isLoading: boolean;
}

export function TopBar({
  onRefresh,
  onOpenSettings,
  isLoading
}: TopBarProps) {
  return (
    <header className="topbar">
      <div className="topbar-left">
        <img 
          src="/simplegrad_v1.svg" 
          alt="SimpleGrad" 
          className="topbar-logo"
        />
      </div>

      <div className="topbar-right">
        {/* Refresh button */}
        <button 
          className="topbar-button"
          onClick={onRefresh}
          disabled={isLoading}
          title="Refresh data"
        >
          <RefreshCw size={16} className={isLoading ? 'spinning' : ''} />
        </button>

        {/* Settings button */}
        <button 
          className="topbar-button"
          onClick={onOpenSettings}
          title="Settings"
        >
          <Settings size={16} />
        </button>
      </div>
    </header>
  );
}
