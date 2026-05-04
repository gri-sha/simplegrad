/**
 * Top navigation bar component
 */

import { RefreshCw, Settings, Sun, Moon } from 'lucide-react';

interface TopBarProps {
  onRefresh: () => void;
  onOpenSettings: () => void;
  isLoading: boolean;
  theme: 'light' | 'dark';
  onToggleTheme: () => void;
}

export function TopBar({ onRefresh, onOpenSettings, isLoading, theme, onToggleTheme }: TopBarProps) {
  return (
    <header className="topbar">
      <div className="topbar-left">
        <span className="topbar-title topbar-brand">SimpleBoard</span>
      </div>

      <div className="topbar-right">
        {/* Refresh button */}
        <button
          className="topbar-button"
          onClick={onRefresh}
          disabled={isLoading}
          title="Refresh data"
          aria-label="Refresh data"
        >
          <RefreshCw size={16} className={isLoading ? 'spinning' : ''} />
        </button>

        {/* Theme toggle */}
        <button
          className="topbar-button"
          onClick={onToggleTheme}
          title={theme === 'light' ? 'Switch to Dark Mode' : 'Switch to Light Mode'}
          aria-label="Toggle theme"
        >
          {theme === 'light' ? <Moon size={16} /> : <Sun size={16} />}
        </button>

        {/* Settings button */}
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
