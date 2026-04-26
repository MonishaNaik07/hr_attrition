import React from 'react';
import { Brain, LayoutDashboard, FileSearch, History, LogOut } from 'lucide-react';

export type TabType = 'dashboard' | 'predict' | 'history';

interface HeaderProps {
  username: string;
  activeTab: TabType;
  onTabChange: (tab: TabType) => void;
  onLogout: () => void;
}

const Header: React.FC<HeaderProps> = ({ username, activeTab, onTabChange, onLogout }) => {
  const tabs: { id: TabType; label: string; icon: React.ReactNode }[] = [
    { id: 'dashboard', label: 'Dashboard', icon: <LayoutDashboard size={18} /> },
    { id: 'predict', label: 'Predict', icon: <FileSearch size={18} /> },
    { id: 'history', label: 'History', icon: <History size={18} /> },
  ];

  return (
    <header className="app-header">
      <div className="header-inner">
        <div className="header-brand">
          <div className="header-logo">
            <Brain size={28} />
          </div>
          <div className="header-brand-text">
            <h1>AttritionIQ</h1>
            <span className="header-tagline">Enterprise AI Platform</span>
          </div>
        </div>

        <nav className="header-nav">
          {tabs.map(tab => (
            <button
              key={tab.id}
              className={`header-nav-btn ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => onTabChange(tab.id)}
            >
              {tab.icon}
              <span>{tab.label}</span>
            </button>
          ))}
        </nav>

        <div className="header-user">
          <div className="header-user-avatar">
            {username.charAt(0).toUpperCase()}
          </div>
          <span className="header-user-name">{username}</span>
          <button className="header-logout-btn" onClick={onLogout} title="Sign Out">
            <LogOut size={18} />
          </button>
        </div>
      </div>
    </header>
  );
};

export default Header;
