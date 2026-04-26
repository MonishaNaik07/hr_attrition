import React, { useState, useCallback } from 'react';
import AuthPage from './components/AuthPage';
import Header, { TabType } from './components/Header';
import Dashboard from './components/Dashboard';
import PredictionForm from './components/PredictionForm';
import HistoryView from './components/History';
import { getCurrentUser, logoutUser } from './utils/mockApi';

const App: React.FC = () => {
  const [user, setUser] = useState(() => getCurrentUser());
  const [activeTab, setActiveTab] = useState<TabType>('dashboard');
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const handleLogin = useCallback((username: string) => {
    setUser({ username, loggedIn: true });
  }, []);

  const handleLogout = useCallback(() => {
    logoutUser();
    setUser(null);
  }, []);

  const triggerRefresh = useCallback(() => {
    setRefreshTrigger(prev => prev + 1);
  }, []);

  if (!user?.loggedIn) {
    return <AuthPage onLogin={handleLogin} />;
  }

  return (
    <div className="app-layout">
      <Header
        username={user.username}
        activeTab={activeTab}
        onTabChange={setActiveTab}
        onLogout={handleLogout}
      />
      <main className="app-main">
        {activeTab === 'dashboard' && (
          <Dashboard refreshTrigger={refreshTrigger} onTabChange={setActiveTab} />
        )}
        {activeTab === 'predict' && (
          <PredictionForm onPrediction={triggerRefresh} />
        )}
        {activeTab === 'history' && (
          <HistoryView refreshTrigger={refreshTrigger} />
        )}
      </main>
      <footer className="app-footer">
        <p>AttritionIQ Enterprise Platform • Dual-Brain AI Engine • VotingClassifier Ensemble</p>
      </footer>
    </div>
  );
};

export default App;
