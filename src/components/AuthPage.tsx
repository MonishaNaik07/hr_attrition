import React, { useState } from 'react';
import { Eye, EyeOff, Shield, Brain, BarChart3, Users, LogIn, UserPlus } from 'lucide-react';
import { loginUser, registerUser } from '../utils/mockApi';

interface AuthPageProps {
  onLogin: (username: string) => void;
}

const AuthPage: React.FC<AuthPageProps> = ({ onLogin }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccess('');

    if (!username.trim() || !password.trim()) {
      setError('Please fill in all fields.');
      return;
    }

    if (isLogin) {
      const result = loginUser(username, password);
      if (result.success) {
        onLogin(username.trim().toLowerCase());
      } else {
        setError(result.message);
      }
    } else {
      if (password !== confirmPassword) {
        setError('Passwords do not match.');
        return;
      }
      if (password.length < 4) {
        setError('Password must be at least 4 characters.');
        return;
      }
      const result = registerUser(username, password);
      if (result.success) {
        setSuccess(result.message);
        setIsLogin(true);
        setPassword('');
        setConfirmPassword('');
      } else {
        setError(result.message);
      }
    }
  };

  return (
    <div className="auth-container">
      {/* Left Panel - Visual */}
      <div className="auth-visual-panel">
        <div className="auth-visual-overlay">
          <div className="auth-visual-content">
            <div className="auth-logo-icon">
              <Brain size={56} strokeWidth={1.5} />
            </div>
            <h1 className="auth-visual-title">AttritionIQ</h1>
            <p className="auth-visual-subtitle">Enterprise Employee Attrition AI Platform</p>
            <div className="auth-visual-divider"></div>
            <p className="auth-visual-description">
              Harness the power of Dual-Brain Machine Learning to predict, explain, and prevent employee attrition across your entire organization.
            </p>
            <div className="auth-features">
              <div className="auth-feature-item">
                <div className="auth-feature-icon"><Brain size={20} /></div>
                <div>
                  <h4>Dual-Brain AI Engine</h4>
                  <p>VotingClassifier with 5 model architectures</p>
                </div>
              </div>
              <div className="auth-feature-item">
                <div className="auth-feature-icon"><Shield size={20} /></div>
                <div>
                  <h4>Flight Risk Analytics</h4>
                  <p>Explainable AI with actionable HR solutions</p>
                </div>
              </div>
              <div className="auth-feature-item">
                <div className="auth-feature-icon"><BarChart3 size={20} /></div>
                <div>
                  <h4>Batch Processing</h4>
                  <p>Analyze entire departments in seconds</p>
                </div>
              </div>
              <div className="auth-feature-item">
                <div className="auth-feature-icon"><Users size={20} /></div>
                <div>
                  <h4>Dual Pipeline</h4>
                  <p>IBM HR & Synthetic data architectures</p>
                </div>
              </div>
            </div>
          </div>
        </div>
        {/* Animated background elements */}
        <div className="auth-bg-circle c1"></div>
        <div className="auth-bg-circle c2"></div>
        <div className="auth-bg-circle c3"></div>
        <div className="auth-bg-grid"></div>
      </div>

      {/* Right Panel - Form */}
      <div className="auth-form-panel">
        <div className="auth-form-wrapper">
          <div className="auth-form-header">
            <div className="auth-form-icon">
              <Brain size={32} className="text-indigo-500" />
            </div>
            <h2>{isLogin ? 'Welcome Back' : 'Create Account'}</h2>
            <p>{isLogin ? 'Sign in to access your AI dashboard' : 'Register to start predicting attrition'}</p>
          </div>

          <div className="auth-toggle">
            <button
              className={`auth-toggle-btn ${isLogin ? 'active' : ''}`}
              onClick={() => { setIsLogin(true); setError(''); setSuccess(''); }}
            >
              <LogIn size={16} /> Sign In
            </button>
            <button
              className={`auth-toggle-btn ${!isLogin ? 'active' : ''}`}
              onClick={() => { setIsLogin(false); setError(''); setSuccess(''); }}
            >
              <UserPlus size={16} /> Register
            </button>
          </div>

          {error && <div className="auth-alert error">{error}</div>}
          {success && <div className="auth-alert success">{success}</div>}

          <form onSubmit={handleSubmit} className="auth-form">
            <div className="auth-field">
              <label htmlFor="username">Username</label>
              <div className="auth-input-wrapper">
                <input
                  id="username"
                  type="text"
                  placeholder="Enter your username"
                  value={username}
                  onChange={e => setUsername(e.target.value)}
                  autoComplete="username"
                />
              </div>
            </div>

            <div className="auth-field">
              <label htmlFor="password">Password</label>
              <div className="auth-input-wrapper">
                <input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  placeholder="Enter your password"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  autoComplete={isLogin ? 'current-password' : 'new-password'}
                />
                <button
                  type="button"
                  className="auth-eye-btn"
                  onClick={() => setShowPassword(!showPassword)}
                  tabIndex={-1}
                >
                  {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                </button>
              </div>
            </div>

            {!isLogin && (
              <div className="auth-field">
                <label htmlFor="confirmPassword">Confirm Password</label>
                <div className="auth-input-wrapper">
                  <input
                    id="confirmPassword"
                    type={showPassword ? 'text' : 'password'}
                    placeholder="Confirm your password"
                    value={confirmPassword}
                    onChange={e => setConfirmPassword(e.target.value)}
                    autoComplete="new-password"
                  />
                </div>
              </div>
            )}

            <button type="submit" className="auth-submit-btn">
              {isLogin ? (
                <><LogIn size={18} /> Sign In to Dashboard</>
              ) : (
                <><UserPlus size={18} /> Create Account</>
              )}
            </button>
          </form>

          <div className="auth-footer">
            <p>
              {isLogin ? "Don't have an account? " : 'Already have an account? '}
              <button
                className="auth-switch-link"
                onClick={() => { setIsLogin(!isLogin); setError(''); setSuccess(''); }}
              >
                {isLogin ? 'Register here' : 'Sign in here'}
              </button>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AuthPage;
