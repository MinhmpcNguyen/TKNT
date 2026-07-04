import React, { useState } from 'react';
import { loginUser, registerUser, storeAuthToken } from '../api';
import { useEditorStore } from '../store';

type AuthMode = 'login' | 'register';

export const AuthScreen: React.FC = () => {
  const setAuthSession = useEditorStore((state) => state.setAuthSession);
  const appendRunLog = useEditorStore((state) => state.appendRunLog);
  const [mode, setMode] = useState<AuthMode>('login');
  const [displayName, setDisplayName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const response =
        mode === 'register'
          ? await registerUser({
              email,
              password,
              display_name: displayName.trim() || null,
            })
          : await loginUser({ email, password });
      storeAuthToken(response.access_token);
      setAuthSession(response.access_token, response.user);
      appendRunLog(
        `${mode === 'register' ? 'Registered' : 'Signed in'} as ${response.user.email}`
      );
    } catch (submitError) {
      setError(
        submitError instanceof Error ? submitError.message : 'Authentication failed.'
      );
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="auth-screen">
      <div className="auth-card">
        <div className="auth-card-header">
          <div>
            <h1>TKNT Floorplan Editor</h1>
            <p>
              Sign in to save layouts, keep your image renders, and use the shared
              inventory.
            </p>
          </div>
          <div className="auth-mode-toggle">
            <button
              className={mode === 'login' ? 'active' : ''}
              onClick={() => setMode('login')}
              type="button"
            >
              Sign In
            </button>
            <button
              className={mode === 'register' ? 'active' : ''}
              onClick={() => setMode('register')}
              type="button"
            >
              Register
            </button>
          </div>
        </div>
        <form className="auth-form" onSubmit={handleSubmit}>
          {mode === 'register' ? (
            <label>
              Display name
              <input
                value={displayName}
                onChange={(event) => setDisplayName(event.target.value)}
                placeholder="Minh"
                type="text"
              />
            </label>
          ) : null}
          <label>
            Email
            <input
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              placeholder="you@example.com"
              type="email"
              required
            />
          </label>
          <label>
            Password
            <input
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder="At least 8 characters"
              type="password"
              minLength={8}
              required
            />
          </label>
          {error ? <div className="auth-error">{error}</div> : null}
          <button className="primary-button" disabled={submitting} type="submit">
            {submitting
              ? mode === 'register'
                ? 'Creating account...'
                : 'Signing in...'
              : mode === 'register'
                ? 'Create account'
                : 'Sign in'}
          </button>
        </form>
      </div>
    </div>
  );
};
