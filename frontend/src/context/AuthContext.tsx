import React, { createContext, useContext, useState, useEffect } from 'react';
import { UserInfo } from '../types';
import { api, setAuthToken, getStoredToken } from '../services/api';

interface AuthContextType {
  user: UserInfo | null;
  token: string | null;
  loading: boolean;
  loginAsDemo: (role: 'viewer' | 'analyst' | 'admin') => Promise<void>;
  logout: () => void;
  hasRole: (role: string) => boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<UserInfo | null>(null);
  const [token, setToken] = useState<string | null>(getStoredToken());
  const [loading, setLoading] = useState<boolean>(true);

  const fetchUserInfo = async () => {
    try {
      const userInfo = await api.getUserInfo();
      setUser(userInfo);
    } catch {
      // Failed to validate session token, reset
      setAuthToken(null);
      setToken(null);
      setUser(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (token) {
      fetchUserInfo();
    } else {
      // Default to auto-login as analyst in demo mode for frictionless experience
      loginAsDemo('analyst').finally(() => setLoading(false));
    }
  }, []);

  const loginAsDemo = async (role: 'viewer' | 'analyst' | 'admin') => {
    setLoading(true);
    try {
      const res = await api.getDemoToken(role);
      setAuthToken(res.access_token);
      setToken(res.access_token);
      const userInfo = await api.getUserInfo();
      setUser(userInfo);
    } catch (err) {
      console.error('Demo login failed:', err);
      setUser({
        sub: `demo-${role}`,
        email: `demo-${role}@tensorlens.local`,
        roles: [role],
        is_demo: true,
      });
    } finally {
      setLoading(false);
    }
  };

  const logout = () => {
    setAuthToken(null);
    setToken(null);
    setUser(null);
  };

  const hasRole = (role: string): boolean => {
    if (!user) return false;
    if (user.roles.includes('admin')) return true;
    return user.roles.includes(role);
  };

  return (
    <AuthContext.Provider value={{ user, token, loading, loginAsDemo, logout, hasRole }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
