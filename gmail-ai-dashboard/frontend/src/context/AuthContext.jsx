import { createContext, useCallback, useContext, useEffect, useState } from 'react';
import { getCurrentUser, logout as apiLogout, eraseAllData as apiErase } from '../services/api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
	const [user, setUser] = useState(null);
	const [loading, setLoading] = useState(true);

	const checkAuth = useCallback(async () => {
		try {
			const { data } = await getCurrentUser();
			setUser(data);
		} catch {
			setUser(null);
		} finally {
			setLoading(false);
		}
	}, []);

	useEffect(() => {
		checkAuth();
	}, [checkAuth]);

	const logout = useCallback(async () => {
		try {
			await apiLogout();
		} catch {
			// Cookie may already be gone
		}
		setUser(null);
	}, []);

	const eraseAllData = useCallback(async () => {
		await apiErase();
		setUser(null);
	}, []);

	const value = {
		user,
		loading,
		isAuthenticated: !!user,
		checkAuth,
		logout,
		eraseAllData,
	};

	return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
	const ctx = useContext(AuthContext);
	if (!ctx) throw new Error('useAuth must be used inside AuthProvider');
	return ctx;
}
