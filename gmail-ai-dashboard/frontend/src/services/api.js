	import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const api = axios.create({ baseURL: API_BASE, withCredentials: true, headers: { 'Content-Type': 'application/json' } });

const getClientTimezone = () => {
	try {
		return Intl?.DateTimeFormat ? Intl.DateTimeFormat().resolvedOptions().timeZone : undefined;
	} catch {
		return undefined;
	}
};

api.interceptors.request.use((config) => {
	const tz = getClientTimezone();
	if (tz) {
		config.headers = config.headers || {};
		config.headers['X-Timezone'] = tz;
		if (config.params && typeof config.params === 'object' && !config.params.tz) {
			config.params.tz = tz;
		}
	}
	return config;
});

export const getCurrentUser = () => api.get('/api/auth/me');
export const logout = () => api.post('/api/auth/logout');
export const eraseAllData = () => api.delete('/api/auth/erase');

export const startSync = () => api.post('/api/gmail/sync');
export const getSyncStatus = () => api.get('/api/gmail/sync/status');

export const fetchEmails = (params) => api.get('/api/emails', { params });
export const fetchEmail = (id) => api.get(`/api/emails/${id}`);
export const fetchLabels = () => api.get('/api/labels');
export const fetchAnalyticsSummary = (range, tz) => {
	const userTz = tz || getClientTimezone();
	return api.get('/api/analytics/summary', { params: { range, ...(userTz ? { tz: userTz } : {}) } });
};
export const fetchAnalyticsCharts = (range, tz) => {
	const userTz = tz || getClientTimezone();
	return api.get('/api/analytics/charts', { params: { range, ...(userTz ? { tz: userTz } : {}) } });
};

export const trainModel = () => api.post('/api/classification/train');

// Export endpoints — download emails from the local database
export const getExportStats = () => api.get('/api/export/stats');
export const getExportCSVUrl = (params = {}) => {
	const query = new URLSearchParams(params).toString();
	return `${API_BASE}/api/export/csv${query ? '?' + query : ''}`;
};
export const getExportJSONUrl = (params = {}) => {
	const query = new URLSearchParams(params).toString();
	return `${API_BASE}/api/export/json${query ? '?' + query : ''}`;
};
export const importEmailsFile = (formData) => api.post('/api/export/import', formData, {
	headers: { 'Content-Type': 'multipart/form-data' },
});

export const getGoogleAuthUrl = () => `${API_BASE}/api/auth/google`;
export const getSyncStreamUrl = () => `${API_BASE}/api/gmail/sync/stream`;

export default api;
