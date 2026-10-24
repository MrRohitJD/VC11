	import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE_URL || (import.meta.env.DEV ? 'http://localhost:8000' : window.location.origin);

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
export const enterDemo = () => api.post('/api/auth/demo');
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

// RAG AI Assistant endpoints
export const sendRagChat = async (data, onProgress) => {
	if (!onProgress) return api.post('/api/rag/chat', data);
	const timezone = getClientTimezone();
	const response = await fetch(`${API_BASE}/api/rag/chat/stream`, {
		method: 'POST', credentials: 'include',
		headers: { 'Content-Type': 'application/json', ...(timezone ? { 'X-Timezone': timezone } : {}) },
		body: JSON.stringify(data),
	});
	if (!response.ok) {
		const payload = await response.json().catch(() => ({}));
		const error = new Error(payload.detail || payload.message || 'Mailbox search could not start.');
		error.status = response.status;
		throw error;
	}
	if (!response.body) throw new Error('Mailbox search could not start.');
	const reader = response.body.getReader();
	const decoder = new TextDecoder();
	let buffer = '';
	let result;
	try {
		while (true) {
			const { value, done } = await reader.read();
			buffer += decoder.decode(value, { stream: !done });
			buffer = buffer.replace(/\r\n/g, '\n');
			let boundary;
			while ((boundary = buffer.indexOf('\n\n')) !== -1) {
				const frame = buffer.slice(0, boundary);
				buffer = buffer.slice(boundary + 2);
				const lines = frame.split('\n');
				const kind = lines.find((line) => line.startsWith('event:'))?.slice(6).trim();
				const raw = lines.filter((line) => line.startsWith('data:')).map((line) => line.slice(5).trimStart()).join('\n');
				if (!raw) continue;
				const payload = JSON.parse(raw);
				if (kind === 'activity') onProgress(payload);
				if (kind === 'error') throw new Error(payload.message || 'Mailbox search failed.');
				if (kind === 'result') result = payload;
			}
			if (done) break;
		}
	} finally {
		reader.releaseLock();
	}
	if (!result) throw new Error('The mailbox search ended before returning an answer.');
	return { data: result };
};
export const getRagStatus = () => api.get('/api/rag/status');
export const triggerRagIndex = (data = {}) => api.post('/api/rag/index', data);
export const getRagSessions = (limit = 20) => api.get('/api/rag/sessions', { params: { limit } });
export const getRagSession = (sessionId) => api.get(`/api/rag/sessions/${sessionId}`);
export const deleteRagSession = (sessionId) => api.delete(`/api/rag/sessions/${sessionId}`);
export const getRagApprovals = (statusFilter) => api.get('/api/rag/approvals', { params: { status_filter: statusFilter } });
export const approveRagEmail = (approvalId) => api.post(`/api/rag/approvals/${approvalId}/approve`);
export const rejectRagEmail = (approvalId) => api.post(`/api/rag/approvals/${approvalId}/reject`);
export const getRagSourceEmail = (emailId) => api.get(`/api/rag/sources/${emailId}`);

export default api;
