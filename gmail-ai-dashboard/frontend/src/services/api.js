	import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const api = axios.create({ baseURL: API_BASE, withCredentials: true, headers: { 'Content-Type': 'application/json' } });

export const getCurrentUser = () => api.get('/api/auth/me');
export const logout = () => api.post('/api/auth/logout');
export const eraseAllData = () => api.delete('/api/auth/erase');

export const startSync = () => api.post('/api/gmail/sync');
export const getSyncStatus = () => api.get('/api/gmail/sync/status');

export const fetchEmails = (params) => api.get('/api/emails', { params });
export const fetchEmail = (id) => api.get(`/api/emails/${id}`);
export const fetchLabels = () => api.get('/api/labels');

export const trainModel = () => api.post('/api/classification/train');

export const getGoogleAuthUrl = () => `${API_BASE}/api/auth/google`;
export const getSyncStreamUrl = () => `${API_BASE}/api/gmail/sync/stream`;

export default api;
