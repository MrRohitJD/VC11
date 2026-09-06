	import axios from 'axios';

const api = axios.create({ baseURL: import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000', headers: { 'Content-Type': 'application/json' } });

export const syncGmail = () => api.post('/api/gmail/sync');
export const fetchEmails = (params) => api.get('/api/emails', { params });
export const trainModel = () => api.post('/api/classification/train');
export default api;
