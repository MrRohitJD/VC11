import { fireEvent, render, screen, within } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import App from './App';
import { getDeveloperSettings, getExportStats, getRagStatus, updateDeveloperSettings } from './services/api';

vi.mock('./context/AuthContext', () => ({
	useAuth: () => ({
		isAuthenticated: true,
		loading: false,
		user: { name: 'Mailbox Owner', email: 'owner@example.com', is_demo: false },
	}),
}));

vi.mock('./services/api', async (importOriginal) => ({
	...await importOriginal(),
	getDeveloperSettings: vi.fn(),
	getExportStats: vi.fn(),
	getRagStatus: vi.fn(),
	updateDeveloperSettings: vi.fn(),
}));

beforeEach(() => {
	vi.clearAllMocks();
	getDeveloperSettings.mockResolvedValue({ data: { database_configured: false, llm_key_configured: false } });
	getExportStats.mockResolvedValue({ data: {} });
	getRagStatus.mockResolvedValue({ data: {} });
});

function submitDatabase() {
	render(<MemoryRouter initialEntries={['/settings']}><App /></MemoryRouter>);
	fireEvent.change(screen.getByLabelText('Database URL'), {
		target: { value: 'postgresql://owner:secret@db.example.com/mail' },
	});
	fireEvent.click(screen.getByRole('button', { name: 'Connect Database' }));
}

it('opens the pop-up as soon as Connect Database is clicked', async () => {
	let finishRequest;
	updateDeveloperSettings.mockReturnValue(new Promise((resolve) => { finishRequest = resolve; }));
	submitDatabase();
	const pending = screen.getByRole('alertdialog', { name: 'Connecting database' });
	expect(within(pending).getByText('Checking the database connection and app schema. No existing mail is transferred.')).toBeInTheDocument();
	expect(screen.getByRole('button', { name: 'Connect Database' })).toBeDisabled();
	finishRequest({ data: { database_configured: true } });
	expect(await screen.findByRole('alertdialog', { name: 'Database connected' })).toBeInTheDocument();
});

it('shows a success pop-up after the database is connected', async () => {
	updateDeveloperSettings.mockResolvedValue({ data: {
		database_configured: true,
		database_url_display: 'postgresql+psycopg://db.example.com:5432/mail',
	} });
	submitDatabase();

	const dialog = await screen.findByRole('alertdialog', { name: 'Database connected' });
	expect(within(dialog).getByText('Connected to postgresql+psycopg://db.example.com:5432/mail.')).toBeInTheDocument();
	expect(await screen.findByRole('status')).toHaveTextContent('Connected to postgresql+psycopg://db.example.com:5432/mail');
	expect(updateDeveloperSettings).toHaveBeenCalledWith({ database_url: 'postgresql://owner:secret@db.example.com/mail' });
	fireEvent.click(within(dialog).getByRole('button', { name: 'OK' }));
	expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument();
});

it('shows a failure pop-up with the server error when the connection fails', async () => {
	updateDeveloperSettings.mockRejectedValue({ response: { data: { detail: 'Could not connect to the database' } } });
	submitDatabase();

	const dialog = await screen.findByRole('alertdialog', { name: 'Database connection failed' });
	expect(within(dialog).getByText('Could not connect to the database')).toBeInTheDocument();
	expect(screen.getByLabelText('Database URL')).toHaveValue('postgresql://owner:secret@db.example.com/mail');
});

it('explains when the browser receives no response from the server', async () => {
	updateDeveloperSettings.mockRejectedValue(new Error('Network Error'));
	submitDatabase();
	const dialog = await screen.findByRole('alertdialog', { name: 'Database connection failed' });
	expect(within(dialog).getByText('The connection request did not reach the server or timed out. Check your network and try again.')).toBeInTheDocument();
});

it('disconnects the personal database and returns to the default without transferring data', async () => {
	getDeveloperSettings.mockResolvedValue({ data: {
		database_configured: true,
		database_url_display: 'postgresql://owner:••••@db.example.com:5432/mail',
	} });
	updateDeveloperSettings.mockResolvedValue({ data: { database_configured: false } });
	render(<MemoryRouter initialEntries={['/settings']}><App /></MemoryRouter>);
	fireEvent.click(await screen.findByRole('button', { name: 'Disconnect & Use App Database' }));

	const dialog = await screen.findByRole('alertdialog', { name: 'Database disconnected' });
	expect(within(dialog).getByText('You are connected to the app database. No data was transferred.')).toBeInTheDocument();
	expect(updateDeveloperSettings).toHaveBeenCalledWith({ clear_database: true });
	expect(screen.getByRole('status')).toHaveTextContent('You are connected to the app database.');
	expect(screen.queryByRole('button', { name: 'Disconnect & Use App Database' })).not.toBeInTheDocument();
});

it('reports when the personal database remains connected after a failed disconnect', async () => {
	getDeveloperSettings.mockResolvedValue({ data: {
		database_configured: true,
		database_url_display: 'postgresql://owner:••••@db.example.com:5432/mail',
	} });
	updateDeveloperSettings.mockRejectedValue({ response: { data: { detail: 'Server error' } } });
	render(<MemoryRouter initialEntries={['/settings']}><App /></MemoryRouter>);
	fireEvent.click(await screen.findByRole('button', { name: 'Disconnect & Use App Database' }));

	const dialog = await screen.findByRole('alertdialog', { name: 'Database not disconnected' });
	expect(within(dialog).getByText('Your personal database is still connected. Please try again.')).toBeInTheDocument();
	expect(screen.getByRole('button', { name: 'Disconnect & Use App Database' })).toBeInTheDocument();
});
