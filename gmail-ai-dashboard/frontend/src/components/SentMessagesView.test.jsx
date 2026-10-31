import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import SentMessagesView from './SentMessagesView';
import { fetchEmails } from '../services/api';

vi.mock('../services/api', () => ({
	fetchEmails: vi.fn(),
}));

const mockSentEmailsPage1 = [
	{
		id: 'sent-1',
		from_email: 'me@example.com',
		to_email: 'Jane Doe <jane@example.com>',
		subject: 'Project Kickoff Notes',
		snippet: 'Here are the notes from our kickoff meeting today.',
		received_at: '2026-10-25T14:30:00Z',
		is_read: true,
		is_starred: false,
		labels: [{ id: 'l-1', name: 'SENT', is_ai: false }],
	},
	{
		id: 'sent-2',
		from_email: 'me@example.com',
		to_email: 'Alex Smith <alex@example.com>',
		subject: 'Invoice #4029',
		snippet: 'Attached is the invoice for services rendered.',
		received_at: '2026-10-24T10:00:00Z',
		is_read: true,
		is_starred: true,
		labels: [{ id: 'l-1', name: 'SENT', is_ai: false }],
	},
];

const mockSentEmailsPage2 = [
	{
		id: 'sent-3',
		from_email: 'me@example.com',
		to_email: 'Support Team <support@service.example>',
		subject: 'Support ticket #104',
		snippet: 'Thank you for following up.',
		received_at: '2026-10-20T08:15:00Z',
		is_read: true,
		is_starred: false,
		labels: [{ id: 'l-1', name: 'SENT', is_ai: false }],
	},
];

describe('SentMessagesView Component', () => {
	beforeEach(() => {
		vi.clearAllMocks();
		class MockIntersectionObserver {
			constructor(callback) {
				this.callback = callback;
			}
			observe() {}
			unobserve() {}
			disconnect() {}
		}
		window.IntersectionObserver = MockIntersectionObserver;
	});

	it('renders loading state initially and then lists sent messages', async () => {
		fetchEmails.mockResolvedValueOnce({
			data: {
				total: 2,
				page: 1,
				per_page: 10,
				emails: mockSentEmailsPage1,
			},
		});

		render(<SentMessagesView onSelectEmail={vi.fn()} />);

		expect(screen.getByText('Loading sent messages...')).toBeInTheDocument();

		await waitFor(() => {
			expect(screen.getByText('Sent Messages')).toBeInTheDocument();
		});

		expect(screen.getByText('To: Jane Doe')).toBeInTheDocument();
		expect(screen.getByText('Project Kickoff Notes')).toBeInTheDocument();
		expect(screen.getByText('Here are the notes from our kickoff meeting today.')).toBeInTheDocument();

		expect(screen.getByText('To: Alex Smith')).toBeInTheDocument();
		expect(screen.getByText('Invoice #4029')).toBeInTheDocument();

		expect(fetchEmails).toHaveBeenCalledWith(
			expect.objectContaining({
				sent_only: true,
				page: 1,
				per_page: 10,
			})
		);
	});

	it('lazy loads the next page when scrolled near bottom', async () => {
		fetchEmails
			.mockResolvedValueOnce({
				data: {
					total: 3,
					page: 1,
					per_page: 10,
					emails: mockSentEmailsPage1,
				},
			})
			.mockResolvedValueOnce({
				data: {
					total: 3,
					page: 2,
					per_page: 10,
					emails: mockSentEmailsPage2,
				},
			});

		const { container } = render(<SentMessagesView onSelectEmail={vi.fn()} />);

		await waitFor(() => {
			expect(screen.getByText('Project Kickoff Notes')).toBeInTheDocument();
		});

		const scrollContainer = container.querySelector('.inbox-scroll');
		expect(scrollContainer).toBeInTheDocument();

		// Simulate scrolling near the bottom to trigger lazy loading
		Object.defineProperty(scrollContainer, 'scrollHeight', { value: 1000, configurable: true });
		Object.defineProperty(scrollContainer, 'scrollTop', { value: 700, configurable: true });
		Object.defineProperty(scrollContainer, 'clientHeight', { value: 200, configurable: true });

		fireEvent.scroll(scrollContainer);

		await waitFor(() => {
			expect(fetchEmails).toHaveBeenCalledWith(
				expect.objectContaining({
					sent_only: true,
					page: 2,
					per_page: 10,
				})
			);
		});

		await waitFor(() => {
			expect(screen.getByText('Support ticket #104')).toBeInTheDocument();
		});
	});

	it('calls onSelectEmail when a sent message row is clicked', async () => {
		fetchEmails.mockResolvedValueOnce({
			data: {
				total: 2,
				page: 1,
				per_page: 10,
				emails: mockSentEmailsPage1,
			},
		});

		const handleSelect = vi.fn();
		render(<SentMessagesView onSelectEmail={handleSelect} />);

		await waitFor(() => {
			expect(screen.getByText('Project Kickoff Notes')).toBeInTheDocument();
		});

		fireEvent.click(screen.getByText('Project Kickoff Notes'));

		expect(handleSelect).toHaveBeenCalledWith(
			'sent-1',
			expect.objectContaining({
				id: 'sent-1',
				subject: 'Project Kickoff Notes',
			})
		);
	});

	it('renders empty state when there are no sent messages', async () => {
		fetchEmails.mockResolvedValueOnce({
			data: {
				total: 0,
				page: 1,
				per_page: 10,
				emails: [],
			},
		});

		render(<SentMessagesView onSelectEmail={vi.fn()} />);

		await waitFor(() => {
			expect(screen.getByText('No sent messages found.')).toBeInTheDocument();
		});
	});
});
