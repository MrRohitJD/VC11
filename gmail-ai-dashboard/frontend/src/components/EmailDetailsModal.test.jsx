import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import EmailDetailsModal, { formatEmailDate } from './EmailDetailsModal';
import { getRagSourceEmail } from '../services/api';

vi.mock('../services/api', () => ({ getRagSourceEmail: vi.fn() }));

const citation = { id: 'email-one', subject: 'Invoice', snippet: 'Short citation only.' };
const fullMessage = {
    id: 'email-one', subject: 'Full invoice', sender_name: 'Billing Team', from_email: 'billing@example.com',
    to: 'owner@example.com', cc: 'accounts@example.com', received_at: '2026-10-19T09:48:00+00:00',
    body: 'First line.\n\nFull body beyond the short citation.', attachments: [],
};

describe('Full email details', () => {
    beforeEach(() => vi.resetAllMocks());

    it('fetches the full body, sender, recipients, and time instead of showing the citation', async () => {
        let resolve;
        getRagSourceEmail.mockReturnValue(new Promise((done) => { resolve = done; }));
        render(<EmailDetailsModal email={citation} onClose={vi.fn()} />);
        expect(screen.getByRole('status')).toHaveTextContent('Loading full email');
        expect(screen.queryByText(citation.snippet)).not.toBeInTheDocument();
        resolve({ data: fullMessage });
        expect(await screen.findByText('Full invoice')).toBeInTheDocument();
        expect(screen.getByText('Billing Team')).toBeInTheDocument();
        expect(screen.getByText('billing@example.com')).toBeInTheDocument();
        expect(screen.getByText('owner@example.com')).toBeInTheDocument();
        expect(screen.getByText('accounts@example.com')).toBeInTheDocument();
        expect(screen.getByText(/Full body beyond the short citation/)).toBeInTheDocument();
        expect(document.querySelector('time')).toHaveAttribute('dateTime', fullMessage.received_at);
        expect(document.querySelector('time')).toHaveTextContent(formatEmailDate(fullMessage.received_at));
        expect(getRagSourceEmail).toHaveBeenCalledWith('email-one');
    });

    it('renders formatted HTML and removes executable content', async () => {
        getRagSourceEmail.mockResolvedValue({ data: { ...fullMessage,
            body_html: '<p>Formatted <strong>invoice</strong></p><script>evil()</script><img src="x" onerror="evil()">',
        } });
        const { container } = render(<EmailDetailsModal email={citation} onClose={vi.fn()} />);
        expect(await screen.findByText('invoice', { selector: 'strong' })).toBeInTheDocument();
        expect(container.querySelector('script')).toBeNull();
        expect(container.querySelector('[onerror]')).toBeNull();
    });

    it('shows fetch errors and retries without substituting the citation', async () => {
        getRagSourceEmail.mockRejectedValueOnce(new Error('Offline')).mockResolvedValueOnce({ data: fullMessage });
        render(<EmailDetailsModal email={citation} onClose={vi.fn()} />);
        expect(await screen.findByRole('alert')).toHaveTextContent('Could not load the full email');
        expect(screen.queryByText(citation.snippet)).not.toBeInTheDocument();
        fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
        expect(await screen.findByText('Full invoice')).toBeInTheDocument();
        expect(getRagSourceEmail).toHaveBeenCalledTimes(2);
    });

    it('discards a stale response after switching to another email', async () => {
        let resolveFirst;
        getRagSourceEmail.mockReturnValueOnce(new Promise((done) => { resolveFirst = done; }))
            .mockResolvedValueOnce({ data: { ...fullMessage, id: 'email-two', subject: 'Second email' } });
        const { rerender } = render(<EmailDetailsModal email={citation} onClose={vi.fn()} />);
        rerender(<EmailDetailsModal email={{ id: 'email-two', subject: 'Second email' }} onClose={vi.fn()} />);
        await waitFor(() => expect(screen.queryByRole('status')).not.toBeInTheDocument());
        resolveFirst({ data: fullMessage });
        await waitFor(() => expect(screen.getByRole('heading', { name: 'Second email' })).toBeInTheDocument());
        expect(screen.queryByText('Full invoice')).not.toBeInTheDocument();
    });
});
