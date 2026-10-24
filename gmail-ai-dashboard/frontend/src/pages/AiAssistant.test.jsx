import { fireEvent, render, screen, within } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import AiAssistant from './AiAssistant';
import { getRagSourceEmail, getRagStatus, sendRagChat } from '../services/api';

vi.mock('../context/AuthContext', () => ({ useAuth: () => ({ user: { name: 'Mailbox Owner' } }) }));
vi.mock('../services/api', () => ({
    getRagStatus: vi.fn(), sendRagChat: vi.fn(), getRagSourceEmail: vi.fn(),
    approveRagEmail: vi.fn(), rejectRagEmail: vi.fn(),
}));

it('opens the complete message when View email is clicked on a citation', async () => {
    Element.prototype.scrollIntoView = vi.fn();
    getRagStatus.mockResolvedValue({ data: {} });
    sendRagChat.mockResolvedValue({ data: {
        answer: 'Your electricity invoice is due on October 25.', session_id: 'session-one',
        citations: [{ email_id: 'invoice-one', subject: 'Electricity invoice', sender: 'billing@example.com',
            date: '2026-10-19T09:00:00+00:00', snippet: 'Short invoice excerpt.', score: 0.9 }],
    } });
    getRagSourceEmail.mockResolvedValue({ data: {
        id: 'invoice-one', subject: 'Electricity invoice', from_email: 'billing@example.com',
        sender_name: 'Utility Billing', to: 'owner@example.com', received_at: '2026-10-19T09:00:00+00:00',
        body: 'Complete payment instructions and invoice details beyond the citation excerpt.',
    } });
    render(<MemoryRouter><AiAssistant /></MemoryRouter>);
    fireEvent.click(screen.getByRole('button', { name: /Summarize pending bills/ }));
    fireEvent.click(await screen.findByRole('button', { name: 'View email' }));
    const dialog = screen.getByRole('dialog');
    expect(await within(dialog).findByText('Complete payment instructions and invoice details beyond the citation excerpt.')).toBeInTheDocument();
    expect(within(dialog).getByText('Utility Billing')).toBeInTheDocument();
    expect(within(dialog).getByText('owner@example.com')).toBeInTheDocument();
    expect(getRagSourceEmail).toHaveBeenCalledWith('invoice-one');
    expect(sendRagChat).toHaveBeenCalledWith(expect.objectContaining({
        query: 'What are my pending bills and invoices due this month?',
    }), expect.any(Function));
});

it('lets the user collapse both side panels to expand the chat area', async () => {
    getRagStatus.mockResolvedValue({ data: {} });
    render(<MemoryRouter><AiAssistant /></MemoryRouter>);

    fireEvent.click(screen.getByRole('button', { name: 'Close navigation and composer panel' }));
    fireEvent.click(screen.getByRole('button', { name: 'Close key points panel' }));

    expect(screen.queryByRole('heading', { name: 'Key Points' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Open navigation and composer panel' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Open key points panel' })).toBeInTheDocument();
});

it('opens a full-size human review window with the exact draft subject and body', async () => {
    Element.prototype.scrollIntoView = vi.fn();
    getRagStatus.mockResolvedValue({ data: {} });
    sendRagChat.mockResolvedValue({ data: {
        answer: 'A draft is ready for your review.', session_id: 'session-draft', citations: [], steps: [],
        approval_required: true, approval_id: 'approval-one', approval_preview: {
            id: 'approval-one', recipient_email: 'adijadhav651@gmail.com', subject: 'Education load follow-up',
            draft_body: 'Hello,\n\nI am writing to discuss the education load.\n\nBest regards,\nRohini', status: 'PENDING',
        },
    } });

    render(<MemoryRouter><AiAssistant /></MemoryRouter>);
    fireEvent.change(screen.getByPlaceholderText(/Ask about all your emails/), {
        target: { value: 'Draft a professional mail to adijadhav651@gmail.com related to education load' },
    });
    fireEvent.click(screen.getByRole('button', { name: /^Send$/ }));

    const dialog = await screen.findByRole('dialog', { name: 'Review email before sending' });
    expect(within(dialog).getByText('adijadhav651@gmail.com')).toBeInTheDocument();
    expect(within(dialog).getByDisplayValue('Education load follow-up')).toBeInTheDocument();
    expect(within(dialog).getByDisplayValue(/I am writing to discuss the education load/)).toBeInTheDocument();
    expect(within(dialog).getByRole('button', { name: 'Approve & send' })).toBeInTheDocument();
    expect(within(dialog).getByRole('button', { name: 'Discard draft' })).toBeInTheDocument();
});
