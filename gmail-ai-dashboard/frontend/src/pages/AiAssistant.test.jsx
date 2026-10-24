import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import AiAssistant from './AiAssistant';
import { approveRagEmail, getRagSourceEmail, getRagStatus, rejectRagEmail, sendRagChat, updateRagEmailDraft } from '../services/api';

vi.mock('../context/AuthContext', () => ({ useAuth: () => ({ user: { name: 'Mailbox Owner' } }) }));
vi.mock('../services/api', () => ({
    getRagStatus: vi.fn(), sendRagChat: vi.fn(), getRagSourceEmail: vi.fn(),
    approveRagEmail: vi.fn(), rejectRagEmail: vi.fn(), updateRagEmailDraft: vi.fn(),
}));

afterEach(() => vi.unstubAllGlobals());

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

it('opens mobile navigation and planner only when requested', () => {
    window.localStorage.setItem('gmail-ai-navigation-open', 'true');
    window.localStorage.setItem('gmail-ai-key-points-open', 'true');
    vi.stubGlobal('matchMedia', vi.fn(() => ({ matches: true, addEventListener: vi.fn(), removeEventListener: vi.fn() })));
    getRagStatus.mockResolvedValue({ data: {} });

    render(<MemoryRouter><AiAssistant /></MemoryRouter>);
    expect(screen.getByRole('button', { name: 'Open navigation and composer panel' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Open key points panel' })).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'Key Points' })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Open navigation and composer panel' }));
    fireEvent.click(screen.getByRole('button', { name: 'Close navigation' }));
    expect(screen.getByRole('button', { name: 'Open navigation and composer panel' })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Open key points panel' }));
    expect(screen.getByRole('heading', { name: 'Key Points' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Dismiss assistant planner' }));
    expect(screen.queryByRole('heading', { name: 'Key Points' })).not.toBeInTheDocument();
});

it('opens an editable human review window and sends the reviewed draft values', async () => {
    Element.prototype.scrollIntoView = vi.fn();
    getRagStatus.mockResolvedValue({ data: {} });
    sendRagChat.mockResolvedValue({ data: {
        answer: 'A draft is ready for your review.', session_id: 'session-draft', citations: [], steps: [],
        approval_required: true, approval_id: 'approval-one', approval_preview: {
            id: 'approval-one', recipient_email: 'adijadhav651@gmail.com', cc: 'advisor@example.com', bcc: 'archive@example.com', subject: 'Education load follow-up',
            draft_body: 'Hello,\n\nI am writing to discuss the education load.\n\nBest regards,\nRohini', status: 'PENDING',
        },
    } });

    render(<MemoryRouter><AiAssistant /></MemoryRouter>);
    fireEvent.change(screen.getByPlaceholderText(/Ask about all your emails/), {
        target: { value: 'Draft a professional mail to adijadhav651@gmail.com related to education load' },
    });
    fireEvent.click(screen.getByRole('button', { name: /^Send$/ }));

    const dialog = await screen.findByRole('dialog', { name: 'Review email before sending' });
    expect(within(dialog).getByDisplayValue('adijadhav651@gmail.com')).toBeInTheDocument();
    expect(within(dialog).getByDisplayValue('advisor@example.com')).toBeInTheDocument();
    expect(within(dialog).getByDisplayValue('archive@example.com')).toBeInTheDocument();
    expect(within(dialog).getByDisplayValue('Education load follow-up')).toBeInTheDocument();
    expect(within(dialog).getByDisplayValue(/I am writing to discuss the education load/)).toBeInTheDocument();
    expect(within(dialog).getByRole('button', { name: 'Approve & send' })).toBeInTheDocument();
    expect(within(dialog).getByRole('button', { name: 'Discard draft' })).toBeInTheDocument();

    approveRagEmail.mockResolvedValue({ data: { status: 'SENT', message: 'Email sent successfully.' } });
    fireEvent.change(within(dialog).getByRole('textbox', { name: 'Draft recipient' }), {
        target: { value: 'updated@example.com' },
    });
    fireEvent.change(within(dialog).getByRole('textbox', { name: 'Draft cc recipients' }), {
        target: { value: 'updated-cc@example.com' },
    });
    fireEvent.change(within(dialog).getByRole('textbox', { name: 'Draft bcc recipients' }), {
        target: { value: 'updated-bcc@example.com' },
    });
    fireEvent.change(within(dialog).getByRole('textbox', { name: 'Draft subject' }), {
        target: { value: 'Updated education loan question' },
    });
    fireEvent.change(within(dialog).getByRole('textbox', { name: 'Draft message body' }), {
        target: { value: 'Hello,\n\nPlease confirm the education loan payout details.\n\nRegards' },
    });
    fireEvent.click(within(dialog).getByRole('button', { name: 'Approve & send' }));

    await waitFor(() => expect(approveRagEmail).toHaveBeenCalledWith('approval-one', {
        recipient_email: 'updated@example.com',
        cc: 'updated-cc@example.com',
        bcc: 'updated-bcc@example.com',
        subject: 'Updated education loan question',
        draft_body: 'Hello,\n\nPlease confirm the education loan payout details.\n\nRegards',
    }));
});
