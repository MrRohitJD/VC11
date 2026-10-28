import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import AiAssistant from './AiAssistant';
import { approveRagEmail, getRagSession, getRagSessions, getRagSourceEmail, getRagStatus, rejectRagEmail, sendRagChat, updateRagEmailDraft } from '../services/api';

vi.mock('../context/AuthContext', () => ({ useAuth: () => ({ user: { name: 'Mailbox Owner' } }) }));
vi.mock('../services/api', () => ({
    getRagStatus: vi.fn(), getRagSessions: vi.fn(), getRagSession: vi.fn(), sendRagChat: vi.fn(), getRagSourceEmail: vi.fn(),
    approveRagEmail: vi.fn(), rejectRagEmail: vi.fn(), updateRagEmailDraft: vi.fn(),
}));

beforeEach(() => {
    window.localStorage.clear();
    getRagSessions.mockResolvedValue({ data: [] });
});
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
    fireEvent.click(await screen.findByText('Related emails (1)'));
    fireEvent.click(screen.getByRole('button', { name: 'View email' }));
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

    fireEvent.click(screen.getByRole('button', { name: 'Open key points panel' }));
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


it('expands chat into focus mode and restores the workspace', () => {
    getRagStatus.mockResolvedValue({ data: {} });
    render(<MemoryRouter><AiAssistant /></MemoryRouter>);
    fireEvent.click(screen.getByRole('button', { name: 'Focus chat' }));
    expect(screen.getByRole('button', { name: 'Exit focus' })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('log', { name: 'Conversation' }).closest('.assistant-shell-view')).toHaveClass('chat-focused');
    fireEvent.click(screen.getByRole('button', { name: 'Exit focus' }));
    expect(screen.getByRole('button', { name: 'Focus chat' })).toHaveAttribute('aria-pressed', 'false');
});

it('keeps multiple exchanges and opens details for only the selected response', async () => {
    Element.prototype.scrollIntoView = vi.fn();
    getRagStatus.mockResolvedValue({ data: {} });
    sendRagChat.mockResolvedValueOnce({ data: { answer: 'First answer', session_id: 'thread', citations: [], steps: [] } })
        .mockResolvedValueOnce({ data: { answer: 'Second answer', session_id: 'thread', citations: [], steps: [] } });
    render(<MemoryRouter><AiAssistant /></MemoryRouter>);
    const composer = screen.getByPlaceholderText(/Ask about all your emails/);
    fireEvent.change(composer, { target: { value: 'First question' } });
    fireEvent.click(screen.getByRole('button', { name: 'Send' }));
    expect(await screen.findByText('First answer')).toBeInTheDocument();
    fireEvent.change(composer, { target: { value: 'Second question' } });
    fireEvent.click(screen.getByRole('button', { name: 'Send' }));
    expect(await screen.findByText('Second answer')).toBeInTheDocument();
    expect(screen.getByText('First question')).toBeInTheDocument();
    expect(screen.getByText('First answer')).toBeInTheDocument();
    const toggles = screen.getAllByRole('button', { name: 'Details' });
    fireEvent.click(toggles[0]);
    expect(toggles[0]).toHaveAttribute('aria-expanded', 'true');
    expect(toggles[1]).toHaveAttribute('aria-expanded', 'false');
    fireEvent.click(toggles[1]);
    expect(toggles[0]).toHaveAttribute('aria-expanded', 'false');
    expect(toggles[1]).toHaveAttribute('aria-expanded', 'true');
    const log = screen.getByRole('log', { name: 'Conversation' });
    Object.defineProperties(log, { scrollHeight: { value: 1200 }, clientHeight: { value: 400 }, scrollTop: { value: 0 } });
    fireEvent.scroll(log);
    expect(screen.getByRole('button', { name: 'Latest messages' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Latest messages' }));
    expect(screen.queryByRole('button', { name: 'Latest messages' })).not.toBeInTheDocument();
});

it('lists saved chats in the planner and reopens a conversation with its citations', async () => {
    Element.prototype.scrollIntoView = vi.fn();
    getRagStatus.mockResolvedValue({ data: {} });
    getRagSessions.mockResolvedValue({ data: [
        { id: 'recent-one', title: 'Invoice follow-up', updated_at: '2026-10-28T09:00:00' },
        { id: 'recent-two', title: 'Interview dates', updated_at: '2026-10-27T09:00:00' },
    ] });
    getRagSession.mockResolvedValue({ data: { id: 'recent-one', messages: [
        { id: 'user-old', role: 'user', content: 'What is due?', created_at: '2026-10-28T08:59:00' },
        { id: 'assistant-old', role: 'assistant', content: 'The invoice is due soon.', created_at: '2026-10-28T09:00:00',
            citations: [{ email_id: 'invoice-old', subject: 'Invoice', sender: 'billing@example.com', date: '2026-10-27T09:00:00', snippet: 'Amount due' }] },
    ] } });
    sendRagChat.mockResolvedValue({ data: { answer: 'Follow-up answer', session_id: 'recent-one', citations: [] } });

    render(<MemoryRouter><AiAssistant /></MemoryRouter>);
    fireEvent.click(screen.getByRole('button', { name: 'Open key points panel' }));
    expect(await screen.findByRole('heading', { name: 'Recent chats' })).toBeInTheDocument();
    fireEvent.click(await screen.findByRole('button', { name: /Invoice follow-up/ }));
    expect(await screen.findByText('The invoice is due soon.')).toBeInTheDocument();
    expect(screen.getByText('What is due?')).toBeInTheDocument();
    expect(screen.getByText('Related emails (1)')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Invoice follow-up/ })).toHaveAttribute('aria-current', 'true');
    fireEvent.change(screen.getByPlaceholderText(/Ask about all your emails/), { target: { value: 'And after that?' } });
    fireEvent.click(screen.getByRole('button', { name: /^Send$/ }));
    await waitFor(() => expect(sendRagChat).toHaveBeenCalledWith(expect.objectContaining({ session_id: 'recent-one' }), expect.any(Function)));
});
