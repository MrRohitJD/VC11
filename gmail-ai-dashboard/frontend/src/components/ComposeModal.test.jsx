import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import ComposeModal from './ComposeModal';
import { sendEmail } from '../services/api';

vi.mock('../services/api', () => ({
	sendEmail: vi.fn(),
}));

describe('ComposeModal', () => {
	beforeEach(() => {
		vi.resetAllMocks();
	});

	it('does not render when isOpen is false', () => {
		render(<ComposeModal isOpen={false} onClose={vi.fn()} />);
		expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
	});

	it('renders dialog with inputs when isOpen is true', () => {
		render(<ComposeModal isOpen={true} onClose={vi.fn()} />);
		expect(screen.getByRole('dialog')).toBeInTheDocument();
		expect(screen.getByPlaceholderText('Recipients')).toBeInTheDocument();
		expect(screen.getByPlaceholderText('Subject')).toBeInTheDocument();
		expect(screen.getByPlaceholderText('Write your email here...')).toBeInTheDocument();
		expect(screen.getByRole('button', { name: /Send/i })).toBeInTheDocument();
	});

	it('shows error if Send is clicked with empty recipient', () => {
		render(<ComposeModal isOpen={true} onClose={vi.fn()} />);
		fireEvent.click(screen.getByRole('button', { name: /Send/i }));
		expect(screen.getByRole('alert')).toHaveTextContent(/Please enter at least one recipient email address/i);
		expect(sendEmail).not.toHaveBeenCalled();
	});

	it('allows toggling Cc and Bcc inputs', () => {
		render(<ComposeModal isOpen={true} onClose={vi.fn()} />);
		expect(screen.queryByPlaceholderText('Cc recipients')).not.toBeInTheDocument();
		expect(screen.queryByPlaceholderText('Bcc recipients')).not.toBeInTheDocument();

		fireEvent.click(screen.getByRole('button', { name: 'Cc' }));
		expect(screen.getByPlaceholderText('Cc recipients')).toBeInTheDocument();

		fireEvent.click(screen.getByRole('button', { name: 'Bcc' }));
		expect(screen.getByPlaceholderText('Bcc recipients')).toBeInTheDocument();
	});

	it('successfully sends email with filled fields', async () => {
		const onClose = vi.fn();
		const onSent = vi.fn();
		sendEmail.mockResolvedValueOnce({
			data: { status: 'SENT', message: 'Email sent successfully.' },
		});

		render(<ComposeModal isOpen={true} onClose={onClose} onSent={onSent} />);

		fireEvent.change(screen.getByPlaceholderText('Recipients'), {
			target: { value: 'alice@example.com' },
		});
		fireEvent.change(screen.getByPlaceholderText('Subject'), {
			target: { value: 'Hello World' },
		});
		fireEvent.change(screen.getByPlaceholderText('Write your email here...'), {
			target: { value: 'This is a test message.' },
		});

		fireEvent.click(screen.getByRole('button', { name: /Send/i }));

		expect(sendEmail).toHaveBeenCalledWith({
			to: 'alice@example.com',
			subject: 'Hello World',
			body: 'This is a test message.',
		});

		expect(await screen.findByRole('status')).toHaveTextContent('Email sent successfully.');
		expect(onSent).toHaveBeenCalledWith({ status: 'SENT', message: 'Email sent successfully.' });

		await waitFor(() => {
			expect(onClose).toHaveBeenCalled();
		}, { timeout: 1500 });
	});

	it('displays error message when sending fails', async () => {
		sendEmail.mockRejectedValueOnce({
			response: { data: { detail: 'Gmail send permission is missing' } },
		});

		render(<ComposeModal isOpen={true} onClose={vi.fn()} />);

		fireEvent.change(screen.getByPlaceholderText('Recipients'), {
			target: { value: 'bob@example.com' },
		});
		fireEvent.click(screen.getByRole('button', { name: /Send/i }));

		expect(await screen.findByRole('alert')).toHaveTextContent('Gmail send permission is missing');
	});

	it('minimizes and restores when minimize control is clicked', () => {
		render(<ComposeModal isOpen={true} onClose={vi.fn()} />);
		expect(screen.getByRole('dialog')).toBeInTheDocument();

		fireEvent.click(screen.getByLabelText('Minimize compose window'));
		expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
		expect(screen.getByRole('button', { name: 'Restore new message window' })).toBeInTheDocument();

		fireEvent.click(screen.getByRole('button', { name: 'Restore new message window' }));
		expect(screen.getByRole('dialog')).toBeInTheDocument();
	});

	it('calls onClose when discard button is clicked', () => {
		const onClose = vi.fn();
		render(<ComposeModal isOpen={true} onClose={onClose} />);
		fireEvent.click(screen.getByLabelText('Discard email draft'));
		expect(onClose).toHaveBeenCalled();
	});
});
