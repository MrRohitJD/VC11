import { useEffect, useRef, useState } from 'react';
import { AlertCircle, Check, Maximize2, Minimize2, Minus, Send, Trash2, X, RefreshCw } from 'lucide-react';
import { sendEmail } from '../services/api';

export default function ComposeModal({ isOpen, onClose, onSent, initialData = null }) {
	const [to, setTo] = useState('');
	const [cc, setCc] = useState('');
	const [bcc, setBcc] = useState('');
	const [subject, setSubject] = useState('');
	const [body, setBody] = useState('');
	const [showCc, setShowCc] = useState(false);
	const [showBcc, setShowBcc] = useState(false);
	const [isMinimized, setIsMinimized] = useState(false);
	const [isMaximized, setIsMaximized] = useState(false);
	const [sending, setSending] = useState(false);
	const [error, setError] = useState('');
	const [successMessage, setSuccessMessage] = useState('');

	const toInputRef = useRef(null);

	// Synchronize when initialData changes or modal opens
	useEffect(() => {
		if (isOpen) {
			setTo(initialData?.to || '');
			setCc(initialData?.cc || '');
			setBcc(initialData?.bcc || '');
			setSubject(initialData?.subject || '');
			setBody(initialData?.body || '');
			setShowCc(Boolean(initialData?.cc));
			setShowBcc(Boolean(initialData?.bcc));
			setIsMinimized(false);
			setError('');
			setSuccessMessage('');

			// Focus the recipient input on open
			setTimeout(() => {
				toInputRef.current?.focus();
			}, 100);
		}
	}, [isOpen, initialData]);

	// Handle Escape and Ctrl+Enter keyboard shortcuts
	useEffect(() => {
		if (!isOpen || isMinimized) return undefined;

		const handleKeyDown = (event) => {
			if (event.key === 'Escape') {
				onClose();
			} else if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
				event.preventDefault();
				handleSend();
			}
		};

		window.addEventListener('keydown', handleKeyDown);
		return () => window.removeEventListener('keydown', handleKeyDown);
	}, [isOpen, isMinimized, to, cc, bcc, subject, body, sending]);

	if (!isOpen) return null;

	const handleSend = async () => {
		const cleanTo = to.trim();
		if (!cleanTo) {
			setError('Please enter at least one recipient email address.');
			toInputRef.current?.focus();
			return;
		}

		setSending(true);
		setError('');
		setSuccessMessage('');

		try {
			const res = await sendEmail({
				to: cleanTo,
				cc: cc.trim() || undefined,
				bcc: bcc.trim() || undefined,
				subject: subject.trim(),
				body: body.trim(),
			});

			setSuccessMessage(res.data?.message || 'Email sent successfully.');
			if (onSent) {
				onSent(res.data);
			}

			// Brief confirmation before closing
			setTimeout(() => {
				setSending(false);
				onClose();
			}, 700);
		} catch (err) {
			setSending(false);
			const detail =
				err.response?.data?.detail ||
				err.response?.data?.message ||
				err.message ||
				'Failed to send email. Please check the recipient address and try again.';
			setError(typeof detail === 'string' ? detail : JSON.stringify(detail));
		}
	};

	const handleDiscard = () => {
		setTo('');
		setCc('');
		setBcc('');
		setSubject('');
		setBody('');
		setError('');
		onClose();
	};

	// Minimized docked bar view
	if (isMinimized) {
		return (
			<div
				className="compose-docked-minimized"
				role="button"
				tabIndex={0}
				onClick={() => setIsMinimized(false)}
				onKeyDown={(e) => {
					if (e.key === 'Enter' || e.key === ' ') setIsMinimized(false);
				}}
				title="Restore message window"
				aria-label="Restore new message window"
			>
				<span className="docked-title">{subject || 'New Message'}</span>
				<div className="docked-actions" onClick={(e) => e.stopPropagation()}>
					<button
						type="button"
						className="docked-action-btn"
						onClick={() => setIsMinimized(false)}
						aria-label="Restore"
					>
						<Maximize2 size={13} />
					</button>
					<button
						type="button"
						className="docked-action-btn"
						onClick={onClose}
						aria-label="Close"
					>
						<X size={14} />
					</button>
				</div>
			</div>
		);
	}

	return (
		<div
			className={`compose-window-container ${isMaximized ? 'is-maximized' : ''}`}
			role="dialog"
			aria-modal="true"
			aria-labelledby="compose-window-title"
		>
			<header className="compose-header">
				<span id="compose-window-title" className="compose-title">
					{subject ? subject : 'New Message'}
				</span>
				<div className="compose-header-controls">
					<button
						type="button"
						className="compose-control-btn"
						onClick={() => setIsMinimized(true)}
						title="Minimize"
						aria-label="Minimize compose window"
					>
						<Minus size={15} />
					</button>
					<button
						type="button"
						className="compose-control-btn"
						onClick={() => setIsMaximized(!isMaximized)}
						title={isMaximized ? 'Exit full screen' : 'Full screen'}
						aria-label={isMaximized ? 'Exit full screen' : 'Full screen'}
					>
						{isMaximized ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
					</button>
					<button
						type="button"
						className="compose-control-btn close-btn"
						onClick={onClose}
						title="Close (saves draft)"
						aria-label="Close compose window"
					>
						<X size={16} />
					</button>
				</div>
			</header>

			<div className="compose-content">
				{/* Recipients row */}
				<div className="compose-row">
					<span className="compose-row-label">To</span>
					<input
						ref={toInputRef}
						type="email"
						className="compose-input"
						placeholder="Recipients"
						value={to}
						onChange={(e) => {
							setTo(e.target.value);
							if (error) setError('');
						}}
						disabled={sending}
						aria-label="Recipient email"
					/>
					<div className="compose-row-tools">
						{!showCc && (
							<button
								type="button"
								className="compose-tool-btn"
								onClick={() => setShowCc(true)}
								tabIndex={-1}
							>
								Cc
							</button>
						)}
						{!showBcc && (
							<button
								type="button"
								className="compose-tool-btn"
								onClick={() => setShowBcc(true)}
								tabIndex={-1}
							>
								Bcc
							</button>
						)}
					</div>
				</div>

				{/* Cc row */}
				{showCc && (
					<div className="compose-row">
						<span className="compose-row-label">Cc</span>
						<input
							type="text"
							className="compose-input"
							placeholder="Cc recipients"
							value={cc}
							onChange={(e) => setCc(e.target.value)}
							disabled={sending}
							aria-label="Cc recipient email"
						/>
					</div>
				)}

				{/* Bcc row */}
				{showBcc && (
					<div className="compose-row">
						<span className="compose-row-label">Bcc</span>
						<input
							type="text"
							className="compose-input"
							placeholder="Bcc recipients"
							value={bcc}
							onChange={(e) => setBcc(e.target.value)}
							disabled={sending}
							aria-label="Bcc recipient email"
						/>
					</div>
				)}

				{/* Subject row */}
				<div className="compose-row subject-row">
					<input
						type="text"
						className="compose-input compose-subject-input"
						placeholder="Subject"
						value={subject}
						onChange={(e) => setSubject(e.target.value)}
						disabled={sending}
						aria-label="Email subject"
					/>
				</div>

				{/* Message Body */}
				<div className="compose-body-wrap">
					<textarea
						className="compose-textarea"
						placeholder="Write your email here..."
						value={body}
						onChange={(e) => setBody(e.target.value)}
						disabled={sending}
						aria-label="Email body text"
					/>
				</div>

				{/* Feedback status */}
				{error && (
					<div className="compose-alert compose-error" role="alert">
						<AlertCircle size={15} />
						<span>{error}</span>
					</div>
				)}
				{successMessage && (
					<div className="compose-alert compose-success" role="status">
						<Check size={15} />
						<span>{successMessage}</span>
					</div>
				)}
			</div>

			<footer className="compose-footer">
				<div className="compose-footer-left">
					<button
						type="button"
						className="compose-send-btn"
						onClick={handleSend}
						disabled={sending || Boolean(successMessage)}
					>
						{sending ? (
							<>
								<RefreshCw size={15} className="spin" />
								<span>Sending…</span>
							</>
						) : successMessage ? (
							<>
								<Check size={15} />
								<span>Sent!</span>
							</>
						) : (
							<>
								<Send size={15} />
								<span>Send</span>
							</>
						)}
					</button>
					<span className="compose-hint">Press Ctrl+Enter to send</span>
				</div>

				<div className="compose-footer-right">
					<button
						type="button"
						className="compose-discard-btn"
						onClick={handleDiscard}
						title="Discard draft"
						aria-label="Discard email draft"
						disabled={sending}
					>
						<Trash2 size={16} />
					</button>
				</div>
			</footer>
		</div>
	);
}
