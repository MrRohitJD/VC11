import { useEffect, useState } from 'react';
import DOMPurify from 'dompurify';
import { X } from 'lucide-react';
import { getRagSourceEmail } from '../services/api';

export function formatEmailDate(value) {
    if (!value) return 'Date unavailable';
    // Historical SQLite timestamps without an offset are stored in UTC.
    const normalized = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/.test(value)
        && !/(?:Z|[+-]\d{2}:?\d{2})$/i.test(value) ? `${value}Z` : value;
    const date = new Date(normalized);
    if (Number.isNaN(date.getTime())) return 'Date unavailable';
    return new Intl.DateTimeFormat(undefined, {
        year: 'numeric', month: 'short', day: 'numeric',
        hour: 'numeric', minute: '2-digit', timeZoneName: 'short',
    }).format(date);
}

export default function EmailDetailsModal({ email, onClose }) {
    const [message, setMessage] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [attempt, setAttempt] = useState(0);

    useEffect(() => {
        let active = true;
        setMessage(null);
        setLoading(true);
        setError('');
        getRagSourceEmail(email.id)
            .then(({ data }) => { if (active) setMessage(data); })
            .catch(() => {
                if (active) setError('Could not load the full email. Please try again.');
            })
            .finally(() => { if (active) setLoading(false); });
        return () => { active = false; };
    }, [email.id, attempt]);

    useEffect(() => {
        const handleKeyDown = (event) => { if (event.key === 'Escape') onClose(); };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [onClose]);

    const html = message?.body_html ? DOMPurify.sanitize(message.body_html, {
        USE_PROFILES: { html: true },
        FORBID_TAGS: ['style', 'form', 'iframe', 'object', 'embed'],
    }) : '';

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="expanded-preview-modal assistant-email-details" role="dialog"
                aria-modal="true" aria-labelledby="email-detail-title" onClick={(event) => event.stopPropagation()}>
                <section className="panel preview-panel is-modal-panel">
                    <div className="preview-header">
                        <div>
                            <span className="preview-label">EMAIL DETAILS</span>
                            <h2 id="email-detail-title">{message?.subject || email.subject || '(No subject)'}</h2>
                        </div>
                        <button className="more-button" onClick={onClose} aria-label="Close email preview">
                            <X size={18} />
                        </button>
                    </div>
                    {loading ? <p role="status">Loading full email…</p> : error ? (
                        <div role="alert" className="email-detail-error">
                            <p>{error}</p>
                            <button onClick={() => setAttempt((value) => value + 1)}>Retry</button>
                        </div>
                    ) : message && (
                        <>
                            <dl className="email-envelope">
                                <dt>From</dt>
                                <dd>{message.sender_name && <strong>{message.sender_name} </strong>}
                                    {message.from_email || message.from || 'Sender unavailable'}</dd>
                                <dt>To</dt><dd>{message.to || 'Recipient unavailable'}</dd>
                                {message.cc && <><dt>Cc</dt><dd>{message.cc}</dd></>}
                                {message.bcc && <><dt>Bcc</dt><dd>{message.bcc}</dd></>}
                                <dt>Received</dt>
                                <dd><time dateTime={message.received_at || message.date || undefined}>
                                    {formatEmailDate(message.received_at || message.date)}
                                </time></dd>
                            </dl>
                            <div className="preview-body">
                                {html ? (
                                    <div className="email-html-body" dangerouslySetInnerHTML={{ __html: html }} />
                                ) : message.body ? (
                                    <div className="email-text-body">{message.body}</div>
                                ) : <p>No message body is available in the synced email.</p>}
                                {message.attachments?.length > 0 && (
                                    <div className="email-attachment-details">
                                        <h3>Attachments</h3>
                                        <ul>{message.attachments.map((attachment) => (
                                            <li key={attachment.id}>
                                                {attachment.filename}
                                                {attachment.mime_type && ` (${attachment.mime_type})`}
                                            </li>
                                        ))}</ul>
                                    </div>
                                )}
                            </div>
                        </>
                    )}
                    <div className="preview-footer"><button onClick={onClose}>Close Preview</button></div>
                </section>
            </div>
        </div>
    );
}
