import { useCallback, useEffect, useRef, useState } from 'react';
import { Paperclip, Send, Star } from 'lucide-react';
import { fetchEmails } from '../services/api';

const parseRecipientInfo = (toHeader) => {
	if (!toHeader) return { name: 'Unknown Recipient', email: '', initials: 'TO' };
	const match = toHeader.match(/^(.*?)\s*<(.+?)>$/);
	if (match) {
		const name = match[1].replace(/['"]/g, '').trim();
		const email = match[2].trim();
		const initials = name
			? name.split(/[ @]/).filter(Boolean).slice(0, 2).map((part) => part[0]).join('').toUpperCase()
			: email.slice(0, 2).toUpperCase();
		return { name: name || email, email, initials: initials || 'TO' };
	}
	const email = toHeader.trim();
	const initials = email.split(/[ @]/).filter(Boolean).slice(0, 2).map((part) => part[0]).join('').toUpperCase();
	return { name: email, email, initials: initials || 'TO' };
};

const mapSentEmail = (email) => {
	const recipientInfo = parseRecipientInfo(email.to_email);
	return {
		...email,
		sender: email.from_email || 'You',
		recipient: recipientInfo.name,
		recipientEmail: recipientInfo.email,
		initials: recipientInfo.initials,
		avatarColor: '#dcfce7',
		preview: email.snippet || '',
		label: email.ai_label || email.labels?.find((l) => l.is_ai)?.name || 'Sent',
		date: email.received_at ? new Date(email.received_at).toLocaleDateString() : '',
		starred: Boolean(email.is_starred),
	};
};

export default function SentMessagesView({
	selectedId,
	onSelectEmail,
	selectedIds = [],
	onToggleSelect,
	searchQuery = '',
}) {
	const [emails, setEmails] = useState([]);
	const [page, setPage] = useState(1);
	const [total, setTotal] = useState(0);
	const [loading, setLoading] = useState(true);
	const [loadingMore, setLoadingMore] = useState(false);
	const [error, setError] = useState('');
	const [categoryFilter, setCategoryFilter] = useState('All');

	const scrollContainerRef = useRef(null);
	const sentinelRef = useRef(null);
	const isMountedRef = useRef(true);

	const loadSentEmails = useCallback(async (targetPage = 1, append = false) => {
		if (targetPage === 1) {
			setLoading(true);
			setError('');
		} else {
			setLoadingMore(true);
		}

		try {
			const { data } = await fetchEmails({
				sent_only: true,
				page: targetPage,
				per_page: 50,
				search: searchQuery ? searchQuery.trim() : undefined,
			});

			if (!isMountedRef.current) return;

			const mapped = (data.emails || []).map(mapSentEmail);
			setTotal(data.total || 0);
			setPage(targetPage);

			if (append) {
				setEmails((prev) => [...prev, ...mapped]);
			} else {
				setEmails(mapped);
			}
		} catch (err) {
			if (!isMountedRef.current) return;
			setError('Could not load sent messages. Please try again.');
		} finally {
			if (isMountedRef.current) {
				setLoading(false);
				setLoadingMore(false);
			}
		}
	}, [searchQuery]);

	useEffect(() => {
		isMountedRef.current = true;
		loadSentEmails(1, false);
		return () => {
			isMountedRef.current = false;
		};
	}, [loadSentEmails]);

	const hasMore = emails.length < total;

	const handleLoadMore = useCallback(() => {
		if (loading || loadingMore || !hasMore) return;
		loadSentEmails(page + 1, true);
	}, [hasMore, loadSentEmails, loading, loadingMore, page]);

	// IntersectionObserver for lazy loading sentinel element at bottom of scroll list
	useEffect(() => {
		if (typeof IntersectionObserver === 'undefined') return undefined;
		const sentinel = sentinelRef.current;
		const container = scrollContainerRef.current;
		if (!sentinel || !hasMore || loadingMore || loading) return undefined;

		const observer = new IntersectionObserver(
			(entries) => {
				if (entries[0]?.isIntersecting) {
					handleLoadMore();
				}
			},
			{ root: container, rootMargin: '120px' }
		);

		observer.observe(sentinel);
		return () => observer.disconnect();
	}, [handleLoadMore, hasMore, loading, loadingMore]);

	const categories = ['All', 'Primary', 'Social', 'Promotions', 'Updates', 'Forums'];
	const visibleEmails = categoryFilter === 'All'
		? emails
		: emails.filter((email) => email.gmail_category === categoryFilter);

	const allVisibleSelected = visibleEmails.length > 0 && visibleEmails.every((e) => selectedIds.includes(e.id));

	const handleSelectAllToggle = () => {
		if (allVisibleSelected) {
			visibleEmails.forEach((e) => {
				if (selectedIds.includes(e.id)) onToggleSelect?.(e.id);
			});
		} else {
			visibleEmails.forEach((e) => {
				if (!selectedIds.includes(e.id)) onToggleSelect?.(e.id);
			});
		}
	};

	return (
		<section className="panel inbox-view" aria-label="Sent messages view">
			<div className="panel-heading">
				<div>
					<h2>Sent Messages</h2>
					<p>
						{emails.length.toLocaleString()} messages loaded
						{total > 0 ? ` of ${total.toLocaleString()} sent` : ''} • Lazy loaded
					</p>
				</div>
				<button className="view-all" type="button">
					{total.toLocaleString()} sent
				</button>
			</div>

			<div className="mail-category-tabs" role="group" aria-label="Mail categories">
				{categories.map((category) => (
					<button
						key={category}
						type="button"
						className={categoryFilter === category ? 'active' : ''}
						onClick={() => setCategoryFilter(category)}
					>
						{category}
					</button>
				))}
			</div>

			<div
				className="email-table inbox-scroll"
				ref={scrollContainerRef}
				onScroll={(event) => {
					const element = event.currentTarget;
					if (element.scrollHeight - element.scrollTop - element.clientHeight < 160) {
						handleLoadMore();
					}
				}}
			>
				{loading ? (
					<div className="mailbox-loading">Loading sent messages...</div>
				) : error ? (
					<div className="mailbox-empty" style={{ padding: '2.5rem 1rem', textAlign: 'center', color: '#dc2626' }}>
						<p>{error}</p>
						<button
							type="button"
							className="sync-retry-btn"
							onClick={() => loadSentEmails(1, false)}
							style={{ marginTop: 8 }}
						>
							Retry
						</button>
					</div>
				) : visibleEmails.length === 0 ? (
					<div className="mailbox-empty" style={{ padding: '2.5rem 1rem', textAlign: 'center', color: '#6b7280' }}>
						<p style={{ margin: 0, fontSize: '0.95rem' }}>No sent messages found.</p>
					</div>
				) : (
					<>
						<div className="table-head">
							<span className="check-cell">
								<input
									type="checkbox"
									aria-label="Select all sent messages"
									checked={allVisibleSelected}
									onChange={handleSelectAllToggle}
								/>
							</span>
							<span>Recipient</span>
							<span>Subject</span>
							<span>Label</span>
							<span>Date</span>
							<span />
						</div>

						{visibleEmails.map((email) => {
							const isSelected = selectedId === email.id;
							const isChecked = selectedIds.includes(email.id);
							return (
								<button
									className={`email-row ${isSelected ? 'selected' : ''}`}
									key={email.id}
									type="button"
									onClick={() => onSelectEmail?.(email.id, email)}
								>
									<span className="check-cell" onClick={(event) => event.stopPropagation()}>
										<input
											type="checkbox"
											aria-label={`Select email to ${email.recipient}`}
											checked={isChecked}
											onChange={() => onToggleSelect?.(email.id)}
										/>
									</span>
									<span className="sender-cell">
										<span className="sender-avatar" style={{ background: email.avatarColor || '#dcfce7' }}>
											{email.initials}
										</span>
										<strong>To: {email.recipient}</strong>
										{email.hasAttachment && <Paperclip size={13} />}
									</span>
									<span className="subject-cell">
										<strong>{email.subject || '(No subject)'}</strong>
										<small>{email.preview}</small>
									</span>
									<span>
										<em className={`label-badge ${(email.label || 'sent').toLowerCase()}`}>
											{email.label}
										</em>
										{email.gmail_category && (
											<small className="gmail-category-badge">{email.gmail_category}</small>
										)}
									</span>
									<span className="date-cell">{email.date}</span>
									<span className="star" aria-label={email.starred ? 'Starred' : 'Not starred'}>
										<Star size={17} fill={email.starred ? 'currentColor' : 'none'} />
									</span>
								</button>
							);
						})}

						{loadingMore && <div className="mailbox-loading">Loading more sent messages...</div>}
						{!hasMore && emails.length > 0 && (
							<div className="mailbox-end">All sent messages loaded ({emails.length})</div>
						)}
						<div ref={sentinelRef} style={{ height: 1 }} aria-hidden="true" />
					</>
				)}
			</div>
		</section>
	);
}
