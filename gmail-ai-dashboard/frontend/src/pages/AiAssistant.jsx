import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
	AlertCircle, Bot, CalendarDays, Check, CircleDot, Cpu, Database, ExternalLink,
	FileText, Inbox, LayoutDashboard, ListChecks, Mail, MoreHorizontal, PenLine,
	RefreshCw, Search, Send, Settings, Star, Tag, X, Zap,
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import EmailDetailsModal, { formatEmailDate } from '../components/EmailDetailsModal';
import SearchActivity from '../components/SearchActivity';
import {
	approveRagEmail,
	getRagStatus,
	rejectRagEmail,
	sendRagChat,
} from '../services/api';
import './AiAssistant.css';

const renderFormattedLeadText = (text, onCitation) => {
	if (!text) return null;
	if (text.includes('**') || text.match(/\[\d+\]/)) {
		const parts = text.split(/(\*\*.*?\*\*|\[\d+\])/g);
		return parts.map((part, i) => {
			if (part.startsWith('**') && part.endsWith('**')) {
				return <strong key={i}>{part.slice(2, -2)}</strong>;
			}
			const citation = part.match(/^\[(\d+)\]$/);
			if (citation && onCitation) {
				const index = Number(citation[1]) - 1;
				return <button key={i} type="button" className="answer-citation-link" onClick={() => onCitation(index)} aria-label={`Open source ${citation[1]}`}>{part}</button>;
			}
			return part;
		});
	}
	return text;
};

const renderAnswer = (text = '', onCitation) => {
	const blocks = [];
	let list = [];
	let listType = null;
	const flushList = () => {
		if (!list.length) return;
		const Tag = listType === 'ordered' ? 'ol' : 'ul';
		blocks.push(<Tag key={blocks.length}>{list.map((item, i) => <li key={i}>{renderFormattedLeadText(item, onCitation)}</li>)}</Tag>);
		list = [];
	};
	for (const raw of text.split('\n')) {
		const line = raw.trim();
		const bullet = line.match(/^(?:[-*?]\s+|\d+[.)]\s+)(.*)$/);
		if (bullet) {
			const type = /^\d/.test(line) ? 'ordered' : 'unordered';
			if (type !== listType) flushList();
			listType = type;
			list.push(bullet[1]);
			continue;
		}
		flushList();
		if (!line) continue;
		const heading = line.match(/^#{1,6}\s+(.+)$/) || line.match(/^\*\*(.+?)\*\*:?$/);
		blocks.push(heading
			? <h4 key={blocks.length}>{renderFormattedLeadText(heading[1], onCitation)}</h4>
			: <p key={blocks.length}>{renderFormattedLeadText(line, onCitation)}</p>);
	}
	flushList();
	return blocks;
};

const PALETTE = [
	{ bg: '#ffedd5', color: '#ea580c' },
	{ bg: '#f3e8ff', color: '#9333ea' },
	{ bg: '#e0f2fe', color: '#0284c7' },
	{ bg: '#dcfce7', color: '#16a34a' },
];

const SUGGESTED_PROMPTS = [
	{ label: '💼 Find interview emails', query: 'Find all interview related emails from last 3 months and summarize upcoming interviews.' },
	{ label: '🧾 Summarize pending bills', query: 'What are my pending bills and invoices due this month?' },
	{ label: '🔒 Security alerts & logins', query: 'Check for any security alerts, 2FA notifications, or password resets.' },
	{ label: '⚡ Action items in inbox', query: 'Extract urgent action items and deadlines from my emails.' },
	{ label: '📊 Summarize today\'s emails', query: 'Give me a brief summary of all incoming emails received today.' },
];

export default function AiAssistant({ user: propUser }) {
	const navigate = useNavigate();
	const { user: authUser } = useAuth();
	const user = propUser || authUser;

	const [input, setInput] = useState('');
	const [isTyping, setIsTyping] = useState(false);
	const [activitySteps, setActivitySteps] = useState([]);
	const [selectedEmailModal, setSelectedEmailModal] = useState(null);
	const [sessionId, setSessionId] = useState(null);
	const [ragStatus, setRagStatus] = useState(null);
	const [approvalFeedback, setApprovalFeedback] = useState({});
	const messagesEndRef = useRef(null);

	const [messages, setMessages] = useState([]);

	// Fetch system status on mount
	useEffect(() => {
		getRagStatus()
			.then((res) => {
				setRagStatus(res.data);
			})
			.catch(() => {
				// Offline or non-blocking
			});
	}, []);

	useEffect(() => {
		messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
	}, [messages, isTyping]);

	const handleSend = async (queryText) => {
		const text = (queryText || input).trim();
		if (!text || isTyping) return;

		const userMsg = {
			id: `user-${Date.now()}`,
			role: 'user',
			text,
			time: new Date().toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }),
		};

		setMessages((prev) => [...prev, userMsg]);
		if (!queryText) setInput('');
		setIsTyping(true);
		setActivitySteps([]);
		const requestSteps = [];
		const requestStartedAt = typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();

		try {
			// Call the LangGraph mailbox workflow.
			const res = await sendRagChat({
				query: text,
				session_id: sessionId,
				top_k: 5,
			}, (step) => {
				requestSteps.push(step);
				setActivitySteps([...requestSteps]);
			});

			const data = res.data;
			if (data.session_id) {
				setSessionId(data.session_id);
			}

			// Format citations
			const mappedEmails = (data.citations || []).map((c, i) => {
				const pal = PALETTE[i % PALETTE.length];
				const letter = c.subject ? c.subject.trim()[0].toUpperCase() : 'E';
				return {
					id: c.email_id || `cit-${i}`,
					letter,
					avatarBg: pal.bg,
					avatarColor: pal.color,
					subject: c.subject || 'Referenced Email',
					sender: c.sender || '',
					senderName: c.sender ? c.sender.split('<')[0].trim() : 'Email Source',
					date: formatEmailDate(c.date),
					label: 'EMAIL',
					score: Number.isFinite(c.score) ? `${Math.round(c.score * 100)}%` : null,
					snippet: c.snippet || '',
					key_points: c.key_points || [],
				};
			});

			const collectedKeyPoints = (data.key_points && data.key_points.length > 0)
				? data.key_points
				: mappedEmails.flatMap((e) => e.key_points || []).filter(Boolean);

			const assistantMsg = {
				id: `assistant-${Date.now()}`,
				role: 'assistant',
				response: {
					type: 'summary',
					text: data.answer,
					// The stream has the observable timings/statuses; keep it for the
					// completed trace instead of replacing it with the compact step log.
					steps: requestSteps.length ? requestSteps : (data.steps || []),
					search_plan: data.search_plan,
					trace_url: data.trace_url,
					trace_id: data.trace_id,
					tracing_enabled: data.tracing_enabled,
					elapsed_ms: Number.isFinite(Number(data.duration_ms))
						? Number(data.duration_ms)
						: Math.round((typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now()) - requestStartedAt),
					source_count: mappedEmails.length,
					approval_required: data.approval_required,
					approval_id: data.approval_id,
					entities: [],
					emails: mappedEmails,
					key_points: collectedKeyPoints,
				},
				time: new Date().toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }),
			};

			setIsTyping(false);
			setMessages((prev) => [...prev, assistantMsg]);
		} catch (error) {
			console.warn('Backend RAG call failed:', error);
			const rateLimited = error.status === 429 || error.response?.status === 429;
			const responseData = {
				text: rateLimited
					? 'You have used your 10 assistant requests for today. Your quota resets at midnight UTC.'
					: (error.message || 'I could not answer your question right now. Please try again in a moment.'),
				steps: requestSteps,
				elapsed_ms: Math.round((typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now()) - requestStartedAt),
				source_count: 0,
				status: 'error',
			};

			setIsTyping(false);
			const assistantMsg = {
				id: `assistant-${Date.now()}`,
				role: 'assistant',
				response: responseData,
				time: new Date().toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }),
			};
			setMessages((prev) => [...prev, assistantMsg]);
		}
	};

	const handleApprove = async (approvalId) => {
		try {
			await approveRagEmail(approvalId);
			setApprovalFeedback((prev) => ({
				...prev,
				[approvalId]: { status: 'APPROVED', message: 'Your email has been sent.' },
			}));
		} catch (err) {
			setApprovalFeedback((prev) => ({
				...prev,
				[approvalId]: { status: 'ERROR', message: 'Your email could not be sent. Please try again.' },
			}));
		}
	};

	const handleReject = async (approvalId) => {
		try {
			await rejectRagEmail(approvalId);
			setApprovalFeedback((prev) => ({
				...prev,
				[approvalId]: { status: 'REJECTED', message: 'Your draft was discarded.' },
			}));
		} catch (err) {
			setApprovalFeedback((prev) => ({
				...prev,
				[approvalId]: { status: 'ERROR', message: 'Your draft could not be discarded. Please try again.' },
			}));
		}
	};

	const handleOpenEmailModal = (emailItem) => {
		setSelectedEmailModal(emailItem);
	};

	const latestResponse = useMemo(() => {
		for (let i = messages.length - 1; i >= 0; i--) {
			if (messages[i].role === 'assistant' && messages[i].response) {
				return messages[i].response;
			}
		}
		return null;
	}, [messages]);

	const extractedKeyPoints = useMemo(() => {
		// 1. Direct key_points array in latestResponse
		if (latestResponse?.key_points && Array.isArray(latestResponse.key_points) && latestResponse.key_points.length > 0) {
			return latestResponse.key_points;
		}

		// 2. Gather key_points from cited emails
		if (latestResponse?.emails && Array.isArray(latestResponse.emails)) {
			const emailPoints = latestResponse.emails.flatMap((e) => e.key_points || []).filter(Boolean);
			if (emailPoints.length > 0) return emailPoints;
		}

		// 3. Extract bullets from assistant message text
		if (latestResponse?.text && typeof latestResponse.text === 'string') {
			const bulletLines = latestResponse.text
				.split('\n')
				.map((l) => l.trim())
				.filter((l) => l.startsWith('•') || l.startsWith('-') || l.startsWith('*'));
			if (bulletLines.length > 0) return bulletLines;
		}

		return [];
	}, [latestResponse]);

	const displayName = user?.name || user?.email || 'Your account';
	const initials = useMemo(() => {
		if (user?.name) {
			const parts = user.name.split(/[ @._]/).filter(Boolean);
			if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
			if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
		}
		return user?.email?.[0]?.toUpperCase() || '?';
	}, [user]);
	const activeTypingStep = [...activitySteps].reverse().find((step) =>
		['running', 'pending', 'in_progress', 'started'].includes(String(step?.status || '').toLowerCase())
	);

	return (
		<div className="app-shell assistant-shell-view">
			{/* Shared Universal Sidebar */}
			<aside className="sidebar">
				<div className="brand">
					<div className="brand-mark"><Mail size={18} /></div>
					<span>Gmail AI <strong>Dashboard</strong></span>
				</div>
				<button className="compose-button" onClick={() => navigate('/')}>
					<PenLine size={17} /> Compose
				</button>
				<div className="nav-group">
					<button className="nav-item" onClick={() => navigate('/')}>
						<LayoutDashboard size={17} />
						<span>Dashboard</span>
					</button>
				</div>
				<div className="nav-group">
					<p className="nav-label">MAILBOX</p>
					<button className="nav-item" onClick={() => navigate('/')}>
						<Inbox size={17} />
						<span>Inbox</span>
					</button>
					<button className="nav-item" onClick={() => navigate('/')}>
						<Star size={17} />
						<span>Starred</span>
					</button>
					<button className="nav-item" onClick={() => navigate('/')}>
						<Tag size={17} />
						<span>Important</span>
					</button>
				</div>
				<div className="nav-group nav-lower">
					<p className="nav-label">WORKSPACE</p>
					<button className="nav-item active" onClick={() => navigate('/assistant')}>
						<Zap size={17} />
						<span>AI Assistant</span>
						<span className="new-pill">AI</span>
					</button>
					<button className="nav-item" onClick={() => navigate('/settings')}>
						<Settings size={17} />
						<span>Settings</span>
					</button>
				</div>
				<div className="sidebar-footer">
					<div className="plan-icon"><Cpu size={17} /></div>
					<div>
						<strong>Mailbox Assistant</strong>
					</div>
					<MoreHorizontal size={16} />
				</div>
			</aside>

			{/* Main Shell */}
			<div className="main-shell">
				{/* Topbar */}
				<header className="topbar">
					<div className="search-box">
						<Search size={16} />
						<input
							placeholder="Search your Primary emails..."
							value={input}
							onChange={(e) => setInput(e.target.value)}
							onKeyDown={(e) => {
								if (e.key === 'Enter') handleSend();
							}}
						/>
						<kbd>Ctrl</kbd><kbd>K</kbd>
					</div>
					<div className="top-actions">
						<button className="icon-button" onClick={() => navigate('/')} title="Back to Dashboard">
							<LayoutDashboard size={18} />
						</button>
						<button className="icon-button" onClick={() => navigate('/settings')} title="Settings">
							<Settings size={18} />
						</button>
						<div className="user-profile">
							<div className="avatar">{initials}</div>
							<div className="user-copy">
								<strong>{displayName}</strong>
								<span>{user?.email || ''}</span>
							</div>
						</div>
					</div>
				</header>

				{/* Assistant Workspace Content */}
				<main className="content assistant-main-content">
					<div className="assistant-layout-grid">
						{/* Left: Chat & Conversation */}
						<section className="panel assistant-chat-panel">
							<div className="assistant-chat-header">
								<div className="assistant-bot-title">
									<div className="assistant-bot-avatar">
										<Bot size={22} />
									</div>
									<div>
										<h2>Mailbox Assistant · LangGraph</h2>
										<p>Answers use only your Primary category emails.</p>
									</div>
								</div>
								<div className="assistant-header-controls">
									<div className="rag-status-chip" title="Assistant connection">
										<Database size={13} />
										<span>{ragStatus ? 'Connected' : 'Connection unavailable'}</span>
									</div>

									<button
										className="assistant-btn-clear"
										onClick={() => { setMessages([]); setSessionId(null); setActivitySteps([]); }}
										title="Clear conversation history"
									>
										<RefreshCw size={14} /> Clear
									</button>
								</div>
							</div>

							{/* Messages Scroll Area */}
							<div className="assistant-messages-scroll">
								{messages.length === 0 && <p className="key-points-empty">How can I help? Find an email, catch up on your inbox, check deadlines, or draft a reply. Choose a suggestion below or type your question.</p>}
								{messages.map((message) =>
									message.role === 'user' ? (
										<div className="assistant-msg-row user-row" key={message.id}>
											<div className="user-avatar-small">{initials}</div>
											<div className="msg-content-wrapper">
												<div className="msg-bubble user-bubble">{message.text}</div>
												<span className="msg-timestamp">{message.time}</span>
											</div>
										</div>
									) : (
										<div className="assistant-msg-row assistant-row" key={message.id}>
											<div className="bot-avatar-small">
												<Bot size={20} />
											</div>
											<div className="msg-content-wrapper">
												<div className="msg-bubble assistant-bubble">
									<SearchActivity
										steps={message.response.steps}
										plan={message.response.search_plan}
										traceUrl={message.response.trace_url}
										traceId={message.response.trace_id}
										tracingEnabled={message.response.tracing_enabled}
										elapsedMs={message.response.elapsed_ms}
										sourceCount={message.response.source_count}
										failed={message.response.status === 'error'}
									/>
													<div className="assistant-lead-text">
										{renderAnswer(message.response.text, (index) => {
											const source = message.response.emails?.[index];
											if (source) handleOpenEmailModal(source);
										})}
													</div>

													{/* Key Interviews Found List */}
													{message.response.entities?.length > 0 && (
														<div className="assistant-entities-section">
															<h4 className="entities-section-title">
																{message.response.entitiesTitle || 'Important details'}
															</h4>
															<div className="entities-list">
																{message.response.entities.map((item, idx) => {
																	const isCompleted = item.status?.toLowerCase() === 'completed';
																	return (
																		<div className="entity-row" key={idx}>
																			<div className="entity-left">
																				<span className="entity-bullet">
																					<CircleDot size={14} color="#2563eb" />
																				</span>
																				<span className="entity-title">{item.name}</span>
																			</div>
																			<time className="entity-date">{item.date}</time>
																			<span
																				className={`status-badge ${isCompleted ? 'completed' : 'upcoming'}`}
																			>
																				{item.status}
																			</span>
																		</div>
																	);
																})}
															</div>
														</div>
													)}

													{/* Detailed Emails with Links */}
													{message.response.emails?.length > 0 && (
														<div className="assistant-sources-section">
															<p className="sources-section-title">
																{message.response.sourcesTitle || 'Related emails'}
															</p>
															<div className="detailed-emails-list">
																{message.response.emails.map((email) => {
																	const letter = email.letter || (email.subject ? email.subject.trim()[0].toUpperCase() : 'E');
																	const avatarBg = email.avatarBg || '#e0f2fe';
																	const avatarColor = email.avatarColor || '#0284c7';
																	return (
																		<article className="detailed-email-card" key={email.id}>
																			<div className="email-card-left">
																				<div
																					className="email-avatar-circle"
																					style={{ backgroundColor: avatarBg, color: avatarColor }}
																				>
																					{letter}
																				</div>
																				<div className="email-card-info">
																					<h5 className="email-card-subject">{email.subject}</h5>
																					<p className="email-card-meta">
																						{email.sender} • {email.date}
																					</p>
																					<p className="email-card-snippet">{email.snippet}</p>
																				</div>
																			</div>
																			<button
																				className="view-email-btn"
																				onClick={() => handleOpenEmailModal(email)}
																				title="View full email"
																			>
																				<span>View email</span>
																				<ExternalLink size={14} />
																			</button>
																		</article>
																	);
																})}
															</div>
														</div>
													)}

													{/* Human-in-the-Loop Email Draft Approval Card */}
													{message.response.approval_required && message.response.approval_id && (
														<div className="assistant-approval-card" style={{ marginTop: 12, padding: 12, background: '#fef3c7', borderRadius: 8, border: '1px solid #f59e0b' }}>
															<div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#92400e', fontWeight: 600 }}>
																<AlertCircle size={18} />
																<span>Review before sending</span>
															</div>
															<p style={{ margin: '6px 0', fontSize: 13, color: '#78350f' }}>
																Review the draft above. Send it when you are ready, or discard it. Nothing is sent until you confirm.
															</p>
															{approvalFeedback[message.response.approval_id] ? (
																<div style={{ fontSize: 13, fontWeight: 600, color: approvalFeedback[message.response.approval_id].status === 'APPROVED' ? '#16a34a' : '#dc2626' }}>
																	{approvalFeedback[message.response.approval_id].message}
																</div>
															) : (
																<div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
																	<button
																		style={{ padding: '6px 14px', background: '#16a34a', color: '#fff', border: 'none', borderRadius: 6, fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 6 }}
																		onClick={() => handleApprove(message.response.approval_id)}
																	>
																		<Check size={14} /> Send email
																	</button>
																	<button
																		style={{ padding: '6px 14px', background: '#dc2626', color: '#fff', border: 'none', borderRadius: 6, fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 6 }}
																		onClick={() => handleReject(message.response.approval_id)}
																	>
																		<X size={14} /> Discard draft
																	</button>
																</div>
															)}
														</div>
													)}
												</div>
												<span className="msg-timestamp">{message.time}</span>
											</div>
										</div>
									)
								)}

								{isTyping && (
									<div className="assistant-msg-row assistant-row">
										<div className="bot-avatar-small"><Bot size={16} /></div>
										<div className="msg-content-wrapper">
											<div className="msg-bubble assistant-bubble typing-bubble">
												<span className="typing-dot" />
												<span className="typing-dot" />
												<span className="typing-dot" />
												<span className="typing-label" role="status">{activeTypingStep?.observation || 'Checking your emails...'}</span>
											</div>
											<SearchActivity
												steps={activitySteps}
												pending
												plan={activitySteps.find((step) => step.plan)?.plan}
											/>
										</div>
									</div>
								)}
								<div ref={messagesEndRef} />
							</div>

							{/* Suggested Prompt Pills */}
							<div className="assistant-prompt-pills">
								{SUGGESTED_PROMPTS.map((p, idx) => (
									<button
										className="prompt-pill-btn"
										key={idx}
										onClick={() => handleSend(p.query)}
									>
										{p.label}
									</button>
								))}
							</div>

							{/* Composer Box */}
							<div className="assistant-composer-wrap">
								<textarea
									rows={2}
									placeholder="Ask about your Primary emails (e.g. interview schedules or action items)..."
									value={input}
									onChange={(e) => setInput(e.target.value)}
									onKeyDown={(e) => {
										if (e.key === 'Enter' && !e.shiftKey) {
											e.preventDefault();
											handleSend();
										}
									}}
								/>
								<div className="composer-bottom-bar">
									<div className="composer-quick-tools">
										<button onClick={() => setInput('Search emails regarding ')}>
											<Search size={13} /> Search
										</button>
										<button onClick={() => setInput('Summarize unread emails from ')}>
											<FileText size={13} /> Summarize
										</button>
										<button onClick={() => setInput('Draft a professional reply to ')}>
											<PenLine size={13} /> Draft Reply
										</button>
										<button onClick={() => setInput('List upcoming meetings this week')}>
											<CalendarDays size={13} /> Meetings
										</button>
									</div>
									<button
										className="composer-send-btn"
										onClick={() => handleSend()}
										disabled={!input.trim()}
									>
										<span>Send</span>
										<Send size={15} />
									</button>
								</div>
							</div>
						</section>

						{/* Right: RAG Pipeline Info & Grounding Panel */}
						<aside className="assistant-side-panel">
							{/* Key Points Based on User Query */}
							<section className="panel side-info-card key-points-card">
								<div className="side-card-header">
									<h3><ListChecks size={16} /> Key Points</h3>
									<span className="status-indicator-green">Latest answer</span>
								</div>
								<p className="architecture-note">
									Important details from your latest answer:
								</p>
								<div className="key-points-list">
									{extractedKeyPoints.length > 0 ? (
										extractedKeyPoints.map((point, idx) => (
											<div className="key-point-bullet-item" key={idx}>
												<span className="key-point-bullet-symbol">•</span>
												<span className="key-point-bullet-text">{point.replace(/^[•\-\*]\s*/, '')}</span>
											</div>
										))
									) : (
										<p className="key-points-empty">No key points yet. Ask about your emails to get started.</p>
									)}
								</div>
							</section>

							{/* Active Sources Summary */}
							<section className="panel side-info-card">
								<div className="side-card-header">
									<h3><Inbox size={16} /> Related emails</h3>
									<span className="badge-count">{latestResponse?.emails?.length ?? 0} items</span>
								</div>
								<div className="side-sources-list">
									{!latestResponse?.emails?.length && <p className="key-points-empty">Related emails will appear here when available.</p>}
									{(latestResponse?.emails || []).map((email, idx) => (
										<div
											className="side-source-item"
											key={idx}
											onClick={() => handleOpenEmailModal(email)}
										>
											<div className="side-source-avatar">{email.subject[0]}</div>
											<div className="side-source-meta">
												<strong>{email.subject}</strong>
												<small>{email.senderName || email.sender} · {email.date}</small>
											</div>
										</div>
									))}
								</div>
							</section>
						</aside>
					</div>
				</main>
			</div>

			{selectedEmailModal && (
                <EmailDetailsModal
                    key={selectedEmailModal.id}
                    email={selectedEmailModal}
                    onClose={() => setSelectedEmailModal(null)}
                />
            )}
		</div>
	);
}
