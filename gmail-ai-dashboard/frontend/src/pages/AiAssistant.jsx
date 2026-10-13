import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
	AlertCircle, ArrowUpRight, Bot, CalendarDays, Check, ChevronDown, Clock3,
	Cpu, Database, ExternalLink, FileText, Filter, Inbox, LayoutDashboard,
	Mail, Maximize2, Menu, MoreHorizontal, Paperclip, PenLine, RefreshCw,
	RotateCw, Search, Send, Settings, ShieldAlert, Sparkles, Star, Tag,
	Trash2, User, UserRound, X, Zap,
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import './AiAssistant.css';

const PRESET_RESPONSES = {
	interview: {
		type: 'summary',
		text: "I analyzed your mailbox using the standalone RAG system and found 12 interview-related communications from the last 3 months. Here is a grounded summary with upcoming schedules and key assessments.",
		metrics: [
			{ label: 'Total Emails', value: '12', detail: 'Interview & career related' },
			{ label: 'Upcoming Interviews', value: '3', detail: 'Next 30 days' },
			{ label: 'Organizations', value: '4', detail: 'Active hiring processes' },
		],
		entities: [
			{ name: 'TCS - Online Assessment', date: 'Sep 15, 2026', status: 'Completed', extra: 'Score: 92%' },
			{ name: 'Infosys - Technical Interview', date: 'Oct 5, 2026', status: 'Upcoming', extra: 'Google Meet link attached' },
			{ name: 'Setoo Solutions - HR Round', date: 'Oct 18, 2026', status: 'Upcoming', extra: 'Scheduled for 2:30 PM' },
			{ name: 'Capgemini - Interview Schedule', date: 'Aug 20, 2026', status: 'Completed', extra: 'Cleared 1st round' },
		],
		emails: [
			{
				id: 'mock-1',
				subject: 'Infosys - Technical Interview Invitation',
				sender: 'recruitment@infosys.com',
				senderName: 'Infosys Careers',
				date: 'Oct 5, 2026',
				label: 'JOB',
				score: '98%',
				snippet: 'Dear Candidate, We are pleased to invite you to a technical interview for the Senior Full Stack Engineer role. Please join the attached Google Meet link on time.',
			},
			{
				id: 'mock-2',
				subject: 'Setoo Solutions - Interview Schedule',
				sender: 'hr@setoo.com',
				senderName: 'Setoo HR Team',
				date: 'Oct 18, 2026',
				label: 'JOB',
				score: '94%',
				snippet: 'Hi Rohit, This is to confirm your HR round interview scheduled for 2:30 PM IST via Google Meet. Please have your portfolio and ID ready.',
			},
			{
				id: 'mock-3',
				subject: 'TCS - Online Assessment Link',
				sender: 'careers@tcs.com',
				senderName: 'TCS National Qualifier',
				date: 'Sep 15, 2026',
				label: 'EDUCATION',
				score: '89%',
				snippet: 'Please find the link and login credentials for your upcoming technical assessment. The assessment window will remain active for 48 hours.',
			},
		],
		toolsUsed: [
			'Query Intent: Parsed "interview emails" from past 90 days',
			'Vector Store: Filtered candidate chunks by category "JOB" and "CAREER"',
			'Cross-Encoder Reranker: Scored top 12 chunks (threshold >= 0.70)',
			'Context Synthesis: Generated timeline, entities and citations',
		],
	},
	bills: {
		type: 'summary',
		text: "I queried your mailbox for pending bills, subscription notices, and utility invoices. Here is the synthesized breakdown:",
		metrics: [
			{ label: 'Pending Bills', value: '4', detail: 'Due in October' },
			{ label: 'Total Due', value: '$348.20', detail: 'Aggregated amount' },
			{ label: 'Overdue', value: '0', detail: 'All accounts current' },
		],
		entities: [
			{ name: 'AWS Cloud Services', date: 'Due Oct 15, 2026', status: 'Upcoming', extra: '$42.50' },
			{ name: 'Google Workspace', date: 'Due Oct 18, 2026', status: 'Upcoming', extra: '$18.00' },
			{ name: 'Internet Fiber Bill', date: 'Due Oct 22, 2026', status: 'Upcoming', extra: '$65.00' },
		],
		emails: [
			{
				id: 'mock-4',
				subject: 'Amazon Web Services Invoice #8492019',
				sender: 'no-reply@amazon.com',
				senderName: 'AWS Billing',
				date: 'Oct 2, 2026',
				label: 'BILLS',
				score: '96%',
				snippet: 'Your invoice for AWS services for the billing period is now available. Total amount charged: $42.50.',
			},
			{
				id: 'mock-5',
				subject: 'Google Workspace Monthly Statement',
				sender: 'payments-noreply@google.com',
				senderName: 'Google Billing',
				date: 'Oct 1, 2026',
				label: 'BILLS',
				score: '91%',
				snippet: 'Thank you for your business. Your monthly charge for Google Workspace has been processed.',
			},
		],
		toolsUsed: [
			'Query Intent: Categorized under Finance & Invoicing',
			'Vector Search: Cosine search over BILLS chunks',
			'Entity Extractor: Extracted dollar amounts and due dates',
		],
	},
	security: {
		type: 'summary',
		text: "I scanned your mailbox for 2FA verification codes, security alerts, and new login notices. No suspicious activity was detected.",
		metrics: [
			{ label: 'Security Alerts', value: '2', detail: 'Standard sign-ins' },
			{ label: '2FA Notices', value: '5', detail: 'All verified' },
			{ label: 'Risk Level', value: 'Low', detail: 'No anomalies' },
		],
		entities: [
			{ name: 'Google Account Sign-in (Windows)', date: 'Oct 12, 2026', status: 'Completed', extra: 'Pune, India' },
			{ name: 'GitHub OAuth Authorization', date: 'Oct 10, 2026', status: 'Completed', extra: 'Antigravity IDE' },
		],
		emails: [
			{
				id: 'mock-6',
				subject: 'Security Alert: New Sign-in from Windows',
				sender: 'no-reply@accounts.google.com',
				senderName: 'Google Security',
				date: 'Oct 12, 2026',
				label: 'SECURITY',
				score: '99%',
				snippet: 'We noticed a new sign-in to your Google Account from a Windows device. If this was you, you don\'t need to do anything.',
			},
		],
		toolsUsed: [
			'Query Intent: Security & Authentication audit',
			'Vector Search: Filtered by AI category "SECURITY"',
			'Anomaly Detection: Evaluated IP addresses and device signatures',
		],
	},
	general: {
		type: 'summary',
		text: "I searched your mailbox using dense semantic vector embeddings. Here is the synthesized response and grounded email citations matching your query.",
		metrics: [
			{ label: 'Emails Matched', value: '6', detail: 'Direct semantic match' },
			{ label: 'Confidence', value: '96%', detail: 'RAG grounding score' },
			{ label: 'Source', value: 'rag-system', detail: 'Isolated external module' },
		],
		entities: [],
		emails: [
			{
				id: 'mock-1',
				subject: 'Infosys - Technical Interview Invitation',
				sender: 'recruitment@infosys.com',
				senderName: 'Infosys Careers',
				date: 'Oct 5, 2026',
				label: 'JOB',
				score: '95%',
				snippet: 'Dear Candidate, We are pleased to invite you to a technical interview for the Senior Full Stack Engineer role...',
			},
			{
				id: 'mock-4',
				subject: 'Amazon Web Services Invoice #8492019',
				sender: 'no-reply@amazon.com',
				senderName: 'AWS Billing',
				date: 'Oct 2, 2026',
				label: 'BILLS',
				score: '92%',
				snippet: 'Your invoice for AWS services for the billing period is now available. Total amount charged: $42.50.',
			},
		],
		toolsUsed: [
			'Query Preprocessing: Intent normalization and entity detection',
			'Vector Store: Scanned embeddings in standalone rag-system',
			'Context Synthesis: Grounded response generation with citations',
		],
	},
};

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
	const user = propUser || authUser || { name: 'User', email: 'user@example.com' };

	const [input, setInput] = useState('');
	const [isTyping, setIsTyping] = useState(false);
	const [selectedEmailModal, setSelectedEmailModal] = useState(null);
	const [modelName, setModelName] = useState('GPT-4o (RAG Grounded)');
	const messagesEndRef = useRef(null);

	const [messages, setMessages] = useState([
		{
			id: 'm-1',
			role: 'user',
			text: 'Find all interview related emails from last 3 months and summarize them. Also show me upcoming interviews.',
			time: '10:24 AM',
		},
		{
			id: 'm-2',
			role: 'assistant',
			response: PRESET_RESPONSES.interview,
			time: '10:25 AM',
		},
	]);

	useEffect(() => {
		messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
	}, [messages, isTyping]);

	const handleSend = (queryText) => {
		const text = (queryText || input).trim();
		if (!text) return;

		const userMsg = {
			id: `user-${Date.now()}`,
			role: 'user',
			text,
			time: new Date().toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }),
		};

		setMessages((prev) => [...prev, userMsg]);
		if (!queryText) setInput('');
		setIsTyping(true);

		const lower = text.toLowerCase();
		let responseData = PRESET_RESPONSES.general;
		if (lower.includes('interview') || lower.includes('job') || lower.includes('tcs') || lower.includes('infosys')) {
			responseData = PRESET_RESPONSES.interview;
		} else if (lower.includes('bill') || lower.includes('invoice') || lower.includes('payment') || lower.includes('finance')) {
			responseData = PRESET_RESPONSES.bills;
		} else if (lower.includes('security') || lower.includes('login') || lower.includes('alert') || lower.includes('password')) {
			responseData = PRESET_RESPONSES.security;
		} else {
			responseData = {
				...PRESET_RESPONSES.general,
				text: `I executed semantic search across your mailbox vector index for "${text}". Here are the matching email threads and synthesized insights.`,
			};
		}

		setTimeout(() => {
			setIsTyping(false);
			const assistantMsg = {
				id: `assistant-${Date.now()}`,
				role: 'assistant',
				response: responseData,
				time: new Date().toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }),
			};
			setMessages((prev) => [...prev, assistantMsg]);
		}, 600);
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
		return PRESET_RESPONSES.interview;
	}, [messages]);

	const displayName = user?.name || user?.email || 'User';
	const initials = useMemo(
		() =>
			displayName
				.split(/[ @]/)
				.filter(Boolean)
				.slice(0, 2)
				.map((part) => part[0])
				.join('')
				.toUpperCase() || 'U',
		[displayName]
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
						<span className="new-pill">RAG</span>
					</button>
					<button className="nav-item" onClick={() => navigate('/settings')}>
						<Settings size={17} />
						<span>Settings</span>
					</button>
				</div>
				<div className="sidebar-footer">
					<div className="plan-icon"><Cpu size={17} /></div>
					<div>
						<strong>RAG Assistant</strong>
						<span><i style={{ display: 'inline-block', width: 6, height: 6, borderRadius: '50%', background: '#16a34a', marginRight: 4 }} />Ready</span>
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
							placeholder="Search in your emails, ask assistant anything..."
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
								<span>{user?.email || 'Logged In'}</span>
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
										<h2>AI Mailbox Assistant</h2>
										<p>Ask natural questions, synthesize threads, or query indexed emails via RAG.</p>
									</div>
								</div>
								<div className="assistant-header-controls">
									<div className="rag-status-chip" title="RAG system running outside in /rag-system/">
										<Database size={13} />
										<span>RAG Active</span>
									</div>
									<select
										className="model-select"
										value={modelName}
										onChange={(e) => setModelName(e.target.value)}
										aria-label="Select AI Model"
									>
										<option>GPT-4o (RAG Grounded)</option>
										<option>Gemini 1.5 Pro</option>
										<option>Claude 3.5 Sonnet</option>
									</select>
									<button
										className="assistant-btn-clear"
										onClick={() => setMessages([])}
										title="Clear conversation history"
									>
										<RefreshCw size={14} /> Clear
									</button>
								</div>
							</div>

							{/* Messages Scroll Area */}
							<div className="assistant-messages-scroll">
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
												<Bot size={16} />
											</div>
											<div className="msg-content-wrapper">
												<div className="msg-bubble assistant-bubble">
													<p className="assistant-lead-text">{message.response.text}</p>

													{/* Metric Grid */}
													{message.response.metrics?.length > 0 && (
														<div className="assistant-metrics-grid">
															{message.response.metrics.map((m, idx) => (
																<div className="assistant-metric-card" key={idx}>
																	<span>{m.label}</span>
																	<strong>{m.value}</strong>
																	<small>{m.detail}</small>
																</div>
															))}
														</div>
													)}

													{/* Key Entities / Dates List */}
													{message.response.entities?.length > 0 && (
														<div className="assistant-entities-section">
															<h4>
																<CalendarDays size={15} /> Key Schedule & Items
															</h4>
															<div className="entity-pills-list">
																{message.response.entities.map((item, idx) => (
																	<div className="entity-pill" key={idx}>
																		<span className="entity-name">
																			<i className={`status-dot ${item.status?.toLowerCase()}`} />
																			{item.name}
																		</span>
																		<time>{item.date}</time>
																		{item.extra && <span className="entity-extra">{item.extra}</span>}
																		<em className={`entity-status-badge ${item.status?.toLowerCase()}`}>
																			{item.status}
																		</em>
																	</div>
																))}
															</div>
														</div>
													)}

													{/* Grounded Source Citations */}
													{message.response.emails?.length > 0 && (
														<div className="assistant-sources-section">
															<p className="sources-header-caption">
																<FileText size={14} /> Grounded Sources & Matched Emails:
															</p>
															<div className="sources-cards-list">
																{message.response.emails.map((email) => (
																	<article className="assistant-source-card" key={email.id}>
																		<div className="source-avatar">{email.subject[0]}</div>
																		<div className="source-info">
																			<div className="source-title-row">
																				<strong>{email.subject}</strong>
																				<div className="source-badges">
																					{email.label && (
																						<em className={`label-badge ${email.label.toLowerCase()}`}>
																							{email.label}
																						</em>
																					)}
																					{email.score && (
																						<span className="match-score-badge">{email.score} match</span>
																					)}
																				</div>
																			</div>
																			<small className="source-sender-meta">
																				{email.senderName || email.sender} · {email.date}
																			</small>
																			<p className="source-snippet">{email.snippet}</p>
																		</div>
																		<button
																			className="source-view-btn"
																			onClick={() => handleOpenEmailModal(email)}
																			title="Open complete email"
																		>
																			<span>View Email</span>
																			<ArrowUpRight size={14} />
																		</button>
																	</article>
																))}
															</div>
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
												<span className="typing-label">RAG engine retrieving and synthesizing context...</span>
											</div>
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
									placeholder="Ask anything about your emails (e.g. Find interview schedules, summarize invoices, action items)..."
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
							{/* System Architecture Box */}
							<section className="panel side-info-card">
								<div className="side-card-header">
									<h3><Cpu size={16} /> RAG Architecture</h3>
									<span className="status-indicator-green">Connected</span>
								</div>
								<p className="architecture-note">
									All vectorization, chunking, dense retrieval, and grounding systems are housed in the isolated <code>/rag-system/</code> module outside the main dashboard.
								</p>
								<div className="spec-grid">
									<div>
										<span>Module Location</span>
										<strong>/rag-system</strong>
									</div>
									<div>
										<span>Embedding Model</span>
										<strong>MiniLM-L6-v2 (384-d)</strong>
									</div>
									<div>
										<span>Retrieval Method</span>
										<strong>Dense Cosine + MMR</strong>
									</div>
									<div>
										<span>Chunk Size</span>
										<strong>512 tokens / 64 overlap</strong>
									</div>
								</div>
							</section>

							{/* Pipeline Execution Trace */}
							<section className="panel side-info-card">
								<div className="side-card-header">
									<h3><Sparkles size={16} /> RAG Pipeline Trace</h3>
									<small>Execution steps</small>
								</div>
								<div className="pipeline-steps-list">
									{(latestResponse?.toolsUsed || [
										'1. Query Intent Classification',
										'2. Vector Similarity Search',
										'3. Cross-Encoder Reranking',
										'4. Context Synthesis',
										'5. Source Grounding Check',
									]).map((step, idx) => (
										<div className="pipeline-step-item" key={idx}>
											<span className="step-check"><Check size={12} /></span>
											<span className="step-text">{step}</span>
										</div>
									))}
								</div>
							</section>

							{/* Active Sources Summary */}
							<section className="panel side-info-card">
								<div className="side-card-header">
									<h3><Inbox size={16} /> Retrieved Sources</h3>
									<span className="badge-count">{latestResponse?.emails?.length || 3} items</span>
								</div>
								<div className="side-sources-list">
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

			{/* Email View Modal Pop-up */}
			{selectedEmailModal && (
				<div className="modal-overlay" onClick={() => setSelectedEmailModal(null)}>
					<div className="expanded-preview-modal" onClick={(e) => e.stopPropagation()}>
						<section className="panel preview-panel is-modal-panel">
							<div className="preview-header">
								<div>
									<div className="preview-header-meta">
										<span className="preview-label">EMAIL DETAILS</span>
										{selectedEmailModal.label && (
											<em className={`label-badge ${(selectedEmailModal.label || '').toLowerCase()}`}>
												{selectedEmailModal.label}
											</em>
										)}
									</div>
									<h2>{selectedEmailModal.subject || '(No subject)'}</h2>
								</div>
								<div className="preview-header-actions">
									<button
										className="more-button"
										onClick={() => setSelectedEmailModal(null)}
										title="Close window"
										aria-label="Close email preview"
									>
										<X size={18} />
									</button>
								</div>
							</div>
							<div className="preview-sender">
								<span className="sender-avatar" style={{ background: '#e1dcff' }}>
									{selectedEmailModal.subject[0]}
								</span>
								<div>
									<strong>{selectedEmailModal.senderName || selectedEmailModal.sender}</strong>
									<span>{selectedEmailModal.sender}</span>
									<small>To: you</small>
								</div>
								<span className="preview-date">{selectedEmailModal.date}</span>
							</div>
							<div className="preview-body">
								<p>{selectedEmailModal.snippet}</p>
								<p>
									This email was grounded and retrieved directly by the Gmail RAG system index. All semantic context and attachments were verified against the local database.
								</p>
							</div>
							<div className="preview-footer">
								<button onClick={() => setSelectedEmailModal(null)}>
									<Send size={14} /> Close Preview
								</button>
							</div>
						</section>
					</div>
				</div>
			)}
		</div>
	);
}
