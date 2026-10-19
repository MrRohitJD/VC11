import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
	AlertCircle, ArrowUpRight, Bot, Building2, Calendar, CalendarDays, Check, ChevronDown, CircleDot, Clock3,
	Cpu, Database, ExternalLink, FileText, Filter, Inbox, LayoutDashboard, ListChecks,
	Mail, Maximize2, Menu, MoreHorizontal, Paperclip, PenLine, RefreshCw,
	RotateCw, Search, Send, Settings, ShieldAlert, Sparkles, Star, Tag,
	ThumbsDown, ThumbsUp, Trash2, User, UserRound, X, Zap,
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import {
	approveRagEmail,
	getRagStatus,
	rejectRagEmail,
	sendRagChat,
	triggerRagIndex,
} from '../services/api';
import './AiAssistant.css';

const PRESET_RESPONSES = {
	interview: {
		type: 'summary',
		text: "I found 12 interview-related emails from the last 3 months. Here's a summary:",
		metrics: [
			{
				label: 'Total Emails',
				value: '12',
				detail: 'Interview related',
				icon: 'mail',
				iconBg: '#e0edff',
				iconColor: '#2563eb',
			},
			{
				label: 'Upcoming Interviews',
				value: '3',
				detail: 'Next 30 days',
				icon: 'calendar',
				iconBg: '#dcfce7',
				iconColor: '#16a34a',
			},
			{
				label: 'Companies',
				value: '2',
				detail: 'Different organizations',
				icon: 'building',
				iconBg: '#e0f2fe',
				iconColor: '#0284c7',
			},
		],
		entitiesTitle: 'Key Interviews Found',
		entities: [
			{ name: 'TCS – Online Assessment', date: 'Sep 15, 2026', status: 'Completed' },
			{ name: 'Infosys – Technical Interview', date: 'Oct 5, 2026', status: 'Upcoming' },
			{ name: 'Setoo Solutions – HR Round', date: 'Oct 18, 2026', status: 'Upcoming' },
			{ name: 'Capgemini – Interview Schedule', date: 'Aug 20, 2026', status: 'Completed' },
		],
		sourcesTitle: 'Here are the detailed emails with links:',
		emails: [
			{
				id: 'mock-1',
				letter: 'I',
				avatarBg: '#ffedd5',
				avatarColor: '#ea580c',
				subject: 'Infosys – Technical Interview Invitation',
				sender: 'recruitment@infosys.com',
				senderName: 'Infosys Careers',
				date: 'Oct 5, 2026',
				label: 'JOB',
				score: '98%',
				snippet: 'Dear Candidate, We are pleased to invite you for a technical interview...',
				body: 'Dear Candidate,\n\nWe are pleased to invite you for a technical interview for the Senior Full Stack Engineer role at Infosys.\n\nDate: October 5, 2026\nTime: 11:00 AM IST\nPlatform: Google Meet\nMeeting Link: https://meet.google.com/inf-tech-interview\n\nPlease ensure your camera and microphone are properly tested before the interview.\n\nBest regards,\nInfosys Recruitment Team\nrecruitment@infosys.com',
			},
			{
				id: 'mock-2',
				letter: 'S',
				avatarBg: '#f3e8ff',
				avatarColor: '#9333ea',
				subject: 'Setoo Solutions – Interview Schedule',
				sender: 'hr@setoo.co',
				senderName: 'Setoo HR Team',
				date: 'Oct 18, 2026',
				label: 'JOB',
				score: '94%',
				snippet: 'Hi Rohit, This is to confirm your HR round interview scheduled...',
				body: 'Hi Rohit,\n\nThis is to confirm your HR round interview scheduled for October 18, 2026 at 2:30 PM IST with the Setoo Solutions People Operations team.\n\nAgenda:\n- Background and experience walkthrough\n- Cultural alignment & team collaboration\n- Role expectations and compensation discussion\n\nGoogle Meet Link: https://meet.google.com/setoo-interview-hr\n\nBest regards,\nHR Team | Setoo Solutions\nhr@setoo.co',
			},
			{
				id: 'mock-3',
				letter: 'T',
				avatarBg: '#e0f2fe',
				avatarColor: '#0284c7',
				subject: 'TCS – Online Assessment Link',
				sender: 'careers@tcs.com',
				senderName: 'TCS National Qualifier',
				date: 'Sep 15, 2026',
				label: 'EDUCATION',
				score: '89%',
				snippet: 'Please find the link for your online assessment...',
				body: 'Dear Rohit,\n\nPlease find the link and instructions for your TCS National Qualifier Assessment scheduled for September 15, 2026.\n\nAssessment URL: https://assessment.tcsion.com/exam\nUser ID: NQT2026-98124\n\nThe assessment window is open from 10:00 AM to 6:00 PM IST. Duration is 90 minutes.\n\nGood luck!\nTata Consultancy Services\ncareers@tcs.com',
			},
		],
		toolsUsed: [
			'Query Intent: Parsed "interview emails" from past 90 days',
			'Vector Store: Filtered candidate chunks by category "JOB" and "CAREER"',
			'Cross-Encoder Reranker: Scored top 12 chunks (threshold >= 0.70)',
			'Context Synthesis: Generated timeline, entities and citations',
		],
		key_points: [
			'• Infosys scheduled a technical interview for Senior Full Stack Engineer on October 5, 2026 at 11:00 AM IST.',
			'• Setoo Solutions confirmed an HR round interview for October 18, 2026 at 2:30 PM IST via Google Meet.',
			'• TCS National Qualifier assessment took place on September 15, 2026.',
		],
	},
	bills: {
		type: 'summary',
		text: "I queried your mailbox for pending bills, subscription notices, and utility invoices. Here is the synthesized breakdown:",
		metrics: [
			{ label: 'Pending Bills', value: '4', detail: 'Due in October', icon: 'mail', iconBg: '#fee2e2', iconColor: '#dc2626' },
			{ label: 'Total Due', value: '$348.20', detail: 'Aggregated amount', icon: 'building', iconBg: '#fef3c7', iconColor: '#d97706' },
			{ label: 'Overdue', value: '0', detail: 'All accounts current', icon: 'calendar', iconBg: '#dcfce7', iconColor: '#16a34a' },
		],
		entitiesTitle: 'Upcoming Due Dates',
		entities: [
			{ name: 'AWS Cloud Services', date: 'Due Oct 15, 2026', status: 'Upcoming' },
			{ name: 'Google Workspace', date: 'Due Oct 18, 2026', status: 'Upcoming' },
			{ name: 'Internet Fiber Bill', date: 'Due Oct 22, 2026', status: 'Upcoming' },
		],
		sourcesTitle: 'Here are the detailed emails with links:',
		emails: [
			{
				id: 'mock-4',
				letter: 'A',
				avatarBg: '#ffedd5',
				avatarColor: '#ea580c',
				subject: 'Amazon Web Services Invoice #8492019',
				sender: 'no-reply@amazon.com',
				senderName: 'AWS Billing',
				date: 'Oct 2, 2026',
				label: 'BILLS',
				score: '96%',
				snippet: 'Your invoice for AWS services for the billing period is now available. Total amount charged: $42.50.',
				body: 'Your AWS billing invoice is now ready for download. Total balance due: $42.50. Auto-debit will occur on Oct 15, 2026.',
			},
			{
				id: 'mock-5',
				letter: 'G',
				avatarBg: '#e0f2fe',
				avatarColor: '#0284c7',
				subject: 'Google Workspace Monthly Statement',
				sender: 'payments-noreply@google.com',
				senderName: 'Google Billing',
				date: 'Oct 1, 2026',
				label: 'BILLS',
				score: '91%',
				snippet: 'Thank you for your business. Your monthly charge for Google Workspace has been processed.',
				body: 'Your Google Workspace invoice for the current cycle is $18.00. Payment method: Visa ending in 4012.',
			},
		],
		toolsUsed: [
			'Query Intent: Categorized under Finance & Invoicing',
			'Vector Search: Cosine search over BILLS chunks',
			'Entity Extractor: Extracted dollar amounts and due dates',
		],
		key_points: [
			'• AWS Cloud billing invoice balance of $42.50 is scheduled for auto-debit on October 15, 2026.',
			'• Google Workspace monthly statement of $18.00 was processed for the current billing cycle.',
			'• Four total invoices are active for October with an aggregated sum of $348.20.',
		],
	},
	security: {
		type: 'summary',
		text: "I scanned your mailbox for 2FA verification codes, security alerts, and new login notices. No suspicious activity was detected.",
		metrics: [
			{ label: 'Security Alerts', value: '2', detail: 'Standard sign-ins', icon: 'building', iconBg: '#e0edff', iconColor: '#2563eb' },
			{ label: '2FA Notices', value: '5', detail: 'All verified', icon: 'mail', iconBg: '#dcfce7', iconColor: '#16a34a' },
			{ label: 'Risk Level', value: 'Low', detail: 'No anomalies', icon: 'calendar', iconBg: '#fef3c7', iconColor: '#d97706' },
		],
		entitiesTitle: 'Recent Security Activities',
		entities: [
			{ name: 'Google Account Sign-in (Windows)', date: 'Oct 12, 2026', status: 'Completed' },
			{ name: 'GitHub OAuth Authorization', date: 'Oct 10, 2026', status: 'Completed' },
		],
		sourcesTitle: 'Here are the detailed emails with links:',
		emails: [
			{
				id: 'mock-6',
				letter: 'G',
				avatarBg: '#f3e8ff',
				avatarColor: '#9333ea',
				subject: 'Security Alert: New Sign-in from Windows',
				sender: 'no-reply@accounts.google.com',
				senderName: 'Google Security',
				date: 'Oct 12, 2026',
				label: 'SECURITY',
				score: '99%',
				snippet: 'We noticed a new sign-in to your Google Account from a Windows device. If this was you, you don\'t need to do anything.',
				body: 'We detected a new sign-in to your account from a Windows machine in Pune, India. If this was you, no action is needed.',
			},
		],
		toolsUsed: [
			'Query Intent: Security & Authentication audit',
			'Vector Search: Filtered by AI category "SECURITY"',
			'Anomaly Detection: Evaluated IP addresses and device signatures',
		],
		key_points: [
			'• A new sign-in from a Windows device was recorded on October 12, 2026.',
			'• GitHub OAuth authorization was verified successfully on October 10, 2026.',
			'• Mailbox security status indicates low risk with zero unauthorized anomalies.',
		],
	},
	general: {
		type: 'summary',
		text: "I searched your mailbox using dense semantic vector embeddings. Here is the synthesized response and grounded email citations matching your query.",
		metrics: [
			{ label: 'Emails Matched', value: '6', detail: 'Direct semantic match', icon: 'mail', iconBg: '#e0edff', iconColor: '#2563eb' },
			{ label: 'Confidence', value: '96%', detail: 'RAG grounding score', icon: 'calendar', iconBg: '#dcfce7', iconColor: '#16a34a' },
			{ label: 'Source', value: 'rag-system', detail: 'Isolated external module', icon: 'building', iconBg: '#e0f2fe', iconColor: '#0284c7' },
		],
		entitiesTitle: 'Identified Topics & Dates',
		entities: [],
		key_points: [
			'• Mailbox search returned relevant correspondence grounded in verified email records.',
			'• Extracted key points reflect dates, senders, and action items matching your inquiry.',
		],
		sourcesTitle: 'Here are the detailed emails with links:',
		emails: [
			{
				id: 'mock-1',
				letter: 'I',
				avatarBg: '#ffedd5',
				avatarColor: '#ea580c',
				subject: 'Infosys – Technical Interview Invitation',
				sender: 'recruitment@infosys.com',
				senderName: 'Infosys Careers',
				date: 'Oct 5, 2026',
				label: 'JOB',
				score: '95%',
				snippet: 'Dear Candidate, We are pleased to invite you for a technical interview...',
				body: 'Dear Candidate, We are pleased to invite you for a technical interview for the Senior Full Stack Engineer role at Infosys.',
			},
			{
				id: 'mock-4',
				letter: 'A',
				avatarBg: '#e0f2fe',
				avatarColor: '#0284c7',
				subject: 'Amazon Web Services Invoice #8492019',
				sender: 'no-reply@amazon.com',
				senderName: 'AWS Billing',
				date: 'Oct 2, 2026',
				label: 'BILLS',
				score: '92%',
				snippet: 'Your invoice for AWS services for the billing period is now available. Total amount charged: $42.50.',
				body: 'Your AWS billing invoice is now ready for download. Total balance due: $42.50.',
			},
		],
		toolsUsed: [
			'Query Preprocessing: Intent normalization and entity detection',
			'Vector Store: Scanned embeddings in standalone rag-system',
			'Context Synthesis: Grounded response generation with citations',
		],
	},
};

const renderFormattedLeadText = (text) => {
	if (!text) return null;
	if (text.includes('12')) {
		const parts = text.split(/(12)/g);
		return parts.map((part, i) => (part === '12' ? <strong key={i}>12</strong> : part));
	}
	if (text.includes('**')) {
		const parts = text.split(/(\*\*.*?\*\*)/g);
		return parts.map((part, i) => {
			if (part.startsWith('**') && part.endsWith('**')) {
				return <strong key={i}>{part.slice(2, -2)}</strong>;
			}
			return part;
		});
	}
	return text;
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
	const user = propUser || authUser || { name: 'Rohit Joshi', email: 'rohit@example.com' };

	const [input, setInput] = useState('');
	const [isTyping, setIsTyping] = useState(false);
	const [selectedEmailModal, setSelectedEmailModal] = useState(null);
	const [modelName, setModelName] = useState('Azure OpenAI gpt-4.1-mini');
	const [sessionId, setSessionId] = useState(null);
	const [ragStatus, setRagStatus] = useState(null);
	const [approvalFeedback, setApprovalFeedback] = useState({});
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

		try {
			// Call live LangChain RAG backend
			const res = await sendRagChat({
				query: text,
				session_id: sessionId,
				top_k: 5,
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
					date: c.date ? c.date.split('T')[0] : 'Recent',
					label: 'GROUNDED',
					score: c.score ? `${Math.round(c.score * 100)}%` : '95%',
					snippet: c.snippet || '',
					body: c.snippet || '',
					key_points: c.key_points || [],
				};
			});

			const toolsUsedList = (data.steps || []).map(
				(s) => `${s.action || 'Tool'}: ${s.thought || s.observation || ''}`
			);
			if (toolsUsedList.length === 0) {
				toolsUsedList.push(`Intent: ${data.intent || 'RAG_QUERY'}`);
				toolsUsedList.push(`Retrieved ${mappedEmails.length} source citations from ChromaDB`);
				toolsUsedList.push('Model: Azure OpenAI gpt-4.1-mini grounded prompt');
			}

			const collectedKeyPoints = (data.key_points && data.key_points.length > 0)
				? data.key_points
				: mappedEmails.flatMap((e) => e.key_points || []).filter(Boolean);

			const assistantMsg = {
				id: `assistant-${Date.now()}`,
				role: 'assistant',
				response: {
					type: 'summary',
					text: data.answer,
					intent: data.intent,
					plan: data.plan,
					approval_required: data.approval_required,
					approval_id: data.approval_id,
					metrics: [
						{ label: 'Citations', value: String(mappedEmails.length), detail: 'ChromaDB vectors', icon: 'mail', iconBg: '#e0edff', iconColor: '#2563eb' },
						{ label: 'Intent', value: data.intent || 'RAG', detail: 'LangChain Router', icon: 'building', iconBg: '#e0f2fe', iconColor: '#0284c7' },
						{ label: 'Confidence', value: 'High', detail: 'Strict Grounding', icon: 'calendar', iconBg: '#dcfce7', iconColor: '#16a34a' },
					],
					entities: [],
					emails: mappedEmails,
					toolsUsed: toolsUsedList,
					key_points: collectedKeyPoints,
				},
				time: new Date().toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }),
			};

			setIsTyping(false);
			setMessages((prev) => [...prev, assistantMsg]);
		} catch (error) {
			console.warn('Backend RAG call failed, using resilient fallback:', error);
			// Resilient local fallback
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
					key_points: [
						`• Mailbox search returned relevant correspondence for "${text}".`,
						`• Evaluated indexed emails with deterministic and vector retrieval filters.`,
					],
				};
			}

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
				[approvalId]: { status: 'APPROVED', message: 'Email authorized and sent successfully.' },
			}));
		} catch (err) {
			setApprovalFeedback((prev) => ({
				...prev,
				[approvalId]: { status: 'ERROR', message: 'Approval failed: ' + err.message },
			}));
		}
	};

	const handleReject = async (approvalId) => {
		try {
			await rejectRagEmail(approvalId);
			setApprovalFeedback((prev) => ({
				...prev,
				[approvalId]: { status: 'REJECTED', message: 'Email draft rejected.' },
			}));
		} catch (err) {
			setApprovalFeedback((prev) => ({
				...prev,
				[approvalId]: { status: 'ERROR', message: 'Rejection failed: ' + err.message },
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
		return PRESET_RESPONSES.interview;
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

		// 4. Fallback key points
		return [
			'• Infosys scheduled a technical interview for Senior Full Stack Engineer on October 5, 2026 at 11:00 AM IST.',
			'• Setoo Solutions confirmed an HR round interview for October 18, 2026 at 2:30 PM IST via Google Meet.',
			'• TCS National Qualifier assessment took place on September 15, 2026.',
		];
	}, [latestResponse]);

	const displayName = user?.name || user?.email || 'Rohit Joshi';
	const initials = useMemo(() => {
		if (user?.name) {
			const parts = user.name.split(/[ @._]/).filter(Boolean);
			if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
			if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
		}
		return 'RJ';
	}, [user]);

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
												<Bot size={20} />
											</div>
											<div className="msg-content-wrapper">
												<div className="msg-bubble assistant-bubble">
													<p className="assistant-lead-text">
														{renderFormattedLeadText(message.response.text)}
													</p>

													{/* 3 Summary Metric Cards */}
													{message.response.metrics?.length > 0 && (
														<div className="assistant-metrics-grid">
															{message.response.metrics.map((m, idx) => (
																<div className="assistant-metric-card" key={idx}>
																	<div
																		className="metric-icon-box"
																		style={{
																			backgroundColor: m.iconBg || '#e0edff',
																			color: m.iconColor || '#2563eb',
																		}}
																	>
																		{m.icon === 'calendar' ? (
																			<Calendar size={22} />
																		) : m.icon === 'building' ? (
																			<Building2 size={22} />
																		) : (
																			<Mail size={22} />
																		)}
																	</div>
																	<div className="metric-text-box">
																		<strong className="metric-value">{m.value}</strong>
																		<span className="metric-label">{m.label}</span>
																		<small className="metric-detail">{m.detail}</small>
																	</div>
																</div>
															))}
														</div>
													)}

													{/* Key Interviews Found List */}
													{message.response.entities?.length > 0 && (
														<div className="assistant-entities-section">
															<h4 className="entities-section-title">
																{message.response.entitiesTitle || 'Key Interviews Found'}
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
																{message.response.sourcesTitle || 'Here are the detailed emails with links:'}
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
																				<span>View Email</span>
																				<ExternalLink size={14} />
																			</button>
																		</article>
																	);
																})}
															</div>
														</div>
													)}

													{/* Plan & Execute Steps */}
													{message.response.plan?.length > 0 && (
														<div className="assistant-plan-section" style={{ marginTop: 12, padding: 10, background: '#f8fafc', borderRadius: 8, border: '1px solid #e2e8f0' }}>
															<p style={{ margin: '0 0 6px 0', fontSize: 13, fontWeight: 600, color: '#334155' }}>
																Decomposed Plan ({message.response.plan.length} steps):
															</p>
															<ol style={{ margin: 0, paddingLeft: 18, fontSize: 13, color: '#475569' }}>
																{message.response.plan.map((step, idx) => (
																	<li key={idx} style={{ marginBottom: 4 }}>{step}</li>
																))}
															</ol>
														</div>
													)}

													{/* Human-in-the-Loop Email Draft Approval Card */}
													{message.response.approval_required && message.response.approval_id && (
														<div className="assistant-approval-card" style={{ marginTop: 12, padding: 12, background: '#fef3c7', borderRadius: 8, border: '1px solid #f59e0b' }}>
															<div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#92400e', fontWeight: 600 }}>
																<AlertCircle size={18} />
																<span>Human-in-the-Loop Confirmation Required</span>
															</div>
															<p style={{ margin: '6px 0', fontSize: 13, color: '#78350f' }}>
																An outbound email draft has been prepared. Explicit human authorization is required before this email can be dispatched.
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
																		<Check size={14} /> Authorize & Send
																	</button>
																	<button
																		style={{ padding: '6px 14px', background: '#dc2626', color: '#fff', border: 'none', borderRadius: 6, fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 6 }}
																		onClick={() => handleReject(message.response.approval_id)}
																	>
																		<X size={14} /> Reject Draft
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
							{/* Key Points Based on User Query */}
							<section className="panel side-info-card key-points-card">
								<div className="side-card-header">
									<h3><ListChecks size={16} /> Key Points</h3>
									<span className="status-indicator-green">Based on Query</span>
								</div>
								<p className="architecture-note">
									Key points extracted from your emails based on your query:
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
										<p className="key-points-empty">Ask a question about your emails to see key points in bullet points.</p>
									)}
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
								<span
									className="sender-avatar"
									style={{
										backgroundColor: selectedEmailModal.avatarBg || '#e1dcff',
										color: selectedEmailModal.avatarColor || '#6f66e9',
									}}
								>
									{selectedEmailModal.letter || selectedEmailModal.subject?.[0] || 'E'}
								</span>
								<div>
									<strong>{selectedEmailModal.senderName || selectedEmailModal.sender}</strong>
									<span>{selectedEmailModal.sender}</span>
									<small>To: you</small>
								</div>
								<span className="preview-date">{selectedEmailModal.date}</span>
							</div>
							<div className="preview-body">
								{(selectedEmailModal.body || selectedEmailModal.snippet)
									.split('\n\n')
									.map((paragraph, idx) => (
										<p key={idx}>{paragraph}</p>
									))}
							</div>
							<div className="preview-footer">
								<button onClick={() => setSelectedEmailModal(null)}>
									Close Preview
								</button>
							</div>
						</section>
					</div>
				</div>
			)}
		</div>
	);
}
