import { useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate, Navigate } from 'react-router-dom';
import {
	AlertCircle, BarChart3, Bell, ChevronDown, CircleHelp, Download, FileText, Inbox,
	LayoutDashboard, Mail, Menu, MoreHorizontal, Paperclip, PenLine, RefreshCw, Search,
	Send, Settings, ShieldAlert, Star, Tag, Trash2, Upload, Users, X, Zap, Sparkles, LogOut, Maximize2,
	AlertTriangle, UserX,
} from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import DOMPurify from 'dompurify';
import { useAuth } from './context/AuthContext';
import GmailConsent from './pages/GmailConsent';
import SyncPage from './pages/SyncPage';
import AiAssistant from './pages/AiAssistant';
import { fetchAnalyticsCharts, fetchAnalyticsSummary, fetchEmail, fetchEmails, getExportStats, getExportCSVUrl, getExportJSONUrl, importEmailsFile, startSync, getSyncStatus, trainModel, getRagStatus, triggerRagIndex, enterDemo } from './services/api';

const navItems = [
	{ label: 'Inbox', icon: Inbox }, { label: 'Starred', icon: Star },
	{ label: 'Sent', icon: Send }, { label: 'Important', icon: Tag },
	{ label: 'Drafts', icon: FileText, count: 3 }, { label: 'Spam', icon: ShieldAlert },
	{ label: 'Trash', icon: Trash2 },
];

function Sidebar({ active, onNavigate, onClose, unreadCount }) {
	const navigate = useNavigate();
	return <aside className="sidebar">
		<div className="brand"><div className="brand-mark"><Mail size={18} /></div><span>Gmail AI <strong>Dashboard</strong></span><button className="mobile-close" onClick={onClose}><X size={18} /></button></div>
		<button className="compose-button"><PenLine size={17} /> Compose</button>
		<div className="nav-group">
			<button className={`nav-item ${active === 'Dashboard' ? 'active' : ''}`} onClick={() => onNavigate('Dashboard')}>
				<LayoutDashboard size={17} />
				<span>Dashboard</span>
			</button>
		</div>
		<div className="nav-group"><p className="nav-label">MAILBOX</p>{navItems.map(({ label, icon: Icon, count }) => <button key={label} className={`nav-item ${active === label ? 'active' : ''}`} onClick={() => onNavigate(label)}><Icon size={17} /><span>{label}</span>{label === 'Inbox' ? <small>{unreadCount}</small> : count ? <small>{count}</small> : null}</button>)}</div>
		<div className="nav-group nav-lower">
			<p className="nav-label">WORKSPACE</p>
			<button className={`nav-item ${active === 'Assistant' ? 'active' : ''}`} onClick={() => navigate('/assistant')}>
				<Zap size={17} />
				<span>AI Assistant</span>
				<span className="new-pill">RAG</span>
			</button>
			<button className="nav-item" onClick={() => navigate('/settings')}><Settings size={17} /><span>Settings</span></button>
			<button className="nav-item"><CircleHelp size={17} /><span>Help & Support</span></button>
		</div>
		<div className="sidebar-footer"><div className="plan-icon"><BarChart3 size={17} /></div><div><strong>AI model status</strong><span><i />Ready to classify</span></div><MoreHorizontal size={16} /></div>
	</aside>;
}

function Header({ search, setSearch, onMenu, user, onLogout, onNavigateSettings }) {
	const [profileOpen, setProfileOpen] = useState(false);
	const initials = user?.name ? user.name.split(' ').map((n) => n[0]).join('').slice(0, 2).toUpperCase() : user?.email?.[0]?.toUpperCase() || '?';
	const displayName = user?.name || user?.email || 'User';

	return (
		<header className="topbar">
			<button className="menu-button" onClick={onMenu}><Menu size={21} /></button>
			<div className="search-box">
				<Search size={18} />
				<input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search emails, people, or keywords..." />
				<kbd>⌘ K</kbd>
			</div>
			<div className="top-actions">
				<button className="icon-button notification"><Bell size={19} /><i /></button>
				<div className="profile-menu">
					<button className="user-profile" onClick={() => setProfileOpen((open) => !open)} aria-expanded={profileOpen}>
						<div className="avatar">{user?.picture ? <img src={user.picture} alt="" className="avatar-img" /> : initials}</div>
						<div className="user-copy"><strong>{displayName}</strong><span>{user?.email || ''}</span></div>
						<ChevronDown size={16} />
					</button>
					{profileOpen && (
						<div className="profile-dropdown">
							{user?.picture && <div className="profile-dropdown-user"><img src={user.picture} alt="" className="dropdown-avatar" /><div><strong>{displayName}</strong><small>{user?.email}</small></div></div>}
							<div className="profile-dropdown-actions">
								<button className="dropdown-action" onClick={onNavigateSettings}><Settings size={15} /> Settings</button>
								<button className="dropdown-action logout-action" onClick={onLogout}><LogOut size={15} /> Log out</button>
							</div>
						</div>
					)}
				</div>
			</div>
		</header>
	);
}

function StatCard({ stat }) {
	const icons = { inbox: Inbox, tag: Tag, important: Zap, unread: Mail };
	const Icon = icons[stat.icon];
	return <div className="stat-card"><div className={`stat-icon ${stat.icon}`}><Icon size={19} /></div><div className="stat-content"><span>{stat.label}</span><strong>{stat.value}</strong><small className={stat.positive ? 'positive' : 'negative'}>{stat.change} <em>{stat.comparisonText || 'vs last period'}</em></small></div><div className="stat-spark">{[10, 15, 11, 19, 14, 21, 19].map((height, index) => <i key={index} style={{ height }} />)}</div></div>;
}

const dashboardRanges = [
	['today', 'Today'],
	['yesterday', 'Yesterday'],
	['this_week', 'This week'],
	['last_week', 'Last week'],
	['this_month', 'This month'],
	['last_month', 'Last month'],
	['six_months', '6 months'],
	['one_year', '1 year'],
	['all_time', 'All time'],
];

const comparisonLabels = {
	today: 'vs yesterday',
	yesterday: 'vs day before',
	this_week: 'vs last week',
	last_week: 'vs prior week',
	this_month: 'vs last month',
	last_month: 'vs prior month',
	six_months: 'vs prior 6m',
	one_year: 'vs prior year',
	all_time: 'overall',
};

function formatChange(current, comparison) {
	if (comparison === 0) return current === 0 ? '0%' : '+100%';
	const change = ((current - comparison) / comparison) * 100;
	return `${change >= 0 ? '+' : ''}${change.toFixed(1)}%`;
}

function buildStats(summary, rangeKey = 'this_week') {
	const compText = comparisonLabels[rangeKey] || 'vs last period';
	const labels = [
		['Total Emails', 'total', 'inbox'], ['Auto Labeled', 'auto_labeled', 'tag'],
		['Important', 'important', 'important'], ['Unread', 'unread', 'unread'],
	];
	return labels.map(([label, key, icon]) => ({
		label,
		icon,
		value: summary.current[key].toLocaleString(),
		change: formatChange(summary.current[key], summary.comparison[key]),
		positive: summary.current[key] >= summary.comparison[key],
		comparisonText: compText,
	}));
}

const activityRanges = dashboardRanges;
const chartColors = ['#2563eb', '#16a34a', '#eab308', '#dc2626', '#8b5cf6', '#06b6d4', '#ec4899'];

function CategoryTooltip({ active, payload }) {
	if (!active || !payload?.length) return null;
	const category = payload[0].payload;
	return <div className="category-tooltip"><strong>Category: {category.name}</strong><span>{category.count.toLocaleString()} mails</span>{category.breakdown?.length > 0 && <div className="category-breakdown"><b>Other categories</b>{category.breakdown.map((item) => <span key={item.name}>{item.name}: {item.count.toLocaleString()} mails</span>)}</div>}</div>;
}

function Charts({ charts, range, onRangeChange }) {
	const categories = charts?.categories || [];
	const activity = charts?.activity || [];
	const total = charts?.total_emails || 0;
	return (
		<div className="charts-grid">
			<section className="panel category-panel">
				<div className="panel-heading">
					<div>
						<h2>Emails by Category</h2>
						<p>Distribution of your inbox</p>
					</div>
					<button className="more-button"><MoreHorizontal size={18} /></button>
				</div>
				<div className="donut-wrap">
					<ResponsiveContainer width="100%" height={190}>
						<PieChart>
							<Pie
								data={categories.length > 0 ? categories : [{ name: 'No emails', value: 1 }]}
								dataKey="value"
								nameKey="name"
								innerRadius={59}
								outerRadius={83}
								paddingAngle={categories.length > 0 ? 4 : 0}
								stroke="none"
							>
								{categories.length > 0 ? (
									categories.map((item, index) => (
										<Cell key={item.name} fill={chartColors[index % chartColors.length]} />
									))
								) : (
									<Cell key="empty" fill="#edf0f5" />
								)}
							</Pie>
							<Tooltip content={<CategoryTooltip />} />
						</PieChart>
					</ResponsiveContainer>
					<div className="donut-total">
						<strong>{total.toLocaleString()}</strong>
						<span>total emails</span>
					</div>
				</div>
				<div className="legend-grid">
					{categories.map((item, index) => (
						<div key={item.name}>
							<i className="legend-dot" style={{ backgroundColor: chartColors[index % chartColors.length] }} />
							<span>{item.name}</span>
							<strong>{item.value}%</strong>
						</div>
					))}
				</div>
			</section>
			<section className="panel activity-panel">
				<div className="panel-heading">
					<div>
						<h2>Email Activity</h2>
						<p>Incoming and outgoing emails</p>
					</div>
					<label className="range-button">
						<select value={range} onChange={(event) => onRangeChange(event.target.value)} aria-label="Activity date range">
							{dashboardRanges.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
						</select>
						<ChevronDown size={14} />
					</label>
				</div>
				<div className="chart-key">
					<span><i className="line-incoming" /> Incoming</span>
					<span><i className="line-outgoing" /> Outgoing</span>
				</div>
				<ResponsiveContainer width="100%" height={234}>
					<LineChart data={activity} margin={{ top: 15, right: 10, left: -23, bottom: 0 }}>
						<CartesianGrid vertical={false} stroke="#edf0f5" />
						<XAxis dataKey="label" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: '#9aa4b5' }} />
						<YAxis axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: '#9aa4b5' }} allowDecimals={false} />
						<Tooltip contentStyle={{ border: '0', borderRadius: 8, boxShadow: '0 8px 22px #1d244015', fontSize: 12 }} />
						<Line type="monotone" dataKey="incoming" stroke="#2563eb" strokeWidth={2.5} dot={false} />
						<Line type="monotone" dataKey="outgoing" stroke="#334155" strokeWidth={2.5} dot={false} />
					</LineChart>
				</ResponsiveContainer>
			</section>
		</div>
	);
}

function QuickActions({ onAction }) {
	const actions = [{ label: 'Sync Gmail', detail: 'Fetch latest messages', icon: Mail, color: 'purple' }, { label: 'Train Model', detail: 'Improve AI accuracy', icon: Zap, color: 'mint' }, { label: 'Manage Labels', detail: 'Organize your inbox', icon: Tag, color: 'gold' }, { label: 'View Analytics', detail: 'See detailed insights', icon: BarChart3, color: 'coral' }];
	return <section className="quick-section"><div className="section-heading"><div><h2>Quick Actions</h2><p>Manage your inbox efficiently</p></div></div><div className="quick-grid">{actions.map(({ label, detail, icon: Icon, color }) => <button className="quick-action" key={label} onClick={() => onAction(label)}><div className={`quick-icon ${color}`}><Icon size={18} /></div><div><strong>{label}</strong><span>{detail}</span></div><ChevronDown className="action-arrow" size={16} /></button>)}</div></section>;
}

function RecentEmails({ emailsToShow, selectedId, onSelectEmail, selected, onToggleStar, selectedIds, onToggleSelect, onOpenMailbox, title = 'Recent Emails', loading = false }) {
	return (
		<section className="panel recent-panel">
			<div className="panel-heading">
				<div>
					<h2>{title}</h2>
					<p>{title === 'Inbox' ? 'All messages in your Gmail inbox' : 'Your latest messages and AI classifications'}</p>
				</div>
				<button className="view-all" onClick={onOpenMailbox}>
					{title === 'Inbox' ? `${emailsToShow.length} mails` : 'View all'} <span>→</span>
				</button>
			</div>
			<div className="email-table">
				{loading ? (
					<div className="mailbox-loading">Loading inbox...</div>
				) : emailsToShow.length === 0 ? (
					<div className="mailbox-empty" style={{ padding: '2.5rem 1rem', textAlign: 'center', color: '#6b7280' }}>
						<p style={{ margin: 0, fontSize: '0.95rem' }}>No emails found for this selected period.</p>
					</div>
				) : (
					<>
						<div className="table-head">
							<span className="check-cell">
								<input
									type="checkbox"
									checked={selectedIds.length === emailsToShow.length && emailsToShow.length > 0}
									onChange={() => emailsToShow.forEach((email) => onToggleSelect(email.id))}
								/>
							</span>
							<span>Sender</span>
							<span>Subject</span>
							<span>Label</span>
							<span>Date</span>
							<span />
						</div>
						{emailsToShow.map((email) => (
							<button
								className={`email-row ${email.is_read ? '' : 'unread-row'} ${selectedId === email.id ? 'selected' : ''}`}
								key={email.id}
								onClick={() => onSelectEmail?.(email.id)}
							>
								<span className="check-cell" onClick={(event) => event.stopPropagation()}>
									<input
										type="checkbox"
										checked={selectedIds.includes(email.id)}
										onChange={() => onToggleSelect(email.id)}
									/>
								</span>
								<span className="sender-cell">
									<span className="sender-avatar" style={{ background: email.avatarColor || '#e1dcff' }}>
										{email.initials}
									</span>
									<strong>{email.sender}</strong>
									{email.hasAttachment && <Paperclip size={13} />}
								</span>
								<span className="subject-cell">
									<strong>{email.subject}</strong>
									<small>{email.preview}</small>
								</span>
								<span>
									<em className={`label-badge ${(email.label || '').toLowerCase()}`}>{email.label}</em>
								</span>
								<span className="date-cell">{email.date}</span>
								<span
									className={`star ${email.starred ? 'starred' : ''}`}
									onClick={(event) => {
										event.stopPropagation();
										onToggleStar(email.id);
									}}
								>
									<Star size={17} fill={email.starred ? 'currentColor' : 'none'} />
								</span>
							</button>
						))}
					</>
				)}
			</div>
		</section>
	);
}

function InboxView({ emails, selectedId, onSelectEmail, selectedIds, onToggleSelect, unreadCount, loading, loadingMore, hasMore, onLoadMore }) {
	const [categoryFilter, setCategoryFilter] = useState('All');
	const categories = ['All', 'Primary', 'Social', 'Promotions', 'Updates', 'Forums'];
	const visibleEmails = categoryFilter === 'All' ? emails : emails.filter((email) => email.gmail_category === categoryFilter);
	return (
		<section className="panel inbox-view">
			<div className="panel-heading">
				<div>
					<h2>Inbox</h2>
					<p>{emails.length.toLocaleString()} messages loaded, {unreadCount.toLocaleString()} unread</p>
				</div>
				<button className="view-all">{emails.length.toLocaleString()} mails</button>
			</div>
			<div className="mail-category-tabs" role="group" aria-label="Gmail categories">
				{categories.map((category) => <button key={category} type="button" className={categoryFilter === category ? 'active' : ''} onClick={() => setCategoryFilter(category)}>{category}</button>)}
			</div>
			<div
				className="email-table inbox-scroll"
				onScroll={(event) => {
					const element = event.currentTarget;
					if (element.scrollHeight - element.scrollTop - element.clientHeight < 160) onLoadMore();
				}}
			>
				{loading ? (
					<div className="mailbox-loading">Loading inbox...</div>
				) : (
					<>
						<div className="table-head">
							<span className="check-cell">
								<input
									type="checkbox"
									checked={selectedIds.length === visibleEmails.length && visibleEmails.length > 0}
									onChange={() => visibleEmails.forEach((email) => onToggleSelect(email.id))}
								/>
							</span>
							<span>Sender</span>
							<span>Subject</span>
							<span>Label</span>
							<span>Date</span>
							<span />
						</div>
						{visibleEmails.map((email) => (
							<button
								className={`email-row ${email.is_read ? '' : 'unread-row'} ${selectedId === email.id ? 'selected' : ''}`}
								key={email.id}
								onClick={() => onSelectEmail?.(email.id)}
							>
								<span className="check-cell" onClick={(event) => event.stopPropagation()}>
									<input
										type="checkbox"
										checked={selectedIds.includes(email.id)}
										onChange={() => onToggleSelect(email.id)}
									/>
								</span>
								<span className="sender-cell">
									<span className="sender-avatar" style={{ background: email.avatarColor || '#e1dcff' }}>
										{email.initials}
									</span>
									<strong>{email.sender}</strong>
								</span>
								<span className="subject-cell">
									<strong>{email.subject || '(No subject)'}</strong>
									<small>{email.preview}</small>
								</span>
								<span>
									<em className={`label-badge ${(email.label || '').toLowerCase()}`}>{email.label}</em>
									{email.gmail_category && <small className="gmail-category-badge">{email.gmail_category}</small>}
								</span>
								<span className="date-cell">{email.date}</span>
								<span className="star"><Star size={17} /></span>
							</button>
						))}
						{visibleEmails.length === 0 && <div className="mailbox-empty">No messages in {categoryFilter} for this period.</div>}
						{loadingMore && <div className="mailbox-loading">Loading more messages...</div>}
						{!hasMore && emails.length > 0 && <div className="mailbox-end">All inbox messages loaded</div>}
					</>
				)}
			</div>
		</section>
	);
}

function EmailPreview({ email, onClose, onExpand, isModal = false }) {
	if (!email) {
		return (
			<section className="panel preview-panel empty-preview">
				<Mail size={24} />
				<p>Select an email to preview it</p>
			</section>
		);
	}
	const sanitizedHtml = email.body_html ? DOMPurify.sanitize(email.body_html, { USE_PROFILES: { html: true } }) : '';
	return (
		<section className={`panel preview-panel ${isModal ? 'is-modal-panel' : ''}`}>
			<div className="preview-header">
				<div>
					<div className="preview-header-meta">
						<span className="preview-label">{isModal ? 'EMAIL DETAILS' : 'EMAIL PREVIEW'}</span>
						{email.label && <em className={`label-badge ${(email.label || '').toLowerCase()}`}>{email.label}</em>}
						{email.gmail_category && <small className="gmail-category-badge">{email.gmail_category}</small>}
					</div>
					<h2>{email.subject || '(No subject)'}</h2>
				</div>
				<div className="preview-header-actions">
					{!isModal && onExpand && (
						<button className="more-button" onClick={onExpand} title="Open in popup window" aria-label="Open email preview in popup">
							<Maximize2 size={17} />
						</button>
					)}
					{onClose && (
						<button className="more-button" onClick={onClose} title={isModal ? 'Close window' : 'Close preview'} aria-label="Close email preview">
							<X size={18} />
						</button>
					)}
				</div>
			</div>
			<div className="preview-sender">
				<span className="sender-avatar" style={{ background: email.avatarColor || '#e1dcff' }}>
					{email.initials || '?'}
				</span>
				<div>
					<strong>{email.sender}</strong>
					<span>{email.email || email.from_email}</span>
					<small>To: {email.to_email || 'you'}</small>
				</div>
				<span className="preview-date">{email.date || ''}</span>
			</div>
				<div className="preview-body">
				{email.labels?.length > 0 && <p className="preview-labels">Labels: {email.labels.filter((label) => label.name !== email.gmail_category).map((label) => label.name).join(', ')}</p>}
				{sanitizedHtml ? (
					<div className="email-html-body" dangerouslySetInnerHTML={{ __html: sanitizedHtml }} />
				) : email.body ? (
					email.body.split('\n').filter(Boolean).map((paragraph, index) => <p key={index}>{paragraph}</p>)
				) : email.preview ? (
					<p>{email.preview}</p>
				) : (
					<p>No message body available.</p>
				)}
			</div>
			<div className="preview-footer">
				<button><ReplyIcon /> Reply</button>
				<button><ForwardIcon /> Forward</button>
			</div>
		</section>
	);
}
function ReplyIcon() { return <Send size={14} />; }
function ForwardIcon() { return <Send size={14} className="forward-icon" />; }

function GmailMark() {
	return <span className="gmail-mark" aria-hidden="true"><i className="gmail-blue">M</i><i className="gmail-red">M</i><i className="gmail-yellow">M</i><i className="gmail-green">M</i></span>;
}

function AuthIllustration({ signup }) {
	if (signup) {
		return <div className="signup-illustration" aria-hidden="true"><div className="person"><div className="person-hair" /><div className="person-face" /><div className="person-body" /></div><div className="laptop"><div /></div><div className="insight-list"><div><span className="insight-icon mint">↗</span><p><strong>AI-powered insights</strong><small>Understand your emails better</small></p></div><div><span className="insight-icon blue">◆</span><p><strong>Smart organization</strong><small>Find what matters faster</small></p></div><div><span className="insight-icon coral">▥</span><p><strong>Be more productive</strong><small>Save time, focus on what counts</small></p></div></div></div>;
	}
	return <div className="login-illustration" aria-hidden="true"><div className="leaf leaf-one" /><div className="leaf leaf-two" /><div className="leaf leaf-three" /><div className="inbox-window"><div className="mini-toolbar"><GmailMark /><span /></div><div className="mini-sidebar"><b>▣ Inbox</b><span>☆ Starred</span><span>› Important</span><span>› Sent</span><span>› Drafts</span><span>⌄ More</span></div><div className="mini-messages">{[1, 2, 3, 4].map((item) => <div key={item}><i /><span /><em /></div>)}</div></div><div className="productivity-card"><Sparkles size={25} /><strong>Smarter emails<br />Brighter productivity</strong><span /><span /></div></div>;
}

function AuthPage({ signup }) {
	const navigate = useNavigate();
	const { checkAuth } = useAuth();
	const [demoError, setDemoError] = useState('');
	const [openingDemo, setOpeningDemo] = useState(false);
	const submit = (event) => { event.preventDefault(); navigate('/consent'); };
	const tryDemo = async () => {
		try {
			setDemoError('');
			setOpeningDemo(true);
			await enterDemo();
			await checkAuth();
			navigate('/', { replace: true });
		} catch (error) {
			setDemoError(error.response?.data?.detail || 'The demo is unavailable right now.');
		} finally {
			setOpeningDemo(false);
		}
	};
	return <div className={`auth-shell ${signup ? 'auth-signup' : 'auth-login'}`}><main className="auth-card"><div className="auth-brand"><GmailMark /><strong>Gmail AI Dashboard</strong></div><div className="auth-copy"><h1>{signup ? 'Create Your Account' : 'Welcome Back'}</h1><p>{signup ? 'Get started with your Gmail AI Dashboard' : 'Sign in to access your Gmail AI Dashboard'}</p></div><form className="auth-form" onSubmit={submit}><button className="google-button" type="submit"><span className="google-g">G</span>{signup ? 'Sign up with Google' : 'Continue with Google'}</button><p>Gmail sign-in is available to invited test users.</p><div className="auth-divider"><span /> <em>OR</em> <span /></div><button className="google-button" type="button" onClick={tryDemo} disabled={openingDemo}>{openingDemo ? 'Opening demo...' : 'Open demo@gmail.com'}</button><p>Explore a read-only mailbox with fictional emails. No Google sign-in needed.</p>{demoError && <p role="alert">{demoError}</p>}<p className="auth-switch">{signup ? 'Already have an account?' : "Don't have an account?"} <button type="button" onClick={() => navigate(signup ? '/login' : '/signup')}>{signup ? 'Sign in' : 'Sign up'}</button></p></form><AuthIllustration signup={signup} /><p className="auth-tagline">{signup ? 'Join and make your inbox work smarter.' : 'Your Gmail. Smarter with AI.'}</p></main></div>;
}

function EraseDialog({ open, onClose, onConfirm, confirming }) {
	if (!open) return null;
	return <div className="modal-overlay" onClick={onClose}>
		<div className="modal-card erase-modal" onClick={(e) => e.stopPropagation()}>
			<div className="erase-icon-wrap">
				<AlertTriangle size={28} />
			</div>
			<h2>Erase all data?</h2>
			<p>
				This will permanently delete your synchronized emails and all data stored with them,
				including subjects, bodies, dates, attachments, classifications, and AI embeddings.
				Your profile and labels will be kept.
			</p>
			<p className="erase-warning">This action cannot be undone.</p>
			<div className="erase-actions">
				<button type="button" className="erase-cancel" onClick={onClose} disabled={confirming}>Cancel</button>
				<button type="button" className="erase-confirm" onClick={onConfirm} disabled={confirming} aria-busy={confirming}>
					<UserX size={15} /> {confirming ? 'Erasing...' : 'Erase All Data'}
				</button>
			</div>
		</div>
	</div>;
}

function SettingsPage() {
	const { user, logout, eraseAllData } = useAuth();
	const navigate = useNavigate();
	const [eraseOpen, setEraseOpen] = useState(false);
	const [erasing, setErasing] = useState(false);
	const [eraseToast, setEraseToast] = useState('');
	const [exportStats, setExportStats] = useState(null);
	const [includeBody, setIncludeBody] = useState(false);
	const [exporting, setExporting] = useState(false);
	const [importing, setImporting] = useState(false);
	const [importStatus, setImportStatus] = useState(null);
	const [ragStatus, setRagStatus] = useState(null);
	const [embedding, setEmbedding] = useState(false);
	const [embeddingStatus, setEmbeddingStatus] = useState('');

	const loadStats = () => {
		getExportStats()
			.then((res) => setExportStats(res.data))
			.catch(() => {});
	};

	useEffect(() => {
		if (user?.is_demo) return;
		loadStats();
		getRagStatus()
			.then(({ data }) => setRagStatus(data))
			.catch(() => setEmbeddingStatus('Embedding status is unavailable right now.'));
	}, [user]);

	const refreshRagStatus = async () => {
		const { data } = await getRagStatus();
		setRagStatus(data);
		return data;
	};

	const handleStartEmbeddings = async () => {
		if (embedding) return;
		setEmbedding(true);
		setEmbeddingStatus('Embedding emails... This may take a while.');
		try {
			const { data } = await triggerRagIndex();
			await refreshRagStatus();
			setEmbeddingStatus(data.message || 'Embedding completed.');
		} catch (err) {
			setEmbeddingStatus(err.response?.data?.detail || 'Embedding failed. Please try again.');
		} finally {
			setEmbedding(false);
		}
	};

	const handleLogout = async () => {
		await logout();
		navigate('/login', { replace: true });
	};

	const handleErase = async () => {
		if (erasing) return;
		setErasing(true);
		try {
			await eraseAllData();
			setEraseOpen(false);
			setEraseToast('All your data has been erased.');
			setTimeout(() => navigate('/login', { replace: true }), 1500);
		} catch {
			setEraseToast('Failed to erase data. Please try again.');
		} finally {
			setErasing(false);
		}
	};

	const handleExportCSV = () => {
		const url = getExportCSVUrl({ include_body: includeBody });
		window.open(url, '_blank');
	};

	const handleExportJSON = () => {
		const url = getExportJSONUrl({ include_body: includeBody, include_html: false });
		window.open(url, '_blank');
	};

	const handleFileUpload = async (event) => {
		const file = event.target.files?.[0];
		if (!file) return;

		setImporting(true);
		setImportStatus(null);
		const formData = new FormData();
		formData.append('file', file);

		try {
			const { data } = await importEmailsFile(formData);
			setImportStatus({ success: true, message: data.message });
			loadStats();
		} catch (err) {
			const message = err.response?.data?.detail || 'Failed to import email file.';
			setImportStatus({ success: false, message });
		} finally {
			setImporting(false);
			event.target.value = '';
		}
	};

	if (user?.is_demo) return <div className="settings-shell"><div className="settings-card"><button className="settings-back" onClick={() => navigate('/')}>← Back to Dashboard</button><h1>Sample mailbox</h1><p>These are fictional emails for the public demo. Gmail sync and account changes are disabled.</p><button className="settings-btn logout-btn" onClick={handleLogout}><LogOut size={16} /> Leave demo</button></div></div>;

	return <div className="settings-shell">
		<div className="settings-card">
			<button className="settings-back" onClick={() => navigate('/')}>← Back to Dashboard</button>
			<h1>Account Settings</h1>

			<section className="settings-section">
				<h2>Profile</h2>
				<div className="settings-profile">
					<div className="settings-avatar">
						{user?.picture ? <img src={user.picture} alt="" /> : <div className="avatar-placeholder">{user?.name?.[0] || '?'}</div>}
					</div>
					<div>
						<strong>{user?.name || 'User'}</strong>
						<span>{user?.email}</span>
					</div>
				</div>
			</section>

			<section className="settings-section">
				<h2>Mail Embeddings</h2>
				<p className="settings-desc">
					Embeddings make your emails searchable by the AI Assistant. Embedding progress is tracked for your mailbox.
				</p>
				{ragStatus?.postgresql ? (
					<div className="export-stats-box embedding-stats-box">
						<div className="export-stat-item">
							<strong>{ragStatus.postgresql.indexed_emails?.toLocaleString() || 0}</strong>
							<span>Emails Embedded</span>
						</div>
						<div className="export-stat-item">
							<strong>{ragStatus.postgresql.pending_emails?.toLocaleString() || 0}</strong>
							<span>Emails Remaining</span>
						</div>
						<div className="export-stat-item">
							<strong>{ragStatus.postgresql.indexed_embeddings?.toLocaleString() || 0}</strong>
							<span>Embeddings Created</span>
						</div>
					</div>
				) : (
					<p className="settings-desc">Loading embedding status...</p>
				)}
				{ragStatus?.postgresql?.pending_emails > 0 && (
					<button className="settings-btn embedding-btn" onClick={handleStartEmbeddings} disabled={embedding}>
						<Sparkles size={16} /> {embedding ? 'Embedding emails...' : 'Start Embeddings'}
					</button>
				)}
				{embeddingStatus && <div className="import-result-msg">{embeddingStatus}</div>}
			</section>

			<section className="settings-section">
				<h2>Export Mail Data</h2>
				<p className="settings-desc">
					Export emails directly from your local database as CSV or JSON. This is fast and runs completely offline from the database without querying the Gmail API.
				</p>

				{exportStats && (
					<div className="export-stats-box">
						<div className="export-stat-item">
							<strong>{exportStats.total_emails?.toLocaleString() || 0}</strong>
							<span>Stored Emails</span>
						</div>
						<div className="export-stat-item">
							<strong>{exportStats.labels_count || 0}</strong>
							<span>Labels</span>
						</div>
						{exportStats.newest_email && (
							<div className="export-stat-item">
								<strong>{new Date(exportStats.newest_email).toLocaleDateString()}</strong>
								<span>Latest Email</span>
							</div>
						)}
					</div>
				)}

				<div className="export-options">
					<label className="export-checkbox-label">
						<input
							type="checkbox"
							checked={includeBody}
							onChange={(e) => setIncludeBody(e.target.checked)}
						/>
						Include full plain-text message bodies (larger file size)
					</label>
				</div>

				<div className="export-actions">
					<button
						className="settings-btn export-btn"
						onClick={handleExportCSV}
						disabled={!exportStats || exportStats.total_emails === 0}
					>
						<Download size={16} /> Export as CSV
					</button>
					<button
						className="settings-btn export-btn"
						onClick={handleExportJSON}
						disabled={!exportStats || exportStats.total_emails === 0}
					>
						<FileText size={16} /> Export as JSON
					</button>
				</div>
			</section>

			<section className="settings-section">
				<h2>Import Mail Data (Parse & Store)</h2>
				<p className="settings-desc">
					Bring your own Gmail JSON or CSV files. The file will be parsed and stored directly into the database without needing to download emails via the Gmail API.
				</p>

				<div className="import-box">
					<label className="import-file-label">
						<Upload size={18} />
						<span>{importing ? 'Parsing and storing emails...' : 'Choose JSON or CSV file to import'}</span>
						<input
							type="file"
							accept=".json,.csv,application/json,text/csv"
							onChange={handleFileUpload}
							disabled={importing}
							style={{ display: 'none' }}
						/>
					</label>
					<p className="import-hint">Supported formats: Gmail export JSON, emails CSV with headers</p>
				</div>

				{importStatus && (
					<div className={`import-result-msg ${importStatus.success ? 'success' : 'error'}`}>
						{importStatus.message}
					</div>
				)}
			</section>

			<section className="settings-section">
				<h2>Session</h2>
				<p className="settings-desc">Log out of your account. Your synced data will be preserved.</p>
				<button className="settings-btn logout-btn" onClick={handleLogout}>
					<LogOut size={16} /> Log out
				</button>
			</section>

			<section className="settings-section settings-danger">
				<h2>Danger Zone</h2>
				<p className="settings-desc">
					Permanently delete your synced emails and all related stored mail data.
					Your profile and labels are preserved. This does not affect your actual Gmail account.
				</p>
				<button type="button" className="settings-btn erase-btn" onClick={() => setEraseOpen(true)}>
					<UserX size={16} /> Erase All Data
				</button>
			</section>
		</div>

		<EraseDialog open={eraseOpen} onClose={() => setEraseOpen(false)} onConfirm={handleErase} confirming={erasing} />
		{eraseToast && <div className="toast"><Zap size={16} /> {eraseToast}</div>}
	</div>;
}

function Dashboard() {
	const [active, setActive] = useState('Dashboard');
	const [search, setSearch] = useState('');
	const [selectedId, setSelectedId] = useState(null);
	const [selectedIds, setSelectedIds] = useState([]);
	const [mobileOpen, setMobileOpen] = useState(false);
	const [toast, setToast] = useState('');
	const [range, setRange] = useState('this_week');
	const [analytics, setAnalytics] = useState(null);
	const [analyticsError, setAnalyticsError] = useState('');
	const [activityRange, setActivityRange] = useState('this_week');
	const [charts, setCharts] = useState(null);
	const [recentEmails, setRecentEmails] = useState([]);
	const [inboxEmails, setInboxEmails] = useState([]);
	const [unreadCount, setUnreadCount] = useState(0);
	const [inboxPage, setInboxPage] = useState(1);
	const [inboxTotal, setInboxTotal] = useState(0);
	const [loadingMoreInbox, setLoadingMoreInbox] = useState(false);
	const [previewEmail, setPreviewEmail] = useState(null);
	const [emailsLoading, setEmailsLoading] = useState(true);
	const [expandedPreview, setExpandedPreview] = useState(false);
	const [isRefreshing, setIsRefreshing] = useState(false);
	const [isSyncing, setIsSyncing] = useState(false);

	const navigate = useNavigate();
	const { user, logout } = useAuth();

	const loadDashboardData = async (targetRange = range, targetActivityRange = activityRange) => {
		setEmailsLoading(true);
		setAnalyticsError('');
		try {
			const [summaryRes, chartsRes, emailsRes] = await Promise.allSettled([
				fetchAnalyticsSummary(targetRange),
				fetchAnalyticsCharts(targetActivityRange),
				fetchEmails({ per_page: 50, page: 1, range: targetRange }),
			]);

			if (summaryRes.status === 'fulfilled') {
				setAnalytics(summaryRes.value.data);
			} else {
				setAnalyticsError('Unable to load mailbox metrics');
			}

			if (chartsRes.status === 'fulfilled') {
				setCharts(chartsRes.value.data);
			}

			if (emailsRes.status === 'fulfilled') {
				const data = emailsRes.value.data;
				const mappedEmails = (data.emails || []).map((email) => {
					const sender = email.from_email || 'Unknown sender';
					const initials = sender.split(/[ @]/).filter(Boolean).slice(0, 2).map((part) => part[0]).join('').toUpperCase();
					return {
						...email,
						sender,
						email: sender,
						initials: initials || '?',
						avatarColor: '#e1dcff',
						preview: email.snippet || '',
						label: email.ai_label || email.labels?.find((l) => l.is_ai)?.name || email.labels?.find((l) => !['INBOX', 'UNREAD', 'STARRED', 'IMPORTANT'].includes(l.name))?.name || email.labels?.[0]?.name || 'Uncategorized',
						date: email.received_at ? new Date(email.received_at).toLocaleDateString() : '',
						starred: false,
					};
				});
				setInboxEmails(mappedEmails);
				setRecentEmails(mappedEmails.slice(0, 5));
				setInboxPage(1);
				setInboxTotal(data.total || 0);
				setUnreadCount(mappedEmails.filter((email) => !email.is_read).length);
			}
		} finally {
			setEmailsLoading(false);
		}
	};

	useEffect(() => {
		loadDashboardData(range, activityRange);
	}, [range, activityRange, user]);

	const handleRangeChange = (newRange) => {
		setRange(newRange);
		setActivityRange(newRange);
	};

	const handleActivityRangeChange = (newRange) => {
		setActivityRange(newRange);
	};

	const handleRefresh = async () => {
		if (isRefreshing || isSyncing) return;
		setIsRefreshing(true);
		setToast('Refreshing dashboard...');
		try {
			await loadDashboardData(range, activityRange);
			setToast('Dashboard is up to date');
		} catch {
			setToast('Could not refresh the dashboard. Please try again.');
		} finally {
			setIsRefreshing(false);
			setTimeout(() => setToast(''), 2500);
		}
	};

	const handleSync = async () => {
		if (isSyncing || isRefreshing) return;
		setIsSyncing(true);
		setToast('Syncing Gmail...');
		try {
			await startSync();
			let attempts = 0;
			const pollInterval = setInterval(async () => {
				attempts++;
				try {
					const { data } = await getSyncStatus();
					if (data.status === 'completed') {
						clearInterval(pollInterval);
						setIsSyncing(false);
						setToast('Gmail synced. Refreshing dashboard...');
						await loadDashboardData(range, activityRange);
						setToast('Dashboard is up to date');
						setTimeout(() => setToast(''), 2500);
					} else if (data.status === 'error') {
						clearInterval(pollInterval);
						setIsSyncing(false);
						setToast(data.error || 'Sync encountered an issue');
						setTimeout(() => setToast(''), 3500);
					} else if (attempts >= 100) {
						clearInterval(pollInterval);
						setIsSyncing(false);
						setToast('Gmail sync is still running. Try again shortly.');
						setTimeout(() => setToast(''), 4000);
					}
				} catch {
					if (attempts >= 100) {
						clearInterval(pollInterval);
						setIsSyncing(false);
						setToast('Could not confirm sync status. Please try again.');
						setTimeout(() => setToast(''), 3500);
					}
				}
			}, 1500);
		} catch (err) {
			setIsSyncing(false);
			const msg = err.response?.data?.detail || 'Unable to start Gmail sync';
			setToast(msg);
			setTimeout(() => setToast(''), 3500);
		}
	};

	useEffect(() => {
		if (!user || user.is_demo) return undefined;

		// Keep the mailbox current while the dashboard is open. The API ignores
		// duplicate requests when a sync is already underway.
		const autoSyncTimer = window.setInterval(() => {
			startSync().catch(() => {});
		}, 40000);

		return () => window.clearInterval(autoSyncTimer);
	}, [user]);

	const loadMoreInbox = async () => {
		if (loadingMoreInbox || inboxEmails.length >= inboxTotal) return;
		setLoadingMoreInbox(true);
		try {
			const nextPage = inboxPage + 1;
			const { data } = await fetchEmails({ per_page: 50, page: nextPage, range });
			const mappedEmails = (data.emails || []).map((email) => {
				const sender = email.from_email || 'Unknown sender';
				const initials = sender.split(/[ @]/).filter(Boolean).slice(0, 2).map((part) => part[0]).join('').toUpperCase();
				return {
					...email,
					sender,
					email: sender,
					initials: initials || '?',
					avatarColor: '#e1dcff',
					preview: email.snippet || '',
					label: email.ai_label || email.labels?.find((l) => l.is_ai)?.name || email.labels?.find((l) => !['INBOX', 'UNREAD', 'STARRED', 'IMPORTANT'].includes(l.name))?.name || email.labels?.[0]?.name || 'Uncategorized',
					date: email.received_at ? new Date(email.received_at).toLocaleDateString() : '',
					starred: false,
				};
			});
			setInboxEmails((items) => [...items, ...mappedEmails]);
			setInboxPage(nextPage);
		} finally {
			setLoadingMoreInbox(false);
		}
	};

	const handleEmailClick = (id) => {
		const email =
			(recentEmails || []).find((item) => item.id === id) ||
			(inboxEmails || []).find((item) => item.id === id) ||
			(filteredEmails || []).find((item) => item.id === id);
		if (email) {
			setPreviewEmail(email);
		}
		setSelectedId(id);
		setExpandedPreview(true);
	};

	useEffect(() => {
		if (!selectedId) {
			return undefined;
		}
		let cancelled = false;
		fetchEmail(selectedId).then(({ data }) => {
			if (cancelled) return;
			const sender = data.from_email || 'Unknown sender';
			const initials = sender.split(/[ @]/).filter(Boolean).slice(0, 2).map((part) => part[0]).join('').toUpperCase();
			setPreviewEmail((prev) => ({
				...(prev || {}),
				...data,
				sender,
				email: sender,
				initials: initials || '?',
				avatarColor: '#e1dcff',
				preview: data.snippet || prev?.preview || '',
				label: data.ai_label || data.labels?.find((l) => l.is_ai)?.name || data.labels?.[0]?.name || prev?.label || 'Uncategorized',
				date: data.received_at ? new Date(data.received_at).toLocaleDateString() : prev?.date || '',
			}));
		}).catch(() => {
			// keep currently set preview data
		});
		return () => { cancelled = true; };
	}, [selectedId]);

	useEffect(() => {
		const handleKeyDown = (e) => {
			if (e.key === 'Escape') {
				setExpandedPreview(false);
			}
		};
		window.addEventListener('keydown', handleKeyDown);
		return () => window.removeEventListener('keydown', handleKeyDown);
	}, []);

	const filteredEmails = useMemo(() => recentEmails.filter((email) => [email.sender, email.subject, email.preview, email.label].join(' ').toLowerCase().includes(search.toLowerCase())), [recentEmails, search]);
	const currentDate = new Intl.DateTimeFormat(undefined, { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' }).format(new Date()).toUpperCase();
	const dashboardStats = analytics ? buildStats(analytics, range) : [];
	const toggleSelect = (id) => setSelectedIds((items) => items.includes(id) ? items.filter((item) => item !== id) : [...items, id]);

	const action = (label) => {
		if (label === 'Sync Gmail') {
			handleSync();
			return;
		}
		if (label === 'Train Model') {
			setToast('Running AI classification on emails...');
			trainModel()
				.then(() => {
					setToast('AI classification running in background!');
					setTimeout(() => loadDashboardData(range, activityRange), 2000);
				})
				.catch(() => {
					setToast('AI classification started!');
				});
			setTimeout(() => setToast(''), 3000);
			return;
		}
		if (label === 'View Analytics') {
			setActive('Dashboard');
			return;
		}
		setToast(`${label} started`);
		setTimeout(() => setToast(''), 2200);
	};

	const handleLogout = async () => {
		await logout();
		navigate('/login', { replace: true });
	};

	if (active !== 'Dashboard') {
		return (
			<div className="app-shell">
				<div className="sidebar-wrap">
					<Sidebar active={active} onNavigate={setActive} onClose={() => {}} unreadCount={unreadCount} />
				</div>
				<div className="main-shell">
					<Header search={search} setSearch={setSearch} onMenu={() => {}} onLogout={handleLogout} user={user} onNavigateSettings={() => navigate('/settings')} />
					<main className="content">
						<InboxView
							emails={inboxEmails}
							selectedId={selectedId}
							onSelectEmail={handleEmailClick}
							selectedIds={selectedIds}
							onToggleSelect={toggleSelect}
							unreadCount={unreadCount}
							loading={emailsLoading}
							loadingMore={loadingMoreInbox}
							hasMore={inboxEmails.length < inboxTotal}
							onLoadMore={loadMoreInbox}
						/>
					</main>
				</div>
				{expandedPreview && previewEmail && (
					<div className="modal-overlay" onClick={() => setExpandedPreview(false)}>
						<div className="expanded-preview-modal" onClick={(event) => event.stopPropagation()}>
							<EmailPreview isModal email={previewEmail} onClose={() => setExpandedPreview(false)} onExpand={() => setExpandedPreview(false)} />
						</div>
					</div>
				)}
			</div>
		);
	}

	return (
		<div className="app-shell">
			<div className={`sidebar-overlay ${mobileOpen ? 'show' : ''}`} onClick={() => setMobileOpen(false)} />
			<div className={mobileOpen ? 'sidebar-wrap open' : 'sidebar-wrap'}>
				<Sidebar active={active} onNavigate={(label) => { setActive(label); setMobileOpen(false); }} onClose={() => setMobileOpen(false)} unreadCount={unreadCount} />
			</div>
			<div className="main-shell">
				<Header search={search} setSearch={setSearch} onMenu={() => setMobileOpen(true)} onLogout={handleLogout} user={user} onNavigateSettings={() => navigate('/settings')} />
				<main className="content">
					<section className="welcome">
						<div>
							<span className="eyebrow">{currentDate}</span>
							<h1>Good morning, {user?.name?.split(' ')[0] || 'there'} <span>✦</span></h1>
							<p>{analyticsError || "Here's what's happening with your inbox today."}</p>
							<p>{user?.is_demo ? 'Demo mode: all emails are fictional sample data.' : (import.meta.env.PROD ? 'Showcase coverage: up to your 500 most recent Gmail messages.' : '')}</p>
						</div>
						<div className="welcome-actions">
							<label className="date-button">
								<BarChart3 size={16} />
								<select value={range} onChange={(event) => handleRangeChange(event.target.value)} aria-label="Analytics date range">
									{dashboardRanges.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
								</select>
								<ChevronDown size={14} />
							</label>
							<button
								className="refresh-button"
								onClick={handleRefresh}
								disabled={isRefreshing || isSyncing}
								aria-busy={isRefreshing}
							>
								<RefreshCw size={15} className={isRefreshing ? 'spin' : ''} />
								{isRefreshing ? 'Refreshing...' : 'Refresh'}
							</button>
			{!user?.is_demo && <button
								className="sync-button"
								onClick={handleSync}
								disabled={isSyncing || isRefreshing}
								aria-busy={isSyncing}
							>
								<Mail size={15} className={isSyncing ? 'spin' : ''} />
								{isSyncing ? 'Syncing...' : 'Sync'}
			</button>}
						</div>
					</section>
					<div className="stats-grid">
						{dashboardStats.length ? dashboardStats.map((stat) => <StatCard key={stat.label} stat={stat} />) : <div className="stats-loading">Loading mailbox metrics...</div>}
					</div>
					<Charts charts={charts} range={activityRange} onRangeChange={handleActivityRangeChange} />
					{!user?.is_demo && <QuickActions onAction={action} />}
					<div className="bottom-grid">
						<RecentEmails
							emailsToShow={filteredEmails}
							selectedId={selectedId}
							onSelectEmail={handleEmailClick}
							selected={previewEmail}
							title="Recent Emails"
							loading={emailsLoading}
							onOpenMailbox={() => setActive('Inbox')}
							onToggleStar={(id) => setRecentEmails((items) => items.map((item) => item.id === id ? { ...item, starred: !item.starred } : item))}
							selectedIds={selectedIds}
							onToggleSelect={toggleSelect}
						/>
						<EmailPreview email={previewEmail} onClose={() => { setSelectedId(null); setPreviewEmail(null); }} onExpand={() => setExpandedPreview(true)} />
					</div>
				</main>
			</div>
			{toast && <div className="toast"><Zap size={16} /> {toast}</div>}
			{expandedPreview && previewEmail && (
				<div className="modal-overlay" onClick={() => setExpandedPreview(false)}>
					<div className="expanded-preview-modal" onClick={(event) => event.stopPropagation()}>
						<EmailPreview isModal email={previewEmail} onClose={() => setExpandedPreview(false)} onExpand={() => setExpandedPreview(false)} />
					</div>
				</div>
			)}
		</div>
	);
}

function ProtectedRoute({ children }) {
	const { isAuthenticated, loading } = useAuth();
	if (loading) return <div className="auth-loading"><Loader size={24} /><p>Loading...</p></div>;
	if (!isAuthenticated) return <Navigate to="/login" replace />;
	return children;
}

function Loader(props) {
	return <svg xmlns="http://www.w3.org/2000/svg" width={props.size || 24} height={props.size || 24} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="spin"><path d="M21 12a9 9 0 1 1-6.219-8.56" /></svg>;
}

function AssistantRoute() {
	const { user } = useAuth();
	const navigate = useNavigate();
	if (user?.is_demo) return <div className="settings-shell"><div className="settings-card"><h1>AI Assistant preview</h1><p>The demo mailbox includes sample AI labels. Assistant search will be available after demo embeddings are added.</p><button className="settings-btn" onClick={() => navigate('/')}>Back to mailbox</button></div></div>;
	return <AiAssistant />;
}

export default function App() {
	const location = useLocation();

	if (location.pathname === '/login') return <AuthPage signup={false} />;
	if (location.pathname === '/signup') return <AuthPage signup={true} />;
	if (location.pathname === '/consent') return <GmailConsent />;
	if (location.pathname === '/sync') return <ProtectedRoute><SyncPage /></ProtectedRoute>;
	if (location.pathname === '/settings') return <ProtectedRoute><SettingsPage /></ProtectedRoute>;
	if (location.pathname === '/assistant') return <ProtectedRoute><AssistantRoute /></ProtectedRoute>;
	return <ProtectedRoute><Dashboard /></ProtectedRoute>;
}
