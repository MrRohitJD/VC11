import { useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate, Navigate } from 'react-router-dom';
import {
	AlertCircle, BarChart3, Bell, ChevronDown, CircleHelp, FileText, Inbox,
	LayoutDashboard, Mail, Menu, MoreHorizontal, Paperclip, PenLine, Search,
	Send, Settings, ShieldAlert, Star, Tag, Trash2, Users, X, Zap, Sparkles, LogOut, Maximize2,
	AlertTriangle, UserX,
} from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import DOMPurify from 'dompurify';
import { useAuth } from './context/AuthContext';
import GmailConsent from './pages/GmailConsent';
import SyncPage from './pages/SyncPage';
import { fetchAnalyticsCharts, fetchAnalyticsSummary, fetchEmail, fetchEmails } from './services/api';

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
		<div className="nav-group"><p className="nav-label">MAILBOX</p>{navItems.map(({ label, icon: Icon, count }) => <button key={label} className={`nav-item ${active === label ? 'active' : ''}`} onClick={() => onNavigate(label)}><Icon size={17} /><span>{label}</span>{label === 'Inbox' ? <small>{unreadCount}</small> : count ? <small>{count}</small> : null}</button>)}</div>
		<div className="nav-group nav-lower"><p className="nav-label">WORKSPACE</p><button className="nav-item"><Zap size={17} /><span>AI Tools</span><span className="new-pill">NEW</span></button><button className="nav-item" onClick={() => navigate('/settings')}><Settings size={17} /><span>Settings</span></button><button className="nav-item"><CircleHelp size={17} /><span>Help & Support</span></button></div>
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
	return <div className="stat-card"><div className={`stat-icon ${stat.icon}`}><Icon size={19} /></div><div className="stat-content"><span>{stat.label}</span><strong>{stat.value}</strong><small className={stat.positive ? 'positive' : 'negative'}>{stat.change} <em>vs last month</em></small></div><div className="stat-spark">{[10, 15, 11, 19, 14, 21, 19].map((height, index) => <i key={index} style={{ height }} />)}</div></div>;
}

const dashboardRanges = [
	['today', 'Today'], ['this_week', 'This week'], ['last_week', 'Last week'],
	['this_month', 'This month'], ['last_month', 'Last month'], ['six_months', '6 months'], ['one_year', '1 year'],
];

function formatChange(current, comparison) {
	if (comparison === 0) return current === 0 ? '0%' : '+100%';
	const change = ((current - comparison) / comparison) * 100;
	return `${change >= 0 ? '+' : ''}${change.toFixed(1)}%`;
}

function buildStats(summary) {
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
	}));
}

const activityRanges = [
	['last_7_days', 'Last 7 days'], ['today', 'Today'], ['this_month', 'This month'],
	['last_month', 'Last month'], ['six_months', '6 months'], ['one_year', '1 year'], ['all_time', 'All time'],
];
const chartColors = ['#6c63ff', '#57c7a5', '#f5b74f', '#ff8176', '#7b9ff2', '#a78bfa'];

function CategoryTooltip({ active, payload }) {
	if (!active || !payload?.length) return null;
	const category = payload[0].payload;
	return <div className="category-tooltip"><strong>Category: {category.name}</strong><span>{category.count.toLocaleString()} mails</span>{category.breakdown?.length > 0 && <div className="category-breakdown"><b>Other categories</b>{category.breakdown.map((item) => <span key={item.name}>{item.name}: {item.count.toLocaleString()} mails</span>)}</div>}</div>;
}

function Charts({ charts, range, onRangeChange }) {
	const categories = charts?.categories || [];
	const activity = charts?.activity || [];
	const total = charts?.total_emails || 0;
	return <div className="charts-grid"><section className="panel category-panel"><div className="panel-heading"><div><h2>Emails by Category</h2><p>Distribution of your inbox</p></div><button className="more-button"><MoreHorizontal size={18} /></button></div><div className="donut-wrap"><ResponsiveContainer width="100%" height={190}><PieChart><Pie data={categories} dataKey="value" nameKey="name" innerRadius={59} outerRadius={83} paddingAngle={4} stroke="none">{categories.map((item, index) => <Cell key={item.name} fill={chartColors[index % chartColors.length]} />)}</Pie><Tooltip content={<CategoryTooltip />} /></PieChart></ResponsiveContainer><div className="donut-total"><strong>{total.toLocaleString()}</strong><span>Total emails</span></div></div><div className="legend-grid">{categories.map((item, index) => <div key={item.name}><i className={`legend-dot dot-${index}`} /><span>{item.name}</span><strong>{item.value}%</strong></div>)}</div></section><section className="panel activity-panel"><div className="panel-heading"><div><h2>Email Activity</h2><p>Incoming and outgoing emails</p></div><label className="range-button"><select value={range} onChange={(event) => onRangeChange(event.target.value)} aria-label="Activity date range">{activityRanges.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select><ChevronDown size={14} /></label></div><div className="chart-key"><span><i className="line-incoming" /> Incoming</span><span><i className="line-outgoing" /> Outgoing</span></div><ResponsiveContainer width="100%" height={234}><LineChart data={activity} margin={{ top: 15, right: 10, left: -23, bottom: 0 }}><CartesianGrid vertical={false} stroke="#edf0f5" /><XAxis dataKey="label" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: '#9aa4b5' }} /><YAxis axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: '#9aa4b5' }} allowDecimals={false} /><Tooltip contentStyle={{ border: '0', borderRadius: 8, boxShadow: '0 8px 22px #1d244015', fontSize: 12 }} /><Line type="monotone" dataKey="incoming" stroke="#6c63ff" strokeWidth={2.5} dot={false} /><Line type="monotone" dataKey="outgoing" stroke="#53c7a0" strokeWidth={2.5} dot={false} /></LineChart></ResponsiveContainer></section></div>;
}

function QuickActions({ onAction }) {
	const actions = [{ label: 'Sync Gmail', detail: 'Fetch latest messages', icon: Mail, color: 'purple' }, { label: 'Train Model', detail: 'Improve AI accuracy', icon: Zap, color: 'mint' }, { label: 'Manage Labels', detail: 'Organize your inbox', icon: Tag, color: 'gold' }, { label: 'View Analytics', detail: 'See detailed insights', icon: BarChart3, color: 'coral' }];
	return <section className="quick-section"><div className="section-heading"><div><h2>Quick Actions</h2><p>Manage your inbox efficiently</p></div></div><div className="quick-grid">{actions.map(({ label, detail, icon: Icon, color }) => <button className="quick-action" key={label} onClick={() => onAction(label)}><div className={`quick-icon ${color}`}><Icon size={18} /></div><div><strong>{label}</strong><span>{detail}</span></div><ChevronDown className="action-arrow" size={16} /></button>)}</div></section>;
			<RecentEmails emailsToShow={filteredEmails} selectedId={selectedId} setSelectedId={setSelectedId} selected={previewEmail} title={active === 'Inbox' ? 'Inbox' : 'Recent Emails'} loading={emailsLoading} onOpenMailbox={openMailboxWindow} onToggleStar={(id) => setRecentEmails((items) => items.map((item) => item.id === id ? { ...item, starred: !item.starred } : item))} selectedIds={selectedIds} onToggleSelect={toggleSelect} />

function RecentEmails({ emailsToShow, selectedId, setSelectedId, selected, onToggleStar, selectedIds, onToggleSelect, onOpenMailbox, title = 'Recent Emails', loading = false }) {
	return <section className="panel recent-panel"><div className="panel-heading"><div><h2>{title}</h2><p>{title === 'Inbox' ? 'All messages in your Gmail inbox' : 'Your latest messages and AI classifications'}</p></div><button className="view-all" onClick={onOpenMailbox}>{title === 'Inbox' ? `${emailsToShow.length} mails` : 'View all'} <span>→</span></button></div><div className="email-table">{loading ? <div className="mailbox-loading">Loading inbox...</div> : <><div className="table-head"><span className="check-cell"><input type="checkbox" checked={selectedIds.length === emailsToShow.length && emailsToShow.length > 0} onChange={() => emailsToShow.forEach((email) => onToggleSelect(email.id))} /></span><span>Sender</span><span>Subject</span><span>Label</span><span>Date</span><span /></div>{emailsToShow.map((email) => <button className={`email-row ${email.is_read ? '' : 'unread-row'} ${selectedId === email.id ? 'selected' : ''}`} key={email.id} onClick={() => { setSelectedId(email.id); onOpenMailbox?.(email.id); }}><span className="check-cell" onClick={(event) => event.stopPropagation()}><input type="checkbox" checked={selectedIds.includes(email.id)} onChange={() => onToggleSelect(email.id)} /></span><span className="sender-cell"><span className="sender-avatar" style={{ background: email.avatarColor }}>{email.initials}</span><strong>{email.sender}</strong>{email.hasAttachment && <Paperclip size={13} />}</span><span className="subject-cell"><strong>{email.subject}</strong><small>{email.preview}</small></span><span><em className={`label-badge ${email.label.toLowerCase()}`}>{email.label}</em></span><span className="date-cell">{email.date}</span><span className={`star ${email.starred ? 'starred' : ''}`} onClick={(event) => { event.stopPropagation(); onToggleStar(email.id); }}><Star size={17} fill={email.starred ? 'currentColor' : 'none'} /></span></button>)}</>}</div></section>;
}

	function InboxView({ emails, selectedId, setSelectedId, selectedIds, onToggleSelect, unreadCount, loading, loadingMore, hasMore, onLoadMore }) {
	return <section className="panel inbox-view"><div className="panel-heading"><div><h2>Inbox</h2><p>{emails.length.toLocaleString()} messages loaded, {unreadCount.toLocaleString()} unread</p></div><button className="view-all">{emails.length.toLocaleString()} mails</button></div><div className="email-table inbox-scroll" onScroll={(event) => { const element = event.currentTarget; if (element.scrollHeight - element.scrollTop - element.clientHeight < 160) onLoadMore(); }}>{loading ? <div className="mailbox-loading">Loading inbox...</div> : <><div className="table-head"><span className="check-cell"><input type="checkbox" checked={selectedIds.length === emails.length && emails.length > 0} onChange={() => emails.forEach((email) => onToggleSelect(email.id))} /></span><span>Sender</span><span>Subject</span><span>Label</span><span>Date</span><span /></div>{emails.map((email) => <button className={`email-row ${email.is_read ? '' : 'unread-row'} ${selectedId === email.id ? 'selected' : ''}`} key={email.id} onClick={() => setSelectedId(email.id)}><span className="check-cell" onClick={(event) => event.stopPropagation()}><input type="checkbox" checked={selectedIds.includes(email.id)} onChange={() => onToggleSelect(email.id)} /></span><span className="sender-cell"><span className="sender-avatar" style={{ background: email.avatarColor }}>{email.initials}</span><strong>{email.sender}</strong></span><span className="subject-cell"><strong>{email.subject || '(No subject)'}</strong><small>{email.preview}</small></span><span><em className={`label-badge ${email.label.toLowerCase()}`}>{email.label}</em></span><span className="date-cell">{email.date}</span><span className="star"><Star size={17} /></span></button>)}{loadingMore && <div className="mailbox-loading">Loading more messages...</div>}{!hasMore && emails.length > 0 && <div className="mailbox-end">All inbox messages loaded</div>}</>}</div></section>;
}

function EmailPreview({ email, onClose, onExpand }) {
	if (!email) return <section className="panel preview-panel empty-preview"><Mail size={24} /><p>Select an email to preview it</p></section>;
	const sanitizedHtml = email.body_html ? DOMPurify.sanitize(email.body_html, { USE_PROFILES: { html: true } }) : '';
	return <section className="panel preview-panel"><div className="preview-header"><div><span className="preview-label">EMAIL PREVIEW</span><h2>{email.subject || '(No subject)'}</h2></div><div className="preview-header-actions"><button className="more-button" onClick={onExpand} aria-label="Open email preview"><Maximize2 size={17} /></button><button className="more-button" onClick={onClose} aria-label="Close email preview"><X size={18} /></button></div></div><div className="preview-sender"><span className="sender-avatar" style={{ background: email.avatarColor }}>{email.initials}</span><div><strong>{email.sender}</strong><span>{email.email}</span><small>To: {email.to_email || 'you'}</small></div><span className="preview-date">{email.date}</span></div><div className="preview-body">{sanitizedHtml ? <div className="email-html-body" dangerouslySetInnerHTML={{ __html: sanitizedHtml }} /> : email.body ? email.body.split('\n').filter(Boolean).map((paragraph, index) => <p key={index}>{paragraph}</p>) : <p>No message body available.</p>}</div><div className="preview-footer"><button><ReplyIcon /> Reply</button><button><ForwardIcon /> Forward</button></div></section>;
}

function MailboxWindow({ emails, selectedId, setSelectedId, selectedIds, onToggleSelect, unreadCount, loading, loadingMore, hasMore, onLoadMore, previewEmail, onClose }) {
	return <div className="mailbox-window-content"><InboxView emails={emails} selectedId={selectedId} setSelectedId={setSelectedId} selectedIds={selectedIds} onToggleSelect={onToggleSelect} unreadCount={unreadCount} loading={loading} loadingMore={loadingMore} hasMore={hasMore} onLoadMore={onLoadMore} /><EmailPreview email={previewEmail} onClose={onClose} onExpand={onClose} /></div>;
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
	const submit = (event) => { event.preventDefault(); navigate('/consent'); };
	return <div className={`auth-shell ${signup ? 'auth-signup' : 'auth-login'}`}><main className="auth-card"><div className="auth-brand"><GmailMark /><strong>Gmail AI Dashboard</strong></div><div className="auth-copy"><h1>{signup ? 'Create Your Account' : 'Welcome Back'}</h1><p>{signup ? 'Get started with your Gmail AI Dashboard' : 'Sign in to access your Gmail AI Dashboard'}</p></div><form className="auth-form" onSubmit={submit}><button className="google-button" type="submit"><span className="google-g">G</span>{signup ? 'Sign up with Google' : 'Continue with Google'}</button><div className="auth-divider"><span /> <em>OR</em> <span /></div><p className="auth-switch">{signup ? 'Already have an account?' : "Don't have an account?"} <button type="button" onClick={() => navigate(signup ? '/login' : '/signup')}>{signup ? 'Sign in' : 'Sign up'}</button></p></form><AuthIllustration signup={signup} /><p className="auth-tagline">{signup ? 'Join and make your inbox work smarter.' : 'Your Gmail. Smarter with AI.'}</p></main></div>;
}

function EraseDialog({ open, onClose, onConfirm }) {
	if (!open) return null;
	return <div className="modal-overlay" onClick={onClose}>
		<div className="modal-card erase-modal" onClick={(e) => e.stopPropagation()}>
			<div className="erase-icon-wrap">
				<AlertTriangle size={28} />
			</div>
			<h2>Erase all data?</h2>
			<p>
				This will permanently delete all data associated with your account from our database,
				including synchronized emails, labels, profile data, and stored OAuth tokens.
			</p>
			<p className="erase-warning">This action cannot be undone.</p>
			<div className="erase-actions">
				<button className="erase-cancel" onClick={onClose}>Cancel</button>
				<button className="erase-confirm" onClick={onConfirm}>
					<UserX size={15} /> Erase All Data
				</button>
			</div>
		</div>
	</div>;
}

function SettingsPage() {
	const { user, logout, eraseAllData } = useAuth();
	const navigate = useNavigate();
	const [eraseOpen, setEraseOpen] = useState(false);
	const [eraseToast, setEraseToast] = useState('');

	const handleLogout = async () => {
		await logout();
		navigate('/login', { replace: true });
	};

	const handleErase = async () => {
		try {
			await eraseAllData();
			setEraseOpen(false);
			setEraseToast('All your data has been erased.');
			setTimeout(() => navigate('/login', { replace: true }), 1500);
		} catch {
			setEraseToast('Failed to erase data. Please try again.');
		}
	};

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
				<h2>Session</h2>
				<p className="settings-desc">Log out of your account. Your synced data will be preserved.</p>
				<button className="settings-btn logout-btn" onClick={handleLogout}>
					<LogOut size={16} /> Log out
				</button>
			</section>

			<section className="settings-section settings-danger">
				<h2>Danger Zone</h2>
				<p className="settings-desc">
					Permanently delete all your data from this application, including synced emails,
					labels, tokens, and your user profile. This does not affect your actual Gmail account.
				</p>
				<button className="settings-btn erase-btn" onClick={() => setEraseOpen(true)}>
					<UserX size={16} /> Erase All Data
				</button>
			</section>
		</div>

		<EraseDialog open={eraseOpen} onClose={() => setEraseOpen(false)} onConfirm={handleErase} />
		{eraseToast && <div className="toast"><Zap size={16} /> {eraseToast}</div>}
	</div>;
}

function Dashboard() {
	const [active, setActive] = useState('Dashboard'); const [search, setSearch] = useState(''); const [selectedId, setSelectedId] = useState(null); const [selectedIds, setSelectedIds] = useState([]); const [mobileOpen, setMobileOpen] = useState(false); const [toast, setToast] = useState(''); const [range, setRange] = useState('this_week'); const [analytics, setAnalytics] = useState(null); const [analyticsError, setAnalyticsError] = useState(''); const [activityRange, setActivityRange] = useState('last_7_days'); const [charts, setCharts] = useState(null); const [recentEmails, setRecentEmails] = useState([]); const [inboxEmails, setInboxEmails] = useState([]); const [unreadCount, setUnreadCount] = useState(0); const [inboxPage, setInboxPage] = useState(1); const [inboxTotal, setInboxTotal] = useState(0); const [loadingMoreInbox, setLoadingMoreInbox] = useState(false); const [previewEmail, setPreviewEmail] = useState(null); const [emailsLoading, setEmailsLoading] = useState(true); const [expandedPreview, setExpandedPreview] = useState(false);
	const [mailboxWindowOpen, setMailboxWindowOpen] = useState(false);
	const navigate = useNavigate();
	const { user, logout } = useAuth();
	useEffect(() => {
		let cancelled = false;
		setAnalyticsError('');
		fetchAnalyticsSummary(range).then(({ data }) => {
			if (!cancelled) setAnalytics(data);
		}).catch(() => {
			if (!cancelled) setAnalyticsError('Unable to load mailbox metrics');
		});
		return () => { cancelled = true; };
	}, [range, user]);
	useEffect(() => {
		let cancelled = false;
		fetchAnalyticsCharts(activityRange).then(({ data }) => {
			if (!cancelled) setCharts(data);
		}).catch(() => {
			if (!cancelled) setCharts(null);
		});
		return () => { cancelled = true; };
	}, [activityRange, user]);
	useEffect(() => {
		let cancelled = false;
		setEmailsLoading(true);
		fetchEmails({ per_page: 50, page: 1, inbox_only: true }).then(({ data }) => {
			if (cancelled) return;
			const mappedEmails = data.emails.map((email) => {
				const sender = email.from_email || 'Unknown sender';
				const initials = sender.split(/[ @]/).filter(Boolean).slice(0, 2).map((part) => part[0]).join('').toUpperCase();
				return {
					...email,
					sender,
					email: sender,
					initials: initials || '?',
					avatarColor: '#e1dcff',
					preview: email.snippet || '',
					label: email.labels?.[0]?.name || 'Uncategorized',
					date: email.received_at ? new Date(email.received_at).toLocaleDateString() : '',
					starred: false,
				};
			});
			setInboxEmails(mappedEmails);
			setRecentEmails(mappedEmails.slice(0, 5));
			setInboxPage(1);
			setInboxTotal(data.total);
			setUnreadCount(mappedEmails.filter((email) => !email.is_read).length);
		}).catch(() => {
			if (!cancelled) setRecentEmails([]);
		}).finally(() => {
			if (!cancelled) setEmailsLoading(false);
		});
		return () => { cancelled = true; };
	}, [user]);
	const loadMoreInbox = async () => {
		if (loadingMoreInbox || inboxEmails.length >= inboxTotal) return;
		setLoadingMoreInbox(true);
		try {
			const nextPage = inboxPage + 1;
			const { data } = await fetchEmails({ per_page: 50, page: nextPage, inbox_only: true });
			const mappedEmails = data.emails.map((email) => {
				const sender = email.from_email || 'Unknown sender';
				const initials = sender.split(/[ @]/).filter(Boolean).slice(0, 2).map((part) => part[0]).join('').toUpperCase();
				return { ...email, sender, email: sender, initials: initials || '?', avatarColor: '#e1dcff', preview: email.snippet || '', label: email.labels?.[0]?.name || 'Uncategorized', date: email.received_at ? new Date(email.received_at).toLocaleDateString() : '', starred: false };
			});
			setInboxEmails((items) => [...items, ...mappedEmails]);
			setInboxPage(nextPage);
		} finally {
			setLoadingMoreInbox(false);
		}
	};
	const openMailboxWindow = (emailId = selectedId) => { if (emailId) setSelectedId(emailId); setMailboxWindowOpen(true); setExpandedPreview(true); };
	useEffect(() => {
		if (selectedId === null) {
			setPreviewEmail(null);
			setExpandedPreview(false);
			return undefined;
		}
		let cancelled = false;
		fetchEmail(selectedId).then(({ data }) => {
			if (cancelled) return;
			const sender = data.from_email || 'Unknown sender';
			setPreviewEmail({ ...data, sender, email: sender, initials: sender[0]?.toUpperCase() || '?', avatarColor: '#e1dcff', date: data.received_at ? new Date(data.received_at).toLocaleDateString() : '' });
		}).catch(() => {
			if (!cancelled) setPreviewEmail(null);
		});
		return () => { cancelled = true; };
	}, [selectedId]);
	const filteredEmails = useMemo(() => recentEmails.filter((email) => [email.sender, email.subject, email.preview, email.label].join(' ').toLowerCase().includes(search.toLowerCase())), [recentEmails, search]);
	const currentDate = new Intl.DateTimeFormat(undefined, { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' }).format(new Date()).toUpperCase();
	const dashboardStats = analytics ? buildStats(analytics) : [];
	const toggleSelect = (id) => setSelectedIds((items) => items.includes(id) ? items.filter((item) => item !== id) : [...items, id]);
	const action = (label) => { setToast(`${label} started`); window.setTimeout(() => setToast(''), 2200); };
	const handleLogout = async () => { await logout(); navigate('/login', { replace: true }); };
	if (active === 'Inbox') {
		return <div className="app-shell"><div className="sidebar-wrap"><Sidebar active={active} onNavigate={setActive} onClose={() => {}} unreadCount={unreadCount} /></div><div className="main-shell"><Header search={search} setSearch={setSearch} onMenu={() => {}} onLogout={handleLogout} user={user} onNavigateSettings={() => navigate('/settings')} /><main className="content"><InboxView emails={inboxEmails} selectedId={selectedId} setSelectedId={setSelectedId} selectedIds={selectedIds} onToggleSelect={toggleSelect} unreadCount={unreadCount} loading={emailsLoading} loadingMore={loadingMoreInbox} hasMore={inboxEmails.length < inboxTotal} onLoadMore={loadMoreInbox} /><EmailPreview email={previewEmail} onClose={() => setSelectedId(null)} onExpand={() => setExpandedPreview(true)} /></main></div>{expandedPreview && previewEmail && <div className="modal-overlay" onClick={() => setExpandedPreview(false)}><div className="expanded-preview-modal" onClick={(event) => event.stopPropagation()}><EmailPreview email={previewEmail} onClose={() => setExpandedPreview(false)} onExpand={() => setExpandedPreview(false)} /></div></div>}</div>;
	}
	return <div className="app-shell"><div className={`sidebar-overlay ${mobileOpen ? 'show' : ''}`} onClick={() => setMobileOpen(false)} /><div className={mobileOpen ? 'sidebar-wrap open' : 'sidebar-wrap'}><Sidebar active={active} onNavigate={(label) => { setActive(label); setMobileOpen(false); }} onClose={() => setMobileOpen(false)} unreadCount={unreadCount} /></div><div className="main-shell"><Header search={search} setSearch={setSearch} onMenu={() => setMobileOpen(true)} onLogout={handleLogout} user={user} onNavigateSettings={() => navigate('/settings')} /><main className="content"><section className="welcome"><div><span className="eyebrow">{currentDate}</span><h1>Good morning, {user?.name?.split(' ')[0] || 'there'} <span>✦</span></h1><p>{analyticsError || "Here's what's happening with your inbox today."}</p></div><label className="date-button"><BarChart3 size={16} /><select value={range} onChange={(event) => setRange(event.target.value)} aria-label="Analytics date range">{dashboardRanges.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select><ChevronDown size={14} /></label></section><div className="stats-grid">{dashboardStats.length ? dashboardStats.map((stat) => <StatCard key={stat.label} stat={stat} />) : <div className="stats-loading">Loading mailbox metrics...</div>}</div><Charts charts={charts} range={activityRange} onRangeChange={setActivityRange} /><QuickActions onAction={action} /><div className="bottom-grid"><RecentEmails emailsToShow={filteredEmails} selectedId={selectedId} setSelectedId={setSelectedId} selected={previewEmail} title={active === 'Inbox' ? 'Inbox' : 'Recent Emails'} loading={emailsLoading} onToggleStar={(id) => setRecentEmails((items) => items.map((item) => item.id === id ? { ...item, starred: !item.starred } : item))} selectedIds={selectedIds} onToggleSelect={toggleSelect} /><EmailPreview email={previewEmail} onClose={() => setSelectedId(null)} onExpand={() => setExpandedPreview(true)} /></div></main></div>{toast && <div className="toast"><Zap size={16} /> {toast}</div>}{expandedPreview && previewEmail && <div className="modal-overlay" onClick={() => setExpandedPreview(false)}><div className="expanded-preview-modal" onClick={(event) => event.stopPropagation()}><EmailPreview email={previewEmail} onClose={() => setExpandedPreview(false)} onExpand={() => setExpandedPreview(false)} /></div></div>}</div>;
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

export default function App() {
	const location = useLocation();

	if (location.pathname === '/login') return <AuthPage signup={false} />;
	if (location.pathname === '/signup') return <AuthPage signup={true} />;
	if (location.pathname === '/consent') return <GmailConsent />;
	if (location.pathname === '/sync') return <ProtectedRoute><SyncPage /></ProtectedRoute>;
	if (location.pathname === '/settings') return <ProtectedRoute><SettingsPage /></ProtectedRoute>;
	return <ProtectedRoute><Dashboard /></ProtectedRoute>;
}
