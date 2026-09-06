import { useMemo, useState } from 'react';
import { NavLink } from 'react-router-dom';
import {
	AlertCircle, BarChart3, Bell, ChevronDown, CircleHelp, FileText, Inbox,
	LayoutDashboard, Mail, Menu, MoreHorizontal, Paperclip, PenLine, Search,
	Send, Settings, ShieldAlert, Star, Tag, Trash2, Users, X, Zap,
} from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import { emailActivity, categoryData, emails, stats } from './mockData';

const navItems = [
	{ label: 'Inbox', icon: Inbox, count: 12 }, { label: 'Starred', icon: Star },
	{ label: 'Sent', icon: Send }, { label: 'Important', icon: Tag },
	{ label: 'Drafts', icon: FileText, count: 3 }, { label: 'Spam', icon: ShieldAlert },
	{ label: 'Trash', icon: Trash2 },
];

function Sidebar({ active, onNavigate, onClose }) {
	return <aside className="sidebar">
		<div className="brand"><div className="brand-mark"><Mail size={18} /></div><span>Gmail AI <strong>Dashboard</strong></span><button className="mobile-close" onClick={onClose}><X size={18} /></button></div>
		<button className="compose-button"><PenLine size={17} /> Compose</button>
		<div className="nav-group"><p className="nav-label">MAILBOX</p>{navItems.map(({ label, icon: Icon, count }) => <button key={label} className={`nav-item ${active === label ? 'active' : ''}`} onClick={() => onNavigate(label)}><Icon size={17} /><span>{label}</span>{count && <small>{count}</small>}</button>)}</div>
		<div className="nav-group nav-lower"><p className="nav-label">WORKSPACE</p><button className="nav-item"><Zap size={17} /><span>AI Tools</span><span className="new-pill">NEW</span></button><button className="nav-item"><Settings size={17} /><span>Settings</span></button><button className="nav-item"><CircleHelp size={17} /><span>Help & Support</span></button></div>
		<div className="sidebar-footer"><div className="plan-icon"><BarChart3 size={17} /></div><div><strong>AI model status</strong><span><i /> Ready to classify</span></div><MoreHorizontal size={16} /></div>
	</aside>;
}

function Header({ search, setSearch, onMenu }) {
	return <header className="topbar"><button className="menu-button" onClick={onMenu}><Menu size={21} /></button><div className="search-box"><Search size={18} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search emails, people, or keywords..." /><kbd>⌘ K</kbd></div><div className="top-actions"><button className="icon-button notification"><Bell size={19} /><i /></button><div className="user-profile"><div className="avatar">JD</div><div className="user-copy"><strong>Jordan Davis</strong><span>Administrator</span></div><ChevronDown size={16} /></div></div></header>;
}

function StatCard({ stat }) {
	const icons = { inbox: Inbox, tag: Tag, important: Zap, unread: Mail };
	const Icon = icons[stat.icon];
	return <div className="stat-card"><div className={`stat-icon ${stat.icon}`}><Icon size={19} /></div><div className="stat-content"><span>{stat.label}</span><strong>{stat.value}</strong><small className={stat.positive ? 'positive' : ''}>{stat.change} <em>vs last month</em></small></div><div className="stat-spark">{[10, 15, 11, 19, 14, 21, 19].map((height, index) => <i key={index} style={{ height }} />)}</div></div>;
}

function Charts() {
	return <div className="charts-grid"><section className="panel category-panel"><div className="panel-heading"><div><h2>Emails by Category</h2><p>Distribution of your inbox</p></div><button className="more-button"><MoreHorizontal size={18} /></button></div><div className="donut-wrap"><ResponsiveContainer width="100%" height={190}><PieChart><Pie data={categoryData} dataKey="value" nameKey="name" innerRadius={59} outerRadius={83} paddingAngle={4} stroke="none"><Cell fill="#6c63ff" /><Cell fill="#57c7a5" /><Cell fill="#f5b74f" /><Cell fill="#ff8176" /></Pie></PieChart></ResponsiveContainer><div className="donut-total"><strong>1,248</strong><span>Total emails</span></div></div><div className="legend-grid">{categoryData.map((item, index) => <div key={item.name}><i className={`legend-dot dot-${index}`} /><span>{item.name}</span><strong>{item.value}%</strong></div>)}</div></section><section className="panel activity-panel"><div className="panel-heading"><div><h2>Email Activity</h2><p>Incoming and outgoing emails</p></div><button className="range-button">Last 7 days <ChevronDown size={14} /></button></div><div className="chart-key"><span><i className="line-incoming" /> Incoming</span><span><i className="line-outgoing" /> Outgoing</span></div><ResponsiveContainer width="100%" height={234}><LineChart data={emailActivity} margin={{ top: 15, right: 10, left: -23, bottom: 0 }}><CartesianGrid vertical={false} stroke="#edf0f5" /><XAxis dataKey="day" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: '#9aa4b5' }} /><YAxis axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: '#9aa4b5' }} ticks={[0, 20, 40, 60]} /><Tooltip contentStyle={{ border: '0', borderRadius: 8, boxShadow: '0 8px 22px #1d244015', fontSize: 12 }} /><Line type="monotone" dataKey="incoming" stroke="#6c63ff" strokeWidth={2.5} dot={false} /><Line type="monotone" dataKey="outgoing" stroke="#53c7a0" strokeWidth={2.5} dot={false} /></LineChart></ResponsiveContainer></section></div>;
}

function QuickActions({ onAction }) {
	const actions = [{ label: 'Sync Gmail', detail: 'Fetch latest messages', icon: Mail, color: 'purple' }, { label: 'Train Model', detail: 'Improve AI accuracy', icon: Zap, color: 'mint' }, { label: 'Manage Labels', detail: 'Organize your inbox', icon: Tag, color: 'gold' }, { label: 'View Analytics', detail: 'See detailed insights', icon: BarChart3, color: 'coral' }];
	return <section className="quick-section"><div className="section-heading"><div><h2>Quick Actions</h2><p>Manage your inbox efficiently</p></div></div><div className="quick-grid">{actions.map(({ label, detail, icon: Icon, color }) => <button className="quick-action" key={label} onClick={() => onAction(label)}><div className={`quick-icon ${color}`}><Icon size={18} /></div><div><strong>{label}</strong><span>{detail}</span></div><ChevronDown className="action-arrow" size={16} /></button>)}</div></section>;
}

function RecentEmails({ emailsToShow, selectedId, setSelectedId, selected, onToggleStar, selectedIds, onToggleSelect }) {
	return <section className="panel recent-panel"><div className="panel-heading"><div><h2>Recent Emails</h2><p>Your latest messages and AI classifications</p></div><button className="view-all">View all <span>→</span></button></div><div className="email-table"><div className="table-head"><span className="check-cell"><input type="checkbox" checked={selectedIds.length === emailsToShow.length && emailsToShow.length > 0} onChange={() => emailsToShow.forEach((email) => onToggleSelect(email.id))} /></span><span>Sender</span><span>Subject</span><span>Label</span><span>Date</span><span /></div>{emailsToShow.map((email) => <button className={`email-row ${selectedId === email.id ? 'selected' : ''}`} key={email.id} onClick={() => setSelectedId(email.id)}><span className="check-cell" onClick={(event) => event.stopPropagation()}><input type="checkbox" checked={selectedIds.includes(email.id)} onChange={() => onToggleSelect(email.id)} /></span><span className="sender-cell"><span className="sender-avatar" style={{ background: email.avatarColor }}>{email.initials}</span><strong>{email.sender}</strong>{email.hasAttachment && <Paperclip size={13} />}</span><span className="subject-cell"><strong>{email.subject}</strong><small>{email.preview}</small></span><span><em className={`label-badge ${email.label.toLowerCase()}`}>{email.label}</em></span><span className="date-cell">{email.date}</span><span className={`star ${email.starred ? 'starred' : ''}`} onClick={(event) => { event.stopPropagation(); onToggleStar(email.id); }}><Star size={17} fill={email.starred ? 'currentColor' : 'none'} /></span></button>)}</div></section>;
}

function EmailPreview({ email, onClose }) {
	if (!email) return <section className="panel preview-panel empty-preview"><Mail size={24} /><p>Select an email to preview it</p></section>;
	return <section className="panel preview-panel"><div className="preview-header"><div><span className="preview-label">EMAIL PREVIEW</span><h2>{email.subject}</h2></div><button className="more-button" onClick={onClose}><X size={18} /></button></div><div className="preview-sender"><span className="sender-avatar" style={{ background: email.avatarColor }}>{email.initials}</span><div><strong>{email.sender}</strong><span>{email.email}</span></div><span className="preview-date">{email.date}</span></div><div className="preview-body"><p>Hi Jordan,</p><p>{email.body}</p><p>Best,<br />{email.sender.split(' ')[0]}</p></div><div className="preview-footer"><button><ReplyIcon /> Reply</button><button><ForwardIcon /> Forward</button></div></section>;
}
function ReplyIcon() { return <Send size={14} />; }
function ForwardIcon() { return <Send size={14} className="forward-icon" />; }

export default function App() {
	const [active, setActive] = useState('Inbox'); const [search, setSearch] = useState(''); const [selectedId, setSelectedId] = useState(1); const [selectedIds, setSelectedIds] = useState([]); const [mobileOpen, setMobileOpen] = useState(false); const [toast, setToast] = useState('');
	const filteredEmails = useMemo(() => emails.filter((email) => [email.sender, email.subject, email.preview, email.label].join(' ').toLowerCase().includes(search.toLowerCase())), [search]);
	const selected = emails.find((email) => email.id === selectedId);
	const toggleSelect = (id) => setSelectedIds((items) => items.includes(id) ? items.filter((item) => item !== id) : [...items, id]);
	const action = (label) => { setToast(`${label} started`); window.setTimeout(() => setToast(''), 2200); };
	return <div className="app-shell"><div className={`sidebar-overlay ${mobileOpen ? 'show' : ''}`} onClick={() => setMobileOpen(false)} /><div className={mobileOpen ? 'sidebar-wrap open' : 'sidebar-wrap'}><Sidebar active={active} onNavigate={(label) => { setActive(label); setMobileOpen(false); }} onClose={() => setMobileOpen(false)} /></div><div className="main-shell"><Header search={search} setSearch={setSearch} onMenu={() => setMobileOpen(true)} /><main className="content"><section className="welcome"><div><span className="eyebrow">MONDAY, JUNE 24, 2024</span><h1>Good morning, Jordan <span>✦</span></h1><p>Here’s what’s happening with your inbox today.</p></div><button className="date-button"><BarChart3 size={16} /> This week <ChevronDown size={14} /></button></section><div className="stats-grid">{stats.map((stat) => <StatCard key={stat.label} stat={stat} />)}</div><Charts /><QuickActions onAction={action} /><div className="bottom-grid"><RecentEmails emailsToShow={filteredEmails} selectedId={selectedId} setSelectedId={setSelectedId} selected={selected} onToggleStar={(id) => { const email = emails.find((item) => item.id === id); email.starred = !email.starred; setSelectedId(id); }} selectedIds={selectedIds} onToggleSelect={toggleSelect} /><EmailPreview email={selected} onClose={() => setSelectedId(null)} /></div></main></div>{toast && <div className="toast"><Zap size={16} /> {toast}</div>}</div>;
}
