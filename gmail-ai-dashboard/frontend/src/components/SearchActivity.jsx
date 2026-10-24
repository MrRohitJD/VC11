import { useEffect, useMemo, useRef, useState } from 'react';
import {
	Activity,
	AlertTriangle,
	CheckCircle2,
	Clock3,
	Database,
	ExternalLink,
	Gauge,
	LoaderCircle,
	Search,
	Sparkles,
	Zap,
} from 'lucide-react';

const ACTION_META = {
	route_request: { label: 'Route request', lane: 'planner', laneLabel: 'Query planner' },
	plan_workflow: { label: 'Plan workflow', lane: 'planner', laneLabel: 'Query planner' },
	plan_search: { label: 'Plan query', lane: 'planner', laneLabel: 'Query planner' },
	gmail_search: { label: 'Search live Gmail', lane: 'gmail', laneLabel: 'Gmail MCP' },
	retrieve_candidates: { label: 'Retrieve candidates', lane: 'semantic', laneLabel: 'Semantic retrieval' },
	keyword_search: { label: 'Run keyword search', lane: 'keyword', laneLabel: 'Keyword retrieval' },
	corrective_search: { label: 'Recover additional matches', lane: 'keyword', laneLabel: 'Keyword retrieval' },
	validate_evidence: { label: 'Validate evidence', lane: 'validator', laneLabel: 'Evidence validator' },
	tool_agent: { label: 'Select mailbox tool', lane: 'agent', laneLabel: 'Tool agent' },
	mailbox_tool: { label: 'Run mailbox tool', lane: 'tool', laneLabel: 'Mailbox tools' },
	review_answer: { label: 'Review evidence coverage', lane: 'validator', laneLabel: 'Evidence validator' },
	compose_answer: { label: 'Compose grounded answer', lane: 'answer', laneLabel: 'Answer composer' },
	answer_ready: { label: 'Answer ready', lane: 'answer', laneLabel: 'Answer composer' },
};

const RUNNING_STATUSES = new Set(['running', 'pending', 'in_progress', 'started']);
const WARNING_STATUSES = new Set(['warning', 'partial', 'degraded']);
const ERROR_STATUSES = new Set(['error', 'failed', 'failure']);

const safeTraceUrl = (value) => {
	try {
		const url = new URL(value);
		return url.protocol === 'https:' && (url.hostname === 'smith.langchain.com' || url.hostname.endsWith('.smith.langchain.com')) ? url.href : null;
	} catch { return null; }
};

const getAction = (step) => String(step?.action || step?.tool || step?.tool_name || step?.name || 'activity');

const humanize = (value) => String(value || 'activity')
	.replace(/[_-]+/g, ' ')
	.replace(/\b\w/g, (letter) => letter.toUpperCase());

const getActionMeta = (action) => ACTION_META[action] || { label: humanize(action), lane: `action-${action}`, laneLabel: humanize(action) };

const getLane = (step, action) => {
	const explicitLane = step?.lane || step?.parallel_lane || step?.parallel_group || step?.source;
	if (explicitLane) {
		const lane = String(explicitLane);
		return { key: lane.toLowerCase().replace(/\s+/g, '-'), label: humanize(lane) };
	}
	const meta = getActionMeta(action);
	return { key: meta.lane, label: meta.laneLabel };
};

const getStatus = (step) => String(step?.status || 'completed').toLowerCase();
const isRunning = (status) => RUNNING_STATUSES.has(status);
const isWarning = (status) => WARNING_STATUSES.has(status);
const isError = (status) => ERROR_STATUSES.has(status);

const statusLabel = (status) => {
	if (isRunning(status)) return 'Running';
	if (isError(status)) return 'Failed';
	if (isWarning(status)) return 'Warning';
	return 'Complete';
};

const formatElapsed = (milliseconds) => {
	if (!Number.isFinite(milliseconds)) return '—';
	const totalSeconds = Math.max(0, milliseconds) / 1000;
	if (totalSeconds < 60) return `${totalSeconds.toFixed(1)}s`;
	const minutes = Math.floor(totalSeconds / 60);
	const seconds = Math.round(totalSeconds % 60);
	return `${minutes}m ${seconds}s`;
};

const formatPlanDate = (value) => {
	if (!value) return null;
	try {
		const date = new Date(value);
		return Number.isNaN(date.getTime()) ? null : date.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
	} catch {
		return null;
	}
};

const normalizeEvents = (steps) => {
	const latest = new Map();
	(steps || []).forEach((step, index) => {
		if (!step || typeof step !== 'object') return;
		const action = getAction(step);
		const lane = getLane(step, action);
		// New stream events carry a stable public event_id, so keep every
		// transition visible. Older payloads have no event id; retain the
		// historical latest-per-stage behavior for those responses.
		const key = step.event_id || step.step_id || step.id || `${lane.key}:${action}`;
		latest.set(key, { ...step, action, lane, index });
	});
	return [...latest.values()].sort((a, b) => a.index - b.index);
};

const StatusIcon = ({ status }) => {
	if (isRunning(status)) return <LoaderCircle size={15} className="search-activity-spinner" aria-hidden="true" />;
	if (isError(status) || isWarning(status)) return <AlertTriangle size={15} aria-hidden="true" />;
	return <CheckCircle2 size={15} aria-hidden="true" />;
};

const LaneIcon = ({ lane }) => {
	if (lane === 'gmail') return <Database size={15} aria-hidden="true" />;
	if (lane === 'semantic' || lane === 'keyword') return <Search size={15} aria-hidden="true" />;
	if (lane === 'answer') return <Sparkles size={15} aria-hidden="true" />;
	return <Zap size={15} aria-hidden="true" />;
};

export default function SearchActivity({
	steps = [],
	plan,
	traceUrl,
	traceId,
	tracingEnabled = false,
	pending = false,
	sourceCount,
	elapsedMs,
	failed = false,
}) {
	const startedAtRef = useRef(Date.now());
	const [liveElapsedMs, setLiveElapsedMs] = useState(0);
	const events = useMemo(() => normalizeEvents(steps), [steps]);
	const traceLink = safeTraceUrl(traceUrl);

	useEffect(() => {
		if (!pending) return undefined;
		const updateElapsed = () => setLiveElapsedMs(Date.now() - startedAtRef.current);
		updateElapsed();
		const timer = window.setInterval(updateElapsed, 100);
		return () => window.clearInterval(timer);
	}, [pending]);

	const laneRows = useMemo(() => {
		const latestByLane = new Map();
		events.forEach((event) => latestByLane.set(event.lane.key, event));
		return [...latestByLane.values()];
	}, [events]);

	const observedElapsedMs = events.reduce((max, event) => {
		const value = Number(event.elapsed_ms);
		return Number.isFinite(value) ? Math.max(max, value) : max;
	}, 0);
	const finalElapsedMs = Number(elapsedMs);
	const displayElapsedMs = pending
		? Math.max(observedElapsedMs, liveElapsedMs)
		: Number.isFinite(finalElapsedMs) ? finalElapsedMs : observedElapsedMs;

	const completedCount = events.filter((event) => !isRunning(getStatus(event))).length;
	const activeEvent = [...events].reverse().find((event) => isRunning(getStatus(event)));
	const hasPendingPlaceholder = pending && !activeEvent;
	const totalStages = Math.max(events.length + (hasPendingPlaceholder ? 1 : 0), 1);
	const progressValue = failed
		? Math.min(92, Math.round((completedCount / totalStages) * 100))
		: pending
			? Math.min(94, Math.round((completedCount / totalStages) * 100))
			: 100;
	const hasWarning = events.some((event) => isWarning(getStatus(event)));
	const panelState = failed ? 'Search stopped' : pending ? 'Retrieval in progress' : hasWarning ? 'Complete with warnings' : 'Search complete';
	const currentLabel = activeEvent
		? getActionMeta(activeEvent.action).label
		: pending ? 'Continuing retrieval' : panelState;

	const inferredCandidateCount = events.reduce((max, event) => {
		const count = Number(event.count ?? event.source_count ?? event.candidates);
		if (!Number.isFinite(count)) return max;
		return ['gmail_search', 'retrieve_candidates', 'keyword_search', 'corrective_search'].includes(event.action) ? Math.max(max, count) : max;
	}, 0);
	const hasSourceCount = sourceCount !== undefined && sourceCount !== null && Number.isFinite(Number(sourceCount));
	const sourceMetric = hasSourceCount ? Number(sourceCount) : inferredCandidateCount || null;
	const sourceMetricLabel = hasSourceCount ? 'Verified sources' : 'Candidates found';

	if (!events.length && !traceId && !pending && !failed) return null;

	const scopeStart = formatPlanDate(plan?.date_start);
	const scopeEnd = formatPlanDate(plan?.date_end);
	const scopeDate = scopeStart && scopeEnd ? `${scopeStart} – ${scopeEnd}` : scopeStart || scopeEnd;
	const scopeLabel = plan?.label || plan?.strategy || 'Primary mail';

	return (
		<details className={`search-activity ${pending ? 'search-activity-live' : ''} ${failed ? 'search-activity-failed' : ''}`} open={pending || undefined}>
			<summary>
				<span className="search-activity-summary-icon" aria-hidden="true"><Activity size={15} /></span>
				<span className="search-activity-summary-copy">
					<strong>Retrieval observability</strong>
					<small>{pending ? currentLabel : `${events.length} tracked stages`}</small>
				</span>
				<span className={`search-activity-summary-state ${failed ? 'is-error' : pending ? 'is-live' : ''}`}>{panelState}</span>
			</summary>

			<div className="search-activity-body">
				<div className="search-activity-overview">
					<div>
						<p className="search-activity-kicker">Mailbox workflow</p>
						<h4>{pending ? 'Watching retrieval as it happens' : failed ? 'The mailbox workflow needs another try' : 'Grounded answer ready'}</h4>
					</div>
					<span className={`search-activity-state-badge ${failed ? 'is-error' : pending ? 'is-live' : hasWarning ? 'is-warning' : 'is-complete'}`} role="status" aria-live="polite">
						<span className="search-activity-state-dot" aria-hidden="true" />
						{panelState}
					</span>
				</div>

				<div className="search-activity-progress-header">
					<span>Stage progress</span>
					<strong>{pending ? `${completedCount}/${totalStages}` : `${events.length}/${Math.max(events.length, 1)}`}</strong>
				</div>
				<div
					className="search-activity-progress-track"
					role="progressbar"
					aria-label="Mailbox retrieval progress"
					aria-valuemin="0"
					aria-valuemax="100"
					aria-valuenow={progressValue}
					aria-valuetext={`${progressValue}% complete`}
				>
					<span style={{ width: `${progressValue}%` }} />
				</div>

				<div className="search-activity-metrics" aria-label="Retrieval metrics">
					<div className="search-activity-metric">
						<Clock3 size={15} aria-hidden="true" />
						<span>Elapsed</span>
						<strong>{formatElapsed(displayElapsedMs)} total</strong>
					</div>
					<div className="search-activity-metric">
						<Database size={15} aria-hidden="true" />
						<span>{sourceMetricLabel}</span>
						<strong>{sourceMetric ?? '—'}</strong>
					</div>
					<div className="search-activity-metric">
						<Gauge size={15} aria-hidden="true" />
						<span>Active lanes</span>
						<strong>{Math.max(laneRows.length, pending ? 1 : 0)}</strong>
					</div>
				</div>

				<p className="search-activity-scope"><span>{scopeLabel}</span>{scopeDate ? ` · ${scopeDate}` : ''}</p>

				<section className="search-activity-section" aria-labelledby="search-activity-timeline-title">
					<div className="search-activity-section-heading">
						<h5 id="search-activity-timeline-title">Step timeline</h5>
						<span>{pending ? 'Live' : 'Final trace'}</span>
					</div>
					<ol className="search-activity-timeline">
						{events.map((event) => {
							const status = getStatus(event);
							const meta = getActionMeta(event.action);
							return (
								<li key={event.event_id || event.step_id || event.id || `${event.lane.key}:${event.action}`} className={`search-activity-step activity-${status}`}>
									<div className="search-activity-step-marker" aria-hidden="true"><StatusIcon status={status} /></div>
									<div className="search-activity-step-copy">
										<div className="search-activity-step-title"><strong>{meta.label}</strong><span>{statusLabel(status)}</span></div>
										<p>{event.observation || 'Stage completed.'}</p>
									</div>
									{Number.isFinite(Number(event.duration_ms))
										? <time title="Stage duration">{formatElapsed(Number(event.duration_ms))}</time>
										: Number.isFinite(Number(event.elapsed_ms)) && (pending || !Number.isFinite(finalElapsedMs)) && <time>{formatElapsed(Number(event.elapsed_ms))}</time>}
								</li>
							);
						})}
						{pending && (
							<li className="search-activity-step activity-running search-activity-placeholder">
								<div className="search-activity-step-marker" aria-hidden="true"><LoaderCircle size={15} className="search-activity-spinner" /></div>
								<div className="search-activity-step-copy">
									<div className="search-activity-step-title"><strong>{currentLabel}</strong><span>Running</span></div>
									<p>{activeEvent?.observation || 'Waiting for the next mailbox event…'}</p>
								</div>
							</li>
						)}
						{!events.length && !pending && <li className="search-activity-empty">No stage details were returned.</li>}
					</ol>
				</section>

				<section className="search-activity-section search-activity-lanes" aria-labelledby="search-activity-lanes-title">
					<div className="search-activity-section-heading">
						<h5 id="search-activity-lanes-title">Tool lanes</h5>
						<span>{laneRows.length > 1 ? `${laneRows.length} lanes` : 'Workflow'}</span>
					</div>
					<div className="search-activity-lane-list">
						{laneRows.length ? laneRows.map((event) => {
							const status = getStatus(event);
							return (
								<div className="search-activity-lane" key={event.lane.key}>
									<div className={`search-activity-lane-icon lane-${event.lane.key}`} aria-hidden="true"><LaneIcon lane={event.lane.key} /></div>
									<div className="search-activity-lane-copy"><strong>{event.lane.label}</strong><span>{getActionMeta(event.action).label}</span></div>
									<span className={`search-activity-lane-status activity-${status}`}>{statusLabel(status)}</span>
								</div>
							);
						}) : (
							<div className="search-activity-lane search-activity-lane-empty">
								<div className="search-activity-lane-icon lane-orchestrator" aria-hidden="true"><Zap size={15} /></div>
								<div className="search-activity-lane-copy"><strong>Retrieval orchestration</strong><span>Waiting for mailbox tools</span></div>
								<span className="search-activity-lane-status activity-running">Starting</span>
							</div>
						)}
					</div>
				</section>

				<div className="search-activity-trace">
					{pending ? <span><span className="search-activity-live-dot" aria-hidden="true" /> Streaming activity from the mailbox workflow</span>
						: traceLink ? <a href={traceLink} target="_blank" rel="noopener noreferrer">Open LangSmith trace <ExternalLink size={13} /></a>
							: <span>{tracingEnabled ? 'LangSmith tracing enabled; trace link unavailable.' : 'LangSmith tracing is off. Activity is available locally.'}</span>}
				</div>
			</div>
		</details>
	);
}
