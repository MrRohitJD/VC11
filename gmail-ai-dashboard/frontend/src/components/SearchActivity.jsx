import { Activity, ExternalLink } from 'lucide-react';

const safeTraceUrl = (value) => {
	try {
		const url = new URL(value);
		return url.protocol === 'https:' && (url.hostname === 'smith.langchain.com' || url.hostname.endsWith('.smith.langchain.com')) ? url.href : null;
	} catch { return null; }
};

export default function SearchActivity({ steps = [], plan, traceUrl, traceId, tracingEnabled = false, pending = false }) {
	const latest = new Map();
	steps.forEach((step) => latest.set(step.action, step));
	const events = [...latest.values()];
	const traceLink = safeTraceUrl(traceUrl);
	if (!events.length && !traceId && !pending) return null;
	return (
		<details className="search-activity" open={pending}>
			<summary><Activity size={15} /> Search activity <span>{pending ? 'In progress' : `${events.length} steps`}</span></summary>
			<div className="search-activity-body">
				{plan && <p className="search-activity-scope">Primary mail{plan.date_start ? ` · ${new Date(plan.date_start).toLocaleDateString()} – ${new Date(plan.date_end).toLocaleDateString()}` : ''}</p>}
				<ol>{events.map((step) => (
					<li key={step.action} className={`activity-${step.status || 'completed'}`}>
						<span className="activity-indicator" aria-hidden="true" />
						<span>{step.observation}</span>
						{Number.isFinite(step.elapsed_ms) && <time>{(step.elapsed_ms / 1000).toFixed(1)}s</time>}
					</li>
				))}</ol>
				{!pending && <div className="search-activity-trace">
					{traceLink ? <a href={traceLink} target="_blank" rel="noopener noreferrer">Open LangSmith trace <ExternalLink size={13} /></a>
						: <span>{tracingEnabled ? 'LangSmith tracing enabled; trace link unavailable.' : 'LangSmith tracing is off. Activity is available locally.'}</span>}
				</div>}
			</div>
		</details>
	);
}
