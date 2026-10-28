import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Mail, Check, Loader, Tag, Database, Wifi, Shield, AlertCircle } from 'lucide-react';
import { startSync, getSyncStatus, getSyncStreamUrl } from '../services/api';

const stages = [
	{ key: 'starting', label: 'Connecting to Gmail...', icon: Wifi },
	{ key: 'authenticating', label: 'Authenticating with Gmail...', icon: Shield },
	{ key: 'labels', label: 'Syncing labels...', icon: Tag },
	{ key: 'listing', label: 'Fetching your emails...', icon: Mail },
	{ key: 'fetching', label: 'Downloading email details...', icon: Mail },
	{ key: 'storing', label: 'Saving emails to your account...', icon: Database },
	{ key: 'completed', label: 'Recent Inbox messages ready', icon: Check },
];

function getStageIndex(stage) {
	const idx = stages.findIndex((s) => s.key === stage);
	return idx >= 0 ? idx : 0;
}

export default function SyncPage() {
	const navigate = useNavigate();
	const [syncState, setSyncState] = useState({
		status: 'idle',
		stage: 'starting',
		emails_found: 0,
		emails_synced: 0,
		emails_fetched: 0,
		labels_synced: 0,
		error: null,
	});
	const [startError, setStartError] = useState(null);
	const syncStarted = useRef(false);
	const eventSourceRef = useRef(null);

	useEffect(() => {
		if (syncStarted.current) return;
		syncStarted.current = true;

		let pollTimer;
		let eventSource;
		const updateStatus = () => {
			getSyncStatus()
				.then((response) => setSyncState((prev) => ({ ...prev, ...response.data })))
			.catch(() => {});
		};

		// Start first so the stream cannot capture the pre-sync idle state.
		startSync()
			.then((response) => setSyncState((prev) => ({ ...prev, ...response.data })))
			.then(() => {
				eventSource = new EventSource(getSyncStreamUrl(), { withCredentials: true });
				eventSourceRef.current = eventSource;
				eventSource.onmessage = (event) => {
					try {
						const data = JSON.parse(event.data);
						setSyncState((prev) => ({ ...prev, ...data }));
					} catch {
						// ignore parse errors
					}
				};
				eventSource.onerror = () => eventSource.close();
			})
			.catch((error) => {
				const message = error.response?.data?.detail || 'Unable to start Gmail sync.';
				setStartError(message);
				setSyncState((prev) => ({ ...prev, status: 'error', stage: 'error', error: message }));
			});

		pollTimer = window.setInterval(updateStatus, 2000);

		return () => {
			window.clearInterval(pollTimer);
			eventSource?.close();
		};
	}, []);

	// Redirect to dashboard after completion
	useEffect(() => {
		if (syncState.status === 'completed') {
			const timer = setTimeout(() => navigate('/dashboard', { replace: true }), 2500);
			return () => clearTimeout(timer);
		}
	}, [syncState.status, navigate]);

	const currentStageIndex = getStageIndex(syncState.stage);
	const isCompleted = syncState.status === 'completed';
	const isError = syncState.status === 'error';

	return (
		<div className="sync-shell">
			<main className="sync-card">
				{/* Animated spinner / completion icon */}
				<div className={`sync-spinner-wrap ${isCompleted ? 'completed' : ''} ${isError ? 'error' : ''}`}>
					<div className="sync-spinner-ring" />
					<div className="sync-spinner-icon">
						{isError ? (
							<AlertCircle size={32} />
						) : isCompleted ? (
							<Check size={32} />
						) : (
							<Loader size={32} className="spin" />
						)}
					</div>
				</div>

				{/* Status text */}
				<h1 className="sync-title">
					{isError
						? 'Sync Error'
						: isCompleted
						? 'Inbox Ready'
						: 'Syncing Your Gmail'}
				</h1>
				{isCompleted && syncState.background_status === 'syncing' && <p>The rest of your Inbox is syncing in the background while you use the dashboard.</p>}

				{isError && (
					<p className="sync-error-msg">{startError || syncState.error || 'An unexpected error occurred.'}</p>
				)}

				{/* Progress steps */}
				<div className="sync-steps">
					{stages.map((stage, idx) => {
						const StageIcon = stage.icon;
						const isDone = idx < currentStageIndex || isCompleted;
						const isCurrent = idx === currentStageIndex && !isCompleted && !isError;

						return (
							<div
								key={stage.key}
								className={`sync-step ${isDone ? 'done' : ''} ${isCurrent ? 'current' : ''}`}
							>
								<div className="sync-step-icon">
									{isDone ? <Check size={14} /> : <StageIcon size={14} />}
								</div>
								<span>{stage.label}</span>
								{/* Show counts for specific stages */}
								{isCurrent && stage.key === 'listing' && syncState.emails_found > 0 && (
									<em className="sync-step-count">
										{syncState.emails_found.toLocaleString()} found
									</em>
								)}
								{isCurrent && stage.key === 'fetching' && (
									<em className="sync-step-count">
										{(syncState.emails_fetched || 0).toLocaleString()} / {syncState.emails_found.toLocaleString()}
									</em>
								)}
								{isCurrent && stage.key === 'storing' && (
									<em className="sync-step-count">
										{syncState.emails_synced.toLocaleString()} saved
									</em>
								)}
								{isCurrent && stage.key === 'labels' && syncState.labels_synced > 0 && (
									<em className="sync-step-count">
										{syncState.labels_synced} labels
									</em>
								)}
							</div>
						);
					})}
				</div>

				{/* Completion summary */}
				{isCompleted && (
					<div className="sync-summary">
						<div className="sync-summary-stat">
							<Mail size={18} />
							<div>
								<strong>{syncState.emails_synced?.toLocaleString() || 0}</strong>
								<span>recent Inbox messages ready</span>
							</div>
						</div>
						<div className="sync-summary-stat">
							<Tag size={18} />
							<div>
								<strong>{syncState.labels_synced || 0}</strong>
								<span>labels synchronized</span>
							</div>
						</div>
						<p className="sync-redirect-note">Redirecting to dashboard...</p>
					</div>
				)}

				{isError && (
					<button className="sync-retry-btn" onClick={() => window.location.reload()}>
						Try Again
					</button>
				)}
			</main>
		</div>
	);
}
