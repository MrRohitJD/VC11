import { Mail, Shield, MessageSquare, Tag, Paperclip, Clock, Users, FileText } from 'lucide-react';
import { getGoogleAuthUrl } from '../services/api';

const accessItems = [
	{ icon: Mail, text: 'Emails and email content' },
	{ icon: Users, text: 'Sender and recipient information' },
	{ icon: FileText, text: 'Subject and timestamps' },
	{ icon: MessageSquare, text: 'Gmail threads' },
	{ icon: Tag, text: 'Gmail labels' },
	{ icon: Paperclip, text: 'Attachment metadata' },
];

export default function GmailConsent() {
	const handleContinue = () => {
		window.location.assign(getGoogleAuthUrl());
	};

	return (
		<div className="consent-shell">
			<main className="consent-card">
				<div className="consent-icon-wrap">
					<div className="consent-icon">
						<Mail size={28} />
					</div>
					<div className="consent-icon-ring" />
				</div>

				<h1>Connect Your Gmail</h1>
				<p className="consent-subtitle">
					To provide the email management features, this application needs permission
					to read your Gmail messages and labels.
				</p>

				<div className="consent-access-box">
					<div className="consent-access-header">
						<Shield size={16} />
						<span>We may access:</span>
					</div>
					<ul className="consent-access-list">
						{accessItems.map(({ icon: Icon, text }) => (
							<li key={text}>
								<Icon size={15} />
								<span>{text}</span>
							</li>
						))}
					</ul>
				</div>

				<p className="consent-note">
					This data will be synchronized to your account database so the application
					can provide its email features. We will never modify or delete anything in
					your Gmail account.
				</p>

				<button className="consent-continue-btn" onClick={handleContinue}>
					<span className="google-g">G</span>
					Continue with Google
				</button>

				<p className="consent-fine-print">
					You will be taken to Google's official consent screen where you can
					review and approve access.
				</p>
			</main>
		</div>
	);
}
