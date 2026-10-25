import { Link } from 'react-router-dom';
import './PublicPages.css';

const CONTACT = 'bibinvid4@gmail.com';

function PublicLayout({ children }) {
  return <div className="public-page">
    <header className="public-header"><Link to="/" className="public-brand">Gmail AI Dashboard</Link><nav><Link to="/privacy">Privacy policy</Link><Link to="/login">Sign in</Link></nav></header>
    <main className="public-main">{children}</main>
    <footer className="public-footer"><span>Gmail AI Dashboard</span><Link to="/privacy">Privacy policy</Link><a href={`mailto:${CONTACT}`}>Contact</a></footer>
  </div>;
}

export function PublicHome() {
  return <PublicLayout>
    <section className="public-hero"><p className="public-kicker">Gmail AI Dashboard</p><h1>Understand and search your Gmail with AI</h1><p>Connect your Gmail account to view mailbox activity, organize messages, search your mail, and ask questions about your email. You control when the app sends a message on your behalf.</p><div className="public-actions"><Link className="public-primary" to="/login">Get started</Link><Link to="/privacy">How we use your data</Link></div></section>
    <section className="public-section"><h2>What the app does</h2><div className="public-features"><article><h3>Mailbox dashboard</h3><p>Sync messages and view email activity, categories, and important mail.</p></article><article><h3>Search and assistant</h3><p>Search your mailbox and use an AI assistant to answer questions grounded in your messages.</p></article><article><h3>Send with your approval</h3><p>Compose and send email from your Gmail account when you choose to do so.</p></article></div></section>
    <section className="public-section"><h2>Google access requested</h2><p>The app requests permission to read Gmail messages and send email on your behalf. Reading is used for synchronization, search, summaries, and AI features. Sending is used for user initiated email. Read our <Link to="/privacy">privacy policy</Link> before connecting.</p></section>
  </PublicLayout>;
}

export function PrivacyPolicy() {
  return <PublicLayout><article className="privacy-content">
    <p className="public-kicker">Privacy policy</p><h1>Gmail AI Dashboard privacy policy</h1><p>Last updated: 25 October 2026</p>
    <p>This policy describes how Gmail AI Dashboard accesses and uses data when you connect a Google account. For questions or data requests, contact <a href={`mailto:${CONTACT}`}>{CONTACT}</a>.</p>
    <h2>Information we access</h2><p>With your permission, we receive your Google account identifier, name, email address, and profile image. The Gmail permissions requested are read-only access to messages and permission to send messages. We may retrieve message headers, bodies, snippets, labels, and attachment metadata to provide the app's features. The app also receives the prompts you enter and messages you approve for sending.</p>
    <h2>How we use it</h2><p>We use your account details to sign you in. We use Gmail data to synchronize and display your mailbox, classify and search email, show analytics, and answer your questions in the AI assistant. The Gmail MCP server searches messages for the signed-in user and passes the results into the app's search workflow. We use send permission only for messages you initiate or approve in the app.</p>
    <h2>Storage and service providers</h2><p>The app stores account details, OAuth credentials, synchronized message content and metadata, classifications, search indexes, and assistant conversation history on its server. Hosting, database, and configured AI model providers process data needed to run these features. Relevant email content and prompts may be sent to a configured AI provider to generate classifications, embeddings, or answers. These providers process data on our behalf to operate the app.</p>
    <h2>Sharing and advertising</h2><p>We do not use Gmail data for advertising. We do not sell Gmail data. Access to data is limited to providing and maintaining the app's features and to service providers needed for those features.</p>
    <h2>Your choices and deletion</h2><p>You can disconnect the app from your Google Account permissions page to stop future Google API access. In the app's settings, you can erase stored mailbox messages, derived indexes, and assistant history; this also clears the stored Google OAuth credentials. The app retains your profile and label records after this action. To request complete account deletion or ask about retained records, email <a href={`mailto:${CONTACT}`}>{CONTACT}</a>. Erasing app data does not delete messages from Gmail.</p>
    <h2>Security</h2><p>We use access controls and authenticated sessions to limit access to each user's mailbox data. No internet service can guarantee absolute security. If you believe your account has been affected, contact us at <a href={`mailto:${CONTACT}`}>{CONTACT}</a>.</p>
    <h2>Changes</h2><p>We may update this policy as the app changes. The current version will be published on this page with its update date.</p>
  </article></PublicLayout>;
}
