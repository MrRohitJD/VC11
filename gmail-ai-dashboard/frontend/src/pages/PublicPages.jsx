import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { enterDemo } from '../services/api';
import './PublicPages.css';

const CONTACT = 'info@rohit-jadhav.dev';
const APP_NAME = "GAI – Rohit's Mail Assistant";

function PublicLayout({ children }) {
  const { isAuthenticated } = useAuth();
  return <div className="public-page">
    <header className="public-header"><Link to="/" className="public-brand"><strong>GAI</strong><small>Rohit's Mail Assistant</small></Link><nav><Link to="/privacy">Privacy policy</Link><Link to="/terms">Terms of Service</Link><Link to="/contact">Contact</Link><Link to={isAuthenticated ? '/dashboard' : '/login'}>{isAuthenticated ? 'Dashboard' : 'Sign in'}</Link></nav></header>
    <main className="public-main">{children}</main>
    <footer className="public-footer"><span><strong>GAI</strong> – Rohit's Mail Assistant</span><Link to="/privacy">Privacy policy</Link><Link to="/terms">Terms of Service</Link><Link to="/contact">Contact</Link></footer>
  </div>;
}

export function PublicHome() {
  const { isAuthenticated, loading, checkAuth } = useAuth();
  const navigate = useNavigate();
  const [openingDemo, setOpeningDemo] = useState(false);
  const [demoError, setDemoError] = useState('');
  const openDemo = async () => {
    try {
      setDemoError('');
      setOpeningDemo(true);
      await enterDemo();
      await checkAuth();
      navigate('/dashboard', { replace: true });
    } catch (error) {
      setDemoError(error.response?.data?.detail || 'The demo is unavailable right now.');
    } finally {
      setOpeningDemo(false);
    }
  };
  return <PublicLayout>
    <section className="public-hero"><p className="public-kicker">{APP_NAME}</p><h1>Understand and search your Gmail with AI</h1><p><strong>GAI is an AI-powered Gmail assistant.</strong> GAI – Rohit's Mail Assistant connects to your Gmail to help you view mailbox activity, organize messages, search your mail, and ask questions about your email. You control when the app sends a message on your behalf.</p><div className="public-actions">{loading ? <span className="public-primary" aria-busy="true">Get started</span> : <Link className="public-primary" to={isAuthenticated ? '/dashboard' : '/login'}>Get started</Link>}<button className="public-demo" type="button" onClick={openDemo} disabled={openingDemo}>{openingDemo ? 'Opening demo...' : 'Open demo'}</button><Link to="/privacy">How we use your data</Link></div>{demoError && <p role="alert">{demoError}</p>}</section>
    <section className="public-section"><h2>What the app does</h2><div className="public-features"><article><h3>Mailbox dashboard</h3><p>Sync messages and view email activity, categories, and important mail.</p></article><article><h3>Search and assistant</h3><p>Search your mailbox and use an AI assistant to answer questions grounded in your messages.</p></article><article><h3>Send with your approval</h3><p>Compose and send email from your Gmail account when you choose to do so.</p></article></div></section>
    <section className="public-section"><h2>Google access requested</h2><p>GAI currently requests Google sign-in/profile access, Gmail read-only access (<code>gmail.readonly</code>), and Gmail send access (<code>gmail.send</code>). Read access supports synchronization, search, summaries, categorization, embeddings, and RAG answers; sending is limited to messages you explicitly approve. The current implementation reads Gmail metadata such as headers, labels, snippets, message IDs, and thread IDs through the read-only scope, but does not request <code>gmail.metadata</code> as a separate OAuth scope. Read our <Link to="/privacy">privacy policy</Link> before connecting.</p></section>
  </PublicLayout>;
}

export function PrivacyPolicy() {
  return <PublicLayout><article className="privacy-content">
    <p className="public-kicker">Privacy policy</p><h1>GAI – Rohit's Mail Assistant privacy policy</h1><p>Last updated: 14 November 2026</p>
    <p>GAI is the service brand for Rohit's Mail Assistant, an AI-powered Gmail assistant. This policy describes how GAI accesses and uses data when you connect a Google account. For questions or data requests, contact <a href={`mailto:${CONTACT}`}>{CONTACT}</a>.</p>
    <h2>Gmail OAuth authentication and scopes</h2><p>When you choose Google sign-in, GAI uses Google's OAuth flow to authenticate you and stores the Google account identifier, name, email address, and profile image returned for the signed-in account. The current application requests <code>gmail.readonly</code> and <code>gmail.send</code>, along with the Google OpenID and user profile scopes used for sign-in. <code>gmail.readonly</code> is used to read and synchronize Gmail; <code>gmail.send</code> is used only when you explicitly approve a message for delivery. The current code does not request <code>gmail.metadata</code> as a separate OAuth scope. Gmail metadata is nevertheless read as part of message access, including message and thread IDs, headers, labels, snippets, dates, size information, and attachment metadata.</p>
    <h2>Gmail data we access</h2><p>Depending on the features you use, GAI may access message headers and addresses (From, To, Cc, and Bcc), subject lines, plain-text and HTML bodies, snippets, received dates, Gmail labels, read/starred state, message and thread identifiers, history identifiers, size estimates, and attachment metadata such as filename, MIME type, size, and attachment ID. The sync code stores attachment metadata; it does not download attachment file contents as part of the normal message parse. GAI also processes prompts you enter, assistant responses, drafts, and messages you approve for sending.</p>
    <h2>How GAI uses Gmail data</h2><p>GAI uses account data to sign you in and Gmail data to synchronize and display your mailbox, show analytics, classify messages, support keyword and semantic search, and answer questions in the AI assistant. The Gmail search workflow can retrieve relevant live messages for the signed-in user. Email sending is user-controlled: the app can create drafts or approval records, but delivery requires your explicit approval.</p>
    <h2>AI categorization, embeddings, and RAG</h2><p>For categorization, the application analyzes message fields such as sender, subject, snippet, and body and stores the resulting category and classification details. For semantic search, email content is split into chunks and converted into vector embeddings. The retrieval-augmented generation (RAG) workflow combines those stored embeddings and message metadata with keyword or live Gmail retrieval to ground assistant answers in your mailbox. Assistant prompts, relevant retrieved email content, and related context may be sent to the configured AI provider needed for classification, embeddings, or answer generation.</p>
    <h2>Storage, retention, and deletion</h2><p>GAI stores account and Gmail connection records, OAuth credentials and scopes, synchronized message content and metadata, Gmail labels, classifications, email chunks, embeddings, chat sessions and messages, retrieval citations, and pending or sent approval records in the configured application database. If you connect a personal PostgreSQL database, new mailbox syncs and assistant activity use that database while existing data remains in the application database. Developer database and personal AI credentials are encrypted by the application before storage; Gmail OAuth credentials are stored server-side so the app can refresh the connection.</p><p>The current implementation does not define an automatic time-based mailbox retention period. You can use Settings → Erase All Data to remove synchronized emails and their dependent classifications, chunks, embeddings, attachments, retrieval records, sync history, assistant history, and send approvals; it also clears the stored Gmail OAuth credentials and signs you out. Your profile, Gmail labels, and chat quota records are preserved by that control. The action does not delete messages from Gmail. For a deletion request beyond the available control or questions about records that may remain in infrastructure backups or logs, contact <a href={`mailto:${CONTACT}`}>{CONTACT}</a>.</p>
    <h2>Third-party AI services and safeguards</h2><p>Depending on the deployment configuration, GAI uses third-party AI services such as Azure OpenAI for embeddings and configured language-model providers such as Azure OpenAI, NVIDIA, or ExperientialLabs for classifications and assistant answers. Users may also connect their own Google Gemini or OpenAI key for supported assistant features. These providers receive only the content needed for the requested operation and process it on behalf of the application. Access is scoped to the authenticated user, requests use authenticated sessions, and developer credentials are encrypted before storage. Review the applicable provider terms and privacy commitments before enabling a personal provider.</p>
    <h2>Google Limited Use</h2><p>The <a href="https://developers.google.com/terms/api-services-user-data-policy">Google API Services User Data Policy</a>, including its Limited Use requirements, applies to GAI's use and transfer of information received from Google APIs. GAI's use and transfer of Gmail data is limited to providing or improving the user-facing Gmail features described above, including synchronization, search, categorization, embeddings, RAG answers, analytics, and user-approved sending. We do not use Gmail data for advertising, sell it, or use it to build advertising profiles. We do not transfer Gmail data to third parties except to service providers that are necessary to operate these features and only for the requested processing. GAI does not use Gmail data for unrelated purposes.</p>
    <h2>Revoking access</h2><p>You can revoke GAI's Gmail access by opening <a href="https://myaccount.google.com/connections">your Google Account connections</a>, selecting GAI, and removing its access. Revoking access stops future Gmail API access once the authorization is no longer valid; it does not automatically delete copies already stored by the application. Use Settings → Erase All Data or email <a href={`mailto:${CONTACT}`}>{CONTACT}</a> to request removal of stored application data. Erasing app data does not delete messages from Gmail.</p>
    <h2>Security</h2><p>GAI uses authenticated sessions, access controls, OAuth state and PKCE protections, and secure session cookies in production to limit access to each user's mailbox data. No internet service can guarantee absolute security. If you believe your account has been affected, contact us at <a href={`mailto:${CONTACT}`}>{CONTACT}</a>.</p>
    <h2>Changes</h2><p>We may update this policy as the app changes. The current version will be published on this page with its update date.</p>
  </article></PublicLayout>;
}

export function TermsOfService() {
  return <PublicLayout><article className="privacy-content">
    <p className="public-kicker">Terms of Service</p><h1>GAI – Rohit's Mail Assistant Terms of Service</h1><p>Last updated: 14 November 2026</p>
    <p>These terms govern your use of GAI, the service brand for Rohit's Mail Assistant. For questions, contact <a href={`mailto:${CONTACT}`}>{CONTACT}</a>.</p>
    <h2>Using the app</h2><p>You need a Google account and must authorize the requested Gmail permissions to use connected mailbox features. You are responsible for activity you initiate through your account and for complying with applicable laws and Google's terms when using Gmail.</p>
    <h2>What the app provides</h2><p>GAI – Rohit's Mail Assistant offers mailbox viewing, synchronization, search, analytics, AI-assisted answers, and user-controlled email sending. AI-generated classifications and answers can be incomplete or inaccurate. Review important information and any message before relying on it or sending it.</p>
    <h2>Permissions and your data</h2><p>Connecting Gmail authorizes GAI to read messages and send messages on your behalf as shown on Google's consent screen. The app's use of your information is described in the <Link to="/privacy">privacy policy</Link>. You can revoke Google access through your Google Account settings. You can also erase stored mailbox data through the app's settings; doing so does not remove messages from Gmail.</p>
    <h2>Acceptable use</h2><p>Do not use the app to send spam, impersonate others, violate others' rights, or interfere with the service. Do not attempt to access another user's account or mailbox without authorization.</p>
    <h2>Availability and changes</h2><p>Features may change or be interrupted for maintenance, provider outages, or security reasons. We may update these terms and will publish the current version here with its update date.</p>
    <h2>Contact</h2><p>For support or questions about these terms, email <a href={`mailto:${CONTACT}`}>{CONTACT}</a>.</p>
  </article></PublicLayout>;
}

export function ContactPage() {
  return <PublicLayout><article className="privacy-content contact-content">
    <p className="public-kicker">Contact and support</p><h1>Support for GAI – Rohit's Mail Assistant</h1>
    <p>GAI is the service brand for Rohit's Mail Assistant. If you need help with Gmail connection, mailbox sync, AI assistant behavior, privacy, or data requests, contact the existing support address below.</p>
    <h2>Support contact</h2><p><a className="contact-email" href={`mailto:${CONTACT}`}>{CONTACT}</a></p>
    <p>When writing about a privacy or deletion request, include the Google account email associated with the service and a clear description of what you want reviewed. Do not send passwords, OAuth tokens, or message contents unless support specifically asks for information that is necessary to investigate your request.</p>
    <p>See the <Link to="/privacy">privacy policy</Link> for Gmail data handling and the <Link to="/terms">Terms of Service</Link> for service terms.</p>
  </article></PublicLayout>;
}
