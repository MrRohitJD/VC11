const CATEGORY_HUES = {
	WORK: 200, JOB: 200, RECRUITMENT_UPDATES: 200,
	EDUCATION: 165, AI_DEVELOPMENT: 165, TECHNICAL: 165, PROJECTS: 165,
	FINANCE: 125, BILLS: 125, RECEIPTS: 125, ORDERS: 125, SHIPPING: 125,
	PROMOTIONS: 40, SOCIAL: 40, EVENTS: 40, NEWSLETTERS: 40,
	TRAVEL: 185, MEETINGS: 185, CALENDAR: 185, UPDATES: 185,
	PERSONAL: 345, HEALTH: 345, CUSTOMER_SUPPORT: 345,
	SPAM: 10, SECURITY: 10, LEGAL: 10, GOVERNMENT: 10,
	ACCOUNT: 85, SUBSCRIPTIONS: 85, OTHER: 220,
};

const CATEGORY_OFFSETS = {
	WORK: 0, JOB: 9, RECRUITMENT_UPDATES: 18,
	EDUCATION: 0, AI_DEVELOPMENT: 9, TECHNICAL: 18, PROJECTS: 27,
	FINANCE: 0, BILLS: 9, RECEIPTS: 18, ORDERS: 27, SHIPPING: 36,
	PROMOTIONS: 0, SOCIAL: 9, EVENTS: 18, NEWSLETTERS: 27,
	TRAVEL: 0, MEETINGS: 9, CALENDAR: 18, UPDATES: 27,
	PERSONAL: 0, HEALTH: 9, CUSTOMER_SUPPORT: 18,
	SPAM: 0, SECURITY: 9, LEGAL: 18, GOVERNMENT: 27,
	ACCOUNT: 0, SUBSCRIPTIONS: 9, OTHER: 0,
};

export default function MailLabelBadge({ email }) {
	const label = email.ai_label || email.labels?.find((item) => item.is_ai)?.name;
	if (!label) return null;
	const key = label.toUpperCase().replace(/[^A-Z0-9]+/g, '_');
	const hue = CATEGORY_HUES[key] ?? 200;
	const offset = CATEGORY_OFFSETS[key] ?? 0;
	const categoryHue = (hue + offset) % 360;
	const displayLabel = label.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, (letter) => letter.toUpperCase());
	return <em className="label-badge category-label" style={{ backgroundColor: `hsl(${categoryHue} 45% 97%)`, color: `hsl(${categoryHue} 32% 48%)`, border: `1px solid hsl(${categoryHue} 30% 91%)` }}>{displayLabel}</em>;
}
