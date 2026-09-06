export const stats = [
	{ label: 'Total Emails', value: '1,248', change: '+12.5%', positive: true, icon: 'inbox' },
	{ label: 'Auto Labeled', value: '892', change: '+8.2%', positive: true, icon: 'tag' },
	{ label: 'Important', value: '156', change: '+4.8%', positive: true, icon: 'important' },
	{ label: 'Unread', value: '42', change: '-18.3%', positive: true, icon: 'unread' },
];
export const categoryData = [{ name: 'Work', value: 42 }, { name: 'Personal', value: 28 }, { name: 'Promotions', value: 18 }, { name: 'Social', value: 12 }];
export const emailActivity = [{ day: 'Mon', incoming: 42, outgoing: 26 }, { day: 'Tue', incoming: 48, outgoing: 34 }, { day: 'Wed', incoming: 36, outgoing: 25 }, { day: 'Thu', incoming: 55, outgoing: 39 }, { day: 'Fri', incoming: 47, outgoing: 32 }, { day: 'Sat', incoming: 26, outgoing: 17 }, { day: 'Sun', incoming: 32, outgoing: 23 }];
export const emails = [
	{ id: 1, sender: 'Sarah Mitchell', email: 'sarah.mitchell@acme.com', initials: 'SM', avatarColor: '#e1dcff', subject: 'Q3 Marketing Strategy & Goals', preview: 'Hi Jordan, I wanted to share the updated strategy...', label: 'Work', date: 'Jun 24, 2024', starred: true, hasAttachment: true, body: 'I wanted to share the updated strategy and goals for Q3. The team has incorporated the feedback from our last review and we are excited to get started.' },
	{ id: 2, sender: 'Alex Rivera', email: 'alex.rivera@design.co', initials: 'AR', avatarColor: '#d2f2e7', subject: 'Design handoff: Dashboard v2', preview: 'The latest screens are ready for your review...', label: 'Work', date: 'Jun 24, 2024', starred: false, body: 'The latest screens are ready for your review. Everything is organized in the project file, including the new dashboard states.' },
	{ id: 3, sender: 'Netflix', email: 'info@netflix.com', initials: 'N', avatarColor: '#ffd6d3', subject: 'New releases you might like', preview: 'Check out what is new this week...', label: 'Promotions', date: 'Jun 23, 2024', starred: false, body: 'Check out what is new this week, curated specially for you based on your recent viewing history.' },
	{ id: 4, sender: 'Mom', email: 'mom@example.com', initials: 'M', avatarColor: '#ffe7b2', subject: 'Sunday dinner', preview: 'Are you free for dinner this Sunday?', label: 'Personal', date: 'Jun 23, 2024', starred: true, body: 'Are you free for dinner this Sunday? Let me know what time works best for you.' },
	{ id: 5, sender: 'LinkedIn', email: 'notifications@linkedin.com', initials: 'in', avatarColor: '#d8e8ff', subject: 'You appeared in 14 searches this week', preview: 'See who is looking at your profile...', label: 'Social', date: 'Jun 22, 2024', starred: false, body: 'See who is looking at your profile and discover new opportunities in your network.' },
];