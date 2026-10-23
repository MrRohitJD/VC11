"""Idempotent, fictional mailbox data for the public demo account.

This module writes only ordinary mail, labels, and classification rows. It does
not call Gmail, an LLM, a chunker, or an embedding provider.
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.email import (
    AiCategory, Email, EmailClassification, EmailLabel, GmailAccount, Label, User,
)


DEMO_EMAIL = "demo@gmail.com"
DEMO_GOOGLE_ID = "demo:mailbox"

GMAIL_CATEGORIES = (
    ("CATEGORY_PERSONAL", "Primary"),
    ("CATEGORY_SOCIAL", "Social"),
    ("CATEGORY_PROMOTIONS", "Promotions"),
    ("CATEGORY_UPDATES", "Updates"),
    ("CATEGORY_FORUMS", "Forums"),
)

AI_DESCRIPTIONS = {
    "WORK": "Professional work tasks and team communications.",
    "JOB": "Applications, interviews, and recruiter outreach.",
    "RECRUITMENT_UPDATES": "Updates on job applications.",
    "EDUCATION": "Courses and university messages.",
    "PROJECTS": "Project tasks and collaboration.",
    "MEETINGS": "Meeting invitations and schedules.",
    "FINANCE": "Banking and personal finance.",
    "BILLS": "Bills and payment reminders.",
    "RECEIPTS": "Payment confirmations and receipts.",
    "SECURITY": "Sign-in and account security alerts.",
    "ACCOUNT": "Account and profile notifications.",
    "TRAVEL": "Travel bookings and itineraries.",
    "EVENTS": "Event invitations and registrations.",
    "SOCIAL": "Social network and community activity.",
    "PERSONAL": "Messages from friends and family.",
    "PROMOTIONS": "Offers and marketing messages.",
    "NEWSLETTERS": "Editorial digests and newsletters.",
    "UPDATES": "Platform and product announcements.",
    "TECHNICAL": "Developer tools and technical alerts.",
    "CALENDAR": "Events, reminders, and scheduling.",
    "AI_DEVELOPMENT": "AI and machine learning product updates.",
    "ORDERS": "Purchase and order updates.",
    "SHIPPING": "Shipment and delivery updates.",
    "SUBSCRIPTIONS": "Plan renewals and subscription changes.",
    "CUSTOMER_SUPPORT": "Support tickets and help desk replies.",
    "HEALTH": "Healthcare appointments and records.",
    "LEGAL": "Contracts and legal notices.",
    "GOVERNMENT": "Government and civic notices.",
    "SPAM": "Unsolicited or suspicious messages.",
    "OTHER": "Messages outside the other AI categories.",
}

# subject, sender name, sender address, body, days ago, hour, minute,
# Gmail category, synthetic AI category, personal label
SAMPLE_MAIL = (
    ("Interview invitation: software intern", "Maya at Northstar", "maya@northstar.example", "Hi Demo, we enjoyed reviewing your application. Could you join a software intern interview on Tuesday at 10:00 AM? Please reply with your availability.", 0, 9, 15, "CATEGORY_PERSONAL", "JOB", "Career"),
    ("Your application is in final review", "Atlas Careers", "careers@atlas.example", "Your data analyst application has reached final review. We expect to share the next step next week.", 1, 16, 42, "CATEGORY_PERSONAL", "RECRUITMENT_UPDATES", "Career"),
    ("Project presentation schedule", "Professor Rao", "rao@university.example", "Your dashboard project presentation is on Thursday at 2:00 PM in Room 204. Bring a short live demonstration and architecture diagram.", 2, 11, 5, "CATEGORY_PERSONAL", "EDUCATION", "University"),
    ("API review with the project team", "Nikhil Shah", "nikhil@northstar.example", "Can we review the API integration and test plan at 3:30 PM tomorrow? I added the agenda to our shared document.", 3, 14, 20, "CATEGORY_PERSONAL", "MEETINGS", "Project"),
    ("Resume feedback for your application", "Campus Placement Cell", "placements@university.example", "Your revised resume looks good. Add a concise summary of your mailbox assistant project before the career fair.", 4, 8, 55, "CATEGORY_PERSONAL", "JOB", "Career"),
    ("Scholarship application reminder", "Student Services", "scholarships@university.example", "The scholarship application closes on November 15. Submit your transcript and statement of purpose before the deadline.", 6, 18, 10, "CATEGORY_PERSONAL", "EDUCATION", "University"),
    ("Repository milestone approved", "Leena Patel", "leena@project.example", "The dashboard milestone is approved. Please finish the accessibility review and deployment notes this week.", 8, 10, 31, "CATEGORY_PERSONAL", "PROJECTS", "Project"),
    ("Mentor catch-up this Friday", "Devika Menon", "devika@mentor.example", "Let's meet Friday at 4:00 PM to discuss your internship search and next project milestone.", 11, 20, 4, "CATEGORY_PERSONAL", "MEETINGS", "Career"),
    ("Reference letter request received", "Professor Iyer", "iyer@university.example", "I received your reference letter request and will send the signed copy by next Monday.", 18, 13, 47, "CATEGORY_PERSONAL", "EDUCATION", "University"),
    ("Internship offer discussion", "Northstar Recruiting", "recruiting@northstar.example", "We would like to discuss the internship role, start date, and stipend. Please choose a time from the attached schedule.", 35, 9, 8, "CATEGORY_PERSONAL", "JOB", "Career"),
    ("A friend mentioned you in a post", "Circle Social", "notify@circle.example", "A friend mentioned you in a post about the university hackathon. Open Circle to join the conversation.", 0, 12, 26, "CATEGORY_SOCIAL", "SOCIAL", "Community"),
    ("Your professional profile was viewed", "CareerNet", "alerts@careernet.example", "A recruiter viewed your professional profile this week. Add your latest project to keep it current.", 2, 17, 38, "CATEGORY_SOCIAL", "SOCIAL", "Career"),
    ("New members in the alumni group", "Alumni Network", "updates@alumni.example", "Five new members joined your university alumni group. Introduce yourself in the discussion thread.", 5, 7, 49, "CATEGORY_SOCIAL", "SOCIAL", "Community"),
    ("Hackathon team invitation", "Build Club", "hello@buildclub.example", "Your classmates invited you to join their hackathon team. The kickoff call is Saturday at noon.", 9, 15, 17, "CATEGORY_SOCIAL", "EVENTS", "Community"),
    ("Community meetup photos", "City Developers", "photos@devcommunity.example", "Photos from last week's developer meetup are ready. Thanks for joining the discussion on accessible design.", 22, 19, 3, "CATEGORY_SOCIAL", "SOCIAL", "Community"),
    ("Student laptop sale this weekend", "Campus Store", "offers@campusstore.example", "Save on student laptops and accessories this weekend. The offer ends Sunday evening.", 1, 6, 35, "CATEGORY_PROMOTIONS", "PROMOTIONS", "Offers"),
    ("Cloud learning credits available", "Cloud Academy", "news@cloudacademy.example", "Explore free cloud labs and a limited student credit offer for new learners.", 4, 12, 12, "CATEGORY_PROMOTIONS", "PROMOTIONS", "Offers"),
    ("October developer newsletter", "Dev Weekly", "digest@devweekly.example", "This month's digest covers API design, testing tips, and deployment stories from open source teams.", 7, 8, 24, "CATEGORY_PROMOTIONS", "NEWSLETTERS", "Reading"),
    ("Conference early-bird registration", "Tech Summit", "events@techsummit.example", "Early-bird registration for the technology summit closes this week. Student passes are available.", 14, 16, 9, "CATEGORY_PROMOTIONS", "EVENTS", "Offers"),
    ("New AI learning path", "Learn Lab", "hello@learnlab.example", "Our new learning path covers retrieval, evaluation, and responsible AI application design.", 39, 11, 52, "CATEGORY_PROMOTIONS", "AI_DEVELOPMENT", "Reading"),
    ("Electricity bill due on the 28th", "City Utilities", "billing@cityutilities.example", "Your sample electricity bill is 1,250 INR and is due on the 28th. This is fictional demo data; no payment is required.", 0, 18, 41, "CATEGORY_UPDATES", "BILLS", "Finance"),
    ("Tuition payment receipt", "University Payments", "receipts@university.example", "We recorded a sample tuition payment of 12,000 INR. This fictional receipt does not represent a real transaction.", 3, 10, 56, "CATEGORY_UPDATES", "RECEIPTS", "Finance"),
    ("New sign-in alert", "Account Security", "security@accounts.example", "A new sign-in to your sample account was recorded. Review your account settings if you did not expect this activity.", 6, 23, 14, "CATEGORY_UPDATES", "SECURITY", "Accounts"),
    ("Train booking confirmation", "Rail Tickets", "tickets@rail.example", "Your fictional train booking to Mumbai is confirmed for next Saturday. Departure is scheduled for 7:15 AM.", 10, 5, 47, "CATEGORY_UPDATES", "TRAVEL", "Travel"),
    ("Package is out for delivery", "Parcel Desk", "tracking@parcel.example", "Your sample order is out for delivery today. The expected arrival window is 2:00 PM to 6:00 PM.", 16, 13, 28, "CATEGORY_UPDATES", "SHIPPING", "Orders"),
    ("Monthly account statement ready", "Example Bank", "statements@examplebank.example", "Your sample monthly account statement is ready to view. All balances shown in this demo are fictional.", 44, 9, 44, "CATEGORY_UPDATES", "FINANCE", "Finance"),
    ("Question about database migrations", "Backend Forum", "reply@backendforum.example", "A member replied to your question about safe database migrations and version tracking.", 1, 21, 7, "CATEGORY_FORUMS", "TECHNICAL", "Project"),
    ("Replies to the accessibility thread", "Design Forum", "threads@designforum.example", "Three members shared suggestions for keyboard navigation and clearer contrast in the dashboard.", 5, 13, 33, "CATEGORY_FORUMS", "PROJECTS", "Project"),
    ("Weekly machine learning discussion", "ML Study Group", "group@mlstudy.example", "This week's study thread compares search ranking methods and evaluation datasets.", 13, 18, 59, "CATEGORY_FORUMS", "AI_DEVELOPMENT", "Reading"),
    ("Open source issue discussion", "Code Community", "discuss@codecommunity.example", "Maintainers replied to your issue with suggestions for an API test and a documentation update.", 28, 7, 16, "CATEGORY_FORUMS", "TECHNICAL", "Community"),
)


def seed_demo_mailbox(db: Session) -> User:
    """Create or complete the one public, read-only demo mailbox."""
    user = db.scalar(select(User).where(User.email == DEMO_EMAIL))
    if user is not None and (user.google_id != DEMO_GOOGLE_ID or not user.is_demo):
        raise RuntimeError("demo@gmail.com already belongs to a non-demo user")
    if user is None:
        user = User(google_id=DEMO_GOOGLE_ID, email=DEMO_EMAIL,
                    name="Demo Account", is_demo=True)
        db.add(user)
        db.flush()

    account = db.scalar(select(GmailAccount).where(GmailAccount.user_id == user.id))
    if account is None:
        account = GmailAccount(user_id=user.id, email=DEMO_EMAIL, provider="demo",
                               access_token="", ai_classification_enabled=False)
        db.add(account)
        db.flush()
    elif account.provider != "demo":
        raise RuntimeError("demo@gmail.com has a real Gmail connection")

    specs = [("INBOX", "INBOX", "system"), ("IMPORTANT", "IMPORTANT", "system"),
             ("STARRED", "STARRED", "system")]
    specs += [(key, name, "system") for key, name in GMAIL_CATEGORIES]
    specs += [(f"AI_{name}", name, "ai") for name in AI_DESCRIPTIONS]
    personal_labels = {row[9] for row in SAMPLE_MAIL}
    specs += [(f"DEMO_{name.upper()}", name, "user") for name in sorted(personal_labels)]
    labels = {row.label_id: row for row in db.scalars(
        select(Label).where(Label.gmail_account_id == account.id)).all()}
    for key, name, kind in specs:
        if key not in labels:
            labels[key] = Label(gmail_account_id=account.id, label_id=key,
                                name=name, type=kind)
            db.add(labels[key])
    db.flush()

    categories = {row.name: row for row in db.scalars(select(AiCategory)).all()}
    for name, description in AI_DESCRIPTIONS.items():
        if name not in categories:
            categories[name] = AiCategory(name=name, description=description)
            db.add(categories[name])
    db.flush()

    existing = {row.gmail_message_id: row for row in db.scalars(
        select(Email).where(Email.gmail_account_id == account.id)).all()}
    links = set(db.execute(select(EmailLabel.email_id, EmailLabel.label_id)
                           .where(EmailLabel.gmail_account_id == account.id)).all())
    now = datetime.now(timezone.utc)
    for index, (subject, sender_name, sender, body, days, hour, minute,
                gmail_category, ai_category, personal_label) in enumerate(SAMPLE_MAIL, 1):
        message_id = f"synthetic-demo-{index:03d}"
        email = existing.get(message_id)
        if email is None:
            received_at = (now - timedelta(days=days)).replace(hour=hour, minute=minute,
                                                                 second=0, microsecond=0)
            if received_at > now:
                received_at -= timedelta(days=1)
            email = Email(gmail_account_id=account.id, gmail_message_id=message_id,
                          thread_id=f"synthetic-thread-{index:03d}",
                          from_email=sender, sender_name=sender_name, to_email=DEMO_EMAIL,
                          subject=subject, snippet=body[:160], body=body,
                          received_at=received_at, is_read=index % 4 != 1,
                          is_starred=index in {1, 3, 6, 10, 21})
            db.add(email)
            db.flush()
        elif email.received_at and email.received_at > now:
            email.received_at -= timedelta(days=1)
        label_keys = {"INBOX", gmail_category, f"AI_{ai_category}",
                      f"DEMO_{personal_label.upper()}"}
        if index in {1, 3, 6, 10, 21, 23}:
            label_keys.add("IMPORTANT")
        if email.is_starred:
            label_keys.add("STARRED")
        for key in label_keys:
            pair = (email.id, labels[key].id)
            if pair not in links:
                db.add(EmailLabel(email_id=email.id, label_id=labels[key].id,
                                  gmail_account_id=account.id))
                links.add(pair)
        existing_classification = db.scalar(select(EmailClassification).where(
                EmailClassification.email_id == email.id,
                EmailClassification.is_current.is_(True)))
        if existing_classification is None:
            db.add(EmailClassification(email_id=email.id,
                                       category_id=categories[ai_category].id,
                                       model_name="synthetic-demo",
                                       classification_status="SUCCEEDED",
                                       classification_reason="Assigned by the demo fixture",
                                       is_current=True, classified_at=now))
        elif existing_classification.model_name == "synthetic-demo":
            existing_classification.confidence_score = None
    db.commit()
    db.refresh(user)
    return user
