import os
import base64
import re
from datetime import date, datetime, timedelta
from html import unescape

from bs4 import BeautifulSoup, Comment
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
BACKEND_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
CREDENTIALS_FILE = os.path.join(
    BACKEND_DIR,
    "credentials",
    "credentials.json",
)
TOKEN_FILE = os.path.join(BACKEND_DIR, "token.json")


# --------------------------------------------------
# Gmail Authentication
# --------------------------------------------------

def get_gmail_service():
    creds = None

    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(
            TOKEN_FILE,
            SCOPES
        )

    if not creds or not creds.valid:

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())

        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE,
                SCOPES
            )

            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())

    return build(
        "gmail",
        "v1",
        credentials=creds
    )


# --------------------------------------------------
# Decode Gmail Body
# --------------------------------------------------

def decode_body(data):
    return base64.urlsafe_b64decode(data).decode(
        "utf-8",
        errors="ignore"
    )


def clean_text(text):
    """Normalize whitespace without removing useful URL text."""
    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in unescape(text).splitlines()
    ]

    return "\n".join(
        line
        for line in lines
        if line
    )


# --------------------------------------------------
# Convert HTML → Clean Text
# --------------------------------------------------

def html_to_text(html):

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    # Remove comments and elements that never contain visible email text.
    for comment in soup.find_all(string=lambda value: isinstance(value, Comment)):
        comment.extract()

    for tag in soup([
        "script",
        "style",
        "noscript",
        "head",
        "title",
        "svg",
        "meta",
        "link"
    ]):
        tag.decompose()

    for tag in soup.find_all(
        attrs={"hidden": True}
    ) + soup.find_all(
        attrs={"aria-hidden": re.compile(r"^true$", re.I)}
    ):
        tag.decompose()

    for tag in soup.find_all(
        class_=re.compile(r"(?:^|\s)hidden(?:\s|$)", re.I)
    ) + soup.find_all(
        id=re.compile(r"(?:^|[-_])hidden(?:$|[-_])", re.I)
    ):
        tag.decompose()

    for tag in soup.find_all(style=re.compile(
        r"(?:display\s*:\s*none|visibility\s*:\s*hidden)",
        re.I
    )):
        tag.decompose()

    for link in soup.find_all("a", href=True):
        href = link.get("href", "").strip()
        visible_text = link.get_text(" ", strip=True)
        if href.lower().startswith(("http://", "https://")) and href not in visible_text:
            link.append(f" ({href})")

    # Convert HTML to readable text
    text = soup.get_text(
        separator="\n",
        strip=True
    )

    return clean_text(text)


# --------------------------------------------------
# Extract Email Body
# --------------------------------------------------

def get_email_body(payload):

    plain_text = None
    html_text = None

    def process_part(part):

        nonlocal plain_text
        nonlocal html_text

        mime_type = part.get("mimeType", "")
        data = part.get("body", {}).get("data")

        if mime_type == "text/plain" and data and plain_text is None:
            plain_text = decode_body(data)
        elif mime_type == "text/html" and data and html_text is None:
            html_text = decode_body(data)

        for nested_part in part.get("parts", []):
            process_part(nested_part)

    process_part(payload)

    # ----------------------------------------------
    # Prefer plain text
    # ----------------------------------------------

    if plain_text:

        return clean_text(plain_text)

    # ----------------------------------------------
    # Otherwise convert HTML
    # ----------------------------------------------

    if html_text:

        return html_to_text(
            html_text
        )

    return ""


# --------------------------------------------------
# Clean Subject
# --------------------------------------------------

def clean_subject(subject):

    cleaned = BeautifulSoup(
        subject,
        "html.parser"
    ).get_text(
        separator=" ",
        strip=True
    )

    return clean_text(cleaned)


# --------------------------------------------------
# Get Latest Email
# --------------------------------------------------

def _gmail_date(value, field_name):
    if value is None:
        return None

    try:
        return date.fromisoformat(value) if isinstance(value, str) else value
    except ValueError as error:
        raise ValueError(
            f"{field_name} must use YYYY-MM-DD format"
        ) from error


def extract_email(message):
    payload = message.get("payload", {})
    headers = {
        header.get("name", "").lower(): header.get("value", "")
        for header in payload.get("headers", [])
    }

    return {
        "id": message.get("id", ""),
        "labelIds": message.get("labelIds", []),
        "from": headers.get("from", ""),
        "subject": clean_subject(headers.get("subject", "")),
        "date": headers.get("date", ""),
        "body": get_email_body(payload),
    }


def get_emails(count=1, start_date=None, end_date=None, service=None):
    """Fetch up to count inbox emails in an optional inclusive date range."""
    if not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")

    start_date = _gmail_date(start_date, "start_date")
    end_date = _gmail_date(end_date, "end_date")
    if start_date and end_date and start_date > end_date:
        raise ValueError("start_date cannot be later than end_date")

    query_parts = []
    if start_date:
        query_parts.append(f"after:{start_date.isoformat()}")
    if end_date:
        query_parts.append(
            f"before:{(end_date + timedelta(days=1)).isoformat()}"
        )

    service = service or get_gmail_service()
    result = service.users().messages().list(
        userId="me",
        labelIds=["INBOX"],
        maxResults=count,
        q=" ".join(query_parts),
    ).execute()

    emails = []
    for message_ref in result.get("messages", [])[:count]:
        message = service.users().messages().get(
            userId="me",
            id=message_ref["id"],
            format="full",
        ).execute()
        emails.append(extract_email(message))

    return emails


def get_latest_email():

    emails = get_emails(count=1)

    if not emails:

        print("No emails found.")

        return

    email = emails[0]

    # --------------------------------------------------
    # Print Result
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("EMAIL")
    print("=" * 60)

    print("ID:")
    print(email["id"])

    print("\nFROM:")
    print(email["from"])

    print("\nSUBJECT:")
    print(email["subject"])

    print("\nDATE:")
    print(email["date"])

    print("\n## BODY:")
    print("-" * 60)
    print(email["body"])
    print("-" * 60)

    print("=" * 60)


# --------------------------------------------------
# Main
# --------------------------------------------------
