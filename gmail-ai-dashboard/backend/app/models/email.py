from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
	pass


class User(Base):
	__tablename__ = "users"

	id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
	google_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
	email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
	name: Mapped[str | None] = mapped_column(String(255))
	picture: Mapped[str | None] = mapped_column(String(500))
	created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
	updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

	gmail_accounts: Mapped[list["GmailAccount"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class GmailAccount(Base):
	__tablename__ = "gmail_accounts"

	id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
	user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
	email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
	access_token: Mapped[str] = mapped_column(Text, nullable=False)
	refresh_token: Mapped[str | None] = mapped_column(Text)
	token_expiry: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
	scope: Mapped[str | None] = mapped_column(Text)
	created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
	updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

	user: Mapped[User] = relationship(back_populates="gmail_accounts")
	emails: Mapped[list["Email"]] = relationship(back_populates="gmail_account", cascade="all, delete-orphan")
	labels: Mapped[list["Label"]] = relationship(back_populates="gmail_account", cascade="all, delete-orphan")


class Email(Base):
	__tablename__ = "emails"

	id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
	gmail_account_id: Mapped[UUID] = mapped_column(ForeignKey("gmail_accounts.id", ondelete="CASCADE"), nullable=False)
	gmail_message_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
	thread_id: Mapped[str | None] = mapped_column(String(255))
	subject: Mapped[str | None] = mapped_column(Text)
	body: Mapped[str | None] = mapped_column(Text)
	snippet: Mapped[str | None] = mapped_column(Text)
	from_email: Mapped[str | None] = mapped_column(String(255))
	to_email: Mapped[str | None] = mapped_column(String(255))
	received_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
	is_read: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
	has_attachment: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
	cc: Mapped[str | None] = mapped_column(Text)
	bcc: Mapped[str | None] = mapped_column(Text)
	attachments_meta: Mapped[str | None] = mapped_column(Text)
	size_estimate: Mapped[int | None] = mapped_column(Integer)
	history_id: Mapped[str | None] = mapped_column(String(255))
	created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

	gmail_account: Mapped[GmailAccount] = relationship(back_populates="emails")
	labels: Mapped[list["EmailLabel"]] = relationship(back_populates="email", cascade="all, delete-orphan")
	classifications: Mapped[list["EmailClassification"]] = relationship(back_populates="email", cascade="all, delete-orphan")


class Label(Base):
	__tablename__ = "labels"
	__table_args__ = (UniqueConstraint("gmail_account_id", "label_id"),)

	id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
	gmail_account_id: Mapped[UUID] = mapped_column(ForeignKey("gmail_accounts.id", ondelete="CASCADE"), nullable=False)
	label_id: Mapped[str] = mapped_column(String(255), nullable=False)
	name: Mapped[str] = mapped_column(String(255), nullable=False)
	type: Mapped[str | None] = mapped_column(String(50))
	created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

	gmail_account: Mapped[GmailAccount] = relationship(back_populates="labels")
	emails: Mapped[list["EmailLabel"]] = relationship(back_populates="label", cascade="all, delete-orphan")


class EmailLabel(Base):
	__tablename__ = "email_labels"
	__table_args__ = (UniqueConstraint("email_id", "label_id"),)

	id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
	email_id: Mapped[UUID] = mapped_column(ForeignKey("emails.id", ondelete="CASCADE"), nullable=False)
	label_id: Mapped[UUID] = mapped_column(ForeignKey("labels.id", ondelete="CASCADE"), nullable=False)
	created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

	email: Mapped[Email] = relationship(back_populates="labels")
	label: Mapped[Label] = relationship(back_populates="emails")


class EmailClassification(Base):
	__tablename__ = "email_classifications"

	id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
	email_id: Mapped[UUID] = mapped_column(ForeignKey("emails.id", ondelete="CASCADE"), nullable=False)
	label: Mapped[str] = mapped_column(String(100), nullable=False)
	confidence: Mapped[float | None] = mapped_column(Float)
	model_version: Mapped[str | None] = mapped_column(String(50))
	created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

	email: Mapped[Email] = relationship(back_populates="classifications")
