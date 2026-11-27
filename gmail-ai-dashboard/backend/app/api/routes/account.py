"""User-facing account preferences."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.database.connection import CentralSessionLocal
from app.models.email import User


router = APIRouter(prefix="/api/account", tags=["account"])


def central_db():
    with CentralSessionLocal() as db:
        yield db


class AccountSettingsUpdate(BaseModel):
    mail_sync_limit: int = Field(ge=1, le=50000)


def _public_settings(user: User) -> dict:
    return {"mail_sync_limit": user.mail_sync_limit}


@router.get("/settings")
def read_account_settings(
    user: User = Depends(get_current_user),
    db: Session = Depends(central_db),
):
    if user.is_demo:
        raise HTTPException(403, "Account settings are unavailable in the demo")
    owner = db.get(User, user.id)
    if owner is None:
        raise HTTPException(404, "Account not found")
    return _public_settings(owner)


@router.put("/settings")
def update_account_settings(
    body: AccountSettingsUpdate,
    user: User = Depends(get_current_user),
    db: Session = Depends(central_db),
):
    if user.is_demo:
        raise HTTPException(403, "Account settings are unavailable in the demo")
    owner = db.get(User, user.id)
    if owner is None:
        raise HTTPException(404, "Account not found")
    owner.mail_sync_limit = body.mail_sync_limit
    db.commit()
    return _public_settings(owner)
