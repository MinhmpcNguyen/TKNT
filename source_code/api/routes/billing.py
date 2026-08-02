from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from api.deps import get_current_admin_user
from db.models import BillingSettings, UserAccount
from services.billing_service import BillingService

router = APIRouter(tags=["billing"])


class BillingSettingsResponse(BaseModel):
    per_credit_price_vnd: int
    monthly_price_vnd: int
    monthly_credit_quota: int
    updated_at: str | None = None


class BillingSettingsUpdateRequest(BaseModel):
    per_credit_price_vnd: int = Field(..., gt=0)
    monthly_price_vnd: int = Field(..., gt=0)
    monthly_credit_quota: int = Field(..., gt=0)


def get_billing_service() -> BillingService:
    return BillingService()


@router.get("/billing/settings", response_model=BillingSettingsResponse)
def get_billing_settings(
    service: BillingService = Depends(get_billing_service),
) -> BillingSettingsResponse:
    return _serialize_billing_settings(service.get_settings())


@router.patch("/admin/billing/settings", response_model=BillingSettingsResponse)
def update_billing_settings(
    request: BillingSettingsUpdateRequest,
    _admin: UserAccount = Depends(get_current_admin_user),
    service: BillingService = Depends(get_billing_service),
) -> BillingSettingsResponse:
    settings = service.update_settings(
        per_credit_price_vnd=request.per_credit_price_vnd,
        monthly_price_vnd=request.monthly_price_vnd,
        monthly_credit_quota=request.monthly_credit_quota,
    )
    return _serialize_billing_settings(settings)


def _serialize_billing_settings(settings: BillingSettings) -> BillingSettingsResponse:
    return BillingSettingsResponse(
        per_credit_price_vnd=settings.per_credit_price_vnd,
        monthly_price_vnd=settings.monthly_price_vnd,
        monthly_credit_quota=settings.monthly_credit_quota,
        updated_at=settings.updated_at.isoformat() if settings.updated_at else None,
    )
