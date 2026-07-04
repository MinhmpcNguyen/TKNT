from __future__ import annotations

from db.models import BillingSettings
from db.pg_billing import PostgresBillingSettingsRepository
from db.runtime_init import ensure_user_content_schema

DEFAULT_PER_CREDIT_PRICE_VND = 5000
DEFAULT_MONTHLY_PRICE_VND = 299000
DEFAULT_MONTHLY_CREDIT_QUOTA = 100


class BillingService:
    def __init__(
        self,
        repository: PostgresBillingSettingsRepository | None = None,
    ) -> None:
        ensure_user_content_schema()
        self._repository = repository or PostgresBillingSettingsRepository()

    def get_settings(self) -> BillingSettings:
        settings = self._repository.get_settings()
        if settings is not None:
            return settings
        return self._repository.upsert_settings(_default_settings())

    def update_settings(
        self,
        *,
        per_credit_price_vnd: int,
        monthly_price_vnd: int,
        monthly_credit_quota: int,
    ) -> BillingSettings:
        return self._repository.upsert_settings(
            BillingSettings(
                per_credit_price_vnd=per_credit_price_vnd,
                monthly_price_vnd=monthly_price_vnd,
                monthly_credit_quota=monthly_credit_quota,
            )
        )


def _default_settings() -> BillingSettings:
    return BillingSettings(
        per_credit_price_vnd=DEFAULT_PER_CREDIT_PRICE_VND,
        monthly_price_vnd=DEFAULT_MONTHLY_PRICE_VND,
        monthly_credit_quota=DEFAULT_MONTHLY_CREDIT_QUOTA,
    )
