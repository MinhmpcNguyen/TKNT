from __future__ import annotations

from db.models import BillingSettings
from db.pg_utils import ConnectionFactory, RowMapping, to_optional_int
from db.postgres import create_connection

_BILLING_SETTINGS_ID = "global"


class PostgresBillingSettingsRepository:
    def __init__(self, connection_factory: ConnectionFactory | None = None) -> None:
        self._connection_factory = connection_factory or create_connection

    def get_settings(self) -> BillingSettings | None:
        query = "SELECT * FROM billing_settings WHERE id = %(settings_id)s"
        with self._connection_factory() as conn:
            with conn.cursor() as cur:
                cur.execute(query, {"settings_id": _BILLING_SETTINGS_ID})
                row = cur.fetchone()
        if row is None:
            return None
        return _row_to_billing_settings(row)

    def upsert_settings(self, settings: BillingSettings) -> BillingSettings:
        query = """
            INSERT INTO billing_settings (
                id,
                per_credit_price_vnd,
                monthly_price_vnd,
                monthly_credit_quota
            )
            VALUES (
                %(settings_id)s,
                %(per_credit_price_vnd)s,
                %(monthly_price_vnd)s,
                %(monthly_credit_quota)s
            )
            ON CONFLICT (id)
            DO UPDATE SET
                per_credit_price_vnd = EXCLUDED.per_credit_price_vnd,
                monthly_price_vnd = EXCLUDED.monthly_price_vnd,
                monthly_credit_quota = EXCLUDED.monthly_credit_quota,
                updated_at = NOW()
            RETURNING *
        """
        with self._connection_factory() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    {
                        "settings_id": _BILLING_SETTINGS_ID,
                        "per_credit_price_vnd": settings.per_credit_price_vnd,
                        "monthly_price_vnd": settings.monthly_price_vnd,
                        "monthly_credit_quota": settings.monthly_credit_quota,
                    },
                )
                row = cur.fetchone()
        if row is None:
            return settings
        return _row_to_billing_settings(row)


def _row_to_billing_settings(row: RowMapping) -> BillingSettings:
    per_credit_price_vnd = to_optional_int(row.get("per_credit_price_vnd"))
    monthly_price_vnd = to_optional_int(row.get("monthly_price_vnd"))
    monthly_credit_quota = to_optional_int(row.get("monthly_credit_quota"))
    if (
        per_credit_price_vnd is None
        or monthly_price_vnd is None
        or monthly_credit_quota is None
    ):
        raise ValueError("Billing settings row has missing required values.")
    return BillingSettings(
        per_credit_price_vnd=per_credit_price_vnd,
        monthly_price_vnd=monthly_price_vnd,
        monthly_credit_quota=monthly_credit_quota,
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )
