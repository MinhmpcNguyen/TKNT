from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import services.admin_service as admin_service_module  # noqa: E402
import services.auth_service as auth_service_module  # noqa: E402
import services.billing_service as billing_service_module  # noqa: E402
import services.user_content_service as user_content_service_module  # noqa: E402
from db.models import (  # noqa: E402
    Asset,
    AssetFile,
    AssetFilter,
    AssetId,
    AuthSession,
    BillingSettings,
    SessionId,
    TenantId,
    UserAccount,
    UserAuthRecord,
    UserId,
    UserRole,
)
from services.admin_service import AdminOperationError, AdminService  # noqa: E402
from services.auth_service import AuthService, AuthenticationError  # noqa: E402
from services.billing_service import BillingService  # noqa: E402
from services.user_content_service import (  # noqa: E402
    UploadedAssetContent,
    UserContentService,
)


class FakeUserRepository:
    def __init__(self) -> None:
        self.auth_by_email: dict[str, UserAuthRecord] = {}
        self.users: dict[UserId, UserAccount] = {}
        self.sessions: dict[SessionId, AuthSession] = {}

    def create_user(self, record: UserAuthRecord) -> None:
        self.auth_by_email[record.user.email.lower()] = record
        self.users[record.user.id] = record.user

    def get_user_auth_by_email(self, email: str) -> UserAuthRecord | None:
        return self.auth_by_email.get(email.lower())

    def get_user_by_id(self, user_id: UserId) -> UserAccount | None:
        return self.users.get(user_id)

    def list_users(self) -> list[UserAccount]:
        return list(self.users.values())

    def update_user_account(
        self,
        *,
        user_id: UserId,
        display_name: str | None,
        role: UserRole,
        is_active: bool,
    ) -> UserAccount | None:
        user = self.users.get(user_id)
        if user is None:
            return None
        updated = user.model_copy(
            update={
                "display_name": display_name,
                "role": role,
                "is_active": is_active,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self.users[user_id] = updated
        record = self.auth_by_email.get(updated.email.lower())
        if record is not None:
            self.auth_by_email[updated.email.lower()] = record.model_copy(
                update={"user": updated}
            )
        return updated

    def create_session(self, session: AuthSession) -> None:
        self.sessions[session.id] = session

    def get_session_by_token_hash(self, token_hash: str) -> AuthSession | None:
        return next(
            (
                session
                for session in self.sessions.values()
                if session.token_hash == token_hash
            ),
            None,
        )

    def revoke_session(self, session_id: SessionId) -> None:
        session = self.sessions[session_id]
        self.sessions[session_id] = session.model_copy(
            update={"revoked_at": datetime.now(timezone.utc)}
        )

    def revoke_user_sessions(self, user_id: UserId) -> None:
        for session_id, session in list(self.sessions.items()):
            if session.user_id == user_id and session.revoked_at is None:
                self.sessions[session_id] = session.model_copy(
                    update={"revoked_at": datetime.now(timezone.utc)}
                )

    def touch_session(self, session_id: SessionId) -> None:
        session = self.sessions[session_id]
        self.sessions[session_id] = session.model_copy(
            update={"last_seen_at": datetime.now(timezone.utc)}
        )


class FakeBillingRepository:
    def __init__(self) -> None:
        self.settings: BillingSettings | None = None

    def get_settings(self) -> BillingSettings | None:
        return self.settings

    def upsert_settings(self, settings: BillingSettings) -> BillingSettings:
        self.settings = settings.model_copy(
            update={"updated_at": datetime.now(timezone.utc)}
        )
        return self.settings


class FakeAssetRepository:
    def __init__(self) -> None:
        self.assets: dict[AssetId, Asset] = {}
        self.files: dict[AssetId, list[AssetFile]] = {}

    def upsert_asset(self, asset: Asset) -> None:
        self.assets[asset.id] = asset

    def upsert_asset_file(self, asset_file: AssetFile) -> None:
        self.files.setdefault(asset_file.asset_id, []).append(asset_file)

    def list_assets(self, asset_filter: AssetFilter) -> list[Asset]:
        return [
            asset
            for asset in self.assets.values()
            if asset.tenant_id == asset_filter.tenant_id
            and (asset_filter.type is None or asset.type == asset_filter.type)
        ]

    def list_asset_files(self, asset_id: AssetId) -> list[AssetFile]:
        return list(self.files.get(asset_id, []))


class AdminRoleManagementTest(unittest.TestCase):
    def setUp(self) -> None:
        self.original_admin_emails = os.environ.get("TKNT_ADMIN_EMAILS")
        self.original_shared_tenant = os.environ.get("TKNT_SHARED_INVENTORY_TENANT_ID")
        self.original_auth_schema_init = auth_service_module.ensure_user_content_schema
        self.original_admin_schema_init = admin_service_module.ensure_user_content_schema
        self.original_billing_schema_init = (
            billing_service_module.ensure_user_content_schema
        )
        self.original_content_schema_init = (
            user_content_service_module.ensure_user_content_schema
        )
        auth_service_module.ensure_user_content_schema = lambda: None
        admin_service_module.ensure_user_content_schema = lambda: None
        billing_service_module.ensure_user_content_schema = lambda: None
        user_content_service_module.ensure_user_content_schema = lambda: None

    def tearDown(self) -> None:
        if self.original_admin_emails is None:
            os.environ.pop("TKNT_ADMIN_EMAILS", None)
        else:
            os.environ["TKNT_ADMIN_EMAILS"] = self.original_admin_emails
        if self.original_shared_tenant is None:
            os.environ.pop("TKNT_SHARED_INVENTORY_TENANT_ID", None)
        else:
            os.environ["TKNT_SHARED_INVENTORY_TENANT_ID"] = (
                self.original_shared_tenant
            )
        auth_service_module.ensure_user_content_schema = (
            self.original_auth_schema_init
        )
        admin_service_module.ensure_user_content_schema = (
            self.original_admin_schema_init
        )
        billing_service_module.ensure_user_content_schema = (
            self.original_billing_schema_init
        )
        user_content_service_module.ensure_user_content_schema = (
            self.original_content_schema_init
        )

    def test_allowlisted_email_registers_as_admin(self) -> None:
        os.environ["TKNT_ADMIN_EMAILS"] = "admin@example.com"
        repo = FakeUserRepository()
        service = AuthService(repository=repo)

        session = service.register_user(
            email="Admin@Example.com",
            password="password123",
            display_name="Admin",
        )

        self.assertEqual(session.user.role, "admin")
        self.assertTrue(session.user.is_active)

    def test_inactive_user_cannot_login_or_use_existing_session(self) -> None:
        repo = FakeUserRepository()
        service = AuthService(repository=repo)
        session = service.register_user(
            email="user@example.com",
            password="password123",
        )
        repo.update_user_account(
            user_id=session.user.id,
            display_name=session.user.display_name,
            role="user",
            is_active=False,
        )

        with self.assertRaises(AuthenticationError):
            service.login_user(email="user@example.com", password="password123")
        with self.assertRaises(AuthenticationError):
            service.get_authenticated_session(token=session.token)

    def test_deactivate_user_revokes_sessions(self) -> None:
        repo = FakeUserRepository()
        auth_service = AuthService(repository=repo)
        admin = auth_service.create_user(
            email="admin@example.com",
            password="password123",
            role="admin",
        )
        user_session = auth_service.register_user(
            email="user@example.com",
            password="password123",
        )
        service = AdminService(repository=repo, auth_service=auth_service)

        updated = service.update_user(
            actor=admin,
            user_id=str(user_session.user.id),
            display_name=user_session.user.display_name,
            role="user",
            is_active=False,
        )

        self.assertIsNotNone(updated)
        self.assertFalse(updated.is_active if updated is not None else True)
        self.assertTrue(
            all(
                session.revoked_at is not None
                for session in repo.sessions.values()
                if session.user_id == user_session.user.id
            )
        )

    def test_admin_cannot_demote_self(self) -> None:
        repo = FakeUserRepository()
        auth_service = AuthService(repository=repo)
        admin = auth_service.create_user(
            email="admin@example.com",
            password="password123",
            role="admin",
        )
        service = AdminService(repository=repo, auth_service=auth_service)

        with self.assertRaises(AdminOperationError):
            service.update_user(
                actor=admin,
                user_id=str(admin.id),
                display_name=admin.display_name,
                role="user",
                is_active=True,
            )

    def test_billing_settings_can_be_updated(self) -> None:
        service = BillingService(repository=FakeBillingRepository())

        settings = service.update_settings(
            per_credit_price_vnd=7000,
            monthly_price_vnd=399000,
            monthly_credit_quota=120,
        )

        self.assertEqual(settings.per_credit_price_vnd, 7000)
        self.assertEqual(settings.monthly_price_vnd, 399000)
        self.assertEqual(settings.monthly_credit_quota, 120)

    def test_admin_furniture_creation_serializes_price_and_model_url(self) -> None:
        os.environ["TKNT_SHARED_INVENTORY_TENANT_ID"] = "shared"
        asset_repo = FakeAssetRepository()
        service = UserContentService(
            asset_repository=asset_repo,
            user_repository=FakeUserRepository(),
        )

        persisted = service.create_shared_furniture_asset(
            name="Walnut Chair",
            category="chair",
            style_tags=["modern", "modern"],
            length_mm=700,
            width_mm=640,
            height_mm=820,
            material="walnut",
            price_amount=1500000,
            attributes={"color_hex": "#805a3b"},
            model_reference="https://example.com/chair.glb",
        )
        payload = service.serialize_inventory_asset(
            persisted=persisted,
            file_url_builder=lambda asset_file: f"/inventory/files/{asset_file.id}",
        )

        self.assertEqual(persisted.asset.tenant_id, TenantId("shared"))
        self.assertEqual(payload["type"], "chair")
        self.assertEqual(payload["style_tags"], ["modern"])
        self.assertEqual(payload["price"], {"amount": 1500000.0, "currency": "VND"})
        attributes = payload["attributes"]
        self.assertIsInstance(attributes, dict)
        self.assertEqual(
            attributes.get("model_url") if isinstance(attributes, dict) else None,
            "https://example.com/chair.glb",
        )

    def test_admin_furniture_creation_can_store_uploaded_glb_in_database(self) -> None:
        os.environ["TKNT_SHARED_INVENTORY_TENANT_ID"] = "shared"
        glb_bytes = b"glTF\x02\x00\x00\x00\x14\x00\x00\x00data"
        asset_repo = FakeAssetRepository()
        service = UserContentService(
            asset_repository=asset_repo,
            user_repository=FakeUserRepository(),
        )

        persisted = service.create_shared_furniture_asset(
            name="Upload Chair",
            category="chair",
            style_tags=["japandi"],
            model_upload=UploadedAssetContent(
                filename="upload-chair.glb",
                content=glb_bytes,
                mime_type="model/gltf-binary",
            ),
        )
        payload = service.serialize_inventory_asset(
            persisted=persisted,
            file_url_builder=lambda asset_file: f"/inventory/files/{asset_file.id}",
        )

        self.assertEqual(len(persisted.files), 1)
        model_file = persisted.files[0]
        self.assertEqual(model_file.provider, "database")
        self.assertEqual(model_file.mime, "model/gltf-binary")
        self.assertEqual(model_file.bytes_size, len(glb_bytes))
        self.assertEqual(model_file.content_bytes, glb_bytes)
        self.assertTrue(model_file.storage_key.startswith("db://asset_files/"))
        attributes = payload["attributes"]
        self.assertIsInstance(attributes, dict)
        self.assertEqual(
            attributes.get("model_url") if isinstance(attributes, dict) else None,
            f"/inventory/files/{model_file.id}",
        )


if __name__ == "__main__":
    unittest.main()
