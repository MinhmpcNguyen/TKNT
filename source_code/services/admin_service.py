from __future__ import annotations

from db.models import UserAccount, UserId, UserRole
from db.pg_user_content import PostgresUserContentRepository
from db.runtime_init import ensure_user_content_schema
from services.auth_service import AuthService


class AdminOperationError(ValueError):
    pass


class AdminService:
    def __init__(
        self,
        *,
        repository: PostgresUserContentRepository | None = None,
        auth_service: AuthService | None = None,
    ) -> None:
        ensure_user_content_schema()
        self._repository = repository or PostgresUserContentRepository()
        self._auth_service = auth_service or AuthService(repository=self._repository)

    def list_users(self) -> list[UserAccount]:
        return list(self._repository.list_users())

    def get_user(self, user_id: str) -> UserAccount | None:
        return self._repository.get_user_by_id(UserId(user_id))

    def create_user(
        self,
        *,
        email: str,
        password: str,
        display_name: str | None = None,
        role: UserRole = "user",
    ) -> UserAccount:
        return self._auth_service.create_user(
            email=email,
            password=password,
            display_name=display_name,
            role=role,
        )

    def update_user(
        self,
        *,
        actor: UserAccount,
        user_id: str,
        display_name: str | None,
        role: UserRole,
        is_active: bool,
    ) -> UserAccount | None:
        target = self._repository.get_user_by_id(UserId(user_id))
        if target is None:
            return None
        if target.id == actor.id and (role != target.role or not is_active):
            raise AdminOperationError(
                "Admins cannot deactivate or demote their own account."
            )
        updated = self._repository.update_user_account(
            user_id=target.id,
            display_name=display_name,
            role=role,
            is_active=is_active,
        )
        if updated is not None and not updated.is_active:
            self._repository.revoke_user_sessions(updated.id)
        return updated
