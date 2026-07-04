from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.deps import get_current_admin_user
from db.models import AssetFile, JsonValue, UserAccount, UserRole
from services.admin_service import AdminOperationError, AdminService
from services.auth_service import AuthenticationError, get_shared_inventory_tenant_id
from services.user_content_service import UserContentService

router = APIRouter(prefix="/admin", tags=["admin"])


class AdminUserResponse(BaseModel):
    id: str
    email: str
    display_name: str | None = None
    role: UserRole
    is_active: bool
    created_at: str | None = None
    updated_at: str | None = None


class AdminUserCreateRequest(BaseModel):
    email: str
    password: str = Field(..., min_length=8)
    display_name: str | None = Field(default=None, max_length=120)
    role: UserRole = "user"


class AdminUserUpdateRequest(BaseModel):
    display_name: str | None = Field(default=None, max_length=120)
    role: UserRole | None = None
    is_active: bool | None = None


class AdminInventoryItemResponse(BaseModel):
    id: str
    name: str
    type: str
    style_tags: list[str] = Field(default_factory=list)
    material: str | None = None
    brand: str | None = None
    dimensions: dict[str, float | None] | None = None
    price: dict[str, object] | None = None
    attributes: dict[str, object] = Field(default_factory=dict)


class AdminInventoryListResponse(BaseModel):
    tenant_id: str
    items: list[AdminInventoryItemResponse]


class AdminFurnitureCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=160)
    category: str = Field(..., min_length=1, max_length=120)
    style_tags: list[str] = Field(default_factory=list)
    material: str | None = Field(default=None, max_length=120)
    brand: str | None = Field(default=None, max_length=120)
    length_mm: float | None = Field(default=None, gt=0)
    width_mm: float | None = Field(default=None, gt=0)
    height_mm: float | None = Field(default=None, gt=0)
    price_amount: float | None = Field(default=None, ge=0)
    price_currency: str = Field(default="VND", min_length=1, max_length=12)
    attributes: dict[str, JsonValue] = Field(default_factory=dict)
    model_reference: str | None = Field(default=None, max_length=1000)
    preview_reference: str | None = Field(default=None, max_length=1000)


def get_admin_service() -> AdminService:
    return AdminService()


def get_user_content_service() -> UserContentService:
    return UserContentService()


@router.get("/users", response_model=list[AdminUserResponse])
def list_users(
    _admin: UserAccount = Depends(get_current_admin_user),
    service: AdminService = Depends(get_admin_service),
) -> list[AdminUserResponse]:
    return [_serialize_user(user) for user in service.list_users()]


@router.post("/users", response_model=AdminUserResponse)
def create_user(
    request: AdminUserCreateRequest,
    _admin: UserAccount = Depends(get_current_admin_user),
    service: AdminService = Depends(get_admin_service),
) -> AdminUserResponse:
    try:
        user = service.create_user(
            email=request.email,
            password=request.password,
            display_name=request.display_name,
            role=request.role,
        )
    except AuthenticationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _serialize_user(user)


@router.patch("/users/{user_id}", response_model=AdminUserResponse)
def update_user(
    user_id: str,
    request: AdminUserUpdateRequest,
    admin: UserAccount = Depends(get_current_admin_user),
    service: AdminService = Depends(get_admin_service),
) -> AdminUserResponse:
    target = service.get_user(user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="User not found.")

    display_name = target.display_name
    if "display_name" in request.model_fields_set:
        display_name = request.display_name.strip() if request.display_name else None
    role = request.role or target.role
    is_active = target.is_active if request.is_active is None else request.is_active

    try:
        updated = service.update_user(
            actor=admin,
            user_id=user_id,
            display_name=display_name,
            role=role,
            is_active=is_active,
        )
    except AdminOperationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if updated is None:
        raise HTTPException(status_code=404, detail="User not found.")
    return _serialize_user(updated)


@router.get("/inventory/items", response_model=AdminInventoryListResponse)
def list_inventory_items(
    _admin: UserAccount = Depends(get_current_admin_user),
    service: UserContentService = Depends(get_user_content_service),
) -> AdminInventoryListResponse:
    items = [
        AdminInventoryItemResponse(
            **service.serialize_inventory_asset(
                persisted=persisted,
                file_url_builder=_build_inventory_file_url,
            )
        )
        for persisted in service.list_shared_inventory_assets()
    ]
    return AdminInventoryListResponse(
        tenant_id=get_shared_inventory_tenant_id(),
        items=items,
    )


@router.post("/inventory/items", response_model=AdminInventoryItemResponse)
def create_inventory_item(
    request: AdminFurnitureCreateRequest,
    _admin: UserAccount = Depends(get_current_admin_user),
    service: UserContentService = Depends(get_user_content_service),
) -> AdminInventoryItemResponse:
    try:
        persisted = service.create_shared_furniture_asset(
            name=request.name,
            category=request.category,
            style_tags=request.style_tags,
            length_mm=request.length_mm,
            width_mm=request.width_mm,
            height_mm=request.height_mm,
            material=request.material,
            brand=request.brand,
            price_amount=request.price_amount,
            price_currency=request.price_currency,
            attributes=request.attributes,
            model_reference=request.model_reference,
            preview_reference=request.preview_reference,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AdminInventoryItemResponse(
        **service.serialize_inventory_asset(
            persisted=persisted,
            file_url_builder=_build_inventory_file_url,
        )
    )


def _serialize_user(user: UserAccount) -> AdminUserResponse:
    return AdminUserResponse(
        id=str(user.id),
        email=user.email,
        display_name=user.display_name,
        role=user.role,
        is_active=user.is_active,
        created_at=user.created_at.isoformat() if user.created_at else None,
        updated_at=user.updated_at.isoformat() if user.updated_at else None,
    )


def _build_inventory_file_url(asset_file: AssetFile) -> str:
    return f"/inventory/files/{asset_file.id}"
