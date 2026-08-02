from __future__ import annotations

import json
import os

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from api.deps import get_current_admin_user
from db.models import AssetFile, JsonValue, UserAccount, UserRole
from services.admin_service import AdminOperationError, AdminService
from services.auth_service import AuthenticationError, get_shared_inventory_tenant_id
from services.user_content_service import UploadedAssetContent, UserContentService

router = APIRouter(prefix="/admin", tags=["admin"])
_ADMIN_GLB_UPLOAD_MAX_BYTES_ENV = "TKNT_ADMIN_GLB_UPLOAD_MAX_BYTES"
_DEFAULT_ADMIN_GLB_UPLOAD_MAX_BYTES = 64 * 1024 * 1024


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


@router.post("/inventory/items/upload", response_model=AdminInventoryItemResponse)
async def create_inventory_item_with_upload(
    name: str = Form(..., min_length=1, max_length=160),
    category: str = Form(..., min_length=1, max_length=120),
    style_tags: str = Form(default=""),
    material: str | None = Form(default=None, max_length=120),
    brand: str | None = Form(default=None, max_length=120),
    length_mm: str | None = Form(default=None),
    width_mm: str | None = Form(default=None),
    height_mm: str | None = Form(default=None),
    price_amount: str | None = Form(default=None),
    price_currency: str = Form(default="VND", min_length=1, max_length=12),
    attributes_json: str = Form(default="{}"),
    preview_reference: str | None = Form(default=None, max_length=1000),
    model_file: UploadFile = File(...),
    _admin: UserAccount = Depends(get_current_admin_user),
    service: UserContentService = Depends(get_user_content_service),
) -> AdminInventoryItemResponse:
    max_upload_bytes = _admin_glb_upload_max_bytes()
    content = await model_file.read(max_upload_bytes + 1)
    await model_file.close()
    if len(content) > max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"GLB upload is too large. Maximum size is {max_upload_bytes} bytes.",
        )
    try:
        persisted = service.create_shared_furniture_asset(
            name=name,
            category=category,
            style_tags=_parse_form_tags(style_tags),
            length_mm=_parse_positive_float_form(length_mm, "length_mm"),
            width_mm=_parse_positive_float_form(width_mm, "width_mm"),
            height_mm=_parse_positive_float_form(height_mm, "height_mm"),
            material=_clean_form_optional_text(material),
            brand=_clean_form_optional_text(brand),
            price_amount=_parse_non_negative_float_form(
                price_amount,
                "price_amount",
            ),
            price_currency=price_currency,
            attributes=_parse_attributes_json(attributes_json),
            preview_reference=_clean_form_optional_text(preview_reference),
            model_upload=UploadedAssetContent(
                filename=model_file.filename or "model.glb",
                content=content,
                mime_type=model_file.content_type,
            ),
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


def _admin_glb_upload_max_bytes() -> int:
    raw_value = os.getenv(_ADMIN_GLB_UPLOAD_MAX_BYTES_ENV, "").strip()
    if not raw_value:
        return _DEFAULT_ADMIN_GLB_UPLOAD_MAX_BYTES
    try:
        parsed = int(raw_value)
    except ValueError:
        return _DEFAULT_ADMIN_GLB_UPLOAD_MAX_BYTES
    return parsed if parsed > 0 else _DEFAULT_ADMIN_GLB_UPLOAD_MAX_BYTES


def _parse_form_tags(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _parse_positive_float_form(raw_value: str | None, label: str) -> float | None:
    value = _parse_optional_float_form(raw_value, label)
    if value is not None and value <= 0:
        raise ValueError(f"{label} must be positive.")
    return value


def _parse_non_negative_float_form(raw_value: str | None, label: str) -> float | None:
    value = _parse_optional_float_form(raw_value, label)
    if value is not None and value < 0:
        raise ValueError(f"{label} must be >= 0.")
    return value


def _parse_optional_float_form(raw_value: str | None, label: str) -> float | None:
    if raw_value is None:
        return None
    text = raw_value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be a number.") from exc


def _clean_form_optional_text(raw_value: str | None) -> str | None:
    if raw_value is None:
        return None
    text = raw_value.strip()
    return text or None


def _parse_attributes_json(raw_value: str) -> dict[str, JsonValue]:
    text = raw_value.strip()
    if not text:
        return {}
    try:
        parsed: object = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("attributes_json must be valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("attributes_json must be a JSON object.")
    out: dict[str, JsonValue] = {}
    for key, value in parsed.items():
        if not isinstance(key, str):
            raise ValueError("attributes_json keys must be strings.")
        out[key] = _parse_json_value(value)
    return out


def _parse_json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        parsed_items: list[object] = []
        for item in value:
            parsed_items.append(_parse_json_value(item))
        return parsed_items
    if isinstance(value, dict):
        parsed_dict: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("Nested attributes_json keys must be strings.")
            parsed_dict[key] = _parse_json_value(item)
        return parsed_dict
    raise ValueError("attributes_json contains a non-JSON value.")
