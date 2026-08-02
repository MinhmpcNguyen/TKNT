from __future__ import annotations

import base64
import binascii
import mimetypes
import re
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from db.models import (
    Asset,
    AssetDimensions,
    AssetFile,
    AssetFileId,
    AssetFilter,
    AssetId,
    AssetPrice,
    FileKind,
    GeneratedRender,
    GeneratedRenderId,
    JsonValue,
    SavedLayout,
    SavedLayoutCase,
    SavedLayoutCaseId,
    SavedLayoutId,
    TenantId,
    UserAccount,
    UserId,
)
from db.pg_assets import PostgresAssetRepository
from db.pg_user_content import PostgresUserContentRepository
from db.runtime_init import ensure_user_content_schema
from services.auth_service import get_shared_inventory_tenant_id


@dataclass(frozen=True)
class PersistedInventoryAsset:
    asset: Asset
    files: list[AssetFile]


@dataclass(frozen=True)
class UploadedAssetContent:
    filename: str
    content: bytes
    mime_type: str | None = None


@dataclass(frozen=True)
class LayoutOptionPersistenceRequest:
    option_id: str
    option_index: int
    name: str
    floorplan_json: dict[str, JsonValue]
    design_json: dict[str, JsonValue] | None = None
    styled_result_json: dict[str, JsonValue] | None = None
    meta: dict[str, JsonValue] | None = None


@dataclass(frozen=True)
class RenderLayoutContext:
    root_layout_id: str | None
    case_id: str | None
    option_id: str | None
    option_index: int | None


class UserContentService:
    def __init__(
        self,
        *,
        asset_repository: PostgresAssetRepository | None = None,
        user_repository: PostgresUserContentRepository | None = None,
    ) -> None:
        ensure_user_content_schema()
        self._asset_repository = asset_repository or PostgresAssetRepository()
        self._user_repository = user_repository or PostgresUserContentRepository()

    def get_user_by_id(self, user_id: str) -> UserAccount | None:
        return self._user_repository.get_user_by_id(UserId(user_id))

    def list_shared_inventory_assets(
        self,
        *,
        style_tags: list[str] | None = None,
    ) -> list[PersistedInventoryAsset]:
        resolved_shared_tenant = get_shared_inventory_tenant_id()
        shared_assets = self._asset_repository.list_assets(
            AssetFilter(
                tenant_id=TenantId(resolved_shared_tenant),
                style_tags=list(style_tags or []),
            )
        )

        persisted: list[PersistedInventoryAsset] = []
        for asset in shared_assets:
            files = list(self._asset_repository.list_asset_files(asset.id))
            persisted.append(PersistedInventoryAsset(asset=asset, files=files))
        return persisted

    def create_shared_furniture_asset(
        self,
        *,
        name: str,
        category: str,
        style_tags: list[str],
        length_mm: float | None = None,
        width_mm: float | None = None,
        height_mm: float | None = None,
        material: str | None = None,
        brand: str | None = None,
        price_amount: float | None = None,
        price_currency: str = "VND",
        attributes: dict[str, JsonValue] | None = None,
        model_reference: str | None = None,
        preview_reference: str | None = None,
        model_upload: UploadedAssetContent | None = None,
    ) -> PersistedInventoryAsset:
        normalized_name = _require_clean_text(name, "Furniture name")
        normalized_category = _require_clean_text(category, "Furniture category")
        asset_id = AssetId(_build_asset_id(normalized_name))
        asset_attributes = dict(attributes or {})
        asset_attributes["category"] = normalized_category
        asset_attributes.setdefault("source", "admin")

        files: list[AssetFile] = []
        model_ref = _clean_optional_text(model_reference)
        if model_upload is not None:
            if model_ref is not None:
                raise ValueError("Use either a model upload or a model reference, not both.")
            files.append(
                _asset_file_from_upload(
                    asset_id=asset_id,
                    upload=model_upload,
                    role="model",
                    file_kind="MODEL",
                )
            )
            asset_attributes["model_storage"] = "database"
        elif model_ref is not None:
            model_file = _asset_file_from_reference(
                asset_id=asset_id,
                reference=model_ref,
                role="model",
                file_kind="MODEL",
            )
            if model_file is None:
                asset_attributes["model_url"] = model_ref
            else:
                files.append(model_file)

        preview_ref = _clean_optional_text(preview_reference)
        if preview_ref is not None:
            preview_file = _asset_file_from_reference(
                asset_id=asset_id,
                reference=preview_ref,
                role="preview",
                file_kind="PREVIEW",
            )
            if preview_file is None:
                asset_attributes["preview_url"] = preview_ref
            else:
                files.append(preview_file)

        asset = Asset(
            id=asset_id,
            tenant_id=TenantId(get_shared_inventory_tenant_id()),
            type="FURNITURE",
            name=normalized_name,
            style_tags=_unique_clean_strings(style_tags),
            material=_clean_optional_text(material),
            brand=_clean_optional_text(brand),
            dimensions=AssetDimensions(
                length_mm=length_mm,
                width_mm=width_mm,
                height_mm=height_mm,
            ),
            price=AssetPrice(amount=price_amount, currency=price_currency or "VND")
            if price_amount is not None
            else None,
            attributes=asset_attributes,
        )
        self._asset_repository.upsert_asset(asset)
        for asset_file in files:
            self._asset_repository.upsert_asset_file(asset_file)
        return PersistedInventoryAsset(asset=asset, files=files)

    def save_layout(
        self,
        *,
        user: UserAccount,
        name: str,
        floorplan_json: dict[str, JsonValue],
        design_json: dict[str, JsonValue] | None = None,
        styled_result_json: dict[str, JsonValue] | None = None,
        case_id: str | None = None,
        option_id: str | None = None,
        option_index: int | None = None,
        meta: dict[str, JsonValue] | None = None,
    ) -> SavedLayout:
        layout = SavedLayout(
            id=SavedLayoutId(f"layout_{uuid.uuid4().hex}"),
            user_id=user.id,
            name=name,
            floorplan_json=floorplan_json,
            design_json=design_json,
            styled_result_json=styled_result_json,
            case_id=SavedLayoutCaseId(case_id) if case_id else None,
            option_id=option_id,
            option_index=option_index,
            meta=meta or {},
        )
        self._user_repository.create_saved_layout(layout)
        stored = self._user_repository.get_saved_layout(layout.id, user.id)
        return stored or layout

    def upsert_layout_case_options(
        self,
        *,
        user: UserAccount,
        case_id: str,
        case_name: str,
        options: Sequence[LayoutOptionPersistenceRequest],
        meta: dict[str, JsonValue] | None = None,
    ) -> list[SavedLayout]:
        layout_case = SavedLayoutCase(
            id=SavedLayoutCaseId(case_id),
            user_id=user.id,
            name=case_name,
            source_pipeline_case_id=case_id,
            meta=meta or {},
        )
        stored_case = self._user_repository.upsert_saved_layout_case(layout_case)
        saved_layouts: list[SavedLayout] = []
        for option in options:
            option_meta = {
                **(option.meta or {}),
                "case_id": str(stored_case.id),
                "case_name": stored_case.name,
                "option_id": option.option_id,
                "option_index": option.option_index,
            }
            layout = SavedLayout(
                id=SavedLayoutId(f"layout_{uuid.uuid4().hex}"),
                user_id=user.id,
                name=option.name,
                floorplan_json=option.floorplan_json,
                design_json=option.design_json,
                styled_result_json=option.styled_result_json,
                case_id=stored_case.id,
                option_id=option.option_id,
                option_index=option.option_index,
                meta=option_meta,
            )
            saved_layouts.append(
                self._user_repository.upsert_saved_layout_option(layout)
            )
        return saved_layouts

    def update_layout_case_name(
        self,
        *,
        user: UserAccount,
        case_id: str,
        name: str,
    ) -> SavedLayoutCase | None:
        existing = self._user_repository.get_saved_layout_case(
            SavedLayoutCaseId(case_id), user.id
        )
        if existing is None:
            return None
        layout_case = SavedLayoutCase(
            id=existing.id,
            user_id=user.id,
            name=name,
            source_pipeline_case_id=existing.source_pipeline_case_id,
            meta=dict(existing.meta or {}),
        )
        return self._user_repository.upsert_saved_layout_case(layout_case)

    def update_layout(
        self,
        *,
        user: UserAccount,
        layout_id: str,
        name: str,
        floorplan_json: dict[str, JsonValue],
        design_json: dict[str, JsonValue] | None = None,
        styled_result_json: dict[str, JsonValue] | None = None,
        meta: dict[str, JsonValue] | None = None,
    ) -> SavedLayout | None:
        existing = self.get_layout(user=user, layout_id=layout_id)
        if existing is None:
            return None
        layout = SavedLayout(
            id=existing.id,
            user_id=user.id,
            name=name,
            floorplan_json=floorplan_json,
            design_json=design_json,
            styled_result_json=styled_result_json,
            case_id=existing.case_id,
            option_id=existing.option_id,
            option_index=existing.option_index,
            meta={**dict(existing.meta or {}), **(meta or {})},
        )
        return self._user_repository.update_saved_layout(layout)

    def list_layouts(
        self, *, user: UserAccount, case_id: str | None = None
    ) -> list[SavedLayout]:
        return list(
            self._user_repository.list_saved_layouts(
                user.id,
                SavedLayoutCaseId(case_id) if case_id else None,
            )
        )

    def get_layout(self, *, user: UserAccount, layout_id: str) -> SavedLayout | None:
        return self._user_repository.get_saved_layout(SavedLayoutId(layout_id), user.id)

    def resolve_render_layout_context(
        self,
        *,
        user: UserAccount,
        root_layout_id: str | None = None,
        case_id: str | None = None,
        option_id: str | None = None,
    ) -> RenderLayoutContext | None:
        normalized_root_layout_id = _clean_optional_text(root_layout_id)
        normalized_case_id = _clean_optional_text(case_id)
        normalized_option_id = _clean_optional_text(option_id)
        if normalized_case_id and normalized_option_id:
            layout = self._find_saved_layout_option(
                user=user,
                case_id=normalized_case_id,
                option_id=normalized_option_id,
            )
            if layout is None:
                return None
            return _render_layout_context_from_layout(layout)
        if normalized_root_layout_id:
            layout = self.get_layout(user=user, layout_id=normalized_root_layout_id)
            if layout is None:
                return None
            return _render_layout_context_from_layout(layout)
        return RenderLayoutContext(
            root_layout_id=None,
            case_id=None,
            option_id=None,
            option_index=None,
        )

    def list_generated_renders(self, *, user: UserAccount) -> list[GeneratedRender]:
        return list(self._user_repository.list_generated_renders(user.id))

    def get_generated_render(
        self, *, user: UserAccount, render_id: str
    ) -> GeneratedRender | None:
        return self._user_repository.get_generated_render(
            GeneratedRenderId(render_id),
            user.id,
        )

    def update_generated_render_meta(
        self,
        *,
        user: UserAccount,
        render_id: str,
        meta_patch: dict[str, JsonValue],
    ) -> GeneratedRender | None:
        existing = self.get_generated_render(user=user, render_id=render_id)
        if existing is None:
            return None
        merged_meta = {
            **dict(existing.meta or {}),
            **meta_patch,
        }
        return self._user_repository.update_generated_render_meta(
            GeneratedRenderId(render_id),
            user.id,
            merged_meta,
        )

    def save_snapshot_render_from_data_url(
        self,
        *,
        user: UserAccount,
        image_data_url: str,
        model_name: str,
        prompt: str,
        negative_prompt: str | None = None,
        meta: dict[str, JsonValue] | None = None,
        root_layout_id: str | None = None,
        case_id: str | None = None,
        option_id: str | None = None,
        option_index: int | None = None,
    ) -> GeneratedRender:
        mime_type, raw_bytes = _decode_data_url(image_data_url)
        render_id = GeneratedRenderId(f"render_{uuid.uuid4().hex}")
        normalized_root_layout_id = _clean_optional_text(root_layout_id)
        normalized_case_id = _clean_optional_text(case_id)
        normalized_option_id = _clean_optional_text(option_id)
        render_meta = _merge_render_layout_context_meta(
            meta or {},
            root_layout_id=normalized_root_layout_id,
            case_id=normalized_case_id,
            option_id=normalized_option_id,
            option_index=option_index,
        )

        render = GeneratedRender(
            id=render_id,
            user_id=user.id,
            source="snapshot_render",
            model_name=model_name,
            prompt=prompt,
            negative_prompt=negative_prompt,
            root_layout_id=SavedLayoutId(normalized_root_layout_id)
            if normalized_root_layout_id
            else None,
            case_id=SavedLayoutCaseId(normalized_case_id)
            if normalized_case_id
            else None,
            option_id=normalized_option_id,
            option_index=option_index,
            image_bytes=raw_bytes,
            mime_type=mime_type,
            meta=render_meta,
        )
        self._user_repository.create_generated_render(render)
        stored = self._user_repository.get_generated_render(render.id, user.id)
        return stored or render

    def can_access_asset(self, *, asset: Asset) -> bool:
        return str(asset.tenant_id) == get_shared_inventory_tenant_id()

    def serialize_inventory_asset(
        self,
        *,
        persisted: PersistedInventoryAsset,
        file_url_builder: Callable[[AssetFile], str],
    ) -> dict[str, object]:
        attributes = dict(persisted.asset.attributes or {})
        serialized_files = []
        for asset_file in persisted.files:
            serialized_files.append(
                {
                    "id": str(asset_file.id),
                    "file_kind": asset_file.file_kind,
                    "provider": asset_file.provider,
                    "storage_key": asset_file.storage_key,
                    "mime": asset_file.mime,
                    "role": asset_file.role,
                    "meta": dict(asset_file.meta or {}),
                    "url": file_url_builder(asset_file),
                }
            )
        attributes["files"] = serialized_files
        attributes["ownership_scope"] = "shared"
        preferred_model_file = _select_preferred_model_file(persisted.files)
        if preferred_model_file is not None:
            attributes["model_url"] = file_url_builder(preferred_model_file)
        category = str(attributes.get("category") or persisted.asset.name)
        return {
            "id": str(persisted.asset.id),
            "name": persisted.asset.name,
            "type": category,
            "style_tags": list(persisted.asset.style_tags or []),
            "material": persisted.asset.material,
            "brand": persisted.asset.brand,
            "dimensions": persisted.asset.dimensions.model_dump()
            if persisted.asset.dimensions
            else None,
            "price": persisted.asset.price.model_dump()
            if persisted.asset.price is not None
            else None,
            "attributes": attributes,
        }

    def serialize_saved_layout(self, layout: SavedLayout) -> dict[str, object]:
        return {
            "id": str(layout.id),
            "name": layout.name,
            "floorplan_json": layout.floorplan_json,
            "design_json": layout.design_json,
            "styled_result_json": layout.styled_result_json,
            "case_id": str(layout.case_id) if layout.case_id else None,
            "option_id": layout.option_id,
            "option_index": layout.option_index,
            "case_name": layout.case_name,
            "meta": dict(layout.meta or {}),
            "created_at": layout.created_at.isoformat() if layout.created_at else None,
            "updated_at": layout.updated_at.isoformat() if layout.updated_at else None,
        }

    def serialize_generated_render(
        self,
        render: GeneratedRender,
        *,
        file_url: str,
    ) -> dict[str, object]:
        return {
            "id": str(render.id),
            "source": render.source,
            "model_name": render.model_name,
            "prompt": render.prompt,
            "negative_prompt": render.negative_prompt,
            "root_layout_id": _serialized_render_root_layout_id(render),
            "case_id": _serialized_render_case_id(render),
            "option_id": _serialized_render_option_id(render),
            "option_index": _serialized_render_option_index(render),
            "mime_type": render.mime_type,
            "meta": dict(render.meta or {}),
            "created_at": render.created_at.isoformat() if render.created_at else None,
            "file_url": file_url,
        }

    def build_generated_render_data_url(self, render: GeneratedRender) -> str:
        raw_bytes = self.read_generated_render_bytes(render)
        image_base64 = base64.b64encode(raw_bytes).decode("ascii")
        return f"data:{render.mime_type};base64,{image_base64}"

    def read_generated_render_bytes(self, render: GeneratedRender) -> bytes:
        if render.image_bytes is not None:
            return render.image_bytes
        raise FileNotFoundError(f"Render {render.id} has no stored binary payload.")

    def _find_saved_layout_option(
        self,
        *,
        user: UserAccount,
        case_id: str,
        option_id: str,
    ) -> SavedLayout | None:
        return next(
            (
                layout
                for layout in self.list_layouts(user=user, case_id=case_id)
                if layout.option_id == option_id
            ),
            None,
        )


_REPO_ROOT = Path(__file__).resolve().parents[1]
_GLB_MAGIC = b"glTF"
_GLB_MIME_TYPE = "model/gltf-binary"
_DIRECT_REFERENCE_PREFIXES = (
    "http://",
    "https://",
    "data:",
    "blob:",
    "/assets/",
    "/assets_gen/",
    "/inventory/",
)


def _build_asset_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    slug = slug[:48] or "furniture"
    return f"asset_{slug}_{uuid.uuid4().hex[:12]}"


def _require_clean_text(value: str, label: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError(f"{label} is required.")
    return text


def _unique_clean_strings(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = value.strip()
        if text and text not in out:
            out.append(text)
    return out


def _asset_file_from_reference(
    *,
    asset_id: AssetId,
    reference: str,
    role: str,
    file_kind: FileKind,
) -> AssetFile | None:
    if _is_direct_reference(reference):
        return None
    path = _resolve_safe_local_asset_path(reference)
    mime, _ = mimetypes.guess_type(path.name)
    if path.suffix.lower() == ".glb":
        mime = _GLB_MIME_TYPE
    return AssetFile(
        id=AssetFileId(f"asset_file_{uuid.uuid4().hex}"),
        asset_id=asset_id,
        file_kind=file_kind,
        provider="local",
        storage_key=str(path),
        mime=mime,
        bytes_size=path.stat().st_size,
        role=role,
        meta={"source": "admin_reference"},
    )


def _asset_file_from_upload(
    *,
    asset_id: AssetId,
    upload: UploadedAssetContent,
    role: str,
    file_kind: FileKind,
) -> AssetFile:
    filename = _safe_uploaded_filename(upload.filename)
    content = upload.content
    if not filename.lower().endswith(".glb"):
        raise ValueError("Uploaded model file must use the .glb extension.")
    if len(content) < 12 or not content.startswith(_GLB_MAGIC):
        raise ValueError("Uploaded model file must be a valid binary GLB file.")
    asset_file_id = AssetFileId(f"asset_file_{uuid.uuid4().hex}")
    return AssetFile(
        id=asset_file_id,
        asset_id=asset_id,
        file_kind=file_kind,
        provider="database",
        storage_key=f"db://asset_files/{asset_file_id}/{filename}",
        mime=_GLB_MIME_TYPE,
        bytes_size=len(content),
        role=role,
        meta={
            "source": "admin_upload",
            "original_filename": filename,
            "uploaded_mime_type": upload.mime_type or "application/octet-stream",
        },
        content_bytes=content,
    )


def _safe_uploaded_filename(filename: str) -> str:
    clean_name = Path(filename.strip()).name
    return clean_name or "model.glb"


def _is_direct_reference(reference: str) -> bool:
    normalized = reference.strip().lower()
    return normalized.startswith(_DIRECT_REFERENCE_PREFIXES)


def _resolve_safe_local_asset_path(reference: str) -> Path:
    raw_path = Path(reference).expanduser()
    candidate = raw_path if raw_path.is_absolute() else _REPO_ROOT / raw_path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(_REPO_ROOT)
    except ValueError as exc:
        raise ValueError(
            "Local asset paths must stay inside the project workspace."
        ) from exc
    if not resolved.is_file():
        raise ValueError("Local asset path does not point to an existing file.")
    return resolved


def _clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _render_layout_context_from_layout(layout: SavedLayout) -> RenderLayoutContext:
    return RenderLayoutContext(
        root_layout_id=str(layout.id),
        case_id=str(layout.case_id) if layout.case_id else None,
        option_id=layout.option_id,
        option_index=layout.option_index,
    )


def _merge_render_layout_context_meta(
    meta: dict[str, JsonValue],
    *,
    root_layout_id: str | None,
    case_id: str | None,
    option_id: str | None,
    option_index: int | None,
) -> dict[str, JsonValue]:
    context_meta: dict[str, JsonValue] = {
        "root_layout_id": root_layout_id,
        "case_id": case_id,
        "option_id": option_id,
        "option_index": option_index,
    }
    return {**meta, **context_meta}


def _serialized_render_root_layout_id(render: GeneratedRender) -> str | None:
    if render.root_layout_id is not None:
        return str(render.root_layout_id)
    return _meta_optional_text(render.meta, "root_layout_id")


def _serialized_render_case_id(render: GeneratedRender) -> str | None:
    if render.case_id is not None:
        return str(render.case_id)
    return _meta_optional_text(render.meta, "case_id")


def _serialized_render_option_id(render: GeneratedRender) -> str | None:
    if render.option_id is not None:
        return render.option_id
    return _meta_optional_text(render.meta, "option_id")


def _serialized_render_option_index(render: GeneratedRender) -> int | None:
    if render.option_index is not None:
        return render.option_index
    raw_value = render.meta.get("option_index")
    if isinstance(raw_value, int):
        return raw_value
    if isinstance(raw_value, float):
        return int(raw_value)
    if isinstance(raw_value, str) and raw_value.strip():
        try:
            return int(raw_value)
        except ValueError:
            return None
    return None


def _meta_optional_text(meta: dict[str, JsonValue], key: str) -> str | None:
    raw_value = meta.get(key)
    if isinstance(raw_value, str) and raw_value.strip():
        return raw_value.strip()
    return None


def _decode_data_url(data_url: str) -> tuple[str, bytes]:
    raw = data_url.strip()
    if not raw.startswith("data:") or "," not in raw:
        raise ValueError("Expected a valid data URL.")
    header, encoded = raw.split(",", 1)
    mime_type = "image/png"
    if ";" in header:
        mime_type = header[5:].split(";", 1)[0] or mime_type
    try:
        return mime_type, base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 payload in data URL.") from exc


def _select_preferred_model_file(files: list[AssetFile]) -> AssetFile | None:
    def score(asset_file: AssetFile) -> int:
        role = (asset_file.role or "").strip().lower()
        file_kind = asset_file.file_kind.strip().upper()
        if role == "model":
            return 0
        if role in {"model_preferred", "model_textured", "3d"}:
            return 1
        if file_kind == "MODEL" and role != "model_source":
            return 10
        if role == "model_source":
            return 20
        return 30

    model_files = [
        asset_file
        for asset_file in files
        if asset_file.file_kind.strip().upper() == "MODEL"
    ]
    if not model_files:
        return None
    return min(model_files, key=score)
