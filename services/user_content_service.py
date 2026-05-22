from __future__ import annotations

import base64
import binascii
import mimetypes
import shutil
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from config.demo_inventory import is_enabled_demo_inventory_tenant
from db.models import (
    Asset,
    AssetDimensions,
    AssetFile,
    AssetFileId,
    AssetFilter,
    AssetId,
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

UserAssetScope = Literal["shared", "owned"]

DEFAULT_USER_CONTENT_ROOT = (
    Path(__file__).resolve().parents[1] / "generated" / "user_content"
)


@dataclass(frozen=True)
class UserAssetPersistenceRequest:
    name: str
    category: str
    width_mm: float
    depth_mm: float
    height_mm: float
    source_mode: Literal["prompt", "image"]
    mesh_glb_path: Path
    preferred_mesh_glb_path: Path | None = None
    texture_png_path: Path | None = None
    source_image_path: Path | None = None
    generated_image_path: Path | None = None
    prompt: str | None = None
    negative_prompt: str | None = None
    reference_model_name: str | None = None
    keep_debug_artifacts: bool = False


@dataclass(frozen=True)
class PersistedUserAsset:
    asset: Asset
    files: list[AssetFile]
    scope: UserAssetScope


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
        content_root: Path = DEFAULT_USER_CONTENT_ROOT,
    ) -> None:
        ensure_user_content_schema()
        self._asset_repository = asset_repository or PostgresAssetRepository()
        self._user_repository = user_repository or PostgresUserContentRepository()
        self._content_root = content_root.expanduser().resolve()

    def get_user_by_id(self, user_id: str) -> UserAccount | None:
        return self._user_repository.get_user_by_id(UserId(user_id))

    def list_inventory_assets_for_user(
        self,
        *,
        user: UserAccount | None,
        shared_tenant_id: str | None = None,
        style_tags: list[str] | None = None,
    ) -> list[PersistedUserAsset]:
        resolved_shared_tenant = shared_tenant_id or get_shared_inventory_tenant_id()
        shared_assets = self._asset_repository.list_assets(
            AssetFilter(
                tenant_id=TenantId(resolved_shared_tenant),
                style_tags=list(style_tags or []),
            )
        )
        owned_assets: list[Asset] = []
        if user is not None:
            tenant_assets = list(
                self._asset_repository.list_assets(
                    AssetFilter(
                        tenant_id=user.tenant_id,
                        style_tags=list(style_tags or []),
                    )
                )
            )
            owned_assets = [
                asset
                for asset in tenant_assets
                if self._can_user_access_owned_asset(user=user, asset=asset)
            ]

        combined = list(shared_assets) + [
            asset
            for asset in owned_assets
            if all(str(asset.id) != str(shared.id) for shared in shared_assets)
        ]
        persisted: list[PersistedUserAsset] = []
        for asset in combined:
            files = list(self._asset_repository.list_asset_files(asset.id))
            scope: UserAssetScope = (
                "shared" if str(asset.tenant_id) == resolved_shared_tenant else "owned"
            )
            persisted.append(PersistedUserAsset(asset=asset, files=files, scope=scope))
        return persisted

    def save_generated_asset(
        self,
        *,
        user: UserAccount,
        request: UserAssetPersistenceRequest,
    ) -> PersistedUserAsset:
        asset_id = AssetId(f"asset_{uuid.uuid4().hex}")
        asset_dir = self._content_root / "assets" / str(user.id) / str(asset_id)
        asset_dir.mkdir(parents=True, exist_ok=True)

        preferred_mesh_path = (
            request.preferred_mesh_glb_path
            if request.preferred_mesh_glb_path is not None
            else request.mesh_glb_path
        )
        mesh_target_path = asset_dir / "mesh.glb"
        shutil.copy2(preferred_mesh_path, mesh_target_path)

        attributes: dict[str, JsonValue] = {
            "category": request.category,
            "ownership_scope": "owned",
            "created_by_user_id": str(user.id),
            "source_mode": request.source_mode,
            "has_texture_baked_model": bool(
                request.preferred_mesh_glb_path is not None
                and request.preferred_mesh_glb_path.resolve()
                != request.mesh_glb_path.resolve()
            ),
            "files": [],
        }
        if request.prompt:
            attributes["generation_prompt"] = request.prompt
        if request.keep_debug_artifacts:
            attributes["debug_kept"] = True

        asset = Asset(
            id=asset_id,
            tenant_id=user.tenant_id,
            type="FURNITURE",
            name=request.name,
            style_tags=[],
            material=None,
            brand="User Upload",
            dimensions=AssetDimensions(
                length_mm=request.depth_mm,
                width_mm=request.width_mm,
                height_mm=request.height_mm,
            ),
            price=None,
            attributes=attributes,
        )
        self._asset_repository.create_asset(asset)

        persisted_files: list[AssetFile] = []
        mesh_file = AssetFile(
            id=AssetFileId(f"asset_file_{uuid.uuid4().hex}"),
            asset_id=asset_id,
            file_kind="MODEL",
            provider="local",
            storage_key=str(mesh_target_path),
            mime="model/gltf-binary",
            role="model",
            meta={"filename": mesh_target_path.name},
        )
        self._asset_repository.create_asset_file(mesh_file)
        persisted_files.append(mesh_file)

        if request.mesh_glb_path.resolve() != preferred_mesh_path.resolve():
            source_mesh_target_path = asset_dir / "mesh_source.glb"
            shutil.copy2(request.mesh_glb_path, source_mesh_target_path)
            source_mesh_file = AssetFile(
                id=AssetFileId(f"asset_file_{uuid.uuid4().hex}"),
                asset_id=asset_id,
                file_kind="MODEL",
                provider="local",
                storage_key=str(source_mesh_target_path),
                mime="model/gltf-binary",
                role="model_source",
                meta={"filename": source_mesh_target_path.name},
            )
            self._asset_repository.create_asset_file(source_mesh_file)
            persisted_files.append(source_mesh_file)

        if request.texture_png_path is not None and request.texture_png_path.exists():
            texture_target_path = asset_dir / "texture.png"
            shutil.copy2(request.texture_png_path, texture_target_path)
            texture_file = AssetFile(
                id=AssetFileId(f"asset_file_{uuid.uuid4().hex}"),
                asset_id=asset_id,
                file_kind="IMAGE",
                provider="local",
                storage_key=str(texture_target_path),
                mime="image/png",
                role="texture",
                meta={"filename": texture_target_path.name},
            )
            self._asset_repository.create_asset_file(texture_file)
            persisted_files.append(texture_file)

        preview_source_path = request.generated_image_path or request.source_image_path
        if preview_source_path is not None and preview_source_path.exists():
            preview_ext = preview_source_path.suffix.lower() or ".png"
            preview_target_path = asset_dir / f"preview{preview_ext}"
            shutil.copy2(preview_source_path, preview_target_path)
            mime_type = mimetypes.guess_type(preview_target_path.name)[0] or "image/png"
            preview_file = AssetFile(
                id=AssetFileId(f"asset_file_{uuid.uuid4().hex}"),
                asset_id=asset_id,
                file_kind="IMAGE",
                provider="local",
                storage_key=str(preview_target_path),
                mime=mime_type,
                role="preview",
                meta={"filename": preview_target_path.name},
            )
            self._asset_repository.create_asset_file(preview_file)
            persisted_files.append(preview_file)

            if request.source_mode == "prompt":
                self._create_generated_render_from_file(
                    user=user,
                    source="object_prompt_reference",
                    model_name=request.reference_model_name or "x/flux2-klein:latest",
                    prompt=request.prompt or request.name,
                    negative_prompt=request.negative_prompt,
                    source_path=preview_target_path,
                    mime_type=mime_type,
                    meta={
                        "asset_id": str(asset_id),
                        "category": request.category,
                    },
                )
            else:
                self._create_generated_render_from_file(
                    user=user,
                    source="object_image_upload",
                    model_name=request.reference_model_name or "user-upload",
                    prompt=request.name,
                    negative_prompt=None,
                    source_path=preview_target_path,
                    mime_type=mime_type,
                    meta={
                        "asset_id": str(asset_id),
                        "category": request.category,
                    },
                )

        return PersistedUserAsset(asset=asset, files=persisted_files, scope="owned")

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
            saved_layouts.append(self._user_repository.upsert_saved_layout_option(layout))
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
            storage_path=None,
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

    def can_access_asset(self, *, user: UserAccount | None, asset: Asset) -> bool:
        if str(asset.tenant_id) == get_shared_inventory_tenant_id():
            return True
        if is_enabled_demo_inventory_tenant(str(asset.tenant_id)):
            return True
        if user is None:
            return False
        if str(asset.tenant_id) != str(user.tenant_id):
            return False
        return self._can_user_access_owned_asset(user=user, asset=asset)

    def serialize_inventory_asset(
        self,
        *,
        persisted: PersistedUserAsset,
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
        attributes["ownership_scope"] = persisted.scope
        preferred_model_file = _select_preferred_model_file(persisted.files)
        if preferred_model_file is not None:
            attributes["model_url"] = file_url_builder(preferred_model_file)
        if persisted.scope == "owned":
            attributes["owner_user_tenant_id"] = str(persisted.asset.tenant_id)
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
            "storage_path": render.storage_path,
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
        if render.storage_path:
            return Path(render.storage_path).read_bytes()
        raise FileNotFoundError(f"Render {render.id} has no stored binary payload.")

    def _create_generated_render_from_file(
        self,
        *,
        user: UserAccount,
        source: Literal["object_prompt_reference", "object_image_upload"],
        model_name: str,
        prompt: str,
        negative_prompt: str | None,
        source_path: Path,
        mime_type: str,
        meta: dict[str, JsonValue],
    ) -> GeneratedRender:
        render_id = GeneratedRenderId(f"render_{uuid.uuid4().hex}")
        image_bytes = source_path.read_bytes()

        render = GeneratedRender(
            id=render_id,
            user_id=user.id,
            source=source,
            model_name=model_name,
            prompt=prompt,
            negative_prompt=negative_prompt,
            storage_path=None,
            image_bytes=image_bytes,
            mime_type=mime_type,
            meta=meta,
        )
        self._user_repository.create_generated_render(render)
        return render

    def _can_user_access_owned_asset(self, *, user: UserAccount, asset: Asset) -> bool:
        attributes = dict(asset.attributes or {})
        ownership_scope = attributes.get("ownership_scope")
        if ownership_scope != "owned":
            return True

        created_by_user_id = attributes.get("created_by_user_id")
        if not isinstance(created_by_user_id, str) or not created_by_user_id.strip():
            return False
        return created_by_user_id == str(user.id)

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
