from __future__ import annotations

import json
import mimetypes
import re
from pathlib import Path
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api.deps import get_current_user
from db.models import UserAccount
from services.object_3d_service import (
    Object3DImageRequest,
    Object3DJobPaths,
    Object3DPromptRequest,
    Object3DServiceConfig,
    create_image_job,
    create_prompt_job,
    get_job_paths,
    load_job_status,
    resolve_artifact_path,
    run_image_job,
    run_prompt_job,
)
from services.user_content_service import (
    UserAssetPersistenceRequest,
    UserContentService,
)

router = APIRouter(prefix="/object-3d", tags=["object-3d"])


class Object3DCreateResponse(BaseModel):
    job_id: str
    job_dir: str
    stage: str
    status_url: str


class Object3DPromptCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=160)
    category: str = Field(..., min_length=1, max_length=120)
    width_mm: float = Field(..., gt=0)
    depth_mm: float = Field(..., gt=0)
    height_mm: float = Field(..., gt=0)
    prompt: str = Field(..., min_length=3)
    negative_prompt: str | None = None
    keep_debug_artifacts: bool = False
    triposr_device: str = "auto"
    triposr_chunk_size: int | None = Field(default=None, ge=0)
    triposr_mc_resolution: int = Field(default=256, ge=64, le=512)
    triposr_mc_threshold: float = Field(default=12.0)


class Object3DStatusResponse(BaseModel):
    job_id: str
    job_dir: str
    mode: str
    stage: str
    updated_at_utc: str
    message: str | None = None
    error: str | None = None
    progress_current: int | None = None
    progress_total: int | None = None
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    artifact_urls: dict[str, str] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    actions: list[dict[str, Any]] = Field(default_factory=list)


def get_user_content_service() -> UserContentService:
    return UserContentService()


@router.post("/jobs/from-prompt", response_model=Object3DCreateResponse)
def create_object_3d_from_prompt(
    req: Object3DPromptCreateRequest,
    background: BackgroundTasks,
    current_user: UserAccount = Depends(get_current_user),
    user_content_service: UserContentService = Depends(get_user_content_service),
) -> Object3DCreateResponse:
    config = Object3DServiceConfig.from_env()
    service_request = Object3DPromptRequest(
        prompt=req.prompt,
        name=req.name,
        negative_prompt=req.negative_prompt,
        keep_debug_artifacts=req.keep_debug_artifacts,
        triposr_device=req.triposr_device,
        triposr_chunk_size=req.triposr_chunk_size,
        triposr_mc_resolution=req.triposr_mc_resolution,
        triposr_mc_threshold=req.triposr_mc_threshold,
    )
    job_paths = create_prompt_job(request=service_request, config=config)
    _bind_job_to_user(job_paths=job_paths, current_user=current_user)
    background.add_task(
        _run_prompt_job_and_persist,
        job_id=job_paths.job_id,
        request=service_request,
        config=config,
        current_user=current_user,
        asset_name=req.name,
        category=req.category,
        width_mm=req.width_mm,
        depth_mm=req.depth_mm,
        height_mm=req.height_mm,
        user_content_service=user_content_service,
    )
    return Object3DCreateResponse(
        job_id=job_paths.job_id,
        job_dir=str(job_paths.root),
        stage="queued",
        status_url=f"/object-3d/jobs/{job_paths.job_id}",
    )


@router.post("/jobs/from-image", response_model=Object3DCreateResponse)
async def create_object_3d_from_image(
    background: BackgroundTasks,
    image: Annotated[UploadFile, File(...)],
    name: Annotated[str, Form(...)],
    category: Annotated[str, Form(...)],
    width_mm: Annotated[float, Form(...)],
    depth_mm: Annotated[float, Form(...)],
    height_mm: Annotated[float, Form(...)],
    current_user: UserAccount = Depends(get_current_user),
    user_content_service: UserContentService = Depends(get_user_content_service),
    keep_debug_artifacts: Annotated[bool, Form()] = False,
    triposr_device: Annotated[str, Form()] = "auto",
    triposr_chunk_size: Annotated[int | None, Form()] = None,
    triposr_mc_resolution: Annotated[int, Form()] = 256,
    triposr_mc_threshold: Annotated[float, Form()] = 12.0,
) -> Object3DCreateResponse:
    filename = (image.filename or "").strip()
    if not filename:
        raise HTTPException(
            status_code=400, detail="Uploaded image must have a filename."
        )
    if width_mm <= 0 or depth_mm <= 0 or height_mm <= 0:
        raise HTTPException(status_code=400, detail="All dimensions must be positive.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    config = Object3DServiceConfig.from_env()
    service_request = Object3DImageRequest(
        source_filename=_normalize_source_filename(filename),
        name=name,
        keep_debug_artifacts=keep_debug_artifacts,
        triposr_device=triposr_device,
        triposr_chunk_size=triposr_chunk_size,
        triposr_mc_resolution=triposr_mc_resolution,
        triposr_mc_threshold=triposr_mc_threshold,
    )
    job_paths = create_image_job(
        request=service_request,
        image_bytes=image_bytes,
        config=config,
    )
    _bind_job_to_user(job_paths=job_paths, current_user=current_user)
    background.add_task(
        _run_image_job_and_persist,
        job_id=job_paths.job_id,
        request=service_request,
        config=config,
        current_user=current_user,
        asset_name=name,
        category=category,
        width_mm=width_mm,
        depth_mm=depth_mm,
        height_mm=height_mm,
        user_content_service=user_content_service,
    )
    return Object3DCreateResponse(
        job_id=job_paths.job_id,
        job_dir=str(job_paths.root),
        stage="queued",
        status_url=f"/object-3d/jobs/{job_paths.job_id}",
    )


@router.get("/jobs/{job_id}", response_model=Object3DStatusResponse)
def get_object_3d_job_status(
    job_id: str,
    current_user: UserAccount = Depends(get_current_user),
) -> Object3DStatusResponse:
    config = Object3DServiceConfig.from_env()
    try:
        job_paths = get_job_paths(job_id, config)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    _ensure_job_owner(job_paths=job_paths, current_user=current_user)
    try:
        payload = load_job_status(job_id, config)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    artifact_paths = payload.get("artifact_paths")
    artifact_path_map = artifact_paths if isinstance(artifact_paths, dict) else {}
    artifact_urls = {
        key: f"/object-3d/jobs/{job_id}/artifacts/{key}"
        for key, value in artifact_path_map.items()
        if isinstance(value, str) and value.strip()
    }
    actions = payload.get("actions") if isinstance(payload.get("actions"), list) else []
    return Object3DStatusResponse(
        job_id=str(payload.get("job_id", job_id)),
        job_dir=str(payload.get("job_dir", job_paths.root)),
        mode=str(payload.get("mode", "unknown")),
        stage=str(payload.get("stage", "unknown")),
        updated_at_utc=str(payload.get("updated_at_utc", "")),
        message=payload.get("message")
        if isinstance(payload.get("message"), str)
        else None,
        error=payload.get("error") if isinstance(payload.get("error"), str) else None,
        progress_current=payload.get("progress_current")
        if isinstance(payload.get("progress_current"), int)
        else None,
        progress_total=payload.get("progress_total")
        if isinstance(payload.get("progress_total"), int)
        else None,
        artifact_paths={
            key: value
            for key, value in artifact_path_map.items()
            if isinstance(key, str) and isinstance(value, str)
        },
        artifact_urls=artifact_urls,
        result=payload.get("result")
        if isinstance(payload.get("result"), dict)
        else None,
        actions=[item for item in actions if isinstance(item, dict)],
    )


@router.get("/jobs/{job_id}/artifacts/{artifact_name}")
def download_object_3d_artifact(
    job_id: str,
    artifact_name: str,
    current_user: UserAccount = Depends(get_current_user),
) -> FileResponse:
    config = Object3DServiceConfig.from_env()
    try:
        job_paths = get_job_paths(job_id, config)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    _ensure_job_owner(job_paths=job_paths, current_user=current_user)
    try:
        artifact_path = resolve_artifact_path(
            job_id=job_id,
            artifact_name=artifact_name,
            config=config,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    media_type, _ = mimetypes.guess_type(artifact_path.name)
    if artifact_path.suffix == ".glb":
        media_type = "model/gltf-binary"
    return FileResponse(
        artifact_path,
        media_type=media_type or "application/octet-stream",
        filename=artifact_path.name,
    )


def _run_prompt_job_and_persist(
    *,
    job_id: str,
    request: Object3DPromptRequest,
    config: Object3DServiceConfig,
    current_user: UserAccount,
    asset_name: str,
    category: str,
    width_mm: float,
    depth_mm: float,
    height_mm: float,
    user_content_service: UserContentService,
) -> None:
    run_prompt_job(job_id=job_id, request=request, config=config)
    try:
        _persist_job_result(
            job_id=job_id,
            mode="prompt",
            config=config,
            current_user=current_user,
            asset_name=asset_name,
            category=category,
            width_mm=width_mm,
            depth_mm=depth_mm,
            height_mm=height_mm,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            keep_debug_artifacts=request.keep_debug_artifacts,
            reference_model_name=config.ollama_image_model,
            user_content_service=user_content_service,
        )
    except Exception as exc:
        _augment_status_result(
            job_id=job_id,
            config=config,
            result_update={"persistence_error": str(exc)},
            action_message="Mesh generation succeeded, but saving the asset failed.",
        )


def _run_image_job_and_persist(
    *,
    job_id: str,
    request: Object3DImageRequest,
    config: Object3DServiceConfig,
    current_user: UserAccount,
    asset_name: str,
    category: str,
    width_mm: float,
    depth_mm: float,
    height_mm: float,
    user_content_service: UserContentService,
) -> None:
    run_image_job(job_id=job_id, request=request, config=config)
    try:
        _persist_job_result(
            job_id=job_id,
            mode="image",
            config=config,
            current_user=current_user,
            asset_name=asset_name,
            category=category,
            width_mm=width_mm,
            depth_mm=depth_mm,
            height_mm=height_mm,
            prompt=None,
            negative_prompt=None,
            keep_debug_artifacts=request.keep_debug_artifacts,
            reference_model_name="user-upload",
            user_content_service=user_content_service,
        )
    except Exception as exc:
        _augment_status_result(
            job_id=job_id,
            config=config,
            result_update={"persistence_error": str(exc)},
            action_message="Mesh generation succeeded, but saving the asset failed.",
        )


def _persist_job_result(
    *,
    job_id: str,
    mode: str,
    config: Object3DServiceConfig,
    current_user: UserAccount,
    asset_name: str,
    category: str,
    width_mm: float,
    depth_mm: float,
    height_mm: float,
    prompt: str | None,
    negative_prompt: str | None,
    keep_debug_artifacts: bool,
    reference_model_name: str,
    user_content_service: UserContentService,
) -> None:
    status_payload = load_job_status(job_id, config)
    if str(status_payload.get("stage")) != "done":
        return

    artifact_paths = status_payload.get("artifact_paths")
    if not isinstance(artifact_paths, dict):
        return
    mesh_glb_raw = artifact_paths.get("mesh_glb")
    if not isinstance(mesh_glb_raw, str) or not mesh_glb_raw.strip():
        return
    preferred_mesh_glb_raw = artifact_paths.get("mesh_glb_preferred")
    texture_png_raw = artifact_paths.get("texture_png")

    persisted = user_content_service.save_generated_asset(
        user=current_user,
        request=UserAssetPersistenceRequest(
            name=asset_name,
            category=category,
            width_mm=width_mm,
            depth_mm=depth_mm,
            height_mm=height_mm,
            source_mode="prompt" if mode == "prompt" else "image",
            mesh_glb_path=Path(mesh_glb_raw),
            preferred_mesh_glb_path=_read_optional_path(preferred_mesh_glb_raw),
            texture_png_path=_read_optional_path(texture_png_raw),
            source_image_path=_read_optional_path(artifact_paths.get("source_image")),
            generated_image_path=_read_optional_path(
                artifact_paths.get("generated_image")
            ),
            prompt=prompt,
            negative_prompt=negative_prompt,
            reference_model_name=reference_model_name,
            keep_debug_artifacts=keep_debug_artifacts,
        ),
    )
    _augment_status_result(
        job_id=job_id,
        config=config,
        result_update={
            "persisted_asset_id": str(persisted.asset.id),
            "persisted_asset_name": persisted.asset.name,
            "inventory_refresh_required": True,
        },
        action_message="Saved generated object into the user's inventory.",
    )


def _bind_job_to_user(
    *, job_paths: Object3DJobPaths, current_user: UserAccount
) -> None:
    payload = json.loads(job_paths.request_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        payload = {}
    payload["owner_user_id"] = str(current_user.id)
    payload["owner_tenant_id"] = str(current_user.tenant_id)
    job_paths.request_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _ensure_job_owner(
    *, job_paths: Object3DJobPaths, current_user: UserAccount
) -> None:
    payload = json.loads(job_paths.request_path.read_text(encoding="utf-8"))
    owner_user_id = payload.get("owner_user_id") if isinstance(payload, dict) else None
    if not isinstance(owner_user_id, str) or owner_user_id != str(current_user.id):
        raise HTTPException(status_code=403, detail="Access denied for this 3D job.")


def _augment_status_result(
    *,
    job_id: str,
    config: Object3DServiceConfig,
    result_update: dict[str, Any],
    action_message: str,
) -> None:
    job_paths = get_job_paths(job_id, config)
    payload = json.loads(job_paths.status_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return
    result = payload.get("result")
    if not isinstance(result, dict):
        result = {}
    result.update(result_update)
    payload["result"] = result
    actions = payload.get("actions")
    action_list = (
        [item for item in actions if isinstance(item, dict)]
        if isinstance(actions, list)
        else []
    )
    action_list.append(
        {
            "stage": "persisted_inventory_asset",
            "message": action_message,
            "updated_at_utc": payload.get("updated_at_utc"),
        }
    )
    payload["actions"] = action_list
    job_paths.status_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _read_optional_path(raw_value: object) -> Path | None:
    if isinstance(raw_value, str) and raw_value.strip():
        return Path(raw_value)
    return None


def _normalize_source_filename(filename: str) -> str:
    stripped = filename.strip().replace("\\", "/").split("/")[-1]
    if not stripped:
        return "source.png"
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", stripped)
    return normalized or "source.png"
