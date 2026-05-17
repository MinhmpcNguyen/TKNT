from __future__ import annotations

import argparse
import html
import json
import mimetypes
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from urllib.parse import parse_qs, urlencode, urlparse

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(BASE_DIR))

from inspect_inventory_glb_orientation import (  # noqa: E402
    DEFAULT_CONTEXTS,
    DEFAULT_HOST,
    DEFAULT_MODELS_PUBLIC_DIR,
    DEFAULT_MODELS_URL_PREFIX,
    InventoryGlbOrientationInspector,
    JsonObject,
    MODEL_FRONT_AXIS_BY_OFFSET,
    normalize_cardinal_offset,
    normalize_quarter_turn_tilt,
    path_within,
    read_json_object,
    read_bool,
    read_string,
)
from preload_inventory_glb import (  # noqa: E402
    DEFAULT_CHUNK_SIZE,
    RetrievalQuery,
    build_destination_path,
    build_model_metadata,
    build_query_text,
    canonicalize_style_key,
    collect_model_file_paths,
    encode_queries,
    ensure_retrieval_dependencies,
    get_device,
    get_existing_variant_metadata,
    load_embedding_store,
    load_text_encoder,
    retrieve_top_matches,
    safe_filename_segment,
    strip_reintroduced_suffix,
)

from db import PostgresAssetRepository  # noqa: E402
from db.models import Asset, AssetFilter, AssetId, TenantId  # noqa: E402

if TYPE_CHECKING:
    import torch
    from transformers import AutoTokenizer, CLIPTextModelWithProjection

DEFAULT_PORT = 8788
DEFAULT_SEMANTIC_PROGRAM_PATH = ROOT_DIR / "stylist" / "compiled_semantic_program.json"
DEFAULT_CURATION_ROOM_TYPES = ("living_room", "kitchen", "bedroom")
DEFAULT_CURATION_STYLES = (
    "modern",
    "japandi",
    "scandinavian",
    "mid century",
    "wabi sabi",
)
DEFAULT_CANDIDATES_PUBLIC_DIR = Path(
    "frontend/public/assets/inventory_model_candidates"
)
DEFAULT_CANDIDATES_URL_PREFIX = "/assets/inventory_model_candidates"
EMPTY_CANDIDATES_REPORT_PREFIX = "manual_inventory_empty_candidate_targets"
SEMANTIC_OBJECT_KEYS = (
    "required",
    "required_if_kept",
    "choose_exactly_one_from",
    "choose_exactly_one_from_if_kept",
    "choose_at_least_one_from",
    "optional",
)
OFFSET_LABELS = {
    0: "Front is -Z",
    90: "Front is +X",
    180: "Front is +Z",
    270: "Front is -X",
}


def escape(value: object) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def target_key(target: JsonObject) -> str:
    return f"{target.get('asset_id')}::{target.get('style_key')}"


def first_target(targets: list[JsonObject]) -> tuple[str, str] | None:
    if not targets:
        return None
    first = targets[0]
    asset_id = str(first.get("asset_id") or "")
    style_key = str(first.get("style_key") or "")
    if not asset_id or not style_key:
        return None
    return asset_id, style_key


def find_target(
    *,
    targets: list[JsonObject],
    asset_id: str,
    style_key: str,
) -> JsonObject | None:
    for target in targets:
        if target.get("asset_id") == asset_id and target.get("style_key") == style_key:
            return target
    return None


def selected_offset(target: JsonObject | None, raw_offset: str | None) -> int:
    if raw_offset:
        try:
            return normalize_cardinal_offset(float(raw_offset))
        except ValueError:
            return 0
    if target is None:
        return 0
    raw_value = target.get("rotation_deg_offset")
    try:
        return normalize_cardinal_offset(float(raw_value or 0))
    except ValueError:
        return 0


def selected_tilt(
    target: JsonObject | None,
    raw_tilt: str | None,
    *,
    axis: str,
) -> int:
    if raw_tilt:
        try:
            return normalize_quarter_turn_tilt(float(raw_tilt))
        except ValueError:
            return 0
    if target is None:
        return 0
    raw_value = target.get(f"rotation_deg_{axis}")
    try:
        return normalize_quarter_turn_tilt(float(raw_value or 0))
    except ValueError:
        return 0


def adjusted_tilt(value: int, delta: int) -> int:
    return normalize_quarter_turn_tilt(float(value + delta))


CandidateSource = Literal["existing", "objaverse"]


@dataclass(frozen=True)
class ObjaverseCandidate:
    rank: int
    uid: str
    similarity: float
    path: Path
    source: CandidateSource
    source_asset_id: str | None = None


@dataclass(frozen=True)
class ObjaverseCurationTarget:
    family_key: str
    assets: tuple[Asset, ...]
    query: RetrievalQuery
    query_text: str
    candidate_results: tuple[tuple[str, float], ...]
    candidates: tuple[ObjaverseCandidate, ...]

    @property
    def asset_id(self) -> str:
        return self.family_key

    @property
    def style_key(self) -> str:
        return self.query.style_key


def curation_target_key(*, asset_id: str, style_key: str) -> str:
    return f"{asset_id}::{style_key}"


def split_curation_target_key(value: str) -> tuple[str, str]:
    if "::" not in value:
        return "", ""
    asset_id, style_key = value.split("::", 1)
    return asset_id, style_key


def normalize_semantic_token(value: str) -> str:
    return "_".join(value.strip().lower().replace("-", "_").split())


def load_semantic_object_ids(
    *,
    semantic_program_path: Path,
    room_types: Sequence[str],
) -> list[str]:
    payload = json.loads(semantic_program_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"Semantic program must be a JSON object: {semantic_program_path}"
        )

    room_type_set = {room_type.strip() for room_type in room_types if room_type.strip()}
    object_ids: list[str] = []
    seen_object_ids: set[str] = set()
    rooms = payload.get("rooms")
    if not isinstance(rooms, list):
        return []

    for raw_room in rooms:
        if not isinstance(raw_room, Mapping):
            continue
        room_type = raw_room.get("room_type")
        if not isinstance(room_type, str) or room_type not in room_type_set:
            continue
        clusters = raw_room.get("clusters")
        if not isinstance(clusters, list):
            continue
        for raw_cluster in clusters:
            if not isinstance(raw_cluster, Mapping):
                continue
            collect_semantic_object_ids(
                raw_cluster.get("object_program"),
                object_ids,
                seen_object_ids,
            )
            activation = raw_cluster.get("activation")
            if not isinstance(activation, Mapping):
                continue
            conditions = activation.get("conditions")
            if not isinstance(conditions, list):
                continue
            for raw_condition in conditions:
                if isinstance(raw_condition, Mapping):
                    collect_semantic_object_ids(
                        raw_condition.get("effects"),
                        object_ids,
                        seen_object_ids,
                    )

    return object_ids


def collect_semantic_object_ids(
    value: object,
    object_ids: list[str],
    seen_object_ids: set[str],
) -> None:
    if not isinstance(value, Mapping):
        return
    for key in SEMANTIC_OBJECT_KEYS:
        raw_items = value.get(key)
        if not isinstance(raw_items, list):
            continue
        for raw_item in raw_items:
            if isinstance(raw_item, str) and raw_item.strip():
                object_id = normalize_semantic_token(raw_item)
                if object_id in seen_object_ids:
                    continue
                seen_object_ids.add(object_id)
                object_ids.append(object_id)


def asset_semantic_tokens(asset: Asset) -> set[str]:
    attributes = dict(asset.attributes or {})
    raw_tokens = [
        str(asset.id),
        strip_reintroduced_suffix(str(asset.id)),
        str(attributes.get("category") or ""),
        str(attributes.get("source_category") or ""),
        str(attributes.get("source_asset_id") or ""),
    ]
    return {normalize_semantic_token(token) for token in raw_tokens if token.strip()}


def asset_matches_semantic_objects(asset: Asset, object_ids: set[str]) -> bool:
    tokens = asset_semantic_tokens(asset)
    for object_id in object_ids:
        if object_id in tokens:
            return True
        prefix = f"{object_id}_"
        if any(token.startswith(prefix) for token in tokens):
            return True
    return False


def semantic_asset_order(asset: Asset, object_order: Mapping[str, int]) -> int:
    family_key = asset_semantic_family(asset, object_order)
    if family_key is None:
        return len(object_order)
    return object_order[family_key]


def asset_semantic_family(
    asset: Asset,
    object_order: Mapping[str, int],
) -> str | None:
    tokens = asset_semantic_tokens(asset)
    for object_id in object_order:
        if object_id in tokens:
            return object_id
        prefix = f"{object_id}_"
        if any(token.startswith(prefix) for token in tokens):
            return object_id
    return None


def existing_variant_has_glb(
    *,
    asset: Asset,
    style_key: str,
    models_public_dir: Path,
    models_url_prefix: str,
) -> bool:
    metadata = get_existing_variant_metadata(asset, style_key=style_key)
    if metadata is None:
        return False
    paths = collect_model_file_paths(
        metadata=metadata,
        models_public_dir=models_public_dir,
        models_url_prefix=models_url_prefix,
    )
    return any(path.expanduser().exists() for path in paths)


def existing_variant_metadata(
    *,
    asset: Asset,
    style_key: str,
) -> JsonObject | None:
    metadata = get_existing_variant_metadata(asset, style_key=style_key)
    if metadata is None:
        return None
    return {str(key): value for key, value in metadata.items()}


def existing_candidate_uid(
    *,
    asset_id: str,
    metadata: JsonObject,
) -> str:
    raw_uid = read_string(metadata.get("objaverse_uid"))
    if raw_uid is not None:
        return raw_uid
    return f"existing:{asset_id}"


def first_existing_model_path(
    *,
    metadata: JsonObject,
    models_public_dir: Path,
    models_url_prefix: str,
) -> Path | None:
    for path in sorted(
        collect_model_file_paths(
            metadata=metadata,
            models_public_dir=models_public_dir,
            models_url_prefix=models_url_prefix,
        ),
        key=str,
    ):
        candidate = path.expanduser()
        if candidate.exists():
            return candidate.resolve()
    return None


def download_candidate_file(
    *,
    uid: str,
    destination_path: Path,
    overwrite: bool,
    protected_roots: Sequence[Path],
) -> None:
    ensure_retrieval_dependencies()
    import objaverse

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists() and not overwrite:
        return

    downloaded_files = objaverse.load_objects(
        uids=[uid],
        download_processes=1,
    )
    source_path_raw = next(iter(downloaded_files.values()), None)
    if not source_path_raw:
        raise FileNotFoundError(f"Objaverse download returned no file for uid={uid}.")

    source_path = Path(source_path_raw)
    if not source_path.exists():
        raise FileNotFoundError(
            f"Downloaded file does not exist for uid={uid}: {source_path}"
        )

    if destination_path.exists():
        destination_path.unlink()
    shutil.copy2(source_path, destination_path)
    delete_unprotected_source_file(source_path, destination_path, protected_roots)


def download_candidate_files_bulk(
    *,
    candidates_by_uid: Mapping[str, Sequence[Path]],
    protected_roots: Sequence[Path],
    download_processes: int,
) -> dict[str, str]:
    ensure_retrieval_dependencies()
    import objaverse

    missing_uids = [
        uid
        for uid, paths in candidates_by_uid.items()
        if any(not path.exists() for path in paths)
    ]
    if not missing_uids:
        return {}

    downloaded_files = objaverse.load_objects(
        uids=missing_uids,
        download_processes=max(1, download_processes),
    )
    errors: dict[str, str] = {}
    for uid in missing_uids:
        source_path_raw = downloaded_files.get(uid)
        if not source_path_raw:
            errors[uid] = f"Objaverse download returned no file for uid={uid}."
            continue

        source_path = Path(source_path_raw)
        if not source_path.exists():
            errors[uid] = f"Downloaded file does not exist for uid={uid}: {source_path}"
            continue

        for destination_path in candidates_by_uid[uid]:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            if destination_path.exists():
                continue
            shutil.copy2(source_path, destination_path)
        delete_unprotected_source_file(
            source_path, candidates_by_uid[uid][0], protected_roots
        )
    return errors


def delete_unprotected_source_file(
    source_path: Path,
    destination_path: Path,
    protected_roots: Sequence[Path],
) -> None:
    source = source_path.expanduser().resolve()
    destination = destination_path.expanduser().resolve()
    if source == destination:
        return
    if any(
        path_within(source, root.expanduser().resolve()) for root in protected_roots
    ):
        return
    try:
        source.unlink()
    except OSError:
        return


def relative_public_url(
    *,
    path: Path,
    public_dir: Path,
    url_prefix: str,
) -> str:
    relative = path.resolve().relative_to(public_dir.resolve())
    prefix = "/" + url_prefix.strip("/").replace("\\", "/")
    return f"{prefix}/{relative.as_posix()}"


def next_available_model_path(
    *,
    tenant_id: str,
    asset_id: str,
    style_key: str,
    uid: str,
    models_public_dir: Path,
) -> Path:
    default_path = build_destination_path(
        tenant_id=tenant_id,
        asset_id=asset_id,
        style_key=style_key,
        models_public_dir=models_public_dir,
    )
    if not default_path.exists():
        return default_path

    base_dir = default_path.parent
    asset_segment = safe_filename_segment(asset_id)
    uid_segment = safe_filename_segment(uid)
    candidate = base_dir / f"{asset_segment}_{uid_segment}.glb"
    if not candidate.exists():
        return candidate

    for index in range(2, 1000):
        candidate = base_dir / f"{asset_segment}_{uid_segment}_{index}.glb"
        if not candidate.exists():
            return candidate
    raise FileExistsError(
        f"Could not find a free destination filename for {asset_id} [{style_key}]."
    )


def build_reviewed_metadata(
    *,
    metadata: JsonObject,
    rotation_offset: int,
    rotation_x_deg: int,
    rotation_z_deg: int,
    contexts: list[str],
    notes: str | None,
) -> JsonObject:
    updated = dict(metadata)
    override = (
        read_json_object(
            updated.get("preview_override") or updated.get("previewOverride")
        )
        or {}
    )
    override["enabled"] = True
    override["contexts"] = contexts
    override["rotation_deg_offset"] = float(rotation_offset)
    override["rotation_deg_x"] = float(rotation_x_deg)
    override["rotation_deg_z"] = float(rotation_z_deg)
    override["notes"] = notes or "Manual Objaverse candidate curation."
    updated["preview_override"] = override
    updated["orientation_review"] = {
        "version": 1,
        "status": "reviewed",
        "front_axis": MODEL_FRONT_AXIS_BY_OFFSET[rotation_offset],
        "rotation_deg_offset": float(rotation_offset),
        "rotation_deg_x": float(rotation_x_deg),
        "rotation_deg_z": float(rotation_z_deg),
        "reference_front": "2D plan +Y / 3D scene -Z",
        "contexts": contexts,
        "notes": notes,
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return updated


class ObjaverseCandidateCurator:
    def __init__(
        self,
        *,
        tenant_id: str,
        semantic_program_path: Path,
        room_types: Sequence[str],
        styles: Sequence[str],
        top_k: int,
        sim_threshold: float,
        embeddings_dir: Path,
        models_public_dir: Path,
        models_url_prefix: str,
        candidates_public_dir: Path,
        candidates_url_prefix: str,
        asset_ids: Sequence[str] | None,
        limit: int | None,
        device_name: str,
        chunk_size: int,
        download_processes: int,
    ) -> None:
        ensure_retrieval_dependencies()
        self._tenant_id = tenant_id
        self._semantic_program_path = semantic_program_path
        self._room_types = tuple(room_types)
        self._styles = tuple(style for style in styles if style.strip())
        self._top_k = top_k
        self._sim_threshold = sim_threshold
        self._embeddings_dir = embeddings_dir
        self._models_public_dir = models_public_dir
        self._models_url_prefix = models_url_prefix
        self._candidates_public_dir = candidates_public_dir
        self._candidates_url_prefix = candidates_url_prefix
        self._asset_ids = {
            asset_id.strip() for asset_id in asset_ids or [] if asset_id.strip()
        }
        self._limit = limit
        self._device_name = device_name
        self._chunk_size = chunk_size
        self._download_processes = download_processes
        self._repo = PostgresAssetRepository()
        self._targets: dict[str, ObjaverseCurationTarget] = {}
        self._target_aliases: dict[str, str] = {}
        self._target_order: list[str] = []
        self._accepted_keys: set[str] = set()
        self._rejected_keys: set[str] = set()
        self._finished_keys: set[str] = set()
        self._completed_asset_ids_by_key: dict[str, set[str]] = {}
        self._saved_candidate_count = 0
        self._deleted_candidate_count = 0
        self._skipped_existing: list[str] = []
        self._missing_asset_ids: list[str] = []
        self._empty_candidate_targets: list[JsonObject] = []
        self._empty_candidate_keys: set[str] = set()
        self._empty_candidate_report_path = (
            ROOT_DIR
            / "outputs"
            / f"{EMPTY_CANDIDATES_REPORT_PREFIX}_{safe_filename_segment(self._tenant_id)}.json"
        )
        self._download_errors: dict[str, list[str]] = {}
        self._uids: Sequence[str] | None = None
        self._feats: torch.Tensor | None = None
        self._text_model: CLIPTextModelWithProjection | None = None
        self._tokenizer: AutoTokenizer | None = None
        self._prepare_targets()

    def list_targets(self) -> list[JsonObject]:
        rows: list[JsonObject] = []
        for key in self._target_order:
            target = self._targets[key]
            if (
                key in self._accepted_keys
                or key in self._rejected_keys
                or key in self._finished_keys
                or not target.candidates
            ):
                continue
            rows.append(self._serialize_target(target))
        return rows

    def stats(self) -> JsonObject:
        pending = len(self.list_targets())
        return {
            "pending": pending,
            "accepted": len(self._accepted_keys),
            "rejected": len(self._rejected_keys),
            "finished": len(self._finished_keys),
            "saved_candidates": self._saved_candidate_count,
            "deleted_candidates": self._deleted_candidate_count,
            "empty_candidate_targets": len(self._empty_candidate_targets),
            "skipped_existing": len(self._skipped_existing),
            "missing_assets": len(self._missing_asset_ids),
            "styles": list(self._styles),
            "room_types": list(self._room_types),
            "top_k": self._top_k,
        }

    def first_pending_key(self) -> str:
        targets = self.list_targets()
        if not targets:
            return ""
        return str(targets[0].get("key") or "")

    def canonical_target_key(self, key: str) -> str:
        return self._target_aliases.get(key, key)

    def target_for_key(self, key: str) -> ObjaverseCurationTarget | None:
        key = self.canonical_target_key(key)
        if (
            key in self._accepted_keys
            or key in self._rejected_keys
            or key in self._finished_keys
        ):
            return None
        target = self._targets.get(key)
        if target is None or not target.candidates:
            return None
        return target

    def candidate_by_rank(
        self,
        *,
        target: ObjaverseCurationTarget,
        rank: int,
    ) -> ObjaverseCandidate | None:
        for candidate in target.candidates:
            if candidate.rank == rank:
                return candidate
        return None

    def next_destination_asset_id(
        self,
        *,
        key: str,
        target: ObjaverseCurationTarget,
    ) -> str | None:
        completed_asset_ids = self._completed_asset_ids_by_key.get(
            self.canonical_target_key(key),
            set(),
        )
        for asset in target.assets:
            asset_id = str(asset.id)
            if asset_id not in completed_asset_ids:
                return asset_id
        return None

    def default_candidate_rank(
        self,
        *,
        key: str,
        target: ObjaverseCurationTarget,
        destination_asset_id: str | None = None,
        prefer_new: bool = False,
    ) -> int:
        if destination_asset_id is None:
            destination_asset_id = self.next_destination_asset_id(
                key=key,
                target=target,
            )
        if destination_asset_id is not None and not prefer_new:
            for candidate in target.candidates:
                if (
                    candidate.source == "existing"
                    and candidate.source_asset_id == destination_asset_id
                ):
                    return candidate.rank
        if destination_asset_id is not None:
            for candidate in target.candidates:
                if candidate.source == "objaverse":
                    return candidate.rank
        if prefer_new:
            for candidate in target.candidates:
                if candidate.source == "objaverse":
                    return candidate.rank
        return target.candidates[0].rank if target.candidates else 1

    def ensure_candidate_files(
        self, target: ObjaverseCurationTarget
    ) -> ObjaverseCurationTarget:
        key = curation_target_key(asset_id=target.asset_id, style_key=target.style_key)
        if not target.candidates:
            raise ValueError(f"No candidates were prepared for {key}.")
        errors: list[str] = []
        candidates_by_uid: dict[str, list[Path]] = {}
        for candidate in target.candidates:
            if candidate.path.exists():
                continue
            candidates_by_uid.setdefault(candidate.uid, []).append(candidate.path)
        try:
            download_errors = download_candidate_files_bulk(
                candidates_by_uid=candidates_by_uid,
                protected_roots=[
                    self._models_public_dir,
                    self._candidates_public_dir,
                ],
                download_processes=self._download_processes,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            download_errors = {
                candidate.uid: str(exc)
                for candidate in target.candidates
                if not candidate.path.exists()
            }
        for candidate in target.candidates:
            if candidate.uid in download_errors:
                errors.append(
                    f"rank {candidate.rank}: {download_errors[candidate.uid]}"
                )
        if errors:
            self._download_errors[key] = errors
        return target

    def resolve_candidate_path(self, *, key: str, rank: int) -> Path:
        target = self.target_for_key(key)
        if target is None:
            raise ValueError("Candidate target is no longer pending.")
        target = self.ensure_candidate_files(target)
        candidate = self.candidate_by_rank(target=target, rank=rank)
        if candidate is None:
            raise ValueError(f"No candidate rank {rank} for {key}.")
        if not candidate.path.exists():
            raise FileNotFoundError(
                f"Candidate rank {rank} did not download for {key}: {candidate.path}"
            )
        if not (
            path_within(candidate.path, self._candidates_public_dir)
            or path_within(candidate.path, self._models_public_dir)
        ):
            raise ValueError(f"Unsafe candidate path: {candidate.path}")
        return candidate.path

    def accept_candidate(self, payload: JsonObject) -> JsonObject:
        key = read_string(payload.get("key"))
        if key is None:
            raise ValueError("key is required.")
        key = self.canonical_target_key(key)
        target = self.target_for_key(key)
        if target is None:
            raise ValueError("Candidate target is no longer pending.")

        rank = int(str(payload.get("rank") or "0"))
        candidate = self.candidate_by_rank(target=target, rank=rank)
        if candidate is None:
            raise ValueError(f"No candidate rank {rank} for {key}.")
        candidate_path = self.resolve_candidate_path(key=key, rank=rank)

        rotation_offset_raw = payload.get("rotation_deg_offset")
        rotation_offset = normalize_cardinal_offset(float(rotation_offset_raw or 0))
        rotation_x_deg = normalize_quarter_turn_tilt(
            float(payload.get("rotation_deg_x") or 0)
        )
        rotation_z_deg = normalize_quarter_turn_tilt(
            float(payload.get("rotation_deg_z") or 0)
        )
        contexts = [
            str(context).strip().lower()
            for context in payload.get("contexts", [])
            if str(context).strip()
        ]
        if not contexts:
            contexts = list(DEFAULT_CONTEXTS)
        notes = read_string(payload.get("notes"))
        delete_unselected_existing = read_bool(
            payload.get("delete_unselected_existing")
        )

        persisted_asset_ids: list[str] = []
        destination_paths: list[str] = []
        public_urls: list[str] = []
        destination_path_set: set[Path] = set()
        for original_asset in target.assets:
            asset = self._repo.get_asset(AssetId(str(original_asset.id)))
            if asset is None or str(asset.tenant_id) != self._tenant_id:
                continue

            destination_path = build_destination_path(
                tenant_id=self._tenant_id,
                asset_id=str(asset.id),
                style_key=target.style_key,
                models_public_dir=self._models_public_dir,
            ).resolve()
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            if candidate_path.resolve() != destination_path:
                shutil.copy2(candidate_path, destination_path)
            destination_path_set.add(destination_path)

            public_url = relative_public_url(
                path=destination_path,
                public_dir=self._models_public_dir,
                url_prefix=self._models_url_prefix,
            )
            member_query = RetrievalQuery(
                asset=asset,
                style=target.query.style,
                style_key=target.style_key,
                material=target.query.material,
                style_tags=target.query.style_tags,
            )
            base_metadata = build_model_metadata(
                query=member_query,
                query_text=target.query_text,
                uid=candidate.uid,
                similarity=candidate.similarity,
                public_url=public_url,
                destination_path=destination_path,
                candidate_results=target.candidate_results,
            )
            base_metadata["curation"] = {
                "status": "accepted",
                "accepted_rank": candidate.rank,
                "accepted_at_utc": datetime.now(timezone.utc).isoformat(),
                "candidate_staging_dir": str(candidate.path.parent),
                "group_key": key,
                "group_family": target.family_key,
                "group_member_asset_ids": [str(member.id) for member in target.assets],
                "candidate_source": candidate.source,
                "candidate_source_asset_id": candidate.source_asset_id,
            }
            reviewed_metadata = build_reviewed_metadata(
                metadata=base_metadata,
                rotation_offset=rotation_offset,
                rotation_x_deg=rotation_x_deg,
                rotation_z_deg=rotation_z_deg,
                contexts=contexts,
                notes=notes,
            )
            self._persist_curated_metadata(
                asset=asset, target=target, metadata=reviewed_metadata
            )
            persisted_asset_ids.append(str(asset.id))
            destination_paths.append(str(destination_path))
            public_urls.append(public_url)

        if delete_unselected_existing:
            protected_paths = {candidate_path.resolve(), *destination_path_set}
            self._delete_unselected_existing_candidates(
                target=target,
                selected_candidate=candidate,
                protected_paths=protected_paths,
            )
        self._delete_staging_dir(target)
        self._accepted_keys.add(key)
        return {
            "ok": True,
            "key": key,
            "asset_id": target.asset_id,
            "asset_ids": persisted_asset_ids,
            "style_key": target.style_key,
            "accepted_rank": candidate.rank,
            "uid": candidate.uid,
            "destination_paths": destination_paths,
            "public_urls": public_urls,
        }

    def save_candidate(self, payload: JsonObject) -> JsonObject:
        key, target, candidate = self._candidate_from_payload(payload)
        candidate_path = self.resolve_candidate_path(key=key, rank=candidate.rank)
        destination_asset_id = self._destination_asset_id(
            candidate=candidate,
            payload=payload,
        )
        asset = self._current_target_asset(
            target=target,
            asset_id=destination_asset_id,
        )
        rotation_offset, rotation_x_deg, rotation_z_deg, contexts, notes = (
            self._review_payload(payload)
        )

        member_query = RetrievalQuery(
            asset=asset,
            style=target.query.style,
            style_key=target.style_key,
            material=target.query.material,
            style_tags=target.query.style_tags,
        )
        destination_path = self._save_candidate_file(
            candidate=candidate,
            candidate_path=candidate_path,
            destination_asset_id=str(asset.id),
            style_key=target.style_key,
        )
        public_url = relative_public_url(
            path=destination_path,
            public_dir=self._models_public_dir,
            url_prefix=self._models_url_prefix,
        )
        base_metadata = self._base_metadata_for_candidate(
            asset=asset,
            candidate=candidate,
            target=target,
            member_query=member_query,
            public_url=public_url,
            destination_path=destination_path,
        )
        base_metadata["curation"] = {
            "status": "saved",
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_rank": candidate.rank,
            "candidate_staging_dir": str(candidate.path.parent),
            "group_key": key,
            "group_family": target.family_key,
            "candidate_source": candidate.source,
            "candidate_source_asset_id": candidate.source_asset_id,
            "destination_asset_id": str(asset.id),
        }
        reviewed_metadata = build_reviewed_metadata(
            metadata=base_metadata,
            rotation_offset=rotation_offset,
            rotation_x_deg=rotation_x_deg,
            rotation_z_deg=rotation_z_deg,
            contexts=contexts,
            notes=notes,
        )
        self._persist_curated_metadata(
            asset=asset,
            target=target,
            metadata=reviewed_metadata,
        )

        self._mark_asset_completed(key=key, asset_id=str(asset.id))
        self._remove_candidate(key=key, rank=candidate.rank)
        if self._target_is_complete(key=key, target=target):
            self._delete_staging_dir(target)
            self._finished_keys.add(key)
        elif candidate.source == "objaverse":
            self._delete_candidate_file(candidate=candidate, skip_if_referenced=False)
        self._saved_candidate_count += 1
        return {
            "ok": True,
            "key": key,
            "asset_id": str(asset.id),
            "style_key": target.style_key,
            "rank": candidate.rank,
            "uid": candidate.uid,
            "destination_path": str(destination_path),
            "public_url": public_url,
        }

    def delete_candidate(self, payload: JsonObject) -> JsonObject:
        key, target, candidate = self._candidate_from_payload(payload)
        removed_asset_ids: list[str] = []
        if candidate.source == "existing":
            removed_asset_ids = self._remove_existing_candidate_metadata(
                target=target,
                candidate=candidate,
            )
            deleted_paths = self._delete_candidate_file(
                candidate=candidate,
                skip_if_referenced=True,
            )
        else:
            deleted_paths = self._delete_candidate_file(
                candidate=candidate,
                skip_if_referenced=False,
            )

        self._remove_candidate(key=key, rank=candidate.rank)
        self._deleted_candidate_count += 1
        remaining = self._targets.get(key)
        if remaining is not None and not remaining.candidates:
            self._delete_staging_dir(target)
            self._finished_keys.add(key)
        return {
            "ok": True,
            "key": key,
            "asset_id": target.asset_id,
            "style_key": target.style_key,
            "rank": candidate.rank,
            "uid": candidate.uid,
            "removed_asset_ids": removed_asset_ids,
            "deleted_paths": deleted_paths,
        }

    def finish_target(self, payload: JsonObject) -> JsonObject:
        key = read_string(payload.get("key"))
        if key is None:
            raise ValueError("key is required.")
        key = self.canonical_target_key(key)
        target = self.target_for_key(key)
        if target is None:
            raise ValueError("Candidate target is no longer pending.")
        self._delete_staging_dir(target)
        self._finished_keys.add(key)
        return {
            "ok": True,
            "key": key,
            "asset_id": target.asset_id,
            "asset_ids": [str(asset.id) for asset in target.assets],
            "style_key": target.style_key,
        }

    def reject_target(self, payload: JsonObject) -> JsonObject:
        key = read_string(payload.get("key"))
        if key is None:
            raise ValueError("key is required.")
        key = self.canonical_target_key(key)
        target = self.target_for_key(key)
        if target is None:
            raise ValueError("Candidate target is no longer pending.")
        self._delete_staging_dir(target)
        self._rejected_keys.add(key)
        return {
            "ok": True,
            "key": key,
            "asset_id": target.asset_id,
            "asset_ids": [str(asset.id) for asset in target.assets],
            "style_key": target.style_key,
        }

    def cleanup_all_staging(self) -> None:
        tenant_dir = self._candidates_public_dir / safe_filename_segment(
            self._tenant_id
        )
        if not tenant_dir.exists():
            return
        if not path_within(tenant_dir, self._candidates_public_dir):
            return
        shutil.rmtree(tenant_dir)

    def download_errors_for_key(self, key: str) -> list[str]:
        return list(self._download_errors.get(key, []))

    def empty_candidate_report_path(self) -> Path:
        return self._empty_candidate_report_path

    def _prepare_targets(self) -> None:
        object_id_list = load_semantic_object_ids(
            semantic_program_path=self._semantic_program_path,
            room_types=self._room_types,
        )
        object_ids = set(object_id_list)
        if not object_ids:
            raise ValueError(
                "No semantic object ids were loaded for candidate curation."
            )

        object_order = {
            object_id: index for index, object_id in enumerate(object_id_list)
        }
        assets = self._load_semantic_assets(
            object_ids=object_ids,
            object_order=object_order,
        )
        targets = self._build_group_targets(assets=assets, object_order=object_order)
        if not targets:
            print("No pending Objaverse candidate curation targets were generated.")
            return

        for target in targets:
            key = curation_target_key(
                asset_id=target.asset_id,
                style_key=target.style_key,
            )
            self._target_order.append(key)
            self._targets[key] = target
            for asset in target.assets:
                self._target_aliases[
                    curation_target_key(
                        asset_id=str(asset.id),
                        style_key=target.style_key,
                    )
                ] = key
        self._retrieve_and_download_all_candidates()
        self._write_empty_candidate_report()
        print(
            "Prepared "
            f"{len(self._targets)} pending grouped candidate target(s); "
            f"skipped_existing={len(self._skipped_existing)} "
            f"missing_assets={len(self._missing_asset_ids)}."
        )
        print(f"Empty candidate target report: {self._empty_candidate_report_path}")

    def _ensure_retrieval_runtime(
        self,
    ) -> tuple[
        Sequence[str],
        torch.Tensor,
        CLIPTextModelWithProjection,
        AutoTokenizer,
    ]:
        if (
            self._uids is not None
            and self._feats is not None
            and self._text_model is not None
            and self._tokenizer is not None
        ):
            return self._uids, self._feats, self._text_model, self._tokenizer

        device = get_device(self._device_name)
        print(f"Candidate curation device: {device}")
        print("Loading text encoder for lazy candidate curation...")
        self._text_model, self._tokenizer = load_text_encoder(device)
        print("Loading Objaverse embeddings for lazy candidate curation...")
        self._uids, self._feats = load_embedding_store(self._embeddings_dir)
        print(f"Loaded {len(self._uids)} Objaverse asset embeddings.")
        return self._uids, self._feats, self._text_model, self._tokenizer

    def _retrieve_and_download_all_candidates(self) -> None:
        if not self._target_order:
            return

        uids, feats, text_model, tokenizer = self._ensure_retrieval_runtime()
        targets = [self._targets[key] for key in self._target_order]
        query_texts = [target.query_text for target in targets]
        print(
            f"Retrieving top {self._top_k} candidates for "
            f"{len(targets)} grouped target(s)..."
        )
        query_embeddings = encode_queries(query_texts, text_model, tokenizer)
        candidates_by_uid: dict[str, list[Path]] = {}

        for key, target, query_embedding in zip(
            self._target_order,
            targets,
            query_embeddings,
            strict=True,
        ):
            existing_candidates = self._existing_candidates(
                target=target,
                start_rank=1,
            )
            results = retrieve_top_matches(
                query_embedding.unsqueeze(0),
                feats,
                uids,
                top_k=self._top_k,
                sim_threshold=self._sim_threshold,
                chunk_size=self._chunk_size,
            )
            if not results:
                if not existing_candidates:
                    self._download_errors[key] = [
                        f"No Objaverse candidates found for {key}."
                    ]
                    self._record_empty_candidate_target(
                        key=key,
                        target=target,
                        reason="no_existing_candidate_and_no_objaverse_match",
                    )
                    continue
                resolved_target = ObjaverseCurationTarget(
                    family_key=target.family_key,
                    assets=target.assets,
                    query=target.query,
                    query_text=target.query_text,
                    candidate_results=(),
                    candidates=existing_candidates,
                )
                self._targets[key] = resolved_target
                continue

            new_candidates = tuple(
                ObjaverseCandidate(
                    rank=index + len(existing_candidates),
                    uid=uid,
                    similarity=similarity,
                    path=self._candidate_path(
                        asset_id=target.asset_id,
                        style_key=target.style_key,
                        rank=index,
                        uid=uid,
                    ),
                    source="objaverse",
                )
                for index, (uid, similarity) in enumerate(results, start=1)
            )
            candidates = (*existing_candidates, *new_candidates)
            resolved_target = ObjaverseCurationTarget(
                family_key=target.family_key,
                assets=target.assets,
                query=target.query,
                query_text=target.query_text,
                candidate_results=tuple(results),
                candidates=candidates,
            )
            self._targets[key] = resolved_target
            for candidate in new_candidates:
                if not candidate.path.exists():
                    candidates_by_uid.setdefault(candidate.uid, []).append(
                        candidate.path
                    )

        if not candidates_by_uid:
            return

        print(
            "Downloading "
            f"{sum(len(paths) for paths in candidates_by_uid.values())} staged "
            f"candidate file(s) across {len(candidates_by_uid)} unique Objaverse UID(s)..."
        )
        download_errors = download_candidate_files_bulk(
            candidates_by_uid=candidates_by_uid,
            protected_roots=[
                self._models_public_dir,
                self._candidates_public_dir,
            ],
            download_processes=self._download_processes,
        )
        for key in self._target_order:
            target = self._targets[key]
            errors = list(self._download_errors.get(key, []))
            for candidate in target.candidates:
                if candidate.uid in download_errors:
                    errors.append(
                        f"rank {candidate.rank}: {download_errors[candidate.uid]}"
                    )
            if errors:
                self._download_errors[key] = errors

    def _ensure_target_candidates(
        self,
        target: ObjaverseCurationTarget,
    ) -> ObjaverseCurationTarget:
        if target.candidates:
            return target

        key = curation_target_key(asset_id=target.asset_id, style_key=target.style_key)
        uids, feats, text_model, tokenizer = self._ensure_retrieval_runtime()
        print(f"Retrieving top {self._top_k} candidates for {key}...")
        existing_candidates = self._existing_candidates(target=target, start_rank=1)
        query_embedding = encode_queries([target.query_text], text_model, tokenizer)[0]
        results = retrieve_top_matches(
            query_embedding.unsqueeze(0),
            feats,
            uids,
            top_k=self._top_k,
            sim_threshold=self._sim_threshold,
            chunk_size=self._chunk_size,
        )
        if not results:
            if not existing_candidates:
                raise ValueError(f"No Objaverse candidates found for {key}.")
            resolved_target = ObjaverseCurationTarget(
                family_key=target.family_key,
                assets=target.assets,
                query=target.query,
                query_text=target.query_text,
                candidate_results=(),
                candidates=existing_candidates,
            )
            self._targets[key] = resolved_target
            return resolved_target

        new_candidates = tuple(
            ObjaverseCandidate(
                rank=index + len(existing_candidates),
                uid=uid,
                similarity=similarity,
                path=self._candidate_path(
                    asset_id=target.asset_id,
                    style_key=target.style_key,
                    rank=index,
                    uid=uid,
                ),
                source="objaverse",
            )
            for index, (uid, similarity) in enumerate(results, start=1)
        )
        candidates = (*existing_candidates, *new_candidates)
        resolved_target = ObjaverseCurationTarget(
            family_key=target.family_key,
            assets=target.assets,
            query=target.query,
            query_text=target.query_text,
            candidate_results=tuple(results),
            candidates=candidates,
        )
        self._targets[key] = resolved_target
        return resolved_target

    def _existing_candidates(
        self,
        *,
        target: ObjaverseCurationTarget,
        start_rank: int,
    ) -> tuple[ObjaverseCandidate, ...]:
        candidates: list[ObjaverseCandidate] = []
        rank = start_rank
        for asset in target.assets:
            metadata = existing_variant_metadata(
                asset=asset,
                style_key=target.style_key,
            )
            if metadata is None:
                continue
            path = first_existing_model_path(
                metadata=metadata,
                models_public_dir=self._models_public_dir,
                models_url_prefix=self._models_url_prefix,
            )
            if path is None:
                continue
            resolved_path = path.resolve()
            candidates.append(
                ObjaverseCandidate(
                    rank=rank,
                    uid=existing_candidate_uid(
                        asset_id=str(asset.id),
                        metadata=metadata,
                    ),
                    similarity=0.0,
                    path=resolved_path,
                    source="existing",
                    source_asset_id=str(asset.id),
                )
            )
            rank += 1
        return tuple(candidates)

    def _delete_unselected_existing_candidates(
        self,
        *,
        target: ObjaverseCurationTarget,
        selected_candidate: ObjaverseCandidate,
        protected_paths: set[Path],
    ) -> None:
        for candidate in target.candidates:
            if candidate.source != "existing":
                continue
            if candidate.rank == selected_candidate.rank:
                continue
            path = candidate.path.expanduser().resolve()
            if path in protected_paths:
                continue
            if not path.exists() or not path_within(path, self._models_public_dir):
                continue
            try:
                path.unlink()
            except OSError:
                continue

    def _candidate_from_payload(
        self,
        payload: JsonObject,
    ) -> tuple[str, ObjaverseCurationTarget, ObjaverseCandidate]:
        key = read_string(payload.get("key"))
        if key is None:
            raise ValueError("key is required.")
        key = self.canonical_target_key(key)
        target = self.target_for_key(key)
        if target is None:
            raise ValueError("Candidate target is no longer pending.")

        rank = int(str(payload.get("rank") or "0"))
        candidate = self.candidate_by_rank(target=target, rank=rank)
        if candidate is None:
            raise ValueError(f"No candidate rank {rank} for {key}.")
        return key, target, candidate

    def _review_payload(
        self,
        payload: JsonObject,
    ) -> tuple[int, int, int, list[str], str | None]:
        rotation_offset = normalize_cardinal_offset(
            float(payload.get("rotation_deg_offset") or 0)
        )
        rotation_x_deg = normalize_quarter_turn_tilt(
            float(payload.get("rotation_deg_x") or 0)
        )
        rotation_z_deg = normalize_quarter_turn_tilt(
            float(payload.get("rotation_deg_z") or 0)
        )
        contexts = [
            str(context).strip().lower()
            for context in payload.get("contexts", [])
            if str(context).strip()
        ]
        if not contexts:
            contexts = list(DEFAULT_CONTEXTS)
        return (
            rotation_offset,
            rotation_x_deg,
            rotation_z_deg,
            contexts,
            read_string(payload.get("notes")),
        )

    def _destination_asset_id(
        self,
        *,
        candidate: ObjaverseCandidate,
        payload: JsonObject,
    ) -> str:
        if candidate.source == "existing":
            if candidate.source_asset_id is None:
                raise ValueError("Existing candidate has no source asset id.")
            return candidate.source_asset_id

        destination_asset_id = read_string(payload.get("destination_asset_id"))
        if destination_asset_id is None:
            raise ValueError("destination_asset_id is required for new candidates.")
        return destination_asset_id

    def _current_target_asset(
        self,
        *,
        target: ObjaverseCurationTarget,
        asset_id: str,
    ) -> Asset:
        if asset_id not in {str(member.id) for member in target.assets}:
            raise ValueError(f"{asset_id} is not a member of {target.asset_id}.")
        asset = self._repo.get_asset(AssetId(asset_id))
        if asset is None or str(asset.tenant_id) != self._tenant_id:
            raise ValueError(f"Inventory asset was not found: {asset_id}.")
        return asset

    def _save_candidate_file(
        self,
        *,
        candidate: ObjaverseCandidate,
        candidate_path: Path,
        destination_asset_id: str,
        style_key: str,
    ) -> Path:
        if candidate.source == "existing":
            return candidate_path.resolve()

        destination_path = next_available_model_path(
            tenant_id=self._tenant_id,
            asset_id=destination_asset_id,
            style_key=style_key,
            uid=candidate.uid,
            models_public_dir=self._models_public_dir,
        ).resolve()
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_path, destination_path)
        return destination_path

    def _base_metadata_for_candidate(
        self,
        *,
        asset: Asset,
        candidate: ObjaverseCandidate,
        target: ObjaverseCurationTarget,
        member_query: RetrievalQuery,
        public_url: str,
        destination_path: Path,
    ) -> JsonObject:
        current_metadata = existing_variant_metadata(
            asset=asset,
            style_key=target.style_key,
        )
        if (
            candidate.source == "existing"
            and current_metadata is not None
            and self._metadata_references_path(
                metadata=current_metadata,
                path=destination_path,
            )
        ):
            return dict(current_metadata)

        return build_model_metadata(
            query=member_query,
            query_text=target.query_text,
            uid=candidate.uid,
            similarity=candidate.similarity,
            public_url=public_url,
            destination_path=destination_path,
            candidate_results=target.candidate_results,
        )

    def _replace_candidate(
        self,
        *,
        key: str,
        candidate: ObjaverseCandidate,
    ) -> None:
        target = self._targets[key]
        candidates = tuple(
            candidate if existing.rank == candidate.rank else existing
            for existing in target.candidates
        )
        self._targets[key] = replace(target, candidates=candidates)

    def _mark_asset_completed(self, *, key: str, asset_id: str) -> None:
        self._completed_asset_ids_by_key.setdefault(key, set()).add(asset_id)

    def _target_is_complete(
        self,
        *,
        key: str,
        target: ObjaverseCurationTarget,
    ) -> bool:
        completed_asset_ids = self._completed_asset_ids_by_key.get(key, set())
        return all(str(asset.id) in completed_asset_ids for asset in target.assets)

    def _remove_candidate(self, *, key: str, rank: int) -> None:
        target = self._targets[key]
        candidates = tuple(
            candidate for candidate in target.candidates if candidate.rank != rank
        )
        self._targets[key] = replace(target, candidates=candidates)

    def _remove_existing_candidate_metadata(
        self,
        *,
        target: ObjaverseCurationTarget,
        candidate: ObjaverseCandidate,
    ) -> list[str]:
        removed_asset_ids: list[str] = []
        candidate_path = candidate.path.expanduser().resolve()
        candidate_asset_ids = (
            {candidate.source_asset_id}
            if candidate.source_asset_id is not None
            else {str(asset.id) for asset in target.assets}
        )
        for original_asset in target.assets:
            if str(original_asset.id) not in candidate_asset_ids:
                continue
            asset = self._repo.get_asset(AssetId(str(original_asset.id)))
            if asset is None or str(asset.tenant_id) != self._tenant_id:
                continue

            attributes = dict(asset.attributes or {})
            removed = False
            variants_raw = read_json_object(attributes.get("model_variants")) or {}
            matching_style_keys = [
                key
                for key in variants_raw
                if canonicalize_style_key(str(key)) == target.style_key
            ]
            for style_key in matching_style_keys:
                variant_metadata = read_json_object(variants_raw.get(style_key))
                if variant_metadata is not None and self._metadata_references_path(
                    metadata=variant_metadata,
                    path=candidate_path,
                ):
                    variants_raw.pop(style_key, None)
                    removed = True

            if variants_raw:
                attributes["model_variants"] = variants_raw
            else:
                attributes.pop("model_variants", None)

            top_level_model = read_json_object(attributes.get("model_3d"))
            top_level_matches = (
                top_level_model is not None
                and self._metadata_references_path(
                    metadata=top_level_model,
                    path=candidate_path,
                )
            )
            if top_level_matches:
                attributes.pop("model_3d", None)
                attributes.pop("model_url", None)
                removed = True

            if not removed:
                continue
            self._repo.upsert_asset(asset.model_copy(update={"attributes": attributes}))
            removed_asset_ids.append(str(asset.id))
        return removed_asset_ids

    def _delete_candidate_file(
        self,
        *,
        candidate: ObjaverseCandidate,
        skip_if_referenced: bool,
    ) -> list[str]:
        path = candidate.path.expanduser().resolve()
        if not path.exists() or path.suffix.lower() != ".glb":
            return []

        allowed_root = (
            self._models_public_dir
            if candidate.source == "existing"
            else self._candidates_public_dir
        )
        if not path_within(path, allowed_root):
            return []
        if skip_if_referenced and self._path_is_referenced_in_db(path):
            return []

        try:
            path.unlink()
        except OSError:
            return []
        return [str(path)]

    def _path_is_referenced_in_db(self, path: Path) -> bool:
        assets = self._repo.list_assets(
            AssetFilter(tenant_id=TenantId(self._tenant_id), type="FURNITURE")
        )
        return any(
            self._asset_references_path(asset=asset, path=path) for asset in assets
        )

    def _asset_references_path(self, *, asset: Asset, path: Path) -> bool:
        attributes = dict(asset.attributes or {})
        top_level_model = read_json_object(attributes.get("model_3d"))
        if top_level_model is not None and self._metadata_references_path(
            metadata=top_level_model,
            path=path,
        ):
            return True

        variants_raw = read_json_object(attributes.get("model_variants")) or {}
        for metadata_raw in variants_raw.values():
            metadata = read_json_object(metadata_raw)
            if metadata is not None and self._metadata_references_path(
                metadata=metadata,
                path=path,
            ):
                return True
        return False

    def _metadata_references_path(self, *, metadata: JsonObject, path: Path) -> bool:
        resolved_path = path.expanduser().resolve()
        candidate_paths = collect_model_file_paths(
            metadata=metadata,
            models_public_dir=self._models_public_dir,
            models_url_prefix=self._models_url_prefix,
        )
        return any(
            candidate_path.expanduser().resolve() == resolved_path
            for candidate_path in candidate_paths
        )

    def _record_empty_candidate_target(
        self,
        *,
        key: str,
        target: ObjaverseCurationTarget,
        reason: str,
    ) -> None:
        if key in self._empty_candidate_keys:
            return
        self._empty_candidate_keys.add(key)
        self._empty_candidate_targets.append(
            {
                "key": key,
                "family_key": target.family_key,
                "style": target.query.style,
                "style_key": target.style_key,
                "member_asset_ids": [str(asset.id) for asset in target.assets],
                "query_text": target.query_text,
                "reason": reason,
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    def _write_empty_candidate_report(self) -> None:
        payload: JsonObject = {
            "tenant_id": self._tenant_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "count": len(self._empty_candidate_targets),
            "targets": self._empty_candidate_targets,
        }
        self._empty_candidate_report_path.parent.mkdir(parents=True, exist_ok=True)
        self._empty_candidate_report_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _load_semantic_assets(
        self,
        *,
        object_ids: set[str],
        object_order: Mapping[str, int],
    ) -> list[Asset]:
        assets = list(
            self._repo.list_assets(
                AssetFilter(tenant_id=TenantId(self._tenant_id), type="FURNITURE")
            )
        )
        if self._asset_ids:
            assets = [asset for asset in assets if str(asset.id) in self._asset_ids]
        assets = [
            asset
            for asset in assets
            if asset_matches_semantic_objects(asset, object_ids)
        ]

        matched_tokens: set[str] = set()
        for asset in assets:
            matched_tokens.update(asset_semantic_tokens(asset) & object_ids)
        self._missing_asset_ids = sorted(object_ids - matched_tokens)

        assets.sort(
            key=lambda asset: (
                semantic_asset_order(asset, object_order),
                str(asset.id),
            )
        )
        if self._limit is not None:
            assets = assets[: max(0, self._limit)]
        return assets

    def _build_group_targets(
        self,
        *,
        assets: Sequence[Asset],
        object_order: Mapping[str, int],
    ) -> list[ObjaverseCurationTarget]:
        grouped_assets: dict[str, list[Asset]] = {}
        for asset in assets:
            family_key = asset_semantic_family(asset, object_order)
            if family_key is None:
                continue
            grouped_assets.setdefault(family_key, []).append(asset)

        targets: list[ObjaverseCurationTarget] = []
        for family_key in object_order:
            members = grouped_assets.get(family_key)
            if not members:
                continue
            members.sort(key=lambda asset: str(asset.id))
            for style in self._styles:
                style_key = canonicalize_style_key(style)
                representative = members[0]
                representative_style_tags: list[str] = []
                seen_tags: set[str] = set()
                for member in members:
                    for raw_tag in member.style_tags:
                        tag = str(raw_tag).strip()
                        if not tag or tag in seen_tags:
                            continue
                        seen_tags.add(tag)
                        representative_style_tags.append(tag)
                material = (
                    representative.material.strip()
                    if isinstance(representative.material, str)
                    and representative.material.strip()
                    else None
                )
                query_asset = representative.model_copy(
                    update={
                        "id": AssetId(family_key),
                        "name": family_key.replace("_", " ").title(),
                    }
                )
                query = RetrievalQuery(
                    asset=query_asset,
                    style=style,
                    style_key=style_key,
                    material=material,
                    style_tags=tuple(representative_style_tags),
                )
                targets.append(
                    ObjaverseCurationTarget(
                        family_key=family_key,
                        assets=tuple(members),
                        query=query,
                        query_text=build_query_text(query),
                        candidate_results=(),
                        candidates=(),
                    )
                )
        return targets

    def _serialize_target(self, target: ObjaverseCurationTarget) -> JsonObject:
        key = curation_target_key(asset_id=target.asset_id, style_key=target.style_key)
        member_keys = [
            curation_target_key(asset_id=str(asset.id), style_key=target.style_key)
            for asset in target.assets
        ]
        member_preview = ", ".join(member_keys[:4])
        if len(member_keys) > 4:
            member_preview = f"{member_preview}, +{len(member_keys) - 4} more"
        return {
            "key": key,
            "asset_id": target.asset_id,
            "name": target.query.asset.name,
            "style": target.query.style,
            "style_key": target.style_key,
            "material": target.query.material,
            "query_text": target.query_text,
            "candidate_count": len(target.candidates),
            "member_asset_ids": [str(asset.id) for asset in target.assets],
            "member_keys": member_keys,
            "display_label": (
                f"{target.query.asset.name} - {target.style_key} "
                f"({len(target.assets)} assets: {member_preview})"
            ),
        }

    def _candidate_path(
        self,
        *,
        asset_id: str,
        style_key: str,
        rank: int,
        uid: str,
    ) -> Path:
        filename = f"rank_{rank:02d}_{safe_filename_segment(uid)}.glb"
        return (
            self._candidates_public_dir
            / safe_filename_segment(self._tenant_id)
            / safe_filename_segment(style_key)
            / safe_filename_segment(asset_id)
            / filename
        )

    def _delete_staging_dir(self, target: ObjaverseCurationTarget) -> None:
        staging_dirs = {
            candidate.path.parent
            for candidate in target.candidates
            if candidate.source == "objaverse"
        }
        for staging_dir in staging_dirs:
            if not staging_dir.exists():
                continue
            if not path_within(staging_dir, self._candidates_public_dir):
                raise ValueError(
                    f"Refusing to delete unsafe staging path: {staging_dir}"
                )
            shutil.rmtree(staging_dir)

    def _persist_curated_metadata(
        self,
        *,
        asset: Asset,
        target: ObjaverseCurationTarget,
        metadata: JsonObject,
    ) -> None:
        attributes = dict(asset.attributes or {})
        variants_raw = read_json_object(attributes.get("model_variants")) or {}
        variants_raw[target.style_key] = metadata
        attributes["model_variants"] = variants_raw
        attributes["model_3d"] = metadata
        attributes["model_url"] = metadata.get("url")

        updated_asset = asset.model_copy(update={"attributes": attributes})
        self._repo.upsert_asset(updated_asset)


class ManualInventoryGlbReviewHandler(BaseHTTPRequestHandler):
    inspector: InventoryGlbOrientationInspector
    vendor_dir: Path

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path in ("", "/"):
                self._send_html(self._build_index(parsed.query))
                return
            if parsed.path == "/viewer":
                self._send_html(self._build_viewer(parsed.query))
                return
            if parsed.path == "/model":
                params = parse_qs(parsed.query)
                asset_id = params.get("asset_id", [""])[0]
                style_key = params.get("style_key", [""])[0]
                model_path = self.inspector.resolve_model_path(
                    asset_id=asset_id,
                    style_key=style_key,
                )
                self._send_file(model_path, default_content_type="model/gltf-binary")
                return
            if parsed.path.startswith("/vendor/"):
                self._send_vendor_file(parsed.path.removeprefix("/vendor/"))
                return
            self._send_plain("Not found.", status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_plain(str(exc), status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path not in ("/save", "/delete"):
                self._send_plain("Not found.", status=HTTPStatus.NOT_FOUND)
                return

            length_raw = self.headers.get("Content-Length")
            length = int(length_raw) if length_raw and length_raw.isdigit() else 0
            body = self.rfile.read(length).decode("utf-8")
            params = parse_qs(body)
            asset_id = params.get("asset_id", [""])[0]
            style_key = params.get("style_key", [""])[0]

            if parsed.path == "/delete":
                payload: JsonObject = {
                    "asset_id": asset_id,
                    "style_key": style_key,
                    "delete_file": params.get("delete_file", [""])[0],
                }
                self.inspector.delete_variant(payload)
                self.send_response(int(HTTPStatus.SEE_OTHER))
                self.send_header("Location", "/?deleted=1")
                self.end_headers()
                return

            offset = params.get("offset", ["0"])[0]
            tilt_x = params.get("tilt_x", ["0"])[0]
            tilt_z = params.get("tilt_z", ["0"])[0]
            contexts = (
                ["all"] if params.get("apply_all", [""])[0] == "on" else ["panel"]
            )
            notes = params.get("notes", [""])[0].strip() or None
            payload: JsonObject = {
                "asset_id": asset_id,
                "style_key": style_key,
                "rotation_deg_offset": normalize_cardinal_offset(float(offset)),
                "rotation_deg_x": normalize_quarter_turn_tilt(float(tilt_x)),
                "rotation_deg_z": normalize_quarter_turn_tilt(float(tilt_z)),
                "contexts": contexts,
                "notes": notes,
            }
            self.inspector.save_orientation(payload)
            query = urlencode(
                {
                    "asset_id": asset_id,
                    "style_key": style_key,
                    "offset": str(payload["rotation_deg_offset"]),
                    "tilt_x": str(payload["rotation_deg_x"]),
                    "tilt_z": str(payload["rotation_deg_z"]),
                    "saved": "1",
                }
            )
            self.send_response(int(HTTPStatus.SEE_OTHER))
            self.send_header("Location", f"/?{query}")
            self.end_headers()
        except Exception as exc:
            self._send_plain(str(exc), status=HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _build_index(self, query: str) -> bytes:
        params = parse_qs(query)
        targets = self.inspector.list_targets()
        requested_asset_id = params.get("asset_id", [""])[0]
        requested_style_key = params.get("style_key", [""])[0]
        target_value = params.get("target", [""])[0]
        if target_value and "::" in target_value:
            requested_asset_id, requested_style_key = target_value.split("::", 1)
        if not requested_asset_id or not requested_style_key:
            first = first_target(targets)
            if first is not None:
                requested_asset_id, requested_style_key = first

        target = find_target(
            targets=targets,
            asset_id=requested_asset_id,
            style_key=requested_style_key,
        )
        offset = selected_offset(target, params.get("offset", [None])[0])
        tilt_x = selected_tilt(target, params.get("tilt_x", [None])[0], axis="x")
        tilt_z = selected_tilt(target, params.get("tilt_z", [None])[0], axis="z")
        saved = params.get("saved", [""])[0] == "1"
        deleted = params.get("deleted", [""])[0] == "1"
        selected_key = (
            f"{requested_asset_id}::{requested_style_key}"
            if requested_asset_id and requested_style_key
            else ""
        )

        option_rows = "\n".join(
            self._target_option(target=target_row, selected_key=selected_key)
            for target_row in targets
        )
        offset_links = " ".join(
            self._offset_link(
                asset_id=requested_asset_id,
                style_key=requested_style_key,
                offset=value,
                tilt_x=tilt_x,
                tilt_z=tilt_z,
                selected=offset == value,
            )
            for value in OFFSET_LABELS
        )
        tilt_x_links = " ".join(
            (
                self._tilt_link(
                    asset_id=requested_asset_id,
                    style_key=requested_style_key,
                    offset=offset,
                    tilt_x=adjusted_tilt(tilt_x, -90),
                    tilt_z=tilt_z,
                    label="X -90",
                ),
                self._tilt_link(
                    asset_id=requested_asset_id,
                    style_key=requested_style_key,
                    offset=offset,
                    tilt_x=adjusted_tilt(tilt_x, 90),
                    tilt_z=tilt_z,
                    label="X +90",
                ),
            )
        )
        tilt_z_links = " ".join(
            (
                self._tilt_link(
                    asset_id=requested_asset_id,
                    style_key=requested_style_key,
                    offset=offset,
                    tilt_x=tilt_x,
                    tilt_z=adjusted_tilt(tilt_z, -90),
                    label="Z -90",
                ),
                self._tilt_link(
                    asset_id=requested_asset_id,
                    style_key=requested_style_key,
                    offset=offset,
                    tilt_x=tilt_x,
                    tilt_z=adjusted_tilt(tilt_z, 90),
                    label="Z +90",
                ),
            )
        )
        tilt_reset_link = self._tilt_link(
            asset_id=requested_asset_id,
            style_key=requested_style_key,
            offset=offset,
            tilt_x=0,
            tilt_z=0,
            label="Reset tilt",
        )
        viewer_src = (
            "/viewer?"
            + urlencode(
                {
                    "asset_id": requested_asset_id,
                    "style_key": requested_style_key,
                    "offset": str(offset),
                    "tilt_x": str(tilt_x),
                    "tilt_z": str(tilt_z),
                }
            )
            if target is not None
            else ""
        )
        status = (
            "Saved orientation to DB."
            if saved
            else "Deleted selected GLB variant from DB."
            if deleted
            else f"Loaded {len(targets)} GLB inventory variants."
        )
        target_name = (
            target.get("display_label") if target is not None else "No model selected"
        )
        notes = ""
        if target is not None and isinstance(target.get("orientation_review"), dict):
            notes = str(target["orientation_review"].get("notes") or "")

        document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Manual GLB Orientation Review</title>
  <style>
    body {{ margin: 0; font: 14px system-ui, sans-serif; color: #111; background: #f6f6f4; }}
    .layout {{ display: grid; grid-template-columns: 360px 1fr; height: 100vh; }}
    aside {{ padding: 14px; border-right: 1px solid #ccc; overflow: auto; background: #fff; }}
    main {{ display: grid; grid-template-rows: auto 1fr; min-width: 0; }}
    h1 {{ font-size: 18px; margin: 0 0 10px; }}
    label {{ display: block; font-weight: 700; margin: 12px 0 6px; }}
    select, textarea, button {{ width: 100%; font: inherit; }}
    select {{ height: 45vh; }}
    textarea {{ min-height: 70px; }}
    .topbar {{ padding: 10px 14px; border-bottom: 1px solid #ccc; background: #fff; }}
    .name {{ font-weight: 800; }}
    .muted {{ color: #666; font-size: 12px; }}
    .status {{ color: #176b32; margin: 8px 0; }}
    .offsets {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 8px 0 12px; }}
    .offsets a {{ display: block; padding: 10px; border: 1px solid #bbb; border-radius: 6px; text-align: center; color: #111; text-decoration: none; background: #fafafa; }}
    .offsets a.selected {{ border-color: #176b32; background: #dcfce7; font-weight: 800; }}
    .tilts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 8px 0; }}
    .tilts a {{ display: block; padding: 10px; border: 1px solid #bbb; border-radius: 6px; text-align: center; color: #111; text-decoration: none; background: #fafafa; }}
    .tilts.single {{ grid-template-columns: 1fr; }}
    iframe {{ width: 100%; height: 100%; border: 0; background: #202020; }}
    .save {{ margin-top: 12px; padding: 11px; border: 0; border-radius: 6px; background: #176b32; color: #fff; font-weight: 800; cursor: pointer; }}
    .delete {{ margin-top: 10px; padding: 11px; border: 0; border-radius: 6px; background: #b42318; color: #fff; font-weight: 800; cursor: pointer; }}
    .finish {{ margin-top: 10px; padding: 11px; border: 0; border-radius: 6px; background: #444; color: #fff; font-weight: 800; cursor: pointer; }}
    .apply {{ display: flex; align-items: center; gap: 8px; font-weight: 400; }}
    .apply input {{ width: auto; }}
    @media (max-width: 900px) {{ .layout {{ grid-template-columns: 1fr; grid-template-rows: auto 65vh; }} select {{ height: 28vh; }} }}
  </style>
</head>
<body>
  <div class="layout">
    <aside>
      <h1>Manual GLB Orientation Review</h1>
      <div class="status">{escape(status)}</div>
      <form method="get" action="/">
        <label for="target">Inventory objects A-Z</label>
        <select id="target" name="target" onchange="const [asset, style] = this.value.split('::'); location.href='/?asset_id=' + encodeURIComponent(asset) + '&style_key=' + encodeURIComponent(style);">
          {option_rows}
        </select>
        <noscript><button type="submit">Open selected</button></noscript>
      </form>

      <label>Pick which model axis is the real front</label>
      <div class="offsets">{offset_links}</div>

      <label>Stand the object upright if needed</label>
      <div class="muted">Current tilt: X={tilt_x}deg, Z={tilt_z}deg</div>
      <div class="tilts">{tilt_x_links}</div>
      <div class="tilts">{tilt_z_links}</div>
      <div class="tilts single">{tilt_reset_link}</div>

      <form method="post" action="/save">
        <input type="hidden" name="asset_id" value="{escape(requested_asset_id)}">
        <input type="hidden" name="style_key" value="{escape(requested_style_key)}">
        <input type="hidden" name="offset" value="{offset}">
        <input type="hidden" name="tilt_x" value="{tilt_x}">
        <input type="hidden" name="tilt_z" value="{tilt_z}">
        <label class="apply"><input type="checkbox" name="apply_all"> Apply to all 3D contexts</label>
        <label for="notes">Notes</label>
        <textarea id="notes" name="notes">{escape(notes)}</textarea>
        <button class="save" type="submit">Save orientation to DB</button>
      </form>

      <form method="post" action="/delete" onsubmit="return confirm('Delete this selected GLB variant from DB?')">
        <input type="hidden" name="asset_id" value="{escape(requested_asset_id)}">
        <input type="hidden" name="style_key" value="{escape(requested_style_key)}">
        <label class="apply"><input type="checkbox" name="delete_file"> Also delete local .glb file</label>
        <button class="delete" type="submit">Delete selected GLB</button>
      </form>

      <p class="muted">Workflow: first stand the object upright with X/Z tilt if it is lying on its side, then choose the front offset, then save.</p>
    </aside>
    <main>
      <div class="topbar">
        <div class="name">{escape(target_name)}</div>
        <div class="muted">offset={offset}deg | {escape(OFFSET_LABELS[offset])} | tiltX={tilt_x}deg | tiltZ={tilt_z}deg</div>
      </div>
      <iframe src="{escape(viewer_src)}" title="GLB viewer"></iframe>
    </main>
  </div>
</body>
</html>"""
        return document.encode("utf-8")

    def _build_viewer(self, query: str) -> bytes:
        params = parse_qs(query)
        asset_id = params.get("asset_id", [""])[0]
        style_key = params.get("style_key", [""])[0]
        offset = selected_offset(None, params.get("offset", ["0"])[0])
        tilt_x = selected_tilt(None, params.get("tilt_x", ["0"])[0], axis="x")
        tilt_z = selected_tilt(None, params.get("tilt_z", ["0"])[0], axis="z")
        model_url = "/model?" + urlencode(
            {"asset_id": asset_id, "style_key": style_key}
        )
        targets = self.inspector.list_targets()
        target = find_target(targets=targets, asset_id=asset_id, style_key=style_key)
        base_rotation_deg = float((target or {}).get("base_rotation_deg") or 0)
        calibration = (target or {}).get("preview_calibration")
        calibration_json = json.dumps(
            calibration if isinstance(calibration, dict) else None
        )
        model_url_json = json.dumps(model_url)
        status_text = (
            f"Loaded {asset_id} [{style_key}] "
            f"offset={offset}deg tiltX={tilt_x}deg tiltZ={tilt_z}deg"
        )
        status_text_json = json.dumps(status_text)
        document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {{ margin: 0; overflow: hidden; background: #202020; color: white; font: 13px system-ui, sans-serif; }}
    #status {{ position: fixed; left: 12px; top: 12px; padding: 8px 10px; background: rgba(0,0,0,.7); border-radius: 6px; z-index: 5; }}
    #hint {{ position: fixed; left: 12px; bottom: 12px; padding: 8px 10px; background: rgba(0,0,0,.7); border-radius: 6px; z-index: 5; }}
  </style>
</head>
<body>
  <div id="status">Loading model...</div>
  <div id="hint">White arrow = 2D plan front. Drag to orbit, scroll to zoom.</div>
  <script>
    window.addEventListener('error', (event) => {{
      const status = document.getElementById('status');
      if (status) status.textContent = 'Viewer error: ' + event.message;
    }});
    window.addEventListener('unhandledrejection', (event) => {{
      const status = document.getElementById('status');
      const reason = event.reason && event.reason.message ? event.reason.message : String(event.reason);
      if (status) status.textContent = 'Viewer error: ' + reason;
    }});
  </script>
  <script type="module">
    import * as THREE from '/vendor/three.module.js';
    import {{ OrbitControls }} from '/vendor/OrbitControls.js';
    import {{ GLTFLoader }} from '/vendor/loaders/GLTFLoader.js';

    const status = document.getElementById('status');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x202020);
    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.shadowMap.enabled = true;
    document.body.appendChild(renderer.domElement);

    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 100);
    camera.position.set(2.6, 2.0, 3.0);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 0.55, 0);

    const world = new THREE.Group();
    scene.add(world);
    scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.7));
    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(4, 6, 5);
    key.castShadow = true;
    scene.add(key);

    const grid = new THREE.GridHelper(4, 8, 0x999999, 0x444444);
    world.add(grid);
    const arrow = new THREE.ArrowHelper(
      new THREE.Vector3(0, 0, -1),
      new THREE.Vector3(0, 0.08, 1.0),
      1.5,
      0xffffff,
      0.25,
      0.12
    );
    world.add(arrow);

    function makeLabel(text, x, y, z, color = '#ffffff') {{
      const c = document.createElement('canvas');
      c.width = 256;
      c.height = 64;
      const ctx = c.getContext('2d');
      ctx.font = '700 28px system-ui';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = color;
      ctx.fillText(text, 128, 32);
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({{ map: new THREE.CanvasTexture(c), transparent: true }}));
      sprite.position.set(x, y, z);
      sprite.scale.set(0.75, 0.18, 1);
      world.add(sprite);
    }}
    makeLabel('PLAN FRONT', 0, 0.1, -0.85);
    makeLabel('BACK', 0, 0.1, 1.0, '#bdbdbd');
    makeLabel('LEFT', -1.25, 0.1, 0, '#bdbdbd');
    makeLabel('RIGHT', 1.25, 0.1, 0, '#bdbdbd');

    window.addEventListener('resize', resize);
    resize();

    function tick() {{
      requestAnimationFrame(tick);
      controls.update();
      renderer.render(scene, camera);
    }}
    tick();

    main().catch((error) => {{
      status.textContent = 'Viewer error: ' + (error instanceof Error ? error.message : String(error));
      console.error(error);
    }});

    async function main() {{
      const modelUrl = {model_url_json};
      status.textContent = 'Fetching GLB...';
      const response = await fetch(modelUrl, {{ cache: 'no-store' }});
      if (!response.ok) {{
        throw new Error(`GLB request failed: ${{response.status}} ${{response.statusText}}`);
      }}
      const bytes = await response.arrayBuffer();
      status.textContent = `Parsing GLB (${{formatBytes(bytes.byteLength)}})...`;
      const gltf = await new Promise((resolve, reject) => {{
        new GLTFLoader().parse(bytes, '', resolve, reject);
      }});
      const sceneSource = gltf.scene || (Array.isArray(gltf.scenes) ? gltf.scenes[0] : null);
      if (!sceneSource || typeof sceneSource.clone !== 'function') {{
        throw new Error('GLB loaded, but it does not contain a usable scene.');
      }}

      const root = new THREE.Group();
      const yawRoot = new THREE.Group();
      const tiltRoot = new THREE.Group();
      const calibrationRoot = new THREE.Group();
      const model = sceneSource.clone(true);
      yawRoot.rotation.y = THREE.MathUtils.degToRad({base_rotation_deg + offset});
      tiltRoot.rotation.x = THREE.MathUtils.degToRad({tilt_x});
      tiltRoot.rotation.z = THREE.MathUtils.degToRad({tilt_z});
      calibrationRoot.add(model);
      tiltRoot.add(calibrationRoot);
      yawRoot.add(tiltRoot);
      applyCalibrationMatrix(calibrationRoot, {calibration_json});
      root.add(yawRoot);
      world.add(root);
      root.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(root);
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      if (!Number.isFinite(maxDim) || maxDim <= 0) {{
        throw new Error('GLB parsed, but its bounds are empty.');
      }}
      const scale = Math.min(
        1.8 / Math.max(size.x, size.z, 0.001),
        1.6 / Math.max(size.x, size.y, size.z, 0.001)
      );
      root.scale.setScalar(scale);
      root.updateMatrixWorld(true);
      const fitted = new THREE.Box3().setFromObject(root);
      const center = fitted.getCenter(new THREE.Vector3());
      root.position.x -= center.x;
      root.position.z -= center.z;
      root.position.y -= fitted.min.y;
      root.traverse((node) => {{
        if (node.isMesh) {{
          node.castShadow = true;
          node.receiveShadow = true;
          if (node.material) {{
            const materials = Array.isArray(node.material) ? node.material : [node.material];
            for (const material of materials) {{
              material.side = THREE.DoubleSide;
              material.needsUpdate = true;
            }}
          }}
        }}
      }});
      status.textContent = {status_text_json};
    }}

    function resize() {{
      renderer.setSize(window.innerWidth, window.innerHeight);
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
    }}

    function formatBytes(size) {{
      if (!Number.isFinite(size)) return 'unknown size';
      if (size < 1024 * 1024) return `${{Math.round(size / 1024)}} KB`;
      return `${{(size / (1024 * 1024)).toFixed(1)}} MB`;
    }}

    function applyCalibrationMatrix(root, calibration) {{
      if (!calibration || calibration.status !== 'calibrated') return;
      const matrixRows = calibration.rotationMatrix || calibration.rotation_matrix;
      if (!Array.isArray(matrixRows) || matrixRows.length !== 4) return;
      const m = new THREE.Matrix4();
      m.set(
        matrixRows[0][0], matrixRows[0][1], matrixRows[0][2], matrixRows[0][3],
        matrixRows[1][0], matrixRows[1][1], matrixRows[1][2], matrixRows[1][3],
        matrixRows[2][0], matrixRows[2][1], matrixRows[2][2], matrixRows[2][3],
        matrixRows[3][0], matrixRows[3][1], matrixRows[3][2], matrixRows[3][3]
      );
      root.applyMatrix4(m);
    }}
  </script>
</body>
</html>"""
        return document.encode("utf-8")

    def _target_option(self, *, target: JsonObject, selected_key: str) -> str:
        key = target_key(target)
        label = (
            target.get("display_label")
            or f"{target.get('name')} - {target.get('style_key')}"
        )
        selected = " selected" if key == selected_key else ""
        return f'<option value="{escape(key)}"{selected}>{escape(label)}</option>'

    def _offset_link(
        self,
        *,
        asset_id: str,
        style_key: str,
        offset: int,
        tilt_x: int,
        tilt_z: int,
        selected: bool,
    ) -> str:
        query = urlencode(
            {
                "asset_id": asset_id,
                "style_key": style_key,
                "offset": str(offset),
                "tilt_x": str(tilt_x),
                "tilt_z": str(tilt_z),
            }
        )
        class_name = " selected" if selected else ""
        return f'<a class="{class_name}" href="/?{query}">{escape(OFFSET_LABELS[offset])}<br>{offset}deg</a>'

    def _tilt_link(
        self,
        *,
        asset_id: str,
        style_key: str,
        offset: int,
        tilt_x: int,
        tilt_z: int,
        label: str,
    ) -> str:
        query = urlencode(
            {
                "asset_id": asset_id,
                "style_key": style_key,
                "offset": str(offset),
                "tilt_x": str(tilt_x),
                "tilt_z": str(tilt_z),
            }
        )
        return f'<a href="/?{query}">{escape(label)}<br>X={tilt_x} Z={tilt_z}</a>'

    def _send_vendor_file(self, relative_path: str) -> None:
        allowed_files = {
            "three.module.js": self.vendor_dir / "three.module.js",
            "OrbitControls.js": self.vendor_dir / "OrbitControls.js",
            "loaders/GLTFLoader.js": self.vendor_dir / "loaders" / "GLTFLoader.js",
            "utils/BufferGeometryUtils.js": self.vendor_dir
            / "utils"
            / "BufferGeometryUtils.js",
        }
        path = allowed_files.get(relative_path)
        if path is None or not path.exists():
            self._send_plain("Vendor file not found.", status=HTTPStatus.NOT_FOUND)
            return
        self._send_file(path, default_content_type="text/javascript")

    def _send_file(self, path: Path, *, default_content_type: str) -> None:
        content_type = mimetypes.guess_type(path.name)[0] or default_content_type
        body = path.read_bytes()
        self.send_response(int(HTTPStatus.OK))
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, body: bytes) -> None:
        self.send_response(int(HTTPStatus.OK))
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_plain(self, text: str, *, status: HTTPStatus) -> None:
        body = text.encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ObjaverseCandidateCurationHandler(ManualInventoryGlbReviewHandler):
    curator: ObjaverseCandidateCurator

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path in ("", "/"):
                self._send_html(self._build_curation_index(parsed.query))
                return
            if parsed.path == "/viewer":
                self._send_html(self._build_curation_viewer(parsed.query))
                return
            if parsed.path == "/candidate-model":
                params = parse_qs(parsed.query)
                key = params.get("key", [""])[0]
                rank = int(params.get("rank", ["1"])[0])
                model_path = self.curator.resolve_candidate_path(key=key, rank=rank)
                self._send_file(model_path, default_content_type="model/gltf-binary")
                return
            if parsed.path.startswith("/vendor/"):
                self._send_vendor_file(parsed.path.removeprefix("/vendor/"))
                return
            self._send_plain("Not found.", status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_plain(str(exc), status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path not in (
                "/accept",
                "/delete-candidate",
                "/finish",
                "/reject",
                "/save-candidate",
            ):
                self._send_plain("Not found.", status=HTTPStatus.NOT_FOUND)
                return

            params = self._read_form_body()
            key = params.get("key", [""])[0]
            if parsed.path == "/finish":
                self.curator.finish_target({"key": key})
                self._redirect_to_next(status_key="finished")
                return
            if parsed.path == "/reject":
                self.curator.reject_target({"key": key})
                self._redirect_to_next(status_key="rejected")
                return
            if parsed.path == "/delete-candidate":
                rank = params.get("rank", ["1"])[0]
                destination_asset_id = params.get("destination_asset_id", [""])[0]
                self.curator.delete_candidate({"key": key, "rank": rank})
                self._redirect_to_target_or_next(
                    key=key,
                    rank=rank,
                    status_key="deleted_candidate",
                    destination_asset_id=destination_asset_id,
                    prefer_new=True,
                )
                return

            offset = params.get("offset", ["0"])[0]
            tilt_x = params.get("tilt_x", ["0"])[0]
            tilt_z = params.get("tilt_z", ["0"])[0]
            contexts = (
                ["all"] if params.get("apply_all", [""])[0] == "on" else ["panel"]
            )
            payload: JsonObject = {
                "key": key,
                "rank": params.get("rank", ["1"])[0],
                "rotation_deg_offset": normalize_cardinal_offset(float(offset)),
                "rotation_deg_x": normalize_quarter_turn_tilt(float(tilt_x)),
                "rotation_deg_z": normalize_quarter_turn_tilt(float(tilt_z)),
                "contexts": contexts,
                "destination_asset_id": params.get("destination_asset_id", [""])[0],
                "delete_unselected_existing": params.get(
                    "delete_unselected_existing", [""]
                )[0],
                "notes": params.get("notes", [""])[0].strip() or None,
            }
            if parsed.path == "/accept":
                self.curator.accept_candidate(payload)
                self._redirect_to_next(status_key="accepted")
                return
            self.curator.save_candidate(payload)
            self._redirect_to_target_or_next(
                key=key,
                rank=params.get("rank", ["1"])[0],
                status_key="saved_candidate",
                destination_asset_id="",
                prefer_new=False,
            )
        except Exception as exc:
            self._send_plain(str(exc), status=HTTPStatus.BAD_REQUEST)

    def _read_form_body(self) -> dict[str, list[str]]:
        length_raw = self.headers.get("Content-Length")
        length = int(length_raw) if length_raw and length_raw.isdigit() else 0
        body = self.rfile.read(length).decode("utf-8")
        return parse_qs(body)

    def _redirect_to_next(self, *, status_key: str) -> None:
        next_key = self.curator.first_pending_key()
        if status_key == "accepted":
            query = {"accepted": "1"}
        elif status_key == "finished":
            query = {"finished": "1"}
        else:
            query = {"rejected": "1"}
        if next_key:
            query["key"] = next_key
        self.send_response(int(HTTPStatus.SEE_OTHER))
        self.send_header("Location", "/?" + urlencode(query))
        self.end_headers()

    def _redirect_to_target_or_next(
        self,
        *,
        key: str,
        rank: str,
        status_key: str,
        destination_asset_id: str,
        prefer_new: bool,
    ) -> None:
        target = self.curator.target_for_key(key)
        query = {status_key: "1"}
        if target is not None:
            query["key"] = self.curator.canonical_target_key(key)
            query["rank"] = rank
            if destination_asset_id:
                query["destination_asset_id"] = destination_asset_id
            if prefer_new:
                query["prefer_new"] = "1"
        else:
            next_key = self.curator.first_pending_key()
            if next_key:
                query["key"] = next_key
        self.send_response(int(HTTPStatus.SEE_OTHER))
        self.send_header("Location", "/?" + urlencode(query))
        self.end_headers()

    def _build_curation_index(self, query: str) -> bytes:
        params = parse_qs(query)
        targets = self.curator.list_targets()
        requested_key = params.get("key", [""])[0]
        if not requested_key:
            requested_key = self.curator.first_pending_key()
        target = self.curator.target_for_key(requested_key)
        if target is None and targets:
            requested_key = str(targets[0].get("key") or "")
            target = self.curator.target_for_key(requested_key)
        if target is not None:
            requested_key = self.curator.canonical_target_key(requested_key)
        if target is not None:
            target = self.curator.ensure_candidate_files(target)

        selected_destination_asset_id = (
            params.get("destination_asset_id", [None])[0]
            or (
                self.curator.next_destination_asset_id(
                    key=requested_key,
                    target=target,
                )
                if target is not None
                else None
            )
            or ""
        )
        default_rank = (
            self.curator.default_candidate_rank(
                key=requested_key,
                target=target,
                destination_asset_id=selected_destination_asset_id or None,
                prefer_new=params.get("prefer_new", [""])[0] == "1",
            )
            if target is not None
            else None
        )
        rank = selected_candidate_rank(
            target,
            params.get("rank", [None])[0],
            default_rank=default_rank,
        )
        candidate = (
            self.curator.candidate_by_rank(target=target, rank=rank)
            if target is not None
            else None
        )
        offset = selected_offset(None, params.get("offset", [None])[0])
        tilt_x = selected_tilt(None, params.get("tilt_x", [None])[0], axis="x")
        tilt_z = selected_tilt(None, params.get("tilt_z", [None])[0], axis="z")

        stats = self.curator.stats()
        accepted = params.get("accepted", [""])[0] == "1"
        rejected = params.get("rejected", [""])[0] == "1"
        finished = params.get("finished", [""])[0] == "1"
        saved_candidate = params.get("saved_candidate", [""])[0] == "1"
        deleted_candidate = params.get("deleted_candidate", [""])[0] == "1"
        status = (
            "Saved this candidate to DB."
            if saved_candidate
            else "Deleted this candidate."
            if deleted_candidate
            else "Finished target and cleaned remaining new staged files."
            if finished
            else "Accepted candidate, overwrote group metadata/orientation, then cleaned new staging."
            if accepted
            else "Rejected target candidates and cleaned new staging."
            if rejected
            else (
                f"Pending {stats['pending']} grouped target(s). "
                f"Saved {stats['saved_candidates']} candidate(s), "
                f"deleted {stats['deleted_candidates']}, "
                f"finished {stats['finished']} target(s), "
                f"empty {stats['empty_candidate_targets']} target(s)."
            )
        )

        target_options = "\n".join(
            self._curation_target_option(target=row, selected_key=requested_key)
            for row in targets
        )
        candidate_options = (
            self._candidate_options(target=target, selected_rank=rank)
            if target is not None
            else ""
        )
        destination_control = self._destination_asset_control(
            target=target,
            candidate=candidate,
            selected_asset_id=selected_destination_asset_id,
        )
        errors = (
            "<br>".join(
                escape(error)
                for error in self.curator.download_errors_for_key(requested_key)
            )
            if requested_key
            else ""
        )
        error_block = f'<div class="error">{errors}</div>' if errors else ""
        target_name = (
            f"{target.query.asset.name} - {target.style_key} "
            f"({len(target.assets)} grouped assets)"
            if target is not None
            else "No pending target"
        )
        member_text = (
            ", ".join(
                curation_target_key(asset_id=str(asset.id), style_key=target.style_key)
                for asset in target.assets
            )
            if target is not None
            else ""
        )
        query_text = target.query_text if target is not None else ""
        offset_links = " ".join(
            self._curation_offset_link(
                key=requested_key,
                rank=rank,
                offset=value,
                tilt_x=tilt_x,
                tilt_z=tilt_z,
                selected=offset == value,
            )
            for value in OFFSET_LABELS
        )
        tilt_x_links = " ".join(
            (
                self._curation_tilt_link(
                    key=requested_key,
                    rank=rank,
                    offset=offset,
                    tilt_x=adjusted_tilt(tilt_x, -90),
                    tilt_z=tilt_z,
                    label="X -90",
                ),
                self._curation_tilt_link(
                    key=requested_key,
                    rank=rank,
                    offset=offset,
                    tilt_x=adjusted_tilt(tilt_x, 90),
                    tilt_z=tilt_z,
                    label="X +90",
                ),
            )
        )
        tilt_z_links = " ".join(
            (
                self._curation_tilt_link(
                    key=requested_key,
                    rank=rank,
                    offset=offset,
                    tilt_x=tilt_x,
                    tilt_z=adjusted_tilt(tilt_z, -90),
                    label="Z -90",
                ),
                self._curation_tilt_link(
                    key=requested_key,
                    rank=rank,
                    offset=offset,
                    tilt_x=tilt_x,
                    tilt_z=adjusted_tilt(tilt_z, 90),
                    label="Z +90",
                ),
            )
        )
        tilt_reset_link = self._curation_tilt_link(
            key=requested_key,
            rank=rank,
            offset=offset,
            tilt_x=0,
            tilt_z=0,
            label="Reset tilt",
        )
        viewer_src = (
            "/viewer?"
            + urlencode(
                {
                    "key": requested_key,
                    "rank": str(rank),
                    "offset": str(offset),
                    "tilt_x": str(tilt_x),
                    "tilt_z": str(tilt_z),
                }
            )
            if target is not None
            else ""
        )
        candidate_select_js = (
            "location.href='/?key=' + encodeURIComponent(document.getElementById('target').value)"
            " + '&rank=' + encodeURIComponent(this.value)"
            f" + '&offset={offset}&tilt_x={tilt_x}&tilt_z={tilt_z}';"
        )
        delete_candidate_text = (
            "Delete current candidate from DB and local model file?"
            if candidate is not None and candidate.source == "existing"
            else "Delete this staged new candidate file?"
        )
        save_button_text = (
            "Save orientation for this current candidate"
            if candidate is not None and candidate.source == "existing"
            else "Save this new candidate to selected asset"
        )
        delete_destination_asset_id = (
            candidate.source_asset_id
            if candidate is not None
            and candidate.source == "existing"
            and candidate.source_asset_id is not None
            else selected_destination_asset_id
        )

        document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Objaverse Candidate Curation</title>
  <style>
    body {{ margin: 0; font: 14px system-ui, sans-serif; color: #111; background: #f6f6f4; }}
    .layout {{ display: grid; grid-template-columns: 390px 1fr; height: 100vh; }}
    aside {{ padding: 14px; border-right: 1px solid #ccc; overflow: auto; background: #fff; }}
    main {{ display: grid; grid-template-rows: auto 1fr; min-width: 0; }}
    h1 {{ font-size: 18px; margin: 0 0 10px; }}
    label {{ display: block; font-weight: 700; margin: 12px 0 6px; }}
    select, textarea, button {{ width: 100%; font: inherit; }}
    select.target {{ height: 31vh; }}
    textarea {{ min-height: 70px; }}
    .topbar {{ padding: 10px 14px; border-bottom: 1px solid #ccc; background: #fff; }}
    .name {{ font-weight: 800; }}
    .muted {{ color: #666; font-size: 12px; }}
    .status {{ color: #176b32; margin: 8px 0; }}
    .error {{ color: #b42318; margin: 8px 0; font-size: 12px; }}
    .offsets {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 8px 0 12px; }}
    .offsets a {{ display: block; padding: 10px; border: 1px solid #bbb; border-radius: 6px; text-align: center; color: #111; text-decoration: none; background: #fafafa; }}
    .offsets a.selected {{ border-color: #176b32; background: #dcfce7; font-weight: 800; }}
    .tilts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 8px 0; }}
    .tilts a {{ display: block; padding: 10px; border: 1px solid #bbb; border-radius: 6px; text-align: center; color: #111; text-decoration: none; background: #fafafa; }}
    .tilts.single {{ grid-template-columns: 1fr; }}
    iframe {{ width: 100%; height: 100%; border: 0; background: #202020; }}
    .save {{ margin-top: 12px; padding: 11px; border: 0; border-radius: 6px; background: #176b32; color: #fff; font-weight: 800; cursor: pointer; }}
    .delete {{ margin-top: 10px; padding: 11px; border: 0; border-radius: 6px; background: #b42318; color: #fff; font-weight: 800; cursor: pointer; }}
    .finish {{ margin-top: 10px; padding: 11px; border: 0; border-radius: 6px; background: #444; color: #fff; font-weight: 800; cursor: pointer; }}
    .apply {{ display: flex; align-items: center; gap: 8px; font-weight: 400; }}
    .apply input {{ width: auto; }}
    @media (max-width: 900px) {{ .layout {{ grid-template-columns: 1fr; grid-template-rows: auto 65vh; }} select.target {{ height: 24vh; }} }}
  </style>
</head>
<body>
  <div class="layout">
    <aside>
      <h1>Objaverse Candidate Curation</h1>
      <div class="status">{escape(status)}</div>
      {error_block}
      <form method="get" action="/">
        <label for="target">Semantic furniture targets</label>
        <select class="target" id="target" name="key" onchange="location.href='/?key=' + encodeURIComponent(this.value);">
          {target_options}
        </select>
        <label for="rank">Current/new candidate</label>
        <select id="rank" name="rank" onchange="{escape(candidate_select_js)}">
          {candidate_options}
        </select>
        <noscript><button type="submit">Open selected</button></noscript>
      </form>

      <label>Pick which model axis is the real front</label>
      <div class="offsets">{offset_links}</div>

      <label>Stand the object upright if needed</label>
      <div class="muted">Current tilt: X={tilt_x}deg, Z={tilt_z}deg</div>
      <div class="tilts">{tilt_x_links}</div>
      <div class="tilts">{tilt_z_links}</div>
      <div class="tilts single">{tilt_reset_link}</div>

      <form method="post" action="/save-candidate">
        <input type="hidden" name="key" value="{escape(requested_key)}">
        <input type="hidden" name="rank" value="{rank}">
        <input type="hidden" name="offset" value="{offset}">
        <input type="hidden" name="tilt_x" value="{tilt_x}">
        <input type="hidden" name="tilt_z" value="{tilt_z}">
        {destination_control}
        <label class="apply"><input type="checkbox" name="apply_all"> Apply to all 3D contexts</label>
        <label for="notes">Notes</label>
        <textarea id="notes" name="notes"></textarea>
        <button class="save" type="submit">{escape(save_button_text)}</button>
      </form>

      <form method="post" action="/delete-candidate" onsubmit="return confirm('{escape(delete_candidate_text)}')">
        <input type="hidden" name="key" value="{escape(requested_key)}">
        <input type="hidden" name="rank" value="{rank}">
        <input type="hidden" id="delete_destination_asset_id" name="destination_asset_id" value="{escape(delete_destination_asset_id)}">
        <button class="delete" type="submit">Delete this candidate</button>
      </form>

      <form method="post" action="/finish" onsubmit="return confirm('Finish this target and delete all remaining staged new candidates?')">
        <input type="hidden" name="key" value="{escape(requested_key)}">
        <button class="finish" type="submit">Finish target and clean staged new files</button>
      </form>

      <p class="muted">Save works on the selected candidate only. Current candidates save orientation in place. New candidates are copied into the selected inventory item. Delete removes only the selected candidate; current files are unlinked only when no DB row still references them.</p>
    </aside>
    <main>
      <div class="topbar">
        <div class="name">{escape(target_name)}</div>
        <div class="muted">rank={rank} | offset={offset}deg | {escape(OFFSET_LABELS[offset])} | tiltX={tilt_x}deg | tiltZ={tilt_z}deg</div>
        <div class="muted">{escape(member_text)}</div>
        <div class="muted">{escape(query_text)}</div>
      </div>
      <iframe src="{escape(viewer_src)}" title="Candidate GLB viewer"></iframe>
    </main>
  </div>
</body>
</html>"""
        return document.encode("utf-8")

    def _build_curation_viewer(self, query: str) -> bytes:
        params = parse_qs(query)
        key = params.get("key", [""])[0]
        rank = int(params.get("rank", ["1"])[0])
        offset = selected_offset(None, params.get("offset", ["0"])[0])
        tilt_x = selected_tilt(None, params.get("tilt_x", ["0"])[0], axis="x")
        tilt_z = selected_tilt(None, params.get("tilt_z", ["0"])[0], axis="z")
        target = self.curator.target_for_key(key)
        candidate = (
            self.curator.candidate_by_rank(target=target, rank=rank) if target else None
        )
        model_url = "/candidate-model?" + urlencode({"key": key, "rank": str(rank)})
        model_url_json = json.dumps(model_url)
        status_text = (
            f"Loaded {target.asset_id if target else ''} "
            f"[{target.style_key if target else ''}] rank={rank} "
            f"uid={candidate.uid if candidate else ''} sim={candidate.similarity if candidate else 0:.4f}"
        )
        status_text_json = json.dumps(status_text)
        document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {{ margin: 0; overflow: hidden; background: #202020; color: white; font: 13px system-ui, sans-serif; }}
    #status {{ position: fixed; left: 12px; top: 12px; padding: 8px 10px; background: rgba(0,0,0,.7); border-radius: 6px; z-index: 5; }}
    #hint {{ position: fixed; left: 12px; bottom: 12px; padding: 8px 10px; background: rgba(0,0,0,.7); border-radius: 6px; z-index: 5; }}
  </style>
</head>
<body>
  <div id="status">Loading candidate...</div>
  <div id="hint">White arrow = 2D plan front. Drag to orbit, scroll to zoom.</div>
  <script>
    window.addEventListener('error', (event) => {{
      const status = document.getElementById('status');
      if (status) status.textContent = 'Viewer error: ' + event.message;
    }});
    window.addEventListener('unhandledrejection', (event) => {{
      const status = document.getElementById('status');
      const reason = event.reason && event.reason.message ? event.reason.message : String(event.reason);
      if (status) status.textContent = 'Viewer error: ' + reason;
    }});
  </script>
  <script type="module">
    import * as THREE from '/vendor/three.module.js';
    import {{ OrbitControls }} from '/vendor/OrbitControls.js';
    import {{ GLTFLoader }} from '/vendor/loaders/GLTFLoader.js';

    const status = document.getElementById('status');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x202020);
    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.shadowMap.enabled = true;
    document.body.appendChild(renderer.domElement);

    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 100);
    camera.position.set(2.6, 2.0, 3.0);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 0.55, 0);

    const world = new THREE.Group();
    scene.add(world);
    scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.7));
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.2);
    keyLight.position.set(4, 6, 5);
    keyLight.castShadow = true;
    scene.add(keyLight);

    const grid = new THREE.GridHelper(4, 8, 0x999999, 0x444444);
    world.add(grid);
    const arrow = new THREE.ArrowHelper(
      new THREE.Vector3(0, 0, -1),
      new THREE.Vector3(0, 0.08, 1.0),
      1.5,
      0xffffff,
      0.25,
      0.12
    );
    world.add(arrow);

    function makeLabel(text, x, y, z, color = '#ffffff') {{
      const c = document.createElement('canvas');
      c.width = 256;
      c.height = 64;
      const ctx = c.getContext('2d');
      ctx.font = '700 28px system-ui';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = color;
      ctx.fillText(text, 128, 32);
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({{ map: new THREE.CanvasTexture(c), transparent: true }}));
      sprite.position.set(x, y, z);
      sprite.scale.set(0.75, 0.18, 1);
      world.add(sprite);
    }}
    makeLabel('PLAN FRONT', 0, 0.1, -0.85);
    makeLabel('BACK', 0, 0.1, 1.0, '#bdbdbd');
    makeLabel('LEFT', -1.25, 0.1, 0, '#bdbdbd');
    makeLabel('RIGHT', 1.25, 0.1, 0, '#bdbdbd');

    window.addEventListener('resize', resize);
    resize();

    function tick() {{
      requestAnimationFrame(tick);
      controls.update();
      renderer.render(scene, camera);
    }}
    tick();

    main().catch((error) => {{
      status.textContent = 'Viewer error: ' + (error instanceof Error ? error.message : String(error));
      console.error(error);
    }});

    async function main() {{
      const modelUrl = {model_url_json};
      status.textContent = 'Fetching candidate GLB...';
      const response = await fetch(modelUrl, {{ cache: 'no-store' }});
      if (!response.ok) {{
        throw new Error(`GLB request failed: ${{response.status}} ${{response.statusText}}`);
      }}
      const bytes = await response.arrayBuffer();
      status.textContent = `Parsing GLB (${{formatBytes(bytes.byteLength)}})...`;
      const gltf = await new Promise((resolve, reject) => {{
        new GLTFLoader().parse(bytes, '', resolve, reject);
      }});
      const sceneSource = gltf.scene || (Array.isArray(gltf.scenes) ? gltf.scenes[0] : null);
      if (!sceneSource || typeof sceneSource.clone !== 'function') {{
        throw new Error('GLB loaded, but it does not contain a usable scene.');
      }}

      const root = new THREE.Group();
      const yawRoot = new THREE.Group();
      const tiltRoot = new THREE.Group();
      const model = sceneSource.clone(true);
      yawRoot.rotation.y = THREE.MathUtils.degToRad({offset});
      tiltRoot.rotation.x = THREE.MathUtils.degToRad({tilt_x});
      tiltRoot.rotation.z = THREE.MathUtils.degToRad({tilt_z});
      tiltRoot.add(model);
      yawRoot.add(tiltRoot);
      root.add(yawRoot);
      world.add(root);
      root.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(root);
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      if (!Number.isFinite(maxDim) || maxDim <= 0) {{
        throw new Error('GLB parsed, but its bounds are empty.');
      }}
      const scale = Math.min(
        1.8 / Math.max(size.x, size.z, 0.001),
        1.6 / Math.max(size.x, size.y, size.z, 0.001)
      );
      root.scale.setScalar(scale);
      root.updateMatrixWorld(true);
      const fitted = new THREE.Box3().setFromObject(root);
      const center = fitted.getCenter(new THREE.Vector3());
      root.position.x -= center.x;
      root.position.z -= center.z;
      root.position.y -= fitted.min.y;
      root.traverse((node) => {{
        if (node.isMesh) {{
          node.castShadow = true;
          node.receiveShadow = true;
          if (node.material) {{
            const materials = Array.isArray(node.material) ? node.material : [node.material];
            for (const material of materials) {{
              material.side = THREE.DoubleSide;
              material.needsUpdate = true;
            }}
          }}
        }}
      }});
      status.textContent = {status_text_json};
    }}

    function resize() {{
      renderer.setSize(window.innerWidth, window.innerHeight);
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
    }}

    function formatBytes(size) {{
      if (!Number.isFinite(size)) return 'unknown size';
      if (size < 1024 * 1024) return `${{Math.round(size / 1024)}} KB`;
      return `${{(size / (1024 * 1024)).toFixed(1)}} MB`;
    }}
  </script>
</body>
</html>"""
        return document.encode("utf-8")

    def _curation_target_option(self, *, target: JsonObject, selected_key: str) -> str:
        key = str(target.get("key") or "")
        label = (
            target.get("display_label")
            or f"{target.get('name')} - {target.get('style_key')}"
        )
        selected = " selected" if key == selected_key else ""
        return f'<option value="{escape(key)}"{selected}>{escape(label)}</option>'

    def _candidate_options(
        self,
        *,
        target: ObjaverseCurationTarget | None,
        selected_rank: int,
    ) -> str:
        if target is None:
            return ""
        rows: list[str] = []
        for candidate in target.candidates:
            selected = " selected" if candidate.rank == selected_rank else ""
            status = "ready" if candidate.path.exists() else "missing"
            source_label = (
                f"current {candidate.source_asset_id}"
                if candidate.source == "existing"
                else "new objaverse"
            )
            label = (
                f"#{candidate.rank:02d} {source_label} "
                f"sim={candidate.similarity:.4f} {candidate.uid} ({status})"
            )
            rows.append(
                f'<option value="{candidate.rank}"{selected}>{escape(label)}</option>'
            )
        return "\n".join(rows)

    def _destination_asset_control(
        self,
        *,
        target: ObjaverseCurationTarget | None,
        candidate: ObjaverseCandidate | None,
        selected_asset_id: str,
    ) -> str:
        if target is None or candidate is None:
            return ""
        if candidate.source == "existing":
            asset_id = candidate.source_asset_id or selected_asset_id
            return (
                f'<input type="hidden" name="destination_asset_id" '
                f'value="{escape(asset_id)}">'
                f'<p class="muted">Current candidate belongs to {escape(asset_id)}. '
                "Save keeps it there, removes it from this list, then moves on.</p>"
            )

        rows: list[str] = []
        for asset in target.assets:
            asset_id = str(asset.id)
            selected = " selected" if asset_id == selected_asset_id else ""
            rows.append(
                f'<option value="{escape(asset_id)}"{selected}>{escape(asset_id)}</option>'
            )
        return (
            '<label for="destination_asset_id">Save new candidate into</label>'
            '<select id="destination_asset_id" name="destination_asset_id" '
            "onchange=\"const deleteDest = document.getElementById('delete_destination_asset_id'); "
            'if (deleteDest) deleteDest.value = this.value;">'
            f"{''.join(rows)}"
            "</select>"
        )

    def _curation_offset_link(
        self,
        *,
        key: str,
        rank: int,
        offset: int,
        tilt_x: int,
        tilt_z: int,
        selected: bool,
    ) -> str:
        query = urlencode(
            {
                "key": key,
                "rank": str(rank),
                "offset": str(offset),
                "tilt_x": str(tilt_x),
                "tilt_z": str(tilt_z),
            }
        )
        class_name = " selected" if selected else ""
        return f'<a class="{class_name}" href="/?{query}">{escape(OFFSET_LABELS[offset])}<br>{offset}deg</a>'

    def _curation_tilt_link(
        self,
        *,
        key: str,
        rank: int,
        offset: int,
        tilt_x: int,
        tilt_z: int,
        label: str,
    ) -> str:
        query = urlencode(
            {
                "key": key,
                "rank": str(rank),
                "offset": str(offset),
                "tilt_x": str(tilt_x),
                "tilt_z": str(tilt_z),
            }
        )
        return f'<a href="/?{query}">{escape(label)}<br>X={tilt_x} Z={tilt_z}</a>'


def selected_candidate_rank(
    target: ObjaverseCurationTarget | None,
    raw_rank: str | None,
    *,
    default_rank: int | None = None,
) -> int:
    if target is None:
        return 1
    if raw_rank:
        try:
            rank = int(raw_rank)
        except ValueError:
            rank = 1
        if any(candidate.rank == rank for candidate in target.candidates):
            return rank
    if default_rank is not None and any(
        candidate.rank == default_rank for candidate in target.candidates
    ):
        return default_rank
    return target.candidates[0].rank if target.candidates else 1


def build_handler(
    *,
    inspector: InventoryGlbOrientationInspector,
    vendor_dir: Path,
) -> type[ManualInventoryGlbReviewHandler]:
    class BoundManualInventoryGlbReviewHandler(ManualInventoryGlbReviewHandler):
        pass

    BoundManualInventoryGlbReviewHandler.inspector = inspector
    BoundManualInventoryGlbReviewHandler.vendor_dir = vendor_dir
    return BoundManualInventoryGlbReviewHandler


def build_curation_handler(
    *,
    curator: ObjaverseCandidateCurator,
    vendor_dir: Path,
) -> type[ObjaverseCandidateCurationHandler]:
    class BoundObjaverseCandidateCurationHandler(ObjaverseCandidateCurationHandler):
        pass

    BoundObjaverseCandidateCurationHandler.curator = curator
    BoundObjaverseCandidateCurationHandler.vendor_dir = vendor_dir
    return BoundObjaverseCandidateCurationHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-retrieve grouped Objaverse candidates and save the selected "
            "GLB plus manual orientation metadata to the inventory DB."
        )
    )
    parser.add_argument("--tenant-id", default="demo_tenant")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--asset-ids", nargs="*", default=None)
    parser.add_argument("--style-keys", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--models-public-dir", type=Path, default=DEFAULT_MODELS_PUBLIC_DIR
    )
    parser.add_argument("--models-url-prefix", default=DEFAULT_MODELS_URL_PREFIX)
    parser.add_argument(
        "--review-existing",
        action="store_true",
        help="Use the older mode that only reviews orientation for already stored GLBs.",
    )
    parser.add_argument(
        "--semantic-program-path",
        type=Path,
        default=DEFAULT_SEMANTIC_PROGRAM_PATH,
        help="Compiled semantic program used to choose furniture object ids.",
    )
    parser.add_argument(
        "--room-types",
        nargs="*",
        default=list(DEFAULT_CURATION_ROOM_TYPES),
        help="Room types to read from the semantic program.",
    )
    parser.add_argument(
        "--styles",
        nargs="*",
        default=list(DEFAULT_CURATION_STYLES),
        help="Style prompts to curate for each grouped semantic furniture family.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of Objaverse candidates downloaded for each grouped family/style.",
    )
    parser.add_argument(
        "--sim-threshold",
        type=float,
        default=0.1,
        help="Minimum OpenShape similarity score for candidate retrieval.",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path("OpenShape-Embeddings"),
        help="Directory used to cache Objaverse embeddings.",
    )
    parser.add_argument(
        "--candidates-public-dir",
        type=Path,
        default=DEFAULT_CANDIDATES_PUBLIC_DIR,
        help="Temporary staging directory for downloaded candidate GLBs.",
    )
    parser.add_argument(
        "--candidates-url-prefix",
        default=DEFAULT_CANDIDATES_URL_PREFIX,
        help="Public URL prefix for staged candidate GLBs.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Device used for the text encoder.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Embedding chunk size used during similarity search.",
    )
    parser.add_argument(
        "--download-processes",
        type=int,
        default=4,
        help="Worker processes used by Objaverse when downloading staged candidates.",
    )
    parser.add_argument(
        "--keep-candidate-staging",
        action="store_true",
        help="Keep temporary staged candidate files when the server stops.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    curator: ObjaverseCandidateCurator | None = None
    if args.review_existing:
        inspector = InventoryGlbOrientationInspector(
            tenant_id=str(args.tenant_id),
            models_public_dir=(ROOT_DIR / args.models_public_dir).resolve(),
            models_url_prefix=str(args.models_url_prefix),
            asset_ids=args.asset_ids,
            style_keys=args.style_keys,
            limit=args.limit,
        )
        handler = build_handler(
            inspector=inspector,
            vendor_dir=(ROOT_DIR / "frontend/src/vendor/three").resolve(),
        )
        mode_label = "Manual inventory GLB orientation review"
    else:
        curator = ObjaverseCandidateCurator(
            tenant_id=str(args.tenant_id),
            semantic_program_path=(ROOT_DIR / args.semantic_program_path).resolve(),
            room_types=args.room_types,
            styles=args.styles,
            top_k=int(args.top_k),
            sim_threshold=float(args.sim_threshold),
            embeddings_dir=(ROOT_DIR / args.embeddings_dir).resolve(),
            models_public_dir=(ROOT_DIR / args.models_public_dir).resolve(),
            models_url_prefix=str(args.models_url_prefix),
            candidates_public_dir=(ROOT_DIR / args.candidates_public_dir).resolve(),
            candidates_url_prefix=str(args.candidates_url_prefix),
            asset_ids=args.asset_ids,
            limit=args.limit,
            device_name=str(args.device),
            chunk_size=int(args.chunk_size),
            download_processes=int(args.download_processes),
        )
        handler = build_curation_handler(
            curator=curator,
            vendor_dir=(ROOT_DIR / "frontend/src/vendor/three").resolve(),
        )
        mode_label = "Grouped Objaverse candidate curation"

    server = ThreadingHTTPServer((str(args.host), int(args.port)), handler)
    print(f"{mode_label}: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping review server.")
    finally:
        server.server_close()
        if curator is not None and not bool(args.keep_candidate_staging):
            curator.cleanup_all_staging()


if __name__ == "__main__":
    main()
