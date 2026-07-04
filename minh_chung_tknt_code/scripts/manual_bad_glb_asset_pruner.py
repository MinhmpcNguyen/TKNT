from __future__ import annotations

import argparse
import html
import json
import mimetypes
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
sys.path.append(str(ROOT_DIR))

from db import PostgresAssetRepository  # noqa: E402
from db.models import (  # noqa: E402
    Asset,
    AssetFile,
    AssetFilter,
    AssetId,
    AssetType,
    TenantId,
)
from db.postgres import create_connection  # noqa: E402

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8794
DEFAULT_TENANT_ID = "demo_tenant"
DEFAULT_ASSET_TYPE: AssetType = "FURNITURE"
SUPPORTED_ASSET_TYPES: tuple[AssetType, ...] = ("FURNITURE", "TEMPLATE")
DEFAULT_MODELS_PUBLIC_DIR = Path("frontend/public/assets/inventory_models")
DEFAULT_MODELS_URL_PREFIX = "/assets/inventory_models"
VENDOR_DIR = ROOT_DIR / "frontend" / "src" / "vendor" / "three"
BASIS_DIR = ROOT_DIR / "frontend" / "public" / "assets" / "three" / "basis"
DIRECT_STYLE_KEY = "__direct__"

JsonObject = dict[str, object]

ASSET_COUNTS_SQL = """
SELECT
    (SELECT COUNT(*) FROM asset_files WHERE asset_id = %(asset_id)s) AS asset_files,
    (SELECT COUNT(*) FROM asset_embeddings WHERE asset_id = %(asset_id)s) AS embeddings
"""

DELETE_ASSET_SQL = """
DELETE FROM assets
WHERE id = %(asset_id)s
  AND tenant_id = %(tenant_id)s
  AND type = %(asset_type)s
RETURNING id
"""


@dataclass(frozen=True)
class GlbReference:
    asset_id: str
    asset_name: str
    source: str
    source_key: str
    display_style: str
    path: Path
    public_url: str | None
    category: str | None

    @property
    def key(self) -> str:
        return "::".join((self.asset_id, self.source, self.source_key))


@dataclass(frozen=True)
class AssetDeletionResult:
    asset_id: str
    asset_deleted: bool
    cascaded_asset_files: int
    cascaded_embeddings: int
    deleted_paths: tuple[str, ...]
    skipped_paths: tuple[str, ...]
    deleted_at_utc: str


def escape(value: object) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def read_json_object(value: object) -> JsonObject | None:
    if not isinstance(value, dict):
        return None
    return {str(key): item for key, item in value.items()}


def read_string(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def read_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def canonicalize_style_key(style: str) -> str:
    return " ".join(style.strip().lower().split())


def parse_asset_type(value: str) -> AssetType:
    normalized = value.strip().upper()
    if normalized in SUPPORTED_ASSET_TYPES:
        return normalized
    raise ValueError(
        f"Unsupported asset type {value!r}. Choose one of: "
        f"{', '.join(SUPPORTED_ASSET_TYPES)}."
    )


def resolve_cli_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (ROOT_DIR / expanded).resolve()


def path_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def model_url_to_path(
    *,
    url: str,
    models_public_dir: Path,
    models_url_prefix: str,
) -> Path | None:
    normalized_prefix = "/" + models_url_prefix.strip("/").replace("\\", "/")
    normalized_url = url.strip()
    if not normalized_url.startswith(normalized_prefix):
        return None

    relative = normalized_url.removeprefix(normalized_prefix).lstrip("/")
    if not relative:
        return None

    candidate = (models_public_dir / Path(relative)).resolve()
    if not path_within(candidate, models_public_dir):
        return None
    return candidate


def storage_key_to_path(storage_key: str) -> Path:
    candidate = Path(storage_key).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (ROOT_DIR / candidate).resolve()


def metadata_model_paths(
    *,
    metadata: JsonObject,
    models_public_dir: Path,
    models_url_prefix: str,
) -> set[Path]:
    paths: set[Path] = set()

    local_path = read_string(metadata.get("local_path"))
    if local_path is not None:
        paths.add(storage_key_to_path(local_path))

    storage_key = read_string(metadata.get("storage_key"))
    if storage_key is not None:
        paths.add(storage_key_to_path(storage_key))

    public_url = read_string(metadata.get("url"))
    if public_url is not None:
        resolved = model_url_to_path(
            url=public_url,
            models_public_dir=models_public_dir,
            models_url_prefix=models_url_prefix,
        )
        if resolved is not None:
            paths.add(resolved)

    return {path for path in paths if path.suffix.lower() == ".glb"}


def asset_file_model_path(asset_file: AssetFile) -> Path | None:
    storage_key = asset_file.storage_key.strip()
    if not storage_key:
        return None

    is_glb = (
        storage_key.lower().endswith(".glb")
        or (asset_file.mime or "").strip().lower() == "model/gltf-binary"
        or asset_file.file_kind == "MODEL"
    )
    if not is_glb:
        return None

    path = storage_key_to_path(storage_key)
    if path.suffix.lower() != ".glb":
        return None
    return path


def asset_category(asset: Asset) -> str | None:
    category = read_string(dict(asset.attributes or {}).get("category"))
    if category is not None:
        return category
    return read_string(dict(asset.attributes or {}).get("source_category"))


def format_file_size(path: Path) -> str:
    if not path.exists():
        return "missing"
    size = path.stat().st_size
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


class BadGlbAssetPruner:
    def __init__(
        self,
        *,
        tenant_id: str,
        asset_type: AssetType,
        models_public_dir: Path,
        models_url_prefix: str,
        asset_ids: Sequence[str] | None,
        style_keys: Sequence[str] | None,
        limit: int | None,
        delete_roots: Sequence[Path],
    ) -> None:
        self._tenant_id = tenant_id
        self._asset_type = asset_type
        self._models_public_dir = models_public_dir
        self._models_url_prefix = models_url_prefix
        self._asset_ids = {
            asset_id.strip() for asset_id in asset_ids or [] if asset_id.strip()
        }
        self._style_keys = (
            {canonicalize_style_key(style) for style in style_keys if style.strip()}
            if style_keys
            else None
        )
        self._limit = limit
        self._delete_roots = tuple(root.resolve() for root in delete_roots)
        self._repo = PostgresAssetRepository()
        self._last_result: AssetDeletionResult | None = None

    def list_targets(self) -> list[GlbReference]:
        references: list[GlbReference] = []
        for asset in self._load_assets():
            references.extend(self._iter_glb_references(asset))
        references.sort(
            key=lambda target: (
                target.asset_name.lower(),
                target.asset_id,
                target.display_style,
                str(target.path),
            )
        )
        return references

    def find_target(self, key: str) -> GlbReference | None:
        asset_id, source, source_key = split_target_key(key)
        if not asset_id:
            return None
        asset = self._repo.get_asset(AssetId(asset_id))
        if (
            asset is None
            or str(asset.tenant_id) != self._tenant_id
            or asset.type != self._asset_type
        ):
            return None
        for target in self._iter_glb_references(asset):
            if target.source == source and target.source_key == source_key:
                return target
        return None

    def first_target(self) -> GlbReference | None:
        targets = self.list_targets()
        return targets[0] if targets else None

    def last_result(self) -> AssetDeletionResult | None:
        return self._last_result

    def delete_roots(self) -> tuple[Path, ...]:
        return self._delete_roots

    def delete_target(
        self,
        *,
        target_key: str,
        delete_files: bool,
        confirmed: bool,
    ) -> AssetDeletionResult:
        if not confirmed:
            raise ValueError("Deletion requires the confirmation checkbox.")

        target = self.find_target(target_key)
        if target is None:
            raise ValueError(f"GLB target no longer exists: {target_key}")

        asset = self._repo.get_asset(AssetId(target.asset_id))
        if asset is None:
            raise ValueError(f"Asset no longer exists: {target.asset_id}")

        glb_paths = self._collect_asset_glb_paths(asset)
        counts = self._asset_child_counts(asset_id=target.asset_id)
        deleted_asset = self._delete_asset_row(asset_id=target.asset_id)
        deleted_paths: list[str] = []
        skipped_paths: list[str] = []

        if delete_files and deleted_asset:
            deleted_paths, skipped_paths = self._delete_unreferenced_files(
                paths=glb_paths,
                deleted_asset_id=target.asset_id,
            )

        result = AssetDeletionResult(
            asset_id=target.asset_id,
            asset_deleted=deleted_asset,
            cascaded_asset_files=counts[0],
            cascaded_embeddings=counts[1],
            deleted_paths=tuple(deleted_paths),
            skipped_paths=tuple(skipped_paths),
            deleted_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        self._last_result = result
        return result

    def _load_assets(self) -> list[Asset]:
        assets = list(
            self._repo.list_assets(
                AssetFilter(
                    tenant_id=TenantId(self._tenant_id),
                    type=self._asset_type,
                )
            )
        )
        if self._asset_ids:
            assets = [asset for asset in assets if str(asset.id) in self._asset_ids]
        assets.sort(key=lambda asset: str(asset.id))
        if self._limit is not None:
            assets = assets[: max(0, self._limit)]
        return assets

    def _iter_glb_references(self, asset: Asset) -> list[GlbReference]:
        references: list[GlbReference] = []
        seen_paths: set[Path] = set()
        attributes = dict(asset.attributes or {})
        category = asset_category(asset)

        variants_raw = read_json_object(attributes.get("model_variants")) or {}
        for metadata_key, raw_metadata in variants_raw.items():
            style_key = canonicalize_style_key(metadata_key)
            if self._style_keys and style_key not in self._style_keys:
                continue
            metadata = read_json_object(raw_metadata)
            if metadata is None:
                continue
            for path in sorted(
                metadata_model_paths(
                    metadata=metadata,
                    models_public_dir=self._models_public_dir,
                    models_url_prefix=self._models_url_prefix,
                ),
                key=str,
            ):
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                references.append(
                    GlbReference(
                        asset_id=str(asset.id),
                        asset_name=asset.name,
                        source="variant",
                        source_key=metadata_key,
                        display_style=style_key,
                        path=path,
                        public_url=read_string(metadata.get("url")),
                        category=category,
                    )
                )

        model_metadata = read_json_object(attributes.get("model_3d"))
        if model_metadata is not None:
            for path in sorted(
                metadata_model_paths(
                    metadata=model_metadata,
                    models_public_dir=self._models_public_dir,
                    models_url_prefix=self._models_url_prefix,
                ),
                key=str,
            ):
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                references.append(
                    GlbReference(
                        asset_id=str(asset.id),
                        asset_name=asset.name,
                        source="top_level",
                        source_key=DIRECT_STYLE_KEY,
                        display_style=DIRECT_STYLE_KEY,
                        path=path,
                        public_url=read_string(model_metadata.get("url"))
                        or read_string(attributes.get("model_url")),
                        category=category,
                    )
                )

        for asset_file in self._repo.list_asset_files(asset.id):
            path = asset_file_model_path(asset_file)
            if path is None or path in seen_paths:
                continue
            seen_paths.add(path)
            references.append(
                GlbReference(
                    asset_id=str(asset.id),
                    asset_name=asset.name,
                    source="asset_file",
                    source_key=str(asset_file.id),
                    display_style=asset_file.role or "asset_file",
                    path=path,
                    public_url=None,
                    category=category,
                )
            )

        return references

    def _collect_asset_glb_paths(self, asset: Asset) -> set[Path]:
        return {target.path for target in self._iter_glb_references(asset)}

    def _asset_child_counts(self, *, asset_id: str) -> tuple[int, int]:
        with create_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(ASSET_COUNTS_SQL, {"asset_id": asset_id})
                row = cur.fetchone()
        if row is None:
            return 0, 0
        return int(row["asset_files"] or 0), int(row["embeddings"] or 0)

    def _delete_asset_row(self, *, asset_id: str) -> bool:
        with create_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    DELETE_ASSET_SQL,
                    {
                        "asset_id": asset_id,
                        "tenant_id": self._tenant_id,
                        "asset_type": self._asset_type,
                    },
                )
                row = cur.fetchone()
        return row is not None

    def _delete_unreferenced_files(
        self,
        *,
        paths: set[Path],
        deleted_asset_id: str,
    ) -> tuple[list[str], list[str]]:
        deleted_paths: list[str] = []
        skipped_paths: list[str] = []

        for path in sorted(paths, key=str):
            resolved = path.expanduser().resolve()
            if not resolved.exists():
                skipped_paths.append(f"{resolved} (missing)")
                continue
            if resolved.suffix.lower() != ".glb":
                skipped_paths.append(f"{resolved} (not a .glb)")
                continue
            if not any(path_within(resolved, root) for root in self._delete_roots):
                skipped_paths.append(f"{resolved} (outside allowed delete roots)")
                continue
            if self._path_is_referenced_by_other_asset(
                path=resolved,
                deleted_asset_id=deleted_asset_id,
            ):
                skipped_paths.append(f"{resolved} (still referenced by another asset)")
                continue
            try:
                resolved.unlink()
            except OSError as exc:
                skipped_paths.append(f"{resolved} ({exc})")
                continue
            deleted_paths.append(str(resolved))

        return deleted_paths, skipped_paths

    def _path_is_referenced_by_other_asset(
        self,
        *,
        path: Path,
        deleted_asset_id: str,
    ) -> bool:
        for asset in self._load_assets():
            if str(asset.id) == deleted_asset_id:
                continue
            if path in {
                candidate.resolve()
                for candidate in self._collect_asset_glb_paths(asset)
            }:
                return True
        return False


def split_target_key(key: str) -> tuple[str, str, str]:
    parts = key.split("::", 2)
    if len(parts) != 3:
        return "", "", ""
    return parts[0], parts[1], parts[2]


class BadGlbPruneHandler(BaseHTTPRequestHandler):
    pruner: BadGlbAssetPruner
    vendor_dir: Path
    basis_dir: Path

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
                target = self._selected_target(params)
                if target is None:
                    raise FileNotFoundError("No GLB target is available.")
                self._send_file(target.path, default_content_type="model/gltf-binary")
                return
            if parsed.path.startswith("/vendor/"):
                self._send_vendor_file(parsed.path.removeprefix("/vendor/"))
                return
            if parsed.path.startswith("/basis/"):
                self._send_basis_file(parsed.path.removeprefix("/basis/"))
                return
            self._send_plain("Not found.", status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_plain(str(exc), status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path != "/delete-asset":
                self._send_plain("Not found.", status=HTTPStatus.NOT_FOUND)
                return

            length_raw = self.headers.get("Content-Length")
            length = int(length_raw) if length_raw and length_raw.isdigit() else 0
            body = self.rfile.read(length).decode("utf-8")
            params = parse_qs(body)
            target_key = params.get("target", [""])[0]
            self.pruner.delete_target(
                target_key=target_key,
                delete_files=read_bool(params.get("delete_files", [""])[0]),
                confirmed=read_bool(params.get("confirm_delete", [""])[0]),
            )
            self.send_response(int(HTTPStatus.SEE_OTHER))
            self.send_header("Location", "/?deleted=1")
            self.end_headers()
        except Exception as exc:
            self._send_plain(str(exc), status=HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _build_index(self, query: str) -> bytes:
        params = parse_qs(query)
        targets = self.pruner.list_targets()
        target = self._selected_target(params)
        if target is None and targets:
            target = targets[0]

        deleted_now = params.get("deleted", [""])[0] == "1"
        status = self._status_text(deleted_now=deleted_now)
        options = "\n".join(
            self._target_option(target=candidate, selected=target == candidate)
            for candidate in targets
        )
        file_links = "\n".join(
            self._target_link(target=candidate, selected=target == candidate)
            for candidate in targets
        )
        details = self._target_details(target=target)
        viewer_src = (
            "/viewer?" + urlencode({"target": target.key}) if target is not None else ""
        )
        iframe = (
            f'<iframe src="{escape(viewer_src)}" title="GLB preview"></iframe>'
            if viewer_src
            else '<div class="empty">No GLB targets found.</div>'
        )

        document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Bad GLB Asset Pruner</title>
  <style>
    body {{ margin: 0; font: 14px system-ui, sans-serif; color: #111; background: #f6f6f4; }}
    .layout {{ display: grid; grid-template-columns: 420px 1fr; height: 100vh; }}
    aside {{ padding: 14px; border-right: 1px solid #ccc; overflow: auto; background: #fff; }}
    main {{ display: grid; grid-template-rows: auto 1fr; min-width: 0; }}
    h1 {{ font-size: 18px; margin: 0 0 8px; }}
    h2 {{ font-size: 14px; margin: 16px 0 8px; }}
    label {{ display: block; margin-top: 8px; font-weight: 700; }}
    select {{ width: 100%; min-height: 40px; padding: 8px; }}
    button {{ width: 100%; margin-top: 10px; padding: 11px; border: 0; border-radius: 6px; background: #b42318; color: #fff; font-weight: 800; cursor: pointer; }}
    button:disabled {{ background: #999; cursor: not-allowed; }}
    .muted {{ color: #666; font-size: 12px; }}
    .status {{ padding: 8px 10px; border: 1px solid #ddd; border-radius: 6px; background: #fafafa; }}
    .topbar {{ padding: 10px 14px; border-bottom: 1px solid #ccc; background: #fff; }}
    .name {{ font-weight: 800; }}
    .files {{ display: grid; gap: 6px; margin: 8px 0 12px; }}
    .files a {{ display: block; padding: 8px 10px; border: 1px solid #ddd; border-radius: 6px; color: #111; text-decoration: none; background: #fafafa; }}
    .files a.selected {{ border-color: #b42318; background: #fee4e2; font-weight: 800; }}
    .danger {{ border: 1px solid #fecdca; border-radius: 6px; background: #fff1f0; padding: 10px; }}
    .check {{ display: flex; gap: 8px; align-items: center; font-weight: 500; }}
    .check input {{ width: auto; }}
    .kv {{ display: grid; grid-template-columns: 110px minmax(0, 1fr); gap: 6px 10px; margin-top: 8px; }}
    code {{ overflow-wrap: anywhere; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    pre {{ padding: 10px; border-radius: 6px; background: #111; color: #d8f3dc; overflow: auto; font-size: 12px; }}
    iframe {{ width: 100%; height: 100%; border: 0; background: #202020; }}
    .empty {{ padding: 20px; color: #666; }}
    @media (max-width: 980px) {{ .layout {{ grid-template-columns: 1fr; grid-template-rows: auto 60vh; }} }}
  </style>
</head>
<body>
  <div class="layout">
    <aside>
      <h1>Bad GLB Asset Pruner</h1>
      <div class="status">{escape(status)}</div>
      <div class="muted" style="margin-top:6px;">Targets: {len(targets)}</div>

      <h2>1. Pick a DB GLB target</h2>
      <form method="get" action="/">
        <select name="target" onchange="this.form.submit()">
          {options}
        </select>
        <noscript><button type="submit">Open selected</button></noscript>
      </form>
      <div class="files">{file_links}</div>

      <h2>2. Inspect and delete</h2>
      {details}
    </aside>
    <main>
      <div class="topbar">
        <div class="name">{escape(target.asset_name if target else "No target")}</div>
        <div class="muted">{escape(target.asset_id if target else "")}</div>
      </div>
      {iframe}
    </main>
  </div>
</body>
</html>"""
        return document.encode("utf-8")

    def _build_viewer(self, query: str) -> bytes:
        params = parse_qs(query)
        target = self._selected_target(params)
        if target is None:
            return b"<!doctype html><html><body>No GLB target.</body></html>"
        model_url_json = json.dumps(f"/model?{urlencode({'target': target.key})}")
        status_text_json = json.dumps(
            f"Loaded {target.asset_id} [{target.display_style}] "
            f"{target.path.name} ({format_file_size(target.path)})"
        )
        document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {{ margin: 0; overflow: hidden; background: #202020; color: white; font: 13px system-ui, sans-serif; }}
    #status {{ position: fixed; left: 12px; top: 12px; padding: 8px 10px; background: rgba(0,0,0,.72); border-radius: 6px; z-index: 5; }}
    #hint {{ position: fixed; left: 12px; bottom: 12px; padding: 8px 10px; background: rgba(0,0,0,.72); border-radius: 6px; z-index: 5; }}
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
    import {{ KTX2Loader }} from '/vendor/loaders/KTX2Loader.js';

    const status = document.getElementById('status');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x202020);
    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);
    const ktx2Loader = new KTX2Loader()
      .setTranscoderPath('/basis/')
      .setWorkerLimit(2)
      .detectSupport(renderer);

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
    scene.add(key);

    world.add(new THREE.GridHelper(4, 8, 0x999999, 0x444444));
    world.add(new THREE.ArrowHelper(
      new THREE.Vector3(0, 0, -1),
      new THREE.Vector3(0, 0.08, 1.0),
      1.5,
      0xffffff,
      0.25,
      0.12
    ));

    window.addEventListener('resize', resize);
    resize();
    tick();
    main().catch((error) => {{
      status.textContent = 'Viewer error: ' + (error instanceof Error ? error.message : String(error));
      console.error(error);
    }});

    async function main() {{
      status.textContent = 'Fetching GLB...';
      const response = await fetch({model_url_json}, {{ cache: 'no-store' }});
      if (!response.ok) {{
        throw new Error(`GLB request failed: ${{response.status}} ${{response.statusText}}`);
      }}
      const bytes = await response.arrayBuffer();
      status.textContent = `Parsing GLB (${{formatBytes(bytes.byteLength)}})...`;
      const loader = new GLTFLoader();
      if (typeof loader.setKTX2Loader === 'function') {{
        loader.setKTX2Loader(ktx2Loader);
      }}
      const gltf = await new Promise((resolve, reject) => {{
        loader.parse(bytes, '', resolve, reject);
      }});
      const sceneSource = gltf.scene || (Array.isArray(gltf.scenes) ? gltf.scenes[0] : null);
      if (!sceneSource || typeof sceneSource.clone !== 'function') {{
        throw new Error('GLB loaded, but it does not contain a usable scene.');
      }}
      const root = new THREE.Group();
      root.add(sceneSource.clone(true));
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
        if (!node.isMesh || !node.material) return;
        const materials = Array.isArray(node.material) ? node.material : [node.material];
        for (const material of materials) {{
          material.side = THREE.DoubleSide;
          material.needsUpdate = true;
        }}
      }});
      status.textContent = {status_text_json};
    }}

    function tick() {{
      requestAnimationFrame(tick);
      controls.update();
      renderer.render(scene, camera);
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

    def _selected_target(
        self,
        params: dict[str, list[str]],
    ) -> GlbReference | None:
        requested_key = params.get("target", [""])[0]
        if requested_key:
            target = self.pruner.find_target(requested_key)
            if target is not None:
                return target
        return self.pruner.first_target()

    def _target_option(self, *, target: GlbReference, selected: bool) -> str:
        selected_attr = " selected" if selected else ""
        label = f"{target.asset_name} - {target.display_style} ({target.asset_id})"
        return (
            f'<option value="{escape(target.key)}"{selected_attr}>'
            f"{escape(label)}</option>"
        )

    def _target_link(self, *, target: GlbReference, selected: bool) -> str:
        query = urlencode({"target": target.key})
        classes = "selected" if selected else ""
        exists = "exists" if target.path.exists() else "missing"
        return (
            f'<a class="{classes}" href="/?{query}">'
            f"{escape(target.asset_name)}<br>"
            f'<span class="muted">{escape(target.asset_id)} | '
            f"{escape(target.display_style)} | {escape(exists)} | "
            f"{escape(format_file_size(target.path))}</span></a>"
        )

    def _target_details(self, *, target: GlbReference | None) -> str:
        if target is None:
            return '<div class="empty">No DB-backed GLB targets were found.</div>'

        delete_roots = "\n".join(f"  - {root}" for root in self.pruner.delete_roots())
        public_url = target.public_url or "-"
        delete_allowed = any(
            path_within(target.path, root) for root in self.pruner.delete_roots()
        )
        delete_allowed_label = "yes" if delete_allowed else "no"
        warning = (
            "This deletes the whole asset row. asset_files and asset_embeddings "
            "cascade through the database schema."
        )
        return f"""
      <div class="danger">
        <strong>{escape(warning)}</strong>
        <div class="kv">
          <span>asset_id</span><code>{escape(target.asset_id)}</code>
          <span>style/source</span><code>{escape(target.display_style)} / {escape(target.source)}</code>
          <span>category</span><code>{escape(target.category or "-")}</code>
          <span>file</span><code>{escape(target.path)}</code>
          <span>public_url</span><code>{escape(public_url)}</code>
          <span>file_size</span><code>{escape(format_file_size(target.path))}</code>
          <span>unlink_allowed</span><code>{escape(delete_allowed_label)}</code>
        </div>
      </div>
      <form method="post" action="/delete-asset" onsubmit="return confirm('Delete this asset and its selected bad GLB references?')">
        <input type="hidden" name="target" value="{escape(target.key)}">
        <label class="check"><input type="checkbox" name="confirm_delete" required> Delete this entire DB asset</label>
        <label class="check"><input type="checkbox" name="delete_files" checked> Also unlink unreferenced local .glb files under allowed roots</label>
        <button type="submit">Delete Bad GLB Asset</button>
      </form>
      <h2>Allowed file delete roots</h2>
      <pre>{escape(delete_roots or "  - none")}</pre>
"""

    def _status_text(self, *, deleted_now: bool) -> str:
        result = self.pruner.last_result()
        if deleted_now and result is not None:
            return (
                f"Deleted asset={result.asset_deleted} for {result.asset_id}; "
                f"cascaded asset_files={result.cascaded_asset_files}, "
                f"embeddings={result.cascaded_embeddings}; "
                f"deleted_files={len(result.deleted_paths)}, "
                f"skipped_files={len(result.skipped_paths)}."
            )
        return "Inspect DB-backed GLBs and delete the whole asset when the GLB is bad."

    def _send_vendor_file(self, relative_path: str) -> None:
        relative = Path(relative_path)
        candidate = (self.vendor_dir / relative).resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Vendor file not found: {candidate}")
        if not path_within(candidate, self.vendor_dir):
            raise FileNotFoundError(
                f"Vendor file is outside allowed directory: {candidate}"
            )
        self._send_file(candidate)

    def _send_basis_file(self, relative_path: str) -> None:
        relative = Path(relative_path)
        candidate = (self.basis_dir / relative).resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Basis transcoder file not found: {candidate}")
        if not path_within(candidate, self.basis_dir):
            raise FileNotFoundError(
                f"Basis file is outside allowed directory: {candidate}"
            )
        self._send_file(candidate)

    def _send_file(
        self,
        path: Path,
        *,
        default_content_type: str | None = None,
    ) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        content_type = (
            default_content_type
            or mimetypes.guess_type(path.name)[0]
            or "application/octet-stream"
        )
        data = path.read_bytes()
        self.send_response(int(HTTPStatus.OK))
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, document: bytes) -> None:
        self.send_response(int(HTTPStatus.OK))
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(document)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(document)

    def _send_plain(self, message: str, *, status: HTTPStatus) -> None:
        data = message.encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def build_handler(
    *,
    pruner: BadGlbAssetPruner,
    vendor_dir: Path,
    basis_dir: Path,
) -> type[BadGlbPruneHandler]:
    class BoundBadGlbPruneHandler(BadGlbPruneHandler):
        pass

    BoundBadGlbPruneHandler.pruner = pruner
    BoundBadGlbPruneHandler.vendor_dir = vendor_dir
    BoundBadGlbPruneHandler.basis_dir = basis_dir
    return BoundBadGlbPruneHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Manual DB-backed GLB viewer for pruning bad GLB files and deleting "
            "the associated inventory asset row."
        )
    )
    parser.add_argument("--tenant-id", default=DEFAULT_TENANT_ID)
    parser.add_argument(
        "--asset-type",
        choices=SUPPORTED_ASSET_TYPES,
        default=DEFAULT_ASSET_TYPE,
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--asset-ids", nargs="*", default=None)
    parser.add_argument("--style-keys", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--models-public-dir",
        type=Path,
        default=DEFAULT_MODELS_PUBLIC_DIR,
    )
    parser.add_argument("--models-url-prefix", default=DEFAULT_MODELS_URL_PREFIX)
    parser.add_argument(
        "--delete-roots",
        nargs="*",
        type=Path,
        default=None,
        help=(
            "Directories where local .glb files may be unlinked. Defaults to "
            "--models-public-dir only."
        ),
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    models_public_dir = resolve_cli_path(args.models_public_dir)
    delete_roots = (
        [resolve_cli_path(path) for path in args.delete_roots]
        if args.delete_roots
        else [models_public_dir]
    )
    pruner = BadGlbAssetPruner(
        tenant_id=str(args.tenant_id),
        asset_type=parse_asset_type(str(args.asset_type)),
        models_public_dir=models_public_dir,
        models_url_prefix=str(args.models_url_prefix),
        asset_ids=args.asset_ids,
        style_keys=args.style_keys,
        limit=args.limit,
        delete_roots=delete_roots,
    )
    handler = build_handler(
        pruner=pruner,
        vendor_dir=VENDOR_DIR.resolve(),
        basis_dir=BASIS_DIR.resolve(),
    )
    server = ThreadingHTTPServer((str(args.host), int(args.port)), handler)
    print(f"Bad GLB asset pruner: http://{args.host}:{args.port}")
    print("Deletes the DB asset row and can unlink unreferenced local .glb files.")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping prune server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
