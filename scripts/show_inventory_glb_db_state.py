from __future__ import annotations

import argparse
import html
import json
import mimetypes
import sys
from collections.abc import Mapping, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TypedDict
from urllib.parse import parse_qs, urlencode, urlparse

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(BASE_DIR))

from db import PostgresAssetRepository  # noqa: E402
from db.models import Asset, AssetFilter, TenantId  # noqa: E402
from preload_inventory_glb import (  # noqa: E402
    DEFAULT_MODELS_PUBLIC_DIR,
    DEFAULT_MODELS_URL_PREFIX,
    canonicalize_style_key,
    collect_model_file_paths,
    strip_reintroduced_suffix,
)


class VariantMatch(TypedDict):
    metadata_key: str
    metadata: dict[str, object]


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8790


def escape(value: object) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def normalize_token(value: str) -> str:
    return "_".join(value.strip().lower().replace("-", "_").split())


def asset_text_attribute(asset: Asset, key: str) -> str:
    value = dict(asset.attributes or {}).get(key)
    return value if isinstance(value, str) else ""


def asset_family_tokens(asset: Asset) -> set[str]:
    raw_tokens = [
        str(asset.id),
        strip_reintroduced_suffix(str(asset.id)),
        asset_text_attribute(asset, "category"),
        asset_text_attribute(asset, "source_category"),
        asset_text_attribute(asset, "source_asset_id"),
    ]
    return {normalize_token(token) for token in raw_tokens if token.strip()}


def asset_matches_family(asset: Asset, family_key: str) -> bool:
    family = normalize_token(family_key)
    tokens = asset_family_tokens(asset)
    if family in tokens:
        return True
    prefix = f"{family}_"
    return any(token.startswith(prefix) for token in tokens)


def read_variant(asset: Asset, style_key: str) -> VariantMatch | None:
    variants_raw = dict(asset.attributes or {}).get("model_variants")
    if not isinstance(variants_raw, Mapping):
        return None

    for raw_key, raw_metadata in variants_raw.items():
        if canonicalize_style_key(str(raw_key)) != style_key:
            continue
        if isinstance(raw_metadata, Mapping):
            return {
                "metadata_key": str(raw_key),
                "metadata": {str(key): value for key, value in raw_metadata.items()},
            }
    return None


def read_json_object(value: object) -> dict[str, object] | None:
    if isinstance(value, Mapping):
        return {str(key): nested for key, nested in value.items()}
    return None


def resolve_model_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (ROOT_DIR / expanded).resolve()


def format_bool(value: bool) -> str:
    return "yes" if value else "no"


def format_file_size(path: Path) -> str:
    if not path.exists():
        return "missing"
    size = path.stat().st_size
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def format_scalar(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value if value else "-"
    if isinstance(value, int | float | bool):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def metadata_summary(
    *,
    metadata: dict[str, object],
    models_public_dir: Path,
    models_url_prefix: str,
) -> list[str]:
    orientation = read_json_object(metadata.get("orientation_review")) or {}
    override = read_json_object(metadata.get("preview_override")) or {}
    curation = read_json_object(metadata.get("curation")) or {}
    paths = sorted(
        (
            resolve_model_path(path)
            for path in collect_model_file_paths(
                metadata=metadata,
                models_public_dir=models_public_dir,
                models_url_prefix=models_url_prefix,
            )
        ),
        key=str,
    )

    lines = [
        f"  url: {format_scalar(metadata.get('url'))}",
        f"  local_path: {format_scalar(metadata.get('local_path'))}",
        (
            "  objaverse_uid: "
            f"{format_scalar(metadata.get('objaverse_uid'))} "
            f"similarity={format_scalar(metadata.get('similarity'))}"
        ),
        (
            "  curation: "
            f"status={format_scalar(curation.get('status'))} "
            f"source={format_scalar(curation.get('candidate_source'))} "
            f"source_asset={format_scalar(curation.get('candidate_source_asset_id'))}"
        ),
        (
            "  orientation: "
            f"status={format_scalar(orientation.get('status'))} "
            f"front={format_scalar(orientation.get('front_axis'))} "
            f"offset={format_scalar(orientation.get('rotation_deg_offset'))} "
            f"x={format_scalar(orientation.get('rotation_deg_x'))} "
            f"z={format_scalar(orientation.get('rotation_deg_z'))}"
        ),
        (
            "  preview_override: "
            f"enabled={format_bool(bool(override.get('enabled')))} "
            f"contexts={format_scalar(override.get('contexts'))}"
        ),
    ]

    if not paths:
        lines.append("  files: none")
        return lines

    lines.append("  files:")
    for path in paths:
        lines.append(
            f"    - {path} exists={format_bool(path.exists())} size={format_file_size(path)}"
        )
    return lines


def load_assets(*, tenant_id: str, family_key: str) -> list[Asset]:
    repo = PostgresAssetRepository()
    assets = list(
        repo.list_assets(AssetFilter(tenant_id=TenantId(tenant_id), type="FURNITURE"))
    )
    matched = [asset for asset in assets if asset_matches_family(asset, family_key)]
    matched.sort(key=lambda asset: str(asset.id))
    return matched


def render_rows(
    *,
    assets: Sequence[Asset],
    style_key: str,
    models_public_dir: Path,
    models_url_prefix: str,
) -> str:
    lines: list[str] = []
    rows_with_variant = 0
    rows_with_existing_file = 0
    for index, asset in enumerate(assets, start=1):
        variant = read_variant(asset, style_key)
        lines.append(f"[{index:02d}] {asset.id} | {asset.name}")
        lines.append(f"  material: {format_scalar(asset.material)}")
        lines.append(f"  style_tags: {format_scalar(asset.style_tags)}")
        if variant is None:
            lines.append(f"  variant[{style_key}]: missing")
            lines.append("")
            continue

        rows_with_variant += 1
        metadata = variant["metadata"]
        file_paths = [
            resolve_model_path(path)
            for path in collect_model_file_paths(
                metadata=metadata,
                models_public_dir=models_public_dir,
                models_url_prefix=models_url_prefix,
            )
        ]
        if any(path.exists() for path in file_paths):
            rows_with_existing_file += 1
        lines.append(
            f"  variant[{style_key}]: present metadata_key={variant['metadata_key']}"
        )
        lines.extend(
            metadata_summary(
                metadata=metadata,
                models_public_dir=models_public_dir,
                models_url_prefix=models_url_prefix,
            )
        )
        lines.append("")

    lines.insert(
        0,
        (
            f"matched_assets={len(assets)} "
            f"variants_present={rows_with_variant} "
            f"variants_with_existing_file={rows_with_existing_file}"
        ),
    )
    return "\n".join(lines).rstrip() + "\n"


def variant_model_paths(
    *,
    asset: Asset,
    style_key: str,
    models_public_dir: Path,
    models_url_prefix: str,
) -> list[Path]:
    variant = read_variant(asset, style_key)
    if variant is None:
        return []
    return sorted(
        (
            resolve_model_path(path)
            for path in collect_model_file_paths(
                metadata=variant["metadata"],
                models_public_dir=models_public_dir,
                models_url_prefix=models_url_prefix,
            )
        ),
        key=str,
    )


def first_existing_variant_path(
    *,
    asset: Asset,
    style_key: str,
    models_public_dir: Path,
    models_url_prefix: str,
) -> Path | None:
    paths = variant_model_paths(
        asset=asset,
        style_key=style_key,
        models_public_dir=models_public_dir,
        models_url_prefix=models_url_prefix,
    )
    for path in paths:
        if path.exists():
            return path
    return paths[0] if paths else None


def find_asset(*, assets: Sequence[Asset], asset_id: str) -> Asset | None:
    for asset in assets:
        if str(asset.id) == asset_id:
            return asset
    return None


def first_visual_asset(
    *,
    assets: Sequence[Asset],
    style_key: str,
    models_public_dir: Path,
    models_url_prefix: str,
) -> Asset | None:
    for asset in assets:
        path = first_existing_variant_path(
            asset=asset,
            style_key=style_key,
            models_public_dir=models_public_dir,
            models_url_prefix=models_url_prefix,
        )
        if path is not None and path.exists():
            return asset
    return assets[0] if assets else None


def read_number(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def metadata_rotation(metadata: dict[str, object]) -> tuple[float, float, float]:
    override = read_json_object(metadata.get("preview_override")) or {}
    review = read_json_object(metadata.get("orientation_review")) or {}
    offset = read_number(override.get("rotation_deg_offset"))
    if offset is None:
        offset = read_number(review.get("rotation_deg_offset")) or 0.0
    rotation_x = read_number(override.get("rotation_deg_x"))
    if rotation_x is None:
        rotation_x = read_number(review.get("rotation_deg_x")) or 0.0
    rotation_z = read_number(override.get("rotation_deg_z"))
    if rotation_z is None:
        rotation_z = read_number(review.get("rotation_deg_z")) or 0.0
    return offset, rotation_x, rotation_z


class VisualInventoryHandler(BaseHTTPRequestHandler):
    assets: Sequence[Asset]
    tenant_id: str
    family_key: str
    style_key: str
    models_public_dir: Path
    models_url_prefix: str
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
                model_path = self._resolve_model_path(asset_id=asset_id)
                self._send_file(model_path, default_content_type="model/gltf-binary")
                return
            if parsed.path.startswith("/vendor/"):
                self._send_vendor_file(parsed.path.removeprefix("/vendor/"))
                return
            self._send_plain("Not found.", status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_plain(str(exc), status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _selected_asset(self, raw_asset_id: str) -> Asset | None:
        requested = find_asset(assets=self.assets, asset_id=raw_asset_id)
        if requested is not None:
            return requested
        return first_visual_asset(
            assets=self.assets,
            style_key=self.style_key,
            models_public_dir=self.models_public_dir,
            models_url_prefix=self.models_url_prefix,
        )

    def _resolve_model_path(self, *, asset_id: str) -> Path:
        asset = find_asset(assets=self.assets, asset_id=asset_id)
        if asset is None:
            raise FileNotFoundError(f"Asset was not found: {asset_id}")
        path = first_existing_variant_path(
            asset=asset,
            style_key=self.style_key,
            models_public_dir=self.models_public_dir,
            models_url_prefix=self.models_url_prefix,
        )
        if path is None:
            raise FileNotFoundError(
                f"No GLB variant for {asset_id} [{self.style_key}]."
            )
        if not path.exists():
            raise FileNotFoundError(f"GLB file does not exist: {path}")
        return path

    def _build_index(self, query: str) -> bytes:
        params = parse_qs(query)
        selected_asset = self._selected_asset(params.get("asset_id", [""])[0])
        selected_asset_id = str(selected_asset.id) if selected_asset is not None else ""
        options = "\n".join(
            self._asset_option(asset=asset, selected_asset_id=selected_asset_id)
            for asset in self.assets
        )
        metadata_block = (
            self._metadata_block(asset=selected_asset)
            if selected_asset is not None
            else "No matching assets."
        )
        viewer_src = (
            "/viewer?" + urlencode({"asset_id": selected_asset_id})
            if selected_asset is not None
            else ""
        )
        rows_with_variant = sum(
            1
            for asset in self.assets
            if read_variant(asset, self.style_key) is not None
        )
        rows_with_file = sum(
            1
            for asset in self.assets
            if any(
                path.exists()
                for path in variant_model_paths(
                    asset=asset,
                    style_key=self.style_key,
                    models_public_dir=self.models_public_dir,
                    models_url_prefix=self.models_url_prefix,
                )
            )
        )
        document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Inventory GLB Visual Check</title>
  <style>
    body {{ margin: 0; font: 14px system-ui, sans-serif; color: #111; background: #f6f6f4; }}
    .layout {{ display: grid; grid-template-columns: 430px 1fr; height: 100vh; }}
    aside {{ padding: 14px; border-right: 1px solid #ccc; overflow: auto; background: #fff; }}
    main {{ min-width: 0; background: #202020; }}
    h1 {{ font-size: 18px; margin: 0 0 10px; }}
    label {{ display: block; font-weight: 700; margin: 12px 0 6px; }}
    select {{ width: 100%; height: 34vh; font: inherit; }}
    iframe {{ width: 100%; height: 100%; border: 0; }}
    pre {{ white-space: pre-wrap; word-break: break-word; padding: 10px; background: #f4f4f2; border: 1px solid #ddd; border-radius: 6px; font-size: 12px; }}
    .muted {{ color: #666; font-size: 12px; }}
    .status {{ color: #176b32; margin: 8px 0; }}
    @media (max-width: 900px) {{ .layout {{ grid-template-columns: 1fr; grid-template-rows: auto 65vh; }} select {{ height: 22vh; }} }}
  </style>
</head>
<body>
  <div class="layout">
    <aside>
      <h1>Inventory GLB Visual Check</h1>
      <div class="status">tenant={escape(self.tenant_id)} | family={escape(self.family_key)} | style={escape(self.style_key)}</div>
      <div class="muted">matched={len(self.assets)} | variants={rows_with_variant} | files={rows_with_file}</div>
      <form method="get" action="/">
        <label for="asset_id">Assets</label>
        <select id="asset_id" name="asset_id" onchange="location.href='/?asset_id=' + encodeURIComponent(this.value);">
          {options}
        </select>
        <noscript><button type="submit">Open selected</button></noscript>
      </form>
      <label>DB metadata</label>
      <pre>{escape(metadata_block)}</pre>
    </aside>
    <main>
      <iframe src="{escape(viewer_src)}" title="GLB viewer"></iframe>
    </main>
  </div>
</body>
</html>"""
        return document.encode("utf-8")

    def _build_viewer(self, query: str) -> bytes:
        params = parse_qs(query)
        asset_id = params.get("asset_id", [""])[0]
        asset = find_asset(assets=self.assets, asset_id=asset_id)
        if asset is None:
            raise FileNotFoundError(f"Asset was not found: {asset_id}")
        variant = read_variant(asset, self.style_key)
        metadata = variant["metadata"] if variant is not None else {}
        offset, rotation_x, rotation_z = metadata_rotation(metadata)
        model_url = "/model?" + urlencode({"asset_id": asset_id})
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
  <div id="status">Loading...</div>
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
    world.add(new THREE.GridHelper(4, 8, 0x999999, 0x444444));
    world.add(new THREE.ArrowHelper(
      new THREE.Vector3(0, 0, -1),
      new THREE.Vector3(0, 0.08, 1.0),
      1.5,
      0xffffff,
      0.25,
      0.12
    ));

    function tick() {{
      requestAnimationFrame(tick);
      controls.update();
      renderer.render(scene, camera);
    }}
    tick();
    window.addEventListener('resize', () => {{
      renderer.setSize(window.innerWidth, window.innerHeight);
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
    }});

    main().catch((error) => {{
      status.textContent = 'Viewer error: ' + (error instanceof Error ? error.message : String(error));
      console.error(error);
    }});

    async function main() {{
      status.textContent = 'Fetching GLB...';
      const response = await fetch({json.dumps(model_url)}, {{ cache: 'no-store' }});
      if (!response.ok) throw new Error(`GLB request failed: ${{response.status}}`);
      const bytes = await response.arrayBuffer();
      const gltf = await new Promise((resolve, reject) => {{
        new GLTFLoader().parse(bytes, '', resolve, reject);
      }});
      const sceneSource = gltf.scene || (Array.isArray(gltf.scenes) ? gltf.scenes[0] : null);
      if (!sceneSource || typeof sceneSource.clone !== 'function') throw new Error('GLB has no scene.');

      const root = new THREE.Group();
      const yawRoot = new THREE.Group();
      const tiltRoot = new THREE.Group();
      const model = sceneSource.clone(true);
      yawRoot.rotation.y = THREE.MathUtils.degToRad({offset});
      tiltRoot.rotation.x = THREE.MathUtils.degToRad({rotation_x});
      tiltRoot.rotation.z = THREE.MathUtils.degToRad({rotation_z});
      tiltRoot.add(model);
      yawRoot.add(tiltRoot);
      root.add(yawRoot);
      world.add(root);

      root.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(root);
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      if (!Number.isFinite(maxDim) || maxDim <= 0) throw new Error('Empty GLB bounds.');
      root.scale.setScalar(Math.min(
        1.8 / Math.max(size.x, size.z, 0.001),
        1.6 / Math.max(size.x, size.y, size.z, 0.001)
      ));
      root.updateMatrixWorld(true);
      const fitted = new THREE.Box3().setFromObject(root);
      const center = fitted.getCenter(new THREE.Vector3());
      root.position.x -= center.x;
      root.position.z -= center.z;
      root.position.y -= fitted.min.y;
      root.traverse((node) => {{
        if (!node.isMesh) return;
        node.castShadow = true;
        node.receiveShadow = true;
        const materials = Array.isArray(node.material) ? node.material : [node.material];
        for (const material of materials) {{
          material.side = THREE.DoubleSide;
          material.needsUpdate = true;
        }}
      }});
      status.textContent = {json.dumps(f"{asset_id} [{self.style_key}] offset={offset:g} x={rotation_x:g} z={rotation_z:g}")};
    }}
  </script>
</body>
</html>"""
        return document.encode("utf-8")

    def _asset_option(self, *, asset: Asset, selected_asset_id: str) -> str:
        variant = read_variant(asset, self.style_key)
        paths = variant_model_paths(
            asset=asset,
            style_key=self.style_key,
            models_public_dir=self.models_public_dir,
            models_url_prefix=self.models_url_prefix,
        )
        status = "ready" if any(path.exists() for path in paths) else "missing"
        if variant is None:
            status = "no variant"
        selected = " selected" if str(asset.id) == selected_asset_id else ""
        label = f"{asset.id} | {status}"
        return f'<option value="{escape(asset.id)}"{selected}>{escape(label)}</option>'

    def _metadata_block(self, *, asset: Asset | None) -> str:
        if asset is None:
            return "No asset selected."
        variant = read_variant(asset, self.style_key)
        if variant is None:
            return f"{asset.id}\nvariant[{self.style_key}]: missing"
        return "\n".join(
            [
                f"{asset.id} | {asset.name}",
                f"material: {format_scalar(asset.material)}",
                f"style_tags: {format_scalar(asset.style_tags)}",
                f"variant[{self.style_key}]: present metadata_key={variant['metadata_key']}",
                *metadata_summary(
                    metadata=variant["metadata"],
                    models_public_dir=self.models_public_dir,
                    models_url_prefix=self.models_url_prefix,
                ),
            ]
        )

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


def build_visual_handler(
    *,
    assets: Sequence[Asset],
    tenant_id: str,
    family_key: str,
    style_key: str,
    models_public_dir: Path,
    models_url_prefix: str,
    vendor_dir: Path,
) -> type[VisualInventoryHandler]:
    class BoundVisualInventoryHandler(VisualInventoryHandler):
        pass

    BoundVisualInventoryHandler.assets = assets
    BoundVisualInventoryHandler.tenant_id = tenant_id
    BoundVisualInventoryHandler.family_key = family_key
    BoundVisualInventoryHandler.style_key = style_key
    BoundVisualInventoryHandler.models_public_dir = models_public_dir
    BoundVisualInventoryHandler.models_url_prefix = models_url_prefix
    BoundVisualInventoryHandler.vendor_dir = vendor_dir
    return BoundVisualInventoryHandler


def serve_visual(
    *,
    assets: Sequence[Asset],
    tenant_id: str,
    family_key: str,
    style_key: str,
    models_public_dir: Path,
    models_url_prefix: str,
    host: str,
    port: int,
) -> None:
    handler = build_visual_handler(
        assets=assets,
        tenant_id=tenant_id,
        family_key=family_key,
        style_key=style_key,
        models_public_dir=models_public_dir,
        models_url_prefix=models_url_prefix,
        vendor_dir=(ROOT_DIR / "frontend/src/vendor/three").resolve(),
    )
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Inventory GLB visual check: http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping visual check server.")
    finally:
        server.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect one inventory GLB family/style directly from Postgres."
    )
    parser.add_argument("--tenant-id", default="demo_tenant")
    parser.add_argument("--family", default="coffee_table")
    parser.add_argument("--style", default="modern")
    parser.add_argument(
        "--models-public-dir",
        type=Path,
        default=DEFAULT_MODELS_PUBLIC_DIR,
    )
    parser.add_argument("--models-url-prefix", default=DEFAULT_MODELS_URL_PREFIX)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    tenant_id = str(args.tenant_id)
    family_key = normalize_token(str(args.family))
    style_key = canonicalize_style_key(str(args.style))
    models_public_dir = (ROOT_DIR / args.models_public_dir).resolve()
    assets = load_assets(tenant_id=tenant_id, family_key=family_key)
    if args.serve:
        serve_visual(
            assets=assets,
            tenant_id=tenant_id,
            family_key=family_key,
            style_key=style_key,
            models_public_dir=models_public_dir,
            models_url_prefix=str(args.models_url_prefix),
            host=str(args.host),
            port=int(args.port),
        )
        return

    header = (
        f"tenant={tenant_id}\n"
        f"family={family_key}\n"
        f"style={style_key}\n"
        f"models_public_dir={models_public_dir}\n"
    )
    sys.stdout.write(header)
    sys.stdout.write(
        render_rows(
            assets=assets,
            style_key=style_key,
            models_public_dir=models_public_dir,
            models_url_prefix=str(args.models_url_prefix),
        )
    )


if __name__ == "__main__":
    main()
