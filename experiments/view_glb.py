from __future__ import annotations

import argparse
import html
import mimetypes
import shutil
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

try:
    import trimesh
except ImportError:  # pragma: no cover - optional helper dependency
    trimesh = None


DEFAULT_GLB_PATH = Path(__file__).resolve().parent / "hunyuan3d_outputs" / "chair.glb"
THREE_VENDOR_ROOT = (
    Path(__file__).resolve().parents[1] / "frontend" / "src" / "vendor" / "three"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve and view a local GLB file in the browser."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_GLB_PATH,
        help="Path to the GLB file to inspect.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for the local viewer.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for the local viewer.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the browser automatically.",
    )
    return parser.parse_args()


def summarize_glb(glb_path: Path) -> str:
    if trimesh is None:
        return "Mesh summary unavailable because trimesh is not installed."

    scene = trimesh.load(glb_path, force="scene")
    geometry_count = len(scene.geometry)
    vertex_count = 0
    face_count = 0

    for geometry in scene.geometry.values():
        if isinstance(geometry, trimesh.Trimesh):
            vertex_count += len(geometry.vertices)
            face_count += len(geometry.faces)

    extents = scene.extents
    return "\n".join(
        [
            f"file={glb_path.name}",
            f"geometry={geometry_count}",
            f"vertices={vertex_count}",
            f"faces={face_count}",
            f"extents={extents[0]:.4f}, {extents[1]:.4f}, {extents[2]:.4f}",
        ]
    )


def build_html(*, model_name: str, metadata: str) -> bytes:
    template = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>GLB Viewer</title>
    <style>
      :root {
        color-scheme: light;
        font-family: "SF Pro Display", "Helvetica Neue", Helvetica, Arial, sans-serif;
        background:
          radial-gradient(circle at top, rgba(255, 234, 201, 0.95), rgba(244, 239, 230, 0.9) 42%, rgba(227, 231, 236, 0.95) 100%);
        color: #1f2937;
      }

      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        min-height: 100vh;
        overflow: hidden;
      }

      canvas {
        display: block;
      }

      .panel {
        position: fixed;
        top: 18px;
        left: 18px;
        width: min(360px, calc(100vw - 36px));
        padding: 16px 18px;
        border: 1px solid rgba(148, 163, 184, 0.35);
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.76);
        backdrop-filter: blur(18px);
        box-shadow: 0 18px 50px rgba(15, 23, 42, 0.14);
      }

      .eyebrow {
        margin: 0 0 8px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #b45309;
      }

      h1 {
        margin: 0;
        font-size: 20px;
        line-height: 1.2;
      }

      .status {
        margin-top: 10px;
        font-size: 14px;
        color: #475569;
      }

      pre {
        margin: 14px 0 0;
        padding: 12px;
        border-radius: 12px;
        background: rgba(248, 250, 252, 0.9);
        font-size: 12px;
        line-height: 1.55;
        color: #0f172a;
        overflow: auto;
      }

      .hint {
        margin-top: 12px;
        font-size: 12px;
        color: #64748b;
      }
    </style>
  </head>
  <body>
    <aside class="panel">
      <p class="eyebrow">Local Viewer</p>
      <h1>__MODEL_NAME__</h1>
      <div class="status" id="status">Loading model...</div>
      <pre>__METADATA__</pre>
      <div class="hint">Drag to orbit, scroll to zoom, right drag to pan.</div>
    </aside>

    <script type="module">
      import * as THREE from "/vendor/three.module.js";
      import { OrbitControls } from "/vendor/OrbitControls.js";
      import { GLTFLoader } from "/vendor/loaders/GLTFLoader.js";

      const statusEl = document.getElementById("status");
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0xf4efe6);

      const renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setPixelRatio(window.devicePixelRatio);
      renderer.setSize(window.innerWidth, window.innerHeight);
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.shadowMap.enabled = true;
      document.body.appendChild(renderer.domElement);

      const camera = new THREE.PerspectiveCamera(42, window.innerWidth / window.innerHeight, 0.01, 100);
      camera.position.set(2.6, 1.8, 2.6);

      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.autoRotate = true;
      controls.autoRotateSpeed = 1.1;
      controls.target.set(0, 0.4, 0);

      const hemiLight = new THREE.HemisphereLight(0xfffbeb, 0x94a3b8, 1.8);
      scene.add(hemiLight);

      const keyLight = new THREE.DirectionalLight(0xffffff, 2.1);
      keyLight.position.set(4, 6, 5);
      keyLight.castShadow = true;
      scene.add(keyLight);

      const fillLight = new THREE.DirectionalLight(0xfef3c7, 0.8);
      fillLight.position.set(-3, 2, -4);
      scene.add(fillLight);

      const floor = new THREE.Mesh(
        new THREE.CircleGeometry(6, 96),
        new THREE.MeshStandardMaterial({
          color: 0xe2e8f0,
          metalness: 0.05,
          roughness: 0.92,
        })
      );
      floor.rotation.x = -Math.PI / 2;
      floor.position.y = -0.85;
      floor.receiveShadow = true;
      scene.add(floor);

      const grid = new THREE.GridHelper(8, 20, 0xcbd5e1, 0xe2e8f0);
      grid.position.y = -0.84;
      scene.add(grid);

      const loader = new GLTFLoader();
      loader.load(
        "/model.glb",
        (gltf) => {
          const root = gltf.scene ?? gltf.scenes[0];
          root.traverse((node) => {
            if (node.isMesh) {
              node.castShadow = true;
              node.receiveShadow = true;
            }
          });
          scene.add(root);
          frameModel(root);
          statusEl.textContent = "Ready";
        },
        (event) => {
          if (!event.total) {
            statusEl.textContent = "Loading model...";
            return;
          }
          const progress = Math.round((event.loaded / event.total) * 100);
          statusEl.textContent = `Loading model... ${progress}%`;
        },
        (error) => {
          statusEl.textContent = "Failed to load model. Check the terminal for details.";
          console.error(error);
        }
      );

      function frameModel(object) {
        const box = new THREE.Box3().setFromObject(object);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDimension = Math.max(size.x, size.y, size.z, 0.001);
        const scale = 2.2 / maxDimension;

        object.position.sub(center);
        object.scale.setScalar(scale);

        const framedBox = new THREE.Box3().setFromObject(object);
        const framedSize = framedBox.getSize(new THREE.Vector3());
        const radius = Math.max(framedSize.length() * 0.5, 0.6);

        camera.position.set(radius * 1.8, radius * 1.15, radius * 1.8);
        camera.near = Math.max(radius / 100, 0.01);
        camera.far = radius * 30;
        camera.updateProjectionMatrix();

        controls.target.set(0, 0, 0);
        controls.maxDistance = radius * 10;
        controls.update();
      }

      function render() {
        requestAnimationFrame(render);
        controls.update();
        renderer.render(scene, camera);
      }

      window.addEventListener("resize", () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
      });

      render();
    </script>
  </body>
</html>
"""
    rendered = template.replace("__MODEL_NAME__", html.escape(model_name)).replace(
        "__METADATA__",
        html.escape(metadata),
    )
    return rendered.encode("utf-8")


def build_handler(*, glb_path: Path, metadata: str) -> type[BaseHTTPRequestHandler]:
    html_payload = build_html(model_name=glb_path.name, metadata=metadata)

    class ViewerHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            request_path = parsed.path

            if request_path in {"/", "/index.html"}:
                self._send_bytes(html_payload, "text/html; charset=utf-8")
                return

            if request_path == "/model.glb":
                self._send_file(glb_path, "model/gltf-binary")
                return

            if request_path.startswith("/vendor/"):
                relative_path = Path(request_path.removeprefix("/vendor/"))
                vendor_path = (THREE_VENDOR_ROOT / relative_path).resolve()
                try:
                    vendor_path.relative_to(THREE_VENDOR_ROOT.resolve())
                except ValueError:
                    self.send_error(HTTPStatus.NOT_FOUND, "Unknown asset")
                    return

                if not vendor_path.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND, "Unknown asset")
                    return

                self._send_file(vendor_path)
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def log_message(self, format: str, *args: object) -> None:
            print(f"[view_glb] {self.address_string()} - {format % args}")

        def _send_bytes(self, payload: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_file(self, path: Path, content_type: str | None = None) -> None:
            resolved_content_type = (
                content_type
                or mimetypes.guess_type(path.name)[0]
                or "application/octet-stream"
            )
            if path.suffix == ".js":
                resolved_content_type = "text/javascript; charset=utf-8"

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", resolved_content_type)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()

            with path.open("rb") as input_file:
                shutil.copyfileobj(input_file, self.wfile)

    return ViewerHandler


def main() -> int:
    args = parse_args()
    glb_path = args.input.resolve()

    if not glb_path.exists():
        raise FileNotFoundError(f"GLB file not found: {glb_path}")
    if not THREE_VENDOR_ROOT.exists():
        raise FileNotFoundError(
            f"Three.js vendor directory not found: {THREE_VENDOR_ROOT}"
        )

    metadata = summarize_glb(glb_path)
    handler = build_handler(glb_path=glb_path, metadata=metadata)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    viewer_url = f"http://{args.host}:{args.port}/"

    print(f"Serving {glb_path}")
    print(metadata)
    print(f"Viewer URL: {viewer_url}")

    if args.open:
        webbrowser.open(viewer_url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down viewer...")
    finally:
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
