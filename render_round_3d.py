from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_payload(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "design_plan" in data:
        plan = data.get("design_plan")
        if isinstance(plan, dict):
            return {"plan": plan, "styled": data.get("styled_design")}
    if isinstance(data, dict):
        return {"plan": data, "styled": None}
    raise ValueError("Input JSON must be a design_plan or contain design_plan.")


def _compute_bounds(placements: list[dict], walls: list[dict]) -> dict:
    if not placements and not walls:
        return {"min_x": -2000, "max_x": 2000, "min_y": -2000, "max_y": 2000}

    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for item in placements:
        pos = item.get("position_mm", {})
        footprint = item.get("footprint_mm", {})
        x = float(pos.get("x", 0.0))
        y = float(pos.get("y", 0.0))
        w = float(footprint.get("w", 0.0))
        d = float(footprint.get("d", 0.0))

        min_x = min(min_x, x - w / 2)
        max_x = max(max_x, x + w / 2)
        min_y = min(min_y, y - d / 2)
        max_y = max(max_y, y + d / 2)

    for wall in walls:
        start = wall.get("start_mm", {})
        end = wall.get("end_mm", {})
        sx = float(start.get("x", 0.0))
        sy = float(start.get("y", 0.0))
        ex = float(end.get("x", 0.0))
        ey = float(end.get("y", 0.0))
        min_x = min(min_x, sx, ex)
        max_x = max(max_x, sx, ex)
        min_y = min(min_y, sy, ey)
        max_y = max(max_y, sy, ey)

    padding = 500.0
    return {
        "min_x": min_x - padding,
        "max_x": max_x + padding,
        "min_y": min_y - padding,
        "max_y": max_y + padding,
    }


def _render_html(payload: dict, bounds: dict) -> str:
    packed = json.dumps(
        {"plan": payload["plan"], "styled": payload["styled"], "bounds": bounds},
        ensure_ascii=True,
    )
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>Round 8 3D Layout</title>
  <style>
    html, body {{ height: 100%; margin: 0; background: #f7f6f2; color: #2a2a2a; }}
    #canvas {{ width: 100%; height: 100%; display: block; }}
    .hud {{ position: fixed; top: 12px; left: 12px; padding: 8px 12px; background: rgba(255,255,255,0.8); color: #111; font-family: sans-serif; font-size: 12px; border-radius: 6px; }}
  </style>
</head>
<body>
  <div class=\"hud\">Round 8 3D (units: mm, scale: 1m = 1000mm)</div>
  <canvas id=\"canvas\"></canvas>

  <script type="module">
    import * as THREE from '../assets/three/three.module.js';
    import {{ OrbitControls }} from '../assets/three/OrbitControls.js';

    const DATA = {packed};
    const plan = DATA.plan;
    const styled = DATA.styled || null;
    const bounds = DATA.bounds;

    const canvas = document.getElementById('canvas');
    const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true }});
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setClearColor(0xf7f6f2, 1);

    const scene = new THREE.Scene();

    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 2000);

    const scale = 0.001; // mm -> meters
    const maxDim = Math.max((bounds.max_x - bounds.min_x) * scale, (bounds.max_y - bounds.min_y) * scale);
    camera.position.set(0, Math.max(3, maxDim * 0.9), Math.max(4, maxDim * 1.2));
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0);
    controls.update();

    const ambient = new THREE.AmbientLight(0xffffff, 0.9);
    scene.add(ambient);

    const directional = new THREE.DirectionalLight(0xffffff, 0.9);
    directional.position.set(6, 10, 4);
    scene.add(directional);

    const hemi = new THREE.HemisphereLight(0xffffff, 0xf3efe7, 0.6);
    scene.add(hemi);

    const centerX = (bounds.min_x + bounds.max_x) / 2;
    const centerY = (bounds.min_y + bounds.max_y) / 2;

    const floorW = (bounds.max_x - bounds.min_x) * scale;
    const floorD = (bounds.max_y - bounds.min_y) * scale;
    const floorGeom = new THREE.PlaneGeometry(floorW, floorD);
    const floorMat = new THREE.MeshStandardMaterial({{ color: 0xf0ece4, side: THREE.DoubleSide }});
    const floor = new THREE.Mesh(floorGeom, floorMat);
    floor.rotation.x = -Math.PI / 2;
    scene.add(floor);

    const grid = new THREE.GridHelper(Math.max(floorW, floorD), 20, 0xd6d1c7, 0xd6d1c7);
    scene.add(grid);

    const paletteColors = styled?.global_style?.palette?.colors || {{}};
    const assetStyles = new Map((styled?.asset_styles || []).map((s) => [s.asset_id, s]));
    const wallStyles = new Map((styled?.wall_styles || []).map((s) => [s.wall_id, s]));
    const windowStyles = new Map((styled?.window_styles || []).map((s) => [s.window_id, s]));
    const palette = [0x8ecae6, 0x219ebc, 0x023047, 0xffb703, 0xfb8500, 0x90be6d];
    const categoryColor = new Map();

    function pickColor(category, colorRole) {{
      if (colorRole && paletteColors[colorRole]) {{
        return new THREE.Color(paletteColors[colorRole]);
      }}
      if (!categoryColor.has(category)) {{
        const idx = categoryColor.size % palette.length;
        categoryColor.set(category, palette[idx]);
      }}
      return new THREE.Color(categoryColor.get(category));
    }}

    function toColor(hex, fallback) {{
      if (!hex || typeof hex !== 'string') {{
        return fallback;
      }}
      try {{
        return new THREE.Color(hex);
      }} catch (e) {{
        return fallback;
      }}
    }}

    function addBox(group, w, h, d, color) {{
      const geom = new THREE.BoxGeometry(w, h, d);
      const mat = new THREE.MeshStandardMaterial({{ color }});
      const mesh = new THREE.Mesh(geom, mat);
      mesh.position.y = h / 2;
      group.add(mesh);
    }}

    function addTable(group, w, h, d, color) {{
      const topH = Math.max(0.04, h * 0.12);
      const legH = Math.max(0.1, h - topH);
      const top = new THREE.Mesh(
        new THREE.BoxGeometry(w, topH, d),
        new THREE.MeshStandardMaterial({{ color }})
      );
      top.position.y = legH + topH / 2;
      group.add(top);

      const legSize = Math.min(w, d) * 0.08;
      const offsets = [
        [w / 2 - legSize / 2, d / 2 - legSize / 2],
        [-w / 2 + legSize / 2, d / 2 - legSize / 2],
        [w / 2 - legSize / 2, -d / 2 + legSize / 2],
        [-w / 2 + legSize / 2, -d / 2 + legSize / 2],
      ];
      offsets.forEach(([ox, oz]) => {{
        const leg = new THREE.Mesh(
          new THREE.BoxGeometry(legSize, legH, legSize),
          new THREE.MeshStandardMaterial({{ color: 0x8d8a84 }})
        );
        leg.position.set(ox, legH / 2, oz);
        group.add(leg);
      }});
    }}

    function addSofa(group, w, h, d, color) {{
      const seatH = h * 0.45;
      const backH = h * 0.55;
      const seatD = d * 0.65;
      const backD = d * 0.25;
      const seat = new THREE.Mesh(
        new THREE.BoxGeometry(w, seatH, seatD),
        new THREE.MeshStandardMaterial({{ color }})
      );
      seat.position.set(0, seatH / 2, 0);
      group.add(seat);

      const back = new THREE.Mesh(
        new THREE.BoxGeometry(w, backH, backD),
        new THREE.MeshStandardMaterial({{ color: new THREE.Color(color).multiplyScalar(0.85) }})
      );
      back.position.set(0, seatH + backH / 2, -(d / 2 - backD / 2));
      group.add(back);
    }}

    function addChair(group, w, h, d, color) {{
      const seatH = h * 0.4;
      const backH = h * 0.6;
      const seat = new THREE.Mesh(
        new THREE.BoxGeometry(w, seatH, d * 0.7),
        new THREE.MeshStandardMaterial({{ color }})
      );
      seat.position.set(0, seatH / 2, 0);
      group.add(seat);

      const back = new THREE.Mesh(
        new THREE.BoxGeometry(w, backH, d * 0.2),
        new THREE.MeshStandardMaterial({{ color: new THREE.Color(color).multiplyScalar(0.8) }})
      );
      back.position.set(0, seatH + backH / 2, -(d / 2 - d * 0.1));
      group.add(back);
    }}

    function addBed(group, w, h, d, color) {{
      const baseH = h * 0.5;
      const headH = h * 0.7;
      const base = new THREE.Mesh(
        new THREE.BoxGeometry(w, baseH, d),
        new THREE.MeshStandardMaterial({{ color }})
      );
      base.position.set(0, baseH / 2, 0);
      group.add(base);

      const head = new THREE.Mesh(
        new THREE.BoxGeometry(w, headH, d * 0.2),
        new THREE.MeshStandardMaterial({{ color: new THREE.Color(color).multiplyScalar(0.8) }})
      );
      head.position.set(0, baseH + headH / 2, -(d / 2 - d * 0.1));
      group.add(head);
    }}

    function addWardrobe(group, w, h, d, color) {{
      const body = new THREE.Mesh(
        new THREE.BoxGeometry(w, h, d),
        new THREE.MeshStandardMaterial({{ color }})
      );
      body.position.set(0, h / 2, 0);
      group.add(body);
    }}

    function addWall(wall) {{
      const start = wall.start_mm || {{ x: 0, y: 0 }};
      const end = wall.end_mm || {{ x: 0, y: 0 }};
      const dx = end.x - start.x;
      const dy = end.y - start.y;
      const length = Math.max(0.01, Math.sqrt(dx * dx + dy * dy) * scale);
      const thickness = Math.max(0.05, (wall.thickness_mm || 200) * scale);
      const height = Math.max(0.5, (wall.height_mm || 2700) * scale);
      const midX = (start.x + end.x) / 2;
      const midY = (start.y + end.y) / 2;
      const angle = Math.atan2(dy, dx);
      const style = wallStyles.get(wall.wall_id);
      const fallback = new THREE.Color('#d8d3c8');
      const color = toColor(style?.color_hex || wall.color_hex, fallback);

      const geom = new THREE.BoxGeometry(length, height, thickness);
      const mat = new THREE.MeshStandardMaterial({{ color }});
      const mesh = new THREE.Mesh(geom, mat);
      mesh.position.set((midX - centerX) * scale, height / 2, (midY - centerY) * scale);
      mesh.rotation.y = angle;
      scene.add(mesh);
    }}

    function addWindow(window) {{
      const start = window.start_mm || {{ x: 0, y: 0 }};
      const end = window.end_mm || {{ x: 0, y: 0 }};
      const dx = end.x - start.x;
      const dy = end.y - start.y;
      const len2 = Math.sqrt(dx * dx + dy * dy) || 1;
      const length = Math.max(0.01, len2 * scale);
      const height = Math.max(0.4, (window.height_mm || 900) * scale);
      const sill = Math.max(0.0, (window.sill_height_mm || 900) * scale);
      const thickness = Math.max(0.03, (window.thickness_mm || 80) * scale);
      const midX = (start.x + end.x) / 2;
      const midY = (start.y + end.y) / 2;
      const angle = Math.atan2(dy, dx);
      const nx = -dy / len2;
      const nz = dx / len2;
      const style = windowStyles.get(window.window_id);
      const frameColor = toColor(
        style?.frame_color_hex || window.color_hex,
        new THREE.Color('#6d6b66')
      );

      const frameThickness = Math.min(Math.max(Math.min(length, height) * 0.08, 0.02), 0.12);
      const frameDepth = Math.max(thickness * 0.5, 0.02);
      const offset = thickness / 2 + frameDepth / 2 + 0.01;
      const frame = new THREE.Group();

      const horizGeom = new THREE.BoxGeometry(length, frameThickness, frameDepth);
      const vertGeom = new THREE.BoxGeometry(frameThickness, height, frameDepth);
      const frameMat = new THREE.MeshStandardMaterial({{
        color: frameColor,
        roughness: 0.45,
        metalness: 0.08,
        side: THREE.DoubleSide,
        emissive: frameColor.clone().multiplyScalar(0.08),
      }});

      const topBar = new THREE.Mesh(horizGeom, frameMat);
      topBar.position.set(0, height / 2 - frameThickness / 2, 0);
      const bottomBar = new THREE.Mesh(horizGeom, frameMat);
      bottomBar.position.set(0, -height / 2 + frameThickness / 2, 0);
      const leftBar = new THREE.Mesh(vertGeom, frameMat);
      leftBar.position.set(-length / 2 + frameThickness / 2, 0, 0);
      const rightBar = new THREE.Mesh(vertGeom, frameMat);
      rightBar.position.set(length / 2 - frameThickness / 2, 0, 0);

      frame.add(topBar);
      frame.add(bottomBar);
      frame.add(leftBar);
      frame.add(rightBar);

      const frameFront = frame.clone();
      frameFront.position.set(
        (midX - centerX) * scale + nx * offset,
        sill + height / 2,
        (midY - centerY) * scale + nz * offset
      );
      frameFront.rotation.y = angle;
      scene.add(frameFront);

      const frameBack = frame.clone();
      frameBack.position.set(
        (midX - centerX) * scale - nx * offset,
        sill + height / 2,
        (midY - centerY) * scale - nz * offset
      );
      frameBack.rotation.y = angle;
      scene.add(frameBack);
    }}

    const walls = Array.isArray(plan.walls) ? plan.walls : [];
    const planWindows = Array.isArray(plan.windows) ? plan.windows : [];
    walls.forEach((wall) => addWall(wall));

    function windowFromPlacement(item) {{
      const pos = item.position_mm || {{ x: 0, y: 0 }};
      const footprint = item.footprint_mm || {{ w: 0, d: 0 }};
      const length = footprint.w || 1000;
      const angle = THREE.MathUtils.degToRad(item.rotation_deg || 0);
      const dx = Math.cos(angle) * (length / 2);
      const dy = Math.sin(angle) * (length / 2);
      return {{
        window_id: item.asset_id,
        start_mm: {{ x: pos.x - dx, y: pos.y - dy }},
        end_mm: {{ x: pos.x + dx, y: pos.y + dy }},
        height_mm: item.height_mm || 900,
        sill_height_mm: 900,
        thickness_mm: footprint.d || 80,
      }};
    }}

    const derivedWindows = [];
    if (planWindows.length === 0) {{
      const placementsAll = Array.isArray(plan.placements) ? plan.placements : [];
      placementsAll.forEach((item) => {{
        const category = (item.category || '').toLowerCase();
        if (category.includes('window')) {{
          derivedWindows.push(windowFromPlacement(item));
        }}
      }});
    }}

    const windows = planWindows.length > 0 ? planWindows : derivedWindows;
    windows.forEach((window) => addWindow(window));

    const placements = Array.isArray(plan.placements) ? plan.placements : [];
    placements.forEach((item) => {{
      const category = item.category || 'item';
      if (String(category).toLowerCase().includes('window')) {{
        return;
      }}
      const pos = item.position_mm || {{ x: 0, y: 0 }};
      const footprint = item.footprint_mm || {{ w: 0, d: 0 }};
      const heightMm = item.height_mm ?? 300;
      const w = footprint.w * scale;
      const d = footprint.d * scale;
      const h = heightMm * scale;

      const style = assetStyles.get(item.asset_id);
      const colorRole = style?.color_role || null;
      const color = toColor(
        style?.color_hex || item.color_hex,
        pickColor(category, colorRole)
      );

      const group = new THREE.Group();
      if (category.includes('sofa')) {{
        addSofa(group, w, h, d, color);
      }} else if (category.includes('chair')) {{
        addChair(group, w, h, d, color);
      }} else if (category.includes('table')) {{
        addTable(group, w, h, d, color);
      }} else if (category.includes('bed')) {{
        addBed(group, w, h, d, color);
      }} else if (category.includes('wardrobe') || category.includes('cabinet')) {{
        addWardrobe(group, w, h, d, color);
      }} else {{
        addBox(group, w, h, d, color);
      }}

      group.position.set((pos.x - centerX) * scale, 0, (pos.y - centerY) * scale);
      group.rotation.y = THREE.MathUtils.degToRad(item.rotation_deg || 0);
      scene.add(group);
    }});

    function onResize() {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }}
    window.addEventListener('resize', onResize);

    function animate() {{
      requestAnimationFrame(animate);
      renderer.render(scene, camera);
    }}
    animate();
  </script>
</body>
</html>
    """


def render_html_from_payload(payload: dict, output_path: Path) -> None:
    plan = payload.get("plan")
    if not isinstance(plan, dict):
        raise ValueError("Payload missing plan.")
    placements = plan.get("placements") if isinstance(plan, dict) else None
    walls = plan.get("walls") if isinstance(plan, dict) else None
    placements_list = placements if isinstance(placements, list) else []
    walls_list = walls if isinstance(walls, list) else []
    bounds = _compute_bounds(placements_list, walls_list)
    html = _render_html(payload, bounds)
    output_path.write_text(html, encoding="utf-8")


def render_html_from_json(input_path: Path, output_path: Path) -> None:
    payload = _load_payload(input_path)
    render_html_from_payload(payload, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render round JSON to 3D HTML.")
    parser.add_argument(
        "input",
        nargs="?",
        default="outputs/iterations/round_8.json",
        help="Path to round JSON (default: outputs/iterations/round_8.json)",
    )
    parser.add_argument(
        "--output",
        default="outputs/round_8_3d.html",
        help="Output HTML path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    render_html_from_json(input_path, output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
