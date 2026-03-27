import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useEditorStore } from '../store';
import { polygonBounds } from '../utils/geometry';
import { getNumericBbox, getWorldBboxFromPlacement, placementMatchesStyledObject } from '../utils/styledLayout';
import { buildObjectProxy } from '../three/proxies';

type RenderObject = {
  id: string;
  type: string;
  colorHex: string;
  material?: string | null;
  rotationCcw?: number;
  bbox: { minX: number; minY: number; maxX: number; maxY: number };
  placeOn?: { method: string; targetId?: string | null } | null;
  draggable?: boolean;
};

const MM_TO_M = 0.001;

function defaultHeightMm(type: string): number {
  const t = type.toLowerCase();
  if (t.includes('rug') || t.includes('mat')) return 10;
  if (t.includes('wall_art') || t.includes('clock') || t.includes('mirror')) return 30;
  if (t.includes('ceiling_light') || t.includes('pendant_light') || t.includes('track_light'))
    return 200;
  if (t.includes('air_conditioner')) return 250;
  if (t.includes('lamp')) return 1400;
  if (t.includes('chair')) return 900;
  if (t.includes('table')) return 500;
  return 800;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function planRotationToYawRad(rotationDeg: number): number {
  return (rotationDeg * Math.PI) / 180 + Math.PI;
}

function yawRadToPlanRotationDeg(yawRad: number): number {
  const rotationDeg = (yawRad * 180) / Math.PI - 180;
  return ((rotationDeg % 360) + 360) % 360;
}

export const ThreeDPreviewPanel: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const {
    viewMode,
    floorplan,
    design,
    styledResult,
    inventory,
    userInput,
    updatePlacement,
    removePlacement,
    setDesign,
  } = useEditorStore();
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [editMode, setEditMode] = useState<'idle' | 'edit'>('idle');
  const initialDesignRef = useRef<any>(null);
  const activeFloorplan = floorplan ?? design?.floorplan_geometry ?? null;
  const editModeRef = useRef<'idle' | 'edit'>('idle');
  const selectedIdRef = useRef<string | null>(null);
  const updateRotatorsVisibilityRef = useRef<(() => void) | null>(null);
  const controlsRef = useRef<any>(null);

  useEffect(() => {
    editModeRef.current = editMode;
    selectedIdRef.current = selectedId;
    if (!selectedId && editMode !== 'idle') {
      setEditMode('idle');
      return;
    }
    if (controlsRef.current) {
      controlsRef.current.enabled = editMode === 'idle';
    }
    if (import.meta.env.DEV) {
      console.debug('[3D]', {
        editMode,
        selectedId,
        controlsEnabled: controlsRef.current?.enabled,
      });
    }
    updateRotatorsVisibilityRef.current?.();
  }, [editMode, selectedId]);

  const objects: RenderObject[] = useMemo(() => {
    const out: RenderObject[] = [];
    const placements = design?.placements ?? [];
    const placementsById = new Map<string, any>();
    for (const p of placements) {
      const pid = String(p.placement_id ?? p.id ?? p.object_id ?? '');
      if (!pid) continue;
      placementsById.set(pid, p);
    }

    const rawObjects = styledResult?.objects;
    if (Array.isArray(rawObjects)) {
      const used = new Set<string>();
      for (const obj of rawObjects) {
        const id = String(obj.instance_id ?? obj.id ?? obj.object_type ?? `obj_${out.length + 1}`);
        const objType = String(obj.object_type ?? obj.category ?? 'item');
        const placeOn = obj.place_on
          ? {
              method: String(obj.place_on.method ?? 'floor'),
              targetId: obj.place_on.target_instance_id ?? null,
            }
          : null;
        const placement = placementsById.get(id);
        const styledBbox = getNumericBbox(obj);
        let minX = styledBbox.minX;
        let minY = styledBbox.minY;
        let maxX = styledBbox.maxX;
        let maxY = styledBbox.maxY;
        let rotationCcw = Number(obj.rotation_ccw ?? 0);

        if (placement) {
          used.add(id);
          if (!placementMatchesStyledObject(placement, styledBbox, rotationCcw)) {
            const placementBbox = getWorldBboxFromPlacement(placement);
            minX = placementBbox.minX;
            minY = placementBbox.minY;
            maxX = placementBbox.maxX;
            maxY = placementBbox.maxY;
            rotationCcw = Number(placement.rotation_deg ?? rotationCcw ?? 0);
          }
        }

        out.push({
          id,
          type: objType,
          colorHex: String(obj.color_hex ?? '#cfcac2'),
          material: obj.material ?? null,
          rotationCcw,
          bbox: { minX, minY, maxX, maxY },
          placeOn,
          draggable: placementsById.has(id),
        });
      }

      for (const p of placements) {
        const pid = String(p.placement_id ?? p.id ?? p.object_id ?? '');
        if (!pid || used.has(pid)) continue;
        const bbox = getWorldBboxFromPlacement(p);
        out.push({
          id: pid,
          type: p.category,
          colorHex: p.color_hex ?? '#cfcac2',
          material: null,
          rotationCcw: Number(p.rotation_deg ?? 0),
          bbox,
          placeOn: { method: 'floor', targetId: 'floor' },
          draggable: true,
        });
      }

      return out;
    }

    for (const placement of placements) {
      const pid = String(placement.placement_id ?? placement.id ?? placement.object_id ?? '');
      out.push({
        id: pid || String(out.length + 1),
        type: placement.category,
        colorHex: placement.color_hex ?? '#cfcac2',
        material: null,
        rotationCcw: Number(placement.rotation_deg ?? 0),
        bbox: getWorldBboxFromPlacement(placement),
        placeOn: { method: 'floor', targetId: 'floor' },
        draggable: true,
      });
    }
    return out;
  }, [styledResult, design]);

  useEffect(() => {
    if (viewMode !== '3d') return;
    if (!containerRef.current) return;
    if (!activeFloorplan) {
      setError('Missing floorplan data for 3D preview.');
      return;
    }

    let disposed = false;
    let raf = 0;
    let renderer: any = null;
    let controls: any = null;
    let resizeObserver: ResizeObserver | null = null;

    const el = containerRef.current;
    setError(null);

    const run = async (width: number, height: number) => {
      try {
        const THREE_URL = new URL('../vendor/three/three.module.js', import.meta.url).toString();
        const ORBIT_URL = new URL('../vendor/three/OrbitControls.js', import.meta.url).toString();

        const THREE: any = await import(/* @vite-ignore */ THREE_URL);
        const controlsMod: any = await import(/* @vite-ignore */ ORBIT_URL);
        const OrbitControls = controlsMod.OrbitControls;

        if (disposed || !el.isConnected) return;
        if (import.meta.env.DEV) {
          console.debug('[3D] init', { width, height });
        }

        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setPixelRatio(window.devicePixelRatio || 1);
        renderer.setSize(width, height);
        renderer.setClearColor(0xf6f4ef, 1);
        renderer.domElement.style.touchAction = 'none';
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        renderer.domElement.style.width = '100%';
        renderer.domElement.style.height = '100%';
        renderer.domElement.style.display = 'block';
        const gl = renderer.getContext?.();
        if (!gl || (typeof gl.isContextLost === 'function' && gl.isContextLost())) {
          setError('WebGL context unavailable. Please enable WebGL in your browser.');
          return;
        }
        while (el.firstChild) {
          el.removeChild(el.firstChild);
        }
        el.appendChild(renderer.domElement);

        const scene = new THREE.Scene();
        const ambient = new THREE.AmbientLight(0xffffff, 0.85);
        scene.add(ambient);
        const dir = new THREE.DirectionalLight(0xffffff, 0.9);
        dir.position.set(6, 10, 4);
        dir.castShadow = true;
        dir.shadow.mapSize.width = 2048;
        dir.shadow.mapSize.height = 2048;
        dir.shadow.camera.near = 0.1;
        dir.shadow.camera.far = 60;
        scene.add(dir);
        const hemi = new THREE.HemisphereLight(0xffffff, 0xf3efe7, 0.6);
        scene.add(hemi);

        const roomPtsRaw =
          activeFloorplan.room?.polygon_mm ??
          activeFloorplan.room?.polygon_ccw ??
          styledResult?.room?.polygon_mm ??
          styledResult?.room?.polygon_ccw ??
          [];
        const roomPts = Array.isArray(roomPtsRaw)
          ? roomPtsRaw
              .map((p: any) => ({ x: Number(p?.x ?? 0), y: Number(p?.y ?? 0) }))
              .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y))
          : [];
        if (!initialDesignRef.current && design) {
          initialDesignRef.current = JSON.parse(JSON.stringify(design));
        }
        const roomBounds = roomPts.length >= 3 ? polygonBounds(roomPts) : null;
        let minX = roomBounds?.minX ?? 0;
        let minY = roomBounds?.minY ?? 0;
        let maxX = roomBounds?.maxX ?? 6000;
        let maxY = roomBounds?.maxY ?? 4000;
        if (!roomBounds && objects.length) {
          minX = Infinity;
          minY = Infinity;
          maxX = -Infinity;
          maxY = -Infinity;
        }

        const roomCenterX = roomBounds ? (roomBounds.minX + roomBounds.maxX) / 2 : (minX + maxX) / 2;
        const roomCenterY = roomBounds ? (roomBounds.minY + roomBounds.maxY) / 2 : (minY + maxY) / 2;
        const roomSpan = roomBounds
          ? Math.max(1, roomBounds.maxX - roomBounds.minX, roomBounds.maxY - roomBounds.minY)
          : null;
        const outlierLimit = roomSpan ? roomSpan * 4 : null;

        for (const obj of objects) {
          const { minX: ox1, minY: oy1, maxX: ox2, maxY: oy2 } = obj.bbox;
          if (![ox1, oy1, ox2, oy2].every((v) => Number.isFinite(v))) continue;
          if (outlierLimit !== null) {
            const cx = (ox1 + ox2) / 2;
            const cy = (oy1 + oy2) / 2;
            const dist = Math.hypot(cx - roomCenterX, cy - roomCenterY);
            if (dist > outlierLimit) continue;
          }
          minX = Math.min(minX, ox1);
          minY = Math.min(minY, oy1);
          maxX = Math.max(maxX, ox2);
          maxY = Math.max(maxY, oy2);
        }

        if (!Number.isFinite(minX) || !Number.isFinite(minY) || !Number.isFinite(maxX) || !Number.isFinite(maxY)) {
          minX = 0;
          minY = 0;
          maxX = 6000;
          maxY = 4000;
        }

        const bounds = { minX, minY, maxX, maxY };
        const spanX = Math.max(1, maxX - minX);
        const spanY = Math.max(1, maxY - minY);
        const centerXmm = (minX + maxX) / 2;
        const centerYmm = (minY + maxY) / 2;

        const maxDimM = Math.max(spanX, spanY) * MM_TO_M;
        if (!Number.isFinite(maxDimM) || !Number.isFinite(centerXmm) || !Number.isFinite(centerYmm)) {
          setError('Invalid floorplan bounds for 3D preview.');
          return;
        }
        const camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 500);
        camera.far = Math.max(500, maxDimM * 8);
        camera.updateProjectionMatrix();
        camera.position.set(0, Math.max(2.5, maxDimM * 0.9), Math.max(3.5, maxDimM * 1.2));
        controls = new OrbitControls(camera, renderer.domElement);
        controls.enabled = editModeRef.current === 'idle';
        controls.enableRotate = true;
        controls.enablePan = true;
        controls.enableZoom = true;
        controlsRef.current = controls;
        controls.target.set(0, 0.4, 0);
        controls.update();
        camera.lookAt(0, 0.4, 0);

        const floorColorHex =
          styledResult?.room?.surfaces?.floor_color_hex ??
          styledResult?.room?.surfaces?.floorColorHex ??
          '#f0ece4';

        const floorPts =
          roomPts.length >= 3
            ? roomPts
            : [
                { x: minX, y: minY },
                { x: maxX, y: minY },
                { x: maxX, y: maxY },
                { x: minX, y: maxY },
              ];
        if (floorPts.length >= 3) {
          const shapePts = floorPts.map(
            (p) => new THREE.Vector2((p.x - centerXmm) * MM_TO_M, (p.y - centerYmm) * MM_TO_M)
          );
          const shape = new THREE.Shape(shapePts);
          const geom = new THREE.ShapeGeometry(shape);
          geom.rotateX(-Math.PI / 2);
          const mat = new THREE.MeshStandardMaterial({ color: new THREE.Color(floorColorHex), side: THREE.DoubleSide });
          const floor = new THREE.Mesh(geom, mat);
          floor.receiveShadow = true;
          floor.castShadow = false;
          scene.add(floor);
        }

        const grid = new THREE.GridHelper(Math.max(spanX, spanY) * MM_TO_M * 1.2, 20, 0xd6d1c7, 0xd6d1c7);
        grid.position.y = 0.001;
        scene.add(grid);

        const wallHeightMm = Number(userInput.height ?? 2800);
        const wallHeight = wallHeightMm * MM_TO_M;
        const wallColorHex =
          styledResult?.room?.surfaces?.wall_color_hex ??
          styledResult?.room?.surfaces?.wallColorHex ??
          '#555555';

        const wallPts = roomPts.length >= 2 ? roomPts : floorPts;
        const wallsRaw = activeFloorplan.walls ?? [];
        const walls =
          wallsRaw.length > 0
            ? wallsRaw
            : wallPts.length >= 2
              ? wallPts.map((p, idx) => ({
                  p1_mm: p,
                  p2_mm: wallPts[(idx + 1) % wallPts.length],
                  thickness_mm: 100,
                }))
              : [];
        const openings =
          activeFloorplan.openings ??
          styledResult?.room?.openings ?? {
            doors: [],
            windows: [],
          };
        const resolveSegment = (item: any): { p1: any; p2: any } | null => {
          if (item?.start_mm && item?.end_mm) return { p1: item.start_mm, p2: item.end_mm };
          if (Array.isArray(item?.segment_mm) && item.segment_mm.length >= 2) {
            return { p1: item.segment_mm[0], p2: item.segment_mm[1] };
          }
          return null;
        };
        for (const wall of walls) {
          const x1 = Number(wall.p1_mm?.x ?? 0);
          const y1 = Number(wall.p1_mm?.y ?? 0);
          const x2 = Number(wall.p2_mm?.x ?? 0);
          const y2 = Number(wall.p2_mm?.y ?? 0);
          const dx = x2 - x1;
          const dy = y2 - y1;
          const lenMm = Math.hypot(dx, dy);
          if (lenMm <= 0) continue;
          const midX = (x1 + x2) / 2;
          const midY = (y1 + y2) / 2;
          const angle = Math.atan2(dy, dx);
          const thicknessMm = Number(wall.thickness_mm ?? 100);

          const geom = new THREE.BoxGeometry(lenMm * MM_TO_M, wallHeight, thicknessMm * MM_TO_M);
          const mat = new THREE.MeshStandardMaterial({ color: new THREE.Color(wallColorHex) });
          const mesh = new THREE.Mesh(geom, mat);
          mesh.castShadow = true;
          mesh.receiveShadow = true;
          // 2D plan uses +Y "up". Our 3D preview maps plan Y to -Z so 3D matches the 2D canvas orientation.
          mesh.position.set((midX - centerXmm) * MM_TO_M, wallHeight / 2, -(midY - centerYmm) * MM_TO_M);
          mesh.rotation.y = angle;
          scene.add(mesh);
        }

        type WallSeg = {
          x1: number;
          y1: number;
          x2: number;
          y2: number;
          dx: number;
          dy: number;
          len2: number;
          thicknessMm: number;
          angle: number;
        };

        const wallSegs: WallSeg[] = (walls ?? [])
          .map((wall) => {
            const x1 = Number(wall.p1_mm?.x ?? 0);
            const y1 = Number(wall.p1_mm?.y ?? 0);
            const x2 = Number(wall.p2_mm?.x ?? 0);
            const y2 = Number(wall.p2_mm?.y ?? 0);
            const dx = x2 - x1;
            const dy = y2 - y1;
            const len2 = dx * dx + dy * dy;
            return {
              x1,
              y1,
              x2,
              y2,
              dx,
              dy,
              len2,
              thicknessMm: Number(wall.thickness_mm ?? 100),
              angle: Math.atan2(dy, dx),
            };
          })
          .filter((w) => w.len2 > 1e-6);

        const findWallMount = (
          pxMm: number,
          pyMm: number
        ): { xMm: number; yMm: number; rotY: number } | null => {
          if (wallSegs.length === 0) return null;
          let best: { seg: WallSeg; t: number; projX: number; projY: number; dist2: number } | null = null;

          for (const seg of wallSegs) {
            const vx = pxMm - seg.x1;
            const vy = pyMm - seg.y1;
            const t = clamp((vx * seg.dx + vy * seg.dy) / seg.len2, 0, 1);
            const projX = seg.x1 + seg.dx * t;
            const projY = seg.y1 + seg.dy * t;
            const dx = pxMm - projX;
            const dy = pyMm - projY;
            const dist2 = dx * dx + dy * dy;
            if (!best || dist2 < best.dist2) {
              best = { seg, t, projX, projY, dist2 };
            }
          }

          if (!best) return null;
          const seg = best.seg;
          const len = Math.sqrt(seg.len2) || 1;
          const leftNx = -seg.dy / len;
          const leftNy = seg.dx / len;
          const rightNx = -leftNx;
          const rightNy = -leftNy;
          const offsetMm = seg.thicknessMm / 2 + 35;

          const leftX = best.projX + leftNx * offsetMm;
          const leftY = best.projY + leftNy * offsetMm;
          const rightX = best.projX + rightNx * offsetMm;
          const rightY = best.projY + rightNy * offsetMm;

          const dl2 = (leftX - centerXmm) ** 2 + (leftY - centerYmm) ** 2;
          const dr2 = (rightX - centerXmm) ** 2 + (rightY - centerYmm) ** 2;

          // Rotation convention: our proxy meshes face +Z as "front".
          // With planY -> -Z mapping, rotY = angle makes local +Z point to wall's RIGHT normal,
          // and rotY = angle + PI makes local +Z point to wall's LEFT normal.
          const useLeft = dl2 <= dr2;
          const rotY = useLeft ? seg.angle + Math.PI : seg.angle;
          return { xMm: useLeft ? leftX : rightX, yMm: useLeft ? leftY : rightY, rotY };
        };

        const findInteriorOffsetPoint = (
          pxMm: number,
          pyMm: number,
          extraMm: number
        ): { xMm: number; yMm: number; seg: WallSeg } | null => {
          if (wallSegs.length === 0) return null;
          let best: { seg: WallSeg; t: number; projX: number; projY: number; dist2: number } | null = null;

          for (const seg of wallSegs) {
            const vx = pxMm - seg.x1;
            const vy = pyMm - seg.y1;
            const t = clamp((vx * seg.dx + vy * seg.dy) / seg.len2, 0, 1);
            const projX = seg.x1 + seg.dx * t;
            const projY = seg.y1 + seg.dy * t;
            const dx = pxMm - projX;
            const dy = pyMm - projY;
            const dist2 = dx * dx + dy * dy;
            if (!best || dist2 < best.dist2) {
              best = { seg, t, projX, projY, dist2 };
            }
          }
          if (!best) return null;

          const seg = best.seg;
          const len = Math.sqrt(seg.len2) || 1;
          const leftNx = -seg.dy / len;
          const leftNy = seg.dx / len;
          const rightNx = -leftNx;
          const rightNy = -leftNy;
          const offsetMm = seg.thicknessMm / 2 + extraMm;

          const leftX = best.projX + leftNx * offsetMm;
          const leftY = best.projY + leftNy * offsetMm;
          const rightX = best.projX + rightNx * offsetMm;
          const rightY = best.projY + rightNy * offsetMm;

          const dl2 = (leftX - centerXmm) ** 2 + (leftY - centerYmm) ** 2;
          const dr2 = (rightX - centerXmm) ** 2 + (rightY - centerYmm) ** 2;
          const useLeft = dl2 <= dr2;
          return { xMm: useLeft ? leftX : rightX, yMm: useLeft ? leftY : rightY, seg };
        };

        const invByType = new Map<string, number>();
        const invById = new Map<string, number>();
        for (const item of inventory ?? []) {
          const height = Number(item.height_mm ?? 0);
          if (item?.asset_id && height > 0) {
            invById.set(item.asset_id, height);
          }
          if (item?.category && height > 0 && !invByType.has(item.category)) {
            invByType.set(item.category, height);
          }
        }

        const heightById = new Map<string, number>();
        for (const obj of objects) {
          const invHeight = invById.get(obj.id) ?? invByType.get(obj.type);
          const hMm = invHeight && invHeight > 0 ? invHeight : defaultHeightMm(obj.type);
          heightById.set(obj.id, hMm);
        }

        const objectGroup = new THREE.Group();
        scene.add(objectGroup);

        const openingsGroup = new THREE.Group();
        scene.add(openingsGroup);

        const rotatorGroup = new THREE.Group();
        scene.add(rotatorGroup);

        const openingColors = styledResult?.room?.opening_colors ?? styledResult?.room?.openingColors ?? null;
        const doorColorById = new Map<string, string>();
        const winColorById = new Map<string, string>();
        if (openingColors?.doors && Array.isArray(openingColors.doors)) {
          for (const d of openingColors.doors) {
            if (d?.id && typeof d.id === 'string' && d?.color_hex) {
              doorColorById.set(d.id, String(d.color_hex));
            }
          }
        }
        if (openingColors?.windows && Array.isArray(openingColors.windows)) {
          for (const w of openingColors.windows) {
            if (w?.id && typeof w.id === 'string' && w?.color_hex) {
              winColorById.set(w.id, String(w.color_hex));
            }
          }
        }

        // Doors (simple leaf proxy) + Windows (frame + glass proxy)
        for (const door of openings.doors ?? []) {
          const seg = resolveSegment(door);
          const p1 = seg?.p1 ?? null;
          const p2 = seg?.p2 ?? null;
          if (!p1 || !p2) continue;
          const x1 = Number(p1.x ?? 0);
          const y1 = Number(p1.y ?? 0);
          const x2 = Number(p2.x ?? 0);
          const y2 = Number(p2.y ?? 0);
          const dx = x2 - x1;
          const dy = y2 - y1;
          const lenMm = Math.hypot(dx, dy);
          if (!(lenMm > 1)) continue;

          const midX = (x1 + x2) / 2;
          const midY = (y1 + y2) / 2;
          const angle = Math.atan2(dy, dx);
          const doorId = String(door?.door_id ?? door?.id ?? 'door');
          const doorColor = doorColorById.get(doorId) ?? '#8c7a63';

          const thicknessM = 0.04;
          const extraMm = thicknessM * 0.5 / MM_TO_M + 6; // fully inside wall face + a small gap
          const anchor = findInteriorOffsetPoint(midX, midY, extraMm);
          const pxMm = anchor?.xMm ?? midX;
          const pyMm = anchor?.yMm ?? midY;

          const hM = Math.min(2.05, Math.max(1.8, wallHeight - 0.1));
          const wM = lenMm * MM_TO_M;

          const leaf = new THREE.Mesh(
            new THREE.BoxGeometry(wM, hM, thicknessM),
            new THREE.MeshStandardMaterial({
              color: new THREE.Color(doorColor),
              roughness: 0.65,
              metalness: 0.08,
            })
          );
          leaf.castShadow = true;
          leaf.receiveShadow = true;
          leaf.position.set((pxMm - centerXmm) * MM_TO_M, hM / 2, -(pyMm - centerYmm) * MM_TO_M);
          leaf.rotation.y = angle;
          openingsGroup.add(leaf);
        }

        for (const win of openings.windows ?? []) {
          const seg = resolveSegment(win);
          const p1 = seg?.p1 ?? null;
          const p2 = seg?.p2 ?? null;
          if (!p1 || !p2) continue;
          const x1 = Number(p1.x ?? 0);
          const y1 = Number(p1.y ?? 0);
          const x2 = Number(p2.x ?? 0);
          const y2 = Number(p2.y ?? 0);
          const dx = x2 - x1;
          const dy = y2 - y1;
          const lenMm = Math.hypot(dx, dy);
          if (!(lenMm > 1)) continue;

          const midX = (x1 + x2) / 2;
          const midY = (y1 + y2) / 2;
          const angle = Math.atan2(dy, dx);
          const winId = String(win?.window_id ?? win?.id ?? 'window');
          const frameColor = winColorById.get(winId) ?? '#c0c0c0';

          const sillMm = Number(win?.sill_height_mm ?? 900);
          const heightMm = Number(win?.height_mm ?? 1200);
          const bottomMm = clamp(sillMm, 0, wallHeightMm - 200);
          const maxHeightMm = Math.max(200, wallHeightMm - bottomMm - 120);
          const actualHeightMm = clamp(heightMm, 200, maxHeightMm);

          const frameDepthM = 0.08;
          const extraMm = frameDepthM * 0.5 / MM_TO_M + 6;
          const anchor = findInteriorOffsetPoint(midX, midY, extraMm);
          const pxMm = anchor?.xMm ?? midX;
          const pyMm = anchor?.yMm ?? midY;

          const group = new THREE.Group();
          group.position.set(
            (pxMm - centerXmm) * MM_TO_M,
            (bottomMm + actualHeightMm / 2) * MM_TO_M,
            -(pyMm - centerYmm) * MM_TO_M
          );
          group.rotation.y = angle;

          const frame = new THREE.Mesh(
            new THREE.BoxGeometry(lenMm * MM_TO_M, actualHeightMm * MM_TO_M, frameDepthM),
            new THREE.MeshStandardMaterial({
              color: new THREE.Color(frameColor),
              roughness: 0.35,
              metalness: 0.55,
            })
          );
          frame.castShadow = true;
          frame.receiveShadow = true;
          group.add(frame);

          const glass = new THREE.Mesh(
            new THREE.BoxGeometry(lenMm * MM_TO_M * 0.92, actualHeightMm * MM_TO_M * 0.9, 0.01),
            new THREE.MeshStandardMaterial({
              color: new THREE.Color('#9fc9e8'),
              roughness: 0.12,
              metalness: 0.02,
              transparent: true,
              opacity: 0.35,
            })
          );
          glass.position.z = frameDepthM / 2 - 0.006;
          glass.castShadow = false;
          glass.receiveShadow = false;
          group.add(glass);

          openingsGroup.add(group);
        }

        const draggableById = new Map<string, boolean>();
        for (const obj of objects) {
          draggableById.set(obj.id, !!obj.draggable);
        }

        const proxyById = new Map<string, any>();
        const rotatorById = new Map<string, any>();

        for (const obj of objects) {
          const wMmPlan = Math.max(1, obj.bbox.maxX - obj.bbox.minX);
          const dMmPlan = Math.max(1, obj.bbox.maxY - obj.bbox.minY);
          const cxMm = (obj.bbox.minX + obj.bbox.maxX) / 2;
          const cyMm = (obj.bbox.minY + obj.bbox.maxY) / 2;

          const method = String(obj.placeOn?.method ?? 'floor');
          const targetId = obj.placeOn?.targetId ?? null;
          const typeKey = obj.type.toLowerCase();
          const isCeiling =
            method === 'hang_on' &&
            (String(targetId ?? '').toLowerCase() === 'ceiling' ||
              typeKey.includes('ceiling_light') ||
              typeKey.includes('pendant_light') ||
              typeKey.includes('track_light'));
          const rotCcwDeg = ((Number(obj.rotationCcw ?? 0) % 360) + 360) % 360;

          // Our 2D outputs store a world-axis-aligned bbox for the already-rotated footprint.
          // If we rotate the proxy mesh by rotCcw again, we must use the *unrotated* footprint dims
          // (i.e., de-rotate bbox dims) so the world footprint stays correct.
          let wMm = wMmPlan;
          let dMm = dMmPlan;
          const isWallHung = method === 'hang_on' && !isCeiling;
          if (!isWallHung && (rotCcwDeg === 90 || rotCcwDeg === 270)) {
            const tmp = wMm;
            wMm = dMm;
            dMm = tmp;
          }

          // Default (floor) dimensions.
          const hMmDefault = heightById.get(obj.id) ?? defaultHeightMm(obj.type);
          let wM = wMm * MM_TO_M;
          let dM = dMm * MM_TO_M;
          let hM = hMmDefault * MM_TO_M;

          // Wall-hung items in a 2D plan typically use bbox as (width x height).
          if (isWallHung) {
            hM = Math.max(0.06, dMm * MM_TO_M);
            const thicknessM = clamp(hM * 0.08, 0.02, 0.06);
            dM = thicknessM;
          }

          let baseY = 0;
          if (method === 'on_top' && typeof targetId === 'string' && heightById.has(targetId)) {
            baseY = (heightById.get(targetId) ?? 0) * MM_TO_M + 0.004;
          } else if (method === 'hang_on') {
            if (isCeiling) {
              baseY = wallHeight - hM;
            } else {
              baseY = clamp(wallHeight * 0.6 - hM / 2, 0.6, wallHeight - hM);
            }
          }

          let pxMm = cxMm;
          let pyMm = cyMm;
          // Plan Y is mapped to -Z so 3D matches 2D canvas orientation.
          // Therefore 2D CCW rotation maps to 3D yaw with +angle, plus PI because our proxies face +Z by default.
          let rotY = planRotationToYawRad(rotCcwDeg);
          if (method === 'hang_on' && !isCeiling) {
            const mount = findWallMount(cxMm, cyMm);
            if (mount) {
              pxMm = mount.xMm;
              pyMm = mount.yMm;
              rotY = mount.rotY;
            }
          }

          const proxy = buildObjectProxy(THREE, {
            type: obj.type,
            wM,
            dM,
            hM,
            colorHex: obj.colorHex,
            material: obj.material ?? null,
          });
          proxy.position.set((pxMm - centerXmm) * MM_TO_M, baseY, -(pyMm - centerYmm) * MM_TO_M);
          proxy.rotation.y = rotY;
          proxy.userData = {
            objectId: obj.id,
            draggable: draggableById.get(obj.id) ?? false,
            baseY,
            method,
            wMm,
            dMm,
          };
          objectGroup.add(proxy);
          proxyById.set(obj.id, proxy);

          // Rotation handle (torus + knob) for draggable floor objects.
          if (draggableById.get(obj.id) && method !== 'hang_on') {
            const ringR = Math.max(0.12, Math.max(wM, dM) * 0.65);
            const ringTube = Math.max(0.008, ringR * 0.06);
            const ring = new THREE.Mesh(
              new THREE.TorusGeometry(ringR, ringTube, 12, 36),
              new THREE.MeshStandardMaterial({ color: new THREE.Color('#6b6b6b'), roughness: 0.5, metalness: 0.1 })
            );
            ring.rotation.x = Math.PI / 2;
            ring.position.set(0, Math.max(0.04, hM + 0.04), 0);
            const knob = new THREE.Mesh(
              new THREE.SphereGeometry(Math.max(0.01, ringTube * 1.2), 12, 12),
              new THREE.MeshStandardMaterial({ color: new THREE.Color('#2f2f2f'), roughness: 0.35, metalness: 0.2 })
            );
            knob.position.set(ringR, ring.position.y, 0);
            const rotator = new THREE.Group();
            rotator.add(ring);
            rotator.add(knob);
            rotator.position.copy(proxy.position);
            rotator.rotation.y = proxy.rotation.y;
            rotator.userData = { objectId: obj.id, rotator: true };
            rotator.visible = false;
            rotatorGroup.add(rotator);
            rotatorById.set(obj.id, rotator);
          }
        }

        const updateRotatorsVisibility = () => {
          const mode = editModeRef.current;
          const selected = selectedIdRef.current;
          for (const [oid, rotator] of rotatorById.entries()) {
            rotator.visible = mode === 'edit' && !!selected && oid === selected;
          }
        };
        updateRotatorsVisibilityRef.current = updateRotatorsVisibility;
        updateRotatorsVisibility();

        const raycaster = new THREE.Raycaster();
        const pointer = new THREE.Vector2();
        const gridMm = 50;
        let dragState:
          | {
              mesh: any;
              objectId: string;
              plane: any;
              offset: { x: number; z: number };
              lastValid: { x: number; z: number };
            }
          | null = null;
        let rotateState:
          | {
              mesh: any;
              objectId: string;
              plane: any;
              baseY: number;
            }
          | null = null;
        let orbitClickStart: { x: number; y: number } | null = null;
        let orbitMoved = false;

        const getPointer = (evt: PointerEvent) => {
          const rect = renderer.domElement.getBoundingClientRect();
          pointer.x = ((evt.clientX - rect.left) / rect.width) * 2 - 1;
          pointer.y = -((evt.clientY - rect.top) / rect.height) * 2 + 1;
        };

        const findDraggableRoot = (obj: any): any => {
          let cur = obj;
          while (cur) {
            if (cur.userData?.objectId) return cur;
            cur = cur.parent;
          }
          return null;
        };

        const handlePointerDown = (evt: PointerEvent) => {
          if (evt.button !== 0) return;
          if (!renderer) return;
          getPointer(evt);
          raycaster.setFromCamera(pointer, camera);

          const mode = editModeRef.current;
          if (mode === 'idle') {
            orbitClickStart = { x: evt.clientX, y: evt.clientY };
            orbitMoved = false;
            return;
          }

          if (mode === 'edit') {
            const rotIntersects = raycaster.intersectObjects(rotatorGroup.children, true);
            if (rotIntersects.length) {
              const hitRot = findDraggableRoot(rotIntersects[0].object);
              if (hitRot && hitRot.userData?.rotator) {
                const objectId = hitRot.userData.objectId;
                const target = proxyById.get(objectId);
                if (target) {
                  const baseY = Number(target.userData.baseY ?? 0);
                  const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -baseY);
                  rotateState = { mesh: target, objectId, plane, baseY };
                  setSelectedId(objectId);
                  if (controls) controls.enabled = false;
                  return;
                }
              }
            }
          }

          const intersects = raycaster.intersectObjects(objectGroup.children, true);
          if (!intersects.length) return;
          const hit = findDraggableRoot(intersects[0].object);
          if (mode !== 'edit') return;
          if (!hit || !hit.userData?.draggable) return;
          if (hit.userData?.objectId !== selectedIdRef.current) return;

          const method = String(hit.userData.method ?? 'floor');
          if (method === 'hang_on') return;

          const baseY = Number(hit.userData.baseY ?? 0);
          const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -baseY);
          const intersectPoint = new THREE.Vector3();
          raycaster.ray.intersectPlane(plane, intersectPoint);
          const offset = {
            x: hit.position.x - intersectPoint.x,
            z: hit.position.z - intersectPoint.z,
          };
          dragState = {
            mesh: hit,
            objectId: hit.userData.objectId,
            plane,
            offset,
            lastValid: { x: hit.position.x, z: hit.position.z },
          };
          if (controls) controls.enabled = false;
        };

        const handlePointerMove = (evt: PointerEvent) => {
          if (editModeRef.current === 'idle') {
            if (orbitClickStart) {
              const dx = evt.clientX - orbitClickStart.x;
              const dy = evt.clientY - orbitClickStart.y;
              if (Math.hypot(dx, dy) > 3) orbitMoved = true;
            }
            return;
          }
          if (rotateState) {
            getPointer(evt);
            raycaster.setFromCamera(pointer, camera);
            const intersectPoint = new THREE.Vector3();
            raycaster.ray.intersectPlane(rotateState.plane, intersectPoint);
            const dx = intersectPoint.x - rotateState.mesh.position.x;
            const dz = intersectPoint.z - rotateState.mesh.position.z;
            const angle = Math.atan2(dz, dx);
            rotateState.mesh.rotation.y = angle + Math.PI / 2;
            const rotator = rotatorById.get(rotateState.objectId);
            if (rotator) {
              rotator.rotation.y = rotateState.mesh.rotation.y;
              rotator.position.copy(rotateState.mesh.position);
            }
            return;
          }

          if (!dragState) return;
          getPointer(evt);
          raycaster.setFromCamera(pointer, camera);
          const intersectPoint = new THREE.Vector3();
          raycaster.ray.intersectPlane(dragState.plane, intersectPoint);
          let nextX = intersectPoint.x + dragState.offset.x;
          let nextZ = intersectPoint.z + dragState.offset.z;

          const worldXmm = nextX / MM_TO_M + centerXmm;
          const worldYmm = -(nextZ / MM_TO_M) + centerYmm;
          const placement = design?.placements?.find((p) => p.placement_id === dragState?.objectId);
          const wMm = Number(dragState.mesh.userData?.wMm ?? 0);
          const dMm = Number(dragState.mesh.userData?.dMm ?? 0);
          const halfW = Math.max(0, wMm / 2);
          const halfD = Math.max(0, dMm / 2);

          const snap = (valueMm: number) => Math.round(valueMm / gridMm) * gridMm;
          let snappedXmm = snap(worldXmm);
          let snappedYmm = snap(worldYmm);

          snappedXmm = clamp(snappedXmm, bounds.minX + halfW, bounds.maxX - halfW);
          snappedYmm = clamp(snappedYmm, bounds.minY + halfD, bounds.maxY - halfD);

          const inside = pointInPolygon({ x: snappedXmm, y: snappedYmm }, roomPts);
          if (inside) {
            const snappedX = (snappedXmm - centerXmm) * MM_TO_M;
            const snappedZ = -(snappedYmm - centerYmm) * MM_TO_M;
            nextX = snappedX;
            nextZ = snappedZ;
            dragState.lastValid = { x: nextX, z: nextZ };
          } else {
            nextX = dragState.lastValid.x;
            nextZ = dragState.lastValid.z;
          }

          dragState.mesh.position.x = nextX;
          dragState.mesh.position.z = nextZ;
          const rotator = rotatorById.get(dragState.objectId);
          if (rotator) {
            rotator.position.copy(dragState.mesh.position);
          }
        };

        const handlePointerUp = () => {
          const activeRotate = rotateState;
          const activeDrag = dragState;

          if (activeRotate) {
            const placement = design?.placements?.find(
              (p) => String(p.placement_id ?? p.id ?? p.object_id ?? '') === activeRotate?.objectId
            );
            if (placement) {
              const rot = yawRadToPlanRotationDeg(activeRotate.mesh.rotation.y);
              updatePlacement({ ...placement, rotation_deg: rot });
            }
          } else if (activeDrag) {
            const worldXmm = activeDrag.mesh.position.x / MM_TO_M + centerXmm;
            const worldYmm = -(activeDrag.mesh.position.z / MM_TO_M) + centerYmm;
            const placement = design?.placements?.find(
              (p) => String(p.placement_id ?? p.id ?? p.object_id ?? '') === activeDrag?.objectId
            );
            if (placement) {
              updatePlacement({ ...placement, position_mm: { x: worldXmm, y: worldYmm } });
            }
          } else if (editModeRef.current === 'idle') {
            if (orbitClickStart && !orbitMoved) {
              raycaster.setFromCamera(pointer, camera);
              const intersects = raycaster.intersectObjects(objectGroup.children, true);
              if (intersects.length) {
                const hit = findDraggableRoot(intersects[0].object);
                if (hit?.userData?.objectId) {
                  setSelectedId(hit.userData.objectId);
                }
              }
            }
          }

          rotateState = null;
          dragState = null;
          orbitClickStart = null;
          orbitMoved = false;
          if (controls) controls.enabled = editModeRef.current === 'idle';
        };

        renderer.domElement.addEventListener('pointerdown', handlePointerDown);
        renderer.domElement.addEventListener('pointermove', handlePointerMove);
        renderer.domElement.addEventListener('pointerup', handlePointerUp);
        renderer.domElement.addEventListener('pointerleave', handlePointerUp);
        window.addEventListener('pointerup', handlePointerUp);
        window.addEventListener('pointercancel', handlePointerUp);
        window.addEventListener('blur', handlePointerUp);

        const handleResize = () => {
          if (!renderer || disposed) return;
          const w = el.clientWidth;
          const h = el.clientHeight;
          if (w < 2 || h < 2) return;
          renderer.setSize(w, h);
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
        };
        resizeObserver = new ResizeObserver(handleResize);
        resizeObserver.observe(el);
        window.addEventListener('resize', handleResize);

        const animate = () => {
          if (disposed) return;
          raf = window.requestAnimationFrame(animate);
          controls?.update?.();
          renderer.render(scene, camera);
        };
        renderer.render(scene, camera);
        animate();

        return () => {
          window.removeEventListener('resize', handleResize);
          renderer.domElement.removeEventListener('pointerdown', handlePointerDown);
          renderer.domElement.removeEventListener('pointermove', handlePointerMove);
          renderer.domElement.removeEventListener('pointerup', handlePointerUp);
          renderer.domElement.removeEventListener('pointerleave', handlePointerUp);
          window.removeEventListener('pointerup', handlePointerUp);
          window.removeEventListener('pointercancel', handlePointerUp);
          window.removeEventListener('blur', handlePointerUp);
          if (updateRotatorsVisibilityRef.current === updateRotatorsVisibility) {
            updateRotatorsVisibilityRef.current = null;
          }
        };
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load Three.js');
      }
    };

    let cleanup: (() => void) | null = null;
    const start = () => {
      const w = el.clientWidth;
      const h = el.clientHeight;
      if (w < 2 || h < 2) return false;
      run(w, h)
        .then((fn) => {
          if (disposed) {
            if (typeof fn === 'function') {
              try {
                fn();
              } catch {
                // ignore
              }
            }
            return;
          }
          cleanup = typeof fn === 'function' ? fn : null;
        })
        .catch(() => {
          cleanup = null;
        });
      return true;
    };
    if (!start()) {
      resizeObserver = new ResizeObserver(() => {
        if (disposed || renderer) return;
        if (start() && resizeObserver) {
          resizeObserver.disconnect();
          resizeObserver = null;
        }
      });
      resizeObserver.observe(el);
    }

    return () => {
      disposed = true;
      window.cancelAnimationFrame(raf);
      if (resizeObserver) {
        resizeObserver.disconnect();
        resizeObserver = null;
      }
      try {
        cleanup?.();
      } catch {
        // ignore
      }
      try {
        controls?.dispose?.();
      } catch {
        // ignore
      }
      controlsRef.current = null;
      try {
        renderer?.dispose?.();
      } catch {
        // ignore
      }
      try {
        if (renderer?.domElement && el.contains(renderer.domElement)) {
          el.removeChild(renderer.domElement);
        }
      } catch {
        // ignore
      }
      if (import.meta.env.DEV) {
        console.debug('[3D] cleanup');
      }
    };
  }, [viewMode, activeFloorplan, objects, inventory, styledResult, userInput.height, design, updatePlacement]);

  if (viewMode !== '3d') return null;

  return (
    <div className="three-panel">
      <div className="three-panel-header">
        <div className="three-panel-title">3D Preview</div>
        <div className="three-panel-actions">
          <button
            className="ghost"
            onClick={() => {
              if (!initialDesignRef.current) return;
              setDesign(JSON.parse(JSON.stringify(initialDesignRef.current)));
            }}
          >
            Reset
          </button>
          <button
            className="danger"
            disabled={!selectedId}
            onClick={() => {
              if (!selectedId) return;
              const placement = design?.placements?.find(
                (p) => String(p.placement_id ?? p.id ?? p.object_id ?? '') === selectedId
              );
              if (placement) {
                removePlacement(placement.placement_id);
              }
              setSelectedId(null);
            }}
          >
            Delete
          </button>
          <button
            className={editMode === 'edit' ? 'active' : ''}
            disabled={!selectedId || editMode !== 'idle'}
            onClick={() => setEditMode('edit')}
          >
            Edit
          </button>
          {editMode === 'edit' ? <button onClick={() => setEditMode('idle')}>Done</button> : null}
          {selectedId ? <div className="three-panel-hint">Selected: {selectedId}</div> : null}
          <div className="three-panel-hint">
            Click-drag to orbit. Click an object to select. Press Edit to lock and adjust. Done to unlock.
          </div>
        </div>
      </div>
      <div className="three-panel-body">
        <div ref={containerRef} className="three-panel-canvas" />
        {error ? (
          <div className="three-preview-overlay">
            <div className="muted">{error}</div>
          </div>
        ) : null}
        {!error && objects.length === 0 ? (
          <div className="three-preview-overlay">
            <div className="muted">No objects to render (only room shell).</div>
          </div>
        ) : null}
      </div>
    </div>
  );
};

function pointInPolygon(point: { x: number; y: number }, polygon: Array<{ x: number; y: number }>): boolean {
  if (!polygon || polygon.length < 3) return true;
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x;
    const yi = polygon[i].y;
    const xj = polygon[j].x;
    const yj = polygon[j].y;
    const intersect = yi > point.y !== yj > point.y && point.x < ((xj - xi) * (point.y - yi)) / (yj - yi + 1e-9) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}
