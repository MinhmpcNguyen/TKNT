import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useEditorStore } from '../store';
import { polygonBounds } from '../utils/geometry';
import { getNumericBbox, getWorldBboxFromPlacement, placementMatchesStyledObject } from '../utils/styledLayout';
import { planRotationToYawRad } from '../utils/planOrientation';
import {
  buildGlbPreviewModel,
  disposeThreeObject,
} from '../three/models';
import {
  collectReintroducedObjects,
  getInventoryDimensionsMm,
  isReintroducedIdentifier,
  isUserGeneratedPreviewItem,
  resolveInventoryItemForPreview,
  resolveWallHungPreviewDepthMm,
  type InventoryPreviewResolution,
} from '../three/inventoryDiagnostics';
import { buildObjectProxy } from '../three/proxies';

type RenderObject = {
  id: string;
  assetId?: string | null;
  rawAssetId?: string | null;
  type: string;
  colorHex: string;
  material?: string | null;
  rotationCcw?: number;
  bbox: { minX: number; minY: number; maxX: number; maxY: number };
  placeOn?: { method: string; targetId?: string | null } | null;
  reintroduced?: boolean;
};

type ResolvedRenderObject = RenderObject & {
  inventoryResolution: InventoryPreviewResolution;
};

type ObjectVisualDiagnostic = {
  objectId: string;
  label: string;
  rawAssetId: string | null;
  resolvedAssetId: string | null;
  reintroduced: boolean;
  status:
    | 'glb_loading'
    | 'glb_loaded'
    | 'proxy_no_inventory_match'
    | 'proxy_no_model_descriptor'
    | 'proxy_low_confidence_match'
    | 'proxy_no_loader'
    | 'proxy_model_load_error';
  reason: string;
};

const MM_TO_M = 0.001;
const REFILL_ASSET_ID_PATTERN = /__(?:reintroduced|accessory_refill)(?:_\d+)?$/i;

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

function normalizeRenderAssetId(value: unknown): string | null {
  if (typeof value !== 'string') {
    return null;
  }
  const raw = value.trim();
  if (!raw) {
    return null;
  }
  const normalized = raw.replace(REFILL_ASSET_ID_PATTERN, '').trim();
  return normalized || raw;
}

export const ThreeDPreviewModal: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const {
    threeDOpen,
    setThreeDOpen,
    floorplan,
    design,
    styledResult,
    inventory,
    userInput,
  } = useEditorStore();
  const [error, setError] = useState<string | null>(null);
  const [objectVisualDiagnostics, setObjectVisualDiagnostics] = useState<
    Record<string, ObjectVisualDiagnostic>
  >({});
  const activeFloorplan = floorplan ?? design?.floorplan_geometry ?? null;

  const objects: RenderObject[] = useMemo(() => {
    const out: RenderObject[] = [];
    const placements = design?.placements ?? [];
    const placementsById = new Map<string, any>();
    for (const placement of placements) {
      const placementId = String(placement.placement_id ?? placement.id ?? placement.object_id ?? '');
      if (!placementId) continue;
      placementsById.set(placementId, placement);
    }
    const rawObjects = styledResult?.objects;
    if (Array.isArray(rawObjects)) {
      const used = new Set<string>();
      for (const obj of rawObjects) {
        const id = String(obj.instance_id ?? obj.id ?? obj.object_type ?? `obj_${out.length + 1}`);
        const styledBbox = getNumericBbox(obj);
        let bbox = styledBbox;
        let rotationCcw = Number(obj.rotation_ccw ?? 0);
        const placement = placementsById.get(id);
        if (placement) {
          used.add(id);
          if (!placementMatchesStyledObject(placement, styledBbox, rotationCcw)) {
            bbox = getWorldBboxFromPlacement(placement);
            rotationCcw = Number(placement.rotation_deg ?? rotationCcw ?? 0);
          }
        }
        const rawAssetId =
          placement?.asset_id ??
          obj.asset_id ??
          obj.assetId ??
          obj.object_id ??
          null;
        out.push({
          id,
          assetId: normalizeRenderAssetId(rawAssetId) ?? null,
          rawAssetId: typeof rawAssetId === 'string' ? rawAssetId : null,
          type: String(obj.object_type ?? obj.category ?? 'item'),
          colorHex: String(obj.color_hex ?? '#cfcac2'),
          material: obj.material ?? null,
          rotationCcw,
          bbox,
          placeOn: obj.place_on
            ? {
              method: String(obj.place_on.method ?? 'floor'),
              targetId: obj.place_on.target_instance_id ?? null,
            }
            : null,
          reintroduced:
            isReintroducedIdentifier(obj.object_id) ||
            isReintroducedIdentifier(rawAssetId) ||
            isReintroducedIdentifier(id),
        });
      }
      for (const placement of placements) {
        const placementId = String(placement.placement_id ?? placement.id ?? placement.object_id ?? '');
        if (!placementId || used.has(placementId)) continue;
        out.push({
          id: placementId,
          assetId: normalizeRenderAssetId(placement.asset_id ?? null) ?? null,
          rawAssetId:
            typeof placement.asset_id === 'string' ? placement.asset_id : null,
          type: placement.category,
          colorHex: placement.color_hex ?? '#cfcac2',
          material: null,
          rotationCcw: Number(placement.rotation_deg ?? 0),
          bbox: getWorldBboxFromPlacement(placement),
          placeOn: { method: 'floor', targetId: 'floor' },
          reintroduced:
            isReintroducedIdentifier(placement.asset_id) ||
            isReintroducedIdentifier(placement.placement_id),
        });
      }
      return out;
    }

    if (!design) return out;
    for (const placement of placements) {
      out.push({
        id: placement.placement_id,
        assetId: normalizeRenderAssetId(placement.asset_id ?? null) ?? null,
        rawAssetId:
          typeof placement.asset_id === 'string' ? placement.asset_id : null,
        type: placement.category,
        colorHex: placement.color_hex ?? '#cfcac2',
        material: null,
        rotationCcw: Number(placement.rotation_deg ?? 0),
        bbox: getWorldBboxFromPlacement(placement),
        placeOn: { method: 'floor', targetId: 'floor' },
        reintroduced:
          isReintroducedIdentifier(placement.asset_id) ||
          isReintroducedIdentifier(placement.placement_id),
      });
    }
    return out;
  }, [styledResult, design]);

  const resolvedObjects: ResolvedRenderObject[] = useMemo(
    () =>
      objects.map((obj) => {
        const targetWidthMm = Math.max(1, obj.bbox.maxX - obj.bbox.minX);
        const targetDepthMm = Math.max(1, obj.bbox.maxY - obj.bbox.minY);
        const inventoryResolution = resolveInventoryItemForPreview({
          inventory,
          assetId: obj.assetId ?? obj.rawAssetId ?? null,
          objectType: obj.type,
          material: obj.material ?? null,
          preferredStyle: userInput.style,
          targetWidthMm,
          targetDepthMm,
          targetHeightMm: defaultHeightMm(obj.type),
        });
        return {
          ...obj,
          inventoryResolution,
        };
      }),
    [inventory, objects, userInput.style]
  );

  const reintroducedObjects = useMemo(() => {
    const fromSummary = collectReintroducedObjects(styledResult);
    if (fromSummary.length > 0) {
      return fromSummary;
    }
    return resolvedObjects
      .filter((obj) => obj.reintroduced)
      .map((obj) => ({
        clusterId: null,
        objectId: obj.id,
        objectType: obj.type,
      }));
  }, [resolvedObjects, styledResult]);

  const proxyObjects = useMemo(
    () =>
      Object.values(objectVisualDiagnostics)
        .filter((item) => item.status !== 'glb_loaded' && item.status !== 'glb_loading')
        .sort((left, right) => left.label.localeCompare(right.label)),
    [objectVisualDiagnostics]
  );

  useEffect(() => {
    const nextDiagnostics = Object.fromEntries(
      resolvedObjects.map((obj) => {
        const modelResolution = obj.inventoryResolution.modelResolution;
        let status: ObjectVisualDiagnostic['status'] = 'proxy_no_inventory_match';
        let reason = obj.inventoryResolution.reason;
        if (obj.inventoryResolution.item) {
          if (obj.inventoryResolution.glbEligible && modelResolution?.descriptor) {
            status = 'glb_loading';
            reason = obj.inventoryResolution.glbReason;
          } else if (modelResolution?.descriptor) {
            status = 'proxy_low_confidence_match';
            reason = obj.inventoryResolution.glbReason;
          } else {
            status = 'proxy_no_model_descriptor';
            reason = modelResolution?.reason ?? 'Matched inventory asset has no GLB metadata.';
          }
        }
        return [
          obj.id,
          {
            objectId: obj.id,
            label: obj.type,
            rawAssetId: obj.rawAssetId ?? obj.assetId ?? null,
            resolvedAssetId: obj.inventoryResolution.item?.asset_id ?? null,
            reintroduced: Boolean(obj.reintroduced),
            status,
            reason,
          } satisfies ObjectVisualDiagnostic,
        ];
      })
    );
    setObjectVisualDiagnostics(nextDiagnostics);
  }, [resolvedObjects]);

  useEffect(() => {
    if (!threeDOpen) return;
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
        const GLTF_LOADER_URL = new URL(
          '../vendor/three/loaders/GLTFLoader.js',
          import.meta.url
        ).toString();

        const THREE: any = await import(/* @vite-ignore */ THREE_URL);
        const controlsMod: any = await import(/* @vite-ignore */ ORBIT_URL);
        const OrbitControls = controlsMod.OrbitControls;
        let GLTFLoader: any = null;
        try {
          const gltfLoaderMod: any = await import(/* @vite-ignore */ GLTF_LOADER_URL);
          GLTFLoader = gltfLoaderMod.GLTFLoader ?? null;
        } catch {
          GLTFLoader = null;
        }
        if (!GLTFLoader) {
          setObjectVisualDiagnostics((current) => {
            const next = { ...current };
            for (const obj of resolvedObjects) {
              if (next[obj.id]?.status !== 'glb_loading') {
                continue;
              }
              next[obj.id] = {
                ...next[obj.id],
                status: 'proxy_no_loader',
                reason: 'GLTFLoader is unavailable in the current frontend bundle.',
              };
            }
            return next;
          });
        }

        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setPixelRatio(window.devicePixelRatio || 1);
        renderer.setSize(width, height);
        renderer.setClearColor(0xf6f4ef, 1);
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
        const roomBounds = roomPts.length >= 3 ? polygonBounds(roomPts) : null;
        let minX = roomBounds?.minX ?? 0;
        let minY = roomBounds?.minY ?? 0;
        let maxX = roomBounds?.maxX ?? 6000;
        let maxY = roomBounds?.maxY ?? 4000;
        if (!roomBounds && resolvedObjects.length) {
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

        for (const obj of resolvedObjects) {
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

        const heightById = new Map<string, number>();
        for (const obj of resolvedObjects) {
          const inventoryDimensions = getInventoryDimensionsMm(obj.inventoryResolution.item);
          const hMm = inventoryDimensions?.height ?? defaultHeightMm(obj.type);
          heightById.set(obj.id, hMm);
        }

        const objectGroup = new THREE.Group();
        scene.add(objectGroup);

        const openingsGroup = new THREE.Group();
        scene.add(openingsGroup);

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

        for (const obj of resolvedObjects) {
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
          const inventoryDimensions = getInventoryDimensionsMm(obj.inventoryResolution.item);
          const isWallHung = method === 'hang_on' && !isCeiling;
          let wMm = wMmPlan;
          let dMm = dMmPlan;
          if (!isWallHung && (rotCcwDeg === 90 || rotCcwDeg === 270)) {
            const tmp = wMm;
            wMm = dMm;
            dMm = tmp;
          }

          // Keep GLB and proxy visuals aligned to the styled-layout footprint so 3D
          // stays visually consistent with the 2D plan. Inventory dimensions are still
          // useful for height defaults and wall-hung thickness.
          const hMmDefault = heightById.get(obj.id) ?? defaultHeightMm(obj.type);
          let wM = wMm * MM_TO_M;
          let dM = dMm * MM_TO_M;
          let hM = hMmDefault * MM_TO_M;

          // Wall-hung items in a 2D plan typically use bbox as (width x height).
          if (isWallHung) {
            wM = Math.max(0.06, wMmPlan * MM_TO_M);
            hM = Math.max(0.06, dMmPlan * MM_TO_M);
            const thicknessMm = resolveWallHungPreviewDepthMm({
              objectType: obj.type,
              planWidthMm: wMmPlan,
              planHeightMm: dMmPlan,
              inventoryDimensions,
            });
            dM = Math.max(0.02, thicknessMm * MM_TO_M);
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
          // Plan Y is mapped to -Z so 3D matches the 2D canvas orientation.
          // Our proxies and GLBs both use the same shared plan-rotation mapping.
          let rotY = planRotationToYawRad(rotCcwDeg);
          if (method === 'hang_on' && !isCeiling) {
            const mount = findWallMount(cxMm, cyMm);
            if (mount) {
              pxMm = mount.xMm;
              pyMm = mount.yMm;
              rotY = mount.rotY;
            }
          }

          const objectRoot = new THREE.Group();
          objectRoot.position.set(
            (pxMm - centerXmm) * MM_TO_M,
            baseY,
            -(pyMm - centerYmm) * MM_TO_M
          );
          objectRoot.rotation.y = rotY;

          const fallbackProxy = buildObjectProxy(THREE, {
            type: obj.type,
            wM,
            dM,
            hM,
            colorHex: obj.colorHex,
            material: obj.material ?? null,
          });
          objectRoot.add(fallbackProxy);
          objectGroup.add(objectRoot);

          const modelDescriptor = obj.inventoryResolution.glbEligible
            ? obj.inventoryResolution.modelResolution?.descriptor ?? null
            : null;
          if (GLTFLoader && modelDescriptor) {
            const preserveGeneratedMeshShape = isUserGeneratedPreviewItem(
              obj.inventoryResolution.item
            );
            void buildGlbPreviewModel({
                THREE,
                GLTFLoader,
                descriptor: modelDescriptor,
                wM,
                dM,
                hM,
                fitStrategy: isWallHung
                  ? preserveGeneratedMeshShape
                    ? 'wall-plane-uniform'
                    : 'wall-plane'
                  : preserveGeneratedMeshShape
                    ? 'footprint-uniform'
                    : 'footprint',
                colorHex: obj.colorHex,
                preserveMaterialTextures: preserveGeneratedMeshShape,
                minimumHorizontalAspectRatio: preserveGeneratedMeshShape ? 0.45 : null,
              })
              .then((model) => {
                if (disposed || !objectRoot.parent) {
                  disposeThreeObject(model);
                  return;
                }
                setObjectVisualDiagnostics((current) => ({
                  ...current,
                  [obj.id]: {
                    ...(current[obj.id] ?? {
                      objectId: obj.id,
                      label: obj.type,
                      rawAssetId: obj.rawAssetId ?? obj.assetId ?? null,
                      resolvedAssetId: obj.inventoryResolution.item?.asset_id ?? null,
                      reintroduced: Boolean(obj.reintroduced),
                    }),
                    status: 'glb_loaded',
                    reason: `Loaded GLB for inventory asset "${obj.inventoryResolution.item?.asset_id ?? obj.assetId ?? obj.type}".`,
                  },
                }));
                const previousChildren = [...objectRoot.children];
                for (const child of previousChildren) {
                  objectRoot.remove(child);
                }
                for (const child of previousChildren) {
                  disposeThreeObject(child);
                }
                objectRoot.add(model);
              })
              .catch((modelError: unknown) => {
                setObjectVisualDiagnostics((current) => ({
                  ...current,
                  [obj.id]: {
                    ...(current[obj.id] ?? {
                      objectId: obj.id,
                      label: obj.type,
                      rawAssetId: obj.rawAssetId ?? obj.assetId ?? null,
                      resolvedAssetId: obj.inventoryResolution.item?.asset_id ?? null,
                      reintroduced: Boolean(obj.reintroduced),
                    }),
                    status: 'proxy_model_load_error',
                    reason:
                      modelError instanceof Error
                        ? modelError.message
                        : 'Failed to load GLB, so the object stayed on the proxy mesh.',
                  },
                }));
                // Keep the procedural proxy as fallback when model loading fails.
              });
          }
        }

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
    };
  }, [threeDOpen, activeFloorplan, resolvedObjects, styledResult, userInput.height, design]);

  if (!threeDOpen) return null;

  return (
    <div className="modal-overlay" role="dialog" aria-modal="true">
      <div className="modal-card">
        <div className="modal-header">
          <div className="modal-title">3D Preview</div>
          <button onClick={() => setThreeDOpen(false)}>Close</button>
        </div>
        <div className="modal-body">
          <div ref={containerRef} className="three-canvas" />
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
          {reintroducedObjects.length > 0 || proxyObjects.length > 0 ? (
            <div className="three-debug-overlay">
              {reintroducedObjects.length > 0 ? (
                <div className="three-debug-card compact">
                  <div className="three-debug-title">
                    Reintroduced ({reintroducedObjects.length})
                  </div>
                  <div className="three-debug-list">
                    {reintroducedObjects.map((item) => (
                      <div
                        key={`${item.clusterId ?? 'no-cluster'}-${item.objectId}`}
                        className="three-debug-item"
                      >
                        <strong>{item.objectType}</strong>
                        <span>{item.objectId}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}
              {proxyObjects.length > 0 ? (
                <div className="three-debug-card compact">
                  <div className="three-debug-title">Proxy Fallbacks ({proxyObjects.length})</div>
                  <div className="three-debug-list">
                    {proxyObjects.slice(0, 8).map((item) => (
                      <div key={item.objectId} className="three-debug-item">
                        <strong>{item.label}</strong>
                        <span>
                          raw: {item.rawAssetId ?? 'n/a'} | inventory:{' '}
                          {item.resolvedAssetId ?? 'not matched'}
                        </span>
                        <span>{item.reason}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
};
