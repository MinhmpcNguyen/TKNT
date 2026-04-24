import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  fetchRenderPresets,
  renderSnapshotImage,
  type SnapshotImageEditOperation,
  type SnapshotImagePresetOption,
  type SnapshotImagePresetSelection,
  type SnapshotImagePresets,
  type SnapshotImageRenderApiResponse,
  type SnapshotImageRenderResult,
} from '../api';
import { useEditorStore } from '../store';
import { polygonBounds } from '../utils/geometry';
import { getNumericBbox, getWorldBboxFromPlacement, placementMatchesStyledObject } from '../utils/styledLayout';
import {
  buildThreeSnapshotArtifact,
  type SnapshotOpeningRecord,
  type SnapshotObjectRecord,
  type ThreeSnapshotArtifact,
  type ThreeSnapshotRenderState,
  type VisibleSnapshotObject,
} from '../utils/threeSnapshot';
import { buildLayoutReferenceImageDataUrl } from '../utils/layoutReferenceImage';
import {
  resolvePlanObjectYawRad,
  yawRadToPlanRotationDeg,
} from '../utils/planOrientation';
import {
  applyInventoryModelPreviewOverride,
  buildGlbPreviewModel,
  disposeThreeObject,
} from '../three/models';
import {
  collectReintroducedObjects,
  getInventoryDimensionsMm,
  isReintroducedIdentifier,
  isShelfLikePreviewType,
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
  frontWorld?: unknown;
  frontSideWorld?: unknown;
  frontSideLocal?: unknown;
  bbox: { minX: number; minY: number; maxX: number; maxY: number };
  placeOn?: { method: string; targetId?: string | null } | null;
  draggable?: boolean;
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
  modelUrl?: string | null;
  previewRotationOffsetDeg?: number | null;
  finalDescriptorRotationDeg?: number | null;
  status:
    | 'glb_loading'
    | 'glb_loaded'
    | 'proxy_no_inventory_match'
    | 'proxy_no_model_descriptor'
    | 'proxy_low_confidence_match'
    | 'proxy_aspect_mismatch'
    | 'proxy_no_loader'
    | 'proxy_model_load_error';
  reason: string;
};

type RenderPresetGroupName = 'styles' | 'lights' | 'sceneries';

const MM_TO_M = 0.001;
const SNAPSHOT_WALL_OCCLUSION_OPACITY_THRESHOLD = 0.5;
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

function sanitizeSnapshotFileSegment(value: string | null | undefined): string {
  const normalized = String(value ?? '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return normalized || 'current-layout';
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

function buildSnapshotFileStem(variantId: string | null, capturedAt: string): string {
  const normalizedDate = capturedAt.replace(/[:.]/g, '-');
  return `layout-3d-${sanitizeSnapshotFileSegment(variantId)}-${normalizedDate}`;
}

function resolveOpeningSegmentMm(
  item: any
): { p1: { x: number; y: number }; p2: { x: number; y: number } } | null {
  if (item?.start_mm && item?.end_mm) {
    return {
      p1: { x: Number(item.start_mm.x ?? 0), y: Number(item.start_mm.y ?? 0) },
      p2: { x: Number(item.end_mm.x ?? 0), y: Number(item.end_mm.y ?? 0) },
    };
  }
  if (Array.isArray(item?.segment_mm) && item.segment_mm.length >= 2) {
    return {
      p1: {
        x: Number(item.segment_mm[0]?.x ?? 0),
        y: Number(item.segment_mm[0]?.y ?? 0),
      },
      p2: {
        x: Number(item.segment_mm[1]?.x ?? 0),
        y: Number(item.segment_mm[1]?.y ?? 0),
      },
    };
  }
  if (Array.isArray(item?.segmentMm) && item.segmentMm.length >= 2) {
    return {
      p1: {
        x: Number(item.segmentMm[0]?.x ?? 0),
        y: Number(item.segmentMm[0]?.y ?? 0),
      },
      p2: {
        x: Number(item.segmentMm[1]?.x ?? 0),
        y: Number(item.segmentMm[1]?.y ?? 0),
      },
    };
  }
  return null;
}

function normalizeSnapshotOpenings(items: any[]): SnapshotOpeningRecord[] {
  return items
    .map((item) => {
      const segment = resolveOpeningSegmentMm(item);
      if (!segment) return null;
      return {
        id: String(item?.id ?? 'opening'),
        segmentMm: [
          { x: segment.p1.x, y: segment.p1.y },
          { x: segment.p2.x, y: segment.p2.y },
        ],
      };
    })
    .filter((item): item is SnapshotOpeningRecord => item !== null);
}

function downloadBlob(blob: Blob, filename: string): void {
  const url = window.URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  window.setTimeout(() => {
    window.URL.revokeObjectURL(url);
  }, 0);
}

function canvasToPngBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) {
        resolve(blob);
        return;
      }
      reject(new Error('Unable to encode the 3D canvas as PNG.'));
    }, 'image/png');
  });
}

function canvasToDataUrl(canvas: HTMLCanvasElement): string {
  return canvas.toDataURL('image/png');
}

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        resolve(reader.result);
        return;
      }
      reject(new Error('Unable to read the reference image.'));
    };
    reader.onerror = () => reject(new Error('Unable to read the reference image.'));
    reader.readAsDataURL(file);
  });
}

function firstPresetKey(
  presets: SnapshotImagePresets | null,
  groupName: RenderPresetGroupName,
  preferred: string
): string {
  const group = presets?.[groupName] ?? {};
  if (preferred && group[preferred]) {
    return preferred;
  }
  return Object.keys(group)[0] ?? '';
}

function optionLabel(rawKey: string): string {
  return rawKey
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function visibleObjectName(object: VisibleSnapshotObject | null): string {
  if (!object) {
    return 'object';
  }
  return object.type.replace(/_/g, ' ');
}

function visibleObjectLabel(object: VisibleSnapshotObject): string {
  return `${visibleObjectName(object)} - ${object.id}`;
}

function hasPresetSelection(selection: SnapshotImagePresetSelection): boolean {
  return Boolean(selection.style || selection.lighting || selection.scenery);
}

type RenderPresetSelectorProps = {
  title: string;
  group: Record<string, SnapshotImagePresetOption>;
  value: string | null | undefined;
  disabled: boolean;
  onChange: (value: string | null) => void;
  allowEmpty?: boolean;
  emptyLabel?: string;
  compact?: boolean;
};

function RenderPresetSelector({
  title,
  group,
  value,
  disabled,
  onChange,
  allowEmpty = false,
  emptyLabel = 'No change',
  compact = false,
}: RenderPresetSelectorProps): React.ReactElement {
  const entries = Object.entries(group);
  const className = compact
    ? 'reference-preset-group compact'
    : 'reference-preset-group';

  return (
    <fieldset className={className}>
      <legend>{title}</legend>
      <div className="reference-preset-options">
        {allowEmpty ? (
          <button
            type="button"
            className={!value ? 'reference-preset-card selected' : 'reference-preset-card'}
            disabled={disabled}
            aria-pressed={!value}
            onClick={() => onChange(null)}
          >
            <span className="reference-preset-image is-empty">
              <span>{emptyLabel}</span>
            </span>
            <span className="reference-preset-label">{emptyLabel}</span>
          </button>
        ) : null}
        {entries.length === 0 && !allowEmpty ? (
          <div className="reference-preset-card disabled">
            <span className="reference-preset-image is-empty">
              <span>Loading</span>
            </span>
            <span className="reference-preset-label">{title}</span>
          </div>
        ) : null}
        {entries.map(([key, option]) => {
          const label = option.label || optionLabel(key);
          return (
            <button
              key={key}
              type="button"
              className={
                value === key ? 'reference-preset-card selected' : 'reference-preset-card'
              }
              disabled={disabled}
              aria-pressed={value === key}
              onClick={() => onChange(key)}
            >
              {option.reference_image ? (
                <img
                  className="reference-preset-image"
                  src={option.reference_image}
                  alt={`${label} reference`}
                  loading="lazy"
                  draggable={false}
                />
              ) : (
                <span className="reference-preset-image is-empty">
                  <span>No image</span>
                </span>
              )}
              <span className="reference-preset-label">{label}</span>
            </button>
          );
        })}
      </div>
    </fieldset>
  );
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
    selectedVariantId,
    selectedWallId,
    selectWall,
    updatePlacement,
    removePlacement,
    setDesign,
    upsertSavedRender,
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
  const rendererRef = useRef<any>(null);
  const sceneRef = useRef<any>(null);
  const cameraRef = useRef<any>(null);
  const rotatorGroupRef = useRef<any>(null);
  const renderStateRef = useRef<ThreeSnapshotRenderState | null>(null);
  const [captureStatus, setCaptureStatus] = useState<{
    tone: 'neutral' | 'success' | 'error';
    text: string;
  } | null>(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const [renderStatus, setRenderStatus] = useState<{
    tone: 'neutral' | 'success' | 'error';
    text: string;
  } | null>(null);
  const [isRenderingImage, setIsRenderingImage] = useState(false);
  const [renderedSnapshotImage, setRenderedSnapshotImage] =
    useState<SnapshotImageRenderResult | null>(null);
  const [savedRenderId, setSavedRenderId] = useState<string | null>(null);
  const [renderPrompt, setRenderPrompt] = useState('');
  const [renderPresets, setRenderPresets] =
    useState<SnapshotImagePresets | null>(null);
  const [presetSelection, setPresetSelection] =
    useState<SnapshotImagePresetSelection>({
      style: 'modern',
      lighting: 'daylight_1',
      scenery: 'none',
    });
  const [lastRenderedSnapshot, setLastRenderedSnapshot] =
    useState<ThreeSnapshotArtifact | null>(null);
  const [lastRenderedSnapshotImageDataUrl, setLastRenderedSnapshotImageDataUrl] =
    useState<string | null>(null);
  const [editSelectedObjectId, setEditSelectedObjectId] = useState<string>('');
  const [replacementFile, setReplacementFile] = useState<File | null>(null);
  const [targetColor, setTargetColor] = useState('#c8a26a');
  const [replaceEnabled, setReplaceEnabled] = useState(false);
  const [colorEnabled, setColorEnabled] = useState(false);
  const [editPresetSelection, setEditPresetSelection] =
    useState<SnapshotImagePresetSelection>({
      style: null,
      lighting: null,
      scenery: null,
    });
  const [objectVisualDiagnostics, setObjectVisualDiagnostics] = useState<
    Record<string, ObjectVisualDiagnostic>
  >({});

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

  useEffect(() => {
    if (!renderStateRef.current) return;
    renderStateRef.current.variantId = selectedVariantId;
    renderStateRef.current.selectedObjectId = selectedId;
    renderStateRef.current.selectedWallId = selectedWallId;
    renderStateRef.current.editMode = editMode;
  }, [editMode, selectedId, selectedVariantId, selectedWallId]);

  useEffect(() => {
    let cancelled = false;
    fetchRenderPresets()
      .then((presets) => {
        if (cancelled) return;
        setRenderPresets(presets);
        setPresetSelection((current) => ({
          style: firstPresetKey(presets, 'styles', current.style ?? 'modern'),
          lighting: firstPresetKey(presets, 'lights', current.lighting ?? 'daylight_1'),
          scenery: firstPresetKey(presets, 'sceneries', current.scenery ?? 'none'),
        }));
      })
      .catch((presetError) => {
        if (cancelled) return;
        setRenderStatus({
          tone: 'error',
          text:
            presetError instanceof Error
              ? presetError.message
              : 'Failed to load image presets.',
        });
      });
    return () => {
      cancelled = true;
    };
  }, []);

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

        const rawAssetId =
          placement?.asset_id ??
          obj.asset_id ??
          obj.assetId ??
          obj.object_id ??
          null;
        const frontWorld = placement?.front_world ?? obj.front_world ?? null;
        const frontSideWorld = placement?.front_side_world ?? obj.front_side_world ?? null;
        const frontSideLocal = placement?.front_side_local ?? obj.front_side_local ?? null;
        out.push({
          id,
          assetId: normalizeRenderAssetId(rawAssetId) ?? null,
          rawAssetId: typeof rawAssetId === 'string' ? rawAssetId : null,
          type: objType,
          colorHex: String(obj.color_hex ?? '#cfcac2'),
          material: obj.material ?? null,
          rotationCcw,
          frontWorld,
          frontSideWorld,
          frontSideLocal,
          bbox: { minX, minY, maxX, maxY },
          placeOn,
          draggable:
            placementsById.has(id) &&
            placementsById.get(id)?.constraint_mode !== 'locked',
          reintroduced:
            isReintroducedIdentifier(obj.object_id) ||
            isReintroducedIdentifier(rawAssetId) ||
            isReintroducedIdentifier(id),
        });
      }

      for (const p of placements) {
        const pid = String(p.placement_id ?? p.id ?? p.object_id ?? '');
        if (!pid || used.has(pid)) continue;
        const bbox = getWorldBboxFromPlacement(p);
        out.push({
          id: pid,
          assetId: normalizeRenderAssetId(p.asset_id ?? null) ?? null,
          rawAssetId: typeof p.asset_id === 'string' ? p.asset_id : null,
          type: p.category,
          colorHex: p.color_hex ?? '#cfcac2',
          material: null,
          rotationCcw: Number(p.rotation_deg ?? 0),
          frontWorld: p.front_world ?? null,
          frontSideWorld: p.front_side_world ?? null,
          frontSideLocal: p.front_side_local ?? null,
          bbox,
          placeOn: { method: 'floor', targetId: 'floor' },
          draggable: p.constraint_mode !== 'locked',
          reintroduced:
            isReintroducedIdentifier(p.asset_id) ||
            isReintroducedIdentifier(p.placement_id) ||
            isReintroducedIdentifier(pid),
        });
      }

      return out;
    }

    for (const placement of placements) {
      const pid = String(placement.placement_id ?? placement.id ?? placement.object_id ?? '');
      out.push({
        id: pid || String(out.length + 1),
        assetId: normalizeRenderAssetId(placement.asset_id ?? null) ?? null,
        rawAssetId:
          typeof placement.asset_id === 'string' ? placement.asset_id : null,
        type: placement.category,
        colorHex: placement.color_hex ?? '#cfcac2',
        material: null,
        rotationCcw: Number(placement.rotation_deg ?? 0),
        frontWorld: placement.front_world ?? null,
        frontSideWorld: placement.front_side_world ?? null,
        frontSideLocal: placement.front_side_local ?? null,
        bbox: getWorldBboxFromPlacement(placement),
        placeOn: { method: 'floor', targetId: 'floor' },
        draggable: placement.constraint_mode !== 'locked',
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

  const glbObjects = useMemo(
    () =>
      Object.values(objectVisualDiagnostics)
        .filter((item) => item.status === 'glb_loading' || item.status === 'glb_loaded')
        .sort((left, right) => left.label.localeCompare(right.label)),
    [objectVisualDiagnostics]
  );

  const lastVisibleObjects = lastRenderedSnapshot?.visibleObjects ?? [];
  const visibleObjectById = useMemo(
    () => new Map(lastVisibleObjects.map((object) => [object.id, object])),
    [lastVisibleObjects]
  );
  const effectiveEditObject =
    visibleObjectById.get(editSelectedObjectId) ??
    (lastVisibleObjects.length > 0 ? lastVisibleObjects[0] : null);
  const selectedObjectIsEditable = Boolean(
    selectedId && visibleObjectById.has(selectedId)
  );
  const editPresetActive = hasPresetSelection(editPresetSelection);
  const hasEditOperation =
    (replaceEnabled && replacementFile !== null && effectiveEditObject !== null) ||
    (colorEnabled && effectiveEditObject !== null) ||
    editPresetActive;

  useEffect(() => {
    if (selectedId && visibleObjectById.has(selectedId)) {
      setEditSelectedObjectId(selectedId);
      return;
    }
    if (!editSelectedObjectId && lastVisibleObjects.length > 0) {
      setEditSelectedObjectId(lastVisibleObjects[0].id);
    }
  }, [editSelectedObjectId, lastVisibleObjects, selectedId, visibleObjectById]);

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
            modelUrl: modelResolution?.descriptor?.url ?? null,
            previewRotationOffsetDeg: null,
            finalDescriptorRotationDeg: modelResolution?.descriptor?.rotationDeg ?? null,
            status,
            reason,
          } satisfies ObjectVisualDiagnostic,
        ];
      })
    );
    setObjectVisualDiagnostics(nextDiagnostics);
  }, [resolvedObjects]);

  async function handleCaptureSnapshot(): Promise<void> {
    const renderer = rendererRef.current;
    const scene = sceneRef.current;
    const camera = cameraRef.current;
    const renderState = renderStateRef.current;
    if (
      !renderer ||
      !scene ||
      !camera ||
      !renderState ||
      !(renderer.domElement instanceof HTMLCanvasElement)
    ) {
      setCaptureStatus({
        tone: 'error',
        text: '3D preview is not ready yet. Please wait for the scene to finish loading.',
      });
      return;
    }

    setIsCapturing(true);
    setCaptureStatus({
      tone: 'neutral',
      text: 'Capturing the current 3D view...',
    });

    const rotatorGroup = rotatorGroupRef.current;
    const previousRotatorVisibility =
      typeof rotatorGroup?.visible === 'boolean' ? rotatorGroup.visible : null;

    try {
      if (rotatorGroup && previousRotatorVisibility !== null) {
        rotatorGroup.visible = false;
      }
      controlsRef.current?.update?.();
      renderer.render(scene, camera);

      const snapshotArtifact = buildThreeSnapshotArtifact(renderState);
      const pngBlob = await canvasToPngBlob(renderer.domElement);
      const jsonBlob = new Blob([JSON.stringify(snapshotArtifact, null, 2)], {
        type: 'application/json',
      });
      const fileStem = buildSnapshotFileStem(
        snapshotArtifact.variantId ?? selectedVariantId,
        snapshotArtifact.capturedAt
      );

      downloadBlob(pngBlob, `${fileStem}.png`);
      downloadBlob(jsonBlob, `${fileStem}.json`);

      setCaptureStatus({
        tone: 'success',
        text: `Saved snapshot PNG and JSON with ${snapshotArtifact.visibleObjectCount} visible objects.`,
      });
    } catch (captureError) {
      setCaptureStatus({
        tone: 'error',
        text:
          captureError instanceof Error
            ? captureError.message
            : 'Failed to capture the current 3D snapshot.',
      });
    } finally {
      if (rotatorGroup && previousRotatorVisibility !== null) {
        rotatorGroup.visible = previousRotatorVisibility;
      }
      setIsCapturing(false);
    }
  }

  async function handleRenderSnapshotImage(): Promise<void> {
    const renderer = rendererRef.current;
    const scene = sceneRef.current;
    const camera = cameraRef.current;
    const renderState = renderStateRef.current;
    if (
      !renderer ||
      !scene ||
      !camera ||
      !renderState ||
      !(renderer.domElement instanceof HTMLCanvasElement)
    ) {
      setRenderStatus({
        tone: 'error',
        text: '3D preview is not ready yet. Please wait for the scene to finish loading.',
      });
      return;
    }

    setIsRenderingImage(true);
    setRenderStatus({
      tone: 'neutral',
      text: 'Capturing the current view and sending it to Nano Banana...',
    });

    const hiddenNodes: Array<{ visible: boolean; node: { visible: boolean } }> = [];
    const hideNode = (node: unknown): void => {
      if (!node || typeof node !== 'object' || !('visible' in node)) {
        return;
      }
      const visible = (node as { visible: boolean }).visible;
      hiddenNodes.push({ node: node as { visible: boolean }, visible });
      (node as { visible: boolean }).visible = false;
    };

    try {
      const rotatorGroup = rotatorGroupRef.current;
      if (rotatorGroup && typeof rotatorGroup.visible === 'boolean') {
        hideNode(rotatorGroup);
      }
      scene.traverse((node: any) => {
        if (node?.userData?.aiRenderHide === true && typeof node.visible === 'boolean') {
          hideNode(node);
        }
      });
      controlsRef.current?.update?.();
      renderer.render(scene, camera);
      const snapshotArtifact = buildThreeSnapshotArtifact(renderState);
      const snapshotImageDataUrl = canvasToDataUrl(renderer.domElement);
      const layoutReferenceImageDataUrl = buildLayoutReferenceImageDataUrl({
        floorplan: activeFloorplan,
        design,
      });
      const renderResult: SnapshotImageRenderApiResponse = await renderSnapshotImage(
        {
          snapshotPayload: snapshotArtifact as Record<string, unknown>,
          snapshotImageDataUrl,
          layoutReferenceImageDataUrl,
          userPrompt: renderPrompt,
          renderMode: 'generate',
          presetSelection,
        }
      );
      setRenderedSnapshotImage(renderResult.render);
      setLastRenderedSnapshot(snapshotArtifact);
      setLastRenderedSnapshotImageDataUrl(snapshotImageDataUrl);
      setEditSelectedObjectId(snapshotArtifact.visibleObjects[0]?.id ?? '');
      if (renderResult.saved_render) {
        upsertSavedRender(renderResult.saved_render);
        setSavedRenderId(renderResult.saved_render.id);
      } else {
        setSavedRenderId(null);
      }
      setRenderStatus({
        tone: 'success',
        text: `Generated image with ${renderResult.render.models.image_model_name}.`,
      });
    } catch (renderError) {
      setRenderStatus({
        tone: 'error',
        text:
          renderError instanceof Error
            ? renderError.message
            : 'Failed to generate an image from the current snapshot.',
      });
    } finally {
      for (let index = hiddenNodes.length - 1; index >= 0; index -= 1) {
        hiddenNodes[index].node.visible = hiddenNodes[index].visible;
      }
      controlsRef.current?.update?.();
      renderer.render(scene, camera);
      setIsRenderingImage(false);
    }
  }

  async function handleEditSnapshotImage(): Promise<void> {
    if (!renderedSnapshotImage || !lastRenderedSnapshot || !lastRenderedSnapshotImageDataUrl) {
      setRenderStatus({
        tone: 'error',
        text: 'Generate an image before applying edits.',
      });
      return;
    }
    if (!hasEditOperation) {
      setRenderStatus({
        tone: 'error',
        text: 'Choose at least one edit operation.',
      });
      return;
    }
    if ((replaceEnabled || colorEnabled) && !effectiveEditObject) {
      setRenderStatus({
        tone: 'error',
        text: 'Choose a visible object from the generated image.',
      });
      return;
    }

    setIsRenderingImage(true);
    setRenderStatus({
      tone: 'neutral',
      text: 'Applying the requested edit with Nano Banana...',
    });

    try {
      const editOperations: SnapshotImageEditOperation[] = [];
      if (effectiveEditObject && (replaceEnabled || colorEnabled)) {
        const replacementImageDataUrl =
          replaceEnabled && replacementFile ? await fileToDataUrl(replacementFile) : null;
        editOperations.push({
          object_id: effectiveEditObject.id,
          object_name: visibleObjectName(effectiveEditObject),
          replacement_image_data_url: replacementImageDataUrl,
          target_color: colorEnabled ? targetColor : null,
        });
      }

      const renderResult: SnapshotImageRenderApiResponse = await renderSnapshotImage(
        {
          snapshotPayload: lastRenderedSnapshot as unknown as Record<string, unknown>,
          snapshotImageDataUrl: lastRenderedSnapshotImageDataUrl,
          editSourceImageDataUrl: renderedSnapshotImage.image.data_url,
          userPrompt: renderPrompt,
          renderMode: 'edit',
          presetSelection: editPresetSelection,
          editOperations,
        }
      );
      setRenderedSnapshotImage(renderResult.render);
      if (renderResult.saved_render) {
        upsertSavedRender(renderResult.saved_render);
        setSavedRenderId(renderResult.saved_render.id);
      } else {
        setSavedRenderId(null);
      }
      setRenderStatus({
        tone: 'success',
        text: `Edited image with ${renderResult.render.models.image_model_name}.`,
      });
    } catch (editError) {
      setRenderStatus({
        tone: 'error',
        text:
          editError instanceof Error
            ? editError.message
            : 'Failed to edit the generated image.',
      });
    } finally {
      setIsRenderingImage(false);
    }
  }

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

        if (disposed || !el.isConnected) return;
        if (import.meta.env.DEV) {
          console.debug('[3D] init', { width, height });
        }

        renderer = new THREE.WebGLRenderer({
          antialias: true,
          preserveDrawingBuffer: true,
        });
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
        sceneRef.current = scene;
        rendererRef.current = renderer;
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
        cameraRef.current = camera;
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
        const ceilingColorHex =
          styledResult?.room?.surfaces?.ceiling_color_hex ??
          styledResult?.room?.surfaces?.ceilingColorHex ??
          null;

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
        grid.userData.aiRenderHide = true;
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
                  wall_id: `wall_${idx + 1}`,
                  p1_mm: p,
                  p2_mm: wallPts[(idx + 1) % wallPts.length],
                  thickness_mm: 100,
                }))
              : [];
        const wallOpacityById = new Map<string, number>();
        for (const wall of walls) {
          const wallId = String(wall.wall_id ?? '');
          if (!wallId) continue;
          wallOpacityById.set(wallId, clamp(Number(wall.opacity ?? 1), 0, 1));
        }
        const openings =
          activeFloorplan.openings ??
          styledResult?.room?.openings ?? {
            doors: [],
            windows: [],
          };
        const resolveSegment = (item: any): { p1: any; p2: any } | null => resolveOpeningSegmentMm(item);
        const wallGroup = new THREE.Group();
        scene.add(wallGroup);
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
          const wallId = String(wall.wall_id ?? '');
          const wallOpacity = clamp(Number(wall.opacity ?? 1), 0, 1);
          const isSelectedWall = !!wallId && selectedWallId === wallId;

          const geom = new THREE.BoxGeometry(lenMm * MM_TO_M, wallHeight, thicknessMm * MM_TO_M);
          const mat = new THREE.MeshStandardMaterial({
            color: new THREE.Color(isSelectedWall ? '#d58c54' : wallColorHex),
            transparent: wallOpacity < 1,
            opacity: wallOpacity,
          });
          const mesh = new THREE.Mesh(geom, mat);
          mesh.castShadow = true;
          mesh.receiveShadow = true;
          mesh.userData = {
            wallId,
            wallMesh: true,
            wallOpacity,
          };
          // 2D plan uses +Y "up". Our 3D preview maps plan Y to -Z so 3D matches the 2D canvas orientation.
          mesh.position.set((midX - centerXmm) * MM_TO_M, wallHeight / 2, -(midY - centerYmm) * MM_TO_M);
          mesh.rotation.y = angle;
          wallGroup.add(mesh);
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

        const heightById = new Map<string, number>();
        for (const obj of resolvedObjects) {
          const inventoryDimensions = getInventoryDimensionsMm(obj.inventoryResolution.item);
          const hMm =
            inventoryDimensions?.height ??
            defaultHeightMm(obj.type);
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
          const attachedWallOpacity = clamp(
            Number(wallOpacityById.get(String(door?.wall_id ?? '')) ?? 1),
            0,
            1,
          );

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
              transparent: attachedWallOpacity < 1,
              opacity: attachedWallOpacity,
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
          const attachedWallOpacity = clamp(
            Number(wallOpacityById.get(String(win?.wall_id ?? '')) ?? 1),
            0,
            1,
          );

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
              transparent: attachedWallOpacity < 1,
              opacity: attachedWallOpacity,
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
              opacity: attachedWallOpacity >= 1 ? 0.35 : attachedWallOpacity * 0.35,
            })
          );
          glass.position.z = frameDepthM / 2 - 0.006;
          glass.castShadow = false;
          glass.receiveShadow = false;
          group.add(glass);

          openingsGroup.add(group);
        }

        const draggableById = new Map<string, boolean>();
        for (const obj of resolvedObjects) {
          draggableById.set(obj.id, !!obj.draggable);
        }

        const proxyById = new Map<string, any>();
        const objectMetadataById = new Map<string, SnapshotObjectRecord>();
        const rotatorById = new Map<string, any>();

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

          // Keep the target box anchored to the styled-layout footprint, but let GLBs
          // preserve their native proportions inside that box via uniform scaling.
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
          // Prefer the same explicit front metadata that drives the 2D white marker.
          let rotY = resolvePlanObjectYawRad({
            rotationDeg: rotCcwDeg,
            frontWorld: obj.frontWorld,
            frontSideWorld: obj.frontSideWorld,
            frontSideLocal: obj.frontSideLocal,
          });
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
          objectRoot.userData = {
            objectId: obj.id,
            draggable: draggableById.get(obj.id) ?? false,
            baseY,
            method,
            wMm,
            dMm,
          };

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
          proxyById.set(obj.id, objectRoot);
          objectMetadataById.set(obj.id, {
            id: obj.id,
            type: obj.type,
            colorHex: obj.colorHex,
            material: obj.material ?? null,
            bboxMm: {
              minX: Math.round(obj.bbox.minX),
              minY: Math.round(obj.bbox.minY),
              maxX: Math.round(obj.bbox.maxX),
              maxY: Math.round(obj.bbox.maxY),
            },
            planPositionMm: {
              x: Math.round(pxMm),
              y: Math.round(pyMm),
              z: Math.round((baseY + hM / 2) / MM_TO_M),
            },
            dimensionsMm: {
              width: Math.round(wM / MM_TO_M),
              depth: Math.round(dM / MM_TO_M),
              height: Math.round(hM / MM_TO_M),
            },
            planRotationDeg: Math.round(rotCcwDeg * 100) / 100,
            sceneYawDeg: Math.round(yawRadToPlanRotationDeg(rotY) * 100) / 100,
            placeOn: obj.placeOn
              ? {
                  method: String(obj.placeOn.method ?? 'floor'),
                  targetId: obj.placeOn.targetId ?? null,
                }
              : null,
          });

          const modelDescriptor = obj.inventoryResolution.glbEligible
            ? obj.inventoryResolution.modelResolution?.descriptor ?? null
            : null;
          const preserveGeneratedMeshShape = isUserGeneratedPreviewItem(
            obj.inventoryResolution.item
          );
          const stretchWallHungToLayoutBox =
            isWallHung &&
            isShelfLikePreviewType(obj.type) &&
            !preserveGeneratedMeshShape;
          if (GLTFLoader && modelDescriptor) {
            const resolvedFitStrategy = isWallHung
              ? stretchWallHungToLayoutBox
                ? 'wall-plane'
                : preserveGeneratedMeshShape
                  ? 'wall-plane-uniform'
                  : 'wall-plane-uniform-or-proxy'
              : preserveGeneratedMeshShape
                ? 'footprint-uniform'
                : 'footprint-uniform-or-proxy';
            const previewOverride = applyInventoryModelPreviewOverride({
              descriptor: modelDescriptor,
              context: 'panel',
              fitStrategy: resolvedFitStrategy,
              wM,
              dM,
              hM,
            });
            const resolvedAssetId =
              obj.inventoryResolution.item?.asset_id?.toLowerCase() ?? null;
            const isSectionalSofaFamily =
              obj.type.toLowerCase().includes('sectional_sofa') ||
              resolvedAssetId?.startsWith('sectional_sofa_') === true;
            const needsSectionalFacingFallback =
              isSectionalSofaFamily &&
              !(previewOverride.applied?.rotationDegOffset === 180);
            const descriptorForPanel = needsSectionalFacingFallback
              ? {
                  ...previewOverride.descriptor,
                  rotationDeg: previewOverride.descriptor.rotationDeg + 180,
                }
              : previewOverride.descriptor;
            void buildGlbPreviewModel({
                THREE,
                GLTFLoader,
                descriptor: descriptorForPanel,
                wM: previewOverride.wM,
                dM: previewOverride.dM,
                hM: previewOverride.hM,
                fitStrategy: previewOverride.fitStrategy,
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
                    modelUrl: descriptorForPanel.url,
                    previewRotationOffsetDeg:
                      previewOverride.applied?.rotationDegOffset ?? 0,
                    finalDescriptorRotationDeg: descriptorForPanel.rotationDeg,
                    status: 'glb_loaded',
                    reason: previewOverride.applied
                      ? `Loaded GLB with preview override for inventory asset "${obj.inventoryResolution.item?.asset_id ?? obj.assetId ?? obj.type}".`
                      : needsSectionalFacingFallback
                        ? `Loaded GLB with sectional sofa panel facing fallback for inventory asset "${obj.inventoryResolution.item?.asset_id ?? obj.assetId ?? obj.type}".`
                      : `Loaded GLB for inventory asset "${obj.inventoryResolution.item?.asset_id ?? obj.assetId ?? obj.type}".`,
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
                const reason =
                  modelError instanceof Error
                    ? modelError.message
                    : 'Failed to load GLB, so the object stayed on the proxy mesh.';
                const isAspectMismatch = reason.includes('proxy fallback stays in place');
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
                    status: isAspectMismatch ? 'proxy_aspect_mismatch' : 'proxy_model_load_error',
                    reason,
                  },
                }));
                // Keep the procedural proxy as fallback when model loading fails.
              });
          }

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
            rotator.position.copy(objectRoot.position);
            rotator.rotation.y = objectRoot.rotation.y;
            rotator.userData = { objectId: obj.id, rotator: true };
            rotator.visible = false;
            rotatorGroup.add(rotator);
            rotatorById.set(obj.id, rotator);
          }
        }

        rotatorGroupRef.current = rotatorGroup;
        renderStateRef.current = {
          THREE,
          renderer,
          camera,
          controls,
          objectGroup,
          wallGroup,
          proxyById,
          objectMetadataById,
          roomBoundsMm: {
            minX: Math.round(bounds.minX),
            minY: Math.round(bounds.minY),
            maxX: Math.round(bounds.maxX),
            maxY: Math.round(bounds.maxY),
          },
          roomPolygonMm: roomPts.map((point) => ({
            x: Number(point.x),
            y: Number(point.y),
          })),
          roomCenterMm: {
            x: centerXmm,
            y: centerYmm,
          },
          roomOpenings: {
            doors: normalizeSnapshotOpenings(Array.isArray(openings.doors) ? openings.doors : []),
            windows: normalizeSnapshotOpenings(
              Array.isArray(openings.windows) ? openings.windows : []
            ),
          },
          roomSurfaceColors: {
            wallColorHex,
            floorColorHex,
            ceilingColorHex,
          },
          variantId: selectedVariantId,
          selectedObjectId: selectedIdRef.current,
          selectedWallId,
          editMode: editModeRef.current,
          wallOcclusionOpacityThreshold: SNAPSHOT_WALL_OCCLUSION_OPACITY_THRESHOLD,
        };

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
                  selectWall(null);
                  return;
                }
              }
              const wallIntersects = raycaster.intersectObjects(wallGroup.children, true);
              if (wallIntersects.length) {
                const wallHit = wallIntersects[0].object;
                const wallId = String(wallHit?.userData?.wallId ?? '');
                if (wallId) {
                  setSelectedId(null);
                  setEditMode('idle');
                  selectWall(wallId);
                  return;
                }
              }
              setSelectedId(null);
              selectWall(null);
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
      rendererRef.current = null;
      sceneRef.current = null;
      cameraRef.current = null;
      rotatorGroupRef.current = null;
      renderStateRef.current = null;
      if (import.meta.env.DEV) {
        console.debug('[3D] cleanup');
      }
    };
  }, [
    viewMode,
    activeFloorplan,
    resolvedObjects,
    styledResult,
    userInput.height,
    design,
    updatePlacement,
    selectedWallId,
    selectWall,
  ]);

  if (viewMode !== '3d') return null;

  return (
    <div className="three-panel">
      <div className="three-panel-header">
        <div className="three-panel-title">3D Preview</div>
        <div className="three-panel-actions">
          <button
            className="ghost"
            disabled={isCapturing}
            onClick={() => {
              void handleCaptureSnapshot();
            }}
          >
            {isCapturing ? 'Capturing...' : 'Capture'}
          </button>
          <button
            className="ghost"
            disabled={isRenderingImage}
            onClick={() => {
              void handleRenderSnapshotImage();
            }}
          >
            {isRenderingImage ? 'Generating...' : 'Generate Image'}
          </button>
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
          {selectedId ? <div className="three-panel-hint">Selected object: {selectedId}</div> : null}
          {!selectedId && selectedWallId ? (
            <div className="three-panel-hint">Selected wall: {selectedWallId}</div>
          ) : null}
          {captureStatus ? (
            <div
              aria-live="polite"
              className={`three-panel-status ${captureStatus.tone}`}
            >
              {captureStatus.text}
            </div>
          ) : null}
          {renderStatus ? (
            <div
              aria-live="polite"
              className={`three-panel-status ${renderStatus.tone}`}
            >
              {renderStatus.text}
            </div>
          ) : null}
          <div className="three-panel-prompt">
            <div className="reference-preset-stack">
              <RenderPresetSelector
                title="Style"
                group={renderPresets?.styles ?? {}}
                value={presetSelection.style}
                disabled={isRenderingImage || !renderPresets}
                onChange={(value) =>
                  setPresetSelection((current) => ({
                    ...current,
                    style: value,
                  }))
                }
              />
              <RenderPresetSelector
                title="Lighting"
                group={renderPresets?.lights ?? {}}
                value={presetSelection.lighting}
                disabled={isRenderingImage || !renderPresets}
                onChange={(value) =>
                  setPresetSelection((current) => ({
                    ...current,
                    lighting: value,
                  }))
                }
              />
              <RenderPresetSelector
                title="Scenery"
                group={renderPresets?.sceneries ?? {}}
                value={presetSelection.scenery}
                disabled={isRenderingImage || !renderPresets}
                onChange={(value) =>
                  setPresetSelection((current) => ({
                    ...current,
                    scenery: value,
                  }))
                }
              />
            </div>
            <label htmlFor="three-render-prompt">Image Prompt</label>
            <textarea
              id="three-render-prompt"
              className="three-panel-prompt-input"
              rows={3}
              disabled={isRenderingImage}
              placeholder="Describe the look you want, for example: warm Japandi materials, soft natural daylight, premium photorealistic styling."
              value={renderPrompt}
              onChange={(event) => setRenderPrompt(event.target.value)}
            />
            <div className="three-panel-prompt-note">
              The current 3D view is captured as the structural source image. Floor grid lines are removed before sending it to Nano Banana.
            </div>
            {renderedSnapshotImage ? (
              <div className="three-edit-controls">
                <div className="three-edit-object-field">
                  <label>
                    Object
                    <select
                      disabled={isRenderingImage || lastVisibleObjects.length === 0}
                      value={effectiveEditObject?.id ?? ''}
                      onChange={(event) => setEditSelectedObjectId(event.target.value)}
                    >
                      {lastVisibleObjects.map((object) => (
                        <option key={object.id} value={object.id}>
                          {visibleObjectLabel(object)}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
                <div className="reference-preset-stack compact">
                  <RenderPresetSelector
                    title="Edit Style"
                    group={renderPresets?.styles ?? {}}
                    value={editPresetSelection.style}
                    disabled={isRenderingImage || !renderPresets}
                    allowEmpty
                    emptyLabel="No change"
                    compact
                    onChange={(value) =>
                      setEditPresetSelection((current) => ({
                        ...current,
                        style: value,
                      }))
                    }
                  />
                  <RenderPresetSelector
                    title="Edit Lighting"
                    group={renderPresets?.lights ?? {}}
                    value={editPresetSelection.lighting}
                    disabled={isRenderingImage || !renderPresets}
                    allowEmpty
                    emptyLabel="No change"
                    compact
                    onChange={(value) =>
                      setEditPresetSelection((current) => ({
                        ...current,
                        lighting: value,
                      }))
                    }
                  />
                  <RenderPresetSelector
                    title="Edit Scenery"
                    group={renderPresets?.sceneries ?? {}}
                    value={editPresetSelection.scenery}
                    disabled={isRenderingImage || !renderPresets}
                    allowEmpty
                    emptyLabel="No change"
                    compact
                    onChange={(value) =>
                      setEditPresetSelection((current) => ({
                        ...current,
                        scenery: value,
                      }))
                    }
                  />
                </div>
                <div className="three-edit-row">
                  <label>
                    <input
                      type="checkbox"
                      checked={replaceEnabled}
                      onChange={(event) => setReplaceEnabled(event.target.checked)}
                    />
                    Replace
                  </label>
                  <input
                    type="file"
                    accept="image/png,image/jpeg,image/webp"
                    disabled={!replaceEnabled || isRenderingImage}
                    onChange={(event) => {
                      setReplacementFile(event.target.files?.[0] ?? null);
                    }}
                  />
                  <label>
                    <input
                      type="checkbox"
                      checked={colorEnabled}
                      onChange={(event) => setColorEnabled(event.target.checked)}
                    />
                    Color
                  </label>
                  <input
                    type="color"
                    disabled={!colorEnabled || isRenderingImage}
                    value={targetColor}
                    onChange={(event) => setTargetColor(event.target.value)}
                    aria-label="Target color"
                  />
                  <button
                    className="ghost"
                    disabled={isRenderingImage || !hasEditOperation}
                    onClick={() => {
                      void handleEditSnapshotImage();
                    }}
                  >
                    {isRenderingImage ? 'Editing...' : 'Apply Edit'}
                  </button>
                </div>
                {selectedId && !selectedObjectIsEditable ? (
                  <div className="three-panel-prompt-note">
                    Selected 3D object is not visible in this generated image.
                  </div>
                ) : null}
              </div>
            ) : null}
          </div>
          <div className="three-panel-hint">
            Click-drag to orbit. Click an object to select it, or click a wall to adjust its opacity.
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
      {reintroducedObjects.length > 0 || proxyObjects.length > 0 || glbObjects.length > 0 ? (
        <div className="three-debug-grid">
          {reintroducedObjects.length > 0 ? (
            <div className="three-debug-card">
              <div className="three-debug-title">
                Reintroduced Objects ({reintroducedObjects.length})
              </div>
              <div className="three-debug-list">
                {reintroducedObjects.map((item) => (
                  <div
                    key={`${item.clusterId ?? 'no-cluster'}-${item.objectId}`}
                    className="three-debug-item"
                  >
                    <strong>{item.objectType}</strong>
                    <span>{item.objectId}</span>
                    {item.clusterId ? <span>cluster: {item.clusterId}</span> : null}
                  </div>
                ))}
              </div>
            </div>
          ) : null}
          {glbObjects.length > 0 ? (
            <div className="three-debug-card">
              <div className="three-debug-title">Loaded GLBs ({glbObjects.length})</div>
              <div className="three-debug-list">
                {glbObjects.map((item) => {
                  const modelName = item.modelUrl?.split('/').pop() ?? 'n/a';
                  const offsetLabel =
                    typeof item.previewRotationOffsetDeg === 'number'
                      ? `${Math.round(item.previewRotationOffsetDeg)}deg`
                      : 'pending';
                  const finalRotationLabel =
                    typeof item.finalDescriptorRotationDeg === 'number'
                      ? `${Math.round(item.finalDescriptorRotationDeg)}deg`
                      : 'n/a';
                  return (
                    <div key={item.objectId} className="three-debug-item">
                      <strong>{item.label}</strong>
                      <span>
                        raw: {item.rawAssetId ?? 'n/a'} | inventory:{' '}
                        {item.resolvedAssetId ?? 'not matched'}
                      </span>
                      <span>
                        model: {modelName} | offset: {offsetLabel} | final rot:{' '}
                        {finalRotationLabel}
                      </span>
                      <span>{item.reason}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : null}
          {proxyObjects.length > 0 ? (
            <div className="three-debug-card">
              <div className="three-debug-title">
                Proxy Fallbacks ({proxyObjects.length})
              </div>
              <div className="three-debug-list">
                {proxyObjects.map((item) => (
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
      {renderedSnapshotImage ? (
        <div className="three-render-output">
          <div className="three-render-output-header">
            <div>
              <strong>AI Render</strong>
              <div className="muted">
                Image: {renderedSnapshotImage.models.image_model_name} | Aspect:{' '}
                {renderedSnapshotImage.image.aspect_ratio} | Max tokens:{' '}
                {renderedSnapshotImage.models.max_tokens} | 2D reference:{' '}
                {renderedSnapshotImage.request.layout_reference_used ? 'on' : 'off'}
              </div>
            </div>
            <div className="three-render-chip-row">
              <span className="three-render-chip">Gemini API Nano Banana</span>
              <span className="three-render-chip fallback">
                {renderedSnapshotImage.request.render_mode === 'edit' ? 'Edit' : 'Generate'}
              </span>
              {savedRenderId ? (
                <span className="three-render-chip fallback">
                  Saved to account: {savedRenderId}
                </span>
              ) : null}
            </div>
          </div>
          <div className="three-render-output-grid">
            <div className="three-render-image-wrap">
              <img
                className="three-render-image"
                src={renderedSnapshotImage.image.data_url}
                alt="Generated architectural render from the current 3D snapshot"
              />
            </div>
            <div className="three-render-meta">
              <div className="three-render-section">
                <strong>User Prompt</strong>
                <p>{renderedSnapshotImage.request.user_prompt}</p>
              </div>
              <div className="three-render-section">
                <strong>System Prompt Source</strong>
                <p>Nano Banana system prompt + preset prompt + source image.</p>
              </div>
              {renderedSnapshotImage.request.layout_reference_enabled ? (
                <div className="three-render-section">
                  <strong>2D Layout Reference</strong>
                  <p>
                    {renderedSnapshotImage.request.layout_reference_used
                      ? 'Enabled from backend env and sent as an extra reference image.'
                      : 'Enabled in code path, but no 2D reference image was used for this render.'}
                  </p>
                </div>
              ) : null}
            </div>
          </div>
        </div>
      ) : null}
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
