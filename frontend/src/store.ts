import { create } from 'zustand';
import {
  ensureCurrentBillingPeriod,
  getMonthlyImagesRemaining,
  loadBillingSettings,
  saveBillingSettings,
  type BillingSettings,
} from './billing';
import {
  AuthUser,
  DesignPlanV1,
  Door,
  FloorplanGeometry,
  GeneratedRenderRecord,
  InventoryItem,
  LayoutVariant,
  Placement,
  Point,
  SavedLayoutRecord,
  TemplateItem,
  Violation,
  Wall,
  Window,
} from './types';
import { normalizePoints } from './utils/geometry';
import { getWorldBboxFromPlacement } from './utils/styledLayout';

const DEFAULT_SHARED_TENANT_ID = 'demo_tenant';
const DEMO_INVENTORY_TENANT_ID = 'demo_inventory_test';
const TRUE_ENV_VALUES = new Set(['1', 'true', 'yes', 'on']);

const isDemoInventoryEnabled = () => {
  const raw =
    (import.meta as any).env?.TKNT_DEMO_INVENTORY_ENABLED ??
    (import.meta as any).env?.VITE_TKNT_DEMO_INVENTORY_ENABLED ??
    '';
  return TRUE_ENV_VALUES.has(String(raw).trim().toLowerCase());
};

const DEFAULT_TENANT_ID = isDemoInventoryEnabled()
  ? DEMO_INVENTORY_TENANT_ID
  : DEFAULT_SHARED_TENANT_ID;

export type UserInput = {
  description: string;
  room_type: string;
  style: string;
  special_description: string;
  feng_shui: string;
  floor_area_m2: number;
  height: number;
  shape_points: Point[];
  location_floor: number;
  bathrooms: number;
  windows: number;
  window_direction: string;
};

export type EditorState = {
  tenantId: string;
  authResolved: boolean;
  authToken: string | null;
  currentUser: AuthUser | null;
  floorplan: FloorplanGeometry | null;
  baseFloorplan: FloorplanGeometry | null;
  design: DesignPlanV1 | null;
  styledResult: any | null;
  layoutVariants: LayoutVariant[];
  selectedVariantId: string | null;
  variantPickerOpen: boolean;
  inventory: InventoryItem[];
  viewMode: '2d' | '3d';
  caseId: string | null;
  pipelineStage: string | null;
  clusterPreviews: Record<string, unknown>;
  templates: TemplateItem[];
  violations: Violation[];
  selectedPlacementId: string | null;
  selectedCornerIndex: number | null;
  selectedWallId: string | null;
  selectedOpeningId: string | null;
  selectedOpeningType: 'door' | 'window' | null;
  userInput: UserInput;
  roomEditMode: boolean;
  drawWallMode: boolean;
  runLogs: string[];
  runStatus: string | null;
  runIterations: number | null;
  lastDesignMeta: { iteration: number | null; attempt: number | null; receivedAt: number } | null;
  snapshotEnabled: boolean;
  additionRequests: Record<string, unknown>[];
  threeDOpen: boolean;
  savedLayouts: SavedLayoutRecord[];
  savedRenders: GeneratedRenderRecord[];
  activeRootLayoutId: string | null;
  selectedSavedRenderId: string | null;
  billingSettings: BillingSettings;
  setTenantId: (tenantId: string) => void;
  setAuthResolved: (resolved: boolean) => void;
  setAuthSession: (token: string, user: AuthUser) => void;
  clearAuthSession: () => void;
  setFloorplan: (floorplan: FloorplanGeometry) => void;
  setBaseFloorplan: (floorplan: FloorplanGeometry) => void;
  setDesign: (design: DesignPlanV1) => void;
  setStyledResult: (styled: any | null) => void;
  setLayoutVariants: (
    variants: LayoutVariant[],
    selectedVariantId?: string | null,
  ) => void;
  selectLayoutVariant: (variantId: string | null) => void;
  setVariantPickerOpen: (open: boolean) => void;
  clearLayoutVariants: () => void;
  setInventory: (inventory: InventoryItem[]) => void;
  setViewMode: (mode: '2d' | '3d') => void;
  setCaseId: (caseId: string | null) => void;
  setPipelineStage: (stage: string | null) => void;
  setClusterPreviews: (previews: Record<string, unknown>) => void;
  updateClusterPreview: (clusterId: string, preview: unknown) => void;
  setTemplates: (templates: TemplateItem[]) => void;
  setViolations: (violations: Violation[]) => void;
  selectPlacement: (placementId: string | null) => void;
  selectCorner: (index: number | null) => void;
  selectWall: (wallId: string | null) => void;
  selectOpening: (type: 'door' | 'window' | null, id?: string | null) => void;
  updateWall: (wallId: string, patch: Partial<Wall>) => void;
  updateOpening: (
    type: 'door' | 'window',
    id: string,
    patch: Partial<Door & Window>
  ) => void;
  removeWall: (wallId: string) => void;
  updatePlacement: (placement: Placement) => void;
  removePlacement: (placementId: string) => void;
  removeOpening: (type: 'door' | 'window', id: string) => void;
  setUserInput: (next: UserInput) => void;
  updateUserInput: (patch: Partial<UserInput>) => void;
  setRoomEditMode: (next: boolean) => void;
  setDrawWallMode: (next: boolean) => void;
  setRunLogs: (logs: string[]) => void;
  appendRunLog: (line: string) => void;
  clearRunLogs: () => void;
  setRunStatus: (status: string | null, iterations?: number | null) => void;
  setLastDesignMeta: (meta: { iteration: number | null; attempt: number | null } | null) => void;
  setSnapshotEnabled: (enabled: boolean) => void;
  setAdditionRequests: (requests: Record<string, unknown>[]) => void;
  setThreeDOpen: (open: boolean) => void;
  setSavedLayouts: (layouts: SavedLayoutRecord[]) => void;
  setSavedRenders: (renders: GeneratedRenderRecord[]) => void;
  upsertSavedRender: (render: GeneratedRenderRecord) => void;
  setActiveRootLayoutId: (layoutId: string | null) => void;
  selectSavedRender: (renderId: string | null) => void;
  setBillingSettings: (settings: BillingSettings) => void;
  updateBillingSettings: (patch: Partial<BillingSettings>) => void;
  consumeMonthlyImageCredit: () => boolean;
  recordPaidImageUse: () => void;
};

export const useEditorStore = create<EditorState>((set, get) => ({
  tenantId: DEFAULT_TENANT_ID,
  authResolved: false,
  authToken: null,
  currentUser: null,
  floorplan: null,
  baseFloorplan: null,
  design: null,
  styledResult: null,
  layoutVariants: [],
  selectedVariantId: null,
  variantPickerOpen: false,
  inventory: [],
  viewMode: '2d',
  caseId: null,
  pipelineStage: null,
  clusterPreviews: {},
  templates: [],
  violations: [],
  selectedPlacementId: null,
  selectedCornerIndex: null,
  selectedWallId: null,
  selectedOpeningId: null,
  selectedOpeningType: null,
  userInput: {
    description: 'Auto-generated layout',
    room_type: 'office',
    style: 'japandi',
    special_description: '',
    feng_shui: '',
    floor_area_m2: 0,
    height: 2800,
    shape_points: [],
    location_floor: 1,
    bathrooms: 1,
    windows: 0,
    window_direction: '',
  },
  roomEditMode: false,
  drawWallMode: false,
  runLogs: [],
  runStatus: null,
  runIterations: null,
  lastDesignMeta: null,
  snapshotEnabled: true,
  additionRequests: [],
  threeDOpen: false,
  savedLayouts: [],
  savedRenders: [],
  activeRootLayoutId: null,
  selectedSavedRenderId: null,
  billingSettings: loadBillingSettings(),
  setTenantId: (tenantId) => set({ tenantId }),
  setAuthResolved: (authResolved) => set({ authResolved }),
  setAuthSession: (authToken, currentUser) =>
    set({
      authResolved: true,
      authToken,
      currentUser,
    }),
  clearAuthSession: () =>
    set({
      authResolved: true,
      authToken: null,
      currentUser: null,
      tenantId: DEFAULT_TENANT_ID,
      savedLayouts: [],
      savedRenders: [],
      activeRootLayoutId: null,
      selectedSavedRenderId: null,
      inventory: [],
    }),
  setFloorplan: (floorplan) => {
    const currentState = get();
    const normalized = normalizePoints(floorplan.room.polygon_mm);
    const shiftDx = normalized.dx;
    const shiftDy = normalized.dy;
    if (shiftDx !== 0 || shiftDy !== 0) {
      const shiftPoint = (p: Point) => ({ x: p.x - shiftDx, y: p.y - shiftDy });
      floorplan = {
        ...floorplan,
        room: { ...floorplan.room, polygon_mm: normalized.points },
        walls: floorplan.walls.map((wall) => ({
          ...wall,
          p1_mm: shiftPoint(wall.p1_mm),
          p2_mm: shiftPoint(wall.p2_mm),
        })),
        openings: {
          doors: floorplan.openings.doors.map((door) => ({
            ...door,
            start_mm: shiftPoint(door.start_mm),
            end_mm: shiftPoint(door.end_mm),
          })),
          windows: floorplan.openings.windows.map((win) => ({
            ...win,
            start_mm: shiftPoint(win.start_mm),
            end_mm: shiftPoint(win.end_mm),
          })),
        },
      };
      const design = currentState.design;
      if (design) {
        const shiftedPlacements = design.placements.map((placement) => ({
          ...placement,
          position_mm: shiftPoint(placement.position_mm),
        }));
        set({ design: { ...design, placements: shiftedPlacements } });
      }
    }
    let points = floorplan.room.polygon_mm ?? [];
    let walls = floorplan.walls ?? [];
    if (walls.length === 0 && points.length >= 2 && !currentState.roomEditMode) {
      const thickness = floorplan.walls?.[0]?.thickness_mm ?? 100;
      walls = buildWallsFromPolygon(points, thickness);
    }
    if (walls.length > 0) {
      const polygon = polygonFromWalls(walls);
      set((state) => {
        if (!state.roomEditMode && polygon.length >= 3) {
          points = polygon;
        }
        const nextFloorplan: FloorplanGeometry = {
          ...floorplan,
          room: { ...floorplan.room, polygon_mm: points },
          walls,
        };
      const area = state.roomEditMode ? state.userInput.floor_area_m2 : polygonArea(points);
      return {
        floorplan: nextFloorplan,
        userInput: {
          ...state.userInput,
          shape_points: points,
          floor_area_m2: area,
          windows: nextFloorplan.openings.windows.length,
        },
      };
    });
    return;
  }
    const nextFloorplan: FloorplanGeometry = {
      ...floorplan,
      room: { ...floorplan.room, polygon_mm: points },
      walls,
    };
    const area = currentState.roomEditMode
      ? currentState.userInput.floor_area_m2
      : polygonArea(points);
    set((state) => ({
      floorplan: nextFloorplan,
      userInput: {
        ...state.userInput,
        shape_points: points,
        floor_area_m2: area,
        windows: nextFloorplan.openings.windows.length,
      },
    }));
  },
  setBaseFloorplan: (floorplan) => set({ baseFloorplan: floorplan }),
  setDesign: (design) => {
    const floorplan = get().floorplan ?? design?.floorplan_geometry ?? null;
    const placements = Array.isArray(design?.placements)
      ? design.placements.map((p) => ({
          ...p,
          placement_id: String(p.placement_id),
        }))
      : [];
    const clampedPlacements = floorplan
      ? placements.map((p) => clampPlacementToFloorplan(p as Placement, floorplan))
      : placements;
    set({ design: { ...design, placements: clampedPlacements } });
  },
  setStyledResult: (styledResult) => set({ styledResult }),
  setLayoutVariants: (layoutVariants, selectedVariantId = null) =>
    set({
      layoutVariants,
      selectedVariantId:
        selectedVariantId && layoutVariants.some((item) => item.variant_id === selectedVariantId)
          ? selectedVariantId
          : layoutVariants[0]?.variant_id ?? null,
    }),
  selectLayoutVariant: (selectedVariantId) => set({ selectedVariantId }),
  setVariantPickerOpen: (variantPickerOpen) => set({ variantPickerOpen }),
  clearLayoutVariants: () =>
    set({
      layoutVariants: [],
      selectedVariantId: null,
      variantPickerOpen: false,
    }),
  setInventory: (inventory) => set({ inventory }),
  setViewMode: (viewMode) => set({ viewMode }),
  setCaseId: (caseId) => set({ caseId }),
  setPipelineStage: (pipelineStage) => set({ pipelineStage }),
  setClusterPreviews: (clusterPreviews) => set({ clusterPreviews }),
  updateClusterPreview: (clusterId, preview) =>
    set((state) => ({ clusterPreviews: { ...state.clusterPreviews, [clusterId]: preview } })),
  setTemplates: (templates) => set({ templates }),
  setViolations: (violations) => set({ violations }),
  selectPlacement: (placementId) =>
    set({
      selectedPlacementId: placementId,
      selectedOpeningId: null,
      selectedOpeningType: null,
      selectedWallId: null,
      selectedCornerIndex: null,
    }),
  selectCorner: (index) =>
    set({
      selectedCornerIndex: index,
      selectedOpeningId: null,
      selectedOpeningType: null,
      selectedWallId: null,
      selectedPlacementId: null,
    }),
  selectWall: (wallId) =>
    set({
      selectedWallId: wallId,
      selectedOpeningId: null,
      selectedOpeningType: null,
      selectedPlacementId: null,
      selectedCornerIndex: null,
    }),
  selectOpening: (type, id = null) =>
    set({
      selectedOpeningType: type,
      selectedOpeningId: id,
      selectedPlacementId: null,
      selectedWallId: null,
      selectedCornerIndex: null,
    }),
  updateWall: (wallId, patch) => {
    const state = get();
    const floorplan = state.floorplan ?? state.design?.floorplan_geometry ?? null;
    if (!floorplan) return;
    const walls = floorplan.walls.map((wall) =>
      wall.wall_id === wallId ? { ...wall, ...patch } : wall
    );
    const nextFloorplan = { ...floorplan, walls };
    set({
      floorplan: state.floorplan ? nextFloorplan : state.floorplan,
      design: state.design
        ? {
            ...state.design,
            floorplan_geometry: nextFloorplan,
          }
        : state.design,
    });
  },
  updateOpening: (type, id, patch) => {
    const state = get();
    const floorplan = state.floorplan ?? state.design?.floorplan_geometry ?? null;
    if (!floorplan) return;
    const nextFloorplan: FloorplanGeometry = {
      ...floorplan,
      openings: {
        doors:
          type === 'door'
            ? floorplan.openings.doors.map((door) =>
                door.door_id === id ? { ...door, ...patch } : door
              )
            : floorplan.openings.doors,
        windows:
          type === 'window'
            ? floorplan.openings.windows.map((window) =>
                window.window_id === id
                  ? { ...window, ...patch }
                  : window
              )
            : floorplan.openings.windows,
      },
    };
    set({
      floorplan: state.floorplan ? nextFloorplan : state.floorplan,
      design: state.design
        ? {
            ...state.design,
            floorplan_geometry: nextFloorplan,
          }
        : state.design,
    });
  },
  removeWall: (wallId) => {
    const state = get();
    const floorplan = state.floorplan ?? state.design?.floorplan_geometry ?? null;
    if (!floorplan) return;
    const nextFloorplan = {
      ...floorplan,
      walls: floorplan.walls.filter((wall) => wall.wall_id !== wallId),
    };
    set({
      floorplan: state.floorplan ? nextFloorplan : state.floorplan,
      design: state.design
        ? {
            ...state.design,
            floorplan_geometry: nextFloorplan,
          }
        : state.design,
      selectedWallId: null,
    });
  },
  updatePlacement: (placement) => {
    const state = get();
    const design = state.design;
    if (!design) return;
    const floorplan = state.floorplan ?? design.floorplan_geometry ?? null;
    const next = floorplan ? clampPlacementToFloorplan(placement, floorplan) : placement;
    const placements = design.placements.map((p) => (p.placement_id === next.placement_id ? next : p));
    set({
      design: { ...design, placements },
      styledResult: syncStyledResultPlacement(state.styledResult, next),
    });
  },
  removePlacement: (placementId) => {
    const state = get();
    const design = state.design;
    if (!design) return;
    set({
      design: {
        ...design,
        placements: design.placements.filter((p) => p.placement_id !== placementId),
      },
      styledResult: removePlacementFromStyledResult(state.styledResult, placementId),
      selectedPlacementId:
        state.selectedPlacementId === placementId ? null : state.selectedPlacementId,
    });
  },
  removeOpening: (type, id) => {
    const floorplan = get().floorplan;
    if (!floorplan) return;
    if (type === 'door') {
      set({
        floorplan: {
          ...floorplan,
          openings: { ...floorplan.openings, doors: floorplan.openings.doors.filter((d) => d.door_id !== id) },
        },
        selectedOpeningId: null,
        selectedOpeningType: null,
      });
      return;
    }
    set({
      floorplan: {
        ...floorplan,
        openings: { ...floorplan.openings, windows: floorplan.openings.windows.filter((w) => w.window_id !== id) },
      },
      selectedOpeningId: null,
      selectedOpeningType: null,
    });
  },
  setUserInput: (next) => set({ userInput: next }),
  updateUserInput: (patch) => {
    const current = get().userInput;
    set({ userInput: { ...current, ...patch } });
  },
  setRoomEditMode: (next) => set({ roomEditMode: next }),
  setDrawWallMode: (next) => set({ drawWallMode: next }),
  setRunLogs: (logs) => set({ runLogs: logs }),
  appendRunLog: (line) =>
    set((state) => ({ runLogs: [...state.runLogs, line] })),
  clearRunLogs: () => set({ runLogs: [] }),
  setRunStatus: (status, iterations = null) =>
    set({ runStatus: status, runIterations: iterations }),
  setLastDesignMeta: (meta) =>
    set({
      lastDesignMeta: meta
        ? { iteration: meta.iteration ?? null, attempt: meta.attempt ?? null, receivedAt: Date.now() }
        : null,
    }),
  setSnapshotEnabled: (enabled) => set({ snapshotEnabled: enabled }),
  setAdditionRequests: (requests) => set({ additionRequests: requests }),
  setThreeDOpen: (open) => set({ threeDOpen: open }),
  setSavedLayouts: (savedLayouts) => set({ savedLayouts }),
  setSavedRenders: (savedRenders) => set({ savedRenders }),
  upsertSavedRender: (render) =>
    set((state) => {
      const next = [render, ...state.savedRenders.filter((item) => item.id !== render.id)];
      return { savedRenders: next };
    }),
  setActiveRootLayoutId: (activeRootLayoutId) => set({ activeRootLayoutId }),
  selectSavedRender: (selectedSavedRenderId) => set({ selectedSavedRenderId }),
  setBillingSettings: (settings) => {
    const next = ensureCurrentBillingPeriod(settings);
    saveBillingSettings(next);
    set({ billingSettings: next });
  },
  updateBillingSettings: (patch) =>
    set((state) => {
      const next = ensureCurrentBillingPeriod({
        ...state.billingSettings,
        ...patch,
      });
      saveBillingSettings(next);
      return { billingSettings: next };
    }),
  consumeMonthlyImageCredit: () => {
    const current = ensureCurrentBillingPeriod(get().billingSettings);
    if (current.mode !== 'monthly' || getMonthlyImagesRemaining(current) <= 0) {
      if (current !== get().billingSettings) {
        saveBillingSettings(current);
        set({ billingSettings: current });
      }
      return false;
    }
    const next = {
      ...current,
      monthlyImagesUsed: current.monthlyImagesUsed + 1,
      totalMonthlyUses: current.totalMonthlyUses + 1,
    };
    saveBillingSettings(next);
    set({ billingSettings: next });
    return true;
  },
  recordPaidImageUse: () =>
    set((state) => {
      const next = ensureCurrentBillingPeriod({
        ...state.billingSettings,
        totalPaidUses: state.billingSettings.totalPaidUses + 1,
      });
      saveBillingSettings(next);
      return { billingSettings: next };
    }),
}));

function syncStyledResultPlacement(
  styledResult: unknown,
  placement: Placement
): unknown {
  if (
    !styledResult ||
    typeof styledResult !== 'object' ||
    !Array.isArray((styledResult as { objects?: unknown[] }).objects)
  ) {
    return styledResult;
  }

  const nextStyledResult = {
    ...(styledResult as Record<string, unknown>),
  };
  const worldBbox = getWorldBboxFromPlacement(placement);
  nextStyledResult.objects = (styledResult as { objects: unknown[] }).objects.map((item) => {
    if (!item || typeof item !== 'object') {
      return item;
    }
    const objectRow = item as Record<string, unknown>;
    const objectId =
      typeof objectRow.instance_id === 'string'
        ? objectRow.instance_id
        : typeof objectRow.id === 'string'
          ? objectRow.id
          : readStyledResultManualPlacementId(objectRow);
    if (objectId !== placement.placement_id) {
      return item;
    }
    const existingManualInput =
      objectRow.manual_input && typeof objectRow.manual_input === 'object'
        ? (objectRow.manual_input as Record<string, unknown>)
        : {};
    return {
      ...objectRow,
      object_type: placement.category,
      color_hex: placement.color_hex,
      rotation_ccw: ((placement.rotation_deg % 360) + 360) % 360,
      front_world: placement.front_world ?? null,
      front_side_world: placement.front_side_world ?? null,
      front_side_local: placement.front_side_local ?? null,
      bbox: {
        min_x: worldBbox.minX,
        min_y: worldBbox.minY,
        max_x: worldBbox.maxX,
        max_y: worldBbox.maxY,
      },
      manual_input: {
        ...existingManualInput,
        ...placement,
      },
    };
  });
  return nextStyledResult;
}

function removePlacementFromStyledResult(
  styledResult: unknown,
  placementId: string
): unknown {
  if (
    !styledResult ||
    typeof styledResult !== 'object' ||
    !Array.isArray((styledResult as { objects?: unknown[] }).objects)
  ) {
    return styledResult;
  }

  const nextStyledResult = {
    ...(styledResult as Record<string, unknown>),
  };
  nextStyledResult.objects = (styledResult as { objects: unknown[] }).objects.filter(
    (item) => {
      if (!item || typeof item !== 'object') {
        return true;
      }
      const objectRow = item as Record<string, unknown>;
      const objectId =
        typeof objectRow.instance_id === 'string'
          ? objectRow.instance_id
          : typeof objectRow.id === 'string'
            ? objectRow.id
            : readStyledResultManualPlacementId(objectRow);
      return objectId !== placementId;
    }
  );
  return nextStyledResult;
}

function readStyledResultManualPlacementId(
  objectRow: Record<string, unknown>
): string | null {
  const manualInput = objectRow.manual_input;
  if (!manualInput || typeof manualInput !== 'object') {
    return null;
  }
  const placementId = (manualInput as Record<string, unknown>).placement_id;
  return typeof placementId === 'string' && placementId.trim() ? placementId : null;
}

function polygonArea(points: Point[]): number {
  if (points.length < 3) return 0;
  let sum = 0;
  for (let i = 0; i < points.length; i++) {
    const p1 = points[i];
    const p2 = points[(i + 1) % points.length];
    sum += p1.x * p2.y - p2.x * p1.y;
  }
  return Math.abs(sum / 2) / 1_000_000;
}

function buildWallsFromPolygon(points: Point[], thickness: number): Wall[] {
  if (points.length < 2) return [];
  return points.map((point, idx) => {
    const next = points[(idx + 1) % points.length];
    return {
      wall_id: `wall_${idx + 1}`,
      p1_mm: point,
      p2_mm: next,
      thickness_mm: thickness,
      room_id: 'room_1',
    };
  });
}

function polygonFromWalls(walls: Wall[]): Point[] {
  if (walls.length < 3) return [];
  const key = (p: Point) => `${Math.round(p.x)}:${Math.round(p.y)}`;
  const nodes = new Map<string, Point>();
  const adjacency = new Map<string, string[]>();
  walls.forEach((wall) => {
    const p1 = wall.p1_mm;
    const p2 = wall.p2_mm;
    const k1 = key(p1);
    const k2 = key(p2);
    nodes.set(k1, p1);
    nodes.set(k2, p2);
    adjacency.set(k1, [...(adjacency.get(k1) ?? []), k2]);
    adjacency.set(k2, [...(adjacency.get(k2) ?? []), k1]);
  });
  const startKey = adjacency.keys().next().value as string | undefined;
  if (!startKey) return [];
  const polygon: Point[] = [];
  let current = startKey;
  let prev: string | null = null;
  let closed = false;
  for (let i = 0; i < walls.length + 2; i++) {
    const point = nodes.get(current);
    if (!point) break;
    polygon.push(point);
    const neighbors = adjacency.get(current) ?? [];
    const next = neighbors.find((n) => n !== prev);
    if (!next) break;
    prev = current;
    current = next;
    if (current === startKey) {
      closed = true;
      break;
    }
  }
  return closed ? polygon : [];
}

function clampPlacementToFloorplan(placement: Placement, floorplan: FloorplanGeometry): Placement {
  const nextPos = clampPlacementPositionToRoomInterior(
    placement.position_mm,
    placement.footprint_mm,
    placement.rotation_deg,
    floorplan,
  );
  if (nextPos.x === placement.position_mm.x && nextPos.y === placement.position_mm.y) return placement;
  return { ...placement, position_mm: nextPos };
}

function clampPlacementPositionToRoomInterior(
  position: Point,
  footprint: { w: number; d: number },
  rotationDeg: number,
  floorplan: FloorplanGeometry,
): Point {
  const polygon = floorplan.room?.polygon_mm ?? [];
  const walls = floorplan.walls ?? [];
  if (polygon.length < 3 || walls.length === 0) return position;

  const wHalf = Math.max(0, Number(footprint?.w ?? 0)) / 2;
  const dHalf = Math.max(0, Number(footprint?.d ?? 0)) / 2;
  const theta = (Number(rotationDeg ?? 0) * Math.PI) / 180;
  const cos = Math.cos(theta);
  const sin = Math.sin(theta);

  // Two passes is usually enough to satisfy corner constraints without jitter.
  let p = { ...position };
  for (let pass = 0; pass < 2; pass += 1) {
    let moved = false;
    for (const wall of walls) {
      const proj = projectPointToSegment(p, wall.p1_mm, wall.p2_mm);
      if (!proj) continue;
      // For concave polygons (e.g. stepped ceilings/notches), endpoint projections can incorrectly
      // apply an edge's half-plane constraint to points that are actually valid elsewhere in the room.
      // Skipping endpoint-only projections avoids "invisible barriers" while the final polygon check
      // still prevents dragging objects outside the room.
      if (proj.t <= 1e-6 || proj.t >= 1 - 1e-6) continue;

      const thicknessMm = Math.max(0, Number(wall.thickness_mm ?? 0));
      const dir = proj.dir;
      const leftNormal = { x: -dir.y, y: dir.x };
      const interiorNormal = interiorNormalForWall(proj.point, leftNormal, polygon);

      const halfExtent =
        wHalf * Math.abs(interiorNormal.x * cos + interiorNormal.y * sin) +
        dHalf * Math.abs(interiorNormal.x * -sin + interiorNormal.y * cos);

      // Wall lines are treated as centerlines; inner face is thickness/2 toward the room interior.
      const requiredDist = thicknessMm / 2 + halfExtent;
      const signedDist = dot(
        { x: p.x - proj.point.x, y: p.y - proj.point.y },
        interiorNormal,
      );

      if (signedDist + 1e-6 < requiredDist) {
        const push = requiredDist - signedDist;
        p = {
          x: p.x + interiorNormal.x * push,
          y: p.y + interiorNormal.y * push,
        };
        moved = true;
      }
    }
    if (!moved) break;
  }

  // If we ended up outside due to numeric issues (or non-simple polygons), fall back to the input.
  if (!pointInPolygon(p, polygon)) return position;
  return p;
}

function dot(a: Point, b: Point): number {
  return a.x * b.x + a.y * b.y;
}

function clampNumber(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function projectPointToSegment(
  p: Point,
  a: Point,
  b: Point,
): { t: number; point: Point; dir: Point } | null {
  const vx = b.x - a.x;
  const vy = b.y - a.y;
  const len2 = vx * vx + vy * vy;
  if (len2 < 1e-6) return null;
  let t = ((p.x - a.x) * vx + (p.y - a.y) * vy) / len2;
  t = clampNumber(t, 0, 1);
  const proj = { x: a.x + t * vx, y: a.y + t * vy };
  const len = Math.sqrt(len2) || 1;
  return { t, point: proj, dir: { x: vx / len, y: vy / len } };
}

function polygonCentroid(points: Point[]): Point {
  if (!points || points.length === 0) return { x: 0, y: 0 };
  let x = 0;
  let y = 0;
  for (const p of points) {
    x += p.x;
    y += p.y;
  }
  return { x: x / points.length, y: y / points.length };
}

function interiorNormalForWall(boundaryPoint: Point, leftNormal: Point, polygon: Point[]): Point {
  // Nudge 1mm to avoid the ambiguous "point on boundary" case.
  const eps = 1;
  const leftTest = { x: boundaryPoint.x + leftNormal.x * eps, y: boundaryPoint.y + leftNormal.y * eps };
  const rightTest = { x: boundaryPoint.x - leftNormal.x * eps, y: boundaryPoint.y - leftNormal.y * eps };
  const leftInside = pointInPolygon(leftTest, polygon);
  const rightInside = pointInPolygon(rightTest, polygon);
  if (leftInside && !rightInside) return leftNormal;
  if (rightInside && !leftInside) return { x: -leftNormal.x, y: -leftNormal.y };

  // Fallback: pick the normal that points toward the polygon centroid.
  const center = polygonCentroid(polygon);
  const toCenter = { x: center.x - boundaryPoint.x, y: center.y - boundaryPoint.y };
  return dot(toCenter, leftNormal) >= 0 ? leftNormal : { x: -leftNormal.x, y: -leftNormal.y };
}

function pointInPolygon(point: Point, polygon: Point[]): boolean {
  if (!polygon || polygon.length < 3) return true;
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x;
    const yi = polygon[i].y;
    const xj = polygon[j].x;
    const yj = polygon[j].y;
    const intersect =
      yi > point.y !== yj > point.y &&
      point.x < ((xj - xi) * (point.y - yi)) / (yj - yi + 1e-9) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}
