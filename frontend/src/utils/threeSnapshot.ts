export type SnapshotVector3 = {
  x: number;
  y: number;
  z: number;
};

export type SnapshotBoundsMm = {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
};

export type SnapshotPointMm = {
  x: number;
  y: number;
};

export type SnapshotScreenBboxPx = {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  width: number;
  height: number;
};

export type SnapshotObjectRecord = {
  id: string;
  assetId?: string | null;
  rawAssetId?: string | null;
  resolvedAssetId?: string | null;
  type: string;
  renderIdentityLabel?: string | null;
  renderIdentityHint?: string | null;
  modelUrl?: string | null;
  modelSource?: string | null;
  glbFitStrategy?: string | null;
  colorHex: string;
  material: string | null;
  bboxMm: SnapshotBoundsMm;
  planPositionMm: SnapshotVector3;
  dimensionsMm: {
    width: number;
    depth: number;
    height: number;
  };
  planRotationDeg: number;
  sceneYawDeg: number;
  placeOn: {
    method: string;
    targetId: string | null;
  } | null;
};

export type VisibleSnapshotObject = SnapshotObjectRecord & {
  distanceToCameraM: number;
  screenCenterPx: {
    x: number;
    y: number;
  } | null;
  screenBboxPx: SnapshotScreenBboxPx | null;
  visibleSampleCount: number;
  visibleSampleFraction: number;
  visibilitySource?: 'raycast' | 'projected_visibility';
};

export type SnapshotOpeningRecord = {
  id: string;
  segmentMm: [SnapshotPointMm, SnapshotPointMm];
  screenCenterPx?: {
    x: number;
    y: number;
  } | null;
  screenBboxPx?: SnapshotScreenBboxPx | null;
};

export type ThreeSnapshotRenderState = {
  THREE: any;
  renderer: any;
  camera: any;
  controls: any | null;
  objectGroup: any;
  wallGroup: any | null;
  openingsGroup: any | null;
  proxyById: Map<string, any>;
  objectMetadataById: Map<string, SnapshotObjectRecord>;
  roomBoundsMm: SnapshotBoundsMm;
  roomPolygonMm: SnapshotPointMm[];
  roomCenterMm: {
    x: number;
    y: number;
  };
  roomOpenings: {
    doors: SnapshotOpeningRecord[];
    windows: SnapshotOpeningRecord[];
  };
  roomSurfaceColors: {
    wallColorHex: string | null;
    floorColorHex: string | null;
    ceilingColorHex: string | null;
  };
  variantId: string | null;
  selectedObjectId: string | null;
  selectedWallId: string | null;
  editMode: 'idle' | 'edit';
  wallOcclusionOpacityThreshold: number;
};

export type ThreeSnapshotArtifact = {
  capturedAt: string;
  variantId: string | null;
  selection: {
    selectedObjectId: string | null;
    selectedWallId: string | null;
    editMode: 'idle' | 'edit';
  };
  canvas: {
    widthPx: number;
    heightPx: number;
  };
  room: {
    boundsMm: SnapshotBoundsMm;
    polygonMm: SnapshotPointMm[];
    surfaceColors: {
      wallColorHex: string | null;
      floorColorHex: string | null;
      ceilingColorHex: string | null;
    };
    openings: {
      doors: SnapshotOpeningRecord[];
      windows: SnapshotOpeningRecord[];
    };
  };
  camera: {
    positionM: SnapshotVector3;
    positionRoomMm: SnapshotVector3;
    targetM: SnapshotVector3 | null;
    targetRoomMm: SnapshotVector3 | null;
    fovDeg: number;
    aspect: number;
    near: number;
    far: number;
  };
  totalObjectCount: number;
  visibleObjectCount: number;
  hiddenObjectCount: number;
  visibilityMethod: string;
  visibilityDiagnostics: {
    status: 'ok' | 'degraded';
    allObjectCount: number;
    visibleObjectCount: number;
    raycastVisibleObjectCount: number;
    projectedFallbackObjectCount: number;
    openingBboxMissingCount: number;
    openingIdsMissingBbox: string[];
    reasons: string[];
  };
  wallOcclusionOpacityThreshold: number;
  allObjects: SnapshotObjectRecord[];
  visibleObjects: VisibleSnapshotObject[];
};

const MM_PER_M = 1000;
const OCCLUSION_TOLERANCE_M = 0.03;
const SNAPSHOT_RAYCAST_LINE_THRESHOLD_M = 0.005;

export function buildThreeSnapshotArtifact(
  state: ThreeSnapshotRenderState
): ThreeSnapshotArtifact {
  const {
    THREE,
    renderer,
    camera,
    controls,
    objectGroup,
    wallGroup,
    openingsGroup,
    proxyById,
    objectMetadataById,
} =
    state;
  objectGroup.updateMatrixWorld?.(true);
  wallGroup?.updateMatrixWorld?.(true);
  openingsGroup?.updateMatrixWorld?.(true);
  camera.updateMatrixWorld?.(true);
  camera.updateProjectionMatrix?.();

  const projectionMatrix = new THREE.Matrix4().multiplyMatrices(
    camera.projectionMatrix,
    camera.matrixWorldInverse
  );
  const frustum = new THREE.Frustum().setFromProjectionMatrix(projectionMatrix);
  const cameraWorldPosition = camera.getWorldPosition(new THREE.Vector3());
  const targetVector = controls?.target
    ? vectorToPlain(controls.target)
    : null;
  const visibleObjects: VisibleSnapshotObject[] = [];
  const raycaster = new THREE.Raycaster();
  raycaster.params.Line.threshold = SNAPSHOT_RAYCAST_LINE_THRESHOLD_M;
  raycaster.params.Points.threshold = SNAPSHOT_RAYCAST_LINE_THRESHOLD_M;
  const canvasRect = renderer.domElement.getBoundingClientRect();
  const canvasWidth = Math.max(
    1,
    Math.round(renderer.domElement.width || canvasRect.width || 1)
  );
  const canvasHeight = Math.max(
    1,
    Math.round(renderer.domElement.height || canvasRect.height || 1)
  );

  const visibleCandidates: VisibleSnapshotObject[] = [];
  let raycastVisibleObjectCount = 0;

  for (const [objectId, proxy] of proxyById.entries()) {
    const metadata = materializeSnapshotObjectRecord(
      objectMetadataById.get(objectId) ?? null,
      proxy,
      state.roomCenterMm.x,
      state.roomCenterMm.y
    );
    if (!metadata || !proxy) continue;

    const box = new THREE.Box3().setFromObject(proxy);
    if (box.isEmpty() || !frustum.intersectsBox(box)) {
      continue;
    }

    const samplePoints = getVisibilitySamplePoints(THREE, box);
    const boxCenter = box.getCenter(new THREE.Vector3());
    const screenCenterPx = projectWorldPointToCanvasPx(
      boxCenter,
      camera,
      canvasWidth,
      canvasHeight
    );
    const screenBboxPx = computeScreenBboxPx({
      THREE,
      box,
      camera,
      canvasWidth,
      canvasHeight,
    });
    let visibleSampleCount = 0;

    for (const point of samplePoints) {
      const visible = isSamplePointVisible({
        THREE,
        raycaster,
        cameraWorldPosition,
        samplePoint: point,
        camera,
        objectGroup,
        wallGroup,
        expectedObjectId: objectId,
        wallOcclusionOpacityThreshold: state.wallOcclusionOpacityThreshold,
      });
      if (visible) {
        visibleSampleCount += 1;
      }
    }

    if (!screenBboxPx || screenBboxPx.width <= 0 || screenBboxPx.height <= 0) {
      continue;
    }

    const visibleObject: VisibleSnapshotObject = {
      ...cloneSnapshotObjectRecord(metadata),
      distanceToCameraM: roundTo(cameraWorldPosition.distanceTo(boxCenter), 3),
      screenCenterPx,
      screenBboxPx,
      visibleSampleCount,
      visibleSampleFraction: roundTo(visibleSampleCount / samplePoints.length, 3),
      visibilitySource: visibleSampleCount > 0 ? 'raycast' : 'projected_visibility',
    };
    visibleCandidates.push(visibleObject);
    if (visibleSampleCount > 0) {
      raycastVisibleObjectCount += 1;
      visibleObjects.push(visibleObject);
    }
  }

  let projectedFallbackObjectCount = 0;
  if (visibleObjects.length === 0 && visibleCandidates.length > 0) {
    visibleObjects.push(...visibleCandidates);
    projectedFallbackObjectCount = visibleCandidates.length;
  }

  visibleObjects.sort((left, right) => left.distanceToCameraM - right.distanceToCameraM);

  const allObjects = Array.from(objectMetadataById.entries())
    .map(([objectId, item]) =>
      materializeSnapshotObjectRecord(
        item,
        proxyById.get(objectId),
        state.roomCenterMm.x,
        state.roomCenterMm.y
      )
    )
    .filter((item): item is SnapshotObjectRecord => item !== null);
  const openingScreenLocks = buildOpeningScreenLocks({
    THREE,
    openingsGroup,
    camera,
    canvasWidth,
    canvasHeight,
  });
  const openingIdsMissingBbox: string[] = [];
  for (const item of state.roomOpenings.doors) {
    if (!openingScreenLocks.has(`door:${item.id}`)) {
      openingIdsMissingBbox.push(`door:${item.id}`);
    }
  }
  for (const item of state.roomOpenings.windows) {
    if (!openingScreenLocks.has(`window:${item.id}`)) {
      openingIdsMissingBbox.push(`window:${item.id}`);
    }
  }
  const visibilityReasons: string[] = [];
  if (projectedFallbackObjectCount > 0) {
    visibilityReasons.push('raycast_occlusion_returned_no_visible_objects');
  }
  if (openingIdsMissingBbox.length > 0) {
    visibilityReasons.push('opening_screen_bbox_missing');
  }
  const visibilityDiagnostics = {
    status: visibilityReasons.length > 0 ? 'degraded' as const : 'ok' as const,
    allObjectCount: allObjects.length,
    visibleObjectCount: visibleObjects.length,
    raycastVisibleObjectCount,
    projectedFallbackObjectCount,
    openingBboxMissingCount: openingIdsMissingBbox.length,
    openingIdsMissingBbox,
    reasons: visibilityReasons,
  };

  return {
    capturedAt: new Date().toISOString(),
    variantId: state.variantId,
    selection: {
      selectedObjectId: state.selectedObjectId,
      selectedWallId: state.selectedWallId,
      editMode: state.editMode,
    },
    canvas: {
      widthPx: canvasWidth,
      heightPx: canvasHeight,
    },
    room: {
      boundsMm: { ...state.roomBoundsMm },
      polygonMm: state.roomPolygonMm.map((point) => ({ ...point })),
      surfaceColors: { ...state.roomSurfaceColors },
      openings: {
        doors: state.roomOpenings.doors.map((item) => ({
          id: item.id,
          segmentMm: item.segmentMm.map((point) => ({ ...point })) as [
            SnapshotPointMm,
            SnapshotPointMm,
          ],
          screenCenterPx: openingScreenLocks.get(`door:${item.id}`)?.center ?? null,
          screenBboxPx: openingScreenLocks.get(`door:${item.id}`)?.bbox ?? null,
        })),
        windows: state.roomOpenings.windows.map((item) => ({
          id: item.id,
          segmentMm: item.segmentMm.map((point) => ({ ...point })) as [
            SnapshotPointMm,
            SnapshotPointMm,
          ],
          screenCenterPx: openingScreenLocks.get(`window:${item.id}`)?.center ?? null,
          screenBboxPx: openingScreenLocks.get(`window:${item.id}`)?.bbox ?? null,
        })),
      },
    },
    camera: {
      positionM: vectorToPlain(cameraWorldPosition),
      positionRoomMm: sceneVectorToRoomMm(
        cameraWorldPosition,
        state.roomCenterMm.x,
        state.roomCenterMm.y
      ),
      targetM: targetVector,
      targetRoomMm: targetVector
        ? sceneVectorToRoomMm(
            state.controls.target,
            state.roomCenterMm.x,
            state.roomCenterMm.y
          )
        : null,
      fovDeg: roundTo(Number(camera.fov ?? 0), 3),
      aspect: roundTo(Number(camera.aspect ?? 0), 4),
      near: roundTo(Number(camera.near ?? 0), 4),
      far: roundTo(Number(camera.far ?? 0), 3),
    },
    totalObjectCount: allObjects.length,
    visibleObjectCount: visibleObjects.length,
    hiddenObjectCount: Math.max(0, allObjects.length - visibleObjects.length),
    visibilityMethod:
      'frustum intersection plus center/corner occlusion sampling against current object meshes and walls above the opacity threshold, excluding render-only helpers',
    visibilityDiagnostics,
    wallOcclusionOpacityThreshold: state.wallOcclusionOpacityThreshold,
    allObjects,
    visibleObjects,
  };
}

function buildOpeningScreenLocks(params: {
  THREE: any;
  openingsGroup: any | null;
  camera: any;
  canvasWidth: number;
  canvasHeight: number;
}): Map<string, { bbox: SnapshotScreenBboxPx; center: { x: number; y: number } | null }> {
  const { THREE, openingsGroup, camera, canvasWidth, canvasHeight } = params;
  const locks = new Map<
    string,
    { bbox: SnapshotScreenBboxPx; center: { x: number; y: number } | null }
  >();
  if (!openingsGroup) {
    return locks;
  }

  openingsGroup.children?.forEach?.((openingRoot: any) => {
    const openingId = String(openingRoot?.userData?.openingId ?? '').trim();
    const openingType = String(openingRoot?.userData?.openingType ?? '').trim();
    if (!openingId || !openingType) {
      return;
    }
    const box = new THREE.Box3().setFromObject(openingRoot);
    if (box.isEmpty()) {
      return;
    }
    const bbox = computeScreenBboxPx({
      THREE,
      box,
      camera,
      canvasWidth,
      canvasHeight,
    });
    if (!bbox) {
      return;
    }
    const center = projectWorldPointToCanvasPx(
      box.getCenter(new THREE.Vector3()),
      camera,
      canvasWidth,
      canvasHeight
    );
    locks.set(`${openingType}:${openingId}`, { bbox, center });
  });
  return locks;
}

function cloneSnapshotObjectRecord(item: SnapshotObjectRecord): SnapshotObjectRecord {
  return {
    ...item,
    bboxMm: { ...item.bboxMm },
    planPositionMm: { ...item.planPositionMm },
    dimensionsMm: { ...item.dimensionsMm },
    placeOn: item.placeOn ? { ...item.placeOn } : null,
  };
}

function materializeSnapshotObjectRecord(
  item: SnapshotObjectRecord | null,
  proxy: any,
  centerXmm: number,
  centerYmm: number
): SnapshotObjectRecord | null {
  if (!item) {
    return null;
  }
  const cloned = cloneSnapshotObjectRecord(item);
  if (!proxy) {
    return cloned;
  }

  const widthMm = Math.max(0, Number(cloned.dimensionsMm.width ?? 0));
  const depthMm = Math.max(0, Number(cloned.dimensionsMm.depth ?? 0));
  const heightMm = Math.max(0, Number(cloned.dimensionsMm.height ?? 0));
  const sceneYawDeg = normalizeDegrees((Number(proxy.rotation?.y ?? 0) * 180) / Math.PI);
  const planRotationDeg = sceneYawDeg;
  const rotationRad = (planRotationDeg * Math.PI) / 180;
  const cos = Math.abs(Math.cos(rotationRad));
  const sin = Math.abs(Math.sin(rotationRad));
  const worldWidthMm = widthMm * cos + depthMm * sin;
  const worldDepthMm = widthMm * sin + depthMm * cos;
  const positionRoomMm = sceneVectorToRoomMm(proxy.position, centerXmm, centerYmm);

  cloned.planPositionMm = {
    x: positionRoomMm.x,
    y: positionRoomMm.y,
    z: roundTo(Number(proxy.position?.y ?? 0) * MM_PER_M + heightMm / 2, 1),
  };
  cloned.planRotationDeg = roundTo(planRotationDeg, 2);
  cloned.sceneYawDeg = roundTo(sceneYawDeg, 2);
  cloned.bboxMm = {
    minX: roundTo(positionRoomMm.x - worldWidthMm / 2, 1),
    minY: roundTo(positionRoomMm.y - worldDepthMm / 2, 1),
    maxX: roundTo(positionRoomMm.x + worldWidthMm / 2, 1),
    maxY: roundTo(positionRoomMm.y + worldDepthMm / 2, 1),
  };
  return cloned;
}

function getVisibilitySamplePoints(THREE: any, box: any): any[] {
  const min = box.min;
  const max = box.max;
  const center = box.getCenter(new THREE.Vector3());
  return [
    center,
    new THREE.Vector3(min.x, min.y, min.z),
    new THREE.Vector3(min.x, min.y, max.z),
    new THREE.Vector3(min.x, max.y, min.z),
    new THREE.Vector3(min.x, max.y, max.z),
    new THREE.Vector3(max.x, min.y, min.z),
    new THREE.Vector3(max.x, min.y, max.z),
    new THREE.Vector3(max.x, max.y, min.z),
    new THREE.Vector3(max.x, max.y, max.z),
  ];
}

function isSamplePointVisible(params: {
  THREE: any;
  raycaster: any;
  cameraWorldPosition: any;
  samplePoint: any;
  camera: any;
  objectGroup: any;
  wallGroup: any | null;
  expectedObjectId: string;
  wallOcclusionOpacityThreshold: number;
}): boolean {
  const {
    raycaster,
    cameraWorldPosition,
    samplePoint,
    camera,
    objectGroup,
    wallGroup,
    expectedObjectId,
    wallOcclusionOpacityThreshold,
  } = params;
  const ndc = samplePoint.clone().project(camera);
  if (
    !Number.isFinite(ndc.x) ||
    !Number.isFinite(ndc.y) ||
    !Number.isFinite(ndc.z) ||
    ndc.z < -1 ||
    ndc.z > 1
  ) {
    return false;
  }

  const direction = samplePoint.clone().sub(cameraWorldPosition);
  const distance = direction.length();
  if (!Number.isFinite(distance) || distance <= 1e-6) {
    return false;
  }
  direction.normalize();
  raycaster.set(cameraWorldPosition, direction);
  raycaster.far = distance + OCCLUSION_TOLERANCE_M;
  const raycastTargets = wallGroup
    ? [...objectGroup.children, ...wallGroup.children]
    : objectGroup.children;
  const intersections = raycaster.intersectObjects(raycastTargets, true);
  if (!intersections.length) {
    return false;
  }
  for (const intersection of intersections) {
    const hitObject = intersection.object;
    if (
      intersection.distance <= OCCLUSION_TOLERANCE_M ||
      isSnapshotRaycastHelper(hitObject)
    ) {
      continue;
    }
    const hitRoot = findObjectRoot(hitObject);
    if (String(hitRoot?.userData?.objectId ?? '') === expectedObjectId) {
      return true;
    }
    const wallMesh = findWallMesh(hitObject);
    if (wallMesh) {
      const opacity = clamp(
        Number(
          wallMesh.userData?.wallOpacity ??
            wallMesh.material?.opacity ??
            1
        ),
        0,
        1
      );
      if (opacity >= wallOcclusionOpacityThreshold) {
        return false;
      }
      continue;
    }
    if (hitRoot?.userData?.objectId) {
      return false;
    }
  }
  return false;
}

function isSnapshotRaycastHelper(node: any): boolean {
  let current = node;
  while (current) {
    if (
      current.userData?.aiRenderHide === true ||
      current.userData?.aiStructuralEdge === true ||
      current.userData?.structuralEdgeOverlay === true
    ) {
      return true;
    }
    current = current.parent ?? null;
  }
  return false;
}

function findObjectRoot(node: any): any | null {
  let current = node;
  while (current) {
    if (typeof current.userData?.objectId === 'string' && current.userData.objectId) {
      return current;
    }
    current = current.parent ?? null;
  }
  return null;
}

function findWallMesh(node: any): any | null {
  let current = node;
  while (current) {
    if (current.userData?.wallMesh) {
      return current;
    }
    current = current.parent ?? null;
  }
  return null;
}

function computeScreenBboxPx(params: {
  THREE: any;
  box: any;
  camera: any;
  canvasWidth: number;
  canvasHeight: number;
}): SnapshotScreenBboxPx | null {
  const { THREE, box, camera, canvasWidth, canvasHeight } = params;
  const points = getVisibilitySamplePoints(THREE, box)
    .map((point) => projectWorldPointToCanvasPx(point, camera, canvasWidth, canvasHeight))
    .filter(
      (
        point
      ): point is {
        x: number;
        y: number;
      } => point !== null
    );

  if (!points.length) {
    return null;
  }

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const point of points) {
    minX = Math.min(minX, point.x);
    minY = Math.min(minY, point.y);
    maxX = Math.max(maxX, point.x);
    maxY = Math.max(maxY, point.y);
  }

  if (!Number.isFinite(minX) || !Number.isFinite(minY)) {
    return null;
  }

  const boundedMinX = clamp(Math.round(minX), 0, canvasWidth);
  const boundedMinY = clamp(Math.round(minY), 0, canvasHeight);
  const boundedMaxX = clamp(Math.round(maxX), 0, canvasWidth);
  const boundedMaxY = clamp(Math.round(maxY), 0, canvasHeight);

  return {
    minX: boundedMinX,
    minY: boundedMinY,
    maxX: boundedMaxX,
    maxY: boundedMaxY,
    width: Math.max(0, boundedMaxX - boundedMinX),
    height: Math.max(0, boundedMaxY - boundedMinY),
  };
}

function projectWorldPointToCanvasPx(
  point: any,
  camera: any,
  canvasWidth: number,
  canvasHeight: number
): { x: number; y: number } | null {
  const projected = point.clone().project(camera);
  if (
    !Number.isFinite(projected.x) ||
    !Number.isFinite(projected.y) ||
    !Number.isFinite(projected.z)
  ) {
    return null;
  }
  return {
    x: roundTo((projected.x * 0.5 + 0.5) * canvasWidth, 1),
    y: roundTo((-projected.y * 0.5 + 0.5) * canvasHeight, 1),
  };
}

function sceneVectorToRoomMm(
  vector: { x: number; y: number; z: number },
  centerXmm: number,
  centerYmm: number
): SnapshotVector3 {
  return {
    x: roundTo(centerXmm + vector.x * MM_PER_M, 1),
    y: roundTo(centerYmm - vector.z * MM_PER_M, 1),
    z: roundTo(vector.y * MM_PER_M, 1),
  };
}

function normalizeDegrees(value: number): number {
  const normalized = Number.isFinite(value) ? value % 360 : 0;
  return (normalized + 360) % 360;
}

function vectorToPlain(vector: { x: number; y: number; z: number }): SnapshotVector3 {
  return {
    x: roundTo(vector.x, 4),
    y: roundTo(vector.y, 4),
    z: roundTo(vector.z, 4),
  };
}

function roundTo(value: number, digits: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
