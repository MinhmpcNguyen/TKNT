import type { InventoryItem } from '../types';

export type InventoryModelPreviewCalibration = {
  version: number;
  status: 'calibrated' | 'needs_review' | 'failed';
  confidence: number;
  boundsMode: string | null;
  fitPolicy: string | null;
  floorAnchor: string | null;
  rotationLabel: string | null;
  rotationMatrix: number[][] | null;
  axisAssignment:
    | {
        width: string;
        height: string;
        depth: string;
      }
    | null;
  baseSizeM:
    | {
        width: number;
        height: number;
        depth: number;
      }
    | null;
  pivotOffsetM:
    | {
        x: number;
        y: number;
        z: number;
      }
    | null;
};

export type GlbFitStrategy =
  | 'box'
  | 'footprint-uniform-or-proxy'
  | 'footprint'
  | 'wall-plane'
  | 'wall-plane-uniform-or-proxy'
  | 'footprint-uniform'
  | 'wall-plane-uniform';

export type InventoryModelPreviewOverride = {
  enabled: boolean;
  contexts: string[];
  fitStrategy: GlbFitStrategy | null;
  scaleMultiplier: number | null;
  rotationDegOffset: number;
  dimensionScale:
    | {
        width: number | null;
        height: number | null;
        depth: number | null;
      }
    | null;
  notes: string | null;
};

export type InventoryModelDescriptor = {
  url: string;
  source: string;
  fit: 'contain' | 'cover';
  scale: number;
  rotationDeg: number;
  previewCalibration: InventoryModelPreviewCalibration | null;
  previewOverride: InventoryModelPreviewOverride | null;
};

export type InventoryModelDescriptorResolution = {
  descriptor: InventoryModelDescriptor | null;
  status:
    | 'resolved_style_variant'
    | 'resolved_direct_model'
    | 'resolved_nested_model'
    | 'resolved_file_list_model'
    | 'missing_style_variant'
    | 'missing_model_metadata'
    | 'invalid_attributes';
  reason: string;
  availableStyles: string[];
};

function resolveActivePreviewCalibration(
  calibration: InventoryModelPreviewCalibration | null
): InventoryModelPreviewCalibration | null {
  if (!calibration) {
    return null;
  }
  // Only trust fully calibrated preview metadata at render time.
  // Low-confidence "needs_review" calibrations have produced incorrect
  // axis flips in 3D, so we fall back to raw GLB + runtime fit instead.
  if (calibration.status !== 'calibrated') {
    return null;
  }
  return calibration;
}

type GltfLoaderConstructor = new (manager?: unknown) => {
  loadAsync(url: string): Promise<{
    scene?: unknown;
    scenes?: unknown[];
  }>;
};

type BuildGlbPreviewModelParams = {
  THREE: any;
  GLTFLoader: GltfLoaderConstructor;
  descriptor: InventoryModelDescriptor;
  wM: number;
  dM: number;
  hM: number;
  fitStrategy?: GlbFitStrategy;
  colorHex?: string | null;
  preserveMaterialTextures?: boolean;
  minimumHorizontalAspectRatio?: number | null;
};

const DIRECT_MODEL_KEYS = [
  'model_url',
  'modelUrl',
  'glb_url',
  'glbUrl',
  'gltf_url',
  'gltfUrl',
  'model_path',
  'modelPath',
  'glb_path',
  'glbPath',
  'gltf_path',
  'gltfPath',
  'three_model_url',
  'threeModelUrl',
  'preview_model_url',
  'previewModelUrl',
];

const NESTED_MODEL_KEYS = [
  'model',
  'model_3d',
  'model3d',
  'three_model',
  'threeModel',
  'preview_model',
  'previewModel',
  'glb',
  'gltf',
];

const URL_VALUE_KEYS = [
  'url',
  'href',
  'src',
  'path',
  'model_url',
  'modelUrl',
  'glb_url',
  'glbUrl',
  'gltf_url',
  'gltfUrl',
  'model_path',
  'modelPath',
  'glb_path',
  'glbPath',
  'gltf_path',
  'gltfPath',
];

const SCALE_VALUE_KEYS = ['scale', 'preview_scale', 'previewScale', 'model_scale', 'modelScale'];
const ROTATION_VALUE_KEYS = [
  'rotation_deg',
  'rotationDeg',
  'preview_rotation_deg',
  'previewRotationDeg',
  'yaw_deg',
  'yawDeg',
];
const FIT_VALUE_KEYS = ['fit', 'preview_fit', 'previewFit'];
const MODEL_FILE_LIST_KEYS = ['files', 'asset_files', 'model_files', 'models'];
const MODEL_VARIANT_KEYS = ['model_variants', 'modelVariants'];

const gltfPromiseCache = new Map<string, Promise<any>>();
const PROXY_BOX_ASPECT_FALLBACK_THRESHOLD = 1.25;

function computePreviewBounds(THREE: any, root: any): any {
  const meshEntries: Array<{
    box: any;
    center: any;
    volume: number;
    maxDim: number;
  }> = [];

  root?.traverse?.((node: any) => {
    if (!node?.isMesh || !node.visible || !node.geometry) {
      return;
    }
    if (typeof node.geometry.computeBoundingBox === 'function' && !node.geometry.boundingBox) {
      node.geometry.computeBoundingBox();
    }
    const geometryBox = node.geometry.boundingBox;
    if (!geometryBox?.clone) {
      return;
    }

    const worldBox = geometryBox.clone().applyMatrix4(node.matrixWorld);
    if (typeof worldBox.isEmpty === 'function' && worldBox.isEmpty()) {
      return;
    }

    const size = worldBox.getSize(new THREE.Vector3());
    const volume = Math.max(size.x * size.y * size.z, 1e-9);
    meshEntries.push({
      box: worldBox,
      center: worldBox.getCenter(new THREE.Vector3()),
      volume,
      maxDim: Math.max(size.x, size.y, size.z),
    });
  });

  if (meshEntries.length === 0) {
    return new THREE.Box3().setFromObject(root);
  }

  meshEntries.sort((left, right) => right.volume - left.volume);
  const dominantVolume = meshEntries[0].volume;
  const dominantMaxDim = meshEntries[0].maxDim;
  const seededEntries = meshEntries.filter(
    (entry, index) => index < 12 || entry.volume >= dominantVolume * 0.01
  );

  const weightedCenter = new THREE.Vector3();
  let totalWeight = 0;
  for (const entry of seededEntries) {
    weightedCenter.add(entry.center.clone().multiplyScalar(entry.volume));
    totalWeight += entry.volume;
  }
  if (totalWeight > 0) {
    weightedCenter.multiplyScalar(1 / totalWeight);
  }

  const distanceLimit = Math.max(dominantMaxDim * 3, 0.25);
  const keptEntries = seededEntries.filter(
    (entry, index) => index === 0 || entry.center.distanceTo(weightedCenter) <= distanceLimit
  );
  const finalEntries = keptEntries.length > 0 ? keptEntries : seededEntries;
  const mergedBox = finalEntries[0].box.clone();
  for (const entry of finalEntries.slice(1)) {
    mergedBox.union(entry.box);
  }
  return mergedBox;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function readString(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null;
}

function readNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return null;
}

function applyPreviewColorTint(
  THREE: any,
  root: any,
  colorHex: string | null | undefined,
  options?: { preserveIntrinsicMaterialColor?: boolean }
): void {
  const rawColor = readString(colorHex);
  if (!rawColor) {
    return;
  }
  const targetColor = new THREE.Color(rawColor);
  root?.traverse?.((node: any) => {
    if (!node?.isMesh || !node.material) {
      return;
    }
    const materials = Array.isArray(node.material) ? node.material : [node.material];
    const tintedMaterials = materials.map((material: any) => {
      if (!material?.clone) {
        return material;
      }
      const tinted = material.clone();
      const hasIntrinsicMaterialColor =
        ('map' in tinted && Boolean(tinted.map)) || Boolean(tinted.vertexColors);
      if (options?.preserveIntrinsicMaterialColor && hasIntrinsicMaterialColor) {
        tinted.needsUpdate = true;
        return tinted;
      }
      if (tinted.color?.isColor) {
        tinted.color.copy(targetColor);
      }
      if ('map' in tinted && tinted.map) {
        tinted.map = null;
      }
      if ('vertexColors' in tinted && tinted.vertexColors) {
        tinted.vertexColors = false;
      }
      if (tinted.emissive?.isColor) {
        tinted.emissive.copy(targetColor).multiplyScalar(0.05);
        if (typeof tinted.emissiveIntensity === 'number') {
          tinted.emissiveIntensity = Math.max(tinted.emissiveIntensity, 0.05);
        }
      }
      tinted.needsUpdate = true;
      return tinted;
    });
    node.material = Array.isArray(node.material) ? tintedMaterials : tintedMaterials[0];
  });
}

function readFit(value: unknown): 'contain' | 'cover' | null {
  const raw = readString(value)?.toLowerCase();
  if (raw === 'contain' || raw === 'cover') {
    return raw;
  }
  return null;
}

function canonicalizeStyleKey(value: unknown): string | null {
  const raw = readString(value);
  if (!raw) {
    return null;
  }
  return raw.toLowerCase().replace(/\s+/g, ' ').trim();
}

function resolveUrlFromRecord(record: Record<string, unknown>): { url: string; source: string } | null {
  for (const key of URL_VALUE_KEYS) {
    const url = readString(record[key]);
    if (url) {
      return { url, source: key };
    }
  }
  return null;
}

function resolveUrlFromFileEntries(
  value: unknown
): { url: string; source: string } | null {
  if (!Array.isArray(value)) {
    return null;
  }

  const candidates: Array<{ url: string; source: string; priority: number }> = [];
  for (const entry of value) {
    if (!isRecord(entry)) {
      continue;
    }
    const candidate =
      resolveUrlFromRecord(entry) ??
      (() => {
        const rawUrl = readString(entry.url) ?? readString(entry.href) ?? readString(entry.src);
        return rawUrl ? { url: rawUrl, source: 'url' } : null;
      })();
    if (!candidate) {
      continue;
    }

    const fileKind = readString(entry.file_kind)?.toLowerCase();
    const role = readString(entry.role)?.toLowerCase();
    const mime = readString(entry.mime)?.toLowerCase();
    const urlLower = candidate.url.toLowerCase();
    const looksLikeModel =
      urlLower.endsWith('.glb') ||
      urlLower.endsWith('.gltf') ||
      fileKind === 'model' ||
      fileKind === 'glb' ||
      fileKind === 'gltf' ||
      role === 'model' ||
      role === '3d' ||
      mime === 'model/gltf-binary' ||
      mime === 'model/gltf+json';
    if (looksLikeModel) {
      candidates.push({
        url: candidate.url,
        source: candidate.source,
        priority: scoreModelFileEntry({ fileKind, role, mime, urlLower }),
      });
    }
  }

  if (candidates.length === 0) {
    return null;
  }

  candidates.sort((left, right) => left.priority - right.priority);
  const preferred = candidates[0];
  return { url: preferred.url, source: preferred.source };
}

function scoreModelFileEntry(params: {
  fileKind: string | null;
  role: string | null;
  mime: string | null;
  urlLower: string;
}): number {
  const { fileKind, role, mime, urlLower } = params;
  if (role === 'model') {
    return 0;
  }
  if (role === 'model_preferred' || role === 'model_textured') {
    return 1;
  }
  if (role === '3d') {
    return 2;
  }
  if (fileKind === 'model' && mime === 'model/gltf-binary') {
    return 5;
  }
  if (fileKind === 'model' && mime === 'model/gltf+json') {
    return 6;
  }
  if (fileKind === 'model' || fileKind === 'glb' || fileKind === 'gltf') {
    return 8;
  }
  if (role === 'model_source') {
    return 20;
  }
  if (urlLower.endsWith('.glb')) {
    return 25;
  }
  if (urlLower.endsWith('.gltf')) {
    return 26;
  }
  return 50;
}

function resolveScalarFromRecord(record: Record<string, unknown>, keys: string[]): number | null {
  for (const key of keys) {
    const value = readNumber(record[key]);
    if (value !== null) {
      return value;
    }
  }
  return null;
}

function resolveFitFromRecord(record: Record<string, unknown>): 'contain' | 'cover' | null {
  for (const key of FIT_VALUE_KEYS) {
    const fit = readFit(record[key]);
    if (fit) {
      return fit;
    }
  }
  return null;
}

function readCalibrationMatrix(value: unknown): number[][] | null {
  if (!Array.isArray(value) || value.length !== 4) {
    return null;
  }
  const rows = value.map((row) => {
    if (!Array.isArray(row) || row.length !== 4) {
      return null;
    }
    const numbers = row.map((cell) => readNumber(cell));
    return numbers.every((cell) => cell !== null) ? (numbers as number[]) : null;
  });
  return rows.every((row) => row !== null) ? (rows as number[][]) : null;
}

function readCalibrationAxisAssignment(
  value: unknown
): InventoryModelPreviewCalibration['axisAssignment'] {
  if (!isRecord(value)) {
    return null;
  }
  const width = readString(value.width);
  const height = readString(value.height);
  const depth = readString(value.depth);
  if (!width || !height || !depth) {
    return null;
  }
  return { width, height, depth };
}

function readCalibrationSizeRecord(
  value: unknown,
  widthKey = 'width',
  heightKey = 'height',
  depthKey = 'depth'
): { width: number; height: number; depth: number } | null {
  if (!isRecord(value)) {
    return null;
  }
  const width = readNumber(value[widthKey]);
  const height = readNumber(value[heightKey]);
  const depth = readNumber(value[depthKey]);
  if (
    width === null ||
    height === null ||
    depth === null ||
    width <= 0 ||
    height <= 0 ||
    depth <= 0
  ) {
    return null;
  }
  return { width, height, depth };
}

function readCalibrationOffsetRecord(
  value: unknown
): InventoryModelPreviewCalibration['pivotOffsetM'] {
  if (!isRecord(value)) {
    return null;
  }
  const x = readNumber(value.x);
  const y = readNumber(value.y);
  const z = readNumber(value.z);
  if (x === null || y === null || z === null) {
    return null;
  }
  return { x, y, z };
}

function readStringList(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((entry) => readString(entry))
    .filter((entry): entry is string => Boolean(entry));
}

function readFitStrategy(value: unknown): GlbFitStrategy | null {
  const raw = readString(value);
  if (!raw) {
    return null;
  }
  const supported: GlbFitStrategy[] = [
    'box',
    'footprint-uniform-or-proxy',
    'footprint',
    'wall-plane',
    'wall-plane-uniform-or-proxy',
    'footprint-uniform',
    'wall-plane-uniform',
  ];
  return supported.includes(raw as GlbFitStrategy) ? (raw as GlbFitStrategy) : null;
}

function resolvePreviewOverride(
  record: Record<string, unknown> | null
): InventoryModelPreviewOverride | null {
  if (!record) {
    return null;
  }
  const rawOverride = record.preview_override ?? record.previewOverride;
  if (!isRecord(rawOverride)) {
    return null;
  }

  const contexts = [
    ...readStringList(rawOverride.contexts),
    ...readStringList(rawOverride.apply_to),
    ...readStringList(rawOverride.applyTo),
  ]
    .map((entry) => entry.toLowerCase())
    .filter((entry, index, array) => array.indexOf(entry) === index);
  const dimensionScaleRaw = rawOverride.dimension_scale ?? rawOverride.dimensionScale;
  const widthScale = isRecord(dimensionScaleRaw)
    ? readNumber(dimensionScaleRaw.width)
    : null;
  const heightScale = isRecord(dimensionScaleRaw)
    ? readNumber(dimensionScaleRaw.height)
    : null;
  const depthScale = isRecord(dimensionScaleRaw)
    ? readNumber(dimensionScaleRaw.depth)
    : null;

  return {
    enabled: rawOverride.enabled !== false,
    contexts,
    fitStrategy:
      readFitStrategy(rawOverride.fit_strategy) ?? readFitStrategy(rawOverride.fitStrategy),
    scaleMultiplier:
      readNumber(rawOverride.scale_multiplier) ?? readNumber(rawOverride.scaleMultiplier),
    rotationDegOffset:
      readNumber(rawOverride.rotation_deg_offset) ??
      readNumber(rawOverride.rotationDegOffset) ??
      0,
    dimensionScale:
      widthScale !== null || heightScale !== null || depthScale !== null
        ? {
            width: widthScale,
            height: heightScale,
            depth: depthScale,
          }
        : null,
    notes: readString(rawOverride.notes),
  };
}

function resolvePreviewCalibration(
  record: Record<string, unknown> | null
): InventoryModelPreviewCalibration | null {
  if (!record) {
    return null;
  }
  const rawCalibration = record.preview_calibration ?? record.previewCalibration;
  if (!isRecord(rawCalibration)) {
    return null;
  }

  const statusRaw = readString(rawCalibration.status)?.toLowerCase();
  const status =
    statusRaw === 'calibrated' || statusRaw === 'needs_review' || statusRaw === 'failed'
      ? statusRaw
      : 'calibrated';
  const version = readNumber(rawCalibration.version) ?? 1;
  const confidence = readNumber(rawCalibration.confidence) ?? 0;
  const baseSizeM =
    readCalibrationSizeRecord(rawCalibration.base_size_m) ??
    readCalibrationSizeRecord(rawCalibration.baseSizeM);
  const pivotOffsetM =
    readCalibrationOffsetRecord(rawCalibration.pivot_offset_m) ??
    readCalibrationOffsetRecord(rawCalibration.pivotOffsetM);
  const rotationMatrix =
    readCalibrationMatrix(rawCalibration.rotation_matrix) ??
    readCalibrationMatrix(rawCalibration.rotationMatrix);

  return {
    version,
    status,
    confidence,
    boundsMode: readString(rawCalibration.bounds_mode) ?? readString(rawCalibration.boundsMode),
    fitPolicy: readString(rawCalibration.fit_policy) ?? readString(rawCalibration.fitPolicy),
    floorAnchor: readString(rawCalibration.floor_anchor) ?? readString(rawCalibration.floorAnchor),
    rotationLabel:
      readString(rawCalibration.rotation_label) ?? readString(rawCalibration.rotationLabel),
    rotationMatrix,
    axisAssignment:
      readCalibrationAxisAssignment(rawCalibration.axis_assignment) ??
      readCalibrationAxisAssignment(rawCalibration.axisAssignment),
    baseSizeM,
    pivotOffsetM,
  };
}

function resolveVariantModelEntry(
  attributes: Record<string, unknown>,
  preferredStyle: string | null | undefined
): { match: { url: string; source: string }; metadata: Record<string, unknown> | null } | null {
  const styleKey = canonicalizeStyleKey(preferredStyle);
  if (!styleKey) {
    return null;
  }

  for (const variantsKey of MODEL_VARIANT_KEYS) {
    const variants = attributes[variantsKey];
    if (!isRecord(variants)) {
      continue;
    }

    for (const [candidateKey, candidateValue] of Object.entries(variants)) {
      if (canonicalizeStyleKey(candidateKey) !== styleKey) {
        continue;
      }
      if (typeof candidateValue === 'string' && candidateValue.trim()) {
        return {
          match: { url: candidateValue.trim(), source: `${variantsKey}.${candidateKey}` },
          metadata: null,
        };
      }
      if (isRecord(candidateValue)) {
        const nested = resolveUrlFromRecord(candidateValue);
        if (nested) {
          return {
            match: {
              url: nested.url,
              source: `${variantsKey}.${candidateKey}.${nested.source}`,
            },
            metadata: candidateValue,
          };
        }
      }
    }
  }

  return null;
}

function resolveMatchingModelMetadataRecord(
  attributes: Record<string, unknown>,
  url: string
): Record<string, unknown> | null {
  for (const key of NESTED_MODEL_KEYS) {
    const candidate = attributes[key];
    if (!isRecord(candidate)) {
      continue;
    }
    const nested = resolveUrlFromRecord(candidate);
    if (nested?.url === url) {
      return candidate;
    }
  }
  return null;
}

export function resolveInventoryModelDescriptor(
  item: InventoryItem | null | undefined,
  preferredStyle?: string | null
): InventoryModelDescriptor | null {
  return inspectInventoryModelDescriptor(item, preferredStyle).descriptor;
}

export function inspectInventoryModelDescriptor(
  item: InventoryItem | null | undefined,
  preferredStyle?: string | null
): InventoryModelDescriptorResolution {
  const attributes = item?.attributes;
  if (!isRecord(attributes)) {
    return {
      descriptor: null,
      status: 'invalid_attributes',
      reason: 'Inventory item has no readable attributes payload for 3D model lookup.',
      availableStyles: [],
    };
  }

  let match: { url: string; source: string } | null = null;
  let metadata: Record<string, unknown> | null = null;
  let resolvedStatus: InventoryModelDescriptorResolution['status'] | null = null;
  const availableStyles = new Set<string>();
  for (const variantsKey of MODEL_VARIANT_KEYS) {
    const variants = attributes[variantsKey];
    if (!isRecord(variants)) {
      continue;
    }
    for (const key of Object.keys(variants)) {
      const normalizedKey = canonicalizeStyleKey(key);
      if (normalizedKey) {
        availableStyles.add(normalizedKey);
      }
    }
  }

  const variantMatch = resolveVariantModelEntry(attributes, preferredStyle);
  if (variantMatch) {
    match = variantMatch.match;
    metadata = variantMatch.metadata;
    resolvedStatus = 'resolved_style_variant';
  }

  if (!match) {
    for (const key of DIRECT_MODEL_KEYS) {
      const url = readString(attributes[key]);
      if (url) {
        match = { url, source: key };
        resolvedStatus = 'resolved_direct_model';
        break;
      }
    }
  }

  if (!match) {
    for (const key of NESTED_MODEL_KEYS) {
      const candidate = attributes[key];
      if (typeof candidate === 'string' && candidate.trim()) {
        match = { url: candidate.trim(), source: key };
        resolvedStatus = 'resolved_nested_model';
        break;
      }
      if (isRecord(candidate)) {
        const nested = resolveUrlFromRecord(candidate);
        if (nested) {
          match = nested;
          metadata = candidate;
          resolvedStatus = 'resolved_nested_model';
          break;
        }
      }
    }
  }

  if (match && !metadata) {
    metadata = resolveMatchingModelMetadataRecord(attributes, match.url);
  }

  if (!match) {
    for (const key of MODEL_FILE_LIST_KEYS) {
      const fromFiles = resolveUrlFromFileEntries(attributes[key]);
      if (fromFiles) {
        match = fromFiles;
        resolvedStatus = 'resolved_file_list_model';
        break;
      }
    }
  }

  if (!match) {
    const styleKey = canonicalizeStyleKey(preferredStyle);
    const sortedAvailableStyles = Array.from(availableStyles).sort();
    if (styleKey && sortedAvailableStyles.length > 0) {
      return {
        descriptor: null,
        status: 'missing_style_variant',
        reason: `No GLB metadata matched style "${styleKey}". Available model variants: ${sortedAvailableStyles.join(
          ', '
        )}.`,
        availableStyles: sortedAvailableStyles,
      };
    }
    return {
      descriptor: null,
      status: 'missing_model_metadata',
      reason: 'Inventory item has no GLB metadata in attributes.',
      availableStyles: sortedAvailableStyles,
    };
  }

  const baseRecord = metadata ?? attributes;
  const fit = resolveFitFromRecord(baseRecord) ?? 'contain';
  const scale = resolveScalarFromRecord(baseRecord, SCALE_VALUE_KEYS) ?? 1;
  const rotationDeg = resolveScalarFromRecord(baseRecord, ROTATION_VALUE_KEYS) ?? 0;
  const previewCalibration = resolvePreviewCalibration(baseRecord);
  const previewOverride = resolvePreviewOverride(baseRecord);

  return {
    descriptor: {
      url: match.url,
      source: match.source,
      fit,
      scale,
      rotationDeg,
      previewCalibration,
      previewOverride,
    },
    status: resolvedStatus ?? 'resolved_direct_model',
    reason:
      resolvedStatus === 'resolved_style_variant'
        ? `Using style-specific GLB metadata for "${canonicalizeStyleKey(preferredStyle) ?? preferredStyle ?? ''}".`
        : `Using GLB metadata from "${match.source}".`,
    availableStyles: Array.from(availableStyles).sort(),
  };
}

export function disposeThreeObject(root: any): void {
  if (!root?.traverse) {
    return;
  }

  root.traverse((node: any) => {
    if (typeof node.geometry?.dispose === 'function') {
      node.geometry.dispose();
    }
    const materials = Array.isArray(node.material) ? node.material : [node.material];
    for (const material of materials) {
      if (typeof material?.dispose === 'function') {
        material.dispose();
      }
    }
  });
}

export function applyInventoryModelPreviewOverride(params: {
  descriptor: InventoryModelDescriptor;
  context: string;
  fitStrategy: GlbFitStrategy;
  wM: number;
  dM: number;
  hM: number;
}): {
  descriptor: InventoryModelDescriptor;
  fitStrategy: GlbFitStrategy;
  wM: number;
  dM: number;
  hM: number;
  applied: InventoryModelPreviewOverride | null;
} {
  const { descriptor, context, fitStrategy, wM, dM, hM } = params;
  const previewOverride = descriptor.previewOverride;
  if (!previewOverride?.enabled) {
    return {
      descriptor,
      fitStrategy,
      wM,
      dM,
      hM,
      applied: null,
    };
  }

  const normalizedContext = context.trim().toLowerCase();
  if (
    previewOverride.contexts.length > 0 &&
    !previewOverride.contexts.includes(normalizedContext) &&
    !previewOverride.contexts.includes('all')
  ) {
    return {
      descriptor,
      fitStrategy,
      wM,
      dM,
      hM,
      applied: null,
    };
  }

  const widthScale = previewOverride.dimensionScale?.width ?? 1;
  const heightScale = previewOverride.dimensionScale?.height ?? 1;
  const depthScale = previewOverride.dimensionScale?.depth ?? 1;
  return {
    descriptor: {
      ...descriptor,
      rotationDeg: descriptor.rotationDeg + previewOverride.rotationDegOffset,
      scale:
        descriptor.scale *
        (previewOverride.scaleMultiplier && previewOverride.scaleMultiplier > 0
          ? previewOverride.scaleMultiplier
          : 1),
    },
    fitStrategy: previewOverride.fitStrategy ?? fitStrategy,
    wM: wM * widthScale,
    dM: dM * depthScale,
    hM: hM * heightScale,
    applied: previewOverride,
  };
}

export async function buildGlbPreviewModel(
  params: BuildGlbPreviewModelParams
): Promise<any> {
  const {
    THREE,
    GLTFLoader,
    descriptor,
    wM,
    dM,
    hM,
    fitStrategy = 'box',
    colorHex = null,
    preserveMaterialTextures = false,
    minimumHorizontalAspectRatio = null,
  } = params;

  let gltfPromise = gltfPromiseCache.get(descriptor.url);
  if (!gltfPromise) {
    const loader = new GLTFLoader();
    gltfPromise = loader.loadAsync(descriptor.url);
    gltfPromiseCache.set(descriptor.url, gltfPromise);
  }

  const gltf = await gltfPromise;
  const sceneSource =
    gltf?.scene && typeof gltf.scene === 'object'
      ? gltf.scene
      : Array.isArray(gltf?.scenes)
        ? gltf.scenes[0]
        : null;
  if (!sceneSource || typeof (sceneSource as any).clone !== 'function') {
    throw new Error(`Model "${descriptor.url}" does not contain a scene.`);
  }

  const visualRoot = new THREE.Group();
  const geometryAdjustRoot = new THREE.Group();
  const calibrationRoot = new THREE.Group();
  visualRoot.add(geometryAdjustRoot);
  geometryAdjustRoot.add(calibrationRoot);
  const scene = (sceneSource as any).clone(true);
  calibrationRoot.add(scene);
  scene.rotation.y = (descriptor.rotationDeg * Math.PI) / 180;
  const previewCalibration = resolveActivePreviewCalibration(descriptor.previewCalibration);
  const rotationMatrix = previewCalibration?.rotationMatrix;
  if (rotationMatrix) {
    const matrix4 = new THREE.Matrix4();
    matrix4.set(
      rotationMatrix[0][0],
      rotationMatrix[0][1],
      rotationMatrix[0][2],
      rotationMatrix[0][3],
      rotationMatrix[1][0],
      rotationMatrix[1][1],
      rotationMatrix[1][2],
      rotationMatrix[1][3],
      rotationMatrix[2][0],
      rotationMatrix[2][1],
      rotationMatrix[2][2],
      rotationMatrix[2][3],
      rotationMatrix[3][0],
      rotationMatrix[3][1],
      rotationMatrix[3][2],
      rotationMatrix[3][3]
    );
    calibrationRoot.applyMatrix4(matrix4);
  }
  applyPreviewColorTint(THREE, visualRoot, colorHex, {
    preserveIntrinsicMaterialColor: preserveMaterialTextures,
  });

  visualRoot.updateMatrixWorld(true);
  let baseBox = computePreviewBounds(THREE, visualRoot);
  let computedBaseSize = baseBox.getSize(new THREE.Vector3());
  const calibrationBaseSize = previewCalibration?.baseSizeM;
  let baseSize =
    calibrationBaseSize &&
    calibrationBaseSize.width > 0 &&
    calibrationBaseSize.height > 0 &&
    calibrationBaseSize.depth > 0
      ? new THREE.Vector3(
          calibrationBaseSize.width,
          calibrationBaseSize.height,
          calibrationBaseSize.depth
        )
      : computedBaseSize;

  const shouldThickenCustomDepth =
    typeof minimumHorizontalAspectRatio === 'number' &&
    Number.isFinite(minimumHorizontalAspectRatio) &&
    minimumHorizontalAspectRatio > 0 &&
    fitStrategy.startsWith('footprint');
  if (shouldThickenCustomDepth && computedBaseSize.x > 0 && computedBaseSize.z > 0) {
    const maxHorizontal = Math.max(computedBaseSize.x, computedBaseSize.z);
    const minHorizontal = Math.min(computedBaseSize.x, computedBaseSize.z);
    const currentRatio = minHorizontal / maxHorizontal;
    const targetRatio = Math.min(Math.max(minimumHorizontalAspectRatio ?? 0, 0), 0.95);
    if (currentRatio > 0 && currentRatio < targetRatio) {
      const desiredMinHorizontal = maxHorizontal * targetRatio;
      const scaleMultiplier = Math.min(desiredMinHorizontal / minHorizontal, 2.2);
      if (scaleMultiplier > 1.01) {
        if (computedBaseSize.x <= computedBaseSize.z) {
          geometryAdjustRoot.scale.x = scaleMultiplier;
        } else {
          geometryAdjustRoot.scale.z = scaleMultiplier;
        }
        visualRoot.updateMatrixWorld(true);
        baseBox = computePreviewBounds(THREE, visualRoot);
        computedBaseSize = baseBox.getSize(new THREE.Vector3());
        baseSize = computedBaseSize.clone();
      }
    }
  }
  const metadataScale =
    Number.isFinite(descriptor.scale) && descriptor.scale > 0 ? descriptor.scale : 1;
  let resolvedFitStrategy = fitStrategy;
  if (
    fitStrategy === 'footprint-uniform-or-proxy' ||
    fitStrategy === 'wall-plane-uniform-or-proxy'
  ) {
    const isFootprintFallback = fitStrategy === 'footprint-uniform-or-proxy';
    const basePrimary = baseSize.x;
    const baseSecondary = isFootprintFallback ? baseSize.z : baseSize.y;
    const targetPrimary = wM;
    const targetSecondary = isFootprintFallback ? dM : hM;
    if (
      basePrimary > 0 &&
      baseSecondary > 0 &&
      Number.isFinite(targetPrimary) &&
      targetPrimary > 0 &&
      Number.isFinite(targetSecondary) &&
      targetSecondary > 0
    ) {
      const baseAspect = basePrimary / baseSecondary;
      const targetAspect = targetPrimary / targetSecondary;
      const aspectMismatch = Math.max(baseAspect / targetAspect, targetAspect / baseAspect);
      if (aspectMismatch > PROXY_BOX_ASPECT_FALLBACK_THRESHOLD) {
        resolvedFitStrategy = isFootprintFallback ? 'footprint' : 'wall-plane';
      }
    }
    if (resolvedFitStrategy === fitStrategy) {
      resolvedFitStrategy = isFootprintFallback ? 'footprint-uniform' : 'wall-plane-uniform';
    }
  }

  if (resolvedFitStrategy === 'footprint-uniform' || resolvedFitStrategy === 'wall-plane-uniform') {
    const planeScaleFactors =
      resolvedFitStrategy === 'footprint-uniform'
        ? [
            baseSize.x > 0 ? wM / baseSize.x : null,
            baseSize.z > 0 ? dM / baseSize.z : null,
          ]
        : [
            baseSize.x > 0 ? wM / baseSize.x : null,
            baseSize.y > 0 ? hM / baseSize.y : null,
          ];
    const validPlaneScaleFactors = planeScaleFactors.filter(
      (value): value is number => value !== null && Number.isFinite(value) && value > 0
    );
    const fitScale =
      validPlaneScaleFactors.length === 0
        ? 1
        : descriptor.fit === 'cover'
          ? Math.max(...validPlaneScaleFactors)
          : Math.min(...validPlaneScaleFactors);
    visualRoot.scale.setScalar(fitScale * metadataScale);
  } else if (resolvedFitStrategy === 'footprint' || resolvedFitStrategy === 'wall-plane') {
    const scaleX =
      baseSize.x > 0 && Number.isFinite(wM) && wM > 0 ? (wM / baseSize.x) * metadataScale : metadataScale;
    const scaleY =
      baseSize.y > 0 && Number.isFinite(hM) && hM > 0 ? (hM / baseSize.y) * metadataScale : metadataScale;
    const scaleZ =
      baseSize.z > 0 && Number.isFinite(dM) && dM > 0 ? (dM / baseSize.z) * metadataScale : metadataScale;
    visualRoot.scale.set(scaleX, scaleY, scaleZ);
  } else {
    const scaleFactors = [
      baseSize.x > 0 ? wM / baseSize.x : null,
      baseSize.y > 0 ? hM / baseSize.y : null,
      baseSize.z > 0 ? dM / baseSize.z : null,
    ].filter((value): value is number => value !== null && Number.isFinite(value) && value > 0);

    const fitScale =
      scaleFactors.length === 0
        ? 1
        : descriptor.fit === 'cover'
          ? Math.max(...scaleFactors)
          : Math.min(...scaleFactors);
    const totalScale = fitScale * metadataScale;
    visualRoot.scale.setScalar(totalScale);
  }

  visualRoot.updateMatrixWorld(true);
  const fittedBox = computePreviewBounds(THREE, visualRoot);
  const fittedCenter = fittedBox.getCenter(new THREE.Vector3());
  visualRoot.position.x -= fittedCenter.x;
  visualRoot.position.z -= fittedCenter.z;
  visualRoot.position.y -= fittedBox.min.y;
  visualRoot.updateMatrixWorld(true);

  visualRoot.traverse((node: any) => {
    if (node?.isMesh) {
      node.castShadow = true;
      node.receiveShadow = true;
      if (typeof node.geometry?.computeVertexNormals === 'function' && !node.geometry.attributes?.normal) {
        node.geometry.computeVertexNormals();
      }
    }
  });

  visualRoot.userData = {
    ...visualRoot.userData,
    modelUrl: descriptor.url,
    modelSource: descriptor.source,
  };

  return visualRoot;
}
