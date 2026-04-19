import type { InventoryItem } from '../types';
import {
  inspectInventoryModelDescriptor,
  type InventoryModelDescriptorResolution,
} from './models';

const REFILL_ASSET_ID_PATTERN = /__(?:reintroduced|accessory_refill)(?:_\d+)?$/i;
const TRAILING_NUMERIC_ASSET_SUFFIX_PATTERN = /(?:[_-]\d+)+$/;

export type InventoryPreviewMatchSource =
  | 'exact_asset_id'
  | 'asset_id_prefix'
  | 'category_style_material'
  | 'category_style'
  | 'category_material'
  | 'category_only'
  | 'none';

export type InventoryDimensionsMm = {
  width: number;
  depth: number;
  height: number;
};

export type InventoryPreviewResolution = {
  item: InventoryItem | null;
  source: InventoryPreviewMatchSource;
  reason: string;
  modelResolution: InventoryModelDescriptorResolution | null;
  matchScore: number;
  dimensionScore: number;
  glbEligible: boolean;
  glbReason: string;
};

export type ReintroducedObjectSummary = {
  clusterId: string | null;
  objectId: string;
  objectType: string;
};

function clampNumber(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function readString(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null;
}

function readRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' ? (value as Record<string, unknown>) : null;
}

function canonicalizeToken(value: unknown): string | null {
  const raw = readString(value);
  if (!raw) {
    return null;
  }
  return raw.toLowerCase().replace(REFILL_ASSET_ID_PATTERN, '').replace(/[\s-]+/g, '_').trim();
}

function canonicalizeAssetPrefix(value: unknown): string | null {
  const token = canonicalizeToken(value);
  if (!token) {
    return null;
  }
  return token.replace(TRAILING_NUMERIC_ASSET_SUFFIX_PATTERN, '').trim() || token;
}

function canonicalizeStyle(value: unknown): string | null {
  const raw = readString(value);
  if (!raw) {
    return null;
  }
  return raw.toLowerCase().replace(/\s+/g, ' ').trim();
}

function itemCategory(item: InventoryItem): string | null {
  const fromAttributes =
    item.attributes && typeof item.attributes === 'object'
      ? readString((item.attributes as Record<string, unknown>).category)
      : null;
  return fromAttributes ?? readString(item.category) ?? readString(item.name);
}

function itemMaterial(item: InventoryItem): string | null {
  const direct = readString(item.material);
  if (direct) {
    return direct.toLowerCase();
  }
  const fromTags = Array.isArray(item.material_tags)
    ? item.material_tags.find((tag) => readString(tag))
    : null;
  if (fromTags) {
    return String(fromTags).toLowerCase();
  }
  const fromAttributes =
    item.attributes && typeof item.attributes === 'object'
      ? readString((item.attributes as Record<string, unknown>).material)
      : null;
  return fromAttributes ? fromAttributes.toLowerCase() : null;
}

function scoreDimensionFit(
  item: InventoryItem,
  targetWidthMm?: number,
  targetDepthMm?: number,
  targetHeightMm?: number
): number {
  const dimensions = getInventoryDimensionsMm(item);
  if (!dimensions) {
    return 0;
  }

  let score = 0;
  if (targetWidthMm && targetWidthMm > 0) {
    const relativeDiff = Math.abs(dimensions.width - targetWidthMm) / targetWidthMm;
    score += Math.max(0, 30 - relativeDiff * 30);
  }
  if (targetDepthMm && targetDepthMm > 0) {
    const relativeDiff = Math.abs(dimensions.depth - targetDepthMm) / targetDepthMm;
    score += Math.max(0, 20 - relativeDiff * 20);
  }
  if (targetHeightMm && targetHeightMm > 0) {
    const relativeDiff = Math.abs(dimensions.height - targetHeightMm) / targetHeightMm;
    score += Math.max(0, 15 - relativeDiff * 15);
  }
  return score;
}

export function normalizePreviewAssetId(value: unknown): string | null {
  const raw = readString(value);
  if (!raw) {
    return null;
  }
  const normalized = raw.replace(REFILL_ASSET_ID_PATTERN, '').trim();
  return normalized || raw;
}

export function isReintroducedIdentifier(value: unknown): boolean {
  const raw = readString(value);
  return raw ? REFILL_ASSET_ID_PATTERN.test(raw) : false;
}

export function getInventoryDimensionsMm(
  item: InventoryItem | null | undefined
): InventoryDimensionsMm | null {
  if (!item) {
    return null;
  }
  const width = Number(item.footprint_mm?.w ?? 0);
  const depth = Number(item.footprint_mm?.d ?? 0);
  const height = Number(item.height_mm ?? 0);
  if (width <= 0 || depth <= 0 || height <= 0) {
    return null;
  }
  return { width, depth, height };
}

export function isShelfLikePreviewType(value: string): boolean {
  const token = canonicalizeToken(value) ?? '';
  return (
    token.includes('shelf') ||
    token.includes('bookcase') ||
    token.includes('bookshelf') ||
    token.includes('cabinet') ||
    token.includes('etagere') ||
    token.includes('rack')
  );
}

export function isUserGeneratedPreviewItem(
  item: InventoryItem | null | undefined
): boolean {
  if (!item) {
    return false;
  }

  const attributes = readRecord(item.attributes);
  if (!attributes) {
    return false;
  }

  const ownershipScope = readString(attributes.ownership_scope)?.toLowerCase() ?? null;
  const sourceMode = readString(attributes.source_mode)?.toLowerCase() ?? null;
  const createdByUserId = readString(attributes.created_by_user_id);

  if (ownershipScope !== 'owned') {
    return false;
  }

  return (
    sourceMode === 'prompt' ||
    sourceMode === 'image' ||
    createdByUserId !== null
  );
}

export function resolveWallHungPreviewDepthMm(params: {
  objectType: string;
  planWidthMm: number;
  planHeightMm: number;
  inventoryDimensions: InventoryDimensionsMm | null;
}): number {
  const { objectType, planWidthMm, planHeightMm, inventoryDimensions } = params;
  const token = canonicalizeToken(objectType) ?? '';
  const inventoryDepthMm = Number(inventoryDimensions?.depth ?? 0);
  const hasInventoryDepth = Number.isFinite(inventoryDepthMm) && inventoryDepthMm > 0;
  const referenceSpanMm = Math.max(planWidthMm, planHeightMm, 1);

  if (token.includes('mirror') || token.includes('wall_art') || token.includes('clock')) {
    return Math.round(
      hasInventoryDepth
        ? clampNumber(inventoryDepthMm, 12, 60)
        : clampNumber(referenceSpanMm * 0.03, 12, 40)
    );
  }

  if (isShelfLikePreviewType(objectType)) {
    return Math.round(
      hasInventoryDepth
        ? clampNumber(inventoryDepthMm, 120, 520)
        : clampNumber(planWidthMm * 0.18, 120, 320)
    );
  }

  return Math.round(
    hasInventoryDepth
      ? clampNumber(inventoryDepthMm, 20, 260)
      : clampNumber(referenceSpanMm * 0.08, 20, 90)
  );
}

export function collectReintroducedObjects(
  styledResult: unknown
): ReintroducedObjectSummary[] {
  const summary = (styledResult as Record<string, unknown> | null)?.reintroduced_objects_by_cluster;
  if (!summary || typeof summary !== 'object') {
    return [];
  }

  const results: ReintroducedObjectSummary[] = [];
  for (const [clusterId, items] of Object.entries(summary)) {
    if (!Array.isArray(items)) {
      continue;
    }
    for (const item of items) {
      if (!item || typeof item !== 'object') {
        continue;
      }
      const row = item as Record<string, unknown>;
      const objectId = readString(row.object_id);
      if (!objectId) {
        continue;
      }
      results.push({
        clusterId: clusterId || null,
        objectId,
        objectType: readString(row.object_type) ?? 'item',
      });
    }
  }
  return results;
}

export function resolveInventoryItemForPreview(params: {
  inventory: InventoryItem[];
  assetId?: string | null;
  objectType?: string | null;
  material?: string | null;
  preferredStyle?: string | null;
  targetWidthMm?: number;
  targetDepthMm?: number;
  targetHeightMm?: number;
}): InventoryPreviewResolution {
  const {
    inventory,
    assetId,
    objectType,
    material,
    preferredStyle,
    targetWidthMm,
    targetDepthMm,
    targetHeightMm,
  } = params;

  const normalizedAssetId = canonicalizeToken(assetId);
  const normalizedAssetPrefix = canonicalizeAssetPrefix(assetId);
  const normalizedObjectType = canonicalizeToken(objectType);
  const preferredStyleKey = canonicalizeStyle(preferredStyle);
  const materialKey = readString(material)?.toLowerCase() ?? null;

  const exactMatch = inventory.find(
    (item) => canonicalizeToken(item.asset_id) === normalizedAssetId
  );
  if (exactMatch) {
    const dimensionScore = scoreDimensionFit(
      exactMatch,
      targetWidthMm,
      targetDepthMm,
      targetHeightMm
    );
    const modelResolution = inspectInventoryModelDescriptor(exactMatch, preferredStyle);
    return {
      item: exactMatch,
      source: 'exact_asset_id',
      reason: `Matched inventory asset "${exactMatch.asset_id}" by exact asset id.`,
      modelResolution,
      matchScore: 1000 + dimensionScore,
      dimensionScore,
      glbEligible: Boolean(modelResolution.descriptor),
      glbReason: modelResolution.descriptor
        ? `Using GLB for exact inventory asset "${exactMatch.asset_id}".`
        : modelResolution.reason,
    };
  }

  type Candidate = {
    item: InventoryItem;
    source: Exclude<InventoryPreviewMatchSource, 'exact_asset_id' | 'none'>;
    score: number;
    dimensionScore: number;
    modelResolution: InventoryModelDescriptorResolution;
  };

  const candidates: Candidate[] = [];
  for (const item of inventory) {
    const assetPrefix = canonicalizeAssetPrefix(item.asset_id);
    const category = canonicalizeToken(itemCategory(item));
    const candidateMaterial = itemMaterial(item);
    const hasStyleMatch =
      preferredStyleKey !== null &&
      item.style_tags.some((tag) => canonicalizeStyle(tag) === preferredStyleKey);
    const hasMaterialMatch =
      materialKey !== null && candidateMaterial !== null && candidateMaterial === materialKey;
    const modelResolution = inspectInventoryModelDescriptor(item, preferredStyle);
    const modelScore = modelResolution.descriptor ? 180 : modelResolution.availableStyles.length > 0 ? 40 : 0;
    const dimensionScore = scoreDimensionFit(
      item,
      targetWidthMm,
      targetDepthMm,
      targetHeightMm
    );

    if (
      normalizedAssetPrefix &&
      assetPrefix &&
      normalizedAssetPrefix === assetPrefix
    ) {
      candidates.push({
        item,
        source: 'asset_id_prefix',
        score: 900 + modelScore + dimensionScore,
        dimensionScore,
        modelResolution,
      });
      continue;
    }

    const categoryMatches =
      (normalizedAssetPrefix && category === normalizedAssetPrefix) ||
      (normalizedObjectType && category === normalizedObjectType);
    if (!categoryMatches) {
      continue;
    }

    let source: Candidate['source'] = 'category_only';
    let baseScore = 600;
    if (hasStyleMatch && hasMaterialMatch) {
      source = 'category_style_material';
      baseScore = 820;
    } else if (hasStyleMatch) {
      source = 'category_style';
      baseScore = 760;
    } else if (hasMaterialMatch) {
      source = 'category_material';
      baseScore = 700;
    }

    candidates.push({
      item,
      source,
      score: baseScore + modelScore + dimensionScore,
      dimensionScore,
      modelResolution,
    });
  }

  if (candidates.length === 0) {
    const lookup = normalizePreviewAssetId(assetId) ?? normalizePreviewAssetId(objectType) ?? 'unknown';
    return {
      item: null,
      source: 'none',
      reason: `No inventory asset matched layout lookup "${lookup}".`,
      modelResolution: null,
      matchScore: 0,
      dimensionScore: 0,
      glbEligible: false,
      glbReason: `No inventory asset matched layout lookup "${lookup}".`,
    };
  }

  candidates.sort((left, right) => {
    if (right.score !== left.score) {
      return right.score - left.score;
    }
    return left.item.asset_id.localeCompare(right.item.asset_id);
  });

  const best = candidates[0];
  const by = best.source.replaceAll('_', ' ');
  const strongMatchSources = new Set<InventoryPreviewMatchSource>([
    'exact_asset_id',
    'asset_id_prefix',
    'category_style_material',
  ]);
  const moderateMatchSources = new Set<InventoryPreviewMatchSource>([
    'category_style',
    'category_material',
  ]);
  const hasDescriptor = Boolean(best.modelResolution.descriptor);
  const hasStyleSpecificModel = best.modelResolution.status === 'resolved_style_variant';
  const glbEligible =
    hasDescriptor &&
    (strongMatchSources.has(best.source) ||
      (moderateMatchSources.has(best.source) &&
        hasStyleSpecificModel &&
        best.dimensionScore >= 26));
  const glbReason = !hasDescriptor
    ? best.modelResolution.reason
    : glbEligible
      ? `Using GLB for inventory asset "${best.item.asset_id}" matched by ${by}.`
      : `Matched inventory asset "${best.item.asset_id}" by ${by}, but confidence is too low for GLB preview, so proxy fallback stays safer.`;
  return {
    item: best.item,
    source: best.source,
    reason: `Matched inventory asset "${best.item.asset_id}" by ${by}.`,
    modelResolution: best.modelResolution,
    matchScore: best.score,
    dimensionScore: best.dimensionScore,
    glbEligible,
    glbReason,
  };
}
