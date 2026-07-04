import {
  AdminUser,
  AuthUser,
  GeneratedRenderRecord,
  InventoryItem,
  SavedLayoutRecord,
  UserRole,
} from './types';

const DEFAULT_API_BASE = import.meta.env.PROD
  ? '/api'
  : 'http://localhost:8001';
const API_BASE = normalizeApiBase(
  import.meta.env.VITE_API_BASE ?? DEFAULT_API_BASE
);
const AUTH_TOKEN_STORAGE_KEY = 'tknt.authToken';
const API_RELATIVE_URL_PATTERN =
  /^\/(inventory|account|pipeline|auth|admin|billing|assets_gen)(\/|$)/;

function normalizeApiBase(value: unknown): string {
  const raw = typeof value === 'string' ? value.trim() : '';
  if (!raw || raw === '/') {
    return '';
  }
  return raw.replace(/\/+$/, '');
}

function normalizePath(path: string): string {
  return path.startsWith('/') ? path : `/${path}`;
}

function joinBasePath(base: string, path: string): string {
  const normalizedPath = normalizePath(path);
  if (!base) {
    return normalizedPath;
  }
  return `${base}${normalizedPath}`;
}

function resolveApiUrl(path: string): string {
  return joinBasePath(API_BASE, path);
}

function browserOrigin(): string {
  return typeof window !== 'undefined' ? window.location.origin : 'http://localhost';
}

export type SnapshotImageSceneReferenceMode =
  | 'none'
  | 'target_layout_with_scene_reference'
  | 'scene_reference_camera_transfer';

export type SnapshotImageQuality = 'low' | 'medium' | 'high';
export type SnapshotImageSize = '1024x1024' | '1024x1536' | '1536x1024';
export type SnapshotImageRenderFlow =
  | 'default'
  | 'single_edit_edge_silhouette_guided'
  | 'edit_object_scope_preflight';

export type AuthSessionResponse = {
  access_token: string;
  token_type: 'bearer';
  user: AuthUser;
};

export type BillingSettingsResponse = {
  per_credit_price_vnd: number;
  monthly_price_vnd: number;
  monthly_credit_quota: number;
  updated_at?: string | null;
};

export type AdminUserCreatePayload = {
  email: string;
  password: string;
  display_name?: string | null;
  role?: UserRole;
};

export type AdminUserUpdatePayload = {
  display_name?: string | null;
  role?: UserRole;
  is_active?: boolean;
};

export type AdminFurnitureCreatePayload = {
  name: string;
  category: string;
  style_tags: string[];
  material?: string | null;
  brand?: string | null;
  length_mm?: number | null;
  width_mm?: number | null;
  height_mm?: number | null;
  price_amount?: number | null;
  price_currency?: string;
  attributes?: Record<string, unknown>;
  model_reference?: string | null;
  preview_reference?: string | null;
};

export type SnapshotImageRenderResult = {
  request: {
    render_mode: 'generate' | 'edit' | 'camera_rerender';
    render_flow?: SnapshotImageRenderFlow;
    system_prompt: string;
    user_prompt: string;
    raw_user_prompt?: string | null;
    preset_prompt?: string | null;
    preset_selection?: SnapshotImagePresetSelection;
    image_quality_requested?: SnapshotImageQuality | string | null;
    image_size_requested?: SnapshotImageSize | string | null;
    stream_partial_images?: number;
    streamed?: boolean;
    aspect_ratio: string;
    source_image_mime_type: string;
    structural_image_mime_type?: string | null;
    geometry_references_used?: {
      depth_map: boolean;
      edge_map: boolean;
      segmentation_map: boolean;
    };
    geometry_reference_image_mime_types?: {
      depth_map?: string | null;
      edge_map?: string | null;
      segmentation_map?: string | null;
    };
    layout_reference_enabled: boolean;
    layout_reference_used: boolean;
    layout_reference_image_mime_type?: string | null;
    annotated_reference_used?: boolean;
    edit_region_guide_used?: boolean;
    camera_transfer_reference_used?: boolean;
    annotated_reference_image_mime_type?: string | null;
    scene_reference_mode?: SnapshotImageSceneReferenceMode;
    scene_reference_used?: boolean;
    reference_only_camera_transfer_used?: boolean;
    scene_reference_image_mime_type?: string | null;
    reference_image_count?: number;
    edit_operations?: SnapshotImageEditOperationSummary[];
    layout_lock_mode?: string;
    object_memory_used?: Array<Record<string, unknown>>;
    locked_items?: Array<Record<string, unknown>>;
    visibility_diagnostics?: Record<string, unknown>;
    locked_geometry_references?: Array<Record<string, unknown>>;
    locked_geometry_atlases?: Array<Record<string, unknown>>;
    locked_geometry_atlas_count?: number;
    locked_geometry_object_count?: number;
    geometry_atlas_omitted_object_ids?: string[];
    geometry_lock_policy?: string;
  };
  image: {
    mime_type: string;
    image_base64: string;
    data_url: string;
    size: string;
    quality?: string;
    aspect_ratio: string;
    raw_response: Record<string, unknown>;
    retry_count?: number;
    usage?: Record<string, unknown>;
    partial_image_count?: number;
  };
  models: {
    image_model_name: string;
    max_tokens: number;
  };
  metadata?: {
    camera?: Record<string, unknown> | null;
    visible_objects?: Array<Record<string, unknown>>;
    visible_object_ids?: string[];
    render_mode?: 'generate' | 'edit' | 'camera_rerender';
    render_flow?: SnapshotImageRenderFlow;
    preset_selection?: SnapshotImagePresetSelection;
    edit_operations?: SnapshotImageEditOperationSummary[];
    render_state?: Record<string, unknown>;
    edit_region_guide_used?: boolean;
    camera_transfer_reference_used?: boolean;
    layout_lock_mode?: string;
    object_memory_used?: Array<Record<string, unknown>>;
    locked_items?: Array<Record<string, unknown>>;
    visibility_diagnostics?: Record<string, unknown>;
    locked_geometry_references?: Array<Record<string, unknown>>;
    locked_geometry_atlases?: Array<Record<string, unknown>>;
    locked_geometry_atlas_count?: number;
    locked_geometry_object_count?: number;
    geometry_atlas_omitted_object_ids?: string[];
    geometry_lock_policy?: string;
    validation?: Record<string, unknown>;
    visual_preflight?: Record<string, unknown>;
    object_scope_edit_mask?: Record<string, unknown>;
  };
};

export type SnapshotImageRenderApiResponse = {
  render: SnapshotImageRenderResult;
  saved_render?: GeneratedRenderRecord | null;
};

export type SnapshotImagePartialImageEvent = {
  partial_image_index: number;
  image_data_url: string;
  size: SnapshotImageSize | string;
  quality: SnapshotImageQuality | string;
  attempt: string;
};

type SnapshotImageStreamHandlers = {
  onStatus?: (event: { message: string; attempt: string | null }) => void;
  onPartialImage?: (event: SnapshotImagePartialImageEvent) => void;
  onCompleted?: (event: SnapshotImageRenderApiResponse) => void;
  onError?: (event: { message: string }) => void;
};

type SnapshotImageRenderClientRequest = {
  snapshotPayload: Record<string, unknown>;
  snapshotImageDataUrl: string;
  depthMapImageDataUrl?: string | null;
  edgeMapImageDataUrl?: string | null;
  segmentationMapImageDataUrl?: string | null;
  annotatedReferenceImageDataUrl?: string | null;
  sceneReferenceImageDataUrl?: string | null;
  sceneReferenceMode?: SnapshotImageSceneReferenceMode;
  userPrompt: string;
  renderMode?: 'generate' | 'edit' | 'camera_rerender';
  renderFlow?: SnapshotImageRenderFlow;
  presetSelection?: SnapshotImagePresetSelection;
  rootLayoutId?: string | null;
  caseId?: string | null;
  optionId?: string | null;
  editOperations?: SnapshotImageEditOperation[];
  lockedGeometryReferences?: SnapshotImageLockedGeometryReference[];
  lockedGeometryAtlases?: SnapshotImageLockedGeometryAtlas[];
  editSourceImageDataUrl?: string | null;
  imageQuality?: SnapshotImageQuality | null;
  imageSize?: SnapshotImageSize | null;
};

export type GeneratedRenderDetailResponse = {
  render: GeneratedRenderRecord;
  image_data_url: string;
};

export type SnapshotImagePresetOption = {
  label: string;
  prompt_suffix?: string | null;
  reference_image?: string | null;
};

export type SnapshotImagePresets = {
  styles: Record<string, SnapshotImagePresetOption>;
  lights: Record<string, SnapshotImagePresetOption>;
  sceneries: Record<string, SnapshotImagePresetOption>;
};

export type SnapshotImagePresetSelection = {
  style?: string | null;
  lighting?: string | null;
  scenery?: string | null;
};

export type SnapshotImageEditOperation = {
  object_id: string;
  object_name?: string | null;
  operation_type?: 'prompt' | 'add' | 'replace' | 'recolor' | 'remove' | null;
  prompt?: string | null;
  bbox_norm?: {
    x: number;
    y: number;
    width: number;
    height: number;
  } | null;
  replacement_image_data_url?: string | null;
  target_color?: string | null;
};

type SnapshotImageLockedGeometryReference = {
  object_id: string;
  object_name?: string | null;
  image_data_url: string;
  source?: string | null;
  screen_bbox_px?: Record<string, number> | null;
};

type SnapshotImageLockedGeometryAtlasObject = {
  object_id: string;
  object_name?: string | null;
  screen_bbox_px?: Record<string, number> | null;
  atlas_index: number;
  cell_index: number;
  cell_label?: string | null;
  cell_bbox_norm?: Record<string, number> | null;
};

export type SnapshotImageLockedGeometryAtlas = {
  atlas_index: number;
  image_data_url: string;
  source?: string | null;
  objects: SnapshotImageLockedGeometryAtlasObject[];
};

type SnapshotImageEditOperationSummary = {
  object_id: string;
  object_name?: string | null;
  operation_type?: 'prompt' | 'add' | 'replace' | 'recolor' | 'remove' | null;
  prompt?: string | null;
  bbox_provided?: boolean;
  replacement_image_provided?: boolean;
  target_color?: string | null;
};

export function getStoredAuthToken(): string | null {
  if (typeof window === 'undefined') {
    return null;
  }
  return window.localStorage.getItem(AUTH_TOKEN_STORAGE_KEY);
}

export function storeAuthToken(token: string): void {
  if (typeof window === 'undefined') {
    return;
  }
  window.localStorage.setItem(AUTH_TOKEN_STORAGE_KEY, token);
}

export function clearStoredAuthToken(): void {
  if (typeof window === 'undefined') {
    return;
  }
  window.localStorage.removeItem(AUTH_TOKEN_STORAGE_KEY);
}

function resolveAuthToken(explicitToken?: string | null): string | null {
  if (explicitToken && explicitToken.trim()) {
    return explicitToken.trim();
  }
  return getStoredAuthToken();
}

function withAuthHeaders(
  init: RequestInit = {},
  token?: string | null,
  includeJsonContentType = false
): RequestInit {
  const resolvedToken = resolveAuthToken(token);
  const headers = new Headers(init.headers ?? {});
  if (includeJsonContentType && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }
  if (resolvedToken) {
    headers.set('Authorization', `Bearer ${resolvedToken}`);
  }
  return { ...init, headers };
}

async function parseError(res: Response, fallback: string): Promise<Error> {
  let detail = fallback;
  try {
    const body = await res.json();
    if (typeof body?.detail === 'string' && body.detail.trim()) {
      detail = body.detail.trim();
    }
  } catch {
    // ignore response parsing failures
  }
  return new Error(detail);
}

async function requestJson<T>(
  path: string,
  init: RequestInit = {},
  options: { token?: string | null; includeJsonContentType?: boolean; fallbackError: string }
): Promise<T> {
  let res: Response;
  try {
    res = await fetch(
      resolveApiUrl(path),
      withAuthHeaders(init, options.token, options.includeJsonContentType)
    );
  } catch (error) {
    const detail =
      error instanceof Error && error.message.trim()
        ? error.message.trim()
        : 'Network request failed';
    throw new Error(`${options.fallbackError}. Cannot reach API at ${API_BASE}. ${detail}`);
  }
  if (!res.ok) {
    throw await parseError(res, options.fallbackError);
  }
  return (await res.json()) as T;
}

function appendAccessToken(url: string, token?: string | null): string {
  const rawUrl = url.trim();
  if (!rawUrl) {
    return url;
  }

  const resolvedToken = resolveAuthToken(token);
  const normalizedPath = normalizePath(rawUrl);
  const isAbsoluteUrl = /^https?:\/\//i.test(rawUrl);
  const isApiRelativeUrl = API_RELATIVE_URL_PATTERN.test(normalizedPath);
  const resolvedUrl = isAbsoluteUrl
    ? rawUrl
    : isApiRelativeUrl
      ? resolveApiUrl(normalizedPath)
      : rawUrl;
  const absolute = new URL(resolvedUrl, browserOrigin());

  if (resolvedToken) {
    absolute.searchParams.set('access_token', resolvedToken);
  }

  if (isAbsoluteUrl || isApiRelativeUrl) {
    return absolute.toString();
  }
  if (!resolvedToken) {
    return url;
  }
  return `${absolute.pathname}${absolute.search}${absolute.hash}`;
}

function resolveReferenceImageUrl(referenceImage: string | null | undefined): string | null {
  const rawUrl = referenceImage?.trim();
  if (!rawUrl) {
    return null;
  }
  if (/^(?:https?:|data:|blob:)/i.test(rawUrl)) {
    return rawUrl;
  }
  const normalizedPath = normalizePath(rawUrl);
  if (API_RELATIVE_URL_PATTERN.test(normalizedPath)) {
    return resolveApiUrl(normalizedPath);
  }
  return new URL(normalizedPath, browserOrigin()).toString();
}

function normalizePresetReferenceImages(
  presets: SnapshotImagePresets
): SnapshotImagePresets {
  const normalizeGroup = (
    group: Record<string, SnapshotImagePresetOption>
  ): Record<string, SnapshotImagePresetOption> =>
    Object.fromEntries(
      Object.entries(group).map(([key, option]) => [
        key,
        {
          ...option,
          reference_image: resolveReferenceImageUrl(option.reference_image),
        },
      ])
    );

  return {
    styles: normalizeGroup(presets.styles),
    lights: normalizeGroup(presets.lights),
    sceneries: normalizeGroup(presets.sceneries),
  };
}

function appendTokenToAttributes(
  attributes: Record<string, unknown> | null | undefined,
  token?: string | null
): Record<string, unknown> {
  const nextAttributes = { ...(attributes ?? {}) };

  const appendTokenToModelRecord = (
    value: unknown
  ): Record<string, unknown> | unknown => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      return value;
    }
    const nextRecord = { ...(value as Record<string, unknown>) };
    const nestedUrlKeys = [
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
    for (const key of nestedUrlKeys) {
      if (typeof nextRecord[key] === 'string') {
        nextRecord[key] = appendAccessToken(String(nextRecord[key]), token);
      }
    }
    return nextRecord;
  };

  const files = Array.isArray(nextAttributes.files) ? nextAttributes.files : null;
  if (files) {
    nextAttributes.files = files.map((file) => {
      if (!file || typeof file !== 'object') {
        return file;
      }
      const nextFile = { ...(file as Record<string, unknown>) };
      if (typeof nextFile.url === 'string') {
        nextFile.url = appendAccessToken(nextFile.url, token);
      }
      return nextFile;
    });
  }

  const directUrlKeys = [
    'model_url',
    'modelUrl',
    'glb_url',
    'glbUrl',
    'preview_model_url',
    'previewModelUrl',
  ];
  for (const key of directUrlKeys) {
    if (typeof nextAttributes[key] === 'string') {
      nextAttributes[key] = appendAccessToken(String(nextAttributes[key]), token);
    }
  }

  const nestedModelKeys = [
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
  for (const key of nestedModelKeys) {
    nextAttributes[key] = appendTokenToModelRecord(nextAttributes[key]);
  }

  const variantKeys = ['model_variants', 'modelVariants'];
  for (const key of variantKeys) {
    const variants = nextAttributes[key];
    if (!variants || typeof variants !== 'object' || Array.isArray(variants)) {
      continue;
    }
    nextAttributes[key] = Object.fromEntries(
      Object.entries(variants as Record<string, unknown>).map(([variantKey, variantValue]) => [
        variantKey,
        appendTokenToModelRecord(variantValue),
      ])
    );
  }

  return nextAttributes;
}

export async function registerUser(payload: {
  email: string;
  password: string;
  display_name?: string | null;
}): Promise<AuthSessionResponse> {
  return requestJson<AuthSessionResponse>(
    '/auth/register',
    {
      method: 'POST',
      body: JSON.stringify(payload),
    },
    {
      includeJsonContentType: true,
      fallbackError: 'Failed to register account',
    }
  );
}

export async function loginUser(payload: {
  email: string;
  password: string;
}): Promise<AuthSessionResponse> {
  return requestJson<AuthSessionResponse>(
    '/auth/login',
    {
      method: 'POST',
      body: JSON.stringify(payload),
    },
    {
      includeJsonContentType: true,
      fallbackError: 'Failed to sign in',
    }
  );
}

export async function getCurrentUser(token?: string | null): Promise<AuthUser> {
  return requestJson<AuthUser>('/auth/me', {}, { token, fallbackError: 'Failed to load account' });
}

export async function logoutUser(token?: string | null): Promise<void> {
  await requestJson<{ ok: boolean }>(
    '/auth/logout',
    { method: 'POST' },
    { token, fallbackError: 'Failed to sign out' }
  );
}

export async function fetchBillingSettings(): Promise<BillingSettingsResponse> {
  return requestJson<BillingSettingsResponse>(
    '/billing/settings',
    {},
    { fallbackError: 'Failed to load billing settings' }
  );
}

export async function updateAdminBillingSettings(
  payload: BillingSettingsResponse,
  token?: string | null
): Promise<BillingSettingsResponse> {
  return requestJson<BillingSettingsResponse>(
    '/admin/billing/settings',
    {
      method: 'PATCH',
      body: JSON.stringify(payload),
    },
    {
      token,
      includeJsonContentType: true,
      fallbackError: 'Failed to update billing settings',
    }
  );
}

export async function listAdminUsers(token?: string | null): Promise<AdminUser[]> {
  return requestJson<AdminUser[]>(
    '/admin/users',
    {},
    { token, fallbackError: 'Failed to load users' }
  );
}

export async function createAdminUser(
  payload: AdminUserCreatePayload,
  token?: string | null
): Promise<AdminUser> {
  return requestJson<AdminUser>(
    '/admin/users',
    {
      method: 'POST',
      body: JSON.stringify(payload),
    },
    {
      token,
      includeJsonContentType: true,
      fallbackError: 'Failed to create user',
    }
  );
}

export async function updateAdminUser(
  userId: string,
  payload: AdminUserUpdatePayload,
  token?: string | null
): Promise<AdminUser> {
  return requestJson<AdminUser>(
    `/admin/users/${encodeURIComponent(userId)}`,
    {
      method: 'PATCH',
      body: JSON.stringify(payload),
    },
    {
      token,
      includeJsonContentType: true,
      fallbackError: 'Failed to update user',
    }
  );
}

export async function listAdminInventoryItems(
  token?: string | null
): Promise<InventoryItem[]> {
  const data = await requestJson<{ items?: Array<Record<string, unknown>> }>(
    '/admin/inventory/items',
    {},
    { token, fallbackError: 'Failed to load admin inventory' }
  );
  return (data.items ?? []).map((item) => mapApiInventoryItem(item, token));
}

export async function createAdminInventoryItem(
  payload: AdminFurnitureCreatePayload,
  token?: string | null
): Promise<InventoryItem> {
  const item = await requestJson<Record<string, unknown>>(
    '/admin/inventory/items',
    {
      method: 'POST',
      body: JSON.stringify(payload),
    },
    {
      token,
      includeJsonContentType: true,
      fallbackError: 'Failed to create furniture item',
    }
  );
  return mapApiInventoryItem(item, token);
}

export async function fetchInventory(token?: string | null): Promise<InventoryItem[]> {
  const data = await requestJson<{ items?: Array<Record<string, unknown>> }>(
    '/inventory/items',
    {},
    { token, fallbackError: 'Failed to load inventory' }
  );
  const items = (data?.items ?? []) as Array<Record<string, unknown>>;
  return items.map((item) => mapApiInventoryItem(item, token));
}

function mapApiInventoryItem(
  item: Record<string, unknown>,
  token?: string | null
): InventoryItem {
  const dims = (item.dimensions ?? {}) as Record<string, unknown>;
  const attributes = appendTokenToAttributes(
    (item.attributes ?? {}) as Record<string, unknown>,
    token
  );
  return {
    asset_id: String(item.id ?? ''),
    name: String(item.name ?? ''),
    category: String(item.type ?? ''),
    footprint_mm: {
      w: Number(dims.width_mm ?? 0),
      d: Number(dims.length_mm ?? 0),
    },
    height_mm: Number(dims.height_mm ?? 0),
    top_view: { type: 'rect' },
    style_tags: Array.isArray(item.style_tags) ? (item.style_tags as string[]) : [],
    material: typeof item.material === 'string' ? item.material : null,
    material_tags: Array.isArray(attributes.material_tags)
      ? (attributes.material_tags as string[])
      : [],
    price: readInventoryPrice(item.price),
    color_hex: typeof attributes.color_hex === 'string' ? attributes.color_hex : null,
    room_type: typeof attributes.room_type === 'string' ? attributes.room_type : null,
    attributes,
  };
}

function readInventoryPrice(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    const amount = Number((value as Record<string, unknown>).amount);
    return Number.isFinite(amount) ? amount : null;
  }
  return null;
}

export async function runPipeline(
  payload: {
    user_id: string;
    input_payload: Record<string, unknown>;
    case_name?: string | null;
    description?: string | null;
    special_notes?: string | null;
  },
  token?: string | null
): Promise<{ case_id: string; case_dir: string; status: string }> {
  return requestJson<{ case_id: string; case_dir: string; status: string }>(
    '/pipeline/run',
    {
      method: 'POST',
      body: JSON.stringify(payload),
    },
    {
      token,
      includeJsonContentType: true,
      fallbackError: 'Failed to start pipeline',
    }
  );
}

export async function getPipelineStatus(caseId: string, token?: string | null): Promise<{
  case_id: string;
  stage: string;
  updated_at_utc: string;
  error?: string | null;
  message?: string | null;
  progress_current?: number | null;
  progress_total?: number | null;
  actions?: Array<{
    stage: string;
    message: string;
    updated_at_utc: string;
    progress_current?: number | null;
    progress_total?: number | null;
    error?: string | null;
  }>;
}> {
  return requestJson(
    `/pipeline/${caseId}/status`,
    {},
    { token, fallbackError: 'Failed to fetch status' }
  );
}

export async function getPipelineResult(caseId: string, token?: string | null): Promise<{ result: any }> {
  return requestJson(`/pipeline/${caseId}/result`, {}, { token, fallbackError: 'Failed to fetch result' });
}

export async function getPipelineArtifact(caseId: string, name: string, token?: string | null): Promise<{ payload: any }> {
  return requestJson(
    `/pipeline/${caseId}/artifact/${name}`,
    {},
    { token, fallbackError: 'Failed to fetch artifact' }
  );
}

export async function renderSnapshotImage(
  request: SnapshotImageRenderClientRequest,
  token?: string | null
): Promise<SnapshotImageRenderApiResponse> {
  const response = await requestJson<SnapshotImageRenderApiResponse>(
    '/pipeline/render-snapshot-image',
    {
      method: 'POST',
      body: JSON.stringify(buildSnapshotImageRenderBody(request)),
    },
    {
      token,
      includeJsonContentType: true,
      fallbackError: 'Failed to render snapshot image',
    }
  );
  return appendSnapshotRenderAccessToken(response, token);
}

export async function renderSnapshotImageStream(
  request: SnapshotImageRenderClientRequest,
  handlers: SnapshotImageStreamHandlers = {},
  token?: string | null
): Promise<SnapshotImageRenderApiResponse> {
  let res: Response;
  try {
    res = await fetch(
      resolveApiUrl('/pipeline/render-snapshot-image/stream'),
      withAuthHeaders(
        {
          method: 'POST',
          body: JSON.stringify(buildSnapshotImageRenderBody(request)),
        },
        token,
        true
      )
    );
  } catch (error) {
    const detail =
      error instanceof Error && error.message.trim()
        ? error.message.trim()
        : 'Network request failed';
    throw new Error(
      `Failed to stream snapshot image. Cannot reach API at ${API_BASE}. ${detail}`
    );
  }
  if (!res.ok) {
    throw await parseError(res, 'Failed to stream snapshot image');
  }
  if (!res.body) {
    throw new Error('Snapshot image stream response did not include a body.');
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let completed: SnapshotImageRenderApiResponse | null = null;

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
    const blocks = buffer.replace(/\r\n/g, '\n').split('\n\n');
    buffer = blocks.pop() ?? '';
    for (const block of blocks) {
      const event = parseSnapshotImageSseBlock(block);
      if (!event) {
        continue;
      }
      if (event.eventName === 'status') {
        handlers.onStatus?.(
          event.data as { message: string; attempt: string | null }
        );
      } else if (event.eventName === 'partial_image') {
        handlers.onPartialImage?.(event.data as SnapshotImagePartialImageEvent);
      } else if (event.eventName === 'completed') {
        completed = appendSnapshotRenderAccessToken(
          event.data as SnapshotImageRenderApiResponse,
          token
        );
        handlers.onCompleted?.(completed);
      } else if (event.eventName === 'error') {
        const errorEvent = event.data as { message?: string };
        const message = errorEvent.message?.trim() || 'Snapshot image stream failed.';
        handlers.onError?.({ message });
        throw new Error(message);
      }
    }
    if (done) {
      break;
    }
  }

  const trailingEvent = parseSnapshotImageSseBlock(buffer);
  if (trailingEvent?.eventName === 'completed') {
    completed = appendSnapshotRenderAccessToken(
      trailingEvent.data as SnapshotImageRenderApiResponse,
      token
    );
    handlers.onCompleted?.(completed);
  }
  if (!completed) {
    throw new Error('Snapshot image stream ended before completion.');
  }
  return completed;
}

function buildSnapshotImageRenderBody(
  request: SnapshotImageRenderClientRequest
): Record<string, unknown> {
  return {
    snapshot_payload: request.snapshotPayload,
    snapshot_image_data_url: request.snapshotImageDataUrl,
    depth_map_image_data_url: request.depthMapImageDataUrl ?? null,
    edge_map_image_data_url: request.edgeMapImageDataUrl ?? null,
    segmentation_map_image_data_url: request.segmentationMapImageDataUrl ?? null,
    annotated_reference_image_data_url:
      request.annotatedReferenceImageDataUrl ?? null,
    scene_reference_image_data_url: request.sceneReferenceImageDataUrl ?? null,
    scene_reference_mode: request.sceneReferenceMode ?? 'none',
    user_prompt: request.userPrompt,
    render_mode: request.renderMode ?? 'generate',
    render_flow: request.renderFlow ?? 'default',
    preset_selection: request.presetSelection ?? {},
    root_layout_id: request.rootLayoutId ?? null,
    case_id: request.caseId ?? null,
    option_id: request.optionId ?? null,
    edit_operations: request.editOperations ?? [],
    locked_geometry_references: request.lockedGeometryReferences ?? [],
    locked_geometry_atlases: request.lockedGeometryAtlases ?? [],
    edit_source_image_data_url: request.editSourceImageDataUrl ?? null,
    image_quality: request.imageQuality ?? null,
    image_size: request.imageSize ?? null,
  };
}

function appendSnapshotRenderAccessToken(
  response: SnapshotImageRenderApiResponse,
  token?: string | null
): SnapshotImageRenderApiResponse {
  return {
    ...response,
    saved_render: response.saved_render
      ? {
          ...response.saved_render,
          file_url: appendAccessToken(response.saved_render.file_url, token),
        }
      : response.saved_render,
  };
}

function parseSnapshotImageSseBlock(
  block: string
): { eventName: string; data: unknown } | null {
  const lines = block.split('\n');
  let eventName = 'message';
  const dataLines: string[] = [];
  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (line.startsWith('event:')) {
      eventName = line.slice('event:'.length).trim();
    } else if (line.startsWith('data:')) {
      dataLines.push(line.slice('data:'.length).trim());
    }
  }
  const rawData = dataLines.join('\n').trim();
  if (!rawData || rawData === '[DONE]') {
    return null;
  }
  return {
    eventName,
    data: JSON.parse(rawData) as unknown,
  };
}

export async function fetchRenderPresets(
  token?: string | null
): Promise<SnapshotImagePresets> {
  return requestJson<SnapshotImagePresets>(
    '/pipeline/render-presets',
    {},
    { token, fallbackError: 'Failed to load image presets' }
  ).then(normalizePresetReferenceImages);
}

export async function listSavedLayouts(
  token?: string | null,
  caseId?: string | null
): Promise<SavedLayoutRecord[]> {
  const query = caseId ? `?case_id=${encodeURIComponent(caseId)}` : '';
  return requestJson<SavedLayoutRecord[]>(
    `/account/layouts${query}`,
    {},
    { token, fallbackError: 'Failed to load saved layouts' }
  );
}

export async function updateSavedLayoutCase(
  caseId: string,
  payload: { name: string },
  token?: string | null
): Promise<{ id: string; name: string; created_at?: string | null; updated_at?: string | null }> {
  return requestJson(
    `/account/layout-cases/${encodeURIComponent(caseId)}`,
    {
      method: 'PATCH',
      body: JSON.stringify(payload),
    },
    {
      token,
      includeJsonContentType: true,
      fallbackError: 'Failed to update case name',
    }
  );
}

export async function getGeneratedRenderDetail(
  renderId: string,
  token?: string | null
): Promise<GeneratedRenderDetailResponse> {
  const response = await requestJson<GeneratedRenderDetailResponse>(
    `/account/renders/${renderId}`,
    {},
    { token, fallbackError: 'Failed to load saved render' }
  );
  return {
    ...response,
    render: {
      ...response.render,
      file_url: appendAccessToken(response.render.file_url, token),
    },
  };
}

export async function updateGeneratedRenderMeta(
  renderId: string,
  meta: Record<string, unknown>,
  token?: string | null
): Promise<GeneratedRenderRecord> {
  const response = await requestJson<GeneratedRenderRecord>(
    `/account/renders/${encodeURIComponent(renderId)}/meta`,
    {
      method: 'PATCH',
      body: JSON.stringify({ meta }),
    },
    {
      token,
      includeJsonContentType: true,
      fallbackError: 'Failed to update saved render metadata',
    }
  );
  return {
    ...response,
    file_url: appendAccessToken(response.file_url, token),
  };
}

export async function createSavedLayout(
  payload: {
    name: string;
    floorplan_json: Record<string, unknown>;
    design_json?: Record<string, unknown> | null;
    styled_result_json?: Record<string, unknown> | null;
    meta?: Record<string, unknown>;
  },
  token?: string | null
): Promise<SavedLayoutRecord> {
  return requestJson<SavedLayoutRecord>(
    '/account/layouts',
    {
      method: 'POST',
      body: JSON.stringify(payload),
    },
    {
      token,
      includeJsonContentType: true,
      fallbackError: 'Failed to save layout',
    }
  );
}

export async function updateSavedLayout(
  layoutId: string,
  payload: {
    name: string;
    floorplan_json: Record<string, unknown>;
    design_json?: Record<string, unknown> | null;
    styled_result_json?: Record<string, unknown> | null;
    meta?: Record<string, unknown>;
  },
  token?: string | null
): Promise<SavedLayoutRecord> {
  return requestJson<SavedLayoutRecord>(
    `/account/layouts/${encodeURIComponent(layoutId)}`,
    {
      method: 'PATCH',
      body: JSON.stringify(payload),
    },
    {
      token,
      includeJsonContentType: true,
      fallbackError: 'Failed to update layout',
    }
  );
}

export async function listGeneratedRenders(
  token?: string | null
): Promise<GeneratedRenderRecord[]> {
  const renders = await requestJson<GeneratedRenderRecord[]>(
    '/account/renders',
    {},
    { token, fallbackError: 'Failed to load saved renders' }
  );
  return renders.map((render) => ({
    ...render,
    file_url: appendAccessToken(render.file_url, token),
  }));
}
