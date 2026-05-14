import {
  AuthUser,
  DesignPlanV1,
  FloorplanGeometry,
  GeneratedRenderRecord,
  InventoryItem,
  SavedLayoutRecord,
  Violation,
} from './types';

const API_BASE = (import.meta as any).env?.VITE_API_BASE ?? 'http://localhost:8001';
const AUTH_TOKEN_STORAGE_KEY = 'tknt.authToken';

export type SnapshotImageSceneReferenceMode =
  | 'none'
  | 'target_layout_with_scene_reference'
  | 'scene_reference_camera_transfer';

export type AuthSessionResponse = {
  access_token: string;
  token_type: 'bearer';
  user: AuthUser;
};

export type Object3DCreateResponse = {
  job_id: string;
  job_dir: string;
  stage: string;
  status_url: string;
};

export type Object3DStatusResponse = {
  job_id: string;
  job_dir: string;
  mode: string;
  stage: string;
  updated_at_utc: string;
  message?: string | null;
  error?: string | null;
  progress_current?: number | null;
  progress_total?: number | null;
  artifact_paths: Record<string, string>;
  artifact_urls: Record<string, string>;
  result?: Record<string, unknown> | null;
  actions: Array<{
    stage: string;
    message: string;
    updated_at_utc: string;
    progress_current?: number | null;
    progress_total?: number | null;
    error?: string | null;
  }>;
};

export type SnapshotImageRenderResult = {
  request: {
    render_mode: 'generate' | 'edit';
    system_prompt: string;
    user_prompt: string;
    raw_user_prompt?: string | null;
    preset_prompt?: string | null;
    preset_selection?: SnapshotImagePresetSelection;
    aspect_ratio: string;
    source_image_mime_type: string;
    structural_image_mime_type?: string | null;
    layout_reference_enabled: boolean;
    layout_reference_used: boolean;
    layout_reference_image_mime_type?: string | null;
    annotated_reference_used?: boolean;
    annotated_reference_image_mime_type?: string | null;
    scene_reference_mode?: SnapshotImageSceneReferenceMode;
    scene_reference_used?: boolean;
    reference_only_camera_transfer_used?: boolean;
    scene_reference_image_mime_type?: string | null;
    reference_image_count?: number;
    edit_operations?: SnapshotImageEditOperationSummary[];
  };
  image: {
    mime_type: string;
    image_base64: string;
    data_url: string;
    size: string;
    quality?: string;
    aspect_ratio: string;
    raw_response: Record<string, unknown>;
  };
  models: {
    image_model_name: string;
    max_tokens: number;
  };
  metadata?: {
    camera?: Record<string, unknown> | null;
    visible_objects?: Array<Record<string, unknown>>;
    visible_object_ids?: string[];
    render_mode?: 'generate' | 'edit';
    preset_selection?: SnapshotImagePresetSelection;
    edit_operations?: SnapshotImageEditOperationSummary[];
  };
};

export type SnapshotImageRenderApiResponse = {
  render: SnapshotImageRenderResult;
  saved_render?: GeneratedRenderRecord | null;
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
  operation_type?: 'prompt' | 'add' | 'replace' | 'recolor' | null;
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

export type SnapshotImageEditOperationSummary = {
  object_id: string;
  object_name?: string | null;
  operation_type?: 'prompt' | 'add' | 'replace' | 'recolor' | null;
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
      `${API_BASE}${path}`,
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

export function appendAccessToken(url: string, token?: string | null): string {
  const rawUrl = url.trim();
  if (!rawUrl) {
    return url;
  }

  const resolvedToken = resolveAuthToken(token);
  const isAbsoluteUrl = /^https?:\/\//i.test(rawUrl);
  const isApiRelativeUrl = /^\/(inventory|object-3d|account|pipeline|auth)(\/|$)/.test(rawUrl);
  const base = isAbsoluteUrl
    ? undefined
    : isApiRelativeUrl
      ? API_BASE
      : typeof window !== 'undefined'
        ? window.location.origin
        : API_BASE;
  const absolute = isAbsoluteUrl ? new URL(rawUrl) : new URL(rawUrl, base);

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
  return new URL(rawUrl.startsWith('/') ? rawUrl : `/${rawUrl}`, API_BASE).toString();
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

export async function fetchInventory(
  tenantId: string,
  token?: string | null
): Promise<InventoryItem[]> {
  const data = await requestJson<{ items?: Array<Record<string, unknown>> }>(
    `/inventory/items?tenant_id=${encodeURIComponent(tenantId)}`,
    {},
    { token, fallbackError: 'Failed to load inventory' }
  );
  const items = (data?.items ?? []) as Array<Record<string, unknown>>;
  return items.map((item) => {
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
      price: null,
      color_hex: typeof attributes.color_hex === 'string' ? attributes.color_hex : null,
      room_type: typeof attributes.room_type === 'string' ? attributes.room_type : null,
      attributes,
    } satisfies InventoryItem;
  });
}

export async function createObject3DFromPrompt(
  payload: {
    name: string;
    category: string;
    width_mm: number;
    depth_mm: number;
    height_mm: number;
    prompt: string;
    negative_prompt?: string | null;
    keep_debug_artifacts?: boolean;
    triposr_device?: string;
    triposr_chunk_size?: number | null;
    triposr_mc_resolution?: number;
    triposr_mc_threshold?: number;
  },
  token?: string | null
): Promise<Object3DCreateResponse> {
  return requestJson<Object3DCreateResponse>(
    '/object-3d/jobs/from-prompt',
    {
      method: 'POST',
      body: JSON.stringify(payload),
    },
    {
      token,
      includeJsonContentType: true,
      fallbackError: 'Failed to start 3D generation from prompt',
    }
  );
}

export async function createObject3DFromImage(
  payload: {
    file: File;
    name: string;
    category: string;
    width_mm: number;
    depth_mm: number;
    height_mm: number;
    keep_debug_artifacts?: boolean;
    triposr_device?: string;
    triposr_chunk_size?: number | null;
    triposr_mc_resolution?: number;
    triposr_mc_threshold?: number;
  },
  token?: string | null
): Promise<Object3DCreateResponse> {
  const formData = new FormData();
  formData.set('image', payload.file);
  formData.set('name', payload.name);
  formData.set('category', payload.category);
  formData.set('width_mm', String(payload.width_mm));
  formData.set('depth_mm', String(payload.depth_mm));
  formData.set('height_mm', String(payload.height_mm));
  if (payload.keep_debug_artifacts !== undefined) {
    formData.set('keep_debug_artifacts', String(payload.keep_debug_artifacts));
  }
  if (payload.triposr_device) {
    formData.set('triposr_device', payload.triposr_device);
  }
  if (payload.triposr_chunk_size !== undefined && payload.triposr_chunk_size !== null) {
    formData.set('triposr_chunk_size', String(payload.triposr_chunk_size));
  }
  if (payload.triposr_mc_resolution !== undefined) {
    formData.set('triposr_mc_resolution', String(payload.triposr_mc_resolution));
  }
  if (payload.triposr_mc_threshold !== undefined) {
    formData.set('triposr_mc_threshold', String(payload.triposr_mc_threshold));
  }
  return requestJson<Object3DCreateResponse>(
    '/object-3d/jobs/from-image',
    {
      method: 'POST',
      body: formData,
    },
    {
      token,
      fallbackError: 'Failed to start 3D generation from image',
    }
  );
}

export async function getObject3DJobStatus(
  jobId: string,
  token?: string | null
): Promise<Object3DStatusResponse> {
  const status = await requestJson<Object3DStatusResponse>(
    `/object-3d/jobs/${jobId}`,
    {},
    { token, fallbackError: 'Failed to fetch 3D job status' }
  );
  return {
    ...status,
    artifact_urls: Object.fromEntries(
      Object.entries(status.artifact_urls ?? {}).map(([key, value]) => [
        key,
        appendAccessToken(value, token),
      ])
    ),
  };
}

export async function runPipeline(
  payload: {
    user_id: string;
    input_payload: Record<string, unknown>;
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

export async function getClusterList(caseId: string, token?: string | null): Promise<{ case_id: string; clusters: string[] }> {
  return requestJson(
    `/pipeline/${caseId}/clusters`,
    {},
    { token, fallbackError: 'Failed to fetch clusters' }
  );
}

export async function getClusterComposer(caseId: string, clusterId: string, token?: string | null): Promise<any> {
  return requestJson(
    `/pipeline/${caseId}/clusters/${clusterId}`,
    {},
    { token, fallbackError: 'Failed to fetch cluster composer' }
  );
}

export async function renderSnapshotImage(
  request: {
    snapshotPayload: Record<string, unknown>;
    snapshotImageDataUrl: string;
    layoutReferenceImageDataUrl?: string | null;
    annotatedReferenceImageDataUrl?: string | null;
    sceneReferenceImageDataUrl?: string | null;
    sceneReferenceMode?: SnapshotImageSceneReferenceMode;
    userPrompt: string;
    renderMode?: 'generate' | 'edit';
    presetSelection?: SnapshotImagePresetSelection;
    rootLayoutId?: string | null;
    editOperations?: SnapshotImageEditOperation[];
    editSourceImageDataUrl?: string | null;
  },
  token?: string | null
): Promise<SnapshotImageRenderApiResponse> {
  return requestJson<SnapshotImageRenderApiResponse>(
    '/pipeline/render-snapshot-image',
    {
      method: 'POST',
      body: JSON.stringify({
        snapshot_payload: request.snapshotPayload,
        snapshot_image_data_url: request.snapshotImageDataUrl,
        layout_reference_image_data_url:
          request.layoutReferenceImageDataUrl ?? null,
        annotated_reference_image_data_url:
          request.annotatedReferenceImageDataUrl ?? null,
        scene_reference_image_data_url:
          request.sceneReferenceImageDataUrl ?? null,
        scene_reference_mode: request.sceneReferenceMode ?? 'none',
        user_prompt: request.userPrompt,
        render_mode: request.renderMode ?? 'generate',
        preset_selection: request.presetSelection ?? {},
        root_layout_id: request.rootLayoutId ?? null,
        edit_operations: request.editOperations ?? [],
        edit_source_image_data_url: request.editSourceImageDataUrl ?? null,
      }),
    },
    {
      token,
      includeJsonContentType: true,
      fallbackError: 'Failed to render snapshot image',
    }
  ).then((response) => ({
    ...response,
    saved_render: response.saved_render
      ? {
          ...response.saved_render,
          file_url: appendAccessToken(response.saved_render.file_url, token),
        }
      : response.saved_render,
  }));
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

export async function listSavedLayouts(token?: string | null): Promise<SavedLayoutRecord[]> {
  return requestJson<SavedLayoutRecord[]>(
    '/account/layouts',
    {},
    { token, fallbackError: 'Failed to load saved layouts' }
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

export async function getSavedLayout(
  layoutId: string,
  token?: string | null
): Promise<SavedLayoutRecord> {
  return requestJson<SavedLayoutRecord>(
    `/account/layouts/${layoutId}`,
    {},
    { token, fallbackError: 'Failed to load saved layout' }
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

export async function saveUiSnapshot(_payload: {
  tenant_id: string;
  iteration: number | null;
  attempt: number | null;
  image_base64: string;
}): Promise<void> {
  // Snapshot endpoint removed in new backend; no-op to avoid fetch errors.
}

export async function checkDesign(_payload: {
  floorplan_geometry: FloorplanGeometry;
  placements: DesignPlanV1['placements'];
  constraints?: Record<string, unknown>;
}): Promise<Violation[]> {
  return [];
}
