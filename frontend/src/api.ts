import { DesignPlanV1, FloorplanGeometry, InventoryItem, Violation } from './types';

const API_BASE = (import.meta as any).env?.VITE_API_BASE ?? 'http://localhost:8000';

export async function fetchInventory(tenantId: string): Promise<InventoryItem[]> {
  const res = await fetch(`${API_BASE}/inventory/items?tenant_id=${tenantId}`);
  if (!res.ok) throw new Error('Failed to load inventory');
  const data = await res.json();
  const items = (data?.items ?? []) as any[];
  return items.map((item) => {
    const dims = item.dimensions ?? {};
    const w = Number(dims.length_mm ?? 0);
    const d = Number(dims.width_mm ?? 0);
    const h = Number(dims.height_mm ?? 0);
    return {
      asset_id: item.id,
      name: item.name,
      category: item.type,
      footprint_mm: { w, d },
      height_mm: h,
      top_view: { type: 'rect' },
      style_tags: item.style_tags ?? [],
      material_tags: item.attributes?.material_tags ?? [],
      price: null,
      color_hex: item.attributes?.color_hex ?? null,
      room_type: item.attributes?.room_type ?? null,
      attributes: item.attributes ?? {},
    } satisfies InventoryItem;
  });
}

export async function runPipeline(payload: {
  user_id: string;
  input_payload: Record<string, unknown>;
  description?: string | null;
  special_notes?: string | null;
}): Promise<{ case_id: string; case_dir: string; status: string }> {
  const res = await fetch(`${API_BASE}/pipeline/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error('Failed to start pipeline');
  return res.json();
}

export async function getPipelineStatus(caseId: string): Promise<{
  case_id: string;
  stage: string;
  updated_at_utc: string;
  error?: string | null;
}> {
  const res = await fetch(`${API_BASE}/pipeline/${caseId}/status`);
  if (!res.ok) throw new Error('Failed to fetch status');
  return res.json();
}

export async function getPipelineResult(caseId: string): Promise<{ result: any }> {
  const res = await fetch(`${API_BASE}/pipeline/${caseId}/result`);
  if (!res.ok) throw new Error('Failed to fetch result');
  return res.json();
}

export async function getPipelineArtifact(caseId: string, name: string): Promise<{ payload: any }> {
  const res = await fetch(`${API_BASE}/pipeline/${caseId}/artifact/${name}`);
  if (!res.ok) throw new Error('Failed to fetch artifact');
  return res.json();
}

export async function getClusterList(caseId: string): Promise<{ case_id: string; clusters: string[] }> {
  const res = await fetch(`${API_BASE}/pipeline/${caseId}/clusters`);
  if (!res.ok) throw new Error('Failed to fetch clusters');
  return res.json();
}

export async function getClusterComposer(caseId: string, clusterId: string): Promise<any> {
  const res = await fetch(`${API_BASE}/pipeline/${caseId}/clusters/${clusterId}`);
  if (!res.ok) throw new Error('Failed to fetch cluster composer');
  return res.json();
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
  // Checker endpoint removed in new backend; return empty.
  return [];
}
