export type Point = { x: number; y: number };
export type DirectionVector = { dx: number; dy: number };
export type CardinalSide = 'top' | 'bottom' | 'left' | 'right';

export type Room = {
  room_id: string;
  name: string;
  polygon_mm: Point[];
  floor_color_hex?: string | null;
};

export type Wall = {
  wall_id: string;
  p1_mm: Point;
  p2_mm: Point;
  thickness_mm: number;
  room_id?: string | null;
  locked?: boolean | null;
  color_hex?: string | null;
  opacity?: number | null;
};

export type Door = {
  door_id: string;
  wall_id?: string | null;
  start_mm: Point;
  end_mm: Point;
  color_hex?: string | null;
  leaf_width_mm?: number | null;
  hinge_side?: 'left' | 'right' | null;
  open_angle_deg?: number | null;
  open_direction?: 'in' | 'out' | null;
};

export type Window = {
  window_id: string;
  wall_id?: string | null;
  start_mm: Point;
  end_mm: Point;
  color_hex?: string | null;
  sill_height_mm?: number | null;
  height_mm?: number | null;
};

export type FloorplanGeometry = {
  room: Room;
  walls: Wall[];
  openings: {
    doors: Door[];
    windows: Window[];
  };
};

export type Footprint = { w: number; d: number };

export type PlacementSource = 'generated' | 'manual';

export type PlacementConstraintMode = 'free' | 'pinned' | 'locked';

export type Placement = {
  placement_id: string;
  asset_id: string;
  category: string;
  position_mm: Point;
  rotation_deg: number;
  front_world?: DirectionVector | null;
  front_side_world?: CardinalSide | null;
  front_side_local?: CardinalSide | null;
  footprint_mm: Footprint;
  height_mm: number;
  color_hex: string;
  anchor: 'center';
  source?: PlacementSource;
  constraint_mode?: PlacementConstraintMode;
  place_on?: {
    method: string;
    target_instance_id?: string | null;
  } | null;
  collision_layer?: string | null;
};

export type TopView = {
  type: 'rect' | 'poly';
  points?: Point[] | null;
};

export type InventoryItem = {
  asset_id: string;
  name: string;
  category: string;
  footprint_mm: Footprint;
  height_mm: number;
  top_view: TopView;
  style_tags: string[];
  material?: string | null;
  material_tags: string[];
  price?: number | null;
  color_hex?: string | null;
  room_type?: string | null;
  attributes?: Record<string, unknown> | null;
};

export type AuthUser = {
  id: string;
  email: string;
  display_name?: string | null;
  tenant_id: string;
};

export type SavedLayoutRecord = {
  id: string;
  name: string;
  floorplan_json: Record<string, unknown>;
  design_json?: Record<string, unknown> | null;
  styled_result_json?: Record<string, unknown> | null;
  meta: Record<string, unknown>;
  created_at?: string | null;
  updated_at?: string | null;
};

export type GeneratedRenderRecord = {
  id: string;
  source: string;
  model_name: string;
  prompt: string;
  negative_prompt?: string | null;
  mime_type: string;
  meta: Record<string, unknown>;
  created_at?: string | null;
  file_url: string;
};

export type TemplateItem = {
  template_id: string;
  name: string;
  room_type?: string | null;
  supported_shapes: string[];
  style_tags: string[];
};

export type DesignPlanV1 = {
  floorplan_geometry: FloorplanGeometry;
  placements: Placement[];
};

export type LayoutVariant = {
  variant_id: string;
  label: string;
  source: string;
  reason: string;
  layout_score: number;
  hard_valid: boolean;
  complete?: boolean;
  gallery_eligible?: boolean;
  coverage_ratio?: number;
  missing_cluster_ids?: string[];
  notes: string[];
  absolute_layout: Record<string, unknown>;
  styled_result: Record<string, unknown>;
  design: DesignPlanV1 | null;
  cluster_previews?: Record<string, unknown>;
};

export type Violation = {
  severity: 'error' | 'warning' | 'needs_info';
  code: string;
  message: string;
  placement_ids: string[];
  details: Record<string, unknown>;
  suggested_fixes: { action_type: string; params: Record<string, unknown> }[];
};
