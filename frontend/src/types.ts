export type Point = { x: number; y: number };

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
};

export type Door = {
  door_id: string;
  wall_id?: string | null;
  start_mm: Point;
  end_mm: Point;
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

export type Placement = {
  placement_id: string;
  asset_id: string;
  category: string;
  position_mm: Point;
  rotation_deg: number;
  footprint_mm: Footprint;
  height_mm: number;
  color_hex: string;
  anchor: 'center';
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
  material_tags: string[];
  price?: number | null;
  color_hex?: string | null;
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

export type Violation = {
  severity: 'error' | 'warning' | 'needs_info';
  code: string;
  message: string;
  placement_ids: string[];
  details: Record<string, unknown>;
  suggested_fixes: { action_type: string; params: Record<string, unknown> }[];
};
