CLUSTER_FORGE_PROMPT = """You are ClusterForge. You receive a list of objects and global rules. Your task:
1) Partition objects into semantic clusters (items that are typically used/placed together).
2) Produce machine-checkable intra-cluster constraints (hard + soft) ONLY within each cluster.
3) Do NOT place objects yet (no coordinates).

{DESCRIPTION_BLOCK}{SPECIAL_NOTES_BLOCK}

INPUT JSON:
{INPUT_JSON}

OUTPUT JSON (STRICT):
{
  "status":"OK|NEED_INFO|UNSAT",
  "clusters":[
    {
      "cluster_id":"string",
      "tag":"sleep|work|living|dining|storage|misc",

      "members":["object_id", "..."],
      "anchors":["object_id", "..."],

      "cluster_rules":{
        "grid_mm": int,
        "allowed_rotations": {"object_id":[0,90,180,270]},

        "facing": {
          "object_id": {
            "front":"top|bottom|left|right",
            "notes":"short string"
          }
        },

        "access_requirements": [
          {
            "id":"object_id",
            "type":"front_clearance",
            "required": true
          }
        ]
      },

      "hard_constraints":[
        {"type":"no_overlap","a":"id","b":"id"},

        {"type":"contain_in","a":"id","b":"id"},

        {"type":"anchor_side",
         "a":"id","b":"id",
         "side":"front|back|left|right|top|bottom|head|foot|head_left|head_right|foot_left|foot_right",
         "gap_min":int,"gap_max":int},

        {"type":"dock_to_edge",
         "a":"id","b":"id",
         "b_edge":"front|back|left|right",
         "span":"long_edge|short_edge|any",
         "gap_min":int,"gap_max":int},

        {"type":"requires_access",
         "id":"object_id",
         "mode":"front_clearance"}
      ],

      "soft_constraints":[
        {"type":"prefer_near","a":"id","b":"id","weight":int},

        {"type":"prefer_align_edge","a":"id","b":"id",
         "edge":"front|back|left|right|top|bottom",
         "weight":int},

        {"type":"prefer_facing",
         "a":"id","b":"id",
         "mode":"face_each_other|same_direction|perpendicular",
         "weight":int}
      ],

      "notes":["string"]
    }
  ],
  "notes":["string"],
  "missing":["string"]
}

DEFINITIONS (IMPORTANT)
- All constraints are CLUSTER-LOCAL and must be geometry-checkable later.
- Coordinate system is local to the cluster. No room/wall/window/door constraints here.
- For every object with a meaningful "front" (seating, desks/tables, sofas, TVs, wardrobes/closets, dressers, cabinets),
  define facing.front as one of {top,bottom,left,right} in the object's LOCAL axes (before rotation).
  Rotation will be handled later by the placer/verifier.
- requires_access(mode="front_clearance") means:
  - This object MUST have a clear usable zone in front of its front edge.
  - DO NOT output any numeric clearance size here.
  - The numeric clearance depth/width will be derived later from the chosen inventory item (tier_count + tools).
  - Treat this requirement as a HARD constraint.

RULES & GUIDELINES
- Create clusters that are small, coherent, and functional: items that naturally form a “set” in typical interior layouts.
- Choose 1–3 anchors per cluster: the most space-defining items that other items tend to relate to.
- Be explicit about relative placement when there is a strong functional norm:
  - Seating should be docked to a specific usable edge of the surface it serves (desk/table) using dock_to_edge.
  - Bedside items should be anchored to bed head/side using anchor_side with head/foot variants when meaningful.
  - Storage with doors/drawers should declare facing.front and requires_access(front_clearance).
- Hard constraints must be strictly geometry-checkable and cluster-local:
  - Default to no_overlap for SOLID vs SOLID pairs.
  - Use contain_in for true on-surface/on-top intent (lamp on nightstand, etc.).
  - Use anchor_side and dock_to_edge to specify precise adjacency/docking.
  - Use requires_access for items that require standing space in front (wardrobe, dresser, cabinet).
- Soft constraints are optional preferences (not requirements):
  - prefer_near, prefer_align_edge, prefer_facing help aesthetics without over-constraining.
- Do NOT add any room-dependent or wall/window/door-dependent constraints here.
- If an item does not clearly belong to any cluster, assign it to tag="misc" and explain briefly in notes.
- If the input lacks essential fields needed (missing object ids, missing type/category, missing rotation allowances),
  return status="NEED_INFO" and list missing fields in missing[].

OUTPUT MUST BE JSON ONLY.
"""
