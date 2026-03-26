import json

from agent.stylist import Stylist

case_id = "demo_tenant_20260325T134026Z"
layout_path = f"cases/{case_id}/07_absolute_layout.json"

layout_json = json.load(open(layout_path, "r", encoding="utf-8"))

stylist = Stylist()
out = stylist.generate(
    layout_json=layout_json,
    user_context_json={"user_id": "demo_tenant"},
    tenant_id=None,  # giống orchestrator
)

print("status:", out.get("status"))
print("objects:", len(out.get("objects", [])))
print("missing:", out.get("missing"))
