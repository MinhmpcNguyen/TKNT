# TKNT Floorplan API (V1)

## Run
```bash
PYTHONPATH=/Users/Yuki/TKNT /Users/Yuki/TKNT/.conda_envs/prj-tknt/bin/python -m uvicorn api.main:app --reload --port 8001
```

## Sample requests
```bash
curl -s "http://localhost:8001/inventory?tenant_id=demo_tenant" | jq
curl -s "http://localhost:8001/templates?tenant_id=demo_tenant" | jq

curl -s -X POST "http://localhost:8001/design/generate" \
  -H "Content-Type: application/json" \
  -d @api/sample_generate.json | jq

curl -s -X POST "http://localhost:8001/design/check" \
  -H "Content-Type: application/json" \
  -d @api/sample_check.json | jq
```
