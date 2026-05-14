# TKNT Floorplan Editor (V1)

## Run
```bash
cd frontend
npm install
npm run dev
```

## AI render scene reference flags
Set one of these before starting Vite:

```bash
VITE_TKNT_RENDER_USE_FULL_SCENE_REFERENCE=true npm run dev
VITE_TKNT_RENDER_USE_REFERENCE_ONLY_CAMERA_TRANSFER=true npm run dev
```

`VITE_TKNT_RENDER_USE_FULL_SCENE_REFERENCE` keeps the current 3D snapshot as the layout source and sends the loaded render as a full-scene continuity reference. `VITE_TKNT_RENDER_USE_REFERENCE_ONLY_CAMERA_TRANSFER` uses the loaded render as the source image and asks the image model to transfer it to the current 3D camera from metadata. If both are enabled, reference-only camera transfer wins.

## Backend
Run API:
```bash
PYTHONPATH=/Users/Yuki/TKNT /Users/Yuki/TKNT/.conda_envs/prj-tknt/bin/python -m uvicorn api.main:app --reload --port 8001
```

Then open `http://localhost:5173`.
