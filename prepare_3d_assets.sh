#!/usr/bin/env bash
set -euo pipefail

ASSETS_DIR="/Users/Yuki/TKNT/assets/three"
mkdir -p "$ASSETS_DIR"

THREE_URL="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js"
ORBIT_URL="https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js"

curl -L "$THREE_URL" -o "$ASSETS_DIR/three.module.js"
curl -L "$ORBIT_URL" -o "$ASSETS_DIR/OrbitControls.js"

# Patch OrbitControls to use local three.module.js
sed -i '' "s/from 'three'/from '\\.\\/three.module.js'/" "$ASSETS_DIR/OrbitControls.js"

echo "Downloaded to $ASSETS_DIR"
