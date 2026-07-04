// Procedural "semi-real" proxy meshes for 3D preview (no external models).
//
// We intentionally avoid importing Three.js here. The caller passes the `THREE`
// module (vendored) so this file stays bundler-friendly.

type ProxyBuildParams = {
  type: string;
  wM: number; // width in meters (X axis)
  dM: number; // depth in meters (Z axis)
  hM: number; // height in meters (Y axis)
  colorHex: string;
  material?: string | null;
};

function _safeColor(THREE: any, hex: string, fallbackHex: string): any {
  try {
    return new THREE.Color(hex);
  } catch {
    return new THREE.Color(fallbackHex);
  }
}

function _shade(THREE: any, c: any, factor: number): any {
  const out = c.clone();
  out.multiplyScalar(factor);
  return out;
}

function _clamp(v: number, a: number, b: number): number {
  return Math.max(a, Math.min(b, v));
}

function _matFor(THREE: any, color: any, material?: string | null): any {
  const m = String(material ?? '').toLowerCase();
  // Conservative defaults: warm matte.
  let roughness = 0.78;
  let metalness = 0.08;
  let transparent = false;
  let opacity = 1.0;
  let emissive = new THREE.Color(0x000000);

  if (m.includes('metal')) {
    roughness = 0.35;
    metalness = 0.75;
  } else if (m.includes('glass')) {
    roughness = 0.12;
    metalness = 0.02;
    transparent = true;
    opacity = 0.45;
  } else if (m.includes('stone') || m.includes('marble')) {
    roughness = 0.65;
    metalness = 0.02;
  } else if (m.includes('fabric') || m.includes('cotton') || m.includes('wool') || m.includes('canvas')) {
    roughness = 0.92;
    metalness = 0.0;
  } else if (m.includes('rattan') || m.includes('bamboo')) {
    roughness = 0.85;
    metalness = 0.02;
  } else if (m.includes('plastic')) {
    roughness = 0.55;
    metalness = 0.02;
  } else if (m.includes('light')) {
    // For light fixtures, a subtle emissive helps.
    emissive = color.clone().multiplyScalar(0.08);
    roughness = 0.35;
    metalness = 0.25;
  }

  return new THREE.MeshStandardMaterial({
    color,
    roughness,
    metalness,
    transparent,
    opacity,
    emissive,
  });
}

function _box(THREE: any, w: number, h: number, d: number, mat: any): any {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function _cyl(THREE: any, rTop: number, rBot: number, h: number, mat: any, radialSeg = 16): any {
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(rTop, rBot, h, radialSeg), mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function _openCyl(THREE: any, rTop: number, rBot: number, h: number, mat: any, radialSeg = 16): any {
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(rTop, rBot, h, radialSeg, 1, true), mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function _sphere(THREE: any, r: number, mat: any, seg = 16): any {
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(r, seg, seg), mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function _cone(THREE: any, r: number, h: number, mat: any, radialSeg = 18): any {
  const mesh = new THREE.Mesh(new THREE.ConeGeometry(r, h, radialSeg), mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function _asKey(type: string): string {
  return String(type ?? '').trim().toLowerCase();
}

function _hasAny(t: string, needles: string[]): boolean {
  return needles.some((n) => t.includes(n));
}

function _addLegs(
  THREE: any,
  group: any,
  opts: {
    w: number;
    d: number;
    legH: number;
    legR: number;
    inset: number;
    mat: any;
  }
): void {
  const { w, d, legH, legR, inset, mat } = opts;
  const x = Math.max(0, w / 2 - inset);
  const z = Math.max(0, d / 2 - inset);
  const positions = [
    [x, z],
    [-x, z],
    [x, -z],
    [-x, -z],
  ] as const;
  for (const [px, pz] of positions) {
    const leg = _cyl(THREE, legR, legR, legH, mat, 12);
    leg.position.set(px, legH / 2, pz);
    group.add(leg);
  }
}

function _buildRug(THREE: any, p: ProxyBuildParams): any {
  const g = new THREE.Group();
  const base = _safeColor(THREE, p.colorHex, '#c9c0b2');
  const mat = new THREE.MeshStandardMaterial({ color: base, roughness: 0.95, metalness: 0.0 });
  const t = _clamp(Math.min(p.wM, p.dM) * 0.012, 0.004, 0.02);
  const rug = _box(THREE, p.wM, t, p.dM, mat);
  rug.position.y = t / 2 + 0.001; // avoid z-fighting with floor
  rug.receiveShadow = true;
  rug.castShadow = false;
  g.add(rug);

  // Border trim
  const trimT = t * 0.6;
  const trimInset = Math.min(p.wM, p.dM) * 0.04;
  const trim = _box(THREE, Math.max(0.01, p.wM - trimInset), trimT, Math.max(0.01, p.dM - trimInset), new THREE.MeshStandardMaterial({
    color: _shade(THREE, base, 0.92),
    roughness: 0.95,
    metalness: 0.0,
  }));
  trim.position.y = rug.position.y + t / 2 - trimT / 2;
  trim.receiveShadow = true;
  trim.castShadow = false;
  g.add(trim);
  return g;
}

function _buildCabinet(
  THREE: any,
  p: ProxyBuildParams,
  opts?: { doors?: boolean; shelves?: boolean }
): any {
  const doors = opts?.doors ?? true;
  const shelves = opts?.shelves ?? false;
  const g = new THREE.Group();
  const base = _safeColor(THREE, p.colorHex, '#b9b1a6');
  const matBody = _matFor(THREE, base, p.material);
  const matDark = _matFor(THREE, _shade(THREE, base, 0.62), 'metal');

  const toeH = _clamp(p.hM * 0.08, 0.03, 0.12);
  const bodyH = Math.max(0.05, p.hM - toeH);

  const body = _box(THREE, p.wM, bodyH, p.dM, matBody);
  body.position.y = toeH + bodyH / 2;
  g.add(body);

  const toe = _box(THREE, p.wM * 0.96, toeH, p.dM * 0.92, matDark);
  toe.position.y = toeH / 2;
  g.add(toe);

  // Subtle top lip
  const lipH = _clamp(p.hM * 0.02, 0.006, 0.03);
  const lip = _box(THREE, p.wM * 0.99, lipH, p.dM * 0.98, _matFor(THREE, _shade(THREE, base, 0.92), p.material));
  lip.position.y = toeH + bodyH - lipH / 2;
  g.add(lip);

  if (shelves) {
    const n = Math.max(2, Math.min(5, Math.round(p.hM / 0.45)));
    const shelfT = _clamp(p.hM * 0.02, 0.01, 0.03);
    for (let i = 1; i < n; i += 1) {
      const y = toeH + (bodyH * i) / n;
      const shelf = _box(
        THREE,
        p.wM * 0.94,
        shelfT,
        p.dM * 0.92,
        _matFor(THREE, _shade(THREE, base, 0.86), p.material)
      );
      shelf.position.set(0, y, 0);
      g.add(shelf);
    }
  }

  if (doors) {
    const panelT = Math.max(0.004, p.dM * 0.04);
    const panelH = bodyH * 0.88;
    const panelY = toeH + bodyH * 0.46;
    const z = p.dM / 2 - panelT / 2 + 0.0005;
    const twoDoors = p.wM > 0.9;
    const panelMat = _matFor(THREE, _shade(THREE, base, 0.98), p.material);
    if (twoDoors) {
      const w = p.wM * 0.48;
      const gap = p.wM * 0.02;
      const left = _box(THREE, w, panelH, panelT, panelMat);
      left.position.set(-gap, panelY, z);
      const right = _box(THREE, w, panelH, panelT, panelMat);
      right.position.set(gap, panelY, z);
      g.add(left);
      g.add(right);

      const handleMat = _matFor(THREE, _shade(THREE, base, 0.4), 'metal');
      const handleR = _clamp(panelH * 0.02, 0.004, 0.012);
      const handleH = panelH * 0.22;
      const hx = w / 2 - handleR * 2.2;
      for (const sign of [-1, 1]) {
        const handle = _cyl(THREE, handleR, handleR, handleH, handleMat, 10);
        handle.rotation.z = Math.PI / 2;
        handle.position.set(sign * hx, panelY, z + panelT / 2 + handleR);
        g.add(handle);
      }
    } else {
      const single = _box(THREE, p.wM * 0.92, panelH, panelT, panelMat);
      single.position.set(0, panelY, z);
      g.add(single);
      const handleMat = _matFor(THREE, _shade(THREE, base, 0.42), 'metal');
      const handle = _cyl(THREE, _clamp(panelH * 0.02, 0.004, 0.012), _clamp(panelH * 0.02, 0.004, 0.012), panelH * 0.22, handleMat, 10);
      handle.rotation.z = Math.PI / 2;
      handle.position.set(p.wM * 0.36, panelY, z + panelT / 2 + 0.008);
      g.add(handle);
    }
  }

  return g;
}

function _buildTable(
  THREE: any,
  p: ProxyBuildParams,
  opts?: { withDrawer?: boolean }
): any {
  const withDrawer = opts?.withDrawer ?? false;
  const g = new THREE.Group();
  const base = _safeColor(THREE, p.colorHex, '#c1b7aa');
  const topH = _clamp(p.hM * 0.08, 0.03, 0.07);
  const legH = Math.max(0.05, p.hM - topH);
  const matTop = _matFor(THREE, base, p.material);
  const matLeg = _matFor(THREE, _shade(THREE, base, 0.72), _hasAny(_asKey(p.material ?? ''), ['metal']) ? 'metal' : 'wood');

  const top = _box(THREE, p.wM, topH, p.dM, matTop);
  top.position.y = legH + topH / 2;
  g.add(top);

  const legR = _clamp(Math.min(p.wM, p.dM) * 0.05, 0.01, 0.05);
  const inset = _clamp(Math.min(p.wM, p.dM) * 0.12, legR * 2.2, 0.18);
  _addLegs(THREE, g, { w: p.wM, d: p.dM, legH, legR, inset, mat: matLeg });

  if (withDrawer) {
    const drawerH = _clamp(legH * 0.45, 0.12, 0.35);
    const drawer = _box(THREE, p.wM * 0.32, drawerH, p.dM * 0.6, _matFor(THREE, _shade(THREE, base, 0.92), p.material));
    drawer.position.set(-p.wM * 0.25, legH - drawerH / 2, 0);
    g.add(drawer);

    const handle = _cyl(THREE, legR * 0.55, legR * 0.55, p.wM * 0.08, _matFor(THREE, _shade(THREE, base, 0.45), 'metal'), 10);
    handle.rotation.z = Math.PI / 2;
    handle.position.set(drawer.position.x + drawer.geometry.parameters.width / 2 - legR * 2.2, drawer.position.y, p.dM * 0.3 + 0.01);
    g.add(handle);
  }

  return g;
}

function _buildChair(
  THREE: any,
  p: ProxyBuildParams,
  opts?: { office?: boolean; arms?: boolean }
): any {
  const office = opts?.office ?? false;
  const arms = opts?.arms ?? false;
  const g = new THREE.Group();
  const base = _safeColor(THREE, p.colorHex, '#b8b1a7');
  const matSeat = _matFor(THREE, base, p.material ?? 'fabric');
  const matFrame = _matFor(THREE, _shade(THREE, base, 0.6), 'metal');

  if (office) {
    const seatH = _clamp(p.hM * 0.22, 0.08, 0.16);
    const backH = Math.max(0.18, p.hM - seatH - 0.12);
    const columnH = Math.max(0.1, p.hM * 0.35);

    const column = _cyl(THREE, Math.min(p.wM, p.dM) * 0.05, Math.min(p.wM, p.dM) * 0.06, columnH, matFrame, 16);
    column.position.y = columnH / 2;
    g.add(column);

    // 5-star base
    const starR = Math.min(p.wM, p.dM) * 0.45;
    for (let i = 0; i < 5; i += 1) {
      const a = (i / 5) * Math.PI * 2;
      const leg = _box(THREE, starR * 0.52, 0.02, 0.06, matFrame);
      leg.position.set(Math.cos(a) * starR * 0.26, 0.02 / 2, Math.sin(a) * starR * 0.26);
      leg.rotation.y = -a;
      g.add(leg);

      const wheel = _cyl(THREE, 0.018, 0.018, 0.012, _matFor(THREE, _shade(THREE, base, 0.25), 'plastic'), 12);
      wheel.rotation.x = Math.PI / 2;
      wheel.position.set(Math.cos(a) * starR * 0.5, 0.012 / 2, Math.sin(a) * starR * 0.5);
      g.add(wheel);
    }

    const seat = _box(THREE, p.wM * 0.92, seatH, p.dM * 0.88, matSeat);
    seat.position.y = columnH + seatH / 2;
    g.add(seat);

    const back = _box(THREE, p.wM * 0.9, backH, Math.max(0.03, p.dM * 0.18), _matFor(THREE, _shade(THREE, base, 0.92), p.material ?? 'fabric'));
    back.position.set(0, columnH + seatH + backH / 2, -p.dM * 0.32);
    g.add(back);

    if (arms) {
      const armH = _clamp(seatH * 0.7, 0.05, 0.12);
      const arm = _box(THREE, p.wM * 0.08, armH, p.dM * 0.55, matFrame);
      const y = columnH + seatH + armH / 2;
      const x = p.wM * 0.42;
      arm.position.set(x, y, 0);
      g.add(arm);
      const arm2 = arm.clone();
      arm2.position.set(-x, y, 0);
      g.add(arm2);
    }

    return g;
  }

  const legH = _clamp(p.hM * 0.45, 0.25, p.hM * 0.6);
  const seatH = _clamp(p.hM * 0.18, 0.06, 0.14);
  const backH = Math.max(0.12, p.hM - legH - seatH);

  const seat = _box(THREE, p.wM * 0.9, seatH, p.dM * 0.75, matSeat);
  seat.position.set(0, legH + seatH / 2, 0);
  g.add(seat);

  const back = _box(THREE, p.wM * 0.86, backH, Math.max(0.02, p.dM * 0.12), _matFor(THREE, _shade(THREE, base, 0.86), p.material ?? 'fabric'));
  back.position.set(0, legH + seatH + backH / 2, -p.dM * 0.32);
  g.add(back);

  const legR = _clamp(Math.min(p.wM, p.dM) * 0.045, 0.01, 0.03);
  const inset = _clamp(Math.min(p.wM, p.dM) * 0.18, legR * 2.4, 0.22);
  _addLegs(THREE, g, { w: p.wM * 0.9, d: p.dM * 0.75, legH, legR, inset, mat: matFrame });

  if (arms) {
    const armH = _clamp(backH * 0.32, 0.07, 0.18);
    const armD = p.dM * 0.6;
    const armW = Math.max(0.03, p.wM * 0.12);
    const y = legH + seatH + armH / 2;
    const x = p.wM * 0.42;
    const arm = _box(THREE, armW, armH, armD, _matFor(THREE, _shade(THREE, base, 0.78), p.material ?? 'fabric'));
    arm.position.set(x, y, 0);
    g.add(arm);
    const arm2 = arm.clone();
    arm2.position.set(-x, y, 0);
    g.add(arm2);
  }

  return g;
}

function _buildSofa(
  THREE: any,
  p: ProxyBuildParams,
  opts?: { sectional?: boolean; recliner?: boolean; beanBag?: boolean }
): any {
  const sectional = opts?.sectional ?? false;
  const recliner = opts?.recliner ?? false;
  const beanBag = opts?.beanBag ?? false;
  const g = new THREE.Group();
  const base = _safeColor(THREE, p.colorHex, '#b7b0a5');
  const mat = _matFor(THREE, base, p.material ?? 'fabric');
  const matDark = _matFor(THREE, _shade(THREE, base, 0.72), p.material ?? 'fabric');
  const matLeg = _matFor(THREE, _shade(THREE, base, 0.45), 'metal');

  if (beanBag) {
    const r = Math.min(p.wM, p.dM) * 0.42;
    const blob = _sphere(THREE, r, mat, 22);
    blob.scale.set(p.wM / (2 * r), _clamp(p.hM / (2 * r), 0.45, 0.9), p.dM / (2 * r));
    blob.position.y = p.hM * 0.42;
    g.add(blob);
    return g;
  }

  const seatH = _clamp(p.hM * 0.35, 0.18, 0.42);
  const seatD = p.dM * 0.62;
  const backH = Math.max(0.2, p.hM - seatH);
  const backD = p.dM * 0.18;

  if (sectional) {
    const mainW = p.wM * 0.65;
    const chaiseW = p.wM * 0.45;
    const chaiseD = p.dM * 0.82;
    const seat1 = _box(THREE, mainW, seatH, seatD, mat);
    seat1.position.set(-p.wM * 0.12, seatH / 2, 0);
    g.add(seat1);
    const seat2 = _box(THREE, chaiseW, seatH, chaiseD, mat);
    seat2.position.set(p.wM * 0.22, seatH / 2, -p.dM * 0.12);
    g.add(seat2);
  } else {
    const seat = _box(THREE, p.wM, seatH, seatD, mat);
    seat.position.set(0, seatH / 2, 0);
    g.add(seat);
  }

  const back = _box(THREE, p.wM, backH, backD, matDark);
  back.position.set(0, seatH + backH / 2, -p.dM / 2 + backD / 2);
  g.add(back);

  const armW = Math.max(0.06, p.wM * 0.12);
  const armH = Math.max(seatH * 0.95, p.hM * 0.55);
  const armD = p.dM * 0.62;
  const arm = _box(THREE, armW, armH, armD, matDark);
  arm.position.set(p.wM / 2 - armW / 2, armH / 2, 0);
  g.add(arm);
  const arm2 = arm.clone();
  arm2.position.set(-p.wM / 2 + armW / 2, armH / 2, 0);
  g.add(arm2);

  // Cushions
  const cushH = Math.max(0.04, seatH * 0.25);
  const cush = _box(THREE, p.wM * 0.3, cushH, seatD * 0.48, _matFor(THREE, _shade(THREE, base, 1.06), p.material ?? 'fabric'));
  cush.position.set(-p.wM * 0.18, seatH + cushH / 2, 0.02);
  g.add(cush);
  const cush2 = cush.clone();
  cush2.position.x = p.wM * 0.18;
  g.add(cush2);

  if (recliner) {
    // Footrest hint
    const foot = _box(THREE, p.wM * 0.9, seatH * 0.35, seatD * 0.35, matDark);
    foot.position.set(0, foot.geometry.parameters.height / 2, p.dM * 0.22);
    g.add(foot);
  }

  // legs
  const legH = _clamp(p.hM * 0.06, 0.02, 0.06);
  const legR = _clamp(Math.min(p.wM, p.dM) * 0.03, 0.008, 0.02);
  _addLegs(THREE, g, { w: p.wM * 0.9, d: p.dM * 0.6, legH, legR, inset: legR * 2.2, mat: matLeg });

  return g;
}

function _buildBed(THREE: any, p: ProxyBuildParams): any {
  const g = new THREE.Group();
  const base = _safeColor(THREE, p.colorHex, '#c6bdb0');
  const matBase = _matFor(THREE, base, p.material ?? 'wood');
  const matMatt = _matFor(THREE, _shade(THREE, base, 1.05), 'fabric');

  const platformH = _clamp(p.hM * 0.28, 0.12, 0.25);
  const mattressH = _clamp(p.hM * 0.22, 0.14, 0.28);
  const headH = Math.max(0.2, p.hM - platformH);
  const headDepth = Math.max(0.03, p.dM * 0.08);

  const platform = _box(THREE, p.wM, platformH, p.dM, matBase);
  platform.position.y = platformH / 2;
  g.add(platform);

  const mattress = _box(THREE, p.wM * 0.97, mattressH, p.dM * 0.97, matMatt);
  mattress.position.y = platformH + mattressH / 2;
  g.add(mattress);

  const head = _box(THREE, p.wM, headH, headDepth, _matFor(THREE, _shade(THREE, base, 0.92), p.material ?? 'wood'));
  head.position.set(0, platformH + mattressH + headH / 2 - platformH * 0.15, -p.dM / 2 + headDepth / 2);
  g.add(head);

  // Feet hint
  const legH = _clamp(platformH * 0.18, 0.02, 0.05);
  const legR = _clamp(Math.min(p.wM, p.dM) * 0.02, 0.008, 0.02);
  _addLegs(THREE, g, { w: p.wM * 0.9, d: p.dM * 0.9, legH, legR, inset: legR * 3.0, mat: _matFor(THREE, _shade(THREE, base, 0.55), 'metal') });

  return g;
}

function _buildLamp(
  THREE: any,
  p: ProxyBuildParams,
  opts?: { wall?: boolean; ceiling?: boolean }
): any {
  const wall = opts?.wall ?? false;
  const ceiling = opts?.ceiling ?? false;
  const g = new THREE.Group();
  const base = _safeColor(THREE, p.colorHex, '#d8d3c8');
  const matMetal = _matFor(THREE, _shade(THREE, base, 0.55), 'metal');
  const matShade = _matFor(THREE, _shade(THREE, base, 1.02), p.material ?? 'fabric');

  const r = Math.min(p.wM, p.dM) * 0.22;
  const poleR = Math.max(0.008, r * 0.22);

  if (ceiling) {
    const diskH = Math.max(0.02, p.hM * 0.18);
    const disk = _cyl(THREE, r * 1.2, r * 1.2, diskH, matShade, 24);
    disk.position.y = diskH / 2;
    g.add(disk);
    const glow = _cyl(THREE, r * 0.95, r * 0.95, diskH * 0.4, new THREE.MeshStandardMaterial({
      color: new THREE.Color(0xfff3d6),
      roughness: 0.25,
      metalness: 0.0,
      emissive: new THREE.Color(0xfff3d6).multiplyScalar(0.35),
    }), 24);
    glow.position.y = diskH * 0.15;
    g.add(glow);
    return g;
  }

  if (wall) {
    const plate = _box(THREE, r * 1.1, r * 1.1, Math.max(0.015, r * 0.25), matMetal);
    plate.position.y = plate.geometry.parameters.height / 2;
    g.add(plate);
    const shade = _cone(THREE, r * 0.9, Math.max(0.15, p.hM * 0.5), matShade, 20);
    shade.rotation.x = Math.PI / 2;
    shade.position.set(0, r * 0.9, plate.geometry.parameters.depth / 2 + r * 0.65);
    g.add(shade);
    return g;
  }

  // Floor/table/desk lamp
  const baseH = Math.max(0.02, p.hM * 0.04);
  const baseDisk = _cyl(THREE, r * 1.2, r * 1.2, baseH, matMetal, 22);
  baseDisk.position.y = baseH / 2;
  g.add(baseDisk);

  const poleH = Math.max(0.12, p.hM * 0.55);
  const pole = _cyl(THREE, poleR, poleR, poleH, matMetal, 16);
  pole.position.y = baseH + poleH / 2;
  g.add(pole);

  const shadeH = Math.max(0.18, p.hM - (baseH + poleH) * 0.9);
  const shade = _cone(THREE, r * 1.05, shadeH, matShade, 22);
  shade.position.y = baseH + poleH + shadeH / 2;
  g.add(shade);

  const bulb = _sphere(THREE, Math.max(0.02, r * 0.2), new THREE.MeshStandardMaterial({
    color: new THREE.Color(0xfff3d6),
    roughness: 0.25,
    metalness: 0.0,
    emissive: new THREE.Color(0xfff3d6).multiplyScalar(0.35),
  }), 18);
  bulb.position.y = baseH + poleH + shadeH * 0.25;
  g.add(bulb);

  return g;
}

function _buildPendantLight(THREE: any, p: ProxyBuildParams): any {
  const g = new THREE.Group();
  const base = _safeColor(THREE, p.colorHex, '#d8d3c8');
  const matMetal = _matFor(THREE, _shade(THREE, base, 0.55), 'metal');
  const matShade = _matFor(THREE, _shade(THREE, base, 1.02), p.material ?? 'fabric');

  const totalH = Math.max(0.25, p.hM);
  const shadeH = _clamp(totalH * 0.45, 0.18, 0.55);
  const cordH = Math.max(0.05, totalH - shadeH);
  const r = Math.min(p.wM, p.dM) * 0.28;

  const shade = _cone(THREE, r * 1.05, shadeH, matShade, 24);
  shade.position.y = shadeH / 2;
  g.add(shade);

  const bulb = _sphere(
    THREE,
    Math.max(0.02, r * 0.18),
    new THREE.MeshStandardMaterial({
      color: new THREE.Color(0xfff3d6),
      roughness: 0.25,
      metalness: 0.0,
      emissive: new THREE.Color(0xfff3d6).multiplyScalar(0.35),
    }),
    18
  );
  bulb.position.y = shadeH * 0.25;
  g.add(bulb);

  const cord = _cyl(THREE, 0.006, 0.006, cordH, matMetal, 10);
  cord.position.y = shadeH + cordH / 2;
  g.add(cord);

  const canopy = _cyl(THREE, r * 0.35, r * 0.35, 0.03, matMetal, 18);
  canopy.position.y = totalH - 0.015;
  g.add(canopy);

  return g;
}

function _buildTrackLight(THREE: any, p: ProxyBuildParams): any {
  const g = new THREE.Group();
  const base = _safeColor(THREE, p.colorHex, '#d8d3c8');
  const mat = _matFor(THREE, _shade(THREE, base, 0.6), p.material ?? 'metal');
  const barH = _clamp(p.hM * 0.22, 0.02, 0.06);
  const bar = _box(THREE, p.wM, barH, Math.max(0.04, p.dM * 0.18), mat);
  bar.position.y = p.hM - barH / 2;
  g.add(bar);

  const n = Math.max(3, Math.min(6, Math.round(p.wM / 0.35)));
  for (let i = 0; i < n; i += 1) {
    const t = (i + 0.5) / n;
    const x = (t - 0.5) * p.wM * 0.9;
    const head = _cone(THREE, 0.05, 0.12, _matFor(THREE, _shade(THREE, base, 0.85), p.material ?? 'metal'), 18);
    head.rotation.x = Math.PI;
    head.position.set(x, p.hM - barH - 0.06, 0);
    g.add(head);
  }
  return g;
}

function _buildTowelRack(THREE: any, p: ProxyBuildParams): any {
  const g = new THREE.Group();
  const base = _safeColor(THREE, p.colorHex, '#c0c0c0');
  const mat = _matFor(THREE, base, p.material ?? 'metal');
  const rodR = _clamp(Math.min(p.wM, p.hM) * 0.08, 0.006, 0.018);
  const rodLen = Math.max(0.15, p.wM * 0.9);
  const rod = _cyl(THREE, rodR, rodR, rodLen, mat, 14);
  rod.rotation.z = Math.PI / 2;
  rod.position.set(0, Math.max(rodR * 2.5, p.hM * 0.45), 0);
  g.add(rod);
  const plateW = rodR * 3.0;
  const plate = _box(THREE, plateW, plateW, Math.max(0.01, p.dM), mat);
  plate.position.set(-rodLen / 2, rod.position.y, 0);
  g.add(plate);
  const plate2 = plate.clone();
  plate2.position.x = rodLen / 2;
  g.add(plate2);
  return g;
}

function _buildOttoman(THREE: any, p: ProxyBuildParams): any {
  const g = new THREE.Group();
  const base = _safeColor(THREE, p.colorHex, '#b7b0a5');
  const mat = _matFor(THREE, base, p.material ?? 'fabric');
  const seatH = _clamp(p.hM, 0.18, 0.55);
  const seat = _box(THREE, p.wM, seatH, p.dM, mat);
  seat.position.y = seatH / 2;
  g.add(seat);
  const top = _box(THREE, p.wM * 0.92, Math.max(0.03, seatH * 0.18), p.dM * 0.92, _matFor(THREE, _shade(THREE, base, 1.06), p.material ?? 'fabric'));
  top.position.y = seatH - top.geometry.parameters.height / 2;
  g.add(top);
  const legH = _clamp(seatH * 0.12, 0.02, 0.06);
  const legR = _clamp(Math.min(p.wM, p.dM) * 0.03, 0.008, 0.02);
  _addLegs(THREE, g, { w: p.wM * 0.92, d: p.dM * 0.92, legH, legR, inset: legR * 2.8, mat: _matFor(THREE, _shade(THREE, base, 0.45), 'metal') });
  return g;
}

function _buildStool(THREE: any, p: ProxyBuildParams): any {
  const g = new THREE.Group();
  const base = _safeColor(THREE, p.colorHex, '#b7b0a5');
  const matSeat = _matFor(THREE, base, p.material ?? 'wood');
  const matLeg = _matFor(THREE, _shade(THREE, base, 0.6), 'metal');
  const seatH = _clamp(p.hM * 0.18, 0.04, 0.12);
  const legH = Math.max(0.15, p.hM - seatH);
  const seat = _box(THREE, p.wM * 0.92, seatH, p.dM * 0.92, matSeat);
  seat.position.y = legH + seatH / 2;
  g.add(seat);
  const legR = _clamp(Math.min(p.wM, p.dM) * 0.05, 0.01, 0.03);
  const inset = _clamp(Math.min(p.wM, p.dM) * 0.18, legR * 2.4, 0.22);
  _addLegs(THREE, g, { w: p.wM * 0.9, d: p.dM * 0.9, legH, legR, inset, mat: matLeg });
  return g;
}

function _buildCoatRack(THREE: any, p: ProxyBuildParams): any {
  const g = new THREE.Group();
  const base = _safeColor(THREE, p.colorHex, '#b7b0a5');
  const mat = _matFor(THREE, _shade(THREE, base, 0.6), p.material ?? 'wood');
  const baseR = Math.min(p.wM, p.dM) * 0.35;
  const baseDisk = _cyl(THREE, baseR, baseR * 1.05, 0.03, mat, 24);
  baseDisk.position.y = 0.015;
  g.add(baseDisk);
  const poleH = Math.max(1.2, p.hM);
  const pole = _cyl(THREE, baseR * 0.12, baseR * 0.12, poleH, mat, 16);
  pole.position.y = poleH / 2 + 0.03;
  g.add(pole);
  for (let i = 0; i < 6; i += 1) {
    const a = (i / 6) * Math.PI * 2;
    const hook = _cyl(THREE, 0.008, 0.008, 0.12, _matFor(THREE, _shade(THREE, base, 0.45), 'metal'), 10);
    hook.rotation.z = Math.PI / 2;
    hook.position.set(Math.cos(a) * baseR * 0.18, poleH * 0.82, Math.sin(a) * baseR * 0.18);
    g.add(hook);
  }
  return g;
}

function _buildUmbrellaStand(THREE: any, p: ProxyBuildParams): any {
  const g = new THREE.Group();
  const base = _safeColor(THREE, p.colorHex, '#b7b0a5');
  const mat = _matFor(THREE, base, p.material ?? 'metal');
  const r = Math.min(p.wM, p.dM) * 0.35;
  const h = _clamp(p.hM, 0.25, 0.8);
  const wall = _openCyl(THREE, r, r * 1.02, h, mat, 22);
  wall.position.y = h / 2;
  g.add(wall);
  const bottom = _cyl(THREE, r * 0.95, r * 0.95, 0.015, _matFor(THREE, _shade(THREE, base, 0.9), p.material ?? 'metal'), 22);
  bottom.position.y = 0.0075;
  g.add(bottom);
  return g;
}

function _buildBarCart(THREE: any, p: ProxyBuildParams): any {
  const g = new THREE.Group();
  const base = _safeColor(THREE, p.colorHex, '#c1b7aa');
  const matFrame = _matFor(THREE, _shade(THREE, base, 0.55), 'metal');
  const matShelf = _matFor(THREE, _shade(THREE, base, 0.96), p.material ?? 'wood');

  const topH = _clamp(p.hM * 0.08, 0.02, 0.06);
  const legH = Math.max(0.3, p.hM - topH);
  const top = _box(THREE, p.wM, topH, p.dM, matShelf);
  top.position.y = legH + topH / 2;
  g.add(top);
  const shelf = _box(THREE, p.wM * 0.95, topH, p.dM * 0.95, _matFor(THREE, _shade(THREE, base, 0.88), p.material ?? 'wood'));
  shelf.position.y = legH * 0.5;
  g.add(shelf);

  const legR = _clamp(Math.min(p.wM, p.dM) * 0.035, 0.008, 0.02);
  const inset = _clamp(Math.min(p.wM, p.dM) * 0.12, legR * 2.4, 0.16);
  _addLegs(THREE, g, { w: p.wM, d: p.dM, legH, legR, inset, mat: matFrame });

  // wheels
  const wheel = _cyl(THREE, 0.02, 0.02, 0.012, _matFor(THREE, _shade(THREE, base, 0.35), 'plastic'), 14);
  wheel.rotation.x = Math.PI / 2;
  wheel.position.set(p.wM / 2 - inset, 0.012 / 2, p.dM / 2 - inset);
  g.add(wheel);
  const w2 = wheel.clone();
  w2.position.set(-p.wM / 2 + inset, 0.012 / 2, p.dM / 2 - inset);
  g.add(w2);
  const w3 = wheel.clone();
  w3.position.set(p.wM / 2 - inset, 0.012 / 2, -p.dM / 2 + inset);
  g.add(w3);
  const w4 = wheel.clone();
  w4.position.set(-p.wM / 2 + inset, 0.012 / 2, -p.dM / 2 + inset);
  g.add(w4);

  // push handle
  const handle = _cyl(THREE, legR * 0.6, legR * 0.6, p.dM * 0.6, matFrame, 12);
  handle.rotation.x = Math.PI / 2;
  handle.position.set(p.wM / 2 - legR * 2.0, legH + topH * 0.75, 0);
  g.add(handle);
  return g;
}

function _buildShowerNiche(THREE: any, p: ProxyBuildParams): any {
  const g = new THREE.Group();
  const base = _safeColor(THREE, p.colorHex, '#e9e6df');
  const frame = _box(THREE, p.wM, p.hM, p.dM, new THREE.MeshStandardMaterial({ color: base, roughness: 0.6, metalness: 0.02 }));
  frame.position.y = p.hM / 2;
  g.add(frame);
  const cavity = _box(THREE, p.wM * 0.88, p.hM * 0.78, Math.max(0.02, p.dM * 0.55), new THREE.MeshStandardMaterial({ color: _shade(THREE, base, 0.92), roughness: 0.75, metalness: 0.0 }));
  cavity.position.set(0, p.hM * 0.52, -p.dM * 0.1);
  g.add(cavity);
  return g;
}

function _buildIroningBoard(THREE: any, p: ProxyBuildParams): any {
  const g = new THREE.Group();
  const base = _safeColor(THREE, p.colorHex, '#cbbfaf');
  const matTop = _matFor(THREE, base, 'fabric');
  const matLeg = _matFor(THREE, _shade(THREE, base, 0.55), 'metal');
  const topT = 0.02;
  const top = _box(THREE, p.wM, topT, p.dM, matTop);
  const h = _clamp(p.hM, 0.6, 1.0);
  top.position.y = h;
  g.add(top);
  // simple X legs
  const leg = _box(THREE, 0.03, h, 0.03, matLeg);
  leg.position.set(-p.wM * 0.15, h / 2, 0);
  leg.rotation.z = Math.PI / 12;
  g.add(leg);
  const leg2 = leg.clone();
  leg2.position.x = p.wM * 0.15;
  leg2.rotation.z = -Math.PI / 12;
  g.add(leg2);
  const foot = _box(THREE, p.wM * 0.25, 0.02, 0.05, matLeg);
  foot.position.set(0, 0.01, 0);
  g.add(foot);
  return g;
}

function _buildIron(THREE: any, p: ProxyBuildParams): any {
  const g = new THREE.Group();
  const base = _safeColor(THREE, p.colorHex, '#bdb7af');
  const mat = _matFor(THREE, base, p.material ?? 'plastic');
  const h = _clamp(p.hM, 0.04, 0.12);
  const body = _box(THREE, p.wM, h, p.dM, mat);
  body.position.y = h / 2;
  g.add(body);
  const handle = _cyl(THREE, 0.01, 0.01, p.dM * 0.6, _matFor(THREE, _shade(THREE, base, 0.55), 'plastic'), 12);
  handle.rotation.x = Math.PI / 2;
  handle.position.set(0, h * 1.1, 0);
  g.add(handle);
  return g;
}

function _buildDryingRack(THREE: any, p: ProxyBuildParams): any {
  const g = new THREE.Group();
  const base = _safeColor(THREE, p.colorHex, '#b7b0a5');
  const mat = _matFor(THREE, _shade(THREE, base, 0.6), p.material ?? 'metal');
  const h = _clamp(p.hM, 0.7, 1.6);
  const rail = _cyl(THREE, 0.01, 0.01, p.wM, mat, 12);
  rail.rotation.z = Math.PI / 2;
  rail.position.set(0, h, -p.dM * 0.25);
  g.add(rail);
  const rail2 = rail.clone();
  rail2.position.z = p.dM * 0.25;
  g.add(rail2);
  const leg = _cyl(THREE, 0.01, 0.01, h, mat, 12);
  leg.position.set(-p.wM * 0.35, h / 2, -p.dM * 0.25);
  g.add(leg);
  const l2 = leg.clone();
  l2.position.set(p.wM * 0.35, h / 2, -p.dM * 0.25);
  g.add(l2);
  const l3 = leg.clone();
  l3.position.set(-p.wM * 0.35, h / 2, p.dM * 0.25);
  g.add(l3);
  const l4 = leg.clone();
  l4.position.set(p.wM * 0.35, h / 2, p.dM * 0.25);
  g.add(l4);
  return g;
}

function _buildWallPanel(THREE: any, p: ProxyBuildParams): any {
  const g = new THREE.Group();
  const base = _safeColor(THREE, p.colorHex, '#c9c3bb');
  const frameT = Math.max(0.02, Math.min(p.wM, p.hM) * 0.06);
  const depth = Math.max(0.02, p.dM);
  const matFrame = _matFor(THREE, _shade(THREE, base, 0.78), 'wood');
  const matArt = _matFor(THREE, _shade(THREE, base, 1.05), p.material ?? 'canvas');

  const back = _box(THREE, p.wM, p.hM, depth, matArt);
  back.position.y = p.hM / 2;
  g.add(back);

  const top = _box(THREE, p.wM, frameT, depth * 1.05, matFrame);
  top.position.set(0, p.hM - frameT / 2, 0);
  g.add(top);
  const bot = top.clone();
  bot.position.set(0, frameT / 2, 0);
  g.add(bot);
  const side = _box(THREE, frameT, p.hM, depth * 1.05, matFrame);
  side.position.set(-p.wM / 2 + frameT / 2, p.hM / 2, 0);
  g.add(side);
  const side2 = side.clone();
  side2.position.x = p.wM / 2 - frameT / 2;
  g.add(side2);

  return g;
}

function _buildMirrorPanel(THREE: any, p: ProxyBuildParams): any {
  const g = new THREE.Group();
  const frameColor = _safeColor(THREE, p.colorHex, '#dce4ea');
  const glassColor = new THREE.Color('#edf4f8');
  const frameT = _clamp(Math.min(p.wM, p.hM) * 0.045, 0.01, 0.035);
  const depth = _clamp(Math.max(0.008, p.dM * 0.5), 0.008, 0.02);
  const matFrame = _matFor(THREE, _shade(THREE, frameColor, 0.9), p.material ?? 'metal');
  const matGlass = new THREE.MeshStandardMaterial({
    color: glassColor,
    roughness: 0.08,
    metalness: 0.02,
    transparent: true,
    opacity: 0.72,
  });

  const back = _box(THREE, p.wM, p.hM, depth, _matFor(THREE, _shade(THREE, frameColor, 0.78), 'wood'));
  back.position.y = p.hM / 2;
  g.add(back);

  const glass = _box(
    THREE,
    Math.max(0.04, p.wM - frameT * 2.4),
    Math.max(0.04, p.hM - frameT * 2.4),
    Math.max(0.004, depth * 0.45),
    matGlass,
  );
  glass.position.set(0, p.hM / 2, depth * 0.12);
  g.add(glass);

  const top = _box(THREE, p.wM, frameT, depth * 1.1, matFrame);
  top.position.set(0, p.hM - frameT / 2, 0);
  g.add(top);
  const bot = top.clone();
  bot.position.set(0, frameT / 2, 0);
  g.add(bot);
  const side = _box(THREE, frameT, p.hM, depth * 1.1, matFrame);
  side.position.set(-p.wM / 2 + frameT / 2, p.hM / 2, 0);
  g.add(side);
  const side2 = side.clone();
  side2.position.x = p.wM / 2 - frameT / 2;
  g.add(side2);

  return g;
}

function _buildDevice(
  THREE: any,
  p: ProxyBuildParams,
  opts?: { screen?: boolean; laptop?: boolean }
): any {
  const screen = opts?.screen ?? false;
  const laptop = opts?.laptop ?? false;
  const g = new THREE.Group();
  const base = _safeColor(THREE, p.colorHex, '#bdb7af');
  const matBody = _matFor(THREE, base, p.material ?? 'plastic');
  const matDark = _matFor(THREE, _shade(THREE, base, 0.35), 'plastic');
  const matGlass = new THREE.MeshStandardMaterial({
    color: new THREE.Color(0x111111),
    roughness: 0.15,
    metalness: 0.25,
    transparent: true,
    opacity: 0.65,
  });

  if (screen) {
    const frameT = Math.max(0.02, p.dM * 0.12);
    const panel = _box(THREE, p.wM, p.hM * 0.9, frameT, matBody);
    panel.position.y = p.hM * 0.55;
    g.add(panel);

    const glass = _box(THREE, p.wM * 0.92, p.hM * 0.82, frameT * 0.6, matGlass);
    glass.position.set(0, p.hM * 0.55, frameT * 0.3);
    g.add(glass);

    const stand = _box(THREE, p.wM * 0.18, p.hM * 0.15, p.dM * 0.35, matDark);
    stand.position.set(0, stand.geometry.parameters.height / 2, 0);
    g.add(stand);
    return g;
  }

  if (laptop) {
    const baseH = Math.max(0.02, p.hM * 0.25);
    const body = _box(THREE, p.wM, baseH, p.dM, matBody);
    body.position.y = baseH / 2;
    g.add(body);
    const screenH = Math.max(0.04, p.hM - baseH * 0.8);
    const screenPane = _box(THREE, p.wM, screenH, Math.max(0.01, p.dM * 0.06), matBody);
    screenPane.position.set(0, baseH + screenH / 2, -p.dM / 2 + screenPane.geometry.parameters.depth / 2);
    screenPane.rotation.x = -Math.PI / 7;
    g.add(screenPane);
    const glass = _box(THREE, p.wM * 0.92, screenH * 0.86, screenPane.geometry.parameters.depth * 0.6, matGlass);
    glass.position.set(0, screenPane.position.y, screenPane.position.z + 0.003);
    glass.rotation.x = screenPane.rotation.x;
    g.add(glass);
    return g;
  }

  // Generic device / appliance
  const body = _box(THREE, p.wM, p.hM, p.dM, matBody);
  body.position.y = p.hM / 2;
  g.add(body);

  // Front panel hint
  const panelT = Math.max(0.008, p.dM * 0.06);
  const panel = _box(THREE, p.wM * 0.92, p.hM * 0.85, panelT, _matFor(THREE, _shade(THREE, base, 0.92), p.material ?? 'plastic'));
  panel.position.set(0, p.hM * 0.5, p.dM / 2 - panelT / 2 + 0.0005);
  g.add(panel);

  // Handle or knob
  const knob = _cyl(THREE, Math.max(0.008, p.wM * 0.03), Math.max(0.008, p.wM * 0.03), Math.max(0.02, p.wM * 0.05), _matFor(THREE, _shade(THREE, base, 0.45), 'metal'), 14);
  knob.rotation.z = Math.PI / 2;
  knob.position.set(p.wM * 0.35, p.hM * 0.5, p.dM / 2 + 0.012);
  g.add(knob);

  return g;
}

function _buildToilet(THREE: any, p: ProxyBuildParams): any {
  const g = new THREE.Group();
  const base = _safeColor(THREE, p.colorHex, '#e9e6df');
  const mat = new THREE.MeshStandardMaterial({ color: base, roughness: 0.55, metalness: 0.02 });

  const bowlH = _clamp(p.hM * 0.55, 0.28, 0.5);
  const bowl = _box(THREE, p.wM * 0.9, bowlH, p.dM * 0.95, mat);
  bowl.position.y = bowlH / 2;
  g.add(bowl);

  const tankH = Math.max(0.2, p.hM - bowlH * 0.85);
  const tank = _box(THREE, p.wM * 0.6, tankH, p.dM * 0.35, mat);
  tank.position.set(0, bowlH + tankH / 2 - 0.04, -p.dM * 0.22);
  g.add(tank);

  const lid = _box(THREE, p.wM * 0.78, Math.max(0.02, bowlH * 0.12), p.dM * 0.75, new THREE.MeshStandardMaterial({
    color: _shade(THREE, base, 0.98),
    roughness: 0.65,
    metalness: 0.0,
  }));
  lid.position.set(0, bowlH + lid.geometry.parameters.height / 2 - 0.01, 0.03);
  g.add(lid);
  return g;
}

function _buildBathtubOrShower(
  THREE: any,
  p: ProxyBuildParams,
  opts?: { shower?: boolean }
): any {
  const shower = opts?.shower ?? false;
  const g = new THREE.Group();
  const base = _safeColor(THREE, p.colorHex, '#e7e2d9');
  const mat = new THREE.MeshStandardMaterial({ color: base, roughness: 0.6, metalness: 0.02 });
  const wallMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(0xffffff),
    roughness: 0.35,
    metalness: 0.0,
    transparent: true,
    opacity: 0.35,
  });

  const tubH = _clamp(p.hM * 0.75, shower ? 0.08 : 0.35, shower ? 0.18 : 0.6);
  const tub = _box(THREE, p.wM, tubH, p.dM, mat);
  tub.position.y = tubH / 2;
  g.add(tub);

  // inner cavity hint
  const lip = _box(THREE, p.wM * 0.9, Math.max(0.02, tubH * 0.2), p.dM * 0.9, new THREE.MeshStandardMaterial({
    color: _shade(THREE, base, 0.92),
    roughness: 0.65,
    metalness: 0.0,
  }));
  lip.position.y = tubH * 0.62;
  g.add(lip);

  if (shower) {
    const glass = _box(THREE, p.wM * 0.02, p.hM, p.dM * 0.9, wallMat);
    glass.position.set(p.wM / 2 - glass.geometry.parameters.width / 2, p.hM / 2, 0);
    g.add(glass);
  }
  return g;
}

function _buildPlant(THREE: any, p: ProxyBuildParams): any {
  const g = new THREE.Group();
  const base = _safeColor(THREE, p.colorHex, '#7a6b5c');
  const potMat = _matFor(THREE, base, p.material ?? 'ceramic');
  const leafMat = _matFor(THREE, new THREE.Color('#4e7a5a'), 'fabric');

  const potH = _clamp(p.hM * 0.35, 0.12, 0.3);
  const potR = Math.min(p.wM, p.dM) * 0.28;
  const pot = _cyl(THREE, potR * 0.9, potR, potH, potMat, 22);
  pot.position.y = potH / 2;
  g.add(pot);

  const foliageH = Math.max(0.18, p.hM - potH * 0.8);
  const blob = _sphere(THREE, Math.min(p.wM, p.dM) * 0.35, leafMat, 20);
  blob.scale.set(1.0, _clamp(foliageH / (Math.min(p.wM, p.dM) * 0.7), 0.8, 1.6), 1.0);
  blob.position.y = potH + foliageH * 0.55;
  g.add(blob);
  return g;
}

function _buildVase(THREE: any, p: ProxyBuildParams): any {
  const g = new THREE.Group();
  const base = _safeColor(THREE, p.colorHex, '#b9b1a6');
  const mat = _matFor(THREE, base, p.material ?? 'ceramic');
  const r = Math.min(p.wM, p.dM) * 0.32;
  const bodyH = Math.max(0.12, p.hM * 0.85);
  const body = _cyl(THREE, r * 0.75, r, bodyH, mat, 24);
  body.position.y = bodyH / 2;
  g.add(body);
  const neckH = Math.max(0.04, p.hM - bodyH);
  const neck = _cyl(THREE, r * 0.5, r * 0.6, neckH, mat, 24);
  neck.position.y = bodyH + neckH / 2;
  g.add(neck);
  return g;
}

function _buildCushion(THREE: any, p: ProxyBuildParams): any {
  const g = new THREE.Group();
  const base = _safeColor(THREE, p.colorHex, '#d5cfc4');
  const mat = _matFor(THREE, base, p.material ?? 'fabric');
  const t = _clamp(p.hM, 0.04, 0.12); // cushion thickness
  const r = Math.min(p.wM, p.dM) * 0.45;
  const blob = _sphere(THREE, Math.max(0.04, r), mat, 20);
  blob.scale.set(p.wM / (2 * r), t / (2 * r), p.dM / (2 * r));
  blob.position.y = t / 2;
  g.add(blob);
  // seam hint
  const seam = _box(
    THREE,
    p.wM * 0.86,
    Math.max(0.003, t * 0.22),
    Math.max(0.003, p.dM * 0.05),
    _matFor(THREE, _shade(THREE, base, 0.9), p.material ?? 'fabric')
  );
  seam.position.set(0, t * 0.6, 0);
  seam.castShadow = false;
  seam.receiveShadow = true;
  g.add(seam);
  return g;
}

function _buildThrowBlanket(THREE: any, p: ProxyBuildParams): any {
  const g = new THREE.Group();
  const base = _safeColor(THREE, p.colorHex, '#cbbfaf');
  const mat = _matFor(THREE, base, p.material ?? 'fabric');
  const t = _clamp(p.hM * 0.2, 0.004, 0.02);
  const w = p.wM;
  const d = p.dM;
  const sheet = _box(THREE, w, t, d, mat);
  sheet.position.y = t / 2;
  sheet.castShadow = false;
  sheet.receiveShadow = true;
  g.add(sheet);
  // folded edge
  const fold = _box(THREE, w * 0.35, t * 1.2, d * 0.55, _matFor(THREE, _shade(THREE, base, 0.92), p.material ?? 'fabric'));
  fold.position.set(-w * 0.18, t * 0.9, d * 0.1);
  fold.castShadow = false;
  fold.receiveShadow = true;
  g.add(fold);
  return g;
}

function _buildLaundryBasket(THREE: any, p: ProxyBuildParams): any {
  const g = new THREE.Group();
  const base = _safeColor(THREE, p.colorHex, '#b7b0a5');
  const mat = _matFor(THREE, base, p.material ?? 'plastic');
  const r = Math.min(p.wM, p.dM) * 0.35;
  const h = _clamp(p.hM, 0.25, 0.75);
  const wall = _openCyl(THREE, r * 0.95, r, h, mat, 24);
  wall.position.y = h / 2;
  g.add(wall);
  const bottom = _cyl(THREE, r * 0.92, r * 0.92, Math.max(0.01, h * 0.04), _matFor(THREE, _shade(THREE, base, 0.92), p.material ?? 'plastic'), 24);
  bottom.position.y = bottom.geometry.parameters.height / 2 + 0.002;
  g.add(bottom);
  // handle hint
  const handle = _cyl(THREE, 0.01, 0.01, r * 1.1, _matFor(THREE, _shade(THREE, base, 0.55), 'metal'), 12);
  handle.rotation.z = Math.PI / 2;
  handle.position.set(0, h * 0.78, r + 0.01);
  g.add(handle);
  return g;
}

function _buildTV(THREE: any, p: ProxyBuildParams): any {
  const g = new THREE.Group();
  const frameT = _clamp(p.dM * 0.12, 0.02, 0.08);
  const body = _box(THREE, p.wM, p.hM * 0.92, frameT, _matFor(THREE, new THREE.Color('#2a2a2a'), 'plastic'));
  body.position.y = p.hM * 0.55;
  g.add(body);
  const glass = _box(
    THREE,
    p.wM * 0.92,
    p.hM * 0.84,
    frameT * 0.65,
    new THREE.MeshStandardMaterial({
      color: new THREE.Color(0x111111),
      roughness: 0.15,
      metalness: 0.25,
      transparent: true,
      opacity: 0.7,
    })
  );
  glass.position.set(0, p.hM * 0.55, frameT * 0.25);
  g.add(glass);
  const stand = _box(THREE, p.wM * 0.22, p.hM * 0.1, Math.max(0.08, p.dM * 0.35), _matFor(THREE, new THREE.Color('#3a3a3a'), 'metal'));
  stand.position.set(0, stand.geometry.parameters.height / 2, 0);
  g.add(stand);
  return g;
}

function _buildFan(THREE: any, p: ProxyBuildParams): any {
  const g = new THREE.Group();
  const base = _safeColor(THREE, p.colorHex, '#b8b1a7');
  const matFrame = _matFor(THREE, _shade(THREE, base, 0.6), 'metal');
  const matBlade = _matFor(THREE, _shade(THREE, base, 0.95), 'plastic');
  const standH = _clamp(p.hM * 0.78, 0.6, 1.6);
  const poleR = _clamp(Math.min(p.wM, p.dM) * 0.05, 0.01, 0.03);
  const baseDisk = _cyl(THREE, poleR * 5.5, poleR * 6.0, 0.03, matFrame, 24);
  baseDisk.position.y = 0.015;
  g.add(baseDisk);
  const pole = _cyl(THREE, poleR, poleR, standH, matFrame, 16);
  pole.position.y = standH / 2 + 0.03;
  g.add(pole);
  const headR = Math.min(p.wM, p.dM) * 0.38;
  const hub = _cyl(THREE, headR * 0.18, headR * 0.18, 0.05, matFrame, 18);
  hub.rotation.x = Math.PI / 2;
  hub.position.y = standH + 0.1;
  g.add(hub);
  for (let i = 0; i < 3; i += 1) {
    const a = (i / 3) * Math.PI * 2;
    const blade = _box(THREE, headR * 0.85, 0.01, headR * 0.18, matBlade);
    blade.position.set(Math.cos(a) * headR * 0.32, standH + 0.1, Math.sin(a) * headR * 0.32);
    blade.rotation.y = -a;
    g.add(blade);
  }
  const cage = _openCyl(THREE, headR, headR, headR * 0.25, _matFor(THREE, _shade(THREE, base, 0.75), 'metal'), 24);
  cage.rotation.x = Math.PI / 2;
  cage.position.y = standH + 0.1;
  g.add(cage);
  return g;
}

function _buildSink(THREE: any, p: ProxyBuildParams): any {
  const g = new THREE.Group();
  const base = _safeColor(THREE, p.colorHex, '#c7bfb3');
  const cabinet = _buildCabinet(THREE, { ...p, hM: p.hM * 0.85 }, { doors: true, shelves: false });
  g.add(cabinet);
  const topH = _clamp(p.hM * 0.06, 0.03, 0.06);
  const top = _box(THREE, p.wM * 1.02, topH, p.dM * 1.02, _matFor(THREE, _shade(THREE, base, 0.92), 'stone'));
  top.position.y = p.hM * 0.85 + topH / 2;
  g.add(top);
  const basin = _box(THREE, p.wM * 0.55, topH * 1.3, p.dM * 0.55, new THREE.MeshStandardMaterial({
    color: new THREE.Color('#e9e6df'),
    roughness: 0.45,
    metalness: 0.02,
  }));
  basin.position.set(0, top.position.y + topH * 0.1, 0);
  g.add(basin);
  const faucet = _cyl(THREE, 0.01, 0.01, 0.18, _matFor(THREE, new THREE.Color('#6d6b66'), 'metal'), 14);
  faucet.position.set(p.wM * 0.18, top.position.y + 0.11, -p.dM * 0.12);
  g.add(faucet);
  const spout = _cyl(THREE, 0.008, 0.008, 0.12, _matFor(THREE, new THREE.Color('#6d6b66'), 'metal'), 14);
  spout.rotation.z = Math.PI / 2;
  spout.position.set(p.wM * 0.18, faucet.position.y + 0.06, -p.dM * 0.02);
  g.add(spout);
  return g;
}

function _buildCooktop(THREE: any, p: ProxyBuildParams): any {
  const g = new THREE.Group();
  const baseH = _clamp(p.hM * 0.25, 0.02, 0.08);
  const body = _box(THREE, p.wM, baseH, p.dM, _matFor(THREE, new THREE.Color('#2a2a2a'), 'metal'));
  body.position.y = baseH / 2;
  g.add(body);
  const burnerR = Math.min(p.wM, p.dM) * 0.12;
  for (const [sx, sz] of [
    [-0.22, -0.18],
    [0.22, -0.18],
    [-0.22, 0.18],
    [0.22, 0.18],
  ] as const) {
    const burner = _cyl(THREE, burnerR, burnerR, 0.01, _matFor(THREE, new THREE.Color('#111111'), 'metal'), 20);
    burner.position.set(p.wM * sx, baseH + 0.005, p.dM * sz);
    g.add(burner);
  }
  return g;
}

function _buildRangeHood(THREE: any, p: ProxyBuildParams): any {
  const g = new THREE.Group();
  const base = _safeColor(THREE, p.colorHex, '#d8d3c8');
  const mat = _matFor(THREE, base, p.material ?? 'metal');
  const hood = _box(THREE, p.wM, p.hM * 0.75, p.dM, mat);
  hood.position.y = hood.geometry.parameters.height / 2;
  g.add(hood);
  const chimney = _box(THREE, p.wM * 0.35, p.hM * 0.45, p.dM * 0.45, _matFor(THREE, _shade(THREE, base, 0.92), p.material ?? 'metal'));
  chimney.position.set(0, p.hM * 0.75 + chimney.geometry.parameters.height / 2 - 0.02, 0);
  g.add(chimney);
  // slats
  const slatT = Math.max(0.003, p.dM * 0.03);
  for (let i = 0; i < 4; i += 1) {
    const slat = _box(THREE, p.wM * 0.9, slatT, p.dM * 0.02, _matFor(THREE, _shade(THREE, base, 0.7), 'metal'));
    slat.position.set(0, p.hM * 0.18 + i * slatT * 2.0, p.dM / 2 - slat.geometry.parameters.depth / 2 + 0.0005);
    g.add(slat);
  }
  return g;
}

function _buildProjector(THREE: any, p: ProxyBuildParams): any {
  const g = new THREE.Group();
  const base = _safeColor(THREE, p.colorHex, '#bdb7af');
  const mat = _matFor(THREE, base, p.material ?? 'plastic');
  const bodyH = _clamp(p.hM, 0.08, 0.25);
  const body = _box(THREE, p.wM, bodyH, p.dM, mat);
  body.position.y = bodyH / 2;
  g.add(body);
  const lens = _cyl(THREE, Math.min(p.wM, p.dM) * 0.12, Math.min(p.wM, p.dM) * 0.12, Math.max(0.03, p.dM * 0.15), _matFor(THREE, new THREE.Color('#3a3a3a'), 'metal'), 18);
  lens.rotation.x = Math.PI / 2;
  lens.position.set(p.wM * 0.28, bodyH * 0.6, p.dM / 2 + lens.geometry.parameters.height / 2);
  g.add(lens);
  return g;
}

function _buildGenericWithDetails(THREE: any, p: ProxyBuildParams): any {
  // Fallback that still feels more than "just a box": feet + panel + handle.
  const g = new THREE.Group();
  const base = _safeColor(THREE, p.colorHex, '#c0b7aa');
  const mat = _matFor(THREE, base, p.material);
  const body = _box(THREE, p.wM, p.hM, p.dM, mat);
  body.position.y = p.hM / 2;
  g.add(body);

  const footH = _clamp(p.hM * 0.05, 0.01, 0.05);
  const footR = _clamp(Math.min(p.wM, p.dM) * 0.03, 0.006, 0.02);
  _addLegs(THREE, g, { w: p.wM * 0.9, d: p.dM * 0.9, legH: footH, legR: footR, inset: footR * 2.4, mat: _matFor(THREE, _shade(THREE, base, 0.5), 'metal') });

  const panelT = Math.max(0.008, p.dM * 0.05);
  const panel = _box(THREE, p.wM * 0.92, p.hM * 0.5, panelT, _matFor(THREE, _shade(THREE, base, 0.95), p.material));
  panel.position.set(0, p.hM * 0.6, p.dM / 2 - panelT / 2 + 0.0005);
  g.add(panel);

  const handle = _cyl(THREE, footR * 0.6, footR * 0.6, p.wM * 0.08, _matFor(THREE, _shade(THREE, base, 0.45), 'metal'), 10);
  handle.rotation.z = Math.PI / 2;
  handle.position.set(p.wM * 0.34, p.hM * 0.6, p.dM / 2 + 0.012);
  g.add(handle);

  return g;
}

export function buildObjectProxy(THREE: any, params: ProxyBuildParams): any {
  const t = _asKey(params.type);
  const p = {
    ...params,
    wM: Math.max(0.02, params.wM),
    dM: Math.max(0.02, params.dM),
    hM: Math.max(0.02, params.hM),
  };

  if (t.includes('rug') || t.includes('mat')) return _buildRug(THREE, p);

  if (t === 'bed') return _buildBed(THREE, p);
  if (t.includes('mattress')) return _buildGenericWithDetails(THREE, { ...p, material: 'fabric' });
  if (t.includes('headboard')) return _buildGenericWithDetails(THREE, { ...p, dM: Math.max(0.05, p.dM * 0.35), material: p.material ?? 'wood' });

  if (_hasAny(t, ['sofa', 'sectional'])) return _buildSofa(THREE, p, { sectional: t.includes('sectional'), recliner: false, beanBag: false });
  if (t.includes('recliner')) return _buildSofa(THREE, p, { sectional: false, recliner: true, beanBag: false });
  if (t.includes('bean_bag')) return _buildSofa(THREE, p, { sectional: false, recliner: false, beanBag: true });
  if (t.includes('armchair')) return _buildSofa(THREE, p, { sectional: false, recliner: false, beanBag: false });
  if (t.includes('ottoman')) return _buildOttoman(THREE, { ...p, hM: Math.min(p.hM, 0.55) });

  if (_hasAny(t, ['office_chair'])) return _buildChair(THREE, p, { office: true, arms: true });
  if (t.includes('bar_stool')) return _buildChair(THREE, { ...p, hM: Math.max(p.hM, 0.75) }, { office: false, arms: false });
  if (t.includes('bathroom_stool') || (t.includes('stool') && !t.includes('bar_stool'))) return _buildStool(THREE, p);
  if (_hasAny(t, ['chair', 'dining_chair'])) return _buildChair(THREE, p, { office: false, arms: false });
  if (_hasAny(t, ['bench', 'entry_bench'])) return _buildChair(THREE, { ...p, hM: Math.min(p.hM, 0.65) }, { office: false, arms: false });

  if (_hasAny(t, ['desk'])) return _buildTable(THREE, p, { withDrawer: true });
  if (_hasAny(t, ['dining_table', 'coffee_table', 'side_table', 'console_table', 'table'])) return _buildTable(THREE, p, { withDrawer: false });

  if (_hasAny(t, ['bookshelf', 'media_shelf', 'shelf'])) return _buildCabinet(THREE, p, { doors: false, shelves: true });
  if (_hasAny(t, ['wardrobe', 'cabinet', 'dresser', 'nightstand', 'tv_console', 'vanity', 'sideboard', 'china_cabinet', 'filing_cabinet', 'pedestal', 'shoe_rack'])) {
    return _buildCabinet(THREE, p, { doors: true, shelves: false });
  }
  if (_hasAny(t, ['kitchen_base_cabinet', 'kitchen_wall_cabinet', 'kitchen_tall_cabinet', 'pantry_cabinet', 'wine_cabinet'])) {
    return _buildCabinet(THREE, p, { doors: true, shelves: false });
  }
  if (t.includes('kitchen_island')) {
    // Island as cabinet + countertop.
    const base = _buildCabinet(THREE, { ...p, hM: p.hM * 0.88 }, { doors: true, shelves: false });
    const topH = _clamp(p.hM * 0.06, 0.03, 0.06);
    const top = _box(THREE, p.wM * 1.05, topH, p.dM * 1.05, _matFor(THREE, _safeColor(THREE, p.colorHex, '#d8d3c8'), 'stone'));
    top.position.y = p.hM * 0.88 + topH / 2;
    const g = new THREE.Group();
    g.add(base);
    g.add(top);
    return g;
  }

  if (_hasAny(t, ['floor_lamp', 'desk_lamp', 'table_lamp', 'bedside_lamp', 'lamp'])) return _buildLamp(THREE, p, { wall: false, ceiling: false });
  if (t.includes('wall_sconce')) return _buildLamp(THREE, p, { wall: true, ceiling: false });
  if (t.includes('pendant_light')) return _buildPendantLight(THREE, p);
  if (t.includes('track_light')) return _buildTrackLight(THREE, p);
  if (t.includes('ceiling_light')) return _buildLamp(THREE, p, { wall: false, ceiling: true });

  if (t.includes('mirror')) return _buildMirrorPanel(THREE, p);
  if (_hasAny(t, ['wall_art', 'clock', 'whiteboard', 'projector_screen'])) return _buildWallPanel(THREE, p);
  if (t.includes('towel_rack')) return _buildTowelRack(THREE, p);
  if (t.includes('shower_niche')) return _buildShowerNiche(THREE, p);

  if (t.includes('plant')) return _buildPlant(THREE, p);
  if (t.includes('vase')) return _buildVase(THREE, p);
  if (t.includes('cushion')) return _buildCushion(THREE, p);
  if (t.includes('throw_blanket')) return _buildThrowBlanket(THREE, p);
  if (t.includes('laundry_basket')) return _buildLaundryBasket(THREE, p);
  if (t === 'tv') return _buildTV(THREE, p);
  if (t === 'fan') return _buildFan(THREE, p);
  if (t.includes('range_hood')) return _buildRangeHood(THREE, p);
  if (t.includes('cooktop') || t === 'stove') return _buildCooktop(THREE, p);
  if (t.includes('sink')) return _buildSink(THREE, p);
  if (t === 'projector') return _buildProjector(THREE, p);
  if (t.includes('coat_rack')) return _buildCoatRack(THREE, p);
  if (t.includes('umbrella_stand')) return _buildUmbrellaStand(THREE, p);
  if (t.includes('bar_cart')) return _buildBarCart(THREE, p);
  if (t.includes('ironing_board')) return _buildIroningBoard(THREE, p);
  if (t === 'iron') return _buildIron(THREE, { ...p, hM: Math.min(p.hM, 0.12) });
  if (t.includes('drying_rack')) return _buildDryingRack(THREE, p);
  if (_hasAny(t, ['decor', 'smart_speaker', 'speaker', 'pet_bed'])) return _buildGenericWithDetails(THREE, p);

  if (_hasAny(t, ['monitor'])) return _buildDevice(THREE, p, { screen: true, laptop: false });
  if (_hasAny(t, ['laptop'])) return _buildDevice(THREE, p, { screen: false, laptop: true });
  if (_hasAny(t, ['desktop_pc', 'printer', 'microwave', 'dishwasher', 'oven', 'fridge', 'washing_machine', 'dryer', 'air_purifier', 'dehumidifier', 'heater', 'air_conditioner'])) {
    return _buildDevice(THREE, p, { screen: false, laptop: false });
  }
  if (_hasAny(t, ['rice_cooker', 'electric_kettle', 'coffee_machine', 'toaster', 'air_fryer', 'blender', 'mouse', 'keyboard'])) {
    // Smaller footprint devices: keep height modest.
    return _buildDevice(THREE, { ...p, hM: Math.min(p.hM, 0.35) }, { screen: false, laptop: false });
  }

  if (_hasAny(t, ['toilet', 'bidet'])) return _buildToilet(THREE, p);
  if (t === 'bathtub') return _buildBathtubOrShower(THREE, p, { shower: false });
  if (t === 'shower') return _buildBathtubOrShower(THREE, p, { shower: true });

  // Doors / windows / curtains / blinds: show as thin panels (still better than nothing).
  if (_hasAny(t, ['door', 'window', 'curtain', 'blind'])) return _buildWallPanel(THREE, { ...p, dM: Math.max(0.02, p.dM * 0.12) });

  // Default
  return _buildGenericWithDetails(THREE, p);
}
