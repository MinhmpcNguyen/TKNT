import type { InventoryItem } from '../types';
import {
  applyInventoryModelPreviewOverride,
  buildGlbPreviewModel,
  disposeThreeObject,
  resolveInventoryModelDescriptor,
  type InventoryModelDescriptor,
} from './models';
import { getInventoryDimensionsMm } from './inventoryDiagnostics';

type ThreeThumbnailRuntime = {
  THREE: any;
  GLTFLoader: any;
  KTX2Loader: any | null;
};

const THUMBNAIL_WIDTH_PX = 192;
const THUMBNAIL_HEIGHT_PX = 150;
const FALLBACK_WIDTH_M = 0.8;
const FALLBACK_DEPTH_M = 0.6;
const FALLBACK_HEIGHT_M = 0.8;

let runtimePromise: Promise<ThreeThumbnailRuntime> | null = null;
const thumbnailCache = new Map<string, Promise<string | null>>();

export function hasInventoryThumbnailModel(
  item: InventoryItem,
  preferredStyle?: string | null
): boolean {
  return resolveInventoryModelDescriptor(item, preferredStyle) !== null;
}

export function getInventoryThumbnailDataUrl(
  item: InventoryItem,
  preferredStyle?: string | null
): Promise<string | null> {
  const descriptor = resolveInventoryModelDescriptor(item, preferredStyle);
  if (!descriptor) {
    return Promise.resolve(null);
  }

  const key = buildThumbnailCacheKey(item, preferredStyle, descriptor);
  const cached = thumbnailCache.get(key);
  if (cached) {
    return cached;
  }

  const promise = renderInventoryThumbnailDataUrl(item, descriptor).catch(() => {
    thumbnailCache.delete(key);
    return null;
  });
  thumbnailCache.set(key, promise);
  return promise;
}

function buildThumbnailCacheKey(
  item: InventoryItem,
  preferredStyle: string | null | undefined,
  descriptor: InventoryModelDescriptor
): string {
  const dimensions = getInventoryDimensionsMm(item);
  return [
    item.asset_id,
    preferredStyle ?? '',
    descriptor.url,
    descriptor.rotationDeg,
    descriptor.rotationXDeg,
    descriptor.rotationZDeg,
    dimensions?.width ?? item.footprint_mm.w,
    dimensions?.depth ?? item.footprint_mm.d,
    dimensions?.height ?? item.height_mm,
  ].join('|');
}

async function loadThreeThumbnailRuntime(): Promise<ThreeThumbnailRuntime> {
  if (runtimePromise) {
    return runtimePromise;
  }

  runtimePromise = (async () => {
    const THREE_URL = new URL('../vendor/three/three.module.js', import.meta.url).toString();
    const GLTF_LOADER_URL = new URL(
      '../vendor/three/loaders/GLTFLoader.js',
      import.meta.url
    ).toString();
    const KTX2_LOADER_URL = new URL(
      '../vendor/three/loaders/KTX2Loader.js',
      import.meta.url
    ).toString();

    const THREE: any = await import(/* @vite-ignore */ THREE_URL);
    const gltfLoaderMod: any = await import(/* @vite-ignore */ GLTF_LOADER_URL);
    let KTX2Loader: any | null = null;
    try {
      const ktx2LoaderMod: any = await import(/* @vite-ignore */ KTX2_LOADER_URL);
      KTX2Loader = ktx2LoaderMod.KTX2Loader ?? null;
    } catch {
      KTX2Loader = null;
    }

    return {
      THREE,
      GLTFLoader: gltfLoaderMod.GLTFLoader,
      KTX2Loader,
    };
  })();

  return runtimePromise;
}

async function renderInventoryThumbnailDataUrl(
  item: InventoryItem,
  descriptor: InventoryModelDescriptor
): Promise<string | null> {
  if (typeof document === 'undefined') {
    return null;
  }

  const { THREE, GLTFLoader, KTX2Loader } = await loadThreeThumbnailRuntime();
  if (!GLTFLoader) {
    return null;
  }

  const canvas = document.createElement('canvas');
  canvas.width = THUMBNAIL_WIDTH_PX;
  canvas.height = THUMBNAIL_HEIGHT_PX;

  const renderer = new THREE.WebGLRenderer({
    alpha: false,
    antialias: true,
    canvas,
    preserveDrawingBuffer: true,
  });
  renderer.setPixelRatio(1);
  renderer.setSize(THUMBNAIL_WIDTH_PX, THUMBNAIL_HEIGHT_PX, false);
  renderer.setClearColor(0xf7f3eb, 1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#f7f3eb');
  scene.add(new THREE.HemisphereLight(0xffffff, 0xe7dfd1, 1.45));
  const keyLight = new THREE.DirectionalLight(0xffffff, 1.8);
  keyLight.position.set(2.8, 4, 3);
  scene.add(keyLight);
  const fillLight = new THREE.DirectionalLight(0xffffff, 0.5);
  fillLight.position.set(-3, 2, -2);
  scene.add(fillLight);

  let model: any | null = null;
  try {
    const dimensions = getInventoryDimensionsMm(item);
    const wM = Math.max(
      0.06,
      (dimensions?.width ?? item.footprint_mm.w ?? FALLBACK_WIDTH_M / 0.001) * 0.001
    );
    const dM = Math.max(
      0.06,
      (dimensions?.depth ?? item.footprint_mm.d ?? FALLBACK_DEPTH_M / 0.001) * 0.001
    );
    const hM = Math.max(
      0.06,
      (dimensions?.height ?? item.height_mm ?? FALLBACK_HEIGHT_M / 0.001) * 0.001
    );
    const preview = applyInventoryModelPreviewOverride({
      descriptor,
      context: 'panel',
      fitStrategy: 'box',
      wM,
      dM,
      hM,
    });
    model = await buildGlbPreviewModel({
      THREE,
      GLTFLoader,
      KTX2Loader,
      renderer,
      descriptor: preview.descriptor,
      wM: preview.wM,
      dM: preview.dM,
      hM: preview.hM,
      fitStrategy: preview.fitStrategy,
      colorHex: null,
      preserveMaterialTextures: true,
      minimumHorizontalAspectRatio: 0.45,
    });
    scene.add(model);
    frameModelForThumbnail(THREE, model, renderer, scene);
    return canvas.toDataURL('image/png');
  } finally {
    if (model) {
      disposeThreeObject(model);
    }
    renderer.dispose();
  }
}

function frameModelForThumbnail(
  THREE: any,
  model: any,
  renderer: any,
  scene: any
): void {
  const box = new THREE.Box3().setFromObject(model);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const radius = Math.max(sphere.radius, 0.08);
  const aspect = THUMBNAIL_WIDTH_PX / THUMBNAIL_HEIGHT_PX;
  const camera = new THREE.PerspectiveCamera(32, aspect, 0.01, Math.max(30, radius * 20));
  const halfVerticalFov = (camera.fov * Math.PI) / 360;
  const halfHorizontalFov = Math.atan(Math.tan(halfVerticalFov) * aspect);
  const limitingHalfFov = Math.min(halfVerticalFov, halfHorizontalFov);
  const distance = (radius / Math.sin(limitingHalfFov)) * 1.28;
  const viewDirection = new THREE.Vector3(0.68, 0.42, 0.86).normalize();
  camera.position.copy(center).addScaledVector(viewDirection, distance);
  camera.near = Math.max(0.005, distance - radius * 3);
  camera.far = Math.max(distance + radius * 4, 30);
  camera.lookAt(center.x, center.y + size.y * 0.04, center.z);
  camera.updateProjectionMatrix();
  renderer.render(scene, camera);
}
