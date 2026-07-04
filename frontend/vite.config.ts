import react from '@vitejs/plugin-react';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

const publicDir = process.env.TKNT_VITE_PUBLIC_DIR === 'false' ? false : 'public';
const projectDir = path.dirname(fileURLToPath(import.meta.url));

function copyDirectorySync(source: string, destination: string): void {
  if (!fs.existsSync(source)) {
    return;
  }
  fs.mkdirSync(destination, { recursive: true });
  for (const entry of fs.readdirSync(source, { withFileTypes: true })) {
    const sourcePath = path.join(source, entry.name);
    const destinationPath = path.join(destination, entry.name);
    if (entry.isDirectory()) {
      copyDirectorySync(sourcePath, destinationPath);
      continue;
    }
    if (entry.isFile()) {
      fs.copyFileSync(sourcePath, destinationPath);
    }
  }
}

const copyThreeRuntimePlugin = () => ({
  name: 'copy-three-runtime',
  closeBundle() {
    const threeOutputDir = path.resolve(projectDir, 'dist/assets/three');
    copyDirectorySync(path.resolve(projectDir, 'src/vendor/three'), threeOutputDir);
    copyDirectorySync(path.resolve(projectDir, 'public/assets/three/basis'), path.join(threeOutputDir, 'basis'));
  },
});

export default defineConfig({
  plugins: [react(), copyThreeRuntimePlugin()],
  publicDir,
  envDir: '..',
  server: {
    port: 5173,
  },
});
