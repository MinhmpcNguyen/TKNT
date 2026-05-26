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

const devLogMiddleware = () => ({
  name: 'dev-log-middleware',
  configureServer(server: any) {
    server.middlewares.use('/__dev_log', (req: any, res: any) => {
      if (req.method !== 'POST') {
        res.statusCode = 405;
        res.end();
        return;
      }
      let body = '';
      req.on('data', (chunk: Buffer) => {
        body += chunk.toString();
        if (body.length > 1_000_000) {
          res.statusCode = 413;
          res.end();
          req.destroy();
        }
      });
      req.on('end', () => {
        try {
          const payload = JSON.parse(body || '{}');
          const level = String(payload.level || 'log').toUpperCase();
          const time = payload.time ? ` ${payload.time}` : '';
          const message = payload.message ?? '';
          console.log(`[FE ${level}]${time} ${message}`);
          if (payload.stack) {
            console.log(payload.stack);
          }
        } catch {
          if (body) console.log(`[FE LOG] ${body}`);
        }
        res.statusCode = 204;
        res.end();
      });
    });
  },
});

export default defineConfig({
  plugins: [react(), devLogMiddleware(), copyThreeRuntimePlugin()],
  publicDir,
  envDir: '..',
  envPrefix: ['VITE_', 'TKNT_DEMO_INVENTORY_ENABLED'],
  server: {
    port: 5173,
  },
});
