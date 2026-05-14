import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

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
  plugins: [react(), devLogMiddleware()],
  envDir: '..',
  envPrefix: ['VITE_', 'TKNT_DEMO_INVENTORY_ENABLED'],
  server: {
    port: 5173,
  },
});
