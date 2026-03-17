import React from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App';
import './styles.css';

if (import.meta.env.DEV) {
  const original = {
    log: console.log.bind(console),
    warn: console.warn.bind(console),
    error: console.error.bind(console),
    debug: console.debug.bind(console),
  };

  const formatArg = (arg: unknown): string => {
    if (arg instanceof Error) return `${arg.name}: ${arg.message}`;
    if (typeof arg === 'string') return arg;
    try {
      return JSON.stringify(arg);
    } catch {
      return String(arg);
    }
  };

  const extractErrorStack = (args: unknown[]): string | null => {
    for (const arg of args) {
      if (arg instanceof Error) return arg.stack ?? null;
    }
    return null;
  };

  const postLog = (level: string, args: unknown[]) => {
    const message = args.map(formatArg).join(' ');
    const payload = {
      level,
      message,
      time: new Date().toISOString(),
      stack: extractErrorStack(args),
    };
    fetch('/__dev_log', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).catch(() => {});
  };

  const wrap = (level: string, fn: (...args: unknown[]) => void) => {
    return (...args: unknown[]) => {
      fn(...args);
      postLog(level, args);
    };
  };

  console.log = wrap('log', original.log);
  console.warn = wrap('warn', original.warn);
  console.error = wrap('error', original.error);
  console.debug = wrap('debug', original.debug);

  window.addEventListener('error', (evt) => {
    postLog('error', [evt.message, `${evt.filename}:${evt.lineno}:${evt.colno}`]);
  });
  window.addEventListener('unhandledrejection', (evt) => {
    postLog('error', ['unhandledrejection', (evt as PromiseRejectionEvent).reason]);
  });
}

const root = createRoot(document.getElementById('root')!);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
