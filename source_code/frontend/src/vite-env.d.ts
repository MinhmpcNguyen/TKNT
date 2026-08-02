/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly DEV: boolean;
  readonly PROD: boolean;
  readonly VITE_API_BASE?: string;
  readonly VITE_ENABLE_EVAL_HOOKS?: string;
  readonly VITE_PIPELINE_POLL_TIMEOUT_SECONDS?: string;
  readonly VITE_TKNT_EVAL_HOOKS?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
