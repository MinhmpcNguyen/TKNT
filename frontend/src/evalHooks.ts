const TRUE_ENV_VALUES = new Set(['1', 'true', 'yes', 'on']);

export function isEvalHooksEnabled(): boolean {
  const env = (import.meta as any).env ?? {};
  const rawValue =
    env.VITE_ENABLE_EVAL_HOOKS ??
    env.VITE_TKNT_EVAL_HOOKS ??
    env.TKNT_ENABLE_EVAL_HOOKS ??
    '';
  return TRUE_ENV_VALUES.has(String(rawValue).trim().toLowerCase());
}
