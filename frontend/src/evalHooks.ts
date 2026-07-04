const TRUE_ENV_VALUES = new Set(['1', 'true', 'yes', 'on']);

export function isEvalHooksEnabled(): boolean {
  const rawValue =
    import.meta.env.VITE_ENABLE_EVAL_HOOKS ??
    import.meta.env.VITE_TKNT_EVAL_HOOKS ??
    '';
  return TRUE_ENV_VALUES.has(String(rawValue).trim().toLowerCase());
}
