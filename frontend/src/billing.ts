export type BillingMode = 'monthly' | 'per_use';

export type BillingChargeKind = 'generate' | 'edit';

export type BillingSettings = {
  mode: BillingMode;
  monthlyPriceVnd: number;
  monthlyImageQuota: number;
  monthlyImagesUsed: number;
  monthlyPeriodStartedAt: string;
  perUsePriceVnd: number;
  totalMonthlyUses: number;
  totalPaidUses: number;
};

const BILLING_STORAGE_KEY = 'tknt.billingSettings.v1';
const DEFAULT_MONTHLY_PRICE_VND = 299000;
const DEFAULT_MONTHLY_IMAGE_QUOTA = 100;
const DEFAULT_PER_USE_PRICE_VND = 5000;

export function createDefaultBillingSettings(now = new Date()): BillingSettings {
  return {
    mode: 'per_use',
    monthlyPriceVnd: DEFAULT_MONTHLY_PRICE_VND,
    monthlyImageQuota: DEFAULT_MONTHLY_IMAGE_QUOTA,
    monthlyImagesUsed: 0,
    monthlyPeriodStartedAt: now.toISOString(),
    perUsePriceVnd: DEFAULT_PER_USE_PRICE_VND,
    totalMonthlyUses: 0,
    totalPaidUses: 0,
  };
}

export function ensureCurrentBillingPeriod(
  settings: BillingSettings,
  now = new Date()
): BillingSettings {
  const startedAt = new Date(settings.monthlyPeriodStartedAt);
  if (
    Number.isNaN(startedAt.getTime()) ||
    startedAt.getFullYear() !== now.getFullYear() ||
    startedAt.getMonth() !== now.getMonth()
  ) {
    return {
      ...settings,
      monthlyImagesUsed: 0,
      monthlyPeriodStartedAt: now.toISOString(),
    };
  }
  return settings;
}

export function getMonthlyImagesRemaining(settings: BillingSettings): number {
  return Math.max(0, settings.monthlyImageQuota - settings.monthlyImagesUsed);
}

export function formatVnd(amount: number): string {
  return new Intl.NumberFormat('vi-VN', {
    currency: 'VND',
    maximumFractionDigits: 0,
    style: 'currency',
  }).format(amount);
}

export function buildPaymentReference(kind: BillingChargeKind): string {
  return `TKNT-${kind.toUpperCase()}-${Date.now().toString(36).toUpperCase()}`;
}

export function loadBillingSettings(): BillingSettings {
  if (typeof window === 'undefined') {
    return createDefaultBillingSettings();
  }
  const rawValue = window.localStorage.getItem(BILLING_STORAGE_KEY);
  if (!rawValue) {
    return createDefaultBillingSettings();
  }
  try {
    return ensureCurrentBillingPeriod(
      sanitizeBillingSettings(JSON.parse(rawValue) as unknown)
    );
  } catch {
    return createDefaultBillingSettings();
  }
}

export function saveBillingSettings(settings: BillingSettings): void {
  if (typeof window === 'undefined') {
    return;
  }
  window.localStorage.setItem(BILLING_STORAGE_KEY, JSON.stringify(settings));
}

function sanitizeBillingSettings(value: unknown): BillingSettings {
  const defaults = createDefaultBillingSettings();
  if (!isRecord(value)) {
    return defaults;
  }
  const mode = value.mode === 'monthly' || value.mode === 'per_use'
    ? value.mode
    : defaults.mode;
  return {
    mode,
    monthlyPriceVnd: readPositiveInteger(
      value.monthlyPriceVnd,
      defaults.monthlyPriceVnd
    ),
    monthlyImageQuota: readPositiveInteger(
      value.monthlyImageQuota,
      defaults.monthlyImageQuota
    ),
    monthlyImagesUsed: readNonNegativeInteger(
      value.monthlyImagesUsed,
      defaults.monthlyImagesUsed
    ),
    monthlyPeriodStartedAt:
      typeof value.monthlyPeriodStartedAt === 'string'
        ? value.monthlyPeriodStartedAt
        : defaults.monthlyPeriodStartedAt,
    perUsePriceVnd: readPositiveInteger(
      value.perUsePriceVnd,
      defaults.perUsePriceVnd
    ),
    totalMonthlyUses: readNonNegativeInteger(
      value.totalMonthlyUses,
      defaults.totalMonthlyUses
    ),
    totalPaidUses: readNonNegativeInteger(
      value.totalPaidUses,
      defaults.totalPaidUses
    ),
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function readPositiveInteger(value: unknown, fallback: number): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return fallback;
  }
  return Math.round(parsed);
}

function readNonNegativeInteger(value: unknown, fallback: number): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) {
    return fallback;
  }
  return Math.round(parsed);
}
