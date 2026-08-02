export type BillingMode = 'monthly' | 'per_use';

export type BillingChargeKind = 'generate' | 'edit' | 'add_credit';

export type BillingSettings = {
  mode: BillingMode;
  monthlyPriceVnd: number;
  monthlyCreditQuota: number;
  monthlyCreditsUsed: number;
  monthlyPeriodStartedAt: string;
  paidCreditBalance: number;
  perCreditPriceVnd: number;
  totalMonthlyCredits: number;
  totalPaidCredits: number;
};

type SnapshotImageQuality = 'low' | 'medium' | 'high';
type SnapshotImageSize = '1024x1024' | '1024x1536' | '1536x1024';

export const DEFAULT_IMAGE_RENDER_PARTIAL_IMAGES = 2;

const BILLING_STORAGE_KEY = 'tknt.billingSettings.v1';
const DEFAULT_MONTHLY_PRICE_VND = 299000;
const DEFAULT_MONTHLY_CREDIT_QUOTA = 100;
const DEFAULT_PER_CREDIT_PRICE_VND = 5000;
const GPT_IMAGE_2_OUTPUT_COST_USD: Record<
  SnapshotImageSize,
  Record<SnapshotImageQuality, number>
> = {
  '1024x1024': {
    high: 0.211,
    low: 0.006,
    medium: 0.053,
  },
  '1024x1536': {
    high: 0.165,
    low: 0.005,
    medium: 0.041,
  },
  '1536x1024': {
    high: 0.165,
    low: 0.005,
    medium: 0.041,
  },
};
const PARTIAL_IMAGE_OUTPUT_TOKENS = 100;
const GPT_IMAGE_2_OUTPUT_TOKEN_PRICE_PER_MILLION_USD = 30;

export function estimateImageRenderCredits({
  quality,
  size,
  partialImages = DEFAULT_IMAGE_RENDER_PARTIAL_IMAGES,
}: {
  quality: SnapshotImageQuality;
  size: SnapshotImageSize;
  partialImages?: number;
}): number {
  const outputCost = GPT_IMAGE_2_OUTPUT_COST_USD[size][quality];
  const partialOverhead = estimatePartialImageOverheadUsd(partialImages);
  const baselineUsd =
    GPT_IMAGE_2_OUTPUT_COST_USD['1536x1024'].medium +
    estimatePartialImageOverheadUsd(DEFAULT_IMAGE_RENDER_PARTIAL_IMAGES);
  return Math.max(1, Math.ceil((outputCost + partialOverhead) / baselineUsd));
}

function estimatePartialImageOverheadUsd(partialImages: number): number {
  const normalizedPartialImages = Math.max(0, Math.min(3, Math.round(partialImages)));
  return (
    (normalizedPartialImages *
      PARTIAL_IMAGE_OUTPUT_TOKENS *
      GPT_IMAGE_2_OUTPUT_TOKEN_PRICE_PER_MILLION_USD) /
    1_000_000
  );
}

function createDefaultBillingSettings(now = new Date()): BillingSettings {
  return {
    mode: 'per_use',
    monthlyPriceVnd: DEFAULT_MONTHLY_PRICE_VND,
    monthlyCreditQuota: DEFAULT_MONTHLY_CREDIT_QUOTA,
    monthlyCreditsUsed: 0,
    monthlyPeriodStartedAt: now.toISOString(),
    paidCreditBalance: 0,
    perCreditPriceVnd: DEFAULT_PER_CREDIT_PRICE_VND,
    totalMonthlyCredits: 0,
    totalPaidCredits: 0,
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
      monthlyCreditsUsed: 0,
      monthlyPeriodStartedAt: now.toISOString(),
    };
  }
  return settings;
}

export function getMonthlyCreditsRemaining(settings: BillingSettings): number {
  return Math.max(0, settings.monthlyCreditQuota - settings.monthlyCreditsUsed);
}

export function getPaidCreditsRemaining(settings: BillingSettings): number {
  return Math.max(0, settings.paidCreditBalance);
}

export function formatVnd(amount: number): string {
  return new Intl.NumberFormat('vi-VN', {
    currency: 'VND',
    maximumFractionDigits: 0,
    style: 'currency',
  }).format(amount);
}

export function buildPaymentReference(kind: BillingChargeKind): string {
  const kindToken = kind.toUpperCase().replace('_', '-');
  return `TKNT-${kindToken}-${Date.now().toString(36).toUpperCase()}`;
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
    monthlyCreditQuota: readPositiveInteger(
      value.monthlyCreditQuota,
      defaults.monthlyCreditQuota
    ),
    monthlyCreditsUsed: readNonNegativeInteger(
      value.monthlyCreditsUsed,
      defaults.monthlyCreditsUsed
    ),
    monthlyPeriodStartedAt:
      typeof value.monthlyPeriodStartedAt === 'string'
        ? value.monthlyPeriodStartedAt
        : defaults.monthlyPeriodStartedAt,
    paidCreditBalance: readNonNegativeInteger(
      value.paidCreditBalance,
      defaults.paidCreditBalance
    ),
    perCreditPriceVnd: readPositiveInteger(
      value.perCreditPriceVnd,
      defaults.perCreditPriceVnd
    ),
    totalMonthlyCredits: readNonNegativeInteger(
      value.totalMonthlyCredits,
      defaults.totalMonthlyCredits
    ),
    totalPaidCredits: readNonNegativeInteger(
      value.totalPaidCredits,
      defaults.totalPaidCredits
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
