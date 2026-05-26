import React, { useEffect, useMemo } from 'react';
import {
  formatVnd,
  getMonthlyCreditsRemaining,
  type BillingMode,
} from '../billing';
import { useEditorStore } from '../store';
import type { GeneratedRenderRecord } from '../types';

type ProfileModalProps = {
  onClose: () => void;
};

type RenderMode = 'generate' | 'edit' | 'camera_rerender' | 'unknown';

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function getSavedRenderMode(render: GeneratedRenderRecord): RenderMode {
  if (
    render.meta.render_mode === 'generate' ||
    render.meta.render_mode === 'edit' ||
    render.meta.render_mode === 'camera_rerender'
  ) {
    return render.meta.render_mode;
  }
  if (isRecord(render.meta.request)) {
    const requestMode = render.meta.request.render_mode;
    if (
      requestMode === 'generate' ||
      requestMode === 'edit' ||
      requestMode === 'camera_rerender'
    ) {
      return requestMode;
    }
  }
  return 'unknown';
}

function getPeriodLabel(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return 'Current month';
  }
  return new Intl.DateTimeFormat('en', {
    month: 'long',
    year: 'numeric',
  }).format(date);
}

function clampInteger(value: string, minimum: number): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return minimum;
  }
  return Math.max(minimum, Math.round(parsed));
}

export const ProfileModal: React.FC<ProfileModalProps> = ({ onClose }) => {
  const {
    currentUser,
    savedLayouts,
    savedRenders,
    billingSettings,
    updateBillingSettings,
  } = useEditorStore();
  const monthlyRemaining = getMonthlyCreditsRemaining(billingSettings);
  const usagePercent =
    billingSettings.monthlyCreditQuota > 0
      ? Math.min(
          100,
          Math.round(
            (billingSettings.monthlyCreditsUsed /
              billingSettings.monthlyCreditQuota) *
              100
          )
        )
      : 0;
  const renderStats = useMemo(() => {
    const snapshotRenders = savedRenders.filter(
      (render) => render.source === 'snapshot_render'
    );
    const generated = snapshotRenders.filter(
      (render) => getSavedRenderMode(render) === 'generate'
    ).length;
    const edited = snapshotRenders.filter(
      (render) => getSavedRenderMode(render) === 'edit'
    ).length;
    return {
      edited,
      generated,
      snapshot: snapshotRenders.length,
      total: savedRenders.length,
    };
  }, [savedRenders]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  const selectBillingMode = (mode: BillingMode): void => {
    updateBillingSettings(
      mode === 'monthly' && billingSettings.mode !== 'monthly'
        ? {
            mode,
            monthlyCreditsUsed: 0,
            monthlyPeriodStartedAt: new Date().toISOString(),
          }
        : { mode }
    );
  };

  return (
    <div
      className="profile-modal-overlay"
      role="dialog"
      aria-modal="true"
      aria-labelledby="profile-modal-title"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) {
          onClose();
        }
      }}
    >
      <div className="profile-modal-card">
        <div className="profile-modal-header">
          <div>
            <div id="profile-modal-title" className="profile-modal-title">
              Profile
            </div>
            <div className="profile-modal-subtitle">
              {currentUser?.display_name || currentUser?.email}
            </div>
          </div>
          <button className="profile-close-button" onClick={onClose} type="button">
            Close
          </button>
        </div>

        <div className="profile-modal-body">
          <section className="profile-identity-card">
            <div className="profile-avatar">
              {(currentUser?.display_name || currentUser?.email || 'U')
                .trim()
                .slice(0, 1)
                .toUpperCase()}
            </div>
            <div className="profile-identity-copy">
              <strong>{currentUser?.display_name || currentUser?.email}</strong>
              <span>{currentUser?.email}</span>
              <span>Tenant: {currentUser?.tenant_id}</span>
            </div>
          </section>

          <section className="profile-section">
            <div className="profile-section-header">
              <h2>Workspace Stats</h2>
              <span>{getPeriodLabel(billingSettings.monthlyPeriodStartedAt)}</span>
            </div>
            <div className="profile-stat-grid">
              <div className="profile-stat-card">
                <span>Layouts</span>
                <strong>{savedLayouts.length}</strong>
              </div>
              <div className="profile-stat-card">
                <span>AI Images</span>
                <strong>{renderStats.snapshot}</strong>
              </div>
              <div className="profile-stat-card">
                <span>Generated</span>
                <strong>{renderStats.generated}</strong>
              </div>
              <div className="profile-stat-card">
                <span>Edited</span>
                <strong>{renderStats.edited}</strong>
              </div>
              <div className="profile-stat-card">
                <span>Saved Renders</span>
                <strong>{renderStats.total}</strong>
              </div>
              <div className="profile-stat-card">
                <span>Paid Credits</span>
                <strong>{billingSettings.totalPaidCredits}</strong>
              </div>
            </div>
          </section>

          <section className="profile-section">
            <div className="profile-section-header">
              <h2>Billing</h2>
              <span>
                {billingSettings.mode === 'monthly' ? 'Monthly pack' : 'Pay per credit'}
              </span>
            </div>
            <div className="billing-plan-grid">
              <button
                className={`billing-plan-card${
                  billingSettings.mode === 'monthly' ? ' selected' : ''
                }`}
                type="button"
                onClick={() => selectBillingMode('monthly')}
              >
                <span>Monthly Pack</span>
                <strong>{formatVnd(billingSettings.monthlyPriceVnd)}</strong>
                <small>{billingSettings.monthlyCreditQuota} credits / month</small>
              </button>
              <button
                className={`billing-plan-card${
                  billingSettings.mode === 'per_use' ? ' selected' : ''
                }`}
                type="button"
                onClick={() => selectBillingMode('per_use')}
              >
                <span>Pay Per Credit</span>
                <strong>{formatVnd(billingSettings.perCreditPriceVnd)}</strong>
                <small>QR by selected render credits</small>
              </button>
            </div>

            <div className="billing-usage-card">
              <div className="billing-usage-header">
                <div>
                  <strong>Monthly Image Credits</strong>
                  <span>
                    {billingSettings.monthlyCreditsUsed} used · {monthlyRemaining} left
                  </span>
                </div>
                <strong>
                  {billingSettings.monthlyCreditsUsed}/
                  {billingSettings.monthlyCreditQuota}
                </strong>
              </div>
              <div className="billing-meter" aria-hidden="true">
                <div style={{ width: `${usagePercent}%` }} />
              </div>
            </div>

            <div className="billing-dev-grid">
              <label>
                <span>Monthly price</span>
                <input
                  type="number"
                  min={1000}
                  step={1000}
                  value={billingSettings.monthlyPriceVnd}
                  onChange={(event) =>
                    updateBillingSettings({
                      monthlyPriceVnd: clampInteger(event.target.value, 1000),
                    })
                  }
                />
              </label>
              <label>
                <span>Monthly credit quota</span>
                <input
                  type="number"
                  min={1}
                  step={1}
                  value={billingSettings.monthlyCreditQuota}
                  onChange={(event) =>
                    updateBillingSettings({
                      monthlyCreditQuota: clampInteger(event.target.value, 1),
                      monthlyCreditsUsed: Math.min(
                        billingSettings.monthlyCreditsUsed,
                        clampInteger(event.target.value, 1)
                      ),
                    })
                  }
                />
              </label>
              <label>
                <span>Credit price</span>
                <input
                  type="number"
                  min={1000}
                  step={1000}
                  value={billingSettings.perCreditPriceVnd}
                  onChange={(event) =>
                    updateBillingSettings({
                      perCreditPriceVnd: clampInteger(event.target.value, 1000),
                    })
                  }
                />
              </label>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
};
