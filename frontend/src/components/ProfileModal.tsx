import React, { useEffect, useMemo } from 'react';
import {
  formatVnd,
  getMonthlyImagesRemaining,
  type BillingMode,
} from '../billing';
import { useEditorStore } from '../store';
import type { GeneratedRenderRecord } from '../types';

type ProfileModalProps = {
  onClose: () => void;
};

type RenderMode = 'generate' | 'edit' | 'unknown';

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function getSavedRenderMode(render: GeneratedRenderRecord): RenderMode {
  if (render.meta.render_mode === 'generate' || render.meta.render_mode === 'edit') {
    return render.meta.render_mode;
  }
  if (isRecord(render.meta.request)) {
    const requestMode = render.meta.request.render_mode;
    if (requestMode === 'generate' || requestMode === 'edit') {
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
  const monthlyRemaining = getMonthlyImagesRemaining(billingSettings);
  const usagePercent =
    billingSettings.monthlyImageQuota > 0
      ? Math.min(
          100,
          Math.round(
            (billingSettings.monthlyImagesUsed / billingSettings.monthlyImageQuota) * 100
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
            monthlyImagesUsed: 0,
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
                <span>Paid Single Images</span>
                <strong>{billingSettings.totalPaidUses}</strong>
              </div>
            </div>
          </section>

          <section className="profile-section">
            <div className="profile-section-header">
              <h2>Billing</h2>
              <span>
                {billingSettings.mode === 'monthly' ? 'Monthly pack' : 'Pay per image'}
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
                <small>{billingSettings.monthlyImageQuota} images / month</small>
              </button>
              <button
                className={`billing-plan-card${
                  billingSettings.mode === 'per_use' ? ' selected' : ''
                }`}
                type="button"
                onClick={() => selectBillingMode('per_use')}
              >
                <span>Pay Per Image</span>
                <strong>{formatVnd(billingSettings.perUsePriceVnd)}</strong>
                <small>QR before every generate/edit</small>
              </button>
            </div>

            <div className="billing-usage-card">
              <div className="billing-usage-header">
                <div>
                  <strong>Monthly Image Credits</strong>
                  <span>
                    {billingSettings.monthlyImagesUsed} used · {monthlyRemaining} left
                  </span>
                </div>
                <strong>
                  {billingSettings.monthlyImagesUsed}/{billingSettings.monthlyImageQuota}
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
                <span>Monthly image quota</span>
                <input
                  type="number"
                  min={1}
                  step={1}
                  value={billingSettings.monthlyImageQuota}
                  onChange={(event) =>
                    updateBillingSettings({
                      monthlyImageQuota: clampInteger(event.target.value, 1),
                      monthlyImagesUsed: Math.min(
                        billingSettings.monthlyImagesUsed,
                        clampInteger(event.target.value, 1)
                      ),
                    })
                  }
                />
              </label>
              <label>
                <span>Single image price</span>
                <input
                  type="number"
                  min={1000}
                  step={1000}
                  value={billingSettings.perUsePriceVnd}
                  onChange={(event) =>
                    updateBillingSettings({
                      perUsePriceVnd: clampInteger(event.target.value, 1000),
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
