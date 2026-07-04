import React, { useMemo } from 'react';
import {
  formatVnd,
  type BillingChargeKind,
} from '../billing';

type BillingPaymentModalProps = {
  amountVnd: number;
  chargeKind: BillingChargeKind;
  countdownSeconds?: number | null;
  creditCount?: number;
  confirmLabel?: string;
  reason: string;
  reference: string;
  onCancel: () => void;
  onConfirm?: () => void;
};

const QR_GRID_SIZE = 13;

function buildQrCells(reference: string): boolean[] {
  let hash = 2166136261;
  for (let index = 0; index < reference.length; index += 1) {
    hash ^= reference.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return Array.from({ length: QR_GRID_SIZE * QR_GRID_SIZE }, (_, index) => {
    const row = Math.floor(index / QR_GRID_SIZE);
    const col = index % QR_GRID_SIZE;
    const finder = getFinderCell(row, col);
    if (finder !== null) {
      return finder;
    }
    const value = hash + row * 37 + col * 53 + row * col * 11;
    return (value & 3) === 0 || value % 7 === 0;
  });
}

function getFinderCell(row: number, col: number): boolean | null {
  const finderOrigins = [
    { row: 0, col: 0 },
    { row: 0, col: QR_GRID_SIZE - 5 },
    { row: QR_GRID_SIZE - 5, col: 0 },
  ];
  for (const origin of finderOrigins) {
    const localRow = row - origin.row;
    const localCol = col - origin.col;
    if (localRow < 0 || localCol < 0 || localRow >= 5 || localCol >= 5) {
      continue;
    }
    return (
      localRow === 0 ||
      localCol === 0 ||
      localRow === 4 ||
      localCol === 4 ||
      (localRow === 2 && localCol === 2)
    );
  }
  return null;
}

function getActionLabel(chargeKind: BillingChargeKind): string {
  if (chargeKind === 'add_credit') {
    return 'Add Credits';
  }
  return chargeKind === 'edit' ? 'Edit Image' : 'Generate Image';
}

export const BillingPaymentModal: React.FC<BillingPaymentModalProps> = ({
  amountVnd,
  chargeKind,
  countdownSeconds = null,
  creditCount,
  confirmLabel = 'Confirm payment',
  reason,
  reference,
  onCancel,
  onConfirm,
}) => {
  const cells = useMemo(() => buildQrCells(reference), [reference]);
  const actionLabel = getActionLabel(chargeKind);

  return (
    <div className="billing-payment-overlay" role="dialog" aria-modal="true">
      <div className="billing-payment-card">
        <div className="billing-payment-copy">
          <span className="billing-payment-kicker">{actionLabel}</span>
          <h2>Payment QR</h2>
          <p>{reason}</p>
        </div>
        <div className="billing-payment-grid">
          <div
            className="billing-qr"
            style={{
              gridTemplateColumns: `repeat(${QR_GRID_SIZE}, 1fr)`,
            }}
            aria-label={`Mock payment QR for ${reference}`}
          >
            {cells.map((active, index) => (
              <span
                key={`${reference}-${index}`}
                className={active ? 'active' : ''}
              />
            ))}
          </div>
          <div className="billing-payment-details">
            <div>
              <span>Amount</span>
              <strong>{formatVnd(amountVnd)}</strong>
            </div>
            {creditCount ? (
              <div>
                <span>Credits</span>
                <strong>{creditCount}</strong>
              </div>
            ) : null}
            <div>
              <span>Reference</span>
              <strong>{reference}</strong>
            </div>
            {countdownSeconds !== null ? (
              <div>
                <span>Unlocks in</span>
                <strong>{countdownSeconds}s</strong>
              </div>
            ) : null}
          </div>
        </div>
        <div className="billing-payment-actions">
          <button type="button" onClick={onCancel}>
            Cancel
          </button>
          {onConfirm ? (
            <button className="primary-button" type="button" onClick={onConfirm}>
              {confirmLabel}
            </button>
          ) : null}
        </div>
      </div>
    </div>
  );
};
