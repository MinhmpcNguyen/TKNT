import React, { FormEvent, useEffect, useMemo, useState } from 'react';
import {
  createAdminInventoryItem,
  createAdminUser,
  fetchInventory,
  listAdminInventoryItems,
  listAdminUsers,
  updateAdminBillingSettings,
  updateAdminUser,
} from '../api';
import { formatVnd } from '../billing';
import { useEditorStore } from '../store';
import type { AdminUser, UserRole } from '../types';

type AdminPanelProps = {
  onClose: () => void;
};

type AdminTab = 'users' | 'furniture' | 'billing';

type UserFormState = {
  email: string;
  password: string;
  displayName: string;
  role: UserRole;
};

type FurnitureFormState = {
  name: string;
  category: string;
  styleTags: string;
  material: string;
  brand: string;
  lengthMm: string;
  widthMm: string;
  heightMm: string;
  priceAmount: string;
  priceCurrency: string;
  modelReference: string;
  previewReference: string;
  attributesJson: string;
};

type BillingDraftState = {
  perCreditPriceVnd: string;
  monthlyPriceVnd: string;
  monthlyCreditQuota: string;
};

const DEFAULT_USER_FORM: UserFormState = {
  email: '',
  password: '',
  displayName: '',
  role: 'user',
};

const DEFAULT_FURNITURE_FORM: FurnitureFormState = {
  name: '',
  category: '',
  styleTags: '',
  material: '',
  brand: '',
  lengthMm: '',
  widthMm: '',
  heightMm: '',
  priceAmount: '',
  priceCurrency: 'VND',
  modelReference: '',
  previewReference: '',
  attributesJson: '{}',
};

export const AdminPanel: React.FC<AdminPanelProps> = ({ onClose }) => {
  const {
    appendRunLog,
    authToken,
    billingSettings,
    currentUser,
    setInventory,
    updateBillingSettings,
  } = useEditorStore();
  const [activeTab, setActiveTab] = useState<AdminTab>('users');
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [inventoryCount, setInventoryCount] = useState(0);
  const [userForm, setUserForm] = useState<UserFormState>(DEFAULT_USER_FORM);
  const [furnitureForm, setFurnitureForm] =
    useState<FurnitureFormState>(DEFAULT_FURNITURE_FORM);
  const [billingDraft, setBillingDraft] = useState<BillingDraftState>({
    monthlyCreditQuota: String(billingSettings.monthlyCreditQuota),
    monthlyPriceVnd: String(billingSettings.monthlyPriceVnd),
    perCreditPriceVnd: String(billingSettings.perCreditPriceVnd),
  });
  const [loading, setLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const activeUsers = useMemo(
    () => users.filter((user) => user.is_active).length,
    [users]
  );

  useEffect(() => {
    setBillingDraft({
      monthlyCreditQuota: String(billingSettings.monthlyCreditQuota),
      monthlyPriceVnd: String(billingSettings.monthlyPriceVnd),
      perCreditPriceVnd: String(billingSettings.perCreditPriceVnd),
    });
  }, [
    billingSettings.monthlyCreditQuota,
    billingSettings.monthlyPriceVnd,
    billingSettings.perCreditPriceVnd,
  ]);

  useEffect(() => {
    let cancelled = false;
    const loadAdminData = async () => {
      setLoading(true);
      setErrorMessage(null);
      try {
        const [nextUsers, nextInventory] = await Promise.all([
          listAdminUsers(authToken),
          listAdminInventoryItems(authToken),
        ]);
        if (cancelled) {
          return;
        }
        setUsers(nextUsers);
        setInventoryCount(nextInventory.length);
      } catch (loadError) {
        if (cancelled) {
          return;
        }
        setErrorMessage(errorToMessage(loadError));
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };
    void loadAdminData();
    return () => {
      cancelled = true;
    };
  }, [authToken]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  const refreshUsers = async (): Promise<void> => {
    const nextUsers = await listAdminUsers(authToken);
    setUsers(nextUsers);
  };

  const handleCreateUser = async (event: FormEvent): Promise<void> => {
    event.preventDefault();
    setStatusMessage(null);
    setErrorMessage(null);
    try {
      const created = await createAdminUser(
        {
          display_name: cleanOptionalText(userForm.displayName),
          email: userForm.email.trim(),
          password: userForm.password,
          role: userForm.role,
        },
        authToken
      );
      setUsers((current) => [created, ...current]);
      setUserForm(DEFAULT_USER_FORM);
      setStatusMessage(`Created user ${created.email}.`);
      appendRunLog(`Admin created user ${created.email}.`);
    } catch (error) {
      setErrorMessage(errorToMessage(error));
    }
  };

  const handleUpdateUser = async (
    user: AdminUser,
    patch: { role?: UserRole; is_active?: boolean }
  ): Promise<void> => {
    setStatusMessage(null);
    setErrorMessage(null);
    try {
      const updated = await updateAdminUser(
        user.id,
        {
          display_name: user.display_name ?? null,
          role: patch.role ?? user.role,
          is_active: patch.is_active ?? user.is_active,
        },
        authToken
      );
      setUsers((current) =>
        current.map((item) => (item.id === updated.id ? updated : item))
      );
      setStatusMessage(`Updated user ${updated.email}.`);
      appendRunLog(`Admin updated user ${updated.email}.`);
    } catch (error) {
      setErrorMessage(errorToMessage(error));
      await refreshUsers().catch(() => undefined);
    }
  };

  const handleCreateFurniture = async (event: FormEvent): Promise<void> => {
    event.preventDefault();
    setStatusMessage(null);
    setErrorMessage(null);
    try {
      const attributes = parseAttributesJson(furnitureForm.attributesJson);
      const created = await createAdminInventoryItem(
        {
          attributes,
          brand: cleanOptionalText(furnitureForm.brand),
          category: furnitureForm.category.trim(),
          height_mm: readOptionalNumber(furnitureForm.heightMm),
          length_mm: readOptionalNumber(furnitureForm.lengthMm),
          material: cleanOptionalText(furnitureForm.material),
          model_reference: cleanOptionalText(furnitureForm.modelReference),
          name: furnitureForm.name.trim(),
          preview_reference: cleanOptionalText(furnitureForm.previewReference),
          price_amount: readOptionalNumber(furnitureForm.priceAmount),
          price_currency: furnitureForm.priceCurrency.trim() || 'VND',
          style_tags: splitTags(furnitureForm.styleTags),
          width_mm: readOptionalNumber(furnitureForm.widthMm),
        },
        authToken
      );
      const nextInventory = await fetchInventory(authToken);
      setInventory(nextInventory);
      setInventoryCount(nextInventory.length);
      setFurnitureForm(DEFAULT_FURNITURE_FORM);
      setStatusMessage(`Created furniture ${created.name}.`);
      appendRunLog(`Admin added furniture ${created.name}.`);
    } catch (error) {
      setErrorMessage(errorToMessage(error));
    }
  };

  const handleSaveBilling = async (event: FormEvent): Promise<void> => {
    event.preventDefault();
    setStatusMessage(null);
    setErrorMessage(null);
    try {
      const updated = await updateAdminBillingSettings(
        {
          monthly_credit_quota: readPositiveInteger(
            billingDraft.monthlyCreditQuota
          ),
          monthly_price_vnd: readPositiveInteger(billingDraft.monthlyPriceVnd),
          per_credit_price_vnd: readPositiveInteger(
            billingDraft.perCreditPriceVnd
          ),
        },
        authToken
      );
      updateBillingSettings({
        monthlyCreditQuota: updated.monthly_credit_quota,
        monthlyPriceVnd: updated.monthly_price_vnd,
        perCreditPriceVnd: updated.per_credit_price_vnd,
      });
      setStatusMessage('Billing settings updated.');
      appendRunLog('Admin updated billing settings.');
    } catch (error) {
      setErrorMessage(errorToMessage(error));
    }
  };

  return (
    <div
      className="admin-panel-overlay"
      role="dialog"
      aria-modal="true"
      aria-labelledby="admin-panel-title"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) {
          onClose();
        }
      }}
    >
      <div className="admin-panel-card">
        <header className="admin-panel-header">
          <div>
            <div id="admin-panel-title" className="admin-panel-title">
              Admin
            </div>
            <div className="admin-panel-subtitle">
              {activeUsers}/{users.length} active users - {inventoryCount} furniture
              assets
            </div>
          </div>
          <button type="button" onClick={onClose}>
            Close
          </button>
        </header>

        <div className="admin-tabs" role="tablist" aria-label="Admin sections">
          {(['users', 'furniture', 'billing'] as AdminTab[]).map((tab) => (
            <button
              key={tab}
              type="button"
              className={activeTab === tab ? 'selected' : ''}
              onClick={() => setActiveTab(tab)}
            >
              {tabLabel(tab)}
            </button>
          ))}
        </div>

        <div className="admin-panel-body">
          {loading ? (
            <div className="admin-inline-note">Loading admin data...</div>
          ) : null}
          {errorMessage ? (
            <div className="admin-inline-error">{errorMessage}</div>
          ) : null}
          {statusMessage ? (
            <div className="admin-inline-success">{statusMessage}</div>
          ) : null}
          {activeTab === 'users' ? (
            <UsersAdminSection
              currentUserId={currentUser?.id ?? null}
              userForm={userForm}
              users={users}
              onCreateUser={handleCreateUser}
              onFormChange={setUserForm}
              onUpdateUser={handleUpdateUser}
            />
          ) : null}
          {activeTab === 'furniture' ? (
            <FurnitureAdminSection
              furnitureForm={furnitureForm}
              onCreateFurniture={handleCreateFurniture}
              onFormChange={setFurnitureForm}
            />
          ) : null}
          {activeTab === 'billing' ? (
            <BillingAdminSection
              billingDraft={billingDraft}
              billingSettings={billingSettings}
              onDraftChange={setBillingDraft}
              onSaveBilling={handleSaveBilling}
            />
          ) : null}
        </div>
      </div>
    </div>
  );
};

const UsersAdminSection: React.FC<{
  currentUserId: string | null;
  userForm: UserFormState;
  users: AdminUser[];
  onCreateUser: (event: FormEvent) => Promise<void>;
  onFormChange: (next: UserFormState) => void;
  onUpdateUser: (
    user: AdminUser,
    patch: { role?: UserRole; is_active?: boolean }
  ) => Promise<void>;
}> = ({
  currentUserId,
  userForm,
  users,
  onCreateUser,
  onFormChange,
  onUpdateUser,
}) => (
  <section className="admin-section">
    <form className="admin-form-grid" onSubmit={(event) => void onCreateUser(event)}>
      <label>
        <span>Email</span>
        <input
          type="email"
          required
          value={userForm.email}
          onChange={(event) =>
            onFormChange({ ...userForm, email: event.target.value })
          }
        />
      </label>
      <label>
        <span>Password</span>
        <input
          type="password"
          required
          minLength={8}
          value={userForm.password}
          onChange={(event) =>
            onFormChange({ ...userForm, password: event.target.value })
          }
        />
      </label>
      <label>
        <span>Display name</span>
        <input
          type="text"
          value={userForm.displayName}
          onChange={(event) =>
            onFormChange({ ...userForm, displayName: event.target.value })
          }
        />
      </label>
      <label>
        <span>Role</span>
        <select
          value={userForm.role}
          onChange={(event) =>
            onFormChange({ ...userForm, role: event.target.value as UserRole })
          }
        >
          <option value="user">User</option>
          <option value="admin">Admin</option>
        </select>
      </label>
      <button className="primary-button" type="submit">
        Create User
      </button>
    </form>

    <div className="admin-table">
      <div className="admin-table-row admin-table-head">
        <span>User</span>
        <span>Role</span>
        <span>Status</span>
        <span>Actions</span>
      </div>
      {users.map((user) => {
        const isSelf = user.id === currentUserId;
        return (
          <div className="admin-table-row" key={user.id}>
            <span>
              <strong>{user.display_name || user.email}</strong>
              <small>{user.email}</small>
            </span>
            <span>
              <select
                disabled={isSelf}
                value={user.role}
                onChange={(event) =>
                  void onUpdateUser(user, {
                    role: event.target.value as UserRole,
                  })
                }
              >
                <option value="user">User</option>
                <option value="admin">Admin</option>
              </select>
            </span>
            <span>{user.is_active ? 'Active' : 'Inactive'}</span>
            <span>
              <button
                disabled={isSelf}
                type="button"
                onClick={() =>
                  void onUpdateUser(user, { is_active: !user.is_active })
                }
              >
                {user.is_active ? 'Deactivate' : 'Reactivate'}
              </button>
            </span>
          </div>
        );
      })}
    </div>
  </section>
);

const FurnitureAdminSection: React.FC<{
  furnitureForm: FurnitureFormState;
  onCreateFurniture: (event: FormEvent) => Promise<void>;
  onFormChange: (next: FurnitureFormState) => void;
}> = ({ furnitureForm, onCreateFurniture, onFormChange }) => (
  <section className="admin-section">
    <form
      className="admin-form-grid admin-furniture-form"
      onSubmit={(event) => void onCreateFurniture(event)}
    >
      <label>
        <span>Name</span>
        <input
          required
          value={furnitureForm.name}
          onChange={(event) =>
            onFormChange({ ...furnitureForm, name: event.target.value })
          }
        />
      </label>
      <label>
        <span>Category</span>
        <input
          required
          value={furnitureForm.category}
          onChange={(event) =>
            onFormChange({ ...furnitureForm, category: event.target.value })
          }
        />
      </label>
      <label>
        <span>Style tags</span>
        <input
          placeholder="modern, japandi"
          value={furnitureForm.styleTags}
          onChange={(event) =>
            onFormChange({ ...furnitureForm, styleTags: event.target.value })
          }
        />
      </label>
      <label>
        <span>Material</span>
        <input
          value={furnitureForm.material}
          onChange={(event) =>
            onFormChange({ ...furnitureForm, material: event.target.value })
          }
        />
      </label>
      <label>
        <span>Brand</span>
        <input
          value={furnitureForm.brand}
          onChange={(event) =>
            onFormChange({ ...furnitureForm, brand: event.target.value })
          }
        />
      </label>
      <div className="admin-dimension-grid">
        <label>
          <span>L mm</span>
          <input
            min={1}
            type="number"
            value={furnitureForm.lengthMm}
            onChange={(event) =>
              onFormChange({ ...furnitureForm, lengthMm: event.target.value })
            }
          />
        </label>
        <label>
          <span>W mm</span>
          <input
            min={1}
            type="number"
            value={furnitureForm.widthMm}
            onChange={(event) =>
              onFormChange({ ...furnitureForm, widthMm: event.target.value })
            }
          />
        </label>
        <label>
          <span>H mm</span>
          <input
            min={1}
            type="number"
            value={furnitureForm.heightMm}
            onChange={(event) =>
              onFormChange({ ...furnitureForm, heightMm: event.target.value })
            }
          />
        </label>
      </div>
      <label>
        <span>Price</span>
        <input
          min={0}
          type="number"
          value={furnitureForm.priceAmount}
          onChange={(event) =>
            onFormChange({ ...furnitureForm, priceAmount: event.target.value })
          }
        />
      </label>
      <label>
        <span>Currency</span>
        <input
          value={furnitureForm.priceCurrency}
          onChange={(event) =>
            onFormChange({ ...furnitureForm, priceCurrency: event.target.value })
          }
        />
      </label>
      <label>
        <span>Model URL/path</span>
        <input
          value={furnitureForm.modelReference}
          onChange={(event) =>
            onFormChange({
              ...furnitureForm,
              modelReference: event.target.value,
            })
          }
        />
      </label>
      <label>
        <span>Preview URL/path</span>
        <input
          value={furnitureForm.previewReference}
          onChange={(event) =>
            onFormChange({
              ...furnitureForm,
              previewReference: event.target.value,
            })
          }
        />
      </label>
      <label className="admin-form-wide">
        <span>Attributes JSON</span>
        <textarea
          rows={5}
          value={furnitureForm.attributesJson}
          onChange={(event) =>
            onFormChange({
              ...furnitureForm,
              attributesJson: event.target.value,
            })
          }
        />
      </label>
      <button className="primary-button" type="submit">
        Add Furniture
      </button>
    </form>
  </section>
);

const BillingAdminSection: React.FC<{
  billingDraft: BillingDraftState;
  billingSettings: {
    monthlyCreditQuota: number;
    monthlyPriceVnd: number;
    perCreditPriceVnd: number;
  };
  onDraftChange: (next: BillingDraftState) => void;
  onSaveBilling: (event: FormEvent) => Promise<void>;
}> = ({ billingDraft, billingSettings, onDraftChange, onSaveBilling }) => (
  <section className="admin-section">
    <form className="admin-form-grid" onSubmit={(event) => void onSaveBilling(event)}>
      <label>
        <span>Per credit price</span>
        <input
          min={1}
          required
          type="number"
          value={billingDraft.perCreditPriceVnd}
          onChange={(event) =>
            onDraftChange({
              ...billingDraft,
              perCreditPriceVnd: event.target.value,
            })
          }
        />
      </label>
      <label>
        <span>Monthly price</span>
        <input
          min={1}
          required
          type="number"
          value={billingDraft.monthlyPriceVnd}
          onChange={(event) =>
            onDraftChange({
              ...billingDraft,
              monthlyPriceVnd: event.target.value,
            })
          }
        />
      </label>
      <label>
        <span>Monthly quota</span>
        <input
          min={1}
          required
          type="number"
          value={billingDraft.monthlyCreditQuota}
          onChange={(event) =>
            onDraftChange({
              ...billingDraft,
              monthlyCreditQuota: event.target.value,
            })
          }
        />
      </label>
      <button className="primary-button" type="submit">
        Save Billing
      </button>
    </form>
    <div className="admin-billing-summary">
      <span>Current per credit: {formatVnd(billingSettings.perCreditPriceVnd)}</span>
      <span>Monthly: {formatVnd(billingSettings.monthlyPriceVnd)}</span>
      <span>Quota: {billingSettings.monthlyCreditQuota} credits</span>
    </div>
  </section>
);

function tabLabel(tab: AdminTab): string {
  if (tab === 'users') {
    return 'Users';
  }
  if (tab === 'furniture') {
    return 'Furniture';
  }
  return 'Billing';
}

function cleanOptionalText(value: string): string | null {
  const trimmed = value.trim();
  return trimmed || null;
}

function splitTags(value: string): string[] {
  return value
    .split(',')
    .map((tag) => tag.trim())
    .filter(Boolean);
}

function readOptionalNumber(value: string): number | null {
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }
  const parsed = Number(trimmed);
  if (!Number.isFinite(parsed)) {
    throw new Error('Numeric fields must be valid numbers.');
  }
  return parsed;
}

function readPositiveInteger(value: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error('Billing settings must be positive numbers.');
  }
  return Math.round(parsed);
}

function parseAttributesJson(value: string): Record<string, unknown> {
  const trimmed = value.trim();
  if (!trimmed) {
    return {};
  }
  const parsed = JSON.parse(trimmed) as unknown;
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('Attributes JSON must be an object.');
  }
  return parsed as Record<string, unknown>;
}

function errorToMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Admin action failed.';
}
