import React, { useMemo, useState } from 'react';
import { updateSavedLayoutCase } from '../api';
import { useEditorStore } from '../store';
import {
  DesignPlanV1,
  FloorplanGeometry,
  GeneratedRenderRecord,
  LayoutVariant,
  SavedLayoutRecord,
} from '../types';

function canReuseSavedRender(render: GeneratedRenderRecord): boolean {
  return render.source === 'snapshot_render';
}

function parseTimestampMs(value: string | null | undefined): number | null {
  if (!value) {
    return null;
  }
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function resolveSavedLayoutIdForRender(
  render: GeneratedRenderRecord,
  savedLayouts: SavedLayoutRecord[]
): string | null {
  const caseId = readRenderContextText(render, 'case_id');
  const optionId = readRenderContextText(render, 'option_id');
  if (caseId && optionId) {
    return (
      savedLayouts.find(
        (layout) => layout.case_id === caseId && layout.option_id === optionId
      )?.id ?? null
    );
  }
  const explicitRootLayoutId = readRenderContextText(render, 'root_layout_id');
  if (explicitRootLayoutId) {
    return explicitRootLayoutId;
  }
  return null;
}

function readRenderContextText(
  render: GeneratedRenderRecord,
  key: 'root_layout_id' | 'case_id' | 'option_id'
): string | null {
  const topLevelValue = render[key];
  if (typeof topLevelValue === 'string' && topLevelValue.trim()) {
    return topLevelValue.trim();
  }
  const metaValue = render.meta[key];
  if (typeof metaValue === 'string' && metaValue.trim()) {
    return metaValue.trim();
  }
  return null;
}

export const AccountPanel: React.FC = () => {
  const {
    currentUser,
    savedLayouts,
    savedRenders,
    activeRootLayoutId,
    activeLayoutCaseId,
    selectedSavedRenderId,
    setSavedLayouts,
    setFloorplan,
    setDesign,
    setStyledResult,
    clearLayoutVariants,
    setLayoutVariants,
    selectLayoutVariant,
    setViewMode,
    setActiveRootLayoutId,
    setActiveLayoutCaseId,
    setActiveSavedLayoutId,
    selectSavedRender,
    appendRunLog,
  } = useEditorStore();
  const [layoutSearchQuery, setLayoutSearchQuery] = useState('');
  const [savedLayoutsOpen, setSavedLayoutsOpen] = useState(true);
  const [savedRendersOpen, setSavedRendersOpen] = useState(true);
  const [expandedCaseKeys, setExpandedCaseKeys] = useState<ReadonlySet<string>>(
    () => new Set()
  );
  const [editingCaseKey, setEditingCaseKey] = useState<string | null>(null);
  const [caseNameDraft, setCaseNameDraft] = useState('');
  const [savingCaseKey, setSavingCaseKey] = useState<string | null>(null);
  const [caseNameError, setCaseNameError] = useState<string | null>(null);
  const [isOpen, setIsOpen] = useState(false);
  const visibleSavedRenders = activeRootLayoutId
    ? savedRenders.filter(
        (render) =>
          resolveSavedLayoutIdForRender(render, savedLayouts) === activeRootLayoutId
      )
    : savedRenders;
  const filteredSavedLayouts = useMemo(() => {
    const query = layoutSearchQuery.trim().toLowerCase();
    if (!query) {
      return savedLayouts;
    }
    return savedLayouts.filter((layout) => {
      const haystack = [
        layout.name,
        layout.case_name ?? '',
        layout.created_at ?? '',
        layout.updated_at ?? '',
      ]
        .join(' ')
        .toLowerCase();
      return haystack.includes(query);
    });
  }, [layoutSearchQuery, savedLayouts]);
  const groupedSavedLayouts = useMemo(
    () => groupSavedLayouts(filteredSavedLayouts),
    [filteredSavedLayouts]
  );

  const toggleLayoutCaseGroup = (groupKey: string) => {
    setExpandedCaseKeys((current) => {
      const next = new Set(current);
      if (next.has(groupKey)) {
        next.delete(groupKey);
      } else {
        next.add(groupKey);
      }
      return next;
    });
  };

  const beginEditLayoutCaseName = (group: SavedLayoutGroup) => {
    setEditingCaseKey(group.key);
    setCaseNameDraft(group.title);
    setCaseNameError(null);
  };

  const cancelEditLayoutCaseName = () => {
    setEditingCaseKey(null);
    setCaseNameDraft('');
    setCaseNameError(null);
  };

  const saveLayoutCaseName = async (group: SavedLayoutGroup) => {
    if (!group.caseId) {
      return;
    }
    const nextName = caseNameDraft.trim();
    if (!nextName) {
      setCaseNameError('Case name is required.');
      return;
    }
    setSavingCaseKey(group.key);
    setCaseNameError(null);
    try {
      const savedCase = await updateSavedLayoutCase(group.caseId, { name: nextName });
      setSavedLayouts(
        savedLayouts.map((layout) =>
          layout.case_id === group.caseId
            ? { ...layout, case_name: savedCase.name }
            : layout
        )
      );
      setEditingCaseKey(null);
      setCaseNameDraft('');
      appendRunLog(`Renamed layout case to "${savedCase.name}".`);
    } catch (error) {
      setCaseNameError(
        error instanceof Error ? error.message : 'Failed to rename layout case.'
      );
    } finally {
      setSavingCaseKey(null);
    }
  };

  const loadLayout = (layout: SavedLayoutRecord) => {
    const nextFloorplan = layout.floorplan_json as unknown as FloorplanGeometry;
    const nextDesign = layout.design_json as unknown as DesignPlanV1 | null;
    setFloorplan(nextFloorplan);
    setDesign(
      nextDesign ?? {
        floorplan_geometry: nextFloorplan,
        placements: [],
      }
    );
    setStyledResult(layout.styled_result_json ?? null);
    if (layout.case_id) {
      const siblings = savedLayouts
        .filter((item) => item.case_id === layout.case_id)
        .sort(compareSavedLayoutOptions)
        .map(savedLayoutToVariant);
      setLayoutVariants(siblings, layout.option_id ?? layout.id);
      selectLayoutVariant(layout.option_id ?? layout.id);
      setActiveLayoutCaseId(layout.case_id);
      setActiveSavedLayoutId(layout.id);
      setExpandedCaseKeys((current) => {
        const next = new Set(current);
        next.add(`case:${layout.case_id}`);
        return next;
      });
    } else {
      clearLayoutVariants();
      setActiveLayoutCaseId(null);
      setActiveSavedLayoutId(null);
    }
    setActiveRootLayoutId(layout.id);
    selectSavedRender(null);
    setViewMode('2d');
    setSavedLayoutsOpen(false);
    appendRunLog(
      layout.case_id
        ? `Loaded saved option "${layout.name}" with its case options.`
        : `Loaded saved layout "${layout.name}".`
    );
  };

  const handleReuseSavedRender = (render: GeneratedRenderRecord) => {
    if (!canReuseSavedRender(render)) {
      appendRunLog(
        `Saved render "${render.id}" is not a 3D snapshot render and cannot be reopened.`
      );
      return;
    }
    setViewMode('3d');
    if (selectedSavedRenderId === render.id) {
      selectSavedRender(null);
      window.setTimeout(() => selectSavedRender(render.id), 0);
    } else {
      selectSavedRender(render.id);
    }
    appendRunLog(`Loaded saved render "${render.id}" into the 3D editor for more edits.`);
  };

  return (
    <section className={`panel account-panel collapsible-panel${isOpen ? ' open' : ''}`}>
      <button
        className="panel-collapse-trigger"
        type="button"
        onClick={() => setIsOpen((current) => !current)}
        aria-expanded={isOpen}
      >
        <span>
          <span className="panel-collapse-title">My Workspace</span>
          <span className="panel-collapse-subtitle">
            {savedLayouts.length} layouts · {visibleSavedRenders.length} renders
          </span>
        </span>
        <span className="panel-collapse-caret">{isOpen ? 'Hide' : 'Open'}</span>
      </button>

      {isOpen ? (
        <div className="panel-collapse-body account-panel-body">
          <div className="account-summary">
            <strong>{currentUser?.display_name || currentUser?.email}</strong>
            <span>{currentUser?.email}</span>
          </div>

          <div className="account-section account-section-compact">
            <div className="account-search-row">
              <input
                type="search"
                value={layoutSearchQuery}
                onChange={(event) => {
                  setLayoutSearchQuery(event.target.value);
                  setSavedLayoutsOpen(true);
                }}
                placeholder="Search saved layouts"
              />
            </div>
            <button
              className="account-section-trigger"
              type="button"
              onClick={() => setSavedLayoutsOpen((current) => !current)}
              aria-expanded={savedLayoutsOpen}
            >
              <span>Saved Layouts</span>
              <strong>{filteredSavedLayouts.length}</strong>
            </button>
            {savedLayoutsOpen ? (
              <div className="account-list account-scroll-list">
                {savedLayouts.length === 0 ? (
                  <p className="muted">No layouts saved yet.</p>
                ) : filteredSavedLayouts.length === 0 ? (
                  <p className="muted">No saved layouts match this search.</p>
                ) : (
                  groupedSavedLayouts.map((group) =>
                    group.caseId
                      ? renderLayoutCaseGroup({
                          group,
                          isActive: activeLayoutCaseId === group.caseId,
                          isExpanded: expandedCaseKeys.has(group.key),
                          isEditing: editingCaseKey === group.key,
                          isSaving: savingCaseKey === group.key,
                          caseNameDraft,
                          caseNameError:
                            editingCaseKey === group.key ? caseNameError : null,
                          beginEditLayoutCaseName,
                          cancelEditLayoutCaseName,
                          loadLayout,
                          saveLayoutCaseName,
                          setCaseNameDraft,
                          toggleLayoutCaseGroup,
                        })
                      : renderLayoutListItem(group.layouts[0], loadLayout)
                  )
                )}
              </div>
            ) : null}
          </div>

          <div className="account-section account-section-compact">
            <button
              className="account-section-trigger"
              type="button"
              onClick={() => setSavedRendersOpen((current) => !current)}
              aria-expanded={savedRendersOpen}
            >
              <span>Saved Renders</span>
              <strong>{visibleSavedRenders.length}</strong>
            </button>
            {savedRendersOpen ? (
              <>
                {activeRootLayoutId ? (
                  <p className="muted">
                    Showing renders linked to the loaded root layout.
                  </p>
                ) : null}
                <div className="account-render-grid account-render-scroll">
                  {visibleSavedRenders.length === 0 ? (
                    <p className="muted">
                      {activeRootLayoutId
                        ? 'No renders are linked to this root layout yet.'
                        : 'Your image renders will appear here.'}
                    </p>
                  ) : (
                    visibleSavedRenders.map((render) => (
                      <div
                        key={render.id}
                        className={`saved-render-card ${
                          selectedSavedRenderId === render.id ? 'selected' : ''
                        }`}
                        onDoubleClick={() => handleReuseSavedRender(render)}
                      >
                        <img src={render.file_url} alt={render.prompt} />
                        <div className="saved-render-meta">
                          <strong>{render.model_name}</strong>
                          <span>{render.source}</span>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </>
            ) : null}
          </div>
        </div>
      ) : null}
    </section>
  );
};

type SavedLayoutGroup = {
  key: string;
  title: string;
  displayTitle: string;
  subtitle: string | null;
  caseId: string | null;
  timestampMs: number;
  layouts: SavedLayoutRecord[];
};

type SavedLayoutCaseTitle = {
  displayTitle: string;
  subtitle: string | null;
};

type RenderLayoutCaseGroupParams = {
  group: SavedLayoutGroup;
  isActive: boolean;
  isEditing: boolean;
  isExpanded: boolean;
  isSaving: boolean;
  caseNameDraft: string;
  caseNameError: string | null;
  beginEditLayoutCaseName: (group: SavedLayoutGroup) => void;
  cancelEditLayoutCaseName: () => void;
  loadLayout: (layout: SavedLayoutRecord) => void;
  saveLayoutCaseName: (group: SavedLayoutGroup) => void;
  setCaseNameDraft: (name: string) => void;
  toggleLayoutCaseGroup: (groupKey: string) => void;
};

const MAX_CASE_PRIMARY_TITLE_LENGTH = 24;
const MAX_CASE_SUBTITLE_SEGMENT_LENGTH = 26;

const ROOM_LABELS: Record<string, string> = {
  bedroom: 'Bedroom',
  kitchen: 'Kitchen',
  office: 'Office',
  'living room': 'Living Room',
  studio: 'Studio',
};

function groupSavedLayouts(layouts: SavedLayoutRecord[]): SavedLayoutGroup[] {
  const groups = new Map<string, SavedLayoutGroup>();
  for (const layout of layouts) {
    const caseId = layout.case_id?.trim() || null;
    const key = caseId ? `case:${caseId}` : `layout:${layout.id}`;
    const existing = groups.get(key);
    if (existing) {
      existing.layouts.push(layout);
      continue;
    }
    const title = caseId ? layout.case_name || caseId : layout.name;
    groups.set(key, {
      key,
      title,
      displayTitle: title,
      subtitle: null,
      caseId,
      timestampMs: 0,
      layouts: [layout],
    });
  }
  return Array.from(groups.values())
    .map((group) => {
      const sortedLayouts = [...group.layouts].sort(compareSavedLayoutOptions);
      const timestampMs = layoutsTimestampMs(sortedLayouts);
      const caseTitle = group.caseId
        ? formatSavedLayoutCaseTitle(group.title, timestampMs)
        : { displayTitle: group.title, subtitle: null };
      return {
        ...group,
        ...caseTitle,
        timestampMs,
        layouts: sortedLayouts,
      };
    })
    .sort((left, right) => right.timestampMs - left.timestampMs);
}

function renderLayoutCaseGroup(params: RenderLayoutCaseGroupParams) {
  const {
    group,
    isActive,
    isEditing,
    isExpanded,
    isSaving,
    caseNameDraft,
    caseNameError,
    beginEditLayoutCaseName,
    cancelEditLayoutCaseName,
    loadLayout,
    saveLayoutCaseName,
    setCaseNameDraft,
    toggleLayoutCaseGroup,
  } = params;
  const toggleLabel = `${isExpanded ? 'Hide' : 'Show'} options for ${
    group.displayTitle
  }${group.subtitle ? `, ${group.subtitle}` : ''}`;
  return (
    <div
      key={group.key}
      className={`account-layout-case-group${isActive ? ' active' : ''}`}
    >
      {isEditing ? (
        <div className="account-layout-case-edit-form">
          <input
            value={caseNameDraft}
            onChange={(event) => setCaseNameDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                void saveLayoutCaseName(group);
              }
              if (event.key === 'Escape') {
                cancelEditLayoutCaseName();
              }
            }}
            aria-label="Case name"
            disabled={isSaving}
          />
          <button
            type="button"
            onClick={() => void saveLayoutCaseName(group)}
            disabled={isSaving}
          >
            {isSaving ? 'Saving' : 'Save'}
          </button>
          <button
            type="button"
            onClick={cancelEditLayoutCaseName}
            disabled={isSaving}
          >
            Cancel
          </button>
          {caseNameError ? (
            <span className="account-layout-case-error">{caseNameError}</span>
          ) : null}
        </div>
      ) : (
        <div className="account-layout-case-header">
          <button
            className="account-layout-case-title"
            type="button"
            onClick={() => toggleLayoutCaseGroup(group.key)}
            aria-expanded={isExpanded}
            aria-label={toggleLabel}
            title={group.title}
          >
            <span className="account-layout-case-copy">
              <strong>{group.displayTitle}</strong>
              {group.subtitle ? <span>{group.subtitle}</span> : null}
            </span>
            <span className="account-layout-case-count">
              {group.layouts.length}
            </span>
            <span className="account-layout-case-caret" aria-hidden="true">
              {isExpanded ? '-' : '+'}
            </span>
          </button>
          <button
            className="account-layout-case-edit-button"
            type="button"
            onClick={() => beginEditLayoutCaseName(group)}
          >
            Edit
          </button>
        </div>
      )}
      {isExpanded ? (
        <div className="account-layout-case-options">
          {group.layouts.map((layout) => renderLayoutListItem(layout, loadLayout))}
        </div>
      ) : null}
    </div>
  );
}

function formatSavedLayoutCaseTitle(
  title: string,
  timestampMs: number
): SavedLayoutCaseTitle {
  const normalizedTitle = normalizeCaseTitle(title);
  if (!normalizedTitle) {
    return {
      displayTitle: 'Saved Case',
      subtitle: formatCaseTimestamp(timestampMs),
    };
  }

  const roomLabel = extractRoomLabel(normalizedTitle);
  const intentLabel = extractIntentLabel(normalizedTitle);
  const displayTitle = truncateAtWordBoundary(
    roomLabel ?? normalizedTitle,
    MAX_CASE_PRIMARY_TITLE_LENGTH
  );
  const subtitleSegments = [
    formatCaseTimestamp(timestampMs),
    intentLabel
      ? truncateAtWordBoundary(intentLabel, MAX_CASE_SUBTITLE_SEGMENT_LENGTH)
      : null,
  ].filter((item): item is string => Boolean(item));

  return {
    displayTitle,
    subtitle: subtitleSegments.length > 0 ? subtitleSegments.join(' · ') : null,
  };
}

function formatCaseTimestamp(timestampMs: number): string | null {
  if (!Number.isFinite(timestampMs) || timestampMs <= 0) {
    return null;
  }
  return new Intl.DateTimeFormat(undefined, {
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    month: '2-digit',
  }).format(new Date(timestampMs));
}

function normalizeCaseTitle(title: string): string {
  return title.replace(/\s+/g, ' ').trim();
}

function extractRoomLabel(title: string): string | null {
  const match = title.match(/\b(living room|bedroom|office|studio|kitchen)\b/i);
  if (!match) {
    return null;
  }
  return ROOM_LABELS[match[1].toLowerCase()] ?? toTitleCase(match[1]);
}

function extractIntentLabel(title: string): string | null {
  const match = title.match(/\bdesigned\s+for\s+([^.,;!?]+)/i);
  if (!match) {
    return null;
  }
  const intent = match[1].replace(/\s+and\s+/gi, ' & ').trim();
  return intent ? toTitleCase(intent) : null;
}

function toTitleCase(value: string): string {
  return value
    .split(/\s+/)
    .map((word) =>
      word === '&' ? word : word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
    )
    .join(' ');
}

function truncateAtWordBoundary(value: string, maxLength: number): string {
  if (value.length <= maxLength) {
    return value;
  }
  const trimmed = value.slice(0, maxLength - 3).trimEnd();
  const lastSpaceIndex = trimmed.lastIndexOf(' ');
  const prefix =
    lastSpaceIndex >= Math.floor(maxLength * 0.6)
      ? trimmed.slice(0, lastSpaceIndex)
      : trimmed;
  return `${prefix}...`;
}

function compareSavedLayoutOptions(
  left: SavedLayoutRecord,
  right: SavedLayoutRecord
): number {
  const leftIndex = left.option_index ?? Number.MAX_SAFE_INTEGER;
  const rightIndex = right.option_index ?? Number.MAX_SAFE_INTEGER;
  if (leftIndex !== rightIndex) {
    return leftIndex - rightIndex;
  }
  const rightTime = parseTimestampMs(right.updated_at ?? right.created_at) ?? 0;
  const leftTime = parseTimestampMs(left.updated_at ?? left.created_at) ?? 0;
  return rightTime - leftTime;
}

function layoutsTimestampMs(layouts: SavedLayoutRecord[]): number {
  return Math.max(
    ...layouts.map(
      (layout) => parseTimestampMs(layout.updated_at ?? layout.created_at) ?? 0
    )
  );
}

function renderLayoutListItem(
  layout: SavedLayoutRecord,
  loadLayout: (layout: SavedLayoutRecord) => void
) {
  return (
    <div
      key={layout.id}
      className="account-list-item account-list-item-loadable"
      onDoubleClick={() => loadLayout(layout)}
      title="Double-click to load"
    >
      <div>
        <div className="account-list-title">{layout.name}</div>
        <div className="account-list-meta">
          {layout.updated_at || layout.created_at || 'Saved recently'}
        </div>
      </div>
      <button
        onClick={() => loadLayout(layout)}
        onDoubleClick={(event) => event.stopPropagation()}
        type="button"
      >
        Load
      </button>
    </div>
  );
}

function savedLayoutToVariant(layout: SavedLayoutRecord): LayoutVariant {
  const meta = layout.meta ?? {};
  const absoluteLayout =
    meta.absolute_layout && typeof meta.absolute_layout === 'object'
      ? (meta.absolute_layout as Record<string, unknown>)
      : {};
  return {
    variant_id: layout.option_id || layout.id,
    label: layout.name,
    source:
      typeof meta.variant_source === 'string' ? meta.variant_source : 'saved',
    reason:
      typeof meta.variant_reason === 'string'
        ? meta.variant_reason
        : 'Saved layout option.',
    layout_score: Number(meta.layout_score ?? 0),
    hard_valid: meta.hard_valid === undefined ? true : Boolean(meta.hard_valid),
    complete: meta.complete === undefined ? undefined : Boolean(meta.complete),
    gallery_eligible:
      meta.gallery_eligible === undefined
        ? undefined
        : Boolean(meta.gallery_eligible),
    coverage_ratio:
      meta.coverage_ratio === undefined ? undefined : Number(meta.coverage_ratio ?? 0),
    missing_cluster_ids: Array.isArray(meta.missing_cluster_ids)
      ? meta.missing_cluster_ids.map((item) => String(item))
      : undefined,
    notes: Array.isArray(meta.notes) ? meta.notes.map((item) => String(item)) : [],
    absolute_layout: absoluteLayout,
    styled_result: layout.styled_result_json ?? {},
    design: (layout.design_json as unknown as DesignPlanV1 | null) ?? null,
    saved_layout_id: layout.id,
    saved_layout_case_id: layout.case_id ?? null,
  };
}
