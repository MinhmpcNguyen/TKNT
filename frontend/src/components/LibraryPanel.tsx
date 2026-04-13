import React, { useEffect, useMemo, useState } from 'react';
import {
  createObject3DFromImage,
  createObject3DFromPrompt,
  fetchInventory,
  getObject3DJobStatus,
  listGeneratedRenders,
} from '../api';
import { useEditorStore } from '../store';
import { InventoryItem } from '../types';

type CustomObjectMode = 'prompt' | 'image';

export const LibraryPanel: React.FC = () => {
  const inventory = useEditorStore((state) => state.inventory);
  const roomType = useEditorStore((state) => state.userInput.room_type);
  const tenantId = useEditorStore((state) => state.tenantId);
  const setInventory = useEditorStore((state) => state.setInventory);
  const setSavedRenders = useEditorStore((state) => state.setSavedRenders);
  const appendRunLog = useEditorStore((state) => state.appendRunLog);
  const [searchQuery, setSearchQuery] = useState('');
  const [styleFilter, setStyleFilter] = useState('');
  const [roomTypeFilter, setRoomTypeFilter] = useState('all');
  const [mode, setMode] = useState<CustomObjectMode>('prompt');
  const [objectName, setObjectName] = useState('');
  const [category, setCategory] = useState('custom_object');
  const [widthMm, setWidthMm] = useState('600');
  const [depthMm, setDepthMm] = useState('600');
  const [heightMm, setHeightMm] = useState('900');
  const [prompt, setPrompt] = useState('');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [creatorError, setCreatorError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [activeJobStatus, setActiveJobStatus] = useState<Awaited<
    ReturnType<typeof getObject3DJobStatus>
  > | null>(null);

  const styleOptions = useMemo(() => {
    const tags = new Set<string>();
    for (const item of inventory) {
      for (const tag of item.style_tags ?? []) {
        const normalized = String(tag).trim();
        if (normalized) {
          tags.add(normalized);
        }
      }
    }
    return Array.from(tags).sort((left, right) => left.localeCompare(right));
  }, [inventory]);

  const visibleInventory = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();
    const normalizedStyle = styleFilter.trim().toLowerCase();
    const normalizedRoomType = roomTypeFilter === 'current' ? roomType : roomTypeFilter;

    return inventory.filter((item) => {
      const haystack = [item.name, item.category, ...(item.style_tags ?? [])]
        .join(' ')
        .toLowerCase();
      if (query && !haystack.includes(query)) {
        return false;
      }
      if (
        normalizedStyle &&
        !(item.style_tags ?? []).some((tag) =>
          String(tag).toLowerCase().includes(normalizedStyle)
        )
      ) {
        return false;
      }
      if (
        normalizedRoomType &&
        normalizedRoomType !== 'all' &&
        !matchesRoomType(item, normalizedRoomType)
      ) {
        return false;
      }
      return true;
    });
  }, [inventory, roomType, roomTypeFilter, searchQuery, styleFilter]);

  const ownedInventory = visibleInventory.filter(
    (item) => item.attributes?.ownership_scope === 'owned'
  );
  const sharedInventory = visibleInventory.filter(
    (item) => item.attributes?.ownership_scope !== 'owned'
  );

  useEffect(() => {
    if (!activeJobId) {
      return undefined;
    }
    let cancelled = false;
    const poll = async () => {
      try {
        const status = await getObject3DJobStatus(activeJobId);
        if (cancelled) {
          return;
        }
        setActiveJobStatus(status);
        if (status.stage === 'done') {
          const [nextInventory, nextRenders] = await Promise.all([
            fetchInventory(tenantId),
            listGeneratedRenders(),
          ]);
          if (cancelled) {
            return;
          }
          setInventory(nextInventory);
          setSavedRenders(nextRenders);
          appendRunLog(
            `Custom object "${status.result?.persisted_asset_name ?? objectName}" is now available in your library.`
          );
          setCreating(false);
          return;
        }
        if (status.stage === 'error') {
          setCreatorError(status.error ?? '3D generation failed.');
          setCreating(false);
          return;
        }
        window.setTimeout(poll, 2000);
      } catch (pollError) {
        if (!cancelled) {
          setCreatorError(
            pollError instanceof Error ? pollError.message : 'Failed to track 3D job.'
          );
          setCreating(false);
        }
      }
    };
    void poll();
    return () => {
      cancelled = true;
    };
  }, [
    activeJobId,
    appendRunLog,
    objectName,
    setInventory,
    setSavedRenders,
    tenantId,
  ]);

  const handleCreateObject = async () => {
    setCreatorError(null);
    const normalizedName = objectName.trim();
    const normalizedCategory = category.trim();
    const width = Number(widthMm);
    const depth = Number(depthMm);
    const height = Number(heightMm);
    if (!normalizedName || !normalizedCategory) {
      setCreatorError('Name and category are required.');
      return;
    }
    if (!Number.isFinite(width) || !Number.isFinite(depth) || !Number.isFinite(height)) {
      setCreatorError('Please enter valid numeric dimensions.');
      return;
    }
    if (width <= 0 || depth <= 0 || height <= 0) {
      setCreatorError('Dimensions must be positive.');
      return;
    }
    if (mode === 'prompt' && !prompt.trim()) {
      setCreatorError('Please describe the object you want to generate.');
      return;
    }
    if (mode === 'image' && !imageFile) {
      setCreatorError('Please upload an image for TripoSR preprocessing.');
      return;
    }

    setCreating(true);
    setActiveJobStatus(null);
    try {
      const response =
        mode === 'prompt'
          ? await createObject3DFromPrompt({
              name: normalizedName,
              category: normalizedCategory,
              width_mm: width,
              depth_mm: depth,
              height_mm: height,
              prompt: prompt.trim(),
            })
          : await createObject3DFromImage({
              file: imageFile as File,
              name: normalizedName,
              category: normalizedCategory,
              width_mm: width,
              depth_mm: depth,
              height_mm: height,
            });
      setActiveJobId(response.job_id);
      appendRunLog(`Started custom 3D object job: ${response.job_id}`);
    } catch (creationError) {
      setCreating(false);
      setCreatorError(
        creationError instanceof Error ? creationError.message : 'Failed to create object.'
      );
    }
  };

  return (
    <aside className="panel library-panel">
      <h3>Library</h3>
      <div className="custom-object-card">
        <div className="custom-object-header">
          <strong>Create Custom 3D Object</strong>
          <span>Saved into your private library</span>
        </div>
        <div className="custom-mode-toggle">
          <button
            className={mode === 'prompt' ? 'active' : ''}
            onClick={() => setMode('prompt')}
            type="button"
          >
            Prompt
          </button>
          <button
            className={mode === 'image' ? 'active' : ''}
            onClick={() => setMode('image')}
            type="button"
          >
            Upload Image
          </button>
        </div>
        <div className="custom-object-form">
          <input
            type="text"
            value={objectName}
            onChange={(event) => setObjectName(event.target.value)}
            placeholder="Object name"
          />
          <input
            type="text"
            value={category}
            onChange={(event) => setCategory(event.target.value)}
            placeholder="Category"
          />
          <div className="custom-dimension-grid">
            <input
              type="number"
              min="1"
              value={widthMm}
              onChange={(event) => setWidthMm(event.target.value)}
              placeholder="Width (mm)"
            />
            <input
              type="number"
              min="1"
              value={depthMm}
              onChange={(event) => setDepthMm(event.target.value)}
              placeholder="Depth (mm)"
            />
            <input
              type="number"
              min="1"
              value={heightMm}
              onChange={(event) => setHeightMm(event.target.value)}
              placeholder="Height (mm)"
            />
          </div>
          {mode === 'prompt' ? (
            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              placeholder="A white ceramic floor vase with soft rounded shoulders on a plain white background"
              rows={5}
            />
          ) : (
            <label className="file-input">
              <span>{imageFile ? imageFile.name : 'Choose an object image'}</span>
              <input
                type="file"
                accept="image/png,image/jpeg,image/webp"
                onChange={(event) => setImageFile(event.target.files?.[0] ?? null)}
              />
            </label>
          )}
          {creatorError ? <div className="inline-error">{creatorError}</div> : null}
          {activeJobStatus ? (
            <div className="job-status-card">
              <strong>{activeJobStatus.stage}</strong>
              <span>{activeJobStatus.message || 'Working on your object...'}</span>
              {activeJobStatus.artifact_urls.preprocessed_image ? (
                <img
                  src={activeJobStatus.artifact_urls.preprocessed_image}
                  alt="Preprocessed object reference"
                />
              ) : null}
            </div>
          ) : null}
          <button
            className="primary-button"
            disabled={creating}
            onClick={handleCreateObject}
            type="button"
          >
            {creating ? 'Generating...' : 'Create object'}
          </button>
        </div>
      </div>

      <div className="library-filters">
        <input
          type="search"
          value={searchQuery}
          placeholder="Search by name or type"
          onChange={(event) => setSearchQuery(event.target.value)}
        />
        <select value={styleFilter} onChange={(event) => setStyleFilter(event.target.value)}>
          <option value="">All styles</option>
          {styleOptions.map((style) => (
            <option key={style} value={style}>
              {style}
            </option>
          ))}
        </select>
        <select
          value={roomTypeFilter}
          onChange={(event) => setRoomTypeFilter(event.target.value)}
        >
          <option value="all">All rooms</option>
          <option value="current">Current room</option>
          <option value="bedroom">Bedroom</option>
          <option value="living_room">Living Room</option>
          <option value="studio">Studio</option>
          <option value="office">Office</option>
          <option value="kitchen">Kitchen</option>
        </select>
      </div>

      <div className="library-list">
        {inventory.length === 0 ? (
          <p className="muted">No inventory loaded yet.</p>
        ) : visibleInventory.length === 0 ? (
          <p className="muted">No inventory matches the current filters.</p>
        ) : (
          <>
            <LibrarySection label={`Your objects (${ownedInventory.length})`} items={ownedInventory} />
            <LibrarySection label={`Shared inventory (${sharedInventory.length})`} items={sharedInventory} />
          </>
        )}
      </div>
    </aside>
  );
};

const LibrarySection: React.FC<{ label: string; items: InventoryItem[] }> = ({
  label,
  items,
}) => {
  if (items.length === 0) {
    return null;
  }
  return (
    <div className="library-section">
      <div className="library-section-title">{label}</div>
      {items.map((item) => (
        <LibraryItem key={item.asset_id} item={item} />
      ))}
    </div>
  );
};

const LibraryItem: React.FC<{ item: InventoryItem }> = ({ item }) => {
  const ownershipScope =
    item.attributes?.ownership_scope === 'owned' ? 'Your asset' : 'Shared asset';
  return (
    <div
      className="library-item"
      draggable
      onDragStart={(event) => {
        const serialized = JSON.stringify(item);
        event.dataTransfer.effectAllowed = 'copy';
        event.dataTransfer.setData('application/x-asset', serialized);
        event.dataTransfer.setData('text/plain', serialized);
        event.dataTransfer.setData('text/x-asset-id', item.asset_id);
      }}
    >
      <div className="library-swatch" style={{ background: item.color_hex ?? '#d6d2c8' }} />
      <div className="library-item-copy">
        <div className="library-name">{item.name}</div>
        <div className="library-meta">
          {item.category} · {ownershipScope}
        </div>
      </div>
    </div>
  );
};

function matchesRoomType(item: InventoryItem, roomType: string): boolean {
  const normalizedRoomType = roomType.trim().toLowerCase();
  const category = item.category.trim().toLowerCase();
  const explicitRoomType = String(item.room_type ?? '').trim().toLowerCase();
  const attributesRoomType = String(item.attributes?.room_type ?? '')
    .trim()
    .toLowerCase();
  if (
    explicitRoomType === normalizedRoomType ||
    attributesRoomType === normalizedRoomType
  ) {
    return true;
  }

  const genericKeywords = [
    'door',
    'window',
    'rug',
    'curtain',
    'blind',
    'plant',
    'wall_art',
    'mirror',
    'clock',
    'ceiling_light',
    'pendant_light',
    'track_light',
    'floor_lamp',
    'table_lamp',
    'storage_cabinet',
    'chair',
    'bench',
    'bookshelf',
  ];
  if (genericKeywords.some((keyword) => category.includes(keyword))) {
    return true;
  }

  const roomKeywords: Record<string, string[]> = {
    bedroom: ['bed', 'nightstand', 'wardrobe', 'dresser', 'mattress', 'laundry_basket', 'bedside'],
    living_room: ['sofa', 'coffee_table', 'tv_console', 'console_table', 'recliner', 'armchair', 'side_table'],
    studio: ['bed', 'nightstand', 'sofa', 'coffee_table', 'desk', 'office_chair', 'tv_console', 'wardrobe', 'dresser'],
    office: ['desk', 'office_chair', 'monitor', 'laptop', 'desk_lamp', 'office_pedestal', 'printer'],
    kitchen: ['kitchen_', 'cabinet', 'island', 'fridge', 'microwave', 'coffee_machine', 'kettle', 'dining_table', 'dining_chair', 'bar_stool'],
  };

  return (roomKeywords[normalizedRoomType] ?? []).some((keyword) =>
    category.includes(keyword)
  );
}
