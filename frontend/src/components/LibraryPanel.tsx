import React, { useMemo, useState } from 'react';
import { useEditorStore } from '../store';
import { InventoryItem } from '../types';

export const LibraryPanel: React.FC = () => {
  const inventory = useEditorStore((s) => s.inventory);
  const roomType = useEditorStore((s) => s.userInput.room_type);
  const [searchQuery, setSearchQuery] = useState('');
  const [styleFilter, setStyleFilter] = useState('');
  const [roomTypeFilter, setRoomTypeFilter] = useState('all');

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
    const normalizedRoomType =
      roomTypeFilter === 'current' ? roomType : roomTypeFilter;

    return inventory.filter((item) => {
      const haystack = [
        item.name,
        item.category,
        ...(item.style_tags ?? []),
      ]
        .join(' ')
        .toLowerCase();
      if (query && !haystack.includes(query)) {
        return false;
      }
      if (
        normalizedStyle &&
        !(item.style_tags ?? []).some((tag) =>
          String(tag).toLowerCase().includes(normalizedStyle),
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

  return (
    <aside className="panel library-panel">
      <h3>Library</h3>
      <div className="library-filters">
        <input
          type="search"
          value={searchQuery}
          placeholder="Search by name or type"
          onChange={(evt) => setSearchQuery(evt.target.value)}
        />
        <select
          value={styleFilter}
          onChange={(evt) => setStyleFilter(evt.target.value)}
        >
          <option value="">All styles</option>
          {styleOptions.map((style) => (
            <option key={style} value={style}>
              {style}
            </option>
          ))}
        </select>
        <select
          value={roomTypeFilter}
          onChange={(evt) => setRoomTypeFilter(evt.target.value)}
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
          <p className="muted">No inventory loaded. Check API /inventory/items or tenant_id.</p>
        ) : visibleInventory.length === 0 ? (
          <p className="muted">No inventory matches the current search filters.</p>
        ) : (
          visibleInventory.map((item) => <LibraryItem key={item.asset_id} item={item} />)
        )}
      </div>
    </aside>
  );
};

const LibraryItem: React.FC<{ item: InventoryItem }> = ({ item }) => {
  return (
    <div className="library-item" draggable onDragStart={(evt) => {
      const serialized = JSON.stringify(item);
      evt.dataTransfer.effectAllowed = 'copy';
      evt.dataTransfer.setData('application/x-asset', serialized);
      evt.dataTransfer.setData('text/plain', serialized);
      evt.dataTransfer.setData('text/x-asset-id', item.asset_id);
    }}>
      <div className="library-swatch" style={{ background: item.color_hex ?? '#d6d2c8' }} />
      <div>
        <div className="library-name">{item.name}</div>
        <div className="library-meta">{item.category}</div>
      </div>
    </div>
  );
};

function matchesRoomType(item: InventoryItem, roomType: string): boolean {
  const normalizedRoomType = roomType.trim().toLowerCase();
  const category = item.category.trim().toLowerCase();
  const explicitRoomType = String(item.room_type ?? '').trim().toLowerCase();
  const attributesRoomType = String(
    item.attributes?.room_type ?? '',
  )
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
    bedroom: [
      'bed',
      'nightstand',
      'wardrobe',
      'dresser',
      'mattress',
      'laundry_basket',
      'bedside',
    ],
    living_room: [
      'sofa',
      'coffee_table',
      'tv_console',
      'console_table',
      'recliner',
      'armchair',
      'side_table',
    ],
    studio: [
      'bed',
      'nightstand',
      'sofa',
      'coffee_table',
      'desk',
      'office_chair',
      'tv_console',
      'wardrobe',
      'dresser',
    ],
    office: [
      'desk',
      'office_chair',
      'monitor',
      'laptop',
      'desk_lamp',
      'office_pedestal',
      'printer',
    ],
    kitchen: [
      'kitchen_',
      'cabinet',
      'island',
      'fridge',
      'microwave',
      'coffee_machine',
      'kettle',
      'dining_table',
      'dining_chair',
      'bar_stool',
    ],
  };

  return (roomKeywords[normalizedRoomType] ?? []).some((keyword) =>
    category.includes(keyword),
  );
}
