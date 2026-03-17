import React from 'react';
import { useEditorStore } from '../store';
import { InventoryItem } from '../types';

export const LibraryPanel: React.FC = () => {
  const inventory = useEditorStore((s) => s.inventory);

  return (
    <aside className="panel library-panel">
      <h3>Library</h3>
      <div className="library-list">
        {inventory.length === 0 ? (
          <p className="muted">No inventory loaded. Check API /inventory/items or tenant_id.</p>
        ) : (
          inventory.map((item) => <LibraryItem key={item.asset_id} item={item} />)
        )}
      </div>
    </aside>
  );
};

const LibraryItem: React.FC<{ item: InventoryItem }> = ({ item }) => {
  return (
    <div className="library-item" draggable onDragStart={(evt) => {
      evt.dataTransfer.setData('application/x-asset', JSON.stringify(item));
    }}>
      <div className="library-swatch" style={{ background: item.color_hex ?? '#d6d2c8' }} />
      <div>
        <div className="library-name">{item.name}</div>
        <div className="library-meta">{item.category}</div>
      </div>
    </div>
  );
};
