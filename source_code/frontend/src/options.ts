export const ROOM_TYPE_OPTIONS = [
  { value: 'living_room', label: 'Living Room' },
  { value: 'bedroom', label: 'Bedroom' },
] as const;

export const STYLE_OPTIONS = [
  { value: 'modern', label: 'Modern' },
  { value: 'japandi', label: 'Japandi' },
  { value: 'scandinavian', label: 'Scandinavian' },
  { value: 'boho', label: 'Boho' },
  { value: 'minimal', label: 'Minimal' },
] as const;

type RoomTypeOption = (typeof ROOM_TYPE_OPTIONS)[number]['value'];
type StyleOption = (typeof STYLE_OPTIONS)[number]['value'];

export const DEFAULT_ROOM_TYPE: RoomTypeOption = 'living_room';
const DEFAULT_STYLE: StyleOption = 'modern';

export function normalizeRoomType(value: string | null | undefined): RoomTypeOption {
  const key = String(value ?? '')
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/g, '_');
  if (key === 'living_room' || key === 'livingroom') {
    return 'living_room';
  }
  if (key === 'bedroom') {
    return 'bedroom';
  }
  return DEFAULT_ROOM_TYPE;
}

export function normalizeStyle(value: string | null | undefined): StyleOption {
  return readStyleOption(value) ?? DEFAULT_STYLE;
}

export function matchesStyleOption(
  value: string | null | undefined,
  option: StyleOption,
): boolean {
  return readStyleOption(value) === option;
}

function readStyleOption(value: string | null | undefined): StyleOption | null {
  const key = String(value ?? '').trim().toLowerCase();
  if (key.includes('japandi')) {
    return 'japandi';
  }
  if (key.includes('scandinavian') || key.includes('scandi')) {
    return 'scandinavian';
  }
  if (key.includes('boho')) {
    return 'boho';
  }
  if (key.includes('minimal')) {
    return 'minimal';
  }
  if (key.includes('modern')) {
    return 'modern';
  }
  return null;
}
