from __future__ import annotations

from layout.room_profiles.base import RoomProfile

BEDROOM_LEGACY_ALIASES: dict[str, tuple[str, ...]] = {
    "bed": ("bed", "queen bed", "king bed", "single bed", "double bed"),
    "nightstand": ("nightstand", "night stand", "bedside table"),
    "wardrobe": ("wardrobe", "closet", "armoire"),
    "desk": ("desk", "work desk", "study desk", "vanity"),
    "chair": ("chair", "desk chair", "office chair", "vanity chair"),
    "bookshelf": ("bookshelf", "book shelf", "bookcase", "shelf"),
    "rug": ("rug", "carpet"),
    "ceiling_lamp": ("ceiling lamp", "ceiling light", "overhead light"),
    "floor_lamp": ("floor lamp", "standing lamp", "reading lamp"),
    "side_table": ("side table", "end table"),
}

LIVING_ROOM_LEGACY_ALIASES: dict[str, tuple[str, ...]] = {
    "armchair": ("armchair", "arm chair", "lounge chair", "reading chair"),
    "bookshelf": ("bookshelf", "book shelf", "bookcase", "shelf"),
    "ceiling_lamp": ("ceiling lamp", "ceiling light", "overhead light"),
    "chair": ("chair", "desk chair", "office chair", "vanity chair"),
    "coffee_table": ("coffee table", "tea table", "center table"),
    "floor_lamp": ("floor lamp", "standing lamp", "reading lamp"),
    "rug": ("rug", "carpet"),
    "side_table": ("side table", "end table"),
    "sofa": ("sofa", "couch", "loveseat", "sectional"),
    "tv_console": ("tv console", "media console", "tv stand"),
}

BEDROOM_LEGACY_PROFILE = RoomProfile(
    profile_id="bedroom_legacy",
    room_types=frozenset({"bedroom", "guest_bedroom", "primary_bedroom"}),
    object_aliases=BEDROOM_LEGACY_ALIASES,
)

LIVING_ROOM_LEGACY_PROFILE = RoomProfile(
    profile_id="living_room_legacy",
    room_types=frozenset({"living_room", "living", "lounge", "family_room"}),
    object_aliases=LIVING_ROOM_LEGACY_ALIASES,
)
