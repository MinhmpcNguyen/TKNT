from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

services_package = types.ModuleType("services")
services_package.__path__ = [str(REPO_ROOT / "services")]
sys.modules.setdefault("services", services_package)

import services.user_content_service as user_content_service_module
from db.models import (
    GeneratedRender,
    GeneratedRenderId,
    SavedLayout,
    SavedLayoutCase,
    SavedLayoutCaseId,
    SavedLayoutId,
    TenantId,
    UserAccount,
    UserId,
)
from pipeline.account_layout_autosave import build_saved_layout_option_drafts
from services.user_content_service import (
    LayoutOptionPersistenceRequest,
    UserContentService,
)


class FakeUserContentRepository:
    def __init__(self) -> None:
        self.users: dict[UserId, UserAccount] = {}
        self.cases: dict[SavedLayoutCaseId, SavedLayoutCase] = {}
        self.layouts: dict[SavedLayoutId, SavedLayout] = {}
        self.layout_keys: dict[tuple[UserId, SavedLayoutCaseId, str], SavedLayoutId] = {}
        self.generated_renders: dict[GeneratedRenderId, GeneratedRender] = {}

    def get_user_by_id(self, user_id: UserId) -> UserAccount | None:
        return self.users.get(user_id)

    def upsert_saved_layout_case(
        self, layout_case: SavedLayoutCase
    ) -> SavedLayoutCase:
        self.cases[layout_case.id] = layout_case
        return layout_case

    def create_saved_layout(self, layout: SavedLayout) -> None:
        self.layouts[layout.id] = layout

    def get_saved_layout(
        self, layout_id: SavedLayoutId, user_id: UserId
    ) -> SavedLayout | None:
        layout = self.layouts.get(layout_id)
        if layout is None or layout.user_id != user_id:
            return None
        case_name = (
            self.cases[layout.case_id].name
            if layout.case_id is not None and layout.case_id in self.cases
            else None
        )
        return layout.model_copy(update={"case_name": case_name})

    def upsert_saved_layout_option(self, layout: SavedLayout) -> SavedLayout:
        if layout.case_id is None or layout.option_id is None:
            raise AssertionError("Expected case_id and option_id for option upsert.")
        key = (layout.user_id, layout.case_id, layout.option_id)
        existing_id = self.layout_keys.get(key)
        if existing_id is not None:
            existing = self.layouts[existing_id]
            layout = layout.model_copy(update={"id": existing.id})
        self.layout_keys[key] = layout.id
        self.layouts[layout.id] = layout
        stored = self.get_saved_layout(layout.id, layout.user_id)
        if stored is None:
            raise AssertionError("Stored layout should be readable by owner.")
        return stored

    def update_saved_layout(self, layout: SavedLayout) -> SavedLayout | None:
        existing = self.layouts.get(layout.id)
        if existing is None or existing.user_id != layout.user_id:
            return None
        self.layouts[layout.id] = layout
        return self.get_saved_layout(layout.id, layout.user_id)

    def list_saved_layouts(
        self, user_id: UserId, case_id: SavedLayoutCaseId | None = None
    ) -> list[SavedLayout]:
        layouts = [
            layout
            for layout in self.layouts.values()
            if layout.user_id == user_id
            and (case_id is None or layout.case_id == case_id)
        ]
        return [
            self.get_saved_layout(layout.id, user_id) or layout for layout in layouts
        ]

    def create_generated_render(self, render: GeneratedRender) -> None:
        self.generated_renders[render.id] = render

    def list_generated_renders(self, user_id: UserId) -> list[GeneratedRender]:
        return [
            render
            for render in self.generated_renders.values()
            if render.user_id == user_id
        ]

    def get_generated_render(
        self, render_id: GeneratedRenderId, user_id: UserId
    ) -> GeneratedRender | None:
        render = self.generated_renders.get(render_id)
        if render is None or render.user_id != user_id:
            return None
        return render

    def update_generated_render_meta(
        self,
        render_id: GeneratedRenderId,
        user_id: UserId,
        meta: dict[str, object],
    ) -> GeneratedRender | None:
        render = self.get_generated_render(render_id, user_id)
        if render is None:
            return None
        updated = render.model_copy(update={"meta": meta})
        self.generated_renders[render_id] = updated
        return updated


class UserContentLayoutCasesTest(unittest.TestCase):
    def setUp(self) -> None:
        self._original_schema_init = user_content_service_module.ensure_user_content_schema
        user_content_service_module.ensure_user_content_schema = lambda: None
        self.repository = FakeUserContentRepository()
        self.user = UserAccount(
            id=UserId("user_case_owner"),
            email="owner@example.com",
            display_name="Owner",
            tenant_id=TenantId("tenant_owner"),
        )
        self.other_user = UserAccount(
            id=UserId("user_other"),
            email="other@example.com",
            display_name="Other",
            tenant_id=TenantId("tenant_other"),
        )
        self.repository.users[self.user.id] = self.user
        self.repository.users[self.other_user.id] = self.other_user
        self.service = UserContentService(user_repository=self.repository)

    def tearDown(self) -> None:
        user_content_service_module.ensure_user_content_schema = (
            self._original_schema_init
        )

    def test_upserts_case_options_without_duplicates(self) -> None:
        first_saved = self.service.upsert_layout_case_options(
            user=self.user,
            case_id="case_a",
            case_name="Case A",
            options=[
                _option("variant_1", 1, "Option 1"),
                _option("variant_2", 2, "Option 2"),
            ],
            meta={"source": "test"},
        )
        second_saved = self.service.upsert_layout_case_options(
            user=self.user,
            case_id="case_a",
            case_name="Case A renamed",
            options=[_option("variant_1", 1, "Option 1 updated")],
            meta={"source": "test"},
        )

        self.assertEqual(len(first_saved), 2)
        self.assertEqual(len(second_saved), 1)
        self.assertEqual(len(self.repository.layouts), 2)
        self.assertEqual(second_saved[0].id, first_saved[0].id)
        self.assertEqual(second_saved[0].name, "Option 1 updated")
        self.assertEqual(second_saved[0].case_name, "Case A renamed")

    def test_update_layout_only_allows_owner(self) -> None:
        saved = self.service.upsert_layout_case_options(
            user=self.user,
            case_id="case_patch",
            case_name="Patch Case",
            options=[_option("variant_1", 1, "Original")],
            meta={},
        )[0]

        blocked = self.service.update_layout(
            user=self.other_user,
            layout_id=str(saved.id),
            name="Other edit",
            floorplan_json=_floorplan(),
            design_json=_design(),
            styled_result_json={"objects": []},
            meta={},
        )
        updated = self.service.update_layout(
            user=self.user,
            layout_id=str(saved.id),
            name="Owner edit",
            floorplan_json=_floorplan(),
            design_json=_design(),
            styled_result_json={"objects": []},
            meta={"updated": True},
        )

        self.assertIsNone(blocked)
        self.assertIsNotNone(updated)
        self.assertEqual(updated.name if updated is not None else "", "Owner edit")
        self.assertEqual(updated.case_id, saved.case_id if updated is not None else None)

    def test_save_snapshot_render_links_case_option_context(self) -> None:
        saved = self.service.upsert_layout_case_options(
            user=self.user,
            case_id="case_render",
            case_name="Render Case",
            options=[_option("variant_1", 1, "Option 1")],
            meta={},
        )[0]
        context = self.service.resolve_render_layout_context(
            user=self.user,
            case_id="case_render",
            option_id="variant_1",
        )
        self.assertIsNotNone(context)
        if context is None:
            return

        render = self.service.save_snapshot_render_from_data_url(
            user=self.user,
            image_data_url=_png_data_url(),
            model_name="gpt-image-1.5",
            prompt="",
            root_layout_id=context.root_layout_id,
            case_id=context.case_id,
            option_id=context.option_id,
            option_index=context.option_index,
            meta={"render_mode": "generate"},
        )

        self.assertEqual(render.root_layout_id, saved.id)
        self.assertEqual(render.case_id, saved.case_id)
        self.assertEqual(render.option_id, "variant_1")
        self.assertEqual(render.option_index, 1)
        self.assertEqual(render.meta["root_layout_id"], str(saved.id))
        self.assertEqual(render.meta["case_id"], "case_render")
        self.assertEqual(render.meta["option_id"], "variant_1")

    def test_resolve_snapshot_render_context_from_root_layout_id(self) -> None:
        saved = self.service.upsert_layout_case_options(
            user=self.user,
            case_id="case_root",
            case_name="Root Case",
            options=[_option("variant_2", 2, "Option 2")],
            meta={},
        )[0]

        context = self.service.resolve_render_layout_context(
            user=self.user,
            root_layout_id=str(saved.id),
        )

        self.assertIsNotNone(context)
        self.assertEqual(context.root_layout_id if context else None, str(saved.id))
        self.assertEqual(context.case_id if context else None, "case_root")
        self.assertEqual(context.option_id if context else None, "variant_2")
        self.assertEqual(context.option_index if context else None, 2)

    def test_resolve_render_context_rejects_unowned_case_option(self) -> None:
        self.service.upsert_layout_case_options(
            user=self.user,
            case_id="case_private",
            case_name="Private Case",
            options=[_option("variant_1", 1, "Option 1")],
            meta={},
        )

        context = self.service.resolve_render_layout_context(
            user=self.other_user,
            case_id="case_private",
            option_id="variant_1",
        )

        self.assertIsNone(context)

    def test_fixture_layout_variants_build_saved_option_drafts(self) -> None:
        case_dir = REPO_ROOT / "cases/user_f1a4e69d3672aab8_20260515T032329Z"
        drafts = build_saved_layout_option_drafts(
            case_id="fixture_case",
            case_name="Fixture Case",
            room_payload=_load_json(case_dir / "01_room_interpreter.json"),
            stylist_payload=_load_json(case_dir / "08_stylist.json"),
            absolute_layout_payload=_load_json(case_dir / "07_absolute_layout.json"),
            layout_variants_payload=_load_json(case_dir / "09_layout_variants.json"),
        )

        self.assertGreaterEqual(len(drafts), 1)
        first = drafts[0]
        self.assertEqual(first.option_id, "variant_1")
        self.assertIn("room", first.floorplan_json)
        self.assertIsNotNone(first.design_json)
        self.assertGreater(len(first.design_json["placements"]), 0)
        self.assertIn("absolute_layout", first.meta)


def _option(
    option_id: str, option_index: int, name: str
) -> LayoutOptionPersistenceRequest:
    return LayoutOptionPersistenceRequest(
        option_id=option_id,
        option_index=option_index,
        name=name,
        floorplan_json=_floorplan(),
        design_json=_design(),
        styled_result_json={"objects": []},
        meta={"label": name},
    )


def _floorplan() -> dict[str, object]:
    return {
        "room": {
            "room_id": "room_1",
            "name": "Room",
            "polygon_mm": [
                {"x": 0, "y": 0},
                {"x": 1000, "y": 0},
                {"x": 1000, "y": 1000},
                {"x": 0, "y": 1000},
            ],
        },
        "walls": [],
        "openings": {"doors": [], "windows": []},
    }


def _design() -> dict[str, object]:
    return {
        "floorplan_geometry": _floorplan(),
        "placements": [],
    }


def _png_data_url() -> str:
    return "data:image/png;base64,iVBORw0KGgo="


def _load_json(path: Path) -> object:
    return json.loads(path.read_text())


if __name__ == "__main__":
    unittest.main()
