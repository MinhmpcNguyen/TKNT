from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from agent_schema.cluster_placer_schema import MacroClusterPlacerOutput
from clients.openai_client import OpenAIClient
from cluster_placer.tools import TOOL_REGISTRY, TOOL_SCHEMAS
from prompt.cluster_placer import MACRO_CLUSTER_PLACER_PROMPT
from prompt.system import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MacroClusterPlacer:
    system_prompt: str = SYSTEM_PROMPT
    prompt_template: str = MACRO_CLUSTER_PLACER_PROMPT

    def generate_raw(
        self,
        *,
        room_model_json: dict[str, Any],
        clusters_outlines_json: dict[str, Any],
        relation_plan_json: dict[str, Any] | None = None,
        existing_transforms_json: dict[str, Any] | None = None,
        verifier_feedback_json: dict[str, Any] | None = None,
        grid_mm: int = 50,
    ) -> str:
        user_prompt = _build_prompt(
            template=self.prompt_template,
            room_model_json=room_model_json,
            clusters_outlines_json=clusters_outlines_json,
            relation_plan_json=relation_plan_json,
            existing_transforms_json=existing_transforms_json,
            verifier_feedback_json=verifier_feedback_json,
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
            {
                "role": "user",
                "content": f"Use grid_mm={grid_mm} for all placements and output.",
            },
        ]
        return _run_with_tools(
            messages=messages,
            room_model=room_model_json,
            clusters_outlines=clusters_outlines_json,
            relation_plan_json=relation_plan_json,
            grid_mm=grid_mm,
            existing_transforms_json=existing_transforms_json,
        )

    def generate(
        self,
        *,
        room_model_json: dict[str, Any],
        clusters_outlines_json: dict[str, Any],
        relation_plan_json: dict[str, Any] | None = None,
        existing_transforms_json: dict[str, Any] | None = None,
        verifier_feedback_json: dict[str, Any] | None = None,
        grid_mm: int = 50,
    ) -> dict[str, Any]:
        raw = self.generate_raw(
            room_model_json=room_model_json,
            clusters_outlines_json=clusters_outlines_json,
            relation_plan_json=relation_plan_json,
            existing_transforms_json=existing_transforms_json,
            verifier_feedback_json=verifier_feedback_json,
            grid_mm=grid_mm,
        )
        payload = _parse_json(raw)
        return MacroClusterPlacerOutput.model_validate(payload).model_dump()


def _build_prompt(
    *,
    template: str,
    room_model_json: dict[str, Any],
    clusters_outlines_json: dict[str, Any],
    relation_plan_json: dict[str, Any] | None,
    existing_transforms_json: dict[str, Any] | None,
    verifier_feedback_json: dict[str, Any] | None,
) -> str:
    mapping = {
        "ROOM_MODEL_JSON": _json_block(room_model_json),
        "CLUSTERS_OUTLINES_JSON": _json_block(clusters_outlines_json),
        "RELATION_PLAN_JSON": _json_block(relation_plan_json),
        "EXISTING_TRANSFORMS_JSON": _json_block(existing_transforms_json),
        "VERIFIER_FEEDBACK_JSON": _json_block(verifier_feedback_json),
    }
    output = template
    for key, value in mapping.items():
        output = output.replace("{" + key + "}", value)
    return output


def _json_block(obj: Any) -> str:
    if obj is None:
        return "null"
    return json.dumps(obj, ensure_ascii=True, indent=2)


def _run_with_tools(
    *,
    messages: list[dict[str, Any]],
    room_model: dict[str, Any],
    clusters_outlines: dict[str, Any],
    relation_plan_json: dict[str, Any] | None,
    grid_mm: int,
    existing_transforms_json: dict[str, Any] | None = None,
    max_steps: int = 30,
    soft_improve_budget: int = 30,
    min_explore_valid_rounds: int = 2,
    history_keep_last_blocks: int = 12,
) -> str:
    client = OpenAIClient()

    verifier_called = False
    forced_check = False
    last_verifier_result: str | None = None
    last_draft_transforms: list[dict[str, Any]] | None = None

    expected_cluster_ids = _cluster_ids(clusters_outlines)

    best_valid_score: int | None = None
    best_valid_quality: dict[str, Any] | None = None
    best_valid_transforms: list[dict[str, Any]] | None = None

    improve_rounds_used = 0
    no_improve_streak = 0

    verified_states: dict[str, dict[str, Any]] = {}
    tried_patch_keys_by_state: dict[str, set[str]] = {}
    invalid_state_trace: list[dict[str, Any]] = []

    for step in range(1, max_steps + 1):
        messages = _prune_messages_block_safe(
            messages,
            prefix_count=3,
            keep_last_blocks=history_keep_last_blocks,
        )
        logger.info("MacroClusterPlacer step %s/%s", step, max_steps)

        try:
            response = client.chat_completion(
                messages,
                model_key="gpt_4o",
                temperature=0.0,
                tools=TOOL_SCHEMAS,
            )
        except Exception as exc:
            if _is_context_length_exceeded(exc):
                logger.warning(
                    "MacroClusterPlacer stopped early due to context length exceeded.",
                    exc_info=exc,
                )
                if best_valid_transforms is not None:
                    return _build_best_valid_json(
                        grid_mm=grid_mm,
                        transforms=best_valid_transforms,
                        quality=best_valid_quality,
                    )
                return json.dumps(
                    {
                        "status": "UNSAT",
                        "grid_mm": int(grid_mm),
                        "cluster_transforms": [],
                        "notes": [
                            "Stopped due to context length exceeded before finding a valid layout."
                        ],
                    },
                    ensure_ascii=True,
                )
            raise

        message = _extract_message(response)
        tool_calls = _extract_tool_calls(message)
        assistant_content = getattr(message, "content", "") or ""

        draft = _try_parse_json_object(assistant_content)
        if isinstance(draft, dict) and isinstance(
            draft.get("cluster_transforms"), list
        ):
            last_draft_transforms = _canonicalize_cluster_transforms(
                draft.get("cluster_transforms")
            )

        if tool_calls:
            logger.info(
                "Tool calls requested: %s",
                [(c.get("function", {}) or {}).get("name") for c in tool_calls],
            )

            assistant_history_msg = {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
            }
            messages.append(assistant_history_msg)

            prepared_calls: list[dict[str, Any]] = []
            preflight_error: str | None = None

            for idx, call in enumerate(tool_calls):
                call_id = call.get("id") or f"tool_{idx}"
                fn = call.get("function", {}) or {}
                name = fn.get("name")
                args_text = fn.get("arguments") or "{}"
                args = _safe_json_loads(args_text)

                if not isinstance(name, str) or name not in TOOL_REGISTRY:
                    prepared_calls.append(
                        {
                            "call_id": call_id,
                            "name": name,
                            "args": args,
                            "kind": "unknown_tool",
                            "state_key": None,
                            "canonical_transforms": None,
                        }
                    )
                    continue

                state_key = None
                canonical_transforms: list[dict[str, Any]] | None = None

                if name == "GlobalClusterVerifier":
                    args = _coerce_verifier_args(
                        args=args,
                        room_model=room_model,
                        clusters_outlines=clusters_outlines,
                        grid_mm=grid_mm,
                        assistant_content=assistant_content,
                        existing_transforms_json=existing_transforms_json,
                        last_draft_transforms=last_draft_transforms,
                        relation_plan_json=relation_plan_json,
                    )

                    canonical_transforms = (
                        _canonicalize_cluster_transforms(args["cluster_transforms"])
                        if isinstance(args.get("cluster_transforms"), list)
                        else []
                    )
                    args["cluster_transforms"] = canonical_transforms

                    ok, detail = _validate_transforms_full(
                        canonical_transforms,
                        expected_cluster_ids,
                    )
                    if not ok:
                        preflight_error = (
                            "Before calling GlobalClusterVerifier you MUST provide cluster_transforms for EVERY cluster_id "
                            f"exactly once. Required IDs: {expected_cluster_ids}. Issue: {detail}"
                        )
                        break

                    state_key = _state_key(canonical_transforms)
                    prior = verified_states.get(state_key)
                    if prior is not None:
                        prior_result = prior.get("result")
                        if prior_result == "VALID":
                            preflight_error = (
                                "This exact cluster_transforms state was already verified VALID. "
                                "Do NOT call GlobalClusterVerifier again. Output final JSON now using that verified state."
                            )
                            break

                        errs_preview = prior.get("errors_preview") or []
                        preflight_error = (
                            "This exact cluster_transforms state was already verified INVALID. "
                            "Do NOT repeat it. Change at least one cluster and re-check. "
                            f"Previous errors (preview): {json.dumps(errs_preview, ensure_ascii=True)}"
                        )
                        break

                prepared_calls.append(
                    {
                        "call_id": call_id,
                        "name": name,
                        "args": args,
                        "kind": "normal",
                        "state_key": state_key,
                        "canonical_transforms": canonical_transforms,
                    }
                )

            if preflight_error is not None:
                for idx, call in enumerate(tool_calls):
                    call_id = call.get("id") or f"tool_{idx}"
                    fn = call.get("function", {}) or {}
                    name = fn.get("name")
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": json.dumps(
                                {
                                    "error": "preflight_failed",
                                    "tool": name,
                                    "message": preflight_error,
                                },
                                ensure_ascii=True,
                            ),
                        }
                    )

                messages.append({"role": "user", "content": preflight_error})
                continue

            tool_results: list[dict[str, Any]] = []
            any_tool_failed = False

            for item in prepared_calls:
                call_id = item["call_id"]
                name = item["name"]
                args = item["args"]
                kind = item["kind"]

                if kind == "unknown_tool":
                    tool_output = {"error": "unknown_tool", "tool": name, "args": args}
                else:
                    tool_output = _safe_run_tool(name, args)

                if name == "GlobalClusterVerifier":
                    verifier_called = True
                    last_verifier_result = tool_output.get("result")

                logger.info(
                    "Tool %s output: %s",
                    name,
                    json.dumps(tool_output, ensure_ascii=True),
                )

                tool_output_for_llm = _compact_tool_output_for_llm(tool_output)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(tool_output_for_llm, ensure_ascii=True),
                    }
                )

                if tool_output.get("error") is not None:
                    any_tool_failed = True

                tool_results.append(
                    {
                        "name": name,
                        "args": args,
                        "raw_output": tool_output,
                        "state_key": item.get("state_key"),
                        "canonical_transforms": item.get("canonical_transforms"),
                    }
                )

            if any_tool_failed:
                first_failed = None
                for tr in tool_results:
                    raw = tr["raw_output"]
                    if isinstance(raw, dict) and raw.get("error") is not None:
                        first_failed = raw
                        break

                fail_msg = (
                    str(first_failed.get("message") or first_failed.get("error"))
                    if isinstance(first_failed, dict)
                    else "tool_failed"
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "A tool call failed. Fix your next tool call or JSON draft instead of repeating the same call. "
                            f"Failure: {fail_msg}. "
                            "Important: provide complete cluster_transforms, keep all ids exactly once, and avoid null/invalid scalar values."
                        ),
                    }
                )
                continue

            gv_results = [
                tr for tr in tool_results if tr.get("name") == "GlobalClusterVerifier"
            ]

            if not gv_results:
                messages.append(
                    {
                        "role": "user",
                        "content": "You used tools, but did not call GlobalClusterVerifier. Call it before final output.",
                    }
                )
                continue

            tr = gv_results[-1]
            tool_output = tr["raw_output"]
            result = tool_output.get("result")
            quality = tool_output.get("quality")
            quality = quality if isinstance(quality, dict) else None
            score = _extract_layout_score(tool_output)

            transforms = tr.get("canonical_transforms")
            transforms = transforms if isinstance(transforms, list) else []
            state_key = tr.get("state_key")

            if isinstance(state_key, str):
                errs = tool_output.get("errors", [])
                verified_states[state_key] = {
                    "result": result,
                    "score": score,
                    "quality": quality,
                    "errors_preview": errs[:8] if isinstance(errs, list) else errs,
                }
                tried_patch_keys_by_state.setdefault(state_key, set())

            if result == "VALID":
                invalid_state_trace.clear()

                improved = _is_better_valid_candidate(
                    new_score=score,
                    new_quality=quality,
                    old_score=best_valid_score,
                    old_quality=best_valid_quality,
                )

                if improved or best_valid_transforms is None:
                    best_valid_score = score
                    best_valid_quality = quality
                    best_valid_transforms = transforms
                    no_improve_streak = 0
                else:
                    no_improve_streak += 1

                logger.info(
                    "VALID placement found: score=%s improved=%s no_improve_streak=%s improve_rounds_used=%s",
                    score,
                    improved,
                    no_improve_streak,
                    improve_rounds_used,
                )

                improve_allowed = _should_try_soft_improve(
                    quality=quality,
                    improve_rounds_used=improve_rounds_used,
                    soft_improve_budget=soft_improve_budget,
                    no_improve_streak=no_improve_streak,
                    min_explore_valid_rounds=min_explore_valid_rounds,
                )

                tried_patch_keys = (
                    tried_patch_keys_by_state[state_key]
                    if isinstance(state_key, str)
                    else set()
                )

                selected_improve_patch = None
                if improve_allowed:
                    selected_improve_patch = _select_single_global_improve_patch(
                        tool_output=tool_output,
                        transforms=transforms,
                        seen_state_keys=set(verified_states.keys()),
                        tried_patch_keys=tried_patch_keys,
                        blocked_cluster_ids=set(),
                    )

                if selected_improve_patch is not None:
                    patched = _apply_global_patch(transforms, selected_improve_patch)
                    ok, detail = _validate_transforms_full(
                        patched,
                        expected_cluster_ids,
                    )
                    patched_state_key = _state_key(patched)

                    if ok and patched_state_key not in verified_states:
                        if isinstance(state_key, str):
                            tried_patch_keys_by_state[state_key].add(
                                _global_patch_key(selected_improve_patch)
                            )

                        improve_rounds_used += 1
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "GlobalClusterVerifier returned VALID. "
                                    "A deterministic SINGLE-CLUSTER improvement candidate is available. "
                                    "Use it as your next attempt. Do NOT combine it with any other move in the same iteration. "
                                    "Re-call GlobalClusterVerifier with EXACTLY these cluster_transforms unless you have one clearly better single-cluster improvement.\n\n"
                                    f"selected_improve_patch={json.dumps(selected_improve_patch, ensure_ascii=True)}\n"
                                    f"patched_cluster_transforms={json.dumps(patched, ensure_ascii=True)}"
                                ),
                            }
                        )
                        continue

                    logger.info(
                        "Discarded improve patch due to invalid structure or repeated state: %s",
                        detail,
                    )

                if not _can_finalize_valid_layout(quality):
                    if improve_rounds_used < soft_improve_budget:
                        improve_rounds_used += 1
                        orientation_preview = _compact_orientation_preview(
                            quality.get("orientation_debug")
                            if isinstance(quality, dict)
                            else None,
                            max_items=8,
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "GlobalClusterVerifier returned VALID but the layout still has CRITICAL orientation mismatch. "
                                    "Do NOT finalize. Change only ONE cluster to reduce the orientation mismatch and re-check. "
                                    "Prefer fixing face_each_other / face_cluster / access_faces_other problems first.\n\n"
                                    f"orientation_preview={json.dumps(orientation_preview, ensure_ascii=True)}"
                                ),
                            }
                        )
                        continue

                    if best_valid_transforms is not None:
                        return _build_best_valid_json(
                            grid_mm=grid_mm,
                            transforms=best_valid_transforms,
                            quality=best_valid_quality,
                        )

                if improve_allowed:
                    improve_rounds_used += 1
                    quality_preview = _compact_quality_for_llm(quality)
                    improve_hints_preview = _compact_improve_hints_preview(
                        tool_output.get("improve_hints"),
                        max_items=2,
                        max_candidates=2,
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "GlobalClusterVerifier returned VALID. "
                                f"Current layout_score={score}. "
                                "Do NOT finalize yet. Explore ONE alternative VALID layout by changing only ONE cluster "
                                "if it can improve layout_score, total_soft_penalty_mm, orientation, circulation, or affinity. "
                                "Re-call GlobalClusterVerifier after that single-cluster change. "
                                "If no clearly better VALID alternative exists, then output final JSON.\n\n"
                                f"quality_preview={json.dumps(quality_preview, ensure_ascii=True)}\n"
                                f"improve_hints_preview={json.dumps(improve_hints_preview, ensure_ascii=True)}"
                            ),
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "GlobalClusterVerifier returned VALID. "
                                "Keep this as the current best valid layout. "
                                "If there is any unseen SINGLE-CLUSTER valid improvement for orientation / layout_score, try it next. "
                                "Otherwise you may output final JSON."
                            ),
                        }
                    )
                continue

            errs = tool_output.get("errors", [])
            errs_preview = errs[:8] if isinstance(errs, list) else errs
            hints_preview = _compact_repair_hints_preview(
                tool_output.get("repair_hints"),
                max_items=2,
                max_candidates=2,
            )
            improve_hints_preview = _compact_improve_hints_preview(
                tool_output.get("improve_hints"),
                max_items=2,
                max_candidates=2,
            )

            if isinstance(state_key, str):
                tried_patch_keys_by_state.setdefault(state_key, set())
            tried_patch_keys = (
                tried_patch_keys_by_state[state_key]
                if isinstance(state_key, str)
                else set()
            )

            remaining_hard_issues = _count_geometry_hard_errors(errs)
            orientation_only = _orientation_only_invalid(errs)

            invalid_state_trace.append(
                {
                    "state_key": state_key,
                    "result": "INVALID",
                    "transforms": transforms,
                    "errors": errs if isinstance(errs, list) else [],
                    "remaining_hard_issues": remaining_hard_issues,
                    "orientation_only": orientation_only,
                    "layout_score": score,
                    "quality": quality,
                }
            )
            invalid_state_trace = invalid_state_trace[-8:]

            stagnation = _detect_invalid_stagnation(invalid_state_trace, window=4)
            blocked_cluster_ids = set(stagnation.get("blocked_cluster_ids") or [])

            # Chỉ cho pairwise khi:
            # - hiện tại chỉ còn ORIENTATION_HARD
            # - và search đang stuck thật sự
            allow_pairwise_patch = bool(orientation_only and stagnation.get("is_stuck"))

            selected_patch = _select_single_global_patch(
                tool_output=tool_output,
                transforms=transforms,
                seen_state_keys=set(verified_states.keys()),
                tried_patch_keys=tried_patch_keys,
                blocked_cluster_ids=blocked_cluster_ids,
                allow_pairwise=allow_pairwise_patch,
            )

            # Chỉ dùng improve_hints để rescue khi INVALID mà chỉ còn orientation.
            if selected_patch is None and orientation_only:
                selected_patch = _select_single_global_invalid_improve_patch(
                    tool_output=tool_output,
                    transforms=transforms,
                    seen_state_keys=set(verified_states.keys()),
                    tried_patch_keys=tried_patch_keys,
                    blocked_cluster_ids=blocked_cluster_ids,
                )

            if selected_patch is not None:
                patched = _apply_global_patch(transforms, selected_patch)
                ok, detail = _validate_transforms_full(
                    patched,
                    expected_cluster_ids,
                )
                patched_state_key = _state_key(patched)

                if ok and patched_state_key not in verified_states:
                    if isinstance(state_key, str):
                        tried_patch_keys_by_state[state_key].add(
                            _global_patch_key(selected_patch)
                        )

                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "GlobalClusterVerifier returned INVALID. "
                                "Use the following SINGLE patched candidate as your next attempt. "
                                "Do NOT combine it with other moves in the same iteration. "
                                "Re-call GlobalClusterVerifier with EXACTLY these cluster_transforms unless you have one clearly better single-cluster patch.\n\n"
                                f"selected_patch={json.dumps(selected_patch, ensure_ascii=True)}\n"
                                f"patched_cluster_transforms={json.dumps(patched, ensure_ascii=True)}\n"
                                f"repair_hints_preview={json.dumps(hints_preview, ensure_ascii=True)}\n"
                                f"improve_hints_preview={json.dumps(improve_hints_preview, ensure_ascii=True)}"
                            ),
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "GlobalClusterVerifier returned INVALID. "
                                "A deterministic patch was considered but would still repeat an already verified state or became invalid structurally. "
                                "Choose a DIFFERENT single-cluster repair and re-check. "
                                f"Errors (preview): {json.dumps(errs_preview, ensure_ascii=True)} "
                                f"Repair hints (preview): {json.dumps(hints_preview, ensure_ascii=True)} "
                                f"Patch issue: {detail}"
                            ),
                        }
                    )
            else:
                if stagnation.get("is_stuck"):
                    repeated_cluster_id = stagnation.get("repeated_cluster_id")
                    preferred_other_clusters = (
                        stagnation.get("preferred_other_clusters") or []
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "GlobalClusterVerifier returned INVALID and the search is STUCK in a local crawl. "
                                "The same cluster was changed repeatedly across recent invalid rounds without reducing hard issue count. "
                                "Do NOT nudge that same cluster again by a tiny step. "
                                f"blocked_cluster_id={json.dumps(repeated_cluster_id, ensure_ascii=True)} "
                                f"preferred_other_clusters={json.dumps(preferred_other_clusters, ensure_ascii=True)} "
                                "Change a DIFFERENT cluster next and re-check. "
                                f"Errors (preview): {json.dumps(errs_preview, ensure_ascii=True)} "
                                f"Repair hints (preview): {json.dumps(hints_preview, ensure_ascii=True)} "
                                f"Improve hints (preview): {json.dumps(improve_hints_preview, ensure_ascii=True)}"
                            ),
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "GlobalClusterVerifier returned INVALID. "
                                "Fix ONLY the reported hard issues and re-check. "
                                "Change one cluster if possible, and do NOT repeat previous invalid states. "
                                f"Errors (preview): {json.dumps(errs_preview, ensure_ascii=True)} "
                                f"Repair hints (preview): {json.dumps(hints_preview, ensure_ascii=True)} "
                                f"Improve hints (preview): {json.dumps(improve_hints_preview, ensure_ascii=True)}"
                            ),
                        }
                    )
            continue

        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            messages.append(
                {
                    "role": "user",
                    "content": "Return a JSON object only (no markdown).",
                }
            )
            continue

        logger.info("Assistant content length: %s", len(content))

        if not verifier_called and not forced_check:
            forced_check = True
            messages.append(
                {
                    "role": "user",
                    "content": "You MUST call GlobalClusterVerifier before final output.",
                }
            )
            continue

        draft = _try_parse_json_object(content)
        if not isinstance(draft, dict):
            if best_valid_transforms is not None:
                return _build_best_valid_json(
                    grid_mm=grid_mm,
                    transforms=best_valid_transforms,
                    quality=best_valid_quality,
                )
            messages.append(
                {
                    "role": "user",
                    "content": "Your output must be valid JSON only.",
                }
            )
            continue

        status = draft.get("status")
        if status not in {"OK", "UNSAT", "NEED_INFO"}:
            messages.append(
                {
                    "role": "user",
                    "content": "Top-level field status must be one of OK|UNSAT|NEED_INFO.",
                }
            )
            continue

        transforms = draft.get("cluster_transforms")
        transforms_list = (
            _canonicalize_cluster_transforms(transforms)
            if isinstance(transforms, list)
            else None
        )

        if status == "OK":
            ok, detail = _validate_transforms_full(
                transforms_list,
                expected_cluster_ids,
            )
            if not ok:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "For status=OK, cluster_transforms must contain every cluster exactly once. "
                            f"Issue: {detail}"
                        ),
                    }
                )
                continue

        if last_verifier_result != "VALID":
            tool_output = _safe_run_tool(
                "GlobalClusterVerifier",
                {
                    "room_model": room_model,
                    "clusters_outlines": clusters_outlines,
                    "cluster_transforms": transforms_list
                    if transforms_list is not None
                    else [],
                    "grid_mm": grid_mm,
                    "forbid_obstacle_touch": True,
                    "relation_plan": relation_plan_json
                    if isinstance(relation_plan_json, dict)
                    else None,
                    "generate_hints": True,
                    "consider_rotations": True,
                    "hard_orientation_gate": False,
                    "orientation_hard_threshold_mm": 220,
                    "min_preferred_gap_mm": 180,
                    "max_preferred_gap_mm": 450,
                },
            )

            if tool_output.get("error") is not None:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "GlobalClusterVerifier failed on your final draft. "
                            "Do NOT finalize. Fix the transforms or the tool-call inputs and verify again. "
                            f"Failure: {tool_output.get('message') or tool_output.get('error')}"
                        ),
                    }
                )
                continue

            verifier_called = True
            last_verifier_result = tool_output.get("result")

            if status == "OK" and last_verifier_result != "VALID":
                errs = tool_output.get("errors", [])
                errs_preview = errs[:8] if isinstance(errs, list) else errs
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your final JSON is still INVALID according to GlobalClusterVerifier. "
                            "Fix ONLY these errors and re-check by calling GlobalClusterVerifier again. "
                            f"Errors (preview): {json.dumps(errs_preview, ensure_ascii=True)}"
                        ),
                    }
                )
                continue

            if status != "OK":
                if best_valid_transforms is not None:
                    return _build_best_valid_json(
                        grid_mm=grid_mm,
                        transforms=best_valid_transforms,
                        quality=best_valid_quality,
                    )
                return content

            quality = tool_output.get("quality")
            quality = quality if isinstance(quality, dict) else None
            score = _extract_layout_score(tool_output)

            if transforms_list is not None and _is_better_valid_candidate(
                new_score=score,
                new_quality=quality,
                old_score=best_valid_score,
                old_quality=best_valid_quality,
            ):
                best_valid_score = score
                best_valid_quality = quality
                best_valid_transforms = transforms_list

            if not _can_finalize_valid_layout(quality):
                tried_patch_keys = set()
                selected_improve_patch = _select_single_global_improve_patch(
                    tool_output=tool_output,
                    transforms=transforms_list or [],
                    seen_state_keys=set(verified_states.keys()),
                    tried_patch_keys=tried_patch_keys,
                    blocked_cluster_ids=set(),
                )

                if (
                    selected_improve_patch is not None
                    and improve_rounds_used < soft_improve_budget
                ):
                    improve_rounds_used += 1
                    patched = _apply_global_patch(
                        transforms_list or [], selected_improve_patch
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your latest VALID layout still has critical orientation mismatch. "
                                "Use this deterministic SINGLE-CLUSTER improvement next and re-call GlobalClusterVerifier.\n\n"
                                f"selected_improve_patch={json.dumps(selected_improve_patch, ensure_ascii=True)}\n"
                                f"patched_cluster_transforms={json.dumps(patched, ensure_ascii=True)}"
                            ),
                        }
                    )
                    continue

                if improve_rounds_used < soft_improve_budget:
                    improve_rounds_used += 1
                    orientation_preview = _compact_orientation_preview(
                        quality.get("orientation_debug")
                        if isinstance(quality, dict)
                        else None,
                        max_items=8,
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your latest VALID layout still has critical orientation mismatch. "
                                "Do NOT finalize. Change only ONE cluster and re-call GlobalClusterVerifier. "
                                f"orientation_preview={json.dumps(orientation_preview, ensure_ascii=True)}"
                            ),
                        }
                    )
                    continue

                if best_valid_transforms is not None:
                    return _build_best_valid_json(
                        grid_mm=grid_mm,
                        transforms=best_valid_transforms,
                        quality=best_valid_quality,
                    )

            return content

        if status != "OK":
            if best_valid_transforms is not None:
                return _build_best_valid_json(
                    grid_mm=grid_mm,
                    transforms=best_valid_transforms,
                    quality=best_valid_quality,
                )
            return content

        if transforms_list is None:
            if best_valid_transforms is not None:
                return _build_best_valid_json(
                    grid_mm=grid_mm,
                    transforms=best_valid_transforms,
                    quality=best_valid_quality,
                )
            return content

        tool_output = _safe_run_tool(
            "GlobalClusterVerifier",
            {
                "room_model": room_model,
                "clusters_outlines": clusters_outlines,
                "cluster_transforms": transforms_list,
                "grid_mm": grid_mm,
                "forbid_obstacle_touch": True,
                "relation_plan": relation_plan_json
                if isinstance(relation_plan_json, dict)
                else None,
                "generate_hints": True,
                "consider_rotations": True,
                "hard_orientation_gate": False,
                "orientation_hard_threshold_mm": 220,
                "min_preferred_gap_mm": 180,
                "max_preferred_gap_mm": 450,
            },
        )

        if tool_output.get("error") is not None:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "GlobalClusterVerifier failed on your latest draft. "
                        "Do NOT finalize. Fix the transforms or verifier inputs and verify again. "
                        f"Failure: {tool_output.get('message') or tool_output.get('error')}"
                    ),
                }
            )
            continue

        if tool_output.get("result") == "VALID":
            quality = tool_output.get("quality")
            quality = quality if isinstance(quality, dict) else None
            score = _extract_layout_score(tool_output)

            if _is_better_valid_candidate(
                new_score=score,
                new_quality=quality,
                old_score=best_valid_score,
                old_quality=best_valid_quality,
            ):
                best_valid_score = score
                best_valid_quality = quality
                best_valid_transforms = transforms_list

            if not _can_finalize_valid_layout(quality):
                selected_improve_patch = _select_single_global_improve_patch(
                    tool_output=tool_output,
                    transforms=transforms_list,
                    seen_state_keys=set(verified_states.keys()),
                    tried_patch_keys=set(),
                    blocked_cluster_ids=set(),
                )

                if (
                    selected_improve_patch is not None
                    and improve_rounds_used < soft_improve_budget
                ):
                    improve_rounds_used += 1
                    patched = _apply_global_patch(
                        transforms_list, selected_improve_patch
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your latest VALID layout still has critical orientation mismatch. "
                                "Use this deterministic SINGLE-CLUSTER improvement next and re-call GlobalClusterVerifier.\n\n"
                                f"selected_improve_patch={json.dumps(selected_improve_patch, ensure_ascii=True)}\n"
                                f"patched_cluster_transforms={json.dumps(patched, ensure_ascii=True)}"
                            ),
                        }
                    )
                    continue

                if improve_rounds_used < soft_improve_budget:
                    improve_rounds_used += 1
                    orientation_preview = _compact_orientation_preview(
                        quality.get("orientation_debug")
                        if isinstance(quality, dict)
                        else None,
                        max_items=8,
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your latest VALID layout still has critical orientation mismatch. "
                                "Do NOT finalize. Change only ONE cluster and re-call GlobalClusterVerifier. "
                                f"orientation_preview={json.dumps(orientation_preview, ensure_ascii=True)}"
                            ),
                        }
                    )
                    continue

                if best_valid_transforms is not None:
                    return _build_best_valid_json(
                        grid_mm=grid_mm,
                        transforms=best_valid_transforms,
                        quality=best_valid_quality,
                    )

            if best_valid_transforms is not None:
                best_json = _build_best_valid_json(
                    grid_mm=grid_mm,
                    transforms=best_valid_transforms,
                    quality=best_valid_quality,
                )
                if best_json != content:
                    logger.info(
                        "Returning previously seen better valid state: best_score=%s current_score=%s",
                        best_valid_score,
                        score,
                    )
                    return best_json

        return content

    if best_valid_transforms is not None:
        logger.warning(
            "MacroClusterPlacer hit max_steps; returning best valid state with score=%s quality=%s",
            best_valid_score,
            json.dumps(best_valid_quality, ensure_ascii=True)
            if isinstance(best_valid_quality, dict)
            else None,
        )
        return _build_best_valid_json(
            grid_mm=grid_mm,
            transforms=best_valid_transforms,
            quality=best_valid_quality,
        )

    logger.warning("MacroClusterPlacer hit max_steps with no valid layout.")
    return json.dumps(
        {
            "status": "UNSAT",
            "grid_mm": int(grid_mm),
            "cluster_transforms": [],
            "notes": ["Stopped due to max_steps with no verified valid layout."],
        },
        ensure_ascii=True,
    )


def _extract_message(response: object) -> object:
    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        return getattr(choices[0], "message", None)
    raise ValueError("OpenAI response missing message")


def _extract_tool_calls(message: object) -> list[dict[str, Any]]:
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        return []
    output: list[dict[str, Any]] = []
    for call in tool_calls:
        if isinstance(call, dict):
            output.append(call)
            continue
        fn = getattr(call, "function", None)
        output.append(
            {
                "id": getattr(call, "id", None),
                "type": getattr(call, "type", "function"),
                "function": {
                    "name": getattr(fn, "name", None) if fn is not None else None,
                    "arguments": getattr(fn, "arguments", None)
                    if fn is not None
                    else None,
                },
            }
        )
    return output


def _safe_json_loads(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _safe_run_tool(name: str | None, args: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(name, str) or name not in TOOL_REGISTRY:
        return {"error": "unknown_tool", "tool": name, "args": args}
    try:
        result = TOOL_REGISTRY[name](**args)
        if result is None:
            return {"error": "tool_returned_none", "tool": name, "args": args}
        if isinstance(result, dict):
            return result
        return {"error": "tool_returned_non_dict", "tool": name, "result": str(result)}
    except Exception as exc:
        return {"error": "tool_failed", "tool": name, "message": str(exc), "args": args}


def _coerce_verifier_args(
    *,
    args: dict[str, Any],
    room_model: dict[str, Any],
    clusters_outlines: dict[str, Any],
    grid_mm: int,
    assistant_content: str,
    existing_transforms_json: dict[str, Any] | None,
    last_draft_transforms: list[dict[str, Any]] | None,
    relation_plan_json: dict[str, Any] | None,
) -> dict[str, Any]:
    output = dict(args)

    if "room_model" not in output:
        output["room_model"] = room_model
    if "clusters_outlines" not in output:
        output["clusters_outlines"] = clusters_outlines
    if "grid_mm" not in output:
        output["grid_mm"] = int(grid_mm)

    if "relation_plan" not in output:
        output["relation_plan"] = (
            relation_plan_json if isinstance(relation_plan_json, dict) else None
        )

    if "cluster_transforms" not in output or not isinstance(
        output.get("cluster_transforms"), list
    ):
        draft = _try_parse_json_object(assistant_content)
        if isinstance(draft, dict) and isinstance(
            draft.get("cluster_transforms"), list
        ):
            output["cluster_transforms"] = draft.get("cluster_transforms")

    if "cluster_transforms" not in output or not isinstance(
        output.get("cluster_transforms"), list
    ):
        if existing_transforms_json and isinstance(
            existing_transforms_json.get("cluster_transforms"), list
        ):
            output["cluster_transforms"] = existing_transforms_json.get(
                "cluster_transforms"
            )

    if "cluster_transforms" not in output or not isinstance(
        output.get("cluster_transforms"), list
    ):
        if isinstance(last_draft_transforms, list) and last_draft_transforms:
            output["cluster_transforms"] = last_draft_transforms

    if "cluster_transforms" not in output:
        output["cluster_transforms"] = []

    if isinstance(output.get("cluster_transforms"), list):
        output["cluster_transforms"] = _canonicalize_cluster_transforms(
            output["cluster_transforms"]
        )

    output["generate_hints"] = bool(output.get("generate_hints", True))
    output["consider_rotations"] = bool(output.get("consider_rotations", True))
    output["forbid_obstacle_touch"] = bool(output.get("forbid_obstacle_touch", True))
    # output["hard_orientation_gate"] = bool(output.get("hard_orientation_gate", False))
    output["hard_orientation_gate"] = False
    output["orientation_hard_threshold_mm"] = int(
        output.get("orientation_hard_threshold_mm", 220) or 220
    )

    min_gap = output.get("min_preferred_gap_mm", 180)
    max_gap = output.get("max_preferred_gap_mm", 450)

    if min_gap is None:
        min_gap = 180
    if max_gap is None:
        max_gap = 450

    try:
        min_gap = int(min_gap)
    except Exception:
        min_gap = 180

    try:
        max_gap = int(max_gap)
    except Exception:
        max_gap = 450

    if max_gap < min_gap:
        max_gap = min_gap

    output["min_preferred_gap_mm"] = min_gap
    output["max_preferred_gap_mm"] = max_gap

    if (
        "base_preferred_gap_mm" in output
        and output["base_preferred_gap_mm"] is not None
    ):
        try:
            output["base_preferred_gap_mm"] = int(output["base_preferred_gap_mm"])
        except Exception:
            output["base_preferred_gap_mm"] = None
    else:
        output["base_preferred_gap_mm"] = None

    return output


def _validate_transforms_full(
    transforms: Any,
    expected_cluster_ids: list[str],
) -> tuple[bool, str]:
    if not isinstance(transforms, list) or not transforms:
        return False, "cluster_transforms is missing or empty"

    seen: list[str] = []
    for t in transforms:
        if not isinstance(t, dict):
            continue
        cid = t.get("cluster_id")
        if isinstance(cid, str) and cid:
            seen.append(cid)

    if not seen:
        return False, "cluster_transforms has no valid cluster_id entries"

    missing = [cid for cid in expected_cluster_ids if cid not in seen]
    extra = [cid for cid in seen if cid not in expected_cluster_ids]
    dups = sorted({cid for cid in seen if seen.count(cid) > 1})

    if missing:
        return False, f"missing cluster_ids: {missing}"
    if extra:
        return False, f"unexpected cluster_ids: {extra}"
    if dups:
        return False, f"duplicate cluster_ids: {dups}"
    if len(seen) != len(expected_cluster_ids):
        return False, "cluster_transforms count mismatch"
    return True, "ok"


def _canonicalize_cluster_transforms(
    transforms: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in transforms:
        if not isinstance(t, dict):
            continue
        cid = t.get("cluster_id")
        if not isinstance(cid, str) or not cid:
            continue
        out.append(
            {
                "cluster_id": cid,
                "x": int(t.get("x", 0) or 0),
                "y": int(t.get("y", 0) or 0),
                "rot": int(t.get("rot", 0) or 0) % 360,
            }
        )
    out.sort(key=lambda item: item["cluster_id"])
    return out


def _canonicalize_patch_clusters(
    patch_clusters: Any,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(patch_clusters, list):
        return out

    seen = set()
    for item in patch_clusters:
        if not isinstance(item, dict):
            continue
        cid = item.get("cluster_id")
        if not isinstance(cid, str) or not cid:
            continue
        row = {
            "cluster_id": cid,
            "x": int(item.get("x", 0) or 0),
            "y": int(item.get("y", 0) or 0),
            "rot": int(item.get("rot", 0) or 0) % 360,
        }
        key = (row["cluster_id"], row["x"], row["y"], row["rot"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)

    out.sort(key=lambda r: r["cluster_id"])
    return out


def _candidate_patch_clusters(
    candidate: dict[str, Any],
    *,
    default_cluster_id: str | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(candidate, dict):
        return []

    if isinstance(candidate.get("patch_clusters"), list):
        patch_clusters = _canonicalize_patch_clusters(candidate.get("patch_clusters"))
        if patch_clusters:
            return patch_clusters

    cid = (
        default_cluster_id
        if isinstance(default_cluster_id, str) and default_cluster_id
        else candidate.get("cluster_id")
    )
    x = candidate.get("x")
    y = candidate.get("y")
    rot = candidate.get("rot")

    if (
        isinstance(cid, str)
        and cid
        and all(isinstance(v, (int, float)) for v in (x, y, rot))
    ):
        return _canonicalize_patch_clusters(
            [{"cluster_id": cid, "x": x, "y": y, "rot": rot}]
        )

    return []


def _state_key(transforms: list[dict[str, Any]]) -> str:
    return json.dumps(
        _canonicalize_cluster_transforms(transforms),
        ensure_ascii=True,
        separators=(",", ":"),
    )


def _global_patch_key(patch: dict[str, Any]) -> str:
    patch_clusters = _candidate_patch_clusters(
        patch,
        default_cluster_id=patch.get("cluster_id") if isinstance(patch, dict) else None,
    )
    payload = {
        "trigger": patch.get("trigger"),
        "with_id": patch.get("with_id"),
        "reason": patch.get("reason"),
        "patch_clusters": patch_clusters,
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _extract_transforms_map(
    transforms: list[dict[str, Any]] | None,
) -> dict[str, tuple[int, int, int]]:
    out: dict[str, tuple[int, int, int]] = {}
    if not isinstance(transforms, list):
        return out
    for t in transforms:
        if not isinstance(t, dict):
            continue
        cid = t.get("cluster_id")
        if not isinstance(cid, str) or not cid:
            continue
        out[cid] = (
            int(t.get("x", 0) or 0),
            int(t.get("y", 0) or 0),
            int(t.get("rot", 0) or 0) % 360,
        )
    return out


def _changed_cluster_ids(
    prev_transforms: list[dict[str, Any]] | None,
    curr_transforms: list[dict[str, Any]] | None,
) -> list[str]:
    prev_map = _extract_transforms_map(prev_transforms)
    curr_map = _extract_transforms_map(curr_transforms)
    ids = sorted(set(prev_map.keys()) | set(curr_map.keys()))
    changed: list[str] = []
    for cid in ids:
        if prev_map.get(cid) != curr_map.get(cid):
            changed.append(cid)
    return changed


def _count_geometry_hard_errors(errors: Any) -> int:
    if not isinstance(errors, list):
        return 0
    count = 0
    for e in errors:
        if not isinstance(e, dict):
            continue
        code = e.get("code")
        if code == "CLUSTER_OUT_OF_BOUNDS":
            count += 1
        elif code == "CLUSTER_OVERLAP":
            count += 1
        elif code == "CLUSTER_INTERSECTS_OBSTACLE":
            obs = e.get("obstacles")
            if isinstance(obs, list) and obs:
                count += len(obs)
            else:
                count += 1
    return count


def _orientation_only_invalid(errors: Any) -> bool:
    if not isinstance(errors, list) or not errors:
        return False
    has_orientation = False
    for e in errors:
        if not isinstance(e, dict):
            continue
        code = e.get("code")
        if code == "ORIENTATION_HARD":
            has_orientation = True
            continue
        return False
    return has_orientation


def _dominant_error_clusters(errors: Any, max_items: int = 3) -> list[str]:
    if not isinstance(errors, list):
        return []

    scores: dict[str, int] = {}
    for e in errors:
        if not isinstance(e, dict):
            continue
        cid = e.get("cluster_id")
        if not isinstance(cid, str) or not cid:
            continue

        code = e.get("code")
        penalty = int(e.get("penalty_mm", 0) or 0)

        if code == "ORIENTATION_HARD":
            scores[cid] = scores.get(cid, 0) + max(1, penalty)
        else:
            scores[cid] = scores.get(cid, 0) + 5000 + max(1, penalty)

    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return [cid for cid, _ in ranked[:max_items]]


def _detect_invalid_stagnation(
    invalid_state_trace: list[dict[str, Any]],
    *,
    window: int = 4,
) -> dict[str, Any]:
    if len(invalid_state_trace) < window:
        return {
            "is_stuck": False,
            "blocked_cluster_ids": [],
            "repeated_cluster_id": None,
            "preferred_other_clusters": [],
            "orientation_only": False,
        }

    tail = invalid_state_trace[-window:]
    if any(item.get("result") != "INVALID" for item in tail):
        return {
            "is_stuck": False,
            "blocked_cluster_ids": [],
            "repeated_cluster_id": None,
            "preferred_other_clusters": [],
            "orientation_only": False,
        }

    changed_seq: list[str] = []
    for prev, curr in zip(tail, tail[1:]):
        changed = _changed_cluster_ids(prev.get("transforms"), curr.get("transforms"))
        if len(changed) != 1:
            return {
                "is_stuck": False,
                "blocked_cluster_ids": [],
                "repeated_cluster_id": None,
                "preferred_other_clusters": [],
                "orientation_only": False,
            }
        changed_seq.append(changed[0])

    if not changed_seq or len(set(changed_seq)) != 1:
        return {
            "is_stuck": False,
            "blocked_cluster_ids": [],
            "repeated_cluster_id": None,
            "preferred_other_clusters": [],
            "orientation_only": False,
        }

    repeated_cluster_id = changed_seq[0]
    hard_counts = [
        int(item.get("remaining_hard_issues", 10**9) or 10**9) for item in tail
    ]
    orientation_only = all(bool(item.get("orientation_only")) for item in tail)

    # Kẹt kiểu local crawl: cùng 1 cluster bị chỉnh liên tiếp,
    # số hard geometry issue không giảm.
    if len(set(hard_counts)) != 1:
        return {
            "is_stuck": False,
            "blocked_cluster_ids": [],
            "repeated_cluster_id": None,
            "preferred_other_clusters": [],
            "orientation_only": orientation_only,
        }

    preferred = _dominant_error_clusters(tail[-1].get("errors"), max_items=3)
    preferred = [cid for cid in preferred if cid != repeated_cluster_id]

    return {
        "is_stuck": True,
        "blocked_cluster_ids": [repeated_cluster_id],
        "repeated_cluster_id": repeated_cluster_id,
        "preferred_other_clusters": preferred,
        "orientation_only": orientation_only,
    }


def _apply_global_patch(
    transforms: list[dict[str, Any]],
    patch: dict[str, Any],
) -> list[dict[str, Any]]:
    base = _canonicalize_cluster_transforms(transforms)
    patch_clusters = _candidate_patch_clusters(
        patch,
        default_cluster_id=patch.get("cluster_id") if isinstance(patch, dict) else None,
    )
    if not patch_clusters:
        return base

    patch_map = {p["cluster_id"]: p for p in patch_clusters}
    out: list[dict[str, Any]] = []

    seen_ids = set()
    for t in base:
        cid = t["cluster_id"]
        seen_ids.add(cid)
        if cid in patch_map:
            p = patch_map[cid]
            out.append(
                {
                    "cluster_id": cid,
                    "x": int(p["x"]),
                    "y": int(p["y"]),
                    "rot": int(p["rot"]) % 360,
                }
            )
        else:
            out.append(dict(t))

    for cid, p in patch_map.items():
        if cid in seen_ids:
            continue
        out.append(
            {
                "cluster_id": cid,
                "x": int(p["x"]),
                "y": int(p["y"]),
                "rot": int(p["rot"]) % 360,
            }
        )

    out.sort(key=lambda item: item["cluster_id"])
    return out


def _select_single_global_patch(
    *,
    tool_output: dict[str, Any],
    transforms: list[dict[str, Any]],
    seen_state_keys: set[str],
    tried_patch_keys: set[str],
    blocked_cluster_ids: set[str] | None = None,
    allow_pairwise: bool | None = None,
) -> dict[str, Any] | None:
    repair_hints = tool_output.get("repair_hints")
    if not isinstance(repair_hints, list) or not repair_hints:
        return None

    blocked_cluster_ids = blocked_cluster_ids or set()

    errors = tool_output.get("errors")
    current_geometry_hard_count = _count_geometry_hard_errors(errors)
    orientation_only = _orientation_only_invalid(errors)

    def _current_geometry_hard_score(errs: Any) -> int:
        if not isinstance(errs, list):
            return 10**9
        score = 0
        for e in errs:
            if not isinstance(e, dict):
                continue
            code = e.get("code")
            if code == "CLUSTER_OUT_OF_BOUNDS":
                score += 1000
            elif code == "CLUSTER_INTERSECTS_OBSTACLE":
                obs = e.get("obstacles")
                if isinstance(obs, list) and obs:
                    score += 200 * len(obs)
                else:
                    score += 200
            elif code == "CLUSTER_OVERLAP":
                score += 300
        return score

    current_geometry_hard_score = _current_geometry_hard_score(errors)

    # Chỉ cho pairwise khi caller cho phép rõ ràng.
    # Fallback an toàn: nếu không truyền thì mặc định KHÔNG cho.
    if allow_pairwise is None:
        allow_pairwise = False

    error_priority = {
        "CLUSTER_OUT_OF_BOUNDS": 0,
        "CLUSTER_INTERSECTS_OBSTACLE": 1,
        "CLUSTER_OVERLAP": 2,
        "PAIRWISE_ORIENTATION_HARD": 3,
        "ORIENTATION_HARD": 4,
    }

    candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    for hint_idx, hint in enumerate(repair_hints):
        if not isinstance(hint, dict):
            continue

        cid = hint.get("cluster_id")
        trigger = hint.get("trigger")
        with_id = hint.get("with_id")
        hint_candidates = hint.get("candidates")

        if not isinstance(cid, str) or not cid:
            continue
        if not isinstance(hint_candidates, list):
            continue

        trig_rank = error_priority.get(trigger, 99)

        for cand_idx, cand in enumerate(hint_candidates):
            if not isinstance(cand, dict):
                continue

            patch_clusters = _candidate_patch_clusters(cand, default_cluster_id=cid)
            if not patch_clusters:
                continue

            patch_cluster_ids = [p["cluster_id"] for p in patch_clusters]
            is_pair_patch = len(patch_cluster_ids) > 1

            # Mặc định chỉ cho single patch.
            if is_pair_patch and not allow_pairwise:
                continue

            # Nếu cluster đang bị block vì crawl lặp, chặn single patch vào nó.
            # Pairwise chỉ được đi qua khi caller cố tình allow_pairwise=True.
            if (not is_pair_patch) and any(
                x in blocked_cluster_ids for x in patch_cluster_ids
            ):
                continue

            post = cand.get("postcheck")
            post = post if isinstance(post, dict) else {}

            all_hard_clear = bool(post.get("all_hard_clear", False))
            remaining_hard = int(post.get("remaining_hard_issues", 10**9) or 10**9)
            solves_trigger = bool(post.get("solves_trigger", False))
            hard_score = int(
                post.get("hard_violation_score", post.get("violation_score", 10**9))
                or 10**9
            )

            cluster_orient_rank = int(
                post.get("cluster_orientation_penalty_mm", 10**9) or 10**9
            )
            global_orient_rank = int(
                post.get("global_orientation_penalty_mm", 10**9) or 10**9
            )
            soft_rank = int(post.get("soft_penalty_mm", 10**9) or 10**9)
            cost_rank = int(cand.get("cost", 10**9) or 10**9)

            # ---- HARD MONOTONIC GATE ----
            # Nếu hiện tại chỉ còn orientation lỗi, thì patch tiếp theo
            # BẮT BUỘC phải giữ all_hard_clear, không được tái sinh hard geometry.
            if orientation_only:
                if (not all_hard_clear) or (remaining_hard != 0) or (hard_score != 0):
                    continue
            else:
                # Nếu còn hard geometry, patch phải làm tốt hơn:
                # hoặc clear sạch, hoặc giảm số hard issues,
                # hoặc cùng số issue nhưng hard_score phải giảm.
                improves_hard = False
                if all_hard_clear and remaining_hard == 0 and hard_score == 0:
                    improves_hard = True
                elif remaining_hard < current_geometry_hard_count:
                    improves_hard = True
                elif (
                    remaining_hard == current_geometry_hard_count
                    and hard_score < current_geometry_hard_score
                ):
                    improves_hard = True

                if not improves_hard:
                    continue

            patch = {
                "cluster_id": cid,
                "trigger": trigger,
                "with_id": with_id,
                "reason": cand.get("reason"),
                "cost": cost_rank,
                "postcheck": post,
                "patch_clusters": patch_clusters,
            }

            patch_key = _global_patch_key(patch)
            if patch_key in tried_patch_keys:
                continue

            patched = _apply_global_patch(transforms, patch)
            patched_state_key = _state_key(patched)
            if patched_state_key in seen_state_keys:
                continue

            # Single trước pair. Geometry trước orientation.
            # Chỉ sau khi hard ổn mới xét orientation / soft / cost.
            score = (
                0 if all_hard_clear else 1,
                remaining_hard,
                hard_score,
                0 if not is_pair_patch else 1,
                0 if solves_trigger else 1,
                trig_rank,
                cluster_orient_rank,
                global_orient_rank,
                soft_rank,
                cost_rank,
                hint_idx,
                cand_idx,
                cid,
            )
            candidates.append((score, patch))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _select_single_global_improve_patch(
    *,
    tool_output: dict[str, Any],
    transforms: list[dict[str, Any]],
    seen_state_keys: set[str],
    tried_patch_keys: set[str],
    blocked_cluster_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    improve_hints = tool_output.get("improve_hints")
    if not isinstance(improve_hints, list) or not improve_hints:
        return None

    blocked_cluster_ids = blocked_cluster_ids or set()

    base_score = _extract_layout_score(tool_output)
    base_quality = tool_output.get("quality")
    base_quality = base_quality if isinstance(base_quality, dict) else {}

    candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    for hint_idx, hint in enumerate(improve_hints):
        if not isinstance(hint, dict):
            continue

        cid = hint.get("cluster_id")
        hint_candidates = hint.get("candidates")

        if not isinstance(cid, str) or not cid:
            continue
        if cid in blocked_cluster_ids:
            continue
        if not isinstance(hint_candidates, list):
            continue

        for cand_idx, cand in enumerate(hint_candidates):
            if not isinstance(cand, dict):
                continue

            patch_clusters = _candidate_patch_clusters(cand, default_cluster_id=cid)
            if not patch_clusters:
                continue

            # VALID improve chỉ cho SINGLE-CLUSTER patch.
            if len(patch_clusters) != 1:
                continue

            postcheck = cand.get("postcheck")
            postcheck = postcheck if isinstance(postcheck, dict) else {}

            hard_score = int(
                postcheck.get(
                    "hard_violation_score", postcheck.get("violation_score", 10**9)
                )
                or 10**9
            )
            remaining_hard = int(postcheck.get("remaining_hard_issues", 10**9) or 10**9)
            all_hard_clear = bool(postcheck.get("all_hard_clear", False))

            # VALID improve tuyệt đối không được làm bẩn hard geometry.
            if hard_score != 0 or remaining_hard != 0 or not all_hard_clear:
                continue

            post_layout_score = postcheck.get("layout_score")
            if isinstance(post_layout_score, float):
                post_layout_score = int(round(post_layout_score))
            elif not isinstance(post_layout_score, int):
                post_layout_score = -(10**9)

            post_quality = postcheck.get("quality")
            if not isinstance(post_quality, dict):
                post_quality = {
                    "layout_score": postcheck.get("layout_score"),
                    "total_soft_penalty_mm": postcheck.get("soft_penalty_mm"),
                    "orientation_penalty_mm": postcheck.get(
                        "global_orientation_penalty_mm"
                    ),
                    "circulation_penalty_mm": None,
                    "affinity_penalty_mm": None,
                    "relation_penalty_mm": None,
                    "spread_penalty_mm": None,
                    "min_cluster_gap_mm": None,
                    "avg_cluster_gap_mm": None,
                    "tight_gap_pairs": [],
                    "orientation_debug": [],
                }

            # Phải tốt hơn valid state hiện tại thật sự.
            if not _is_better_valid_candidate(
                new_score=post_layout_score,
                new_quality=post_quality,
                old_score=base_score,
                old_quality=base_quality,
            ):
                continue

            x = cand.get("x")
            y = cand.get("y")
            rot = cand.get("rot")
            if not all(isinstance(v, (int, float)) for v in (x, y, rot)):
                # fallback từ patch_clusters
                only_patch = patch_clusters[0]
                x = only_patch["x"]
                y = only_patch["y"]
                rot = only_patch["rot"]

            cost = int(cand.get("cost", 10**9) or 10**9)

            patch = {
                "cluster_id": cid,
                "trigger": "IMPROVE_VALID",
                "with_id": None,
                "x": int(x),
                "y": int(y),
                "rot": int(rot) % 360,
                "reason": cand.get("reason", "IMPROVE_VALID"),
                "cost": cost,
                "postcheck": postcheck,
            }

            patch_key = _global_patch_key(patch)
            if patch_key in tried_patch_keys:
                continue

            patched = _apply_global_patch(transforms, patch)
            patched_state_key = _state_key(patched)
            if patched_state_key in seen_state_keys:
                continue

            # _quality_rank: tuple càng lớn càng tốt
            quality_rank = _quality_rank(post_layout_score, post_quality)

            rank = (
                *quality_rank,
                -float(cost),  # cost nhỏ hơn tốt hơn
                -float(hint_idx),
                -float(cand_idx),
            )
            candidates.append((rank, patch))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _select_single_global_invalid_improve_patch(
    *,
    tool_output: dict[str, Any],
    transforms: list[dict[str, Any]],
    seen_state_keys: set[str],
    tried_patch_keys: set[str],
    blocked_cluster_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    improve_hints = tool_output.get("improve_hints")
    if not isinstance(improve_hints, list) or not improve_hints:
        return None

    blocked_cluster_ids = blocked_cluster_ids or set()
    base_quality = tool_output.get("quality")
    base_quality = base_quality if isinstance(base_quality, dict) else {}
    base_global_orientation = int(
        base_quality.get("orientation_penalty_mm", 10**9) or 10**9
    )

    candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    for hint_idx, hint in enumerate(improve_hints):
        if not isinstance(hint, dict):
            continue
        cid = hint.get("cluster_id")
        hint_candidates = hint.get("candidates")
        base_cluster_pen = int(
            hint.get("baseline_cluster_orientation_penalty_mm", 10**9) or 10**9
        )

        if not isinstance(cid, str) or not cid:
            continue
        if cid in blocked_cluster_ids:
            continue
        if not isinstance(hint_candidates, list):
            continue

        for cand_idx, cand in enumerate(hint_candidates):
            if not isinstance(cand, dict):
                continue

            x = cand.get("x")
            y = cand.get("y")
            rot = cand.get("rot")
            if not all(isinstance(v, (int, float)) for v in (x, y, rot)):
                continue

            postcheck = cand.get("postcheck")
            postcheck = postcheck if isinstance(postcheck, dict) else {}

            hard_score = int(
                postcheck.get(
                    "hard_violation_score", postcheck.get("violation_score", 10**9)
                )
            )
            if hard_score != 0:
                continue
            if not bool(postcheck.get("all_hard_clear", True)):
                continue

            cluster_orient_pen = int(
                postcheck.get("cluster_orientation_penalty_mm", 10**9) or 10**9
            )
            global_orient_pen = int(
                postcheck.get("global_orientation_penalty_mm", 10**9) or 10**9
            )
            soft_pen = int(postcheck.get("soft_penalty_mm", 10**9) or 10**9)
            cost = int(cand.get("cost", 10**9))

            improved = cluster_orient_pen < base_cluster_pen or (
                cluster_orient_pen == base_cluster_pen
                and global_orient_pen < base_global_orientation
            )
            if not improved:
                continue

            patch = {
                "cluster_id": cid,
                "trigger": "INVALID_ORIENTATION_IMPROVE",
                "with_id": None,
                "x": int(x),
                "y": int(y),
                "rot": int(rot) % 360,
                "reason": cand.get("reason", "INVALID_ORIENTATION_IMPROVE"),
                "cost": cost,
                "postcheck": postcheck,
            }

            patch_key = _global_patch_key(patch)
            if patch_key in tried_patch_keys:
                continue

            patched = _apply_global_patch(transforms, patch)
            patched_state_key = _state_key(patched)
            if patched_state_key in seen_state_keys:
                continue

            score = (
                cluster_orient_pen,
                global_orient_pen,
                soft_pen,
                cost,
                hint_idx,
                cand_idx,
                cid,
            )
            candidates.append((score, patch))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _extract_layout_score(tool_output: dict[str, Any]) -> int | None:
    if not isinstance(tool_output, dict):
        return None

    quality = tool_output.get("quality")
    if isinstance(quality, dict):
        score = quality.get("layout_score")
        if isinstance(score, int):
            return score
        if isinstance(score, float):
            return int(round(score))

    debug = tool_output.get("debug")
    if isinstance(debug, dict):
        q = debug.get("quality")
        if isinstance(q, dict):
            score = q.get("layout_score")
            if isinstance(score, int):
                return score
            if isinstance(score, float):
                return int(round(score))

    return None


def _critical_orientation_metrics(
    quality: dict[str, Any] | None,
) -> tuple[int, int]:
    if not isinstance(quality, dict):
        return (0, 0)

    debug = quality.get("orientation_debug")
    if not isinstance(debug, list):
        return (0, 0)

    critical_count = 0
    worst_penalty = 0

    for item in debug:
        if not isinstance(item, dict):
            continue

        kind = str(item.get("kind") or "")
        relation = str(item.get("relation") or item.get("intent") or "")
        penalty = int(item.get("penalty_mm", 0) or 0)

        is_critical = False

        if kind == "cluster_directional_relation" and relation in {
            "face_each_other",
            "access_faces_other",
            "turn_toward",
        }:
            is_critical = penalty >= 250

        elif kind == "cluster_orientation" and relation in {
            "face_cluster",
            "face_center",
            "face_entry",
            "face_window",
            "inward_to_room",
            "access_to_open_space",
        }:
            is_critical = penalty >= 220

        elif kind == "object_orientation" and relation in {
            "face_object",
            "front_to_open_space",
            "front_to_room_center",
            "front_to_cluster_center",
            "preserve_front_access",
        }:
            is_critical = penalty >= 220

        if is_critical:
            critical_count += 1
            worst_penalty = max(worst_penalty, penalty)

    return critical_count, worst_penalty


def _can_finalize_valid_layout(quality: dict[str, Any] | None) -> bool:
    critical_count, worst_critical = _critical_orientation_metrics(quality)
    if critical_count > 0:
        return False
    if worst_critical >= 250:
        return False
    return True


def _quality_rank(
    score: int | None,
    quality: dict[str, Any] | None,
) -> tuple[float, float, float, float, float, float, float, float, float, float]:
    if not isinstance(quality, dict):
        quality = {}

    critical_count, worst_critical = _critical_orientation_metrics(quality)

    tight_pairs = quality.get("tight_gap_pairs")
    tight_count = len(tight_pairs) if isinstance(tight_pairs, list) else 0

    min_gap = quality.get("min_cluster_gap_mm")
    avg_gap = quality.get("avg_cluster_gap_mm")
    total_soft_penalty = quality.get("total_soft_penalty_mm")
    orientation_penalty = quality.get("orientation_penalty_mm")
    circulation_penalty = quality.get("circulation_penalty_mm")
    affinity_penalty = quality.get("affinity_penalty_mm")

    score_v = float(score) if isinstance(score, int) else -1e12
    min_gap_v = float(min_gap) if isinstance(min_gap, (int, float)) else -1.0
    avg_gap_v = float(avg_gap) if isinstance(avg_gap, (int, float)) else -1.0
    total_soft_v = (
        float(total_soft_penalty)
        if isinstance(total_soft_penalty, (int, float))
        else 1e12
    )
    orientation_v = (
        float(orientation_penalty)
        if isinstance(orientation_penalty, (int, float))
        else 1e12
    )
    circulation_v = (
        float(circulation_penalty)
        if isinstance(circulation_penalty, (int, float))
        else 1e12
    )
    affinity_v = (
        float(affinity_penalty) if isinstance(affinity_penalty, (int, float)) else 1e12
    )

    return (
        score_v,  # ưu tiên layout_score cao nhất
        -float(critical_count),  # tie-break: ít critical orientation hơn
        -float(worst_critical),
        -orientation_v,
        -total_soft_v,
        -circulation_v,
        -affinity_v,
        -float(tight_count),
        min_gap_v,
        avg_gap_v,
    )


def _is_better_valid_candidate(
    *,
    new_score: int | None,
    new_quality: dict[str, Any] | None,
    old_score: int | None,
    old_quality: dict[str, Any] | None,
) -> bool:
    return _quality_rank(new_score, new_quality) > _quality_rank(old_score, old_quality)


def _should_try_soft_improve(
    *,
    quality: dict[str, Any] | None,
    improve_rounds_used: int,
    soft_improve_budget: int,
    no_improve_streak: int,
    min_explore_valid_rounds: int = 2,
) -> bool:
    if improve_rounds_used >= soft_improve_budget:
        return False

    critical_count, worst_critical = _critical_orientation_metrics(quality)

    if critical_count > 0 or worst_critical >= 250:
        return True

    if no_improve_streak >= 2:
        return False

    if improve_rounds_used < min_explore_valid_rounds:
        return True

    if not isinstance(quality, dict):
        return False

    tight_pairs = quality.get("tight_gap_pairs")
    orientation_penalty = quality.get("orientation_penalty_mm")
    circulation_penalty = quality.get("circulation_penalty_mm")
    affinity_penalty = quality.get("affinity_penalty_mm")
    total_soft_penalty = quality.get("total_soft_penalty_mm")

    if isinstance(tight_pairs, list) and len(tight_pairs) > 0:
        return True
    if isinstance(orientation_penalty, (int, float)) and orientation_penalty > 80:
        return True
    if isinstance(circulation_penalty, (int, float)) and circulation_penalty > 120:
        return True
    if isinstance(affinity_penalty, (int, float)) and affinity_penalty > 120:
        return True
    if isinstance(total_soft_penalty, (int, float)) and total_soft_penalty > 0:
        return True

    return False


def _build_best_valid_json(
    *,
    grid_mm: int,
    transforms: list[dict[str, Any]],
    quality: dict[str, Any] | None,
) -> str:
    notes = ["Returned best verified valid layout."]
    if isinstance(quality, dict):
        score = quality.get("layout_score")
        if isinstance(score, (int, float)):
            notes = [
                f"Returned best verified valid layout (layout_score={int(round(score))})."
            ]

    payload = {
        "status": "OK",
        "grid_mm": int(grid_mm),
        "cluster_transforms": _canonicalize_cluster_transforms(transforms),
        "notes": notes,
    }
    return json.dumps(payload, ensure_ascii=True)


def _compact_quality_for_llm(quality: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(quality, dict):
        return {}

    return {
        "layout_score": quality.get("layout_score"),
        "total_soft_penalty_mm": quality.get("total_soft_penalty_mm"),
        "orientation_penalty_mm": quality.get("orientation_penalty_mm"),
        "circulation_penalty_mm": quality.get("circulation_penalty_mm"),
        "affinity_penalty_mm": quality.get("affinity_penalty_mm"),
        "relation_penalty_mm": quality.get("relation_penalty_mm"),
        "spread_penalty_mm": quality.get("spread_penalty_mm"),
        "min_cluster_gap_mm": quality.get("min_cluster_gap_mm"),
        "avg_cluster_gap_mm": quality.get("avg_cluster_gap_mm"),
        "tight_gap_pairs_preview": (
            quality.get("tight_gap_pairs", [])[:3]
            if isinstance(quality.get("tight_gap_pairs"), list)
            else []
        ),
        "orientation_debug_preview": _compact_orientation_preview(
            quality.get("orientation_debug"), max_items=5
        ),
        "circulation_debug_preview": (
            quality.get("circulation_debug", [])[:4]
            if isinstance(quality.get("circulation_debug"), list)
            else []
        ),
        "affinity_debug_preview": (
            quality.get("affinity_debug", [])[:4]
            if isinstance(quality.get("affinity_debug"), list)
            else []
        ),
    }


def _compact_postcheck_for_llm(postcheck: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(postcheck, dict):
        return {}

    q = postcheck.get("quality")
    q = q if isinstance(q, dict) else {}

    return {
        "valid_geom": postcheck.get("valid_geom"),
        "in_room": postcheck.get("in_room"),
        "hit_obstacles": postcheck.get("hit_obstacles", [])[:3]
        if isinstance(postcheck.get("hit_obstacles"), list)
        else [],
        "overlaps": postcheck.get("overlaps", [])[:3]
        if isinstance(postcheck.get("overlaps"), list)
        else [],
        "solves_trigger": postcheck.get("solves_trigger"),
        "hard_violation_score": postcheck.get("hard_violation_score"),
        "violation_score": postcheck.get("violation_score"),
        "remaining_hard_issues": postcheck.get("remaining_hard_issues"),
        "all_hard_clear": postcheck.get("all_hard_clear"),
        "soft_penalty_mm": postcheck.get("soft_penalty_mm"),
        "layout_score": postcheck.get("layout_score"),
        "cluster_orientation_penalty_mm": postcheck.get(
            "cluster_orientation_penalty_mm"
        ),
        "global_orientation_penalty_mm": postcheck.get("global_orientation_penalty_mm"),
        "orientation_penalty_mm": (
            q.get("orientation_penalty_mm")
            if isinstance(q, dict)
            else postcheck.get("global_orientation_penalty_mm")
        ),
    }


def _compact_hint_candidates_for_llm(
    candidates: Any,
    max_items: int = 3,
) -> list[dict[str, Any]]:
    if not isinstance(candidates, list):
        return []

    out: list[dict[str, Any]] = []
    for cand in candidates[:max_items]:
        if not isinstance(cand, dict):
            continue

        payload = {
            "reason": cand.get("reason"),
            "cost": cand.get("cost"),
            "postcheck": _compact_postcheck_for_llm(
                cand.get("postcheck") if isinstance(cand.get("postcheck"), dict) else {}
            ),
        }

        patch_clusters = _candidate_patch_clusters(
            cand,
            default_cluster_id=None,
        )
        if patch_clusters:
            payload["patch_clusters"] = patch_clusters
        else:
            payload.update(
                {
                    "x": cand.get("x"),
                    "y": cand.get("y"),
                    "rot": cand.get("rot"),
                    "dx": cand.get("dx"),
                    "dy": cand.get("dy"),
                }
            )

        out.append(payload)

    return out


def _compact_tool_output_for_llm(tool_output: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(tool_output, dict):
        return {"result": "INVALID", "errors": [{"code": "BAD_TOOL_OUTPUT"}]}

    if tool_output.get("error") is not None:
        return {
            "error": tool_output.get("error"),
            "tool": tool_output.get("tool"),
            "message": tool_output.get("message"),
        }

    errors = tool_output.get("errors")
    errors_preview = errors[:8] if isinstance(errors, list) else []

    repair_hints = tool_output.get("repair_hints")
    improve_hints = tool_output.get("improve_hints")

    compact_repair_hints: list[dict[str, Any]] = []
    if isinstance(repair_hints, list):
        for item in repair_hints[:3]:
            if not isinstance(item, dict):
                continue
            compact_repair_hints.append(
                {
                    "cluster_id": item.get("cluster_id"),
                    "trigger": item.get("trigger"),
                    "with_id": item.get("with_id"),
                    "base_transform": item.get("base_transform"),
                    "candidates": _compact_hint_candidates_for_llm(
                        item.get("candidates"), max_items=3
                    ),
                }
            )

    compact_improve_hints: list[dict[str, Any]] = []
    if isinstance(improve_hints, list):
        for item in improve_hints[:2]:
            if not isinstance(item, dict):
                continue
            compact_improve_hints.append(
                {
                    "cluster_id": item.get("cluster_id"),
                    "trigger": item.get("trigger"),
                    "baseline_cluster_orientation_penalty_mm": item.get(
                        "baseline_cluster_orientation_penalty_mm"
                    ),
                    "baseline_orientation_penalty_mm": item.get(
                        "baseline_orientation_penalty_mm"
                    ),
                    "base_transform": item.get("base_transform"),
                    "candidates": _compact_hint_candidates_for_llm(
                        item.get("candidates"), max_items=2
                    ),
                }
            )

    return {
        "result": tool_output.get("result"),
        "errors": errors_preview,
        "quality": _compact_quality_for_llm(
            tool_output.get("quality")
            if isinstance(tool_output.get("quality"), dict)
            else {}
        ),
        "repair_hints": compact_repair_hints,
        "improve_hints": compact_improve_hints,
    }


def _compact_repair_hints_preview(
    repair_hints: Any,
    *,
    max_items: int = 2,
    max_candidates: int = 2,
) -> list[dict[str, Any]]:
    if not isinstance(repair_hints, list):
        return []

    out: list[dict[str, Any]] = []
    for item in repair_hints[:max_items]:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "cluster_id": item.get("cluster_id"),
                "trigger": item.get("trigger"),
                "with_id": item.get("with_id"),
                "candidates": _compact_hint_candidates_for_llm(
                    item.get("candidates"), max_items=max_candidates
                ),
            }
        )
    return out


def _compact_improve_hints_preview(
    improve_hints: Any,
    *,
    max_items: int = 2,
    max_candidates: int = 2,
) -> list[dict[str, Any]]:
    if not isinstance(improve_hints, list):
        return []

    out: list[dict[str, Any]] = []
    for item in improve_hints[:max_items]:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "cluster_id": item.get("cluster_id"),
                "trigger": item.get("trigger"),
                "baseline_cluster_orientation_penalty_mm": item.get(
                    "baseline_cluster_orientation_penalty_mm"
                ),
                "baseline_orientation_penalty_mm": item.get(
                    "baseline_orientation_penalty_mm"
                ),
                "candidates": _compact_hint_candidates_for_llm(
                    item.get("candidates"), max_items=max_candidates
                ),
            }
        )
    return out


def _compact_orientation_preview(
    orientation_debug: Any,
    *,
    max_items: int = 8,
) -> list[dict[str, Any]]:
    if not isinstance(orientation_debug, list):
        return []

    out: list[dict[str, Any]] = []
    for item in orientation_debug[:max_items]:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "kind": item.get("kind"),
                "cluster_id": item.get("cluster_id"),
                "object_id": item.get("object_id"),
                "a": item.get("a"),
                "b": item.get("b"),
                "intent": item.get("intent"),
                "relation": item.get("relation"),
                "target_cluster_id": item.get("target_cluster_id"),
                "target_object_id": item.get("target_object_id"),
                "resolved_target_cluster_id": item.get("resolved_target_cluster_id"),
                "penalty_mm": item.get("penalty_mm"),
                "dot": item.get("dot"),
            }
        )
    return out


def _prune_messages_block_safe(
    messages: list[dict[str, Any]],
    *,
    prefix_count: int = 3,
    keep_last_blocks: int = 12,
) -> list[dict[str, Any]]:
    if len(messages) <= prefix_count:
        return messages

    prefix = messages[:prefix_count]
    tail = messages[prefix_count:]

    if not tail:
        return messages

    blocks: list[list[dict[str, Any]]] = []
    i = 0
    while i < len(tail):
        msg = tail[i]
        role = msg.get("role")

        if role == "assistant" and msg.get("tool_calls"):
            block = [msg]
            i += 1
            while i < len(tail) and tail[i].get("role") == "tool":
                block.append(tail[i])
                i += 1
            blocks.append(block)
            continue

        blocks.append([msg])
        i += 1

    if len(blocks) <= keep_last_blocks:
        kept_tail = [m for block in blocks for m in block]
        return prefix + kept_tail

    kept_blocks = blocks[-keep_last_blocks:]
    kept_tail = [m for block in kept_blocks for m in block]
    return prefix + kept_tail


def _is_context_length_exceeded(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    if code == "context_length_exceeded":
        return True
    text = str(exc)
    return ("context_length_exceeded" in text) or ("maximum context length" in text)


def _try_parse_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        parsed = json.loads(_extract_json_object(_coerce_json_text(text)))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_json(raw: str) -> dict[str, Any]:
    text = _coerce_json_text(raw)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        text = _extract_json_object(text)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("MacroClusterPlacer returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("MacroClusterPlacer response must be a JSON object")
    return payload


def _coerce_json_text(raw: str) -> str:
    text = raw.strip()
    if not text.startswith("```"):
        return text
    parts = text.split("```")
    for idx in range(1, len(parts), 2):
        candidate = parts[idx].strip()
        if candidate.startswith("json"):
            candidate = candidate[4:].lstrip("\n").strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return candidate
    return text


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start : end + 1]


def _cluster_ids(clusters_outlines: dict[str, Any]) -> list[str]:
    if not isinstance(clusters_outlines, dict):
        return []

    direct_ids = [key for key in clusters_outlines.keys() if isinstance(key, str)]
    direct_cluster_like = []
    for key in direct_ids:
        value = clusters_outlines.get(key)
        if isinstance(value, dict) and (
            "cluster_footprint" in value
            or "outline_polygons_ccw" in value
            or value.get("cluster_id") == key
        ):
            direct_cluster_like.append(key)

    if direct_cluster_like:
        return sorted(direct_cluster_like)

    clusters = clusters_outlines.get("clusters")
    if isinstance(clusters, list):
        out: list[str] = []
        for item in clusters:
            if not isinstance(item, dict):
                continue
            cid = item.get("cluster_id")
            if isinstance(cid, str) and cid:
                out.append(cid)
        return sorted(out)

    return []
