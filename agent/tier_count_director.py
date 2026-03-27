from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from clients.llm_client import get_llm_client
from prompt.tier_count import TIER_COUNT_DIRECTOR, TIER_COUNT_DIRECTOR_SYSTEM
from tier_count.tools import TOOL_REGISTRY, TOOL_SCHEMAS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TierCountDirector:
    system_prompt: str = TIER_COUNT_DIRECTOR_SYSTEM
    prompt_template: str = TIER_COUNT_DIRECTOR

    def generate(
        self,
        *,
        description: str,
        special_notes: str,
        room_model_json: dict[str, Any],
        user_intent_json: dict[str, Any],
        clusters_json: dict[str, Any],
        size_profiles_json: dict[str, Any] | None = None,
        layout_failure_report_json: dict[str, Any] | None = None,
        max_steps: int = 20,
        access_clearance_ratio: float = 0.25,
    ) -> dict[str, Any]:
        clusters_json_clean = _strip_raw_text(
            _unwrap_payload(clusters_json, key="clusters")
        )

        user_prompt = _build_prompt(
            template=self.prompt_template,
            description=description,
            special_notes=special_notes,
            room_model_json=room_model_json,
            user_intent_json=user_intent_json,
            clusters_json=clusters_json_clean,
            size_profiles_json=size_profiles_json,
            layout_failure_report_json=layout_failure_report_json,
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        tenant_id = ""
        if isinstance(user_intent_json, dict):
            tenant_id = str(user_intent_json.get("tenant_id") or "")
        if not tenant_id and isinstance(room_model_json, dict):
            tenant_id = str(room_model_json.get("tenant_id") or "")

        context = {
            "room_model_json": room_model_json,
            "user_intent_json": user_intent_json,
            "special_notes": special_notes,
            "clusters_json": clusters_json_clean,
            "tenant_id": tenant_id or None,
        }

        size_profiles_by_category: dict[str, Any] | None = None
        if isinstance(size_profiles_json, dict):
            sp = size_profiles_json.get("size_profiles_by_category")
            if isinstance(sp, dict):
                size_profiles_by_category = sp

        return _run_with_tools(
            messages=messages,
            context=context,
            max_steps=max_steps,
            size_profiles_by_category=size_profiles_by_category,
        )

    def tools(self) -> list[dict[str, Any]]:
        return list(TOOL_SCHEMAS)


# ============================================================
# Prompt helpers
# ============================================================


def _json_block(obj: Any) -> str:
    if obj is None:
        return "null"
    return json.dumps(obj, ensure_ascii=True, indent=2)


def _build_prompt(
    *,
    template: str,
    description: str,
    special_notes: str,
    room_model_json: dict[str, Any],
    user_intent_json: dict[str, Any],
    clusters_json: dict[str, Any],
    size_profiles_json: dict[str, Any] | None,
    layout_failure_report_json: dict[str, Any] | None,
) -> str:
    mapping = {
        "DESCRIPTION": description or "",
        "SPECIAL_NOTES": special_notes or "",
        "ROOM_MODEL_JSON": _json_block(room_model_json),
        "USER_INTENT_JSON": _json_block(user_intent_json),
        "CLUSTERS_JSON": _json_block(clusters_json),
        "SIZE_PROFILES_JSON": _json_block(size_profiles_json),
        "LAYOUT_FAILURE_REPORT_JSON": _json_block(layout_failure_report_json),
    }
    out = template
    for k, v in mapping.items():
        out = out.replace("{" + k + "}", v)
    return out


def _unwrap_payload(payload: dict[str, Any], *, key: str) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get(key), list):
        return payload
    parsed = payload.get("parsed") if isinstance(payload, dict) else None
    if isinstance(parsed, dict) and isinstance(parsed.get(key), list):
        return parsed
    raw = payload.get("raw") if isinstance(payload, dict) else None
    if isinstance(raw, dict) and isinstance(raw.get(key), list):
        return raw
    return payload


def _strip_raw_text(payload: Any) -> Any:
    if isinstance(payload, dict):
        payload = dict(payload)
        payload.pop("raw_text", None)
        for k, v in list(payload.items()):
            payload[k] = _strip_raw_text(v)
        return payload
    if isinstance(payload, list):
        return [_strip_raw_text(x) for x in payload]
    return payload


# ============================================================
# Core tool loop
# ============================================================


def _run_with_tools(
    *,
    messages: list[dict[str, Any]],
    context: dict[str, Any],
    max_steps: int,
    size_profiles_by_category: dict[str, Any] | None,
) -> dict[str, Any]:
    client = get_llm_client()

    clusters_json = context.get("clusters_json")
    required_types = _extract_member_types(clusters_json)
    members_by_cluster = _extract_members_by_cluster(clusters_json)
    anchors_by_cluster = _extract_anchors_by_cluster(clusters_json)
    droppable_clusters = _extract_droppable_clusters(clusters_json)

    last_draft: dict[str, Any] | None = None
    frozen_cluster_budget_limits_m2: dict[str, float] | None = None
    rescue_mode_used = False

    for step in range(max_steps):
        logger.info("TierCount step %s/%s", step + 1, max_steps)

        response = client.chat_completion(
            messages,
            model_key="primary",
            temperature=0.0,
            max_tokens=None,
            tools=TOOL_SCHEMAS,
        )

        message = _extract_message(response)
        tool_calls = _extract_tool_calls(message)
        content = getattr(message, "content", "") or ""

        draft = _try_parse_json_object(content)
        if isinstance(draft, dict):
            last_draft = draft

        # ----------------------------------------------------
        # TOOL CALLS
        # ----------------------------------------------------
        if tool_calls:
            logger.info(
                "Tool calls requested: %s",
                [(c.get("function", {}) or {}).get("name") for c in tool_calls],
            )

            prepared_calls: list[dict[str, Any]] = []
            preflight_error_message: str | None = None

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
                            "name": str(name),
                            "args": args,
                            "kind": "unknown_tool",
                        }
                    )
                    continue

                if name == "get_size_profiles":
                    args = _coerce_get_size_profiles_args(
                        args=args,
                        context=context,
                        last_draft=last_draft,
                        required_types=required_types,
                    )

                    if (
                        not isinstance(args.get("categories"), list)
                        or not args["categories"]
                    ):
                        logger.warning(
                            "get_size_profiles has no categories after coercion. last_draft=%s required_types=%s",
                            isinstance(last_draft, dict),
                            required_types,
                        )
                        preflight_error_message = (
                            "Before calling get_size_profiles, provide categories or decisions. "
                            "You may include a JSON draft with decisions first, or call get_size_profiles "
                            f"with categories derived from cluster members: {required_types}"
                        )
                        break

                elif name == "estimate_budget":
                    args = _coerce_estimate_budget_args(
                        args=args,
                        context=context,
                        size_profiles_by_category=size_profiles_by_category,
                        last_draft=last_draft,
                        frozen_cluster_budget_limits_m2=frozen_cluster_budget_limits_m2,
                    )

                    ok, detail = _validate_decisions(
                        args.get("decisions"),
                        required_types,
                        members_by_cluster=members_by_cluster,
                        anchors_by_cluster=anchors_by_cluster,
                        droppable_clusters=droppable_clusters,
                    )
                    if not ok:
                        logger.warning(
                            "estimate_budget preflight failed: detail=%s | args_keys=%s | has_last_draft=%s",
                            detail,
                            sorted(list(args.keys())),
                            isinstance(last_draft, dict),
                        )
                        preflight_error_message = (
                            "Before calling estimate_budget, you MUST provide a valid decisions list. "
                            "Requirement: exactly one decision per member type across all clusters, "
                            "quantity must be an integer >= 0, and each CORE cluster must keep at least one anchor "
                            "(quantity >= 1, using CLUSTERS_JSON.anchors when present). "
                            "DROPPABLE clusters (for example misc/decor/accent/accessory) may be reduced to all quantities = 0. "
                            f"Issue: {detail}. "
                            "You can either include decisions in the tool arguments, or output a JSON draft "
                            "with decisions first so the controller can reuse it."
                        )
                        break

                    if not isinstance(args.get("size_profiles_by_category"), dict):
                        cats = _extract_categories_from_decisions(
                            {"decisions": args.get("decisions")}
                        )
                        if not cats:
                            cats = required_types[:]

                        if cats:
                            logger.info(
                                "Auto-fetching size profiles before estimate_budget for categories=%s",
                                cats,
                            )
                            sp_out = TOOL_REGISTRY["get_size_profiles"](
                                categories=cats,
                                tenant_id=context.get("tenant_id"),
                            )
                            if _tool_output_has_error(sp_out):
                                raise RuntimeError(
                                    f"get_size_profiles failed during preflight: {_tool_error_text(sp_out)}"
                                )
                            sp = sp_out.get("size_profiles_by_category")
                            if isinstance(sp, dict):
                                size_profiles_by_category = sp
                                args["size_profiles_by_category"] = sp

                    if not isinstance(args.get("size_profiles_by_category"), dict):
                        logger.warning(
                            "estimate_budget still missing size_profiles_by_category after auto-fetch"
                        )
                        preflight_error_message = (
                            "Size profiles are still missing, so estimate_budget cannot run. "
                            "Call get_size_profiles first, or provide size_profiles_by_category."
                        )
                        break

                prepared_calls.append(
                    {
                        "call_id": call_id,
                        "name": name,
                        "args": args,
                        "kind": "normal",
                    }
                )

            if preflight_error_message is not None:
                messages.append({"role": "user", "content": preflight_error_message})
                continue

            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                }
            )

            for item in prepared_calls:
                call_id = item["call_id"]
                name = item["name"]
                args = item["args"]
                kind = item["kind"]

                if kind == "unknown_tool":
                    tool_output = {
                        "error": "unknown_tool",
                        "tool": name,
                        "args": args,
                    }
                else:
                    tool_output = _safe_run_tool(name, args)

                logger.info(
                    "Tool %s output: %s",
                    name,
                    json.dumps(tool_output, ensure_ascii=True),
                )

                if _tool_output_has_error(tool_output):
                    raise RuntimeError(
                        f"{name} failed: {_tool_error_text(tool_output)}"
                    )

                if name == "get_size_profiles":
                    sp = tool_output.get("size_profiles_by_category")
                    if isinstance(sp, dict):
                        size_profiles_by_category = sp

                if (
                    name == "estimate_budget"
                    and frozen_cluster_budget_limits_m2 is None
                    and isinstance(tool_output.get("cluster_budget_limits_m2"), dict)
                    and not bool(tool_output.get("rescue_mode", False))
                ):
                    frozen_cluster_budget_limits_m2 = tool_output[
                        "cluster_budget_limits_m2"
                    ]

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": str(name),
                        "content": json.dumps(tool_output, ensure_ascii=True),
                    }
                )

                if name != "estimate_budget":
                    continue

                active_budget_out = tool_output
                active_decisions = args.get("decisions")
                if not isinstance(active_decisions, list):
                    active_decisions = []

                input_fit = bool(active_budget_out.get("input_decisions_fit", False))
                recommended_fit = bool(
                    active_budget_out.get("recommended_decisions_fit", False)
                )
                repair_exhausted = _tool_reports_repair_exhausted(active_budget_out)
                hard_unsat = _tool_reports_hard_unsat(active_budget_out)

                repair_hints = _preview_budget_repair_hints(active_budget_out)
                violations_after = active_budget_out.get("violations_after_fit") or []
                recommended_decisions = active_budget_out.get("recommended_decisions")

                if (
                    not input_fit
                    and not recommended_fit
                    and not rescue_mode_used
                    and repair_exhausted
                ):
                    rescue_args = dict(args)
                    if (
                        isinstance(recommended_decisions, list)
                        and recommended_decisions
                    ):
                        rescue_args["decisions"] = recommended_decisions
                    rescue_args["rescue_mode"] = True
                    rescue_args.pop("frozen_cluster_budget_limits_m2", None)

                    rescue_budget_out = _safe_run_tool("estimate_budget", rescue_args)
                    if _tool_output_has_error(rescue_budget_out):
                        raise RuntimeError(
                            f"estimate_budget rescue failed: {_tool_error_text(rescue_budget_out)}"
                        )

                    rescue_mode_used = True

                    logger.info(
                        "Rescue estimate_budget output: %s",
                        json.dumps(rescue_budget_out, ensure_ascii=True),
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": f"{call_id}_rescue",
                            "name": "estimate_budget",
                            "content": json.dumps(rescue_budget_out, ensure_ascii=True),
                        }
                    )

                    active_budget_out = rescue_budget_out
                    input_fit = bool(
                        active_budget_out.get("input_decisions_fit", False)
                    )
                    recommended_fit = bool(
                        active_budget_out.get("recommended_decisions_fit", False)
                    )
                    repair_exhausted = _tool_reports_repair_exhausted(active_budget_out)
                    hard_unsat = _tool_reports_hard_unsat(active_budget_out)

                    repair_hints = _preview_budget_repair_hints(active_budget_out)
                    violations_after = (
                        active_budget_out.get("violations_after_fit") or []
                    )
                    recommended_decisions = active_budget_out.get(
                        "recommended_decisions"
                    )

                if input_fit:
                    result = _build_ok_result(
                        decisions=active_decisions,
                        base_draft=last_draft,
                        size_profiles_by_category=size_profiles_by_category,
                        budget_mode="rescue"
                        if bool(active_budget_out.get("rescue_mode", False))
                        else None,
                    )
                    return result

                if recommended_fit and isinstance(recommended_decisions, list):
                    result = _build_ok_result(
                        decisions=recommended_decisions,
                        base_draft=last_draft,
                        size_profiles_by_category=size_profiles_by_category,
                        budget_mode="rescue"
                        if bool(active_budget_out.get("rescue_mode", False))
                        else None,
                    )
                    return result

                if hard_unsat:
                    result = _build_unsat_result(
                        base_draft=last_draft,
                        fallback_decisions=recommended_decisions
                        if isinstance(recommended_decisions, list)
                        else active_decisions,
                        budget_out=active_budget_out,
                        size_profiles_by_category=size_profiles_by_category,
                    )
                    return result

                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Budget check still fails. "
                            "You must continue revising until input_decisions_fit=true. "
                            "Use the recommended_decisions below as the new base draft and make stronger minimal reductions if needed "
                            "(you may change multiple non-anchor items in one revision). "
                            f"Remaining violations: {json.dumps(_preview_budget_violations(violations_after), ensure_ascii=True)}. "
                            f"Recommended decisions: {json.dumps(recommended_decisions, ensure_ascii=True)}. "
                            f"Repair hints: {json.dumps(repair_hints, ensure_ascii=True)}"
                        ),
                    }
                )

            continue

        # ----------------------------------------------------
        # NO TOOL CALLS => EXPECT FINAL JSON
        # ----------------------------------------------------
        if not isinstance(content, str) or not content.strip():
            messages.append(
                {"role": "user", "content": "Return a JSON object only (no markdown)."}
            )
            continue

        logger.info("Assistant content length: %s", len(content))

        try:
            result = _parse_json(content)
            status = str(result.get("status") or "").upper()
        except ValueError:
            messages.append(
                {
                    "role": "user",
                    "content": "Your output must be valid JSON only (no markdown, no prose).",
                }
            )
            continue

        # IMPORTANT FIX:
        # Only return NEED_INFO if it truly has no valid decision set to continue with.
        if status == "NEED_INFO":
            ok_needinfo, _ = _validate_decisions(
                result.get("decisions"),
                required_types,
                members_by_cluster=members_by_cluster,
                anchors_by_cluster=anchors_by_cluster,
                droppable_clusters=droppable_clusters,
            )
            if not ok_needinfo:
                return result
            result["status"] = "OK"
            status = "OK"

        ok, detail = _validate_decisions(
            result.get("decisions"),
            required_types,
            members_by_cluster=members_by_cluster,
            anchors_by_cluster=anchors_by_cluster,
            droppable_clusters=droppable_clusters,
        )
        if not ok:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Decisions are required before final output. "
                        "You MUST output exactly one decision per member type across all clusters, "
                        "quantity must be an integer >= 0, and each CORE cluster must keep at least one anchor "
                        "(quantity >= 1, using CLUSTERS_JSON.anchors when present). "
                        "DROPPABLE clusters (for example misc/decor/accent/accessory) may be reduced to all quantities = 0. "
                        f"Issue: {detail}"
                    ),
                }
            )
            continue

        if not isinstance(size_profiles_by_category, dict):
            cats = _extract_categories_from_decisions(result)
            if not cats:
                cats = required_types[:]
            if cats:
                logger.info(
                    "Auto-fetching size profiles before finalization for categories=%s",
                    cats,
                )
                sp_out = TOOL_REGISTRY["get_size_profiles"](
                    categories=cats,
                    tenant_id=context.get("tenant_id"),
                )
                if _tool_output_has_error(sp_out):
                    raise RuntimeError(
                        f"get_size_profiles failed during finalization: {_tool_error_text(sp_out)}"
                    )
                sp = sp_out.get("size_profiles_by_category")
                if isinstance(sp, dict):
                    size_profiles_by_category = sp

        if not isinstance(size_profiles_by_category, dict):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Unable to finalize because size profiles are missing. "
                        "Call get_size_profiles first or provide size_profiles_by_category."
                    ),
                }
            )
            continue

        if status == "UNSAT":
            budget_out_unsat = _run_budget_check_for_result(
                result=result,
                context=context,
                size_profiles_by_category=size_profiles_by_category,
                frozen_cluster_budget_limits_m2=frozen_cluster_budget_limits_m2,
                rescue_mode=False,
            )
            if _tool_output_has_error(budget_out_unsat):
                raise RuntimeError(
                    f"estimate_budget failed while verifying UNSAT: {_tool_error_text(budget_out_unsat)}"
                )
            if _tool_reports_hard_unsat(budget_out_unsat):
                return _build_unsat_result(
                    base_draft=result,
                    fallback_decisions=result.get("decisions", []),
                    budget_out=budget_out_unsat,
                    size_profiles_by_category=size_profiles_by_category,
                )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Do not stop with status='UNSAT' yet. "
                        "Continue revising unless the budget tool explicitly confirms true infeasibility."
                    ),
                }
            )
            continue

        budget_out = _run_budget_check_for_result(
            result=result,
            context=context,
            size_profiles_by_category=size_profiles_by_category,
            frozen_cluster_budget_limits_m2=frozen_cluster_budget_limits_m2,
            rescue_mode=False,
        )
        if _tool_output_has_error(budget_out):
            raise RuntimeError(
                f"estimate_budget failed during finalization: {_tool_error_text(budget_out)}"
            )

        logger.info(
            "Finalization budget check output: %s",
            json.dumps(budget_out, ensure_ascii=True),
        )

        if (
            frozen_cluster_budget_limits_m2 is None
            and isinstance(budget_out.get("cluster_budget_limits_m2"), dict)
            and not bool(budget_out.get("rescue_mode", False))
        ):
            frozen_cluster_budget_limits_m2 = budget_out["cluster_budget_limits_m2"]

        input_fit = bool(budget_out.get("input_decisions_fit", False))
        recommended_fit = bool(budget_out.get("recommended_decisions_fit", False))
        repair_exhausted = _tool_reports_repair_exhausted(budget_out)
        hard_unsat = _tool_reports_hard_unsat(budget_out)

        if input_fit:
            return _build_ok_result(
                decisions=result.get("decisions", []),
                base_draft=result,
                size_profiles_by_category=size_profiles_by_category,
                budget_mode=None,
            )

        recommended_decisions = budget_out.get("recommended_decisions")
        if recommended_fit and isinstance(recommended_decisions, list):
            return _build_ok_result(
                decisions=recommended_decisions,
                base_draft=result,
                size_profiles_by_category=size_profiles_by_category,
                budget_mode=None,
            )

        if repair_exhausted and not rescue_mode_used:
            rescue_budget_out = _run_budget_check_for_result(
                result=result,
                context=context,
                size_profiles_by_category=size_profiles_by_category,
                frozen_cluster_budget_limits_m2=None,
                rescue_mode=True,
            )
            if _tool_output_has_error(rescue_budget_out):
                raise RuntimeError(
                    f"estimate_budget rescue failed during finalization: {_tool_error_text(rescue_budget_out)}"
                )

            rescue_mode_used = True
            logger.info(
                "Finalization rescue budget check output: %s",
                json.dumps(rescue_budget_out, ensure_ascii=True),
            )

            rescue_input_fit = bool(rescue_budget_out.get("input_decisions_fit", False))
            rescue_recommended_fit = bool(
                rescue_budget_out.get("recommended_decisions_fit", False)
            )
            rescue_hard_unsat = _tool_reports_hard_unsat(rescue_budget_out)
            rescue_recommended_decisions = rescue_budget_out.get(
                "recommended_decisions"
            )

            if rescue_input_fit:
                return _build_ok_result(
                    decisions=result.get("decisions", []),
                    base_draft=result,
                    size_profiles_by_category=size_profiles_by_category,
                    budget_mode="rescue",
                )

            if rescue_recommended_fit and isinstance(
                rescue_recommended_decisions, list
            ):
                return _build_ok_result(
                    decisions=rescue_recommended_decisions,
                    base_draft=result,
                    size_profiles_by_category=size_profiles_by_category,
                    budget_mode="rescue",
                )

            if rescue_hard_unsat:
                return _build_unsat_result(
                    base_draft=result,
                    fallback_decisions=rescue_recommended_decisions
                    if isinstance(rescue_recommended_decisions, list)
                    else result.get("decisions", []),
                    budget_out=rescue_budget_out,
                    size_profiles_by_category=size_profiles_by_category,
                )

            budget_out = rescue_budget_out
            recommended_decisions = budget_out.get("recommended_decisions")

        if hard_unsat:
            return _build_unsat_result(
                base_draft=result,
                fallback_decisions=recommended_decisions
                if isinstance(recommended_decisions, list)
                else result.get("decisions", []),
                budget_out=budget_out,
                size_profiles_by_category=size_profiles_by_category,
            )

        violations_after = budget_out.get("violations_after_fit") or []
        repair_hints = _preview_budget_repair_hints(budget_out)

        messages.append(
            {
                "role": "user",
                "content": (
                    "Your current decisions still fail budget. "
                    "Do not return UNSAT. Continue revising until input_decisions_fit=true. "
                    "Use the recommended_decisions below as the next base draft and make stronger minimal corrections if needed "
                    "(you may reduce several non-anchor items in one revision). "
                    f"Remaining violations: {json.dumps(_preview_budget_violations(violations_after), ensure_ascii=True)}. "
                    f"Recommended decisions: {json.dumps(recommended_decisions, ensure_ascii=True)}. "
                    f"Repair hints: {json.dumps(repair_hints, ensure_ascii=True)}"
                ),
            }
        )

    # --------------------------------------------------------
    # Timeout fallback
    # --------------------------------------------------------
    if isinstance(last_draft, dict) and isinstance(size_profiles_by_category, dict):
        try:
            budget_out = _run_budget_check_for_result(
                result=last_draft,
                context=context,
                size_profiles_by_category=size_profiles_by_category,
                frozen_cluster_budget_limits_m2=frozen_cluster_budget_limits_m2,
                rescue_mode=False,
            )
            if _tool_output_has_error(budget_out):
                raise RuntimeError(_tool_error_text(budget_out))

            logger.info(
                "Timeout fallback budget check output: %s",
                json.dumps(budget_out, ensure_ascii=True),
            )

            if bool(budget_out.get("input_decisions_fit", False)):
                return _build_ok_result(
                    decisions=last_draft.get("decisions", []),
                    base_draft=last_draft,
                    size_profiles_by_category=size_profiles_by_category,
                    budget_mode=None,
                )

            recommended_decisions = budget_out.get("recommended_decisions")
            if bool(budget_out.get("recommended_decisions_fit", False)) and isinstance(
                recommended_decisions, list
            ):
                return _build_ok_result(
                    decisions=recommended_decisions,
                    base_draft=last_draft,
                    size_profiles_by_category=size_profiles_by_category,
                    budget_mode=None,
                )

            rescue_budget_out = _run_budget_check_for_result(
                result=last_draft,
                context=context,
                size_profiles_by_category=size_profiles_by_category,
                frozen_cluster_budget_limits_m2=None,
                rescue_mode=True,
            )
            if _tool_output_has_error(rescue_budget_out):
                raise RuntimeError(_tool_error_text(rescue_budget_out))

            logger.info(
                "Timeout fallback rescue budget check output: %s",
                json.dumps(rescue_budget_out, ensure_ascii=True),
            )

            if bool(rescue_budget_out.get("input_decisions_fit", False)):
                return _build_ok_result(
                    decisions=last_draft.get("decisions", []),
                    base_draft=last_draft,
                    size_profiles_by_category=size_profiles_by_category,
                    budget_mode="rescue",
                )

            rescue_recommended_decisions = rescue_budget_out.get(
                "recommended_decisions"
            )
            if bool(
                rescue_budget_out.get("recommended_decisions_fit", False)
            ) and isinstance(rescue_recommended_decisions, list):
                return _build_ok_result(
                    decisions=rescue_recommended_decisions,
                    base_draft=last_draft,
                    size_profiles_by_category=size_profiles_by_category,
                    budget_mode="rescue",
                )

            if _tool_reports_hard_unsat(rescue_budget_out):
                return _build_unsat_result(
                    base_draft=last_draft,
                    fallback_decisions=rescue_recommended_decisions
                    if isinstance(rescue_recommended_decisions, list)
                    else last_draft.get("decisions", []),
                    budget_out=rescue_budget_out,
                    size_profiles_by_category=size_profiles_by_category,
                )
        except Exception as exc:
            logger.warning("Timeout fallback failed: %s", exc)

    raise TimeoutError(
        "TierCountDirector exceeded max_steps without reaching a budget-valid final JSON."
    )


# ============================================================
# Tool / args helpers
# ============================================================


def _safe_json_loads(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _safe_run_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    try:
        return TOOL_REGISTRY[name](**args)
    except TypeError as exc:
        return {
            "error": "invalid_tool_arguments",
            "tool": name,
            "message": str(exc),
            "args": args,
        }
    except Exception as exc:
        return {"error": "tool_failed", "tool": name, "message": str(exc), "args": args}


def _tool_output_has_error(tool_output: dict[str, Any]) -> bool:
    return isinstance(tool_output, dict) and isinstance(tool_output.get("error"), str)


def _tool_error_text(tool_output: dict[str, Any]) -> str:
    if not isinstance(tool_output, dict):
        return "unknown tool error"
    return json.dumps(tool_output, ensure_ascii=True)


def _coerce_get_size_profiles_args(
    *,
    args: dict[str, Any],
    context: dict[str, Any],
    last_draft: dict[str, Any] | None,
    required_types: list[str],
) -> dict[str, Any]:
    out = dict(args)

    # get_size_profiles expects furniture/object categories, not cluster tags (e.g. living/storage/misc).
    # Use CLUSTERS_JSON members as the primary source of truth when available.
    if required_types:
        cluster_tags: set[str] = set()
        clusters_json = context.get("clusters_json")
        if isinstance(clusters_json, dict):
            clusters = clusters_json.get("clusters")
            if isinstance(clusters, list):
                for cluster in clusters:
                    if not isinstance(cluster, dict):
                        continue
                    tag = cluster.get("tag")
                    if isinstance(tag, str) and tag.strip():
                        cluster_tags.add(tag.strip().lower())

        provided = out.get("categories")
        provided_cats: list[str] = []
        if isinstance(provided, list):
            for c in provided:
                if isinstance(c, str) and c.strip():
                    provided_cats.append(c.strip())

        deduped: list[str] = []
        seen: set[str] = set()
        for c in provided_cats:
            key = c.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(c)

        required_set = {t.lower() for t in required_types if t}
        filtered: list[str] = []
        dropped: list[str] = []
        for c in deduped:
            ck = c.lower()
            if ck in cluster_tags and ck not in required_set:
                dropped.append(c)
                continue
            filtered.append(c)

        if dropped:
            logger.info(
                "Dropping cluster tags from get_size_profiles.categories: dropped=%s kept=%s",
                dropped,
                filtered,
            )

        if filtered and all(c.lower() in required_set for c in filtered):
            out["categories"] = filtered
        else:
            if filtered:
                logger.info(
                    "Ignoring get_size_profiles.categories not present in CLUSTERS_JSON.members: %s",
                    filtered,
                )
            out["categories"] = required_types[:]
    else:
        cats = out.get("categories")
        if not isinstance(cats, list) or not cats:
            cats = []
            if isinstance(last_draft, dict):
                cats = _extract_categories_from_decisions(last_draft)
            if not cats:
                cats = required_types[:]
            out["categories"] = cats

    if "tenant_id" not in out:
        out["tenant_id"] = context.get("tenant_id")

    return out


def _coerce_estimate_budget_args(
    *,
    args: dict[str, Any],
    context: dict[str, Any],
    size_profiles_by_category: dict[str, Any] | None,
    last_draft: dict[str, Any] | None,
    frozen_cluster_budget_limits_m2: dict[str, float] | None,
) -> dict[str, Any]:
    out = dict(args)

    if not isinstance(out.get("decisions"), list):
        if isinstance(last_draft, dict) and isinstance(
            last_draft.get("decisions"), list
        ):
            out["decisions"] = last_draft["decisions"]

    if "room_model" not in out:
        out["room_model"] = context.get("room_model_json", {})
    if isinstance(out.get("room_model"), str):
        parsed = _safe_json_loads(out["room_model"])
        if parsed:
            out["room_model"] = parsed

    if "size_profiles_by_category" not in out and isinstance(
        size_profiles_by_category, dict
    ):
        out["size_profiles_by_category"] = size_profiles_by_category
    if isinstance(out.get("size_profiles_by_category"), str):
        parsed = _safe_json_loads(out["size_profiles_by_category"])
        if parsed:
            out["size_profiles_by_category"] = parsed

    if "style" not in out:
        style = ""
        user_intent = context.get("user_intent_json") or {}
        if isinstance(user_intent, dict):
            ui = user_intent.get("user_input") or {}
            if isinstance(ui, dict):
                style = str(ui.get("style") or "")
        if not style:
            meta = (context.get("room_model_json") or {}).get("meta") or {}
            if isinstance(meta, dict):
                style = str(meta.get("style") or "")
        out["style"] = style

    if "user_notes" not in out:
        out["user_notes"] = str(context.get("special_notes") or "")

    if "clusters_json" not in out:
        out["clusters_json"] = context.get("clusters_json")
    if isinstance(out.get("clusters_json"), str):
        parsed = _safe_json_loads(out["clusters_json"])
        if parsed:
            out["clusters_json"] = parsed

    rescue_mode = bool(out.get("rescue_mode", False))
    if (
        "frozen_cluster_budget_limits_m2" not in out
        and isinstance(frozen_cluster_budget_limits_m2, dict)
        and not rescue_mode
    ):
        out["frozen_cluster_budget_limits_m2"] = frozen_cluster_budget_limits_m2

    return out


def _run_budget_check_for_result(
    *,
    result: dict[str, Any],
    context: dict[str, Any],
    size_profiles_by_category: dict[str, Any],
    frozen_cluster_budget_limits_m2: dict[str, float] | None,
    rescue_mode: bool,
) -> dict[str, Any]:
    args = {
        "room_model": context.get("room_model_json", {}),
        "decisions": result.get("decisions", []),
        "size_profiles_by_category": size_profiles_by_category,
        "style": _infer_style(context),
        "user_notes": str(context.get("special_notes") or ""),
        "clusters_json": context.get("clusters_json"),
        "rescue_mode": rescue_mode,
    }
    if frozen_cluster_budget_limits_m2 is not None and not rescue_mode:
        args["frozen_cluster_budget_limits_m2"] = frozen_cluster_budget_limits_m2
    return _safe_run_tool("estimate_budget", args)


def _infer_style(context: dict[str, Any]) -> str:
    user_intent = context.get("user_intent_json") or {}
    if isinstance(user_intent, dict):
        ui = user_intent.get("user_input") or {}
        if isinstance(ui, dict):
            style = str(ui.get("style") or "")
            if style:
                return style

    room_model = context.get("room_model_json") or {}
    if isinstance(room_model, dict):
        meta = room_model.get("meta") or {}
        if isinstance(meta, dict):
            style = str(meta.get("style") or "")
            if style:
                return style

    return ""


# ============================================================
# Parsing helpers
# ============================================================


def _extract_message(response: object) -> object:
    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        return getattr(choices[0], "message", None)
    raise ValueError("OpenAI response missing message")


def _extract_tool_calls(message: object) -> list[dict[str, Any]]:
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        return []
    out: list[dict[str, Any]] = []
    for call in tool_calls:
        if isinstance(call, dict):
            out.append(call)
            continue
        fn = getattr(call, "function", None)
        out.append(
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
    return out


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
            raise ValueError("TierCountDirector returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("TierCountDirector response must be a JSON object")
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


# ============================================================
# Draft / result helpers
# ============================================================


def _merge_recommended_decisions_into_draft(
    draft: dict[str, Any] | None,
    recommended_decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    base = dict(draft) if isinstance(draft, dict) else {}
    base["status"] = "OK"
    base["decisions"] = recommended_decisions
    if not isinstance(base.get("assumptions"), list):
        base["assumptions"] = []
    if not isinstance(base.get("global_notes"), list):
        base["global_notes"] = []
    return base


def _build_ok_result(
    *,
    decisions: list[dict[str, Any]],
    base_draft: dict[str, Any] | None,
    size_profiles_by_category: dict[str, Any] | None,
    budget_mode: str | None,
) -> dict[str, Any]:
    result = _merge_recommended_decisions_into_draft(base_draft, decisions)
    result["status"] = "OK"
    result["budget_valid"] = True
    if budget_mode:
        result["budget_mode"] = budget_mode
    if isinstance(size_profiles_by_category, dict):
        _attach_rep_dims(result, size_profiles_by_category)
    return result


def _build_unsat_result(
    *,
    base_draft: dict[str, Any] | None,
    fallback_decisions: list[dict[str, Any]],
    budget_out: dict[str, Any],
    size_profiles_by_category: dict[str, Any] | None,
) -> dict[str, Any]:
    result = dict(base_draft) if isinstance(base_draft, dict) else {}
    result["status"] = "UNSAT"
    result["decisions"] = (
        fallback_decisions if isinstance(fallback_decisions, list) else []
    )
    if not isinstance(result.get("assumptions"), list):
        result["assumptions"] = []
    if not isinstance(result.get("global_notes"), list):
        result["global_notes"] = []

    core_clusters = budget_out.get("core_minimum_infeasible_clusters")
    blocked_clusters = budget_out.get("repair_blocked_clusters")

    notes = list(result.get("global_notes") or [])
    if isinstance(core_clusters, list) and core_clusters:
        notes.append(
            f"Minimum surviving CORE clusters still exceed budget: {core_clusters}"
        )
    elif isinstance(blocked_clusters, list) and blocked_clusters:
        notes.append(f"Repair blocked clusters: {blocked_clusters}")
    else:
        notes.append("Budget remained infeasible after allowed minimal repairs.")

    result["global_notes"] = notes
    result["budget_valid"] = False
    result["budget_reason"] = "hard_unsat"

    if isinstance(size_profiles_by_category, dict):
        _attach_rep_dims(result, size_profiles_by_category)
    return result


def _tool_reports_repair_exhausted(budget_out: dict[str, Any]) -> bool:
    if bool(budget_out.get("repair_exhausted", False)):
        return True
    return _budget_repair_exhausted(budget_out)


def _tool_reports_hard_unsat(budget_out: dict[str, Any]) -> bool:
    if bool(budget_out.get("core_minimum_infeasible", False)):
        return True
    if bool(budget_out.get("repair_exhausted", False)) and bool(
        budget_out.get("global_repair_blocked", False)
    ):
        return True
    return False


# ============================================================
# Post-process: attach rep_dims_m for composer
# ============================================================


def _attach_rep_dims(
    result: dict[str, Any], size_profiles_by_category: dict[str, Any]
) -> None:
    decisions = result.get("decisions")
    if not isinstance(decisions, list):
        return

    for d in decisions:
        if not isinstance(d, dict):
            continue
        if "rep_dims_m" in d and isinstance(d.get("rep_dims_m"), dict):
            continue

        tier = d.get("size_tier")
        if not isinstance(tier, str) or not tier:
            continue

        category = d.get("category")
        object_type = d.get("object_type")

        profile = None
        if isinstance(category, str) and category in size_profiles_by_category:
            profile = size_profiles_by_category.get(category)
        elif isinstance(object_type, str) and object_type in size_profiles_by_category:
            profile = size_profiles_by_category.get(object_type)

        if not isinstance(profile, dict):
            continue

        rep = (profile.get("rep_dims_m") or {}).get(tier.upper())
        if isinstance(rep, dict):
            d["rep_dims_m"] = rep


def _extract_categories_from_decisions(result: dict[str, Any]) -> list[str]:
    decisions = result.get("decisions")
    if not isinstance(decisions, list):
        return []
    cats: list[str] = []
    for d in decisions:
        if not isinstance(d, dict):
            continue
        for key in ("category", "object_type"):
            v = d.get(key)
            if isinstance(v, str) and v and v not in cats:
                cats.append(v)
    return cats


def _extract_member_types(clusters_json: Any) -> list[str]:
    if not isinstance(clusters_json, dict):
        return []
    clusters = clusters_json.get("clusters")
    if not isinstance(clusters, list):
        return []
    out: list[str] = []
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        members = cluster.get("members")
        if not isinstance(members, list):
            continue
        for m in members:
            if isinstance(m, str) and m and m not in out:
                out.append(m)
    return out


def _extract_members_by_cluster(clusters_json: Any) -> dict[str, set[str]]:
    if not isinstance(clusters_json, dict):
        return {}
    clusters = clusters_json.get("clusters")
    if not isinstance(clusters, list):
        return {}

    out: dict[str, set[str]] = {}
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        cluster_id = cluster.get("cluster_id")
        if not isinstance(cluster_id, str) or not cluster_id:
            continue
        members = cluster.get("members")
        if not isinstance(members, list):
            continue
        out[cluster_id] = {m for m in members if isinstance(m, str) and m}
    return out


def _extract_anchors_by_cluster(clusters_json: Any) -> dict[str, set[str]]:
    if not isinstance(clusters_json, dict):
        return {}
    clusters = clusters_json.get("clusters")
    if not isinstance(clusters, list):
        return {}

    out: dict[str, set[str]] = {}
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        cluster_id = cluster.get("cluster_id")
        if not isinstance(cluster_id, str) or not cluster_id:
            continue
        anchors = cluster.get("anchors")
        if not isinstance(anchors, list):
            out.setdefault(cluster_id, set())
            continue
        out[cluster_id] = {a for a in anchors if isinstance(a, str) and a}
    return out


def _extract_droppable_clusters(clusters_json: Any) -> set[str]:
    if not isinstance(clusters_json, dict):
        return set()

    clusters = clusters_json.get("clusters")
    if not isinstance(clusters, list):
        return set()

    out: set[str] = set()
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue

        cluster_id = cluster.get("cluster_id")
        if not isinstance(cluster_id, str) or not cluster_id:
            continue

        tag = str(cluster.get("tag") or "").strip().lower()
        rules = cluster.get("cluster_rules") or {}
        allow_empty = (
            isinstance(rules, dict) and rules.get("allow_empty_cluster") is True
        )

        if tag in {"misc", "decor", "accent", "accessory"} or allow_empty:
            out.add(cluster_id)

    return out


def _validate_decisions(
    decisions: Any,
    required_types: list[str],
    *,
    members_by_cluster: dict[str, set[str]] | None = None,
    anchors_by_cluster: dict[str, set[str]] | None = None,
    droppable_clusters: set[str] | None = None,
) -> tuple[bool, str]:
    if not isinstance(decisions, list) or not decisions:
        return False, "decisions list is missing or empty"

    types: list[str] = []
    bad_qty: list[str] = []
    bad_cluster_id: list[str] = []
    bad_cluster_assignment: list[str] = []
    unexpected_types: list[str] = []
    decisions_by_cluster: dict[str, dict[str, int]] = {}

    for d in decisions:
        if not isinstance(d, dict):
            continue
        t = d.get("object_type") or d.get("category")
        if not isinstance(t, str) or not t:
            continue

        types.append(t)

        if required_types and t not in required_types:
            unexpected_types.append(t)

        qty = d.get("quantity")
        if not isinstance(qty, int) or qty < 0:
            bad_qty.append(t)

        cluster_id = d.get("cluster_id")
        if not isinstance(cluster_id, str) or not cluster_id:
            bad_cluster_id.append(t)
            continue

        if isinstance(members_by_cluster, dict) and members_by_cluster:
            members = members_by_cluster.get(cluster_id)
            if not isinstance(members, set):
                bad_cluster_assignment.append(f"{t}->{cluster_id}")
            elif t not in members:
                bad_cluster_assignment.append(f"{t}->{cluster_id}")

        decisions_by_cluster.setdefault(cluster_id, {})[t] = (
            qty if isinstance(qty, int) else 0
        )

    if bad_qty:
        return False, f"quantity must be an integer >= 0 for: {sorted(set(bad_qty))}"

    if bad_cluster_id:
        return (
            False,
            f"cluster_id must be a non-empty string for: {sorted(set(bad_cluster_id))}",
        )

    if bad_cluster_assignment:
        return (
            False,
            "each decision must reference a valid cluster_id and a member in that cluster. "
            f"invalid assignments: {sorted(set(bad_cluster_assignment))}",
        )

    if unexpected_types:
        return False, f"unexpected decisions for types: {sorted(set(unexpected_types))}"

    if required_types:
        missing = [t for t in required_types if t not in types]
        if missing:
            return False, f"missing decisions for types: {missing}"

    duplicates = {t for t in types if types.count(t) > 1}
    if duplicates:
        return False, f"multiple decisions found for types: {sorted(duplicates)}"

    droppable_clusters = droppable_clusters or set()

    if isinstance(members_by_cluster, dict) and members_by_cluster:
        for cluster_id in sorted(members_by_cluster.keys()):
            if cluster_id in droppable_clusters:
                continue

            members = members_by_cluster.get(cluster_id) or set()
            anchors = (
                anchors_by_cluster.get(cluster_id, set())
                if isinstance(anchors_by_cluster, dict)
                else set()
            )
            cluster_decisions = decisions_by_cluster.get(cluster_id, {})

            if anchors:
                kept = any(cluster_decisions.get(a, 0) >= 1 for a in anchors)
                if not kept:
                    return (
                        False,
                        f"CORE cluster {cluster_id} must keep at least one anchor with quantity >= 1: {sorted(anchors)}",
                    )
            else:
                kept = any(
                    cluster_decisions.get(m, 0) >= 1
                    for m in members
                    if isinstance(m, str)
                )
                if not kept and members:
                    return (
                        False,
                        f"CORE cluster {cluster_id} must keep at least one member with quantity >= 1",
                    )

    return True, "ok"


def _preview_budget_repair_hints(
    budget_out: dict[str, Any],
    limit: int = 8,
) -> list[dict[str, Any]]:
    details = budget_out.get("decision_footprint_details")
    if not isinstance(details, list):
        return []

    out: list[dict[str, Any]] = []
    for row in details:
        if not isinstance(row, dict):
            continue

        orig_qty = row.get("quantity")
        rec_qty = row.get("recommended_quantity")
        orig_tier = row.get("size_tier")
        rec_tier = row.get("recommended_size_tier")

        changed_qty = (
            isinstance(orig_qty, int)
            and isinstance(rec_qty, int)
            and orig_qty != rec_qty
        )
        changed_tier = (
            isinstance(orig_tier, str)
            and isinstance(rec_tier, str)
            and orig_tier != rec_tier
        )

        if not changed_qty and not changed_tier:
            continue

        out.append(
            {
                "cluster_id": row.get("cluster_id"),
                "category": row.get("category"),
                "priority": row.get("priority"),
                "quantity": orig_qty,
                "recommended_quantity": rec_qty,
                "size_tier": orig_tier,
                "recommended_size_tier": rec_tier,
            }
        )

        if len(out) >= limit:
            break

    return out


def _budget_repair_exhausted(budget_out: dict[str, Any]) -> bool:
    if bool(budget_out.get("recommended_decisions_fit", False)):
        return False

    details = budget_out.get("decision_footprint_details")
    if not isinstance(details, list):
        return False

    for row in details:
        if not isinstance(row, dict):
            continue
        if row.get("quantity") != row.get("recommended_quantity"):
            return False
        if row.get("size_tier") != row.get("recommended_size_tier"):
            return False

    return True


def _preview_budget_violations(
    violations: list[dict[str, Any]], limit: int = 5
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for v in violations[:limit]:
        if not isinstance(v, dict):
            continue
        out.append(
            {
                "cluster_id": v.get("cluster_id"),
                "limit_m2": v.get("limit_m2"),
                "footprint_m2": v.get("footprint_m2"),
                "over_by_m2": v.get("over_by_m2"),
            }
        )
    return out
