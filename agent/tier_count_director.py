from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from clients.openai_client import OpenAIClient
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
            # NOTE: clearance ratio is fixed inside estimate_budget (see tier_count/tools.py)
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
    client = OpenAIClient()

    clusters_json = context.get("clusters_json")
    required_types = _extract_member_types(clusters_json)
    members_by_cluster = _extract_members_by_cluster(clusters_json)
    anchors_by_cluster = _extract_anchors_by_cluster(clusters_json)
    estimate_budget_called = False
    last_draft: dict[str, Any] | None = None

    for step in range(max_steps):
        logger.info("TierCount step %s/%s", step + 1, max_steps)

        response = client.chat_completion(
            messages,
            model_key="gpt_4o",
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

            # -------- PREPARE / PREFLIGHT ALL CALLS FIRST --------
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
                    )

                    ok, detail = _validate_decisions(
                        args.get("decisions"),
                        required_types,
                        members_by_cluster=members_by_cluster,
                        anchors_by_cluster=anchors_by_cluster,
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
                            "quantity must be an integer >= 0, and each cluster must keep at least one anchor "
                            "(quantity >= 1, using CLUSTERS_JSON.anchors when present). "
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

            # -------- DO NOT APPEND assistant tool_calls IF PREFLIGHT FAILED --------
            if preflight_error_message is not None:
                messages.append(
                    {
                        "role": "user",
                        "content": preflight_error_message,
                    }
                )
                continue

            # Safe to append now: every tool_call will receive a tool response
            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                }
            )

            # -------- EXECUTE PREPARED CALLS --------
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

                if name == "get_size_profiles":
                    sp = tool_output.get("size_profiles_by_category")
                    if isinstance(sp, dict):
                        size_profiles_by_category = sp

                if name == "estimate_budget":
                    estimate_budget_called = True

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": str(name),
                        "content": json.dumps(tool_output, ensure_ascii=True),
                    }
                )

                if name == "estimate_budget":
                    violations = tool_output.get("violations")
                    if isinstance(violations, list) and violations:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Budget check found violations. Revise decisions minimally and re-check. "
                                    "Prefer reducing size_tier before reducing quantity unless the item is optional. "
                                    f"Violations: {json.dumps(_preview_budget_violations(violations), ensure_ascii=True)}"
                                ),
                            }
                        )
                    else:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Budget check passed. If your decisions are complete, you may now output final JSON only."
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
        except ValueError:
            messages.append(
                {
                    "role": "user",
                    "content": "Your output must be valid JSON only (no markdown, no prose).",
                }
            )
            continue

        ok, detail = _validate_decisions(
            result.get("decisions"),
            required_types,
            members_by_cluster=members_by_cluster,
            anchors_by_cluster=anchors_by_cluster,
        )
        if not ok:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Decisions are required before final output. "
                        "You MUST output exactly one decision per member type across all clusters, "
                        "quantity must be an integer >= 0, and each cluster must keep at least one anchor "
                        "(quantity >= 1, using CLUSTERS_JSON.anchors when present). "
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

        budget_out = _run_budget_check_for_result(
            result=result,
            context=context,
            size_profiles_by_category=size_profiles_by_category,
        )
        logger.info(
            "Finalization budget check output: %s",
            json.dumps(budget_out, ensure_ascii=True),
        )
        estimate_budget_called = True

        violations = budget_out.get("violations")
        if isinstance(violations, list) and violations:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your current decisions violate the area budget. "
                        "Revise them minimally and return updated JSON after re-checking budget. "
                        "Prefer reducing size_tier first for secondary/optional items. "
                        f"Violations: {json.dumps(_preview_budget_violations(violations), ensure_ascii=True)}"
                    ),
                }
            )
            continue

        _attach_rep_dims(result, size_profiles_by_category)
        return result

    raise TimeoutError("TierCountDirector exceeded max_steps without final JSON.")


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


def _coerce_get_size_profiles_args(
    *,
    args: dict[str, Any],
    context: dict[str, Any],
    last_draft: dict[str, Any] | None,
    required_types: list[str],
) -> dict[str, Any]:
    out = dict(args)

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

    return out


def _run_budget_check_for_result(
    *,
    result: dict[str, Any],
    context: dict[str, Any],
    size_profiles_by_category: dict[str, Any],
) -> dict[str, Any]:
    args = {
        "room_model": context.get("room_model_json", {}),
        "decisions": result.get("decisions", []),
        "size_profiles_by_category": size_profiles_by_category,
        "style": _infer_style(context),
        "user_notes": str(context.get("special_notes") or ""),
        "clusters_json": context.get("clusters_json"),
    }
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


def _validate_decisions(
    decisions: Any,
    required_types: list[str],
    *,
    members_by_cluster: dict[str, set[str]] | None = None,
    anchors_by_cluster: dict[str, set[str]] | None = None,
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

        decisions_by_cluster.setdefault(cluster_id, {})[t] = qty if isinstance(qty, int) else 0

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

    if isinstance(members_by_cluster, dict) and members_by_cluster:
        for cluster_id in sorted(members_by_cluster.keys()):
            members = members_by_cluster.get(cluster_id) or set()
            anchors = (
                anchors_by_cluster.get(cluster_id, set())
                if isinstance(anchors_by_cluster, dict)
                else set()
            )
            cluster_decisions = decisions_by_cluster.get(cluster_id, {})

            # If a cluster has anchors, require at least one anchor kept.
            if anchors:
                kept = any(cluster_decisions.get(a, 0) >= 1 for a in anchors)
                if not kept:
                    return (
                        False,
                        f"cluster {cluster_id} must keep at least one anchor with quantity >= 1: {sorted(anchors)}",
                    )
            else:
                # Fallback: require at least one member kept so the cluster isn't empty.
                kept = any(
                    cluster_decisions.get(m, 0) >= 1 for m in members if isinstance(m, str)
                )
                if not kept and members:
                    return (
                        False,
                        f"cluster {cluster_id} must keep at least one member with quantity >= 1",
                    )

    return True, "ok"


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
