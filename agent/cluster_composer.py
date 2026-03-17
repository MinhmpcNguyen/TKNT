from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Any

from agent_schema.cluster_composer_schema import ClusterComposerOutput
from clients.openai_client import OpenAIClient
from cluster_composer.tools import TOOL_REGISTRY, TOOL_SCHEMAS
from prompt.cluster_composer import CLUSTER_COMPOSER_PROMPT
from prompt.system import SYSTEM_PROMPT

logger = logging.getLogger(__name__)
FIXED_ACCESS_CLEARANCE_RATIO = 0.25


@dataclass(frozen=True)
class ClusterComposer:
    system_prompt: str = SYSTEM_PROMPT
    prompt_template: str = CLUSTER_COMPOSER_PROMPT

    def generate_raw(
        self,
        *,
        merged_clusters: dict[str, Any],
        cluster_id: str,
        description: str = "",
        special_notes: str = "",
        verifier_feedback_json: dict[str, Any] | None = None,
        max_steps: int = 30,
        access_clearance_ratio: float = FIXED_ACCESS_CLEARANCE_RATIO,
    ) -> str:
        access_clearance_ratio = FIXED_ACCESS_CLEARANCE_RATIO
        cluster = _find_cluster(merged_clusters, cluster_id)
        user_prompt = _build_prompt(
            template=self.prompt_template,
            description=description,
            special_notes=special_notes,
            input_json=cluster,
            verifier_feedback_json=verifier_feedback_json,
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return _run_with_tools(
            messages=messages,
            cluster=cluster,
            max_steps=max_steps,
            access_clearance_ratio=access_clearance_ratio,
        )

    def generate(
        self,
        *,
        merged_clusters: dict[str, Any],
        cluster_id: str,
        description: str = "",
        special_notes: str = "",
        verifier_feedback_json: dict[str, Any] | None = None,
        max_steps: int = 30,
        access_clearance_ratio: float = FIXED_ACCESS_CLEARANCE_RATIO,
    ) -> dict[str, Any]:
        access_clearance_ratio = FIXED_ACCESS_CLEARANCE_RATIO
        raw = self.generate_raw(
            merged_clusters=merged_clusters,
            cluster_id=cluster_id,
            description=description,
            special_notes=special_notes,
            verifier_feedback_json=verifier_feedback_json,
            max_steps=max_steps,
            access_clearance_ratio=access_clearance_ratio,
        )
        payload = _parse_json(raw)
        payload = _sanitize_cluster_output_payload(payload)
        return ClusterComposerOutput.model_validate(payload).model_dump()


# ---------------------------
# Prompt helpers
# ---------------------------


def _find_cluster(merged_clusters: dict[str, Any], cluster_id: str) -> dict[str, Any]:
    if not isinstance(merged_clusters, dict):
        raise ValueError("merged_clusters missing or invalid (expected dict).")
    clusters = merged_clusters.get("clusters", [])
    if isinstance(clusters, list):
        for cluster in clusters:
            if isinstance(cluster, dict) and cluster.get("cluster_id") == cluster_id:
                return cluster
    available = [
        c.get("cluster_id")
        for c in clusters
        if isinstance(c, dict) and c.get("cluster_id")
    ]
    raise ValueError(f"cluster_id not found: {cluster_id}. available={available}")


def _build_prompt(
    *,
    template: str,
    description: str,
    special_notes: str,
    input_json: dict[str, Any],
    verifier_feedback_json: dict[str, Any] | None,
) -> str:
    mapping = {
        "DESCRIPTION": description or "",
        "SPECIAL_NOTES": special_notes or "",
        "INPUT_JSON": _json_block(input_json),
        "VERIFIER_FEEDBACK_JSON": _json_block(verifier_feedback_json),
    }
    out = template
    for k, v in mapping.items():
        out = out.replace("{" + k + "}", v)
    return out


def _json_block(obj: Any) -> str:
    if obj is None:
        return "null"
    return json.dumps(obj, ensure_ascii=True, indent=2)


# ---------------------------
# Core tool loop
# ---------------------------


def _run_with_tools(
    *,
    messages: list[dict[str, Any]],
    cluster: dict[str, Any],
    max_steps: int,
    access_clearance_ratio: float,
) -> str:
    client = OpenAIClient()
    access_clearance_ratio = FIXED_ACCESS_CLEARANCE_RATIO

    expected_member_ids = _member_ids(cluster)

    last_draft: dict[str, Any] | None = None
    last_verifier_result: str | None = None  # VALID | INVALID | None

    best_valid_draft: dict[str, Any] | None = None
    best_valid_eval: dict[str, Any] | None = None
    best_valid_rank: tuple[Any, ...] | None = None

    verified_state_counts: dict[str, int] = {}
    tried_patch_keys_by_state: dict[str, set[str]] = {}

    # ---------------------------
    # Deterministic seed before first LLM turn
    # ---------------------------
    seeded_placements = _seed_local_layout(cluster)

    if seeded_placements:
        ok, detail = _validate_local_placements_full(
            seeded_placements, expected_member_ids
        )
        if ok:
            last_draft = {
                "cluster_id": cluster.get("cluster_id"),
                "local_placements": _canonicalize_local_placements(seeded_placements),
            }

            seed_eval = _run_verifier_once(
                cluster=cluster,
                placements=seeded_placements,
                access_clearance_ratio=access_clearance_ratio,
            )
            seed_preview = _verifier_preview(seed_eval)

            logger.info(
                "Deterministic seed eval: %s",
                json.dumps(seed_preview, ensure_ascii=True),
            )

            # Nếu seed đã VALID thì tối ưu host-side vài bước rồi đưa cho model tiếp tục
            if seed_eval.get("result") == "VALID":
                current_seed = _canonicalize_local_placements(seeded_placements)
                current_eval = seed_eval
                current_rank = _tool_rank(current_eval)

                best_valid_draft = {
                    "cluster_id": cluster.get("cluster_id"),
                    "local_placements": current_seed,
                }
                best_valid_eval = current_eval
                best_valid_rank = current_rank

                seed_state_sig = _placements_signature(current_seed)
                tried_patch_keys_by_state.setdefault(seed_state_sig, set())

                for _ in range(6):
                    selected_move, patched, patched_eval = _choose_best_single_patch(
                        cluster=cluster,
                        access_clearance_ratio=access_clearance_ratio,
                        tool_output=current_eval,
                        placements=current_seed,
                        tried_patch_keys=tried_patch_keys_by_state[seed_state_sig],
                    )

                    if (
                        selected_move is None
                        or patched is None
                        or not isinstance(patched_eval, dict)
                        or patched_eval.get("result") != "VALID"
                    ):
                        break

                    patched_rank = _tool_rank(patched_eval)
                    if not _rank_better(patched_rank, current_rank):
                        break

                    tried_patch_keys_by_state[seed_state_sig].add(
                        _move_key(selected_move)
                    )

                    current_seed = _canonicalize_local_placements(patched)
                    current_eval = patched_eval
                    current_rank = patched_rank

                    best_valid_draft = {
                        "cluster_id": cluster.get("cluster_id"),
                        "local_placements": current_seed,
                    }
                    best_valid_eval = current_eval
                    best_valid_rank = current_rank

                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "A deterministic seeded VALID layout is available. "
                            "Use these verified local_placements as your current best state. "
                            "Only continue if you can find a strictly better VALID compact layout with a single-object change.\n\n"
                            f"verified_local_placements={json.dumps(current_seed, ensure_ascii=True)}\n"
                            f"seed_eval_preview={json.dumps(_verifier_preview(current_eval), ensure_ascii=True)}"
                        ),
                    }
                )

            # Nếu seed chưa VALID thì ép model dùng seed này làm first attempt
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Use the following deterministic seeded local_placements as your first attempt. "
                        "Call LocalClusterVerifier with EXACTLY these local_placements first, then repair only if needed. "
                        "Do not invent a brand-new layout before checking this seed.\n\n"
                        f"seeded_local_placements={json.dumps(seeded_placements, ensure_ascii=True)}\n"
                        f"seed_eval_preview={json.dumps(seed_preview, ensure_ascii=True)}"
                    ),
                }
            )
        else:
            logger.warning("Deterministic seed rejected: %s", detail)

    for step in range(1, max_steps + 1):
        logger.info("ClusterComposer step %s/%s", step, max_steps)

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
                    "ClusterComposer stopped early due to context length exceeded.",
                    exc_info=exc,
                )
                if best_valid_draft is not None and isinstance(
                    best_valid_draft.get("local_placements"), list
                ):
                    placements = _canonicalize_local_placements(
                        best_valid_draft["local_placements"]
                    )
                    payload = _build_cluster_output_from_placements(
                        cluster=cluster,
                        placements=placements,
                        verifier_eval=best_valid_eval,
                        notes=[
                            "Stopped due to context length exceeded; returned best verified valid layout."
                        ],
                    )
                    return json.dumps(payload, ensure_ascii=True)

                fallback = {
                    "status": "UNSAT",
                    "cluster_id": str(cluster.get("cluster_id") or ""),
                    "local_frame": {
                        "unit": "mm",
                        "grid_mm": _extract_grid_mm(cluster),
                        "origin_note": "(0,0) is an arbitrary local origin for this cluster",
                    },
                    "local_placements": [],
                    "cluster_footprint": {
                        "type": "union_of_rects",
                        "rects": [],
                        "local_bbox": {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0},
                    },
                    "notes": [
                        "Stopped due to context length exceeded before finding a valid layout."
                    ],
                    "missing": [],
                }
                return json.dumps(fallback, ensure_ascii=True)
            raise

        message = _extract_message(response)
        tool_calls = _extract_tool_calls(message)
        content = getattr(message, "content", "") or ""

        draft_in_content = _try_parse_json_object(content)
        if isinstance(draft_in_content, dict) and isinstance(
            draft_in_content.get("local_placements"), list
        ):
            last_draft = draft_in_content

        # ---------------------------
        # Tool calls branch
        # ---------------------------
        if tool_calls:
            logger.info(
                "Tool calls requested: %s",
                [(c.get("function", {}) or {}).get("name") for c in tool_calls],
            )

            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls,
                }
            )

            restart_outer = False
            handled_call_ids: set[str] = set()

            for idx, call in enumerate(tool_calls):
                call_id = call.get("id") or f"tool_{idx}"
                fn = call.get("function", {}) or {}
                name = fn.get("name")
                args_text = fn.get("arguments") or "{}"
                args = _safe_json_loads(args_text)

                if name != "LocalClusterVerifier":
                    tool_output = {
                        "error": "unsupported_tool",
                        "tool": name,
                        "message": "Only LocalClusterVerifier is supported in ClusterComposer.",
                    }
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": str(name),
                            "content": json.dumps(tool_output, ensure_ascii=True),
                        }
                    )
                    handled_call_ids.add(call_id)
                    continue

                if name == "LocalClusterVerifier":
                    args = _coerce_verifier_args(
                        args=args,
                        cluster=cluster,
                        last_draft=last_draft,
                        access_clearance_ratio=access_clearance_ratio,
                    )

                    lp = args.get("local_placements")
                    if not isinstance(lp, list) or not lp:
                        tool_output = {
                            "result": "INVALID",
                            "errors": [
                                {
                                    "code": "MISSING_LOCAL_PLACEMENTS",
                                    "detail": "local_placements is required in tool arguments",
                                }
                            ],
                        }
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "name": name,
                                "content": json.dumps(tool_output, ensure_ascii=True),
                            }
                        )
                        handled_call_ids.add(call_id)
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "You attempted to call LocalClusterVerifier without local_placements. "
                                    "Re-call LocalClusterVerifier and INCLUDE local_placements inside the TOOL CALL arguments. "
                                    f"local_placements must contain every member id exactly once: {expected_member_ids}. "
                                    "Do NOT output a separate draft; put placements directly in tool arguments."
                                ),
                            }
                        )
                        restart_outer = True
                        continue

                    ok, detail = _validate_local_placements_full(
                        lp, expected_member_ids
                    )
                    if not ok:
                        tool_output = {
                            "result": "INVALID",
                            "errors": [
                                {
                                    "code": "INVALID_LOCAL_PLACEMENTS",
                                    "detail": detail,
                                }
                            ],
                        }
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "name": name,
                                "content": json.dumps(tool_output, ensure_ascii=True),
                            }
                        )
                        handled_call_ids.add(call_id)
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Your local_placements is incomplete/invalid. "
                                    f"Requirement: every member id exactly once: {expected_member_ids}. "
                                    f"Issue: {detail}. "
                                    "Re-call LocalClusterVerifier with corrected local_placements in tool arguments."
                                ),
                            }
                        )
                        restart_outer = True
                        continue

                    last_draft = {
                        "cluster_id": cluster.get("cluster_id"),
                        "local_placements": _canonicalize_local_placements(lp),
                    }

                tool_output = _safe_run_tool(name, args)

                logger.info(
                    "Tool %s output: %s",
                    name,
                    json.dumps(tool_output, ensure_ascii=True),
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": json.dumps(tool_output, ensure_ascii=True),
                    }
                )
                handled_call_ids.add(call_id)

                if name != "LocalClusterVerifier":
                    continue

                result = tool_output.get("result")
                lp_verified = args.get("local_placements") or []
                lp_verified = _canonicalize_local_placements(lp_verified)

                state_sig = _placements_signature(lp_verified)
                verified_state_counts[state_sig] = (
                    verified_state_counts.get(state_sig, 0) + 1
                )
                state_repeat_count = verified_state_counts[state_sig]
                tried_patch_keys_by_state.setdefault(state_sig, set())

                if result == "VALID":
                    last_verifier_result = "VALID"
                    current_valid_draft = {
                        "cluster_id": cluster.get("cluster_id"),
                        "local_placements": lp_verified,
                    }
                    current_valid_eval = tool_output
                    last_draft = current_valid_draft

                    current_rank = _tool_rank(tool_output)

                    if best_valid_rank is None or current_rank < best_valid_rank:
                        best_valid_rank = current_rank
                        best_valid_draft = current_valid_draft
                        best_valid_eval = current_valid_eval

                    # VALID state: try one more single-object improvement first.
                    try:
                        selected_move, patched, patched_eval = (
                            _choose_best_single_patch(
                                cluster=cluster,
                                access_clearance_ratio=access_clearance_ratio,
                                tool_output=tool_output,
                                placements=lp_verified,
                                tried_patch_keys=tried_patch_keys_by_state[state_sig],
                            )
                        )
                    except Exception as exc:
                        logger.exception(
                            "Valid-state improvement patch selection failed: %s",
                            exc,
                        )
                        selected_move, patched, patched_eval = None, None, None

                    if (
                        selected_move is not None
                        and patched is not None
                        and isinstance(patched_eval, dict)
                        and patched_eval.get("result") == "VALID"
                        and _tool_rank(patched_eval) < current_rank
                    ):
                        ok, detail = _validate_local_placements_full(
                            patched, expected_member_ids
                        )
                        if ok:
                            move_key = _move_key(selected_move)
                            tried_patch_keys_by_state[state_sig].add(move_key)

                            last_draft = {
                                "cluster_id": cluster.get("cluster_id"),
                                "local_placements": patched,
                            }

                            patched_eval_preview = _verifier_preview(patched_eval)

                            messages.append(
                                {
                                    "role": "user",
                                    "content": (
                                        "LocalClusterVerifier returned VALID, but a strictly better VALID compact layout is available. "
                                        "Use the following SINGLE patched candidate as your next attempt. "
                                        "Do NOT combine it with other suggested moves in the same step. "
                                        "Re-call LocalClusterVerifier with EXACTLY these local_placements unless you have one strictly better single-object patch.\n\n"
                                        f"selected_move={json.dumps(selected_move, ensure_ascii=True)}\n"
                                        f"patched_local_placements={json.dumps(patched, ensure_ascii=True)}\n"
                                        f"patched_eval_preview={json.dumps(patched_eval_preview, ensure_ascii=True)}"
                                    ),
                                }
                            )
                            continue

                    # No better valid patch found -> finalize immediately from host
                    final_draft = best_valid_draft or current_valid_draft
                    final_eval = best_valid_eval or current_valid_eval
                    placements = _canonicalize_local_placements(
                        final_draft["local_placements"]
                    )

                    payload = _build_cluster_output_from_placements(
                        cluster=cluster,
                        placements=placements,
                        verifier_eval=final_eval,
                        notes=["Returned best verified valid layout from controller."],
                    )
                    return json.dumps(payload, ensure_ascii=True)

                last_verifier_result = "INVALID"

                try:
                    selected_move, patched, patched_eval = _choose_best_single_patch(
                        cluster=cluster,
                        access_clearance_ratio=access_clearance_ratio,
                        tool_output=tool_output,
                        placements=lp_verified,
                        tried_patch_keys=tried_patch_keys_by_state[state_sig],
                    )
                except Exception as exc:
                    logger.exception(
                        "Patch selection failed; falling back to manual fix prompt: %s",
                        exc,
                    )
                    selected_move, patched, patched_eval = None, None, None

                if selected_move is not None and patched is not None:
                    ok, detail = _validate_local_placements_full(
                        patched, expected_member_ids
                    )

                    if ok:
                        move_key = _move_key(selected_move)
                        tried_patch_keys_by_state[state_sig].add(move_key)

                        last_draft = {
                            "cluster_id": cluster.get("cluster_id"),
                            "local_placements": patched,
                        }

                        patched_eval_preview = _verifier_preview(patched_eval)

                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "LocalClusterVerifier returned INVALID. "
                                    "Use the following SINGLE patched candidate as your next attempt. "
                                    "Do NOT combine it with other suggested moves in the same step. "
                                    "Re-call LocalClusterVerifier with EXACTLY these local_placements unless you have one strictly better single-object patch.\n\n"
                                    f"selected_move={json.dumps(selected_move, ensure_ascii=True)}\n"
                                    f"patched_local_placements={json.dumps(patched, ensure_ascii=True)}\n"
                                    f"patched_eval_preview={json.dumps(patched_eval_preview, ensure_ascii=True)}"
                                ),
                            }
                        )
                    else:
                        errs = tool_output.get("errors", [])
                        preview = errs[:8] if isinstance(errs, list) else errs
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "LocalClusterVerifier returned INVALID and the deterministic patch could not be applied safely. "
                                    "Fix ONLY the reported issues and re-check. "
                                    "Change only one object if possible, and do not mix multiple suggested moves in the same iteration. "
                                    f"Errors (preview): {json.dumps(preview, ensure_ascii=True)}"
                                ),
                            }
                        )
                else:
                    errs = tool_output.get("errors", [])
                    preview = errs[:8] if isinstance(errs, list) else errs

                    if state_repeat_count >= 2:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "You are repeating the same invalid placement state. "
                                    "Do NOT repeat previous repairs. "
                                    "Either propose a different single-object fix and re-call LocalClusterVerifier, "
                                    "or conclude UNSAT if no grid-valid repair exists with the allowed rotations.\n"
                                    f"Repeated state count={state_repeat_count}\n"
                                    f"Errors (preview): {json.dumps(preview, ensure_ascii=True)}"
                                ),
                            }
                        )
                    else:
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "LocalClusterVerifier returned INVALID. "
                                    "Fix ONLY the reported issues and re-check. "
                                    "Prefer changing one object at a time. "
                                    f"Errors (preview): {json.dumps(preview, ensure_ascii=True)}"
                                ),
                            }
                        )

            if len(handled_call_ids) < len(tool_calls):
                for idx, call in enumerate(tool_calls):
                    call_id = call.get("id") or f"tool_{idx}"
                    if call_id in handled_call_ids:
                        continue
                    fn = call.get("function", {}) or {}
                    name = fn.get("name")
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": name,
                            "content": json.dumps(
                                {
                                    "error": "tool_skipped",
                                    "tool": name,
                                    "message": "Skipped due to restart or earlier tool handling.",
                                },
                                ensure_ascii=True,
                            ),
                        }
                    )

            if restart_outer:
                continue

            continue

        # ---------------------------
        # No tool calls: final JSON
        # ---------------------------
        if not isinstance(content, str) or not content.strip():
            messages.append(
                {"role": "user", "content": "Return a JSON object (no markdown)."}
            )
            continue

        final_payload = _try_parse_json_object(content)
        if not isinstance(final_payload, dict):
            messages.append(
                {
                    "role": "user",
                    "content": "Your output must be valid JSON only (no markdown).",
                }
            )
            continue

        status = final_payload.get("status")
        cid = final_payload.get("cluster_id")
        if status not in {"OK", "UNSAT", "NEED_INFO"} or not isinstance(cid, str):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your JSON must include top-level fields: status (OK|UNSAT|NEED_INFO) and cluster_id (string)."
                    ),
                }
            )
            continue

        if status in {"UNSAT", "NEED_INFO"}:
            if best_valid_draft is not None and isinstance(
                best_valid_draft.get("local_placements"), list
            ):
                logger.warning(
                    "ClusterComposer received status=%s but a verified valid layout already exists; returning best valid.",
                    status,
                )
                placements = _canonicalize_local_placements(
                    best_valid_draft["local_placements"]
                )
                payload = _build_cluster_output_from_placements(
                    cluster=cluster,
                    placements=placements,
                    verifier_eval=best_valid_eval,
                    notes=[
                        "Model returned non-OK status; returned best verified valid layout instead."
                    ],
                )
                return json.dumps(payload, ensure_ascii=True)

            if status == "UNSAT" and last_verifier_result is None:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Do NOT conclude status=UNSAT before attempting LocalClusterVerifier at least once. "
                            "Call LocalClusterVerifier now with local_placements in tool arguments, including every member exactly once: "
                            f"{expected_member_ids}."
                        ),
                    }
                )
                continue

            return content

        if status == "OK" and last_verifier_result != "VALID":
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You cannot output status=OK until LocalClusterVerifier returns VALID. "
                        "Call LocalClusterVerifier now with local_placements inside tool arguments."
                    ),
                }
            )
            continue

        if status == "OK":
            lp = final_payload.get("local_placements")
            ok, detail = _validate_local_placements_full(lp, expected_member_ids)
            if not ok:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your local_placements is incomplete/invalid. "
                            f"Requirement: every member id exactly once: {expected_member_ids}. "
                            f"Issue: {detail}"
                        ),
                    }
                )
                continue

            # Re-verify final output
            final_eval = _run_verifier_once(
                cluster=cluster,
                placements=lp,
                access_clearance_ratio=access_clearance_ratio,
            )

            if final_eval.get("result") != "VALID":
                preview = _verifier_preview(final_eval)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your final JSON does not verify as VALID. "
                            "Do NOT change the schema. "
                            "Re-call LocalClusterVerifier if needed, or output FINAL JSON using the verified placements.\n\n"
                            f"final_eval_preview={json.dumps(preview, ensure_ascii=True)}"
                        ),
                    }
                )
                continue

            final_rank = _tool_rank(final_eval)

            if (
                best_valid_rank is not None
                and best_valid_draft is not None
                and final_rank > best_valid_rank
            ):
                chosen_placements = _canonicalize_local_placements(
                    best_valid_draft["local_placements"]
                )
                payload = _build_cluster_output_from_placements(
                    cluster=cluster,
                    placements=chosen_placements,
                    verifier_eval=best_valid_eval,
                    notes=["Returned better previously verified valid layout."],
                )
                return json.dumps(payload, ensure_ascii=True)

            chosen_placements = _canonicalize_local_placements(lp)

            if best_valid_rank is None or final_rank < best_valid_rank:
                best_valid_rank = final_rank
                best_valid_draft = {
                    "cluster_id": cluster.get("cluster_id"),
                    "local_placements": chosen_placements,
                }
                best_valid_eval = final_eval

            payload = _build_cluster_output_from_placements(
                cluster=cluster,
                placements=chosen_placements,
                verifier_eval=final_eval,
                notes=["Returned verified valid layout."],
            )
            return json.dumps(payload, ensure_ascii=True)

    if best_valid_draft is not None and isinstance(
        best_valid_draft.get("local_placements"), list
    ):
        logger.warning(
            "ClusterComposer hit max_steps; returning best verified valid layout."
        )
        placements = _canonicalize_local_placements(
            best_valid_draft["local_placements"]
        )
        payload = _build_cluster_output_from_placements(
            cluster=cluster,
            placements=placements,
            verifier_eval=best_valid_eval,
            notes=["Stopped due to max_steps; returned best verified valid layout."],
        )
        return json.dumps(payload, ensure_ascii=True)

    logger.warning("ClusterComposer hit max_steps with no valid layout.")
    return json.dumps(
        {
            "status": "UNSAT",
            "cluster_id": str(cluster.get("cluster_id") or ""),
            "local_frame": {
                "unit": "mm",
                "grid_mm": _extract_grid_mm(cluster),
                "origin_note": "(0,0) is an arbitrary local origin for this cluster",
            },
            "local_placements": [],
            "cluster_footprint": {
                "type": "union_of_rects",
                "rects": [],
                "local_bbox": {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0},
            },
            "notes": ["Stopped due to max_steps with no verified valid layout."],
            "missing": [],
        },
        ensure_ascii=True,
    )


# ---------------------------
# Tool / args helpers
# ---------------------------


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
    cluster: dict[str, Any],
    last_draft: dict[str, Any] | None,
    access_clearance_ratio: float,
) -> dict[str, Any]:
    output = dict(args)

    if "hard_constraints" not in output:
        output["hard_constraints"] = cluster.get("hard_constraints", [])

    rules = cluster.get("cluster_rules", {}) if isinstance(cluster, dict) else {}
    if "grid_mm" not in output:
        output["grid_mm"] = int((rules or {}).get("grid_mm", 1) or 1)

    if "objects" not in output:
        output["objects"] = _build_object_specs(cluster)

    if "cluster_rules" not in output:
        output["cluster_rules"] = rules if isinstance(rules, dict) else None

    output["access_clearance_ratio"] = float(FIXED_ACCESS_CLEARANCE_RATIO)

    if not isinstance(output.get("local_placements"), list) or not output.get(
        "local_placements"
    ):
        if isinstance(last_draft, dict) and isinstance(
            last_draft.get("local_placements"), list
        ):
            output["local_placements"] = last_draft["local_placements"]

    if "use_clearance" not in output:
        output["use_clearance"] = True

    placement_ids = _placement_ids(output.get("local_placements"))
    if not _objects_have_valid_dims(output.get("objects")) or (
        placement_ids and not _objects_cover_ids(output.get("objects"), placement_ids)
    ):
        output["objects"] = _build_object_specs(cluster)

    return output


def _build_object_specs(cluster: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    decisions = cluster.get("decisions", [])
    rules = cluster.get("cluster_rules", {}) if isinstance(cluster, dict) else {}
    allowed_raw = rules.get("allowed_rotations") if isinstance(rules, dict) else None
    allowed = allowed_raw if isinstance(allowed_raw, dict) else {}
    facing_raw = rules.get("facing") if isinstance(rules, dict) else None
    facing = facing_raw if isinstance(facing_raw, dict) else {}

    if not isinstance(decisions, list):
        return specs

    for d in decisions:
        if not isinstance(d, dict):
            continue

        obj_id = d.get("object_type") or d.get("category")
        rep = d.get("rep_dims_m") or {}
        if not isinstance(obj_id, str) or not obj_id or not isinstance(rep, dict):
            continue

        L = float(rep.get("L", 0) or 0)
        W = float(rep.get("W", 0) or 0)
        if L <= 0 or W <= 0:
            continue

        w = int(round(L * 1000))
        h = int(round(W * 1000))

        spec: dict[str, Any] = {
            "id": obj_id,
            "w": w,
            "h": h,
            "clearance_mm": int(d.get("clearance_mm", 0) or 0),
            "allowed_rotations": allowed.get(obj_id, [0, 90, 180, 270]),
            "collision": d.get("collision", "solid"),
        }

        f = facing.get(obj_id)
        if isinstance(f, dict) and f.get("front") in {"top", "bottom", "left", "right"}:
            spec["front"] = f["front"]

        specs.append(spec)

    return specs


def _objects_have_valid_dims(objects: Any) -> bool:
    if isinstance(objects, dict):
        if not objects:
            return False
        for obj_id, spec in objects.items():
            if not isinstance(obj_id, str) or not obj_id.strip():
                return False
            if not isinstance(spec, dict):
                return False
            w = int(spec.get("w", 0) or 0)
            h = int(spec.get("h", 0) or 0)
            if w <= 0 or h <= 0:
                return False
        return True

    if not isinstance(objects, list) or not objects:
        return False
    for spec in objects:
        if not isinstance(spec, dict):
            return False
        obj_id = spec.get("id")
        if not isinstance(obj_id, str) or not obj_id.strip():
            return False
        w = int(spec.get("w", 0) or 0)
        h = int(spec.get("h", 0) or 0)
        if w <= 0 or h <= 0:
            return False
    return True


def _objects_cover_ids(objects: Any, required_ids: set[str]) -> bool:
    if not required_ids:
        return True
    if isinstance(objects, dict):
        return required_ids.issubset({k for k in objects.keys() if isinstance(k, str)})
    if isinstance(objects, list):
        ids: set[str] = set()
        for spec in objects:
            if not isinstance(spec, dict):
                continue
            obj_id = spec.get("id")
            if isinstance(obj_id, str) and obj_id:
                ids.add(obj_id)
        return required_ids.issubset(ids)
    return False


def _placement_ids(local_placements: Any) -> set[str]:
    if not isinstance(local_placements, list):
        return set()
    out: set[str] = set()
    for p in local_placements:
        if not isinstance(p, dict):
            continue
        pid = p.get("id")
        if isinstance(pid, str) and pid:
            out.add(pid)
    return out


def _seed_local_layout(cluster: dict[str, Any]) -> list[dict[str, Any]]:
    members = _member_ids(cluster)
    if not members:
        return []

    grid_mm = _extract_grid_mm(cluster)
    spec_by_id = {
        s["id"]: s
        for s in _build_object_specs(cluster)
        if isinstance(s, dict) and isinstance(s.get("id"), str)
    }

    if any(mid not in spec_by_id for mid in members):
        return []

    hard_constraints = (
        cluster.get("hard_constraints", [])
        if isinstance(cluster.get("hard_constraints"), list)
        else []
    )
    anchors = [
        a
        for a in (
            cluster.get("anchors", [])
            if isinstance(cluster.get("anchors"), list)
            else []
        )
        if isinstance(a, str) and a in spec_by_id
    ]
    priority_map = _seed_build_priority_map(cluster)
    access_required = _seed_collect_access_required(cluster)

    dependent_ids: set[str] = set()
    relations: list[dict[str, Any]] = []

    for c in hard_constraints:
        if not isinstance(c, dict):
            continue
        ctype = c.get("type")
        a = c.get("a")
        b = c.get("b")
        if ctype in {"dock_to_edge", "anchor_side", "contain_in"}:
            if isinstance(a, str) and isinstance(b, str):
                if a in spec_by_id and b in spec_by_id:
                    relations.append(dict(c))
                    dependent_ids.add(a)

    freestanding = [m for m in members if m in spec_by_id and m not in dependent_ids]
    if not freestanding:
        freestanding = [m for m in members if m in spec_by_id]

    def _main_sort_key(oid: str) -> tuple[Any, ...]:
        area = int(spec_by_id[oid].get("w", 0) or 0) * int(
            spec_by_id[oid].get("h", 0) or 0
        )
        return (
            0 if oid in anchors else 1,
            int(priority_map.get(oid, 9)),
            0 if oid in access_required else 1,
            -area,
            oid,
        )

    freestanding.sort(key=_main_sort_key)

    base_row_ids: list[str] = []
    back_row_ids: list[str] = []

    for oid in freestanding:
        pr = int(priority_map.get(oid, 9))
        if pr >= 3:
            back_row_ids.append(oid)
        else:
            base_row_ids.append(oid)

    if not base_row_ids and back_row_ids:
        base_row_ids = back_row_ids
        back_row_ids = []

    placements: dict[str, dict[str, Any]] = {}

    row_gap = max(grid_mm, 150)
    back_row_gap = max(grid_mm, 250)

    # ---- place main freestanding row at y = 0
    x_cursor = 0
    for oid in base_row_ids:
        spec = spec_by_id[oid]
        rot = _seed_first_allowed_rotation(spec, prefer=0)
        w, h = _seed_rotated_wh(spec, rot)

        x = _seed_snap_up_value(x_cursor, grid_mm)
        y = 0

        placements[oid] = {"id": oid, "x": int(x), "y": int(y), "rot": int(rot)}
        x_cursor = x + w + row_gap

    # ---- place optional/back-row freestanding below main row
    max_main_h = 0
    for oid in base_row_ids:
        spec = spec_by_id[oid]
        rot = placements[oid]["rot"]
        _, h = _seed_rotated_wh(spec, rot)
        max_main_h = max(max_main_h, h)

    back_y = -_seed_snap_up_value(max_main_h + back_row_gap, grid_mm)
    back_x_cursor = 0

    for oid in back_row_ids:
        spec = spec_by_id[oid]
        rot = _seed_first_allowed_rotation(spec, prefer=0)
        w, h = _seed_rotated_wh(spec, rot)

        x = _seed_snap_up_value(back_x_cursor, grid_mm)
        y = back_y

        placements[oid] = {"id": oid, "x": int(x), "y": int(y), "rot": int(rot)}
        back_x_cursor = x + w + row_gap

    # ---- place dependents relative to their base
    unresolved = relations[:]
    for _ in range(len(relations) + 3):
        progressed = False
        still_unresolved: list[dict[str, Any]] = []

        for rel in unresolved:
            a = rel.get("a")
            b = rel.get("b")
            if not isinstance(a, str) or not isinstance(b, str):
                continue

            if a in placements:
                continue
            if b not in placements:
                still_unresolved.append(rel)
                continue

            placed = _seed_place_relative(
                rel=rel,
                cluster=cluster,
                spec_by_id=spec_by_id,
                placements=placements,
                grid_mm=grid_mm,
            )
            if placed is None:
                still_unresolved.append(rel)
                continue

            placements[a] = placed
            progressed = True

        unresolved = still_unresolved
        if not progressed:
            break

    # ---- any leftover members: place on a far fallback row
    fallback_ids = [m for m in members if m in spec_by_id and m not in placements]
    fallback_y = back_y - _seed_snap_up_value(
        max(max_main_h, 600) + back_row_gap, grid_mm
    )
    fallback_x = 0

    for oid in fallback_ids:
        spec = spec_by_id[oid]
        rot = _seed_first_allowed_rotation(spec, prefer=0)
        w, h = _seed_rotated_wh(spec, rot)

        x = _seed_snap_up_value(fallback_x, grid_mm)
        placements[oid] = {
            "id": oid,
            "x": int(x),
            "y": int(fallback_y),
            "rot": int(rot),
        }
        fallback_x = x + w + row_gap

    return _seed_normalize_to_origin(list(placements.values()))


def _seed_build_priority_map(cluster: dict[str, Any]) -> dict[str, int]:
    rank_map = {"anchor": 0, "primary": 1, "secondary": 2, "optional": 3}
    out: dict[str, int] = {}

    decisions = cluster.get("decisions", [])
    if not isinstance(decisions, list):
        return out

    for d in decisions:
        if not isinstance(d, dict):
            continue
        oid = d.get("object_type") or d.get("category")
        if not isinstance(oid, str) or not oid:
            continue
        priority = str(d.get("priority", "secondary")).lower()
        out[oid] = rank_map.get(priority, 2)

    return out


def _seed_collect_access_required(cluster: dict[str, Any]) -> set[str]:
    out: set[str] = set()

    rules = cluster.get("cluster_rules", {})
    if isinstance(rules, dict):
        ar = rules.get("access_requirements")
        if isinstance(ar, list):
            for item in ar:
                if not isinstance(item, dict):
                    continue
                if (
                    item.get("type") == "front_clearance"
                    and item.get("required") is True
                ):
                    oid = item.get("id")
                    if isinstance(oid, str) and oid:
                        out.add(oid)

    hard = cluster.get("hard_constraints", [])
    if isinstance(hard, list):
        for c in hard:
            if not isinstance(c, dict):
                continue
            if (
                c.get("type") == "requires_access"
                and c.get("mode") == "front_clearance"
            ):
                oid = c.get("id")
                if isinstance(oid, str) and oid:
                    out.add(oid)

    return out


def _seed_first_allowed_rotation(spec: dict[str, Any], prefer: int = 0) -> int:
    allowed = spec.get("allowed_rotations")
    if isinstance(allowed, list) and allowed:
        allowed_norm = [int(x) % 360 for x in allowed if isinstance(x, int)]
        if prefer in allowed_norm:
            return int(prefer)
        if 0 in allowed_norm:
            return 0
        return int(allowed_norm[0])
    return int(prefer) % 360


def _seed_rotated_wh(spec: dict[str, Any], rot: int) -> tuple[int, int]:
    w = int(spec.get("w", 0) or 0)
    h = int(spec.get("h", 0) or 0)
    if rot % 360 in (90, 270):
        return h, w
    return w, h


def _seed_snap_up_value(v: int, grid_mm: int) -> int:
    if grid_mm <= 0:
        return int(v)
    return int(math.ceil(v / grid_mm) * grid_mm)


def _seed_snap_nearest_value(v: int, grid_mm: int) -> int:
    if grid_mm <= 0:
        return int(v)
    down = math.floor(v / grid_mm) * grid_mm
    up = down + grid_mm
    return int(down if abs(v - down) <= abs(up - v) else up)


def _seed_pick_grid_in_interval(
    current: int,
    interval: tuple[int, int],
    grid_mm: int,
) -> int:
    lo, hi = interval
    if lo > hi:
        lo, hi = hi, lo

    if grid_mm <= 0:
        return int(min(max(current, lo), hi))

    first = int(math.ceil(lo / grid_mm) * grid_mm)
    last = int(math.floor(hi / grid_mm) * grid_mm)

    if first <= last:
        cand = _seed_snap_nearest_value(current, grid_mm)
        if cand < first:
            return first
        if cand > last:
            return last
        return int(cand)

    mid = int((lo + hi) // 2)
    return _seed_snap_nearest_value(mid, grid_mm)


def _seed_rotate_side(side: str, rot: int) -> str:
    mapping = {
        0: {"top": "top", "right": "right", "bottom": "bottom", "left": "left"},
        90: {"top": "left", "right": "top", "bottom": "right", "left": "bottom"},
        180: {"top": "bottom", "right": "left", "bottom": "top", "left": "right"},
        270: {"top": "right", "right": "bottom", "bottom": "left", "left": "top"},
    }
    return mapping.get(rot % 360, mapping[0]).get(side, side)


def _seed_get_front_base(
    cluster: dict[str, Any], obj_id: str, spec: dict[str, Any]
) -> str:
    rules = cluster.get("cluster_rules", {})
    if isinstance(rules, dict):
        facing = rules.get("facing", {})
        if isinstance(facing, dict):
            f = facing.get(obj_id)
            if isinstance(f, dict):
                front = f.get("front")
                if front in {"top", "bottom", "left", "right"}:
                    return str(front)

    spec_front = spec.get("front")
    if spec_front in {"top", "bottom", "left", "right"}:
        return str(spec_front)
    return "top"


def _seed_opposite_side(side: str) -> str:
    return {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}.get(
        side, side
    )


def _seed_resolve_edge_token_to_base_side(edge_token: str, front_base: str) -> str:
    if edge_token in {"top", "bottom", "left", "right"}:
        return edge_token
    if edge_token == "front":
        return front_base
    if edge_token == "back":
        return _seed_opposite_side(front_base)
    return "top"


def _seed_place_relative(
    *,
    rel: dict[str, Any],
    cluster: dict[str, Any],
    spec_by_id: dict[str, dict[str, Any]],
    placements: dict[str, dict[str, Any]],
    grid_mm: int,
) -> dict[str, Any] | None:
    a = rel.get("a")
    b = rel.get("b")
    if not isinstance(a, str) or not isinstance(b, str):
        return None
    if a not in spec_by_id or b not in spec_by_id or b not in placements:
        return None

    a_spec = spec_by_id[a]
    b_spec = spec_by_id[b]
    b_pl = placements[b]

    b_rot = int(b_pl["rot"]) % 360
    bx = int(b_pl["x"])
    by = int(b_pl["y"])
    bw, bh = _seed_rotated_wh(b_spec, b_rot)

    a_rot = _seed_first_allowed_rotation(a_spec, prefer=0)
    aw, ah = _seed_rotated_wh(a_spec, a_rot)

    ctype = rel.get("type")

    if ctype == "contain_in":
        x_lo = bx
        x_hi = bx + bw - aw
        y_lo = by
        y_hi = by + bh - ah
        if x_hi < x_lo or y_hi < y_lo:
            return None

        x_mid = bx + (bw - aw) // 2
        y_mid = by + (bh - ah) // 2

        x = _seed_pick_grid_in_interval(x_mid, (x_lo, x_hi), grid_mm)
        y = _seed_pick_grid_in_interval(y_mid, (y_lo, y_hi), grid_mm)
        return {"id": a, "x": int(x), "y": int(y), "rot": int(a_rot)}

    gap_min = int(rel.get("gap_min", 0) or 0)
    gap_max = int(rel.get("gap_max", 0) or 0)

    if ctype == "dock_to_edge":
        b_edge = rel.get("b_edge")
        if not isinstance(b_edge, str):
            return None
        front_base = _seed_get_front_base(cluster, b, b_spec)
        base_side = _seed_resolve_edge_token_to_base_side(b_edge, front_base)
    elif ctype == "anchor_side":
        side = rel.get("side")
        if not isinstance(side, str):
            return None
        if side.startswith("head_"):
            base_side = "top"
        elif side.startswith("foot_"):
            base_side = "bottom"
        elif side in {"head"}:
            base_side = "top"
        elif side in {"foot"}:
            base_side = "bottom"
        else:
            base_side = side
    else:
        return None

    mapped_side = _seed_rotate_side(base_side, b_rot)

    if mapped_side == "top":
        y = _seed_pick_grid_in_interval(
            by + bh + gap_min,
            (by + bh + gap_min, by + bh + gap_max),
            grid_mm,
        )
        x_center = bx + (bw - aw) // 2
        x = _seed_snap_nearest_value(x_center, grid_mm)
        return {"id": a, "x": int(x), "y": int(y), "rot": int(a_rot)}

    if mapped_side == "bottom":
        y = _seed_pick_grid_in_interval(
            by - ah - gap_min,
            (by - ah - gap_max, by - ah - gap_min),
            grid_mm,
        )
        x_center = bx + (bw - aw) // 2
        x = _seed_snap_nearest_value(x_center, grid_mm)
        return {"id": a, "x": int(x), "y": int(y), "rot": int(a_rot)}

    if mapped_side == "left":
        x = _seed_pick_grid_in_interval(
            bx - aw - gap_min,
            (bx - aw - gap_max, bx - aw - gap_min),
            grid_mm,
        )
        y_center = by + (bh - ah) // 2
        y = _seed_snap_nearest_value(y_center, grid_mm)
        return {"id": a, "x": int(x), "y": int(y), "rot": int(a_rot)}

    if mapped_side == "right":
        x = _seed_pick_grid_in_interval(
            bx + bw + gap_min,
            (bx + bw + gap_min, bx + bw + gap_max),
            grid_mm,
        )
        y_center = by + (bh - ah) // 2
        y = _seed_snap_nearest_value(y_center, grid_mm)
        return {"id": a, "x": int(x), "y": int(y), "rot": int(a_rot)}

    return None


def _seed_normalize_to_origin(
    placements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    canonical = _canonicalize_local_placements(placements)
    if not canonical:
        return canonical

    min_x = min(int(p["x"]) for p in canonical)
    min_y = min(int(p["y"]) for p in canonical)

    out: list[dict[str, Any]] = []
    for p in canonical:
        out.append(
            {
                "id": p["id"],
                "x": int(p["x"] - min_x),
                "y": int(p["y"] - min_y),
                "rot": int(p["rot"]) % 360,
            }
        )
    out.sort(key=lambda item: item["id"])
    return out


def _validate_local_placements_full(
    placements: Any,
    expected_ids: list[str],
) -> tuple[bool, str]:
    if not isinstance(placements, list) or not placements:
        return False, "local_placements missing/empty"

    seen: list[str] = []
    for p in placements:
        if not isinstance(p, dict):
            continue
        pid = p.get("id")
        if isinstance(pid, str) and pid:
            seen.append(pid)

    if not seen:
        return False, "local_placements has no valid id entries"

    missing = [cid for cid in expected_ids if cid not in seen]
    extra = [cid for cid in seen if cid not in expected_ids]
    dups = sorted({cid for cid in seen if seen.count(cid) > 1})

    if missing:
        return False, f"missing ids: {missing}"
    if extra:
        return False, f"unexpected ids: {extra}"
    if dups:
        return False, f"duplicate ids: {dups}"
    if len(seen) != len(expected_ids):
        return False, "count mismatch"
    return True, "ok"


# ---------------------------
# Deterministic repair helpers
# ---------------------------


def _canonicalize_local_placements(
    placements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in placements:
        if not isinstance(p, dict):
            continue
        pid = p.get("id")
        if not isinstance(pid, str) or not pid:
            continue
        out.append(
            {
                "id": pid,
                "x": int(p.get("x", 0)),
                "y": int(p.get("y", 0)),
                "rot": int(p.get("rot", 0)) % 360,
            }
        )
    out.sort(key=lambda item: item["id"])
    return out


def _placements_signature(placements: list[dict[str, Any]]) -> str:
    canonical = _canonicalize_local_placements(placements)
    return json.dumps(canonical, ensure_ascii=True, separators=(",", ":"))


def _move_key(move: dict[str, Any]) -> str:
    payload = {
        "reason": move.get("reason"),
        "move_object": move.get("move_object"),
        "dx": move.get("dx"),
        "dy": move.get("dy"),
        "new_x": move.get("new_x"),
        "new_y": move.get("new_y"),
        "new_rot": move.get("new_rot"),
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _choose_best_single_patch(
    *,
    cluster: dict[str, Any],
    access_clearance_ratio: float,
    tool_output: dict[str, Any],
    placements: list[dict[str, Any]],
    tried_patch_keys: set[str],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None, dict[str, Any] | None]:
    baseline_rank = _tool_rank(tool_output)
    grid_mm = _extract_grid_mm(cluster)

    candidates = _collect_patch_candidates(
        cluster=cluster,
        access_clearance_ratio=access_clearance_ratio,
        tool_output=tool_output,
        placements=placements,
        grid_mm=grid_mm,
    )

    best_move: dict[str, Any] | None = None
    best_patched: list[dict[str, Any]] | None = None
    best_eval: dict[str, Any] | None = None
    best_sort_key: tuple[Any, ...] | None = None
    best_rank: tuple[Any, ...] | None = None

    for move in candidates:
        key = _move_key(move)
        if key in tried_patch_keys:
            continue

        patched = _apply_single_move(placements, move)
        eval_out = _run_verifier_once(
            cluster=cluster,
            placements=patched,
            access_clearance_ratio=access_clearance_ratio,
        )

        candidate_rank = _tool_rank(eval_out)
        move_cost = _movement_cost(move)
        reason_rank = _reason_rank(move.get("reason"))

        sort_key = (
            *candidate_rank,
            move_cost,
            reason_rank,
            move.get("move_object") or "",
            key,
        )

        if best_sort_key is None or sort_key < best_sort_key:
            best_sort_key = sort_key
            best_rank = candidate_rank
            best_move = move
            best_patched = patched
            best_eval = eval_out

    if best_move is None or best_rank is None:
        return None, None, None

    if not _rank_better(best_rank, baseline_rank):
        return None, None, None

    return best_move, best_patched, best_eval


def _collect_patch_candidates(
    *,
    cluster: dict[str, Any],
    access_clearance_ratio: float,
    tool_output: dict[str, Any],
    placements: list[dict[str, Any]],
    grid_mm: int,
) -> list[dict[str, Any]]:
    del access_clearance_ratio  # tool ignores it; keep signature stable

    out: list[dict[str, Any]] = []

    suggested = tool_output.get("suggested_moves")
    if isinstance(suggested, list):
        for move in suggested:
            if isinstance(move, dict):
                out.append(move)

    out.extend(
        _generate_overlap_escape_moves(
            tool_output=tool_output,
            placements=placements,
            grid_mm=grid_mm,
        )
    )
    out.extend(
        _generate_access_block_escape_moves(
            tool_output=tool_output,
            placements=placements,
            grid_mm=grid_mm,
        )
    )
    out.extend(
        _generate_rotation_escape_moves(
            cluster=cluster,
            placements=placements,
        )
    )

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for move in out:
        key = _move_key(move)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(move)
    return deduped


def _generate_overlap_escape_moves(
    *,
    tool_output: dict[str, Any],
    placements: list[dict[str, Any]],
    grid_mm: int,
) -> list[dict[str, Any]]:
    debug = tool_output.get("debug") or {}
    rects = debug.get("rects") or {}
    pair_overlaps = debug.get("pair_overlaps") or []

    placement_by_id = {p["id"]: p for p in _canonicalize_local_placements(placements)}
    moves: list[dict[str, Any]] = []

    for ov in pair_overlaps:
        if not isinstance(ov, dict):
            continue
        a = ov.get("a")
        b = ov.get("b")
        if not isinstance(a, str) or not isinstance(b, str):
            continue
        if a not in rects or b not in rects:
            continue
        if a not in placement_by_id or b not in placement_by_id:
            continue

        ra = rects[a]
        rb = rects[b]
        if not isinstance(ra, dict) or not isinstance(rb, dict):
            continue

        ax1 = int(ra.get("x1", 0))
        ay1 = int(ra.get("y1", 0))
        ax2 = int(ra.get("x2", 0))
        ay2 = int(ra.get("y2", 0))

        bx1 = int(rb.get("x1", 0))
        by1 = int(rb.get("y1", 0))
        bx2 = int(rb.get("x2", 0))
        by2 = int(rb.get("y2", 0))

        margin = max(grid_mm, 0)

        candidate_specs = [
            (a, bx2 - ax1 + margin, 0, "HOST_OVERLAP_ESCAPE_RIGHT"),
            (a, -(ax2 - bx1 + margin), 0, "HOST_OVERLAP_ESCAPE_LEFT"),
            (a, 0, by2 - ay1 + margin, "HOST_OVERLAP_ESCAPE_UP"),
            (a, 0, -(ay2 - by1 + margin), "HOST_OVERLAP_ESCAPE_DOWN"),
            (b, ax2 - bx1 + margin, 0, "HOST_OVERLAP_ESCAPE_RIGHT"),
            (b, -(bx2 - ax1 + margin), 0, "HOST_OVERLAP_ESCAPE_LEFT"),
            (b, 0, ay2 - by1 + margin, "HOST_OVERLAP_ESCAPE_UP"),
            (b, 0, -(by2 - ay1 + margin), "HOST_OVERLAP_ESCAPE_DOWN"),
        ]

        for move_object, dx, dy, reason in candidate_specs:
            dx = _host_snap_delta(dx, grid_mm)
            dy = _host_snap_delta(dy, grid_mm)
            if dx == 0 and dy == 0:
                continue

            base = placement_by_id[move_object]
            moves.append(
                {
                    "reason": reason,
                    "a": a,
                    "b": b,
                    "move_object": move_object,
                    "dx": int(dx),
                    "dy": int(dy),
                    "new_x": int(base["x"] + dx),
                    "new_y": int(base["y"] + dy),
                    "note": "Host-generated overlap escape move.",
                }
            )

    return moves


def _generate_access_block_escape_moves(
    *,
    tool_output: dict[str, Any],
    placements: list[dict[str, Any]],
    grid_mm: int,
) -> list[dict[str, Any]]:
    debug = tool_output.get("debug") or {}
    rects = debug.get("rects") or {}
    clearance_rects = debug.get("front_clearance_rects") or {}
    blocks = debug.get("front_clearance_blocks") or []

    placement_by_id = {p["id"]: p for p in _canonicalize_local_placements(placements)}
    moves: list[dict[str, Any]] = []

    for block in blocks:
        if not isinstance(block, dict):
            continue

        owner = block.get("owner")
        blocker = block.get("blocker")
        if not isinstance(owner, str) or not isinstance(blocker, str):
            continue
        if owner not in clearance_rects or blocker not in rects:
            continue
        if blocker not in placement_by_id:
            continue

        cr = clearance_rects[owner]
        br = rects[blocker]
        if not isinstance(cr, dict) or not isinstance(br, dict):
            continue

        cx1 = int(cr.get("x1", 0))
        cy1 = int(cr.get("y1", 0))
        cx2 = int(cr.get("x2", 0))
        cy2 = int(cr.get("y2", 0))

        bx1 = int(br.get("x1", 0))
        by1 = int(br.get("y1", 0))
        bx2 = int(br.get("x2", 0))
        by2 = int(br.get("y2", 0))

        margin = max(grid_mm, 0)

        candidate_specs = [
            (cx2 - bx1 + margin, 0, "HOST_ACCESS_ESCAPE_RIGHT"),
            (-(bx2 - cx1 + margin), 0, "HOST_ACCESS_ESCAPE_LEFT"),
            (0, cy2 - by1 + margin, "HOST_ACCESS_ESCAPE_UP"),
            (0, -(by2 - cy1 + margin), "HOST_ACCESS_ESCAPE_DOWN"),
        ]

        base = placement_by_id[blocker]
        for dx, dy, reason in candidate_specs:
            dx = _host_snap_delta(dx, grid_mm)
            dy = _host_snap_delta(dy, grid_mm)
            if dx == 0 and dy == 0:
                continue

            moves.append(
                {
                    "reason": reason,
                    "owner": owner,
                    "blocker": blocker,
                    "move_object": blocker,
                    "dx": int(dx),
                    "dy": int(dy),
                    "new_x": int(base["x"] + dx),
                    "new_y": int(base["y"] + dy),
                    "note": "Host-generated access-clearance escape move.",
                }
            )

    return moves


def _generate_rotation_escape_moves(
    *,
    cluster: dict[str, Any],
    placements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    decisions = cluster.get("decisions", [])
    rules = cluster.get("cluster_rules", {}) if isinstance(cluster, dict) else {}
    allowed_raw = rules.get("allowed_rotations") if isinstance(rules, dict) else None
    allowed = allowed_raw if isinstance(allowed_raw, dict) else {}

    placement_by_id = {p["id"]: p for p in _canonicalize_local_placements(placements)}
    moves: list[dict[str, Any]] = []

    for d in decisions:
        if not isinstance(d, dict):
            continue
        obj_id = d.get("object_type") or d.get("category")
        if not isinstance(obj_id, str) or obj_id not in placement_by_id:
            continue

        current_rot = int(placement_by_id[obj_id]["rot"]) % 360
        allowed_rots = allowed.get(obj_id, [0, 90, 180, 270])
        if not isinstance(allowed_rots, list):
            continue

        for rot in allowed_rots:
            if not isinstance(rot, int):
                continue
            rot = rot % 360
            if rot == current_rot:
                continue

            moves.append(
                {
                    "reason": "HOST_ROTATION_ESCAPE",
                    "move_object": obj_id,
                    "new_rot": rot,
                    "note": "Host-generated rotation candidate.",
                }
            )

    return moves


def _run_verifier_once(
    *,
    cluster: dict[str, Any],
    placements: list[dict[str, Any]],
    access_clearance_ratio: float,
) -> dict[str, Any]:
    access_clearance_ratio = FIXED_ACCESS_CLEARANCE_RATIO
    args = {
        "hard_constraints": cluster.get("hard_constraints", []),
        "objects": _build_object_specs(cluster),
        "local_placements": _canonicalize_local_placements(placements),
        "grid_mm": _extract_grid_mm(cluster),
        "use_clearance": True,
        "cluster_rules": (
            cluster.get("cluster_rules")
            if isinstance(cluster.get("cluster_rules"), dict)
            else None
        ),
        "access_clearance_ratio": float(access_clearance_ratio),
    }
    return _safe_run_tool("LocalClusterVerifier", args)


def _extract_grid_mm(cluster: dict[str, Any]) -> int:
    rules = cluster.get("cluster_rules", {}) if isinstance(cluster, dict) else {}
    return int((rules or {}).get("grid_mm", 1) or 1)


def _build_cluster_output_from_placements(
    *,
    cluster: dict[str, Any],
    placements: list[dict[str, Any]],
    verifier_eval: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    cluster_id = str(cluster.get("cluster_id") or "")
    grid_mm = _extract_grid_mm(cluster)

    canonical_placements = _canonicalize_local_placements(placements)
    spec_by_id = {
        s["id"]: s
        for s in _build_object_specs(cluster)
        if isinstance(s, dict) and isinstance(s.get("id"), str)
    }

    rects: list[dict[str, Any]] = []
    for p in canonical_placements:
        pid = p.get("id")
        if not isinstance(pid, str) or not pid:
            continue
        spec = spec_by_id.get(pid)
        if spec is None:
            continue

        w = int(spec.get("w", 0) or 0)
        h = int(spec.get("h", 0) or 0)
        rot = int(p.get("rot", 0)) % 360
        if rot in (90, 270):
            w, h = h, w

        rects.append(
            {
                "id": pid,
                "x": int(p.get("x", 0)),
                "y": int(p.get("y", 0)),
                "w": int(w),
                "h": int(h),
            }
        )

    bbox = _local_bbox_from_rects(rects)

    orientation_meta = _infer_orientation_meta(
        cluster=cluster,
        placements=canonical_placements,
        rects=rects,
        spec_by_id=spec_by_id,
        verifier_eval=verifier_eval,
    )

    return {
        "status": "OK",
        "cluster_id": cluster_id,
        "local_frame": {
            "unit": "mm",
            "grid_mm": int(grid_mm),
            "origin_note": "(0,0) is an arbitrary local origin for this cluster",
        },
        "local_placements": canonical_placements,
        "cluster_footprint": {
            "type": "union_of_rects",
            "rects": rects,
            "local_bbox": bbox,
        },
        "orientation_meta": orientation_meta,
        "notes": list(notes or []),
        "missing": [],
    }


def _local_bbox_from_rects(rects: list[dict[str, Any]]) -> dict[str, int]:
    if not rects:
        return {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0}

    min_x = 10**18
    min_y = 10**18
    max_x = -(10**18)
    max_y = -(10**18)

    for r in rects:
        if not isinstance(r, dict):
            continue
        x = int(r.get("x", 0))
        y = int(r.get("y", 0))
        w = int(r.get("w", 0))
        h = int(r.get("h", 0))
        if w <= 0 or h <= 0:
            continue
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

    if min_x == 10**18:
        return {"min_x": 0, "min_y": 0, "max_x": 0, "max_y": 0}

    return {
        "min_x": int(min_x),
        "min_y": int(min_y),
        "max_x": int(max_x),
        "max_y": int(max_y),
    }


# ---------------------------
# Orientation meta inference
# ---------------------------


def _infer_orientation_meta(
    *,
    cluster: dict[str, Any],
    placements: list[dict[str, Any]],
    rects: list[dict[str, Any]],
    spec_by_id: dict[str, dict[str, Any]],
    verifier_eval: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # 1) Prefer verifier-derived orientation evidence from actual VALID layout
    meta_from_verifier = _infer_orientation_meta_from_verifier_eval(
        cluster=cluster,
        verifier_eval=verifier_eval,
    )
    if meta_from_verifier is not None:
        return meta_from_verifier

    # 2) Fallback heuristic only if verifier did not provide orientation_inference
    rect_by_id = {
        r["id"]: r
        for r in rects
        if isinstance(r, dict) and isinstance(r.get("id"), str)
    }
    placement_by_id = {
        p["id"]: p
        for p in placements
        if isinstance(p, dict) and isinstance(p.get("id"), str)
    }

    cluster_bbox = _local_bbox_from_rects(rects)
    cluster_axis = _infer_cluster_axis_local(rects)

    important_ids = _select_orientation_important_object_ids(
        cluster=cluster,
        rect_by_id=rect_by_id,
    )

    front_votes: dict[tuple[int, int], int] = {}
    axis_votes: dict[tuple[int, int], int] = {}
    important_objects: dict[str, dict[str, dict[str, int]]] = {}

    access_required = _seed_collect_access_required(cluster)
    anchors = {
        a
        for a in (
            cluster.get("anchors", [])
            if isinstance(cluster.get("anchors"), list)
            else []
        )
        if isinstance(a, str) and a
    }

    for oid in important_ids:
        rect = rect_by_id.get(oid)
        placement = placement_by_id.get(oid)
        spec = spec_by_id.get(oid)

        if rect is None or placement is None or spec is None:
            continue

        front = _infer_object_front_local_fallback(
            oid=oid,
            rect=rect,
            placement=placement,
            spec=spec,
            cluster_bbox=cluster_bbox,
            access_required=access_required,
        )
        axis = _infer_object_axis_local_fallback(
            rect=rect,
            front=front,
        )

        if front is None and axis is None:
            continue
        if front is None and axis is not None:
            front = _default_front_from_axis(axis)
        if axis is None and front is not None:
            axis = _axis_from_front(front)

        if front is None or axis is None:
            continue

        important_objects[oid] = {
            "front_local": _vec_to_dict(front),
            "axis_local": _vec_to_dict(axis),
        }

        weight = _orientation_vote_weight(
            oid=oid,
            access_required=access_required,
            anchors=anchors,
        )
        front_votes[front] = front_votes.get(front, 0) + weight
        axis_votes[axis] = axis_votes.get(axis, 0) + weight

    if axis_votes:
        cluster_axis = _pick_best_axis_vote(axis_votes, fallback=cluster_axis)

    cluster_front = _infer_cluster_front_local_fallback(
        cluster_axis=cluster_axis,
        front_votes=front_votes,
    )

    return {
        "cluster_front_local": _vec_to_dict(cluster_front),
        "cluster_axis_local": _vec_to_dict(cluster_axis),
        "important_objects": important_objects,
    }


def _infer_orientation_meta_from_verifier_eval(
    *,
    cluster: dict[str, Any],
    verifier_eval: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(verifier_eval, dict):
        return None

    debug = verifier_eval.get("debug")
    if not isinstance(debug, dict):
        return None

    orientation = debug.get("orientation_inference")
    if not isinstance(orientation, dict):
        return None

    cluster_axis = _coerce_axis_vec(orientation.get("cluster_axis_local_candidate"))
    cluster_front = _coerce_axis_vec(orientation.get("cluster_front_local_candidate"))

    objects_raw = orientation.get("objects")
    objects_raw = objects_raw if isinstance(objects_raw, dict) else {}

    anchors = {
        a
        for a in (
            cluster.get("anchors", [])
            if isinstance(cluster.get("anchors"), list)
            else []
        )
        if isinstance(a, str) and a
    }
    access_required = _seed_collect_access_required(cluster)
    member_order = _member_ids(cluster)
    member_set = set(member_order)

    important_objects: dict[str, dict[str, dict[str, int]]] = {}
    front_votes: dict[tuple[int, int], int] = {}
    axis_votes: dict[tuple[int, int], int] = {}

    ordered_ids = [oid for oid in member_order if oid in objects_raw]
    for oid in sorted(objects_raw.keys()):
        if oid not in member_set and oid not in ordered_ids:
            ordered_ids.append(oid)

    for oid in ordered_ids:
        item = objects_raw.get(oid)
        if not isinstance(item, dict):
            continue

        front = _coerce_axis_vec(item.get("front_local"))
        axis = _coerce_axis_vec(item.get("axis_local"))

        if front is None and axis is None:
            continue
        if front is None and axis is not None:
            front = _default_front_from_axis(axis)
        if axis is None and front is not None:
            axis = _axis_from_front(front)

        if front is None or axis is None:
            continue

        include = bool(item.get("important_for_orientation"))
        include = (
            include
            or bool(item.get("has_front_access"))
            or bool(item.get("is_anchor_like"))
            or oid in anchors
            or oid in access_required
            or _looks_directional_object_id(oid)
        )

        if not include:
            continue

        important_objects[oid] = {
            "front_local": _vec_to_dict(front),
            "axis_local": _vec_to_dict(axis),
        }

        weight = _orientation_vote_weight(
            oid=oid,
            access_required=access_required,
            anchors=anchors,
        )
        front_votes[front] = front_votes.get(front, 0) + weight
        axis_votes[axis] = axis_votes.get(axis, 0) + weight

    if cluster_axis is None:
        if axis_votes:
            cluster_axis = _pick_best_axis_vote(axis_votes, fallback=(1, 0))
        else:
            cluster_axis = (1, 0)

    if cluster_front is None:
        if front_votes:
            cluster_front = _infer_cluster_front_local_fallback(
                cluster_axis=cluster_axis,
                front_votes=front_votes,
            )
        else:
            cluster_front = _default_front_from_axis(cluster_axis)

    return {
        "cluster_front_local": _vec_to_dict(cluster_front),
        "cluster_axis_local": _vec_to_dict(cluster_axis),
        "important_objects": important_objects,
    }


def _coerce_axis_vec(value: Any) -> tuple[int, int] | None:
    dx: int | None = None
    dy: int | None = None

    if isinstance(value, dict):
        try:
            dx = int(value.get("dx", 0))
            dy = int(value.get("dy", 0))
        except Exception:
            return None
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            dx = int(value[0])
            dy = int(value[1])
        except Exception:
            return None
    else:
        return None

    if dx is None or dy is None:
        return None
    if (dx, dy) not in {(1, 0), (-1, 0), (0, 1), (0, -1)}:
        return None
    return (dx, dy)


# ---------------------------
# Fallback orientation helpers
# ---------------------------


def _select_orientation_important_object_ids(
    *,
    cluster: dict[str, Any],
    rect_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    members = _member_ids(cluster)
    anchors = {
        a
        for a in (
            cluster.get("anchors", [])
            if isinstance(cluster.get("anchors"), list)
            else []
        )
        if isinstance(a, str) and a
    }
    access_required = _seed_collect_access_required(cluster)

    selected: list[str] = []
    for oid in members:
        if oid not in rect_by_id:
            continue
        if (
            oid in anchors
            or oid in access_required
            or _looks_directional_object_id(oid)
        ):
            selected.append(oid)

    out: list[str] = []
    seen: set[str] = set()
    for oid in selected:
        if oid in seen:
            continue
        seen.add(oid)
        out.append(oid)
    return out


def _looks_directional_object_id(oid: str) -> bool:
    s = oid.lower()
    keywords = (
        "sofa",
        "armchair",
        "chair",
        "bench",
        "bed",
        "desk",
        "table",
        "island",
        "fridge",
        "cabinet",
        "pantry",
        "wardrobe",
        "dresser",
        "bookshelf",
        "console",
        "tv",
        "monitor",
        "sink",
        "stove",
        "dishwasher",
        "washer",
        "dryer",
        "toilet",
        "vanity",
    )
    return any(k in s for k in keywords)


def _orientation_vote_weight(
    *,
    oid: str,
    access_required: set[str],
    anchors: set[str],
) -> int:
    if oid in access_required:
        return 4
    if oid in anchors:
        return 3
    if _looks_directional_object_id(oid):
        return 2
    return 1


def _infer_cluster_axis_local(rects: list[dict[str, Any]]) -> tuple[int, int]:
    if not rects:
        return (1, 0)

    bbox = _local_bbox_from_rects(rects)
    span_x = int(bbox.get("max_x", 0)) - int(bbox.get("min_x", 0))
    span_y = int(bbox.get("max_y", 0)) - int(bbox.get("min_y", 0))

    if span_x >= span_y:
        return (1, 0)
    return (0, 1)


def _pick_best_axis_vote(
    axis_votes: dict[tuple[int, int], int],
    *,
    fallback: tuple[int, int],
) -> tuple[int, int]:
    normalized_votes: dict[tuple[int, int], int] = {}
    for axis, weight in axis_votes.items():
        canonical = _canonical_axis(axis)
        normalized_votes[canonical] = normalized_votes.get(canonical, 0) + int(weight)

    best_axis = fallback
    best_score: tuple[int, int] | None = None

    for axis in ((1, 0), (0, 1)):
        score = (
            normalized_votes.get(axis, 0),
            1 if axis == fallback else 0,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_axis = axis

    return best_axis


def _infer_cluster_front_local_fallback(
    *,
    cluster_axis: tuple[int, int],
    front_votes: dict[tuple[int, int], int],
) -> tuple[int, int]:
    if not front_votes:
        return _default_front_from_axis(cluster_axis)

    best_front: tuple[int, int] | None = None
    best_score: tuple[int, int, int] | None = None

    for front in ((0, 1), (1, 0), (-1, 0), (0, -1)):
        weight = front_votes.get(front, 0)
        perpendicular_bonus = 1 if _is_perpendicular(front, cluster_axis) else 0
        priority = _front_priority(front)
        score = (weight, perpendicular_bonus, -priority)
        if best_score is None or score > best_score:
            best_score = score
            best_front = front

    if best_front is None or front_votes.get(best_front, 0) <= 0:
        return _default_front_from_axis(cluster_axis)
    return best_front


def _infer_object_front_local_fallback(
    *,
    oid: str,
    rect: dict[str, Any],
    placement: dict[str, Any],
    spec: dict[str, Any],
    cluster_bbox: dict[str, int],
    access_required: set[str],
) -> tuple[int, int] | None:
    base_front = spec.get("front")
    rot = int(placement.get("rot", 0)) % 360

    if base_front in {"top", "bottom", "left", "right"}:
        final_side = _seed_rotate_side(str(base_front), rot)
        return _side_to_vec(final_side)

    if oid in access_required:
        return _outward_vec_from_bbox(rect, cluster_bbox)

    return None


def _infer_object_axis_local_fallback(
    *,
    rect: dict[str, Any],
    front: tuple[int, int] | None,
) -> tuple[int, int] | None:
    w = int(rect.get("w", 0) or 0)
    h = int(rect.get("h", 0) or 0)

    if w > h:
        return (1, 0)
    if h > w:
        return (0, 1)
    if front is not None:
        return _axis_from_front(front)
    return None


def _side_to_vec(side: str) -> tuple[int, int] | None:
    mapping = {
        "top": (0, 1),
        "bottom": (0, -1),
        "left": (-1, 0),
        "right": (1, 0),
    }
    return mapping.get(side)


def _vec_to_dict(vec: tuple[int, int]) -> dict[str, int]:
    return {"dx": int(vec[0]), "dy": int(vec[1])}


def _canonical_axis(vec: tuple[int, int]) -> tuple[int, int]:
    dx, dy = int(vec[0]), int(vec[1])
    if abs(dx) == 1:
        return (1, 0)
    return (0, 1)


def _axis_from_front(front: tuple[int, int]) -> tuple[int, int]:
    dx, dy = int(front[0]), int(front[1])
    if abs(dx) == 1:
        return (0, 1)
    return (1, 0)


def _default_front_from_axis(axis: tuple[int, int]) -> tuple[int, int]:
    if _canonical_axis(axis) == (1, 0):
        return (0, 1)
    return (1, 0)


def _is_perpendicular(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return (int(a[0]) * int(b[0]) + int(a[1]) * int(b[1])) == 0


def _front_priority(front: tuple[int, int]) -> int:
    order = {
        (0, 1): 0,
        (1, 0): 1,
        (-1, 0): 2,
        (0, -1): 3,
    }
    return order.get(front, 99)


def _outward_vec_from_bbox(
    rect: dict[str, Any],
    cluster_bbox: dict[str, int],
) -> tuple[int, int]:
    cx = float(rect.get("x", 0)) + float(rect.get("w", 0)) / 2.0
    cy = float(rect.get("y", 0)) + float(rect.get("h", 0)) / 2.0

    bx = (int(cluster_bbox.get("min_x", 0)) + int(cluster_bbox.get("max_x", 0))) / 2.0
    by = (int(cluster_bbox.get("min_y", 0)) + int(cluster_bbox.get("max_y", 0))) / 2.0

    dx = cx - bx
    dy = cy - by

    if abs(dx) >= abs(dy):
        return (1, 0) if dx >= 0 else (-1, 0)
    return (0, 1) if dy >= 0 else (0, -1)


def _is_context_length_exceeded(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    if code == "context_length_exceeded":
        return True
    text = str(exc)
    return ("context_length_exceeded" in text) or ("maximum context length" in text)


def _error_score(tool_output: dict[str, Any]) -> int:
    if not isinstance(tool_output, dict):
        return 10**9
    if tool_output.get("result") == "VALID":
        return 0

    errors = tool_output.get("errors")
    if not isinstance(errors, list):
        return 10**8

    weight = {
        "MISSING_OBJECT_SPECS": 100000,
        "INVALID_DIMS": 100000,
        "UNKNOWN_OBJECT": 50000,
        "GRID_VIOLATION": 2000,
        "ROTATION_NOT_ALLOWED": 1500,
        "CONTAIN_VIOLATION": 3500,
        "ANCHOR_VIOLATION": 3000,
        "DOCK_VIOLATION": 3000,
        "OVERLAP": 5000,
        "ACCESS_BLOCKED": 4500,
        "LOCAL_BBOX_BUDGET_EXCEEDED": 2500,
        "LOCAL_OUTLINE_BUDGET_EXCEEDED": 2500,
        "LOCAL_HULL_BUDGET_EXCEEDED": 2500,
        "LOCAL_FILL_RATIO_TOO_LOW": 2500,
    }

    score = 0
    for err in errors:
        if not isinstance(err, dict):
            score += 10000
            continue
        code = err.get("code")
        score += weight.get(code, 4000)
    return score


def _extract_layout_key(tool_output: dict[str, Any]) -> tuple[float, ...]:
    debug = tool_output.get("debug", {}) if isinstance(tool_output, dict) else {}
    constraint_scores = (
        debug.get("constraint_scores", {}) if isinstance(debug, dict) else {}
    )
    lexi = (
        constraint_scores.get("lexicographic_key", [])
        if isinstance(constraint_scores, dict)
        else []
    )

    quality = tool_output.get("quality", {}) if isinstance(tool_output, dict) else {}
    bbox = quality.get("bbox", {}) if isinstance(quality, dict) else {}

    compact_score = float(quality.get("compact_score", 10**12) or 0.0)
    bbox_area = float(bbox.get("area_mm2", 10**12) or 0.0)
    max_span = float(bbox.get("max_span_mm", 10**12) or 0.0)
    fill_ratio_bbox = float(quality.get("fill_ratio_bbox", 0.0) or 0.0)
    fill_ratio_hull = float(quality.get("fill_ratio_hull", 0.0) or 0.0)
    aspect_ratio = float(bbox.get("aspect_ratio", 10**6) or 0.0)

    def _at(i: int, default: float) -> float:
        if isinstance(lexi, list) and i < len(lexi):
            try:
                return float(lexi[i])
            except Exception:
                return default
        return default

    return (
        _at(0, 0.0),
        _at(1, 0.0),
        _at(2, 0.0),
        _at(3, compact_score),
        _at(4, 0.0),
        compact_score,
        bbox_area,
        max_span,
        -fill_ratio_bbox,
        -fill_ratio_hull,
        aspect_ratio,
    )


def _tool_rank(tool_output: dict[str, Any]) -> tuple[Any, ...]:
    if not isinstance(tool_output, dict):
        return (
            9,
            10**9,
            10**9,
            10**9,
            10**9,
            10**9,
            10**9,
            10**9,
            10**9,
            10**9,
            10**9,
            10**9,
        )

    if tool_output.get("result") == "VALID":
        return (0, *_extract_layout_key(tool_output))

    quality = tool_output.get("quality", {}) if isinstance(tool_output, dict) else {}
    bbox = quality.get("bbox", {}) if isinstance(quality, dict) else {}

    compact_score = float(quality.get("compact_score", 10**12) or 0.0)
    bbox_area = float(bbox.get("area_mm2", 10**12) or 0.0)
    max_span = float(bbox.get("max_span_mm", 10**12) or 0.0)

    return (
        1,
        float(_error_score(tool_output)),
        compact_score,
        bbox_area,
        max_span,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


def _rank_better(
    candidate_rank: tuple[Any, ...], baseline_rank: tuple[Any, ...]
) -> bool:
    return candidate_rank < baseline_rank


def _reason_rank(reason: Any) -> int:
    rank = {
        "GRID_VIOLATION": 0,
        "ROTATION_NOT_ALLOWED": 1,
        "HOST_OVERLAP_ESCAPE_LEFT": 2,
        "HOST_OVERLAP_ESCAPE_RIGHT": 2,
        "HOST_OVERLAP_ESCAPE_UP": 3,
        "HOST_OVERLAP_ESCAPE_DOWN": 3,
        "HOST_ACCESS_ESCAPE_LEFT": 4,
        "HOST_ACCESS_ESCAPE_RIGHT": 4,
        "HOST_ACCESS_ESCAPE_UP": 5,
        "HOST_ACCESS_ESCAPE_DOWN": 5,
        "OVERLAP": 6,
        "ACCESS_BLOCKED": 7,
        "DOCK_VIOLATION": 8,
        "ANCHOR_VIOLATION": 9,
        "CONTAIN_VIOLATION": 10,
        "HOST_ROTATION_ESCAPE": 11,
        "COMPACT_LEFT_EDGE": 12,
        "COMPACT_RIGHT_EDGE": 12,
        "COMPACT_BOTTOM_EDGE": 13,
        "COMPACT_TOP_EDGE": 13,
    }
    return rank.get(reason, 99)


def _movement_cost(move: dict[str, Any]) -> int:
    dx = int(move.get("dx", 0) or 0)
    dy = int(move.get("dy", 0) or 0)
    rot_penalty = 500 if "new_rot" in move and move.get("new_rot") is not None else 0
    return abs(dx) + abs(dy) + rot_penalty


def _verifier_preview(tool_output: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(tool_output, dict):
        return {"result": "INVALID", "error_count": 999999}

    errors = tool_output.get("errors")
    preview = errors[:6] if isinstance(errors, list) else []

    quality = tool_output.get("quality", {}) if isinstance(tool_output, dict) else {}
    bbox = quality.get("bbox", {}) if isinstance(quality, dict) else {}

    return {
        "result": tool_output.get("result"),
        "error_count": len(errors) if isinstance(errors, list) else None,
        "errors_preview": preview,
        "error_score": _error_score(tool_output),
        "tool_rank": list(_tool_rank(tool_output)),
        "compact_score": quality.get("compact_score"),
        "bbox_area_mm2": bbox.get("area_mm2"),
        "max_span_mm": bbox.get("max_span_mm"),
    }


def _apply_single_move(
    placements: list[dict[str, Any]],
    move: dict[str, Any],
) -> list[dict[str, Any]]:
    move_object = move.get("move_object")
    if not isinstance(move_object, str):
        return _canonicalize_local_placements(placements)

    patched = _canonicalize_local_placements(placements)
    out: list[dict[str, Any]] = []

    for p in patched:
        if p["id"] != move_object:
            out.append(dict(p))
            continue

        q = dict(p)

        if "new_x" in move and move.get("new_x") is not None:
            q["x"] = int(move["new_x"])
        else:
            q["x"] = int(q["x"]) + int(move.get("dx", 0) or 0)

        if "new_y" in move and move.get("new_y") is not None:
            q["y"] = int(move["new_y"])
        else:
            q["y"] = int(q["y"]) + int(move.get("dy", 0) or 0)

        if "new_rot" in move and move.get("new_rot") is not None:
            q["rot"] = int(move["new_rot"]) % 360
        else:
            q["rot"] = int(q.get("rot", 0)) % 360

        out.append(q)

    out.sort(key=lambda item: item["id"])
    return out


def _host_snap_delta(delta: int, grid_mm: int) -> int:
    if delta == 0 or grid_mm <= 0:
        return int(delta)
    sign = 1 if delta > 0 else -1
    mag = abs(int(delta))
    if mag % grid_mm == 0:
        return sign * mag
    mag = ((mag // grid_mm) + 1) * grid_mm
    return sign * mag


# ---------------------------
# Parsing helpers
# ---------------------------


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
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("ClusterComposer response must be a JSON object")
    return payload


def _sanitize_cluster_output_payload(payload: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "status",
        "cluster_id",
        "local_frame",
        "local_placements",
        "cluster_footprint",
        "orientation_meta",
        "notes",
        "missing",
    }

    notes: list[str] = []
    if isinstance(payload.get("notes"), list):
        notes = [str(x).strip() for x in payload.get("notes") if str(x).strip()]

    reason = payload.get("reason")
    if isinstance(reason, str) and reason.strip():
        notes.append(reason.strip())

    cleaned: dict[str, Any] = {k: payload[k] for k in allowed_keys if k in payload}
    if notes:
        cleaned["notes"] = notes
    elif "notes" in cleaned and not isinstance(cleaned.get("notes"), list):
        cleaned["notes"] = []
    return cleaned


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


def _member_ids(cluster: dict[str, Any]) -> list[str]:
    members = cluster.get("members") if isinstance(cluster, dict) else None
    if not isinstance(members, list):
        return []
    return [m for m in members if isinstance(m, str)]
