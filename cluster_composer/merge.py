"""Cluster composer merge utilities."""

from __future__ import annotations

from typing import Any


def merge_cluster_outputs(
    cluster_forge: dict[str, Any],
    tier_count: dict[str, Any],
) -> dict[str, Any]:
    """
    Merge ClusterForge output with TierCount decisions.

    - Groups by cluster_id.
    - Drops any `raw_text` field found in the merged output.
    - Filters cluster members/constraints/rules so ClusterComposer only receives objects that
      exist in BOTH ClusterForge.members and TierCount.decisions (and have quantity >= 1).
    """
    cluster_payload = _unwrap_payload(cluster_forge, key="clusters")
    tier_payload = _unwrap_payload(tier_count, key="decisions")

    clusters = cluster_payload.get("clusters", [])
    decisions = tier_payload.get("decisions", [])

    decision_by_type: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        obj_type = _decision_type_id(decision)
        if obj_type is None:
            continue
        qty = decision.get("quantity")
        if not isinstance(qty, int) or qty < 1:
            continue
        decision_by_type.setdefault(obj_type, decision)

    merged_clusters: list[dict[str, Any]] = []
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        cluster_id = str(cluster.get("cluster_id") or "")
        members = cluster.get("members")
        if not isinstance(members, list):
            members = []

        kept_members = [
            m
            for m in members
            if isinstance(m, str) and m in decision_by_type and m.strip()
        ]
        kept_set = set(kept_members)

        merged = dict(cluster)
        merged["members"] = kept_members

        anchors = cluster.get("anchors")
        if isinstance(anchors, list):
            merged["anchors"] = [a for a in anchors if isinstance(a, str) and a in kept_set]

        merged["decisions"] = [
            dict(decision_by_type[m], cluster_id=cluster_id) for m in kept_members
        ]

        for key in ("hard_constraints", "soft_constraints"):
            constraints = cluster.get(key)
            if not isinstance(constraints, list):
                continue
            merged[key] = [
                c
                for c in constraints
                if isinstance(c, dict)
                and _constraint_subjects(c).issubset(kept_set)
            ]

        rules = cluster.get("cluster_rules")
        if isinstance(rules, dict):
            merged["cluster_rules"] = _filter_cluster_rules(rules, kept_set)

        merged_clusters.append(merged)

    merged_output: dict[str, Any] = {
        "status": cluster_payload.get("status", "OK"),
        "clusters": merged_clusters,
        "notes": cluster_payload.get("notes", []),
        "missing": cluster_payload.get("missing", []),
    }

    _remove_key_recursive(merged_output, "raw_text")
    return merged_output


def _remove_key_recursive(value: Any, key: str) -> None:
    if isinstance(value, dict):
        if key in value:
            value.pop(key, None)
        for v in list(value.values()):
            _remove_key_recursive(v, key)
    elif isinstance(value, list):
        for item in value:
            _remove_key_recursive(item, key)


def _unwrap_payload(payload: dict[str, Any], *, key: str) -> dict[str, Any]:
    """
    Accepts either a direct payload or a wrapper with raw/parsed fields.
    Prefers `parsed`, then `raw`, then the original payload.
    """
    if isinstance(payload.get(key), list):
        return payload
    parsed = payload.get("parsed")
    if isinstance(parsed, dict) and isinstance(parsed.get(key), list):
        return parsed
    raw = payload.get("raw")
    if isinstance(raw, dict) and isinstance(raw.get(key), list):
        return raw
    return payload


def _decision_type_id(decision: dict[str, Any]) -> str | None:
    obj_type = decision.get("object_type") or decision.get("category")
    if not isinstance(obj_type, str):
        return None
    obj_type = obj_type.strip()
    return obj_type if obj_type else None


def _constraint_subjects(constraint: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for key in ("a", "b", "id"):
        value = constraint.get(key)
        if isinstance(value, str) and value:
            out.add(value)
    return out


def _filter_cluster_rules(rules: dict[str, Any], kept_ids: set[str]) -> dict[str, Any]:
    out = dict(rules)

    allowed_rotations = out.get("allowed_rotations")
    if isinstance(allowed_rotations, dict):
        out["allowed_rotations"] = {
            k: v for k, v in allowed_rotations.items() if k in kept_ids
        }

    facing = out.get("facing")
    if isinstance(facing, dict):
        out["facing"] = {k: v for k, v in facing.items() if k in kept_ids}

    access_requirements = out.get("access_requirements")
    if isinstance(access_requirements, list):
        kept_reqs: list[dict[str, Any]] = []
        for item in access_requirements:
            if not isinstance(item, dict):
                continue
            oid = item.get("id")
            if isinstance(oid, str) and oid in kept_ids:
                kept_reqs.append(item)
        out["access_requirements"] = kept_reqs

    return out
