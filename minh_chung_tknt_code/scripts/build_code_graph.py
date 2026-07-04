from __future__ import annotations

import ast
import json
import os
import re
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "docs" / "code_graph.html"
ANNOTATIONS_PATH = REPO_ROOT / "docs" / "code_annotations.json"

SOURCE_SUFFIXES = {".py", ".ts", ".tsx"}
ROOT_FILES = (
    "api/main.py",
    "frontend/src/main.tsx",
    "run_case_backend_cli.py",
    "docker-compose.yml",
    "frontend/package.json",
)
EXCLUDED_DIR_PARTS = {
    ".cache",
    ".conda_envs",
    ".conda_pkgs",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "cases",
    "dist",
    "generated",
    "logs",
    "node_modules",
    "OpenShape-Embeddings",
    "other_systems",
    "out",
    "outputs",
    "tmp",
}
EXCLUDED_PREFIXES = (
    "Thesis_Chapters_DOCX/",
    "Thesis_LaTeX/",
    "assets/three/",
    "frontend/src/vendor/",
)
CONFIG_FILES = {
    ".env.example",
    "Dockerfile",
    "app-config-template.yaml",
    "app-config.docker.yaml",
    "docker-compose.yml",
    "frontend/index.html",
    "frontend/package.json",
    "frontend/vite.config.ts",
    "requirements.txt",
    "setup_env.sh",
}
TS_IMPORT_RE = re.compile(
    r"""(?:import|export)\s+(?:[^'"]+\s+from\s+)?['"](?P<target>[^'"]+)['"]"""
)
TS_DYNAMIC_IMPORT_RE = re.compile(r"""import\(\s*['"](?P<target>[^'"]+)['"]\s*\)""")


@dataclass(frozen=True)
class SourceNode:
    path: str
    area: str
    kind: str
    lines: int
    root: bool


@dataclass(frozen=True)
class SourceSymbol:
    name: str
    qualified_name: str
    kind: str
    line_start: int
    line_end: int


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    kind: str


def main() -> int:
    source_paths = sorted(_iter_source_paths())
    py_module_by_name = _build_python_module_index(source_paths)
    node_by_path = {path: _build_node(path) for path in source_paths}
    edges = _collect_edges(source_paths, py_module_by_name, node_by_path.keys())
    graph = _build_graph_payload(node_by_path, edges)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(_render_html(graph), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    return 0


def _iter_source_paths() -> Iterable[str]:
    for current_root, dirnames, filenames in os.walk(REPO_ROOT):
        current_path = Path(current_root)
        rel_dir = current_path.relative_to(REPO_ROOT).as_posix()
        if rel_dir == ".":
            rel_dir = ""
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if not _is_excluded_dir(f"{rel_dir}/{dirname}".strip("/"))
        ]
        for filename in filenames:
            path = current_path / filename
            rel = path.relative_to(REPO_ROOT).as_posix()
            if _is_excluded(rel):
                continue
            if path.suffix in SOURCE_SUFFIXES or rel in CONFIG_FILES:
                yield rel


def _is_excluded_dir(rel_dir: str) -> bool:
    if not rel_dir:
        return False
    parts = set(Path(rel_dir).parts)
    if parts & EXCLUDED_DIR_PARTS:
        return True
    return any(rel_dir.startswith(prefix.rstrip("/")) for prefix in EXCLUDED_PREFIXES)


def _is_excluded(rel_path: str) -> bool:
    parts = set(Path(rel_path).parts)
    if parts & EXCLUDED_DIR_PARTS:
        return True
    return any(rel_path.startswith(prefix) for prefix in EXCLUDED_PREFIXES)


def _build_node(path: str) -> SourceNode:
    file_path = REPO_ROOT / path
    text = _read_text(file_path)
    return SourceNode(
        path=path,
        area=_area_for(path),
        kind=file_path.suffix.lstrip(".") or file_path.name,
        lines=len(text.splitlines()),
        root=path in ROOT_FILES,
    )


def _area_for(path: str) -> str:
    if path.startswith("frontend/src/"):
        return "frontend"
    if path.startswith("test_agent/"):
        return "tests"
    if "/" in path:
        return path.split("/", 1)[0]
    return "root"


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def _build_python_module_index(paths: Sequence[str]) -> dict[str, str]:
    module_by_name: dict[str, str] = {}
    for path in paths:
        if not path.endswith(".py"):
            continue
        module_name = path.removesuffix(".py").replace("/", ".")
        if module_name.endswith(".__init__"):
            module_name = module_name.removesuffix(".__init__")
        module_by_name[module_name] = path
    return module_by_name


def _collect_edges(
    paths: Sequence[str],
    py_module_by_name: Mapping[str, str],
    known_paths: Iterable[str],
) -> list[Edge]:
    known_path_set = set(known_paths)
    edge_keys: set[tuple[str, str, str]] = set()
    for path in paths:
        file_path = REPO_ROOT / path
        if path.endswith(".py"):
            targets = _python_import_targets(path, file_path, py_module_by_name)
            for target in targets:
                edge_keys.add((path, target, "python-import"))
        elif path.endswith((".ts", ".tsx")):
            targets = _typescript_import_targets(path, file_path, known_path_set)
            for target in targets:
                edge_keys.add((path, target, "ts-import"))
    return [
        Edge(source=source, target=target, kind=kind)
        for source, target, kind in sorted(edge_keys)
        if source != target
    ]


def _python_import_targets(
    source_path: str,
    file_path: Path,
    module_by_name: Mapping[str, str],
) -> list[str]:
    text = _read_text(file_path)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    source_module = source_path.removesuffix(".py").replace("/", ".")
    if source_module.endswith(".__init__"):
        source_module = source_module.removesuffix(".__init__")
    source_package = source_module.rsplit(".", 1)[0] if "." in source_module else ""

    targets: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = _resolve_python_module(alias.name, module_by_name)
                if target is not None:
                    targets.append(target)
        elif isinstance(node, ast.ImportFrom):
            module_name = _absolute_import_from_name(
                source_package=source_package,
                module=node.module,
                level=node.level,
            )
            if not module_name:
                continue
            for alias in node.names:
                candidate = f"{module_name}.{alias.name}"
                target = _resolve_python_module(candidate, module_by_name)
                if target is None:
                    target = _resolve_python_module(module_name, module_by_name)
                if target is not None:
                    targets.append(target)
    return sorted(set(targets))


def _absolute_import_from_name(
    *,
    source_package: str,
    module: str | None,
    level: int,
) -> str:
    if level == 0:
        return module or ""
    package_parts = source_package.split(".") if source_package else []
    keep_count = max(0, len(package_parts) - level + 1)
    base = ".".join(package_parts[:keep_count])
    if module:
        return f"{base}.{module}" if base else module
    return base


def _resolve_python_module(
    module_name: str,
    module_by_name: Mapping[str, str],
) -> str | None:
    parts = module_name.split(".")
    for size in range(len(parts), 0, -1):
        candidate = ".".join(parts[:size])
        target = module_by_name.get(candidate)
        if target is not None:
            return target
    return None


def _typescript_import_targets(
    source_path: str,
    file_path: Path,
    known_paths: set[str],
) -> list[str]:
    text = _read_text(file_path)
    targets: list[str] = []
    for match in list(TS_IMPORT_RE.finditer(text)) + list(
        TS_DYNAMIC_IMPORT_RE.finditer(text)
    ):
        raw_target = match.group("target")
        target = _resolve_relative_ts_import(source_path, raw_target, known_paths)
        if target is not None:
            targets.append(target)
    return sorted(set(targets))


def _resolve_relative_ts_import(
    source_path: str,
    raw_target: str,
    known_paths: set[str],
) -> str | None:
    if not raw_target.startswith("."):
        return None
    base_dir = Path(source_path).parent
    raw_path = (base_dir / raw_target).as_posix()
    candidates = [
        raw_path,
        f"{raw_path}.ts",
        f"{raw_path}.tsx",
        f"{raw_path}.js",
        f"{raw_path}.jsx",
        f"{raw_path}/index.ts",
        f"{raw_path}/index.tsx",
    ]
    for candidate in candidates:
        normalized = Path(candidate).as_posix()
        if normalized in known_paths:
            return normalized
    return None


def _build_graph_payload(
    node_by_path: Mapping[str, SourceNode],
    edges: Sequence[Edge],
) -> dict[str, object]:
    imported_by: dict[str, list[str]] = {path: [] for path in node_by_path}
    imports: dict[str, list[str]] = {path: [] for path in node_by_path}
    for edge in edges:
        imports[edge.source].append(edge.target)
        imported_by[edge.target].append(edge.source)

    annotations = _load_annotations()
    depths = _root_depths(node_by_path.keys(), edges)
    nodes = []
    for path, node in sorted(node_by_path.items()):
        note_count = len(annotations.get(path, []))
        symbols = _extract_symbols(path)
        nodes.append(
            {
                "id": path,
                "label": Path(path).name,
                "area": node.area,
                "kind": node.kind,
                "lines": node.lines,
                "root": node.root,
                "depth": depths.get(path),
                "imports": sorted(imports[path]),
                "importedBy": sorted(imported_by[path]),
                "noteCount": note_count,
                "symbolCount": len(symbols),
                "removalImpact": _removal_impact_summary(
                    path=path,
                    is_root=node.root,
                    imported_by=imported_by[path],
                ),
            }
        )

    return {
        "generatedFrom": REPO_ROOT.name,
        "rootFiles": [path for path in ROOT_FILES if path in node_by_path],
        "nodes": nodes,
        "edges": [
            {"source": edge.source, "target": edge.target, "kind": edge.kind}
            for edge in edges
        ],
        "areas": sorted({node.area for node in node_by_path.values()}),
        "annotations": annotations,
        "sources": {
            path: {
                "text": _read_text(REPO_ROOT / path),
                "symbols": [
                    {
                        "name": symbol.name,
                        "qualifiedName": symbol.qualified_name,
                        "kind": symbol.kind,
                        "line_start": symbol.line_start,
                        "line_end": symbol.line_end,
                    }
                    for symbol in _extract_symbols(path)
                ],
            }
            for path in sorted(node_by_path)
        },
    }


def _extract_symbols(path: str) -> list[SourceSymbol]:
    file_path = REPO_ROOT / path
    text = _read_text(file_path)
    if path.endswith(".py"):
        return _extract_python_symbols(text)
    if path.endswith((".ts", ".tsx")):
        return _extract_typescript_symbols(text)
    return []


def _extract_python_symbols(text: str) -> list[SourceSymbol]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    symbols: list[SourceSymbol] = []

    def walk(nodes: Sequence[ast.stmt], parents: tuple[str, ...] = ()) -> None:
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                qualified = (*parents, node.name)
                symbols.append(
                    SourceSymbol(
                        name=node.name,
                        qualified_name=".".join(qualified),
                        kind="class",
                        line_start=node.lineno,
                        line_end=int(getattr(node, "end_lineno", node.lineno)),
                    )
                )
                walk(node.body, qualified)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = (*parents, node.name)
                symbols.append(
                    SourceSymbol(
                        name=node.name,
                        qualified_name=".".join(qualified),
                        kind="async function"
                        if isinstance(node, ast.AsyncFunctionDef)
                        else "function",
                        line_start=node.lineno,
                        line_end=int(getattr(node, "end_lineno", node.lineno)),
                    )
                )
                walk(node.body, qualified)

    walk(tree.body)
    return sorted(
        symbols,
        key=lambda symbol: (
            symbol.line_start,
            symbol.line_end,
            symbol.qualified_name,
        ),
    )


def _extract_typescript_symbols(text: str) -> list[SourceSymbol]:
    symbols: list[SourceSymbol] = []
    class_pattern = re.compile(
        r"^\s*(?:export\s+default\s+|export\s+)?class\s+([A-Za-z0-9_$]+)"
    )
    function_pattern = re.compile(
        r"^\s*(?:export\s+default\s+|export\s+)?(?:async\s+)?function\s+([A-Za-z0-9_$]+)"
    )
    const_pattern = re.compile(r"^\s*(?:export\s+)?const\s+([A-Za-z0-9_$]+)\s*=")
    patterns = (
        ("class", class_pattern),
        ("function", function_pattern),
        ("function", const_pattern),
    )
    lines = text.splitlines()
    for index, line in enumerate(lines, start=1):
        for kind, pattern in patterns:
            match = pattern.search(line)
            if not match:
                continue
            if pattern is const_pattern and not _typescript_const_is_arrow(
                lines, index
            ):
                continue
            name = match.group(1)
            symbols.append(
                SourceSymbol(
                    name=name,
                    qualified_name=name,
                    kind=kind,
                    line_start=index,
                    line_end=_typescript_block_end(lines, index),
                )
            )
            break
    return symbols


def _typescript_const_is_arrow(lines: Sequence[str], start_line: int) -> bool:
    header = []
    for index in range(start_line, min(len(lines), start_line + 30) + 1):
        line = lines[index - 1]
        header.append(line)
        joined = "\n".join(header)
        expression = joined.split("=", 1)[1] if "=" in joined else joined
        if re.search(
            r"^\s*(?:async\s*)?(?:\([^;=]*?\)|[A-Za-z_$][A-Za-z0-9_$]*)\s*(?::[^=;{}]+)?=>",
            expression,
            re.DOTALL,
        ):
            return True
        if line.rstrip().endswith(";"):
            return False
    return False


def _typescript_block_end(lines: Sequence[str], start_line: int) -> int:
    balance = 0
    started = False
    for index in range(start_line, len(lines) + 1):
        line = lines[index - 1]
        balance += line.count("{") - line.count("}")
        if "{" in line:
            started = True
        if started and balance <= 0 and index > start_line:
            return index
        if not started and line.rstrip().endswith((";", ",")):
            return index
    return start_line


def _load_annotations() -> dict[str, list[dict[str, object]]]:
    if not ANNOTATIONS_PATH.exists():
        return {}
    try:
        payload = json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, dict):
        return {}

    annotations: dict[str, list[dict[str, object]]] = {}
    for path, records in files.items():
        if not isinstance(path, str) or not isinstance(records, list):
            continue
        annotations[path] = [record for record in records if isinstance(record, dict)]
    return annotations


def _root_depths(paths: Iterable[str], edges: Sequence[Edge]) -> dict[str, int | None]:
    adjacency: dict[str, list[str]] = {path: [] for path in paths}
    for edge in edges:
        adjacency.setdefault(edge.source, []).append(edge.target)

    depths: dict[str, int | None] = {path: None for path in adjacency}
    queue: deque[str] = deque()
    for root in ROOT_FILES:
        if root not in depths:
            continue
        depths[root] = 0
        queue.append(root)

    while queue:
        current = queue.popleft()
        current_depth = depths[current]
        if current_depth is None:
            continue
        for target in adjacency.get(current, []):
            previous = depths.get(target)
            next_depth = current_depth + 1
            if previous is not None and previous <= next_depth:
                continue
            depths[target] = next_depth
            queue.append(target)
    return depths


def _removal_impact_summary(
    *,
    path: str,
    is_root: bool,
    imported_by: Sequence[str],
) -> str:
    if is_root:
        return "Entry/config root: deleting it can break application boot or local run setup."
    if imported_by:
        callers = ", ".join(imported_by[:3])
        suffix = "" if len(imported_by) <= 3 else f", +{len(imported_by) - 3} more"
        return f"Imported by {len(imported_by)} file(s): {callers}{suffix}."
    if path.startswith("test_agent/"):
        return "Test-only file: deleting it lowers regression coverage."
    return "No internal import caller found. Check routes, scripts, config strings, and manual workflows before deleting."


def _render_html(graph: Mapping[str, object]) -> str:
    graph_json = json.dumps(graph, ensure_ascii=False, indent=2).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__GRAPH_JSON__", graph_json)


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TKNT Code Reader</title>
  <style>
    :root {
      --bg: #f7f7f4;
      --panel: #ffffff;
      --ink: #222222;
      --muted: #656565;
      --line: #d6d2ca;
      --accent: #1f7a8c;
      --accent-2: #b43f5a;
      --root: #214e34;
      --code-bg: #fbfbf9;
      --code-line: #efede8;
      --shadow: 0 10px 24px rgba(30, 30, 30, 0.08);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      overflow: hidden;
    }

    .app {
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr) 430px;
      height: 100vh;
    }

    aside, main {
      min-width: 0;
      min-height: 0;
    }

    .sidebar, .details, .reader {
      background: var(--panel);
    }

    .sidebar {
      border-right: 1px solid var(--line);
      padding: 18px;
      overflow: auto;
    }

    .reader {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      min-width: 0;
    }

    .details {
      border-right: 0;
      border-left: 1px solid var(--line);
      padding: 18px;
      overflow: auto;
    }

    h1 {
      margin: 0 0 8px;
      font-size: 20px;
      letter-spacing: 0;
    }

    h2 {
      margin: 20px 0 8px;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0;
      color: var(--muted);
    }

    p {
      margin: 0 0 12px;
      color: var(--muted);
      line-height: 1.45;
      font-size: 13px;
    }

    input, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      font: inherit;
      background: #fff;
      color: var(--ink);
      outline: none;
    }

    input:focus, select:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(31, 122, 140, 0.14);
    }

    .control {
      display: grid;
      gap: 6px;
      margin: 14px 0;
    }

    label {
      font-size: 12px;
      color: var(--muted);
    }

    .area-list {
      display: grid;
      gap: 8px;
    }

    .area-row {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      color: var(--ink);
    }

    .area-row input {
      width: auto;
    }

    .stat-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }

    .stat {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfbf9;
    }

    .stat strong {
      display: block;
      font-size: 18px;
    }

    .stat span {
      color: var(--muted);
      font-size: 12px;
    }

    .canvas-wrap {
      position: relative;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background:
        linear-gradient(rgba(0, 0, 0, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 0, 0, 0.03) 1px, transparent 1px);
      background-size: 28px 28px;
    }

    .toolbar {
      position: absolute;
      left: 16px;
      top: 16px;
      z-index: 4;
      display: flex;
      gap: 8px;
      align-items: center;
      background: rgba(255, 255, 255, 0.9);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
    }

    button {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 8px;
      padding: 8px 10px;
      font: inherit;
      cursor: pointer;
    }

    button:hover {
      border-color: var(--accent);
    }

    .file-path {
      overflow-wrap: anywhere;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      color: #2b4b54;
      background: #f1f4f4;
      border-radius: 8px;
      padding: 10px;
    }

    .reader-header {
      display: grid;
      gap: 10px;
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.94);
    }

    .reader-title-row {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
    }

    .reader-title {
      min-width: 0;
    }

    .reader-title h1 {
      margin-bottom: 4px;
      overflow-wrap: anywhere;
    }

    .reader-meta {
      color: var(--muted);
      font-size: 12px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      overflow-wrap: anywhere;
    }

    .reader-controls {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 180px;
      gap: 10px;
    }

    .code-body {
      min-height: 0;
      overflow: auto;
      background: var(--code-bg);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.55;
      tab-size: 4;
    }

    .empty-code {
      padding: 30px;
      color: var(--muted);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .code-line {
      display: grid;
      grid-template-columns: 58px minmax(0, 1fr);
      min-height: 21px;
      border-left: 3px solid transparent;
      cursor: default;
    }

    .code-line:hover {
      background: #f1eee8;
    }

    .code-line.in-symbol {
      cursor: pointer;
    }

    .code-line.in-symbol .code-text {
      background: rgba(31, 122, 140, 0.04);
    }

    .code-line.selected-symbol {
      background: rgba(180, 63, 90, 0.08);
    }

    .code-line.has-note {
      border-left-color: var(--accent);
    }

    .line-no {
      color: #8b857b;
      text-align: right;
      padding: 0 12px 0 8px;
      user-select: none;
      border-right: 1px solid var(--code-line);
    }

    .code-text {
      white-space: pre;
      overflow: visible;
      padding: 0 16px;
    }

    .pill-row {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 10px 0;
    }

    .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 12px;
      color: var(--muted);
      background: #fbfbf9;
    }

    .section-actions {
      display: grid;
      gap: 8px;
      margin-top: 8px;
    }

    .list {
      display: grid;
      gap: 8px;
      margin: 0;
      padding: 0;
      list-style: none;
    }

    .list button {
      width: 100%;
      text-align: left;
      overflow-wrap: anywhere;
      font-size: 12px;
      line-height: 1.35;
    }

    .list button.active {
      border-color: var(--accent-2);
      background: #fff8fa;
    }

    .file-list {
      max-height: 34vh;
      overflow: auto;
      padding-right: 2px;
    }

    .annotation {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      margin: 8px 0;
      background: #fbfbf9;
      font-size: 12px;
    }

    .annotation strong {
      display: block;
      margin-bottom: 4px;
    }

    .annotation p {
      margin-top: 8px;
      margin-bottom: 0;
    }

    .hint {
      border-left: 3px solid var(--accent);
      padding-left: 10px;
    }

    @media (max-width: 1100px) {
      .app {
        grid-template-columns: 280px 1fr;
      }

      .details {
        display: none;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <h1>TKNT Code Reader</h1>
      <p>Open a file, click a function or class, and read saved explanations from the side ledger.</p>
      <div class="stat-grid">
        <div class="stat"><strong id="nodeCount">0</strong><span>files</span></div>
        <div class="stat"><strong id="symbolCount">0</strong><span>symbols</span></div>
      </div>
      <div class="control">
        <label for="searchBox">Search path</label>
        <input id="searchBox" type="search" placeholder="pipeline/orchestrator" />
      </div>
      <div class="control">
        <label for="depthSelect">Root distance</label>
        <select id="depthSelect">
          <option value="all">All files</option>
          <option value="0">Root files only</option>
          <option value="1">Depth <= 1</option>
          <option value="2">Depth <= 2</option>
          <option value="3">Depth <= 3</option>
          <option value="unreached">Unreached by roots</option>
        </select>
      </div>
      <h2>Areas</h2>
      <div id="areaList" class="area-list"></div>
      <h2>Root Files</h2>
      <ul id="rootList" class="list"></ul>
      <h2>Matching Files</h2>
      <ul id="fileList" class="list file-list"></ul>
    </aside>
    <main class="reader">
      <div class="reader-header">
        <div class="reader-title-row">
          <div class="reader-title">
            <h1 id="readerTitle">Select a file</h1>
            <div id="readerPath" class="reader-meta">No file selected</div>
          </div>
          <button id="copyPathButton" type="button">Copy path</button>
        </div>
        <div class="reader-controls">
          <select id="symbolSelect" aria-label="Jump to symbol"></select>
          <select id="annotationSelect" aria-label="Jump to annotation"></select>
        </div>
      </div>
      <div id="codeBody" class="code-body"></div>
    </main>
    <aside class="details">
      <h1 id="detailTitle">Select a file</h1>
      <p id="detailIntro">Saved explanations appear here when a file or function is selected.</p>
      <div id="detailBody"></div>
    </aside>
  </div>
  <script id="graph-data" type="application/json">__GRAPH_JSON__</script>
  <script>
    const graph = JSON.parse(document.getElementById('graph-data').textContent);
    const nodes = graph.nodes;
    const nodeById = new Map(nodes.map((node) => [node.id, node]));
    const sourceById = graph.sources || {};
    const areaColors = new Map();
    const palette = ['#1f7a8c', '#b43f5a', '#7768ae', '#3d6b35', '#a95d24', '#335c81', '#7a5c2e', '#4f6d7a'];
    graph.areas.forEach((area, index) => areaColors.set(area, palette[index % palette.length]));

    const nodeCount = document.getElementById('nodeCount');
    const symbolCount = document.getElementById('symbolCount');
    const searchBox = document.getElementById('searchBox');
    const depthSelect = document.getElementById('depthSelect');
    const areaList = document.getElementById('areaList');
    const rootList = document.getElementById('rootList');
    const fileList = document.getElementById('fileList');
    const readerTitle = document.getElementById('readerTitle');
    const readerPath = document.getElementById('readerPath');
    const symbolSelect = document.getElementById('symbolSelect');
    const annotationSelect = document.getElementById('annotationSelect');
    const codeBody = document.getElementById('codeBody');
    const copyPathButton = document.getElementById('copyPathButton');
    const detailTitle = document.getElementById('detailTitle');
    const detailIntro = document.getElementById('detailIntro');
    const detailBody = document.getElementById('detailBody');

    let selectedId = null;
    let selectedSymbolIndex = null;
    const enabledAreas = new Set(graph.areas);

    nodeCount.textContent = nodes.length.toString();
    symbolCount.textContent = nodes.reduce((total, node) => total + Number(node.symbolCount || 0), 0).toString();
    renderAreaFilters();
    renderRootList();
    renderFileList();
    selectNode(graph.rootFiles[0] || nodes[0]?.id || null);

    searchBox.addEventListener('input', renderFileList);
    depthSelect.addEventListener('change', renderFileList);
    symbolSelect.addEventListener('change', () => {
      const value = symbolSelect.value;
      if (value === '') {
        selectSymbol(null);
      } else {
        selectSymbol(Number(value), { scroll: true });
      }
    });
    annotationSelect.addEventListener('change', () => {
      const line = Number(annotationSelect.value || 0);
      if (line > 0) scrollToLine(line);
    });
    copyPathButton.addEventListener('click', async () => {
      if (!selectedId) return;
      try {
        await navigator.clipboard.writeText(selectedId);
        copyPathButton.textContent = 'Copied';
        setTimeout(() => { copyPathButton.textContent = 'Copy path'; }, 900);
      } catch {
        copyPathButton.textContent = 'Copy failed';
        setTimeout(() => { copyPathButton.textContent = 'Copy path'; }, 900);
      }
    });

    function renderAreaFilters() {
      areaList.innerHTML = '';
      graph.areas.forEach((area) => {
        const row = document.createElement('label');
        row.className = 'area-row';
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = true;
        checkbox.addEventListener('change', () => {
          if (checkbox.checked) enabledAreas.add(area);
          else enabledAreas.delete(area);
          renderFileList();
        });
        const color = document.createElement('span');
        color.style.width = '10px';
        color.style.height = '10px';
        color.style.borderRadius = '50%';
        color.style.background = areaColors.get(area);
        const text = document.createElement('span');
        text.textContent = area;
        row.append(checkbox, color, text);
        areaList.append(row);
      });
    }

    function renderRootList() {
      rootList.innerHTML = '';
      graph.rootFiles.forEach((path) => {
        const item = document.createElement('li');
        const button = document.createElement('button');
        button.type = 'button';
        button.textContent = path;
        button.addEventListener('click', () => selectNode(path));
        item.append(button);
        rootList.append(item);
      });
    }

    function renderFileList() {
      const visible = visibleNodes();
      fileList.innerHTML = '';
      if (!visible.length) {
        const item = document.createElement('li');
        item.innerHTML = '<p>No files match the current filters.</p>';
        fileList.append(item);
        return;
      }
      visible.slice(0, 220).forEach((node) => {
        const item = document.createElement('li');
        const button = document.createElement('button');
        button.type = 'button';
        button.className = node.id === selectedId ? 'active' : '';
        button.innerHTML = `${escapeHtml(node.id)}<br><span style="color: var(--muted)">${node.lines} lines | ${node.symbolCount || 0} symbols | ${node.noteCount || 0} notes</span>`;
        button.addEventListener('click', () => selectNode(node.id));
        item.append(button);
        fileList.append(item);
      });
    }

    function visibleNodeIds() {
      return new Set(visibleNodes().map((node) => node.id));
    }

    function visibleNodes() {
      const search = searchBox.value.trim().toLowerCase();
      const depthValue = depthSelect.value;
      return nodes.filter((node) => {
        if (!enabledAreas.has(node.area)) return false;
        if (search && !node.id.toLowerCase().includes(search)) return false;
        if (depthValue === 'unreached' && node.depth !== null) return false;
        if (depthValue !== 'all' && depthValue !== 'unreached') {
          const maxDepth = Number(depthValue);
          if (node.depth === null || node.depth > maxDepth) return false;
        }
        return true;
      }).sort((a, b) => {
        const depthA = a.depth ?? 99;
        const depthB = b.depth ?? 99;
        return depthA - depthB || a.area.localeCompare(b.area) || a.id.localeCompare(b.id);
      });
    }

    function relatedIds(id) {
      const related = new Set();
      if (!id) return related;
      related.add(id);
      const node = nodeById.get(id);
      if (!node) return related;
      node.imports.forEach((value) => related.add(value));
      node.importedBy.forEach((value) => related.add(value));
      return related;
    }

    function selectNode(id, options = {}) {
      if (!id || !nodeById.has(id)) return;
      selectedId = id;
      selectedSymbolIndex = options.symbolIndex ?? null;
      renderFileList();
      renderReader(nodeById.get(id));
      renderDetails(nodeById.get(id));
      if (typeof selectedSymbolIndex === 'number' && options.scroll !== false) {
        const symbol = currentSymbols()[selectedSymbolIndex];
        if (symbol) scrollToLine(symbol.line_start);
      }
    }

    function selectSymbol(index, options = {}) {
      selectedSymbolIndex = index;
      symbolSelect.value = index == null ? '' : String(index);
      renderCode(nodeById.get(selectedId));
      renderDetails(nodeById.get(selectedId));
      if (index != null && options.scroll) {
        const symbol = currentSymbols()[index];
        if (symbol) scrollToLine(symbol.line_start);
      }
    }

    function renderReader(node) {
      readerTitle.textContent = node.label;
      readerPath.textContent = node.id;
      renderSymbolSelect(node);
      renderAnnotationSelect(node);
      renderCode(node);
    }

    function renderSymbolSelect(node) {
      const symbols = currentSymbols(node.id);
      symbolSelect.innerHTML = '<option value="">File overview</option>';
      symbols.forEach((symbol, index) => {
        const option = document.createElement('option');
        option.value = String(index);
        option.textContent = `${symbol.qualifiedName} (${symbol.line_start}-${symbol.line_end})`;
        symbolSelect.append(option);
      });
      symbolSelect.value = selectedSymbolIndex == null ? '' : String(selectedSymbolIndex);
    }

    function renderAnnotationSelect(node) {
      const annotations = graph.annotations[node.id] || [];
      annotationSelect.innerHTML = '<option value="">Jump to annotation</option>';
      annotations.forEach((item) => {
        const line = Number(item.line_start || 0);
        if (!line) return;
        const option = document.createElement('option');
        option.value = String(line);
        option.textContent = `line ${line}: ${String(item.question || item.id || 'annotation').slice(0, 72)}`;
        annotationSelect.append(option);
      });
    }

    function renderCode(node) {
      const source = sourceById[node.id];
      codeBody.innerHTML = '';
      if (!source || typeof source.text !== 'string') {
        codeBody.innerHTML = '<div class="empty-code">No embedded source for this file.</div>';
        return;
      }
      const lines = source.text.split('\\n');
      const symbols = currentSymbols(node.id);
      const symbolByLine = symbolIndexesByLine(symbols, lines.length);
      const noteStarts = new Set((graph.annotations[node.id] || []).map((item) => Number(item.line_start || 0)).filter(Boolean));
      const fragment = document.createDocumentFragment();
      lines.forEach((line, index) => {
        const lineNumber = index + 1;
        const symbolIndex = symbolByLine.get(lineNumber);
        const row = document.createElement('div');
        const classes = ['code-line'];
        if (symbolIndex != null) classes.push('in-symbol');
        if (symbolIndex != null && symbolIndex === selectedSymbolIndex) classes.push('selected-symbol');
        if (noteStarts.has(lineNumber)) classes.push('has-note');
        row.className = classes.join(' ');
        row.dataset.line = String(lineNumber);
        if (symbolIndex != null) {
          row.dataset.symbolIndex = String(symbolIndex);
          row.addEventListener('click', () => selectSymbol(symbolIndex));
        }
        const gutter = document.createElement('div');
        gutter.className = 'line-no';
        gutter.textContent = String(lineNumber);
        const text = document.createElement('div');
        text.className = 'code-text';
        text.textContent = line || ' ';
        row.append(gutter, text);
        fragment.append(row);
      });
      codeBody.append(fragment);
    }

    function symbolIndexesByLine(symbols, lineCount) {
      const byLine = new Map();
      symbols.forEach((symbol, index) => {
        const start = Math.max(1, Number(symbol.line_start || 1));
        const end = Math.min(lineCount, Number(symbol.line_end || start));
        const span = end - start;
        for (let line = start; line <= end; line += 1) {
          const currentIndex = byLine.get(line);
          if (currentIndex == null) {
            byLine.set(line, index);
            continue;
          }
          const current = symbols[currentIndex];
          const currentSpan = Number(current.line_end || current.line_start) - Number(current.line_start || 0);
          if (span <= currentSpan) byLine.set(line, index);
        }
      });
      return byLine;
    }

    function renderDetails(node) {
      if (!node) return;
      const symbols = currentSymbols(node.id);
      const symbol = typeof selectedSymbolIndex === 'number' ? symbols[selectedSymbolIndex] : null;
      detailTitle.textContent = node.label;
      detailIntro.textContent = symbol ? `${symbol.qualifiedName} (${symbol.kind})` : node.id;
      const annotations = graph.annotations[node.id] || [];
      const directAnnotations = symbol ? annotationsForSymbol(symbol, annotations) : [];
      const broadAnnotations = symbol
        ? annotations.filter((item) => !directAnnotations.includes(item) && broadAnnotation(item))
        : annotations;
      detailBody.innerHTML = `
        <div class="file-path">${escapeHtml(node.id)}</div>
        <div class="pill-row">
          <span class="pill">${escapeHtml(node.area)}</span>
          <span class="pill">${escapeHtml(node.kind)}</span>
          <span class="pill">${node.lines} lines</span>
          <span class="pill">${node.symbolCount || 0} symbols</span>
          <span class="pill">depth ${node.depth ?? 'unreached'}</span>
          ${node.root ? '<span class="pill">root</span>' : ''}
        </div>
        ${symbol ? symbolSummary(symbol, directAnnotations, broadAnnotations) : fileSummary(node, annotations)}
        <h2>Symbols</h2>
        ${symbolList(symbols)}
        <h2>Deletion Impact Hint</h2>
        <p>${escapeHtml(node.removalImpact)}</p>
        <h2>Imports</h2>
        ${linkList(node.imports)}
        <h2>Imported By</h2>
        ${linkList(node.importedBy)}
      `;
      detailBody.querySelectorAll('[data-node-id]').forEach((button) => {
        button.addEventListener('click', () => selectNode(button.dataset.nodeId));
      });
      detailBody.querySelectorAll('[data-symbol-index]').forEach((button) => {
        button.addEventListener('click', () => selectSymbol(Number(button.dataset.symbolIndex), { scroll: true }));
      });
    }

    function linkList(paths) {
      if (!paths.length) return '<p>No internal file dependency found.</p>';
      return `<ul class="list">${paths.map((path) => `<li><button type="button" data-node-id="${escapeHtml(path)}">${escapeHtml(path)}</button></li>`).join('')}</ul>`;
    }

    function symbolList(symbols) {
      if (!symbols.length) return '<p>No function/class symbols extracted.</p>';
      return `<ul class="list">${symbols.map((symbol, index) => `
        <li><button type="button" class="${index === selectedSymbolIndex ? 'active' : ''}" data-symbol-index="${index}">
          ${escapeHtml(symbol.qualifiedName)}<br>
          <span style="color: var(--muted)">${escapeHtml(symbol.kind)} | lines ${symbol.line_start}-${symbol.line_end}</span>
        </button></li>
      `).join('')}</ul>`;
    }

    function fileSummary(node, annotations) {
      const manualAnnotations = annotations.filter((item) => !isAutoAnnotation(item));
      const autoAnnotations = annotations.filter(isAutoAnnotation);
      return `
        <h2>Saved Manual Notes</h2>
        ${annotationList(manualAnnotations, 'No manual notes saved for this file yet.')}
        <h2>Auto Function Ledger</h2>
        <p>${autoAnnotations.length} auto-generated function/class notes. Select a symbol or click its code line to read the exact note.</p>
        ${annotationList(autoAnnotations.slice(0, 8), 'No auto-generated symbol notes for this file.')}
      `;
    }

    function symbolSummary(symbol, directAnnotations, broadAnnotations) {
      return `
        <h2>Selected Symbol</h2>
        <div class="annotation hint">
          <strong>${escapeHtml(symbol.qualifiedName)}</strong>
          <div>${escapeHtml(symbol.kind)} | lines ${symbol.line_start}-${symbol.line_end}</div>
        </div>
        <h2>Function Notes</h2>
        ${annotationList(directAnnotations, 'No saved function-specific note yet. Ask about this function, then say "lưu vào ledger".')}
        <h2>Broader File Notes</h2>
        ${annotationList(broadAnnotations.slice(0, 4), 'No broader saved note for this file.')}
      `;
    }

    function annotationList(items, emptyMessage = 'No saved annotations yet.') {
      if (!items.length) return `<p>${escapeHtml(emptyMessage)}</p>`;
      return items.map((item) => `
        <div class="annotation ${isAutoAnnotation(item) ? 'hint' : ''}">
          <strong>${escapeHtml(annotationTitle(item))}</strong>
          <div>${escapeHtml(item.answer || '')}</div>
          ${item.short_explanation ? `<p>${escapeHtml(item.short_explanation)}</p>` : ''}
          ${item.removal_impact ? `<p>${escapeHtml(item.removal_impact)}</p>` : ''}
        </div>
      `).join('');
    }

    function annotationTitle(item) {
      const start = Number(item.line_start || 0);
      const end = Number(item.line_end || start);
      const lineLabel = start ? `lines ${start}${end && end !== start ? `-${end}` : ''}: ` : '';
      return `${lineLabel}${item.question || item.id || 'Question'}`;
    }

    function isAutoAnnotation(item) {
      return item.auto_generated === true || item.type === 'auto_symbol_summary';
    }

    function annotationsForSymbol(symbol, annotations) {
      return annotations.filter((item) => {
        if (item.type === 'file_overview') return false;
        const start = Number(item.line_start || 0);
        const end = Number(item.line_end || start);
        if (!start) return false;
        const span = Math.max(1, end - start + 1);
        const overlaps = start <= symbol.line_end && end >= symbol.line_start;
        const startsInside = start >= symbol.line_start && start <= symbol.line_end;
        return overlaps && (startsInside || span <= 220);
      });
    }

    function broadAnnotation(item) {
      const start = Number(item.line_start || 0);
      const end = Number(item.line_end || start);
      return item.type === 'file_overview' || !start || end - start > 220;
    }

    function currentSymbols(path = selectedId) {
      const source = sourceById[path] || {};
      return Array.isArray(source.symbols) ? source.symbols : [];
    }

    function scrollToLine(line) {
      const row = codeBody.querySelector(`[data-line="${line}"]`);
      if (!row) return;
      row.scrollIntoView({ block: 'center' });
    }

    function escapeHtml(value) {
      return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    }
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
