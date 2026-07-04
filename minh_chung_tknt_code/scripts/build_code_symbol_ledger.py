from __future__ import annotations

import ast
import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from build_code_graph import ANNOTATIONS_PATH, REPO_ROOT, _iter_source_paths, _read_text


AUTO_TYPE = "auto_symbol_summary"
AUTO_ID_PREFIX = "auto-symbol-summary:"
GENERATED_AT = "2026-06-28"
MAX_CALLS = 8
IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


@dataclass(frozen=True)
class SymbolFacts:
    path: str
    name: str
    qualified_name: str
    kind: str
    line_start: int
    line_end: int
    signature: str
    docstring: str | None
    decorators: tuple[str, ...]
    calls: tuple[str, ...]
    raises: tuple[str, ...]
    returns_value: bool
    mutates_state: bool
    has_io: bool


class _PythonFactsVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.raises: list[str] = []
        self.returns_value = False
        self.mutates_state = False
        self.has_io = False

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)
        if name:
            self.calls[name] += 1
            if name in {
                "open",
                "read_text",
                "write_text",
                "json.loads",
                "json.dumps",
                "requests.get",
                "requests.post",
                "fetch",
                "execute",
                "commit",
                "rollback",
            }:
                self.has_io = True
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        if node.exc is not None:
            self.raises.append(_expr_name(node.exc))
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None and not isinstance(node.value, ast.Constant):
            self.returns_value = True
        if isinstance(node.value, ast.Constant) and node.value.value is not None:
            self.returns_value = True
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if any(_target_mutates_state(target) for target in node.targets):
            self.mutates_state = True
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if _target_mutates_state(node.target):
            self.mutates_state = True
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if _target_mutates_state(node.target):
            self.mutates_state = True
        self.generic_visit(node)


def main() -> int:
    source_paths = sorted(
        path for path in _iter_source_paths() if path.endswith((".py", ".ts", ".tsx"))
    )
    token_counts = _identifier_counts(source_paths)
    current_payload = _load_ledger_payload()
    files = current_payload.setdefault("files", {})
    if not isinstance(files, dict):
        files = {}
        current_payload["files"] = files

    generated_count = 0
    for path in source_paths:
        records = files.get(path)
        preserved = _preserved_records(records if isinstance(records, list) else [])
        auto_records = [
            _record_for_symbol(facts, reference_count=token_counts.get(facts.name, 0))
            for facts in _extract_symbol_facts(path)
        ]
        if preserved or auto_records:
            files[path] = [*preserved, *auto_records]
            generated_count += len(auto_records)
        elif path in files:
            del files[path]

    current_payload["version"] = 1
    current_payload["auto_symbol_ledger"] = {
        "type": AUTO_TYPE,
        "generated_at": GENERATED_AT,
        "symbol_count": generated_count,
        "source": "scripts/build_code_symbol_ledger.py",
        "note": "Auto-generated first-pass notes. Manual Q&A entries are preserved above generated notes per file.",
    }
    ANNOTATIONS_PATH.write_text(
        json.dumps(current_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {ANNOTATIONS_PATH.relative_to(REPO_ROOT)} with {generated_count} auto notes"
    )
    return 0


def _load_ledger_payload() -> dict[str, object]:
    if not ANNOTATIONS_PATH.exists():
        return {"version": 1, "files": {}}
    try:
        payload = json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"version": 1, "files": {}}
    if not isinstance(payload, dict):
        return {"version": 1, "files": {}}
    return payload


def _preserved_records(records: Sequence[object]) -> list[dict[str, object]]:
    preserved: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        record_id = str(record.get("id", ""))
        if record.get("auto_generated") is True:
            continue
        if record.get("type") == AUTO_TYPE:
            continue
        if record_id.startswith(AUTO_ID_PREFIX):
            continue
        preserved.append(record)
    return preserved


def _identifier_counts(paths: Iterable[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in paths:
        counts.update(IDENTIFIER_RE.findall(_read_text(REPO_ROOT / path)))
    return counts


def _extract_symbol_facts(path: str) -> list[SymbolFacts]:
    text = _read_text(REPO_ROOT / path)
    if path.endswith(".py"):
        return _extract_python_facts(path, text)
    return _extract_typescript_facts(path, text)


def _extract_python_facts(path: str, text: str) -> list[SymbolFacts]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    facts: list[SymbolFacts] = []

    def walk(nodes: Sequence[ast.stmt], parents: tuple[str, ...] = ()) -> None:
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                qualified = (*parents, node.name)
                facts.append(_python_class_facts(path, node, qualified))
                walk(node.body, qualified)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = (*parents, node.name)
                facts.append(_python_function_facts(path, node, qualified))
                walk(node.body, qualified)

    walk(tree.body)
    return sorted(
        facts,
        key=lambda item: (item.line_start, item.line_end, item.qualified_name),
    )


def _python_class_facts(
    path: str,
    node: ast.ClassDef,
    qualified: tuple[str, ...],
) -> SymbolFacts:
    method_names = tuple(
        child.name
        for child in node.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    bases = tuple(_expr_name(base) for base in node.bases)
    decorators = tuple(_expr_name(decorator) for decorator in node.decorator_list)
    signature = f"class {node.name}"
    if bases:
        signature = f"{signature}({', '.join(bases)})"
    return SymbolFacts(
        path=path,
        name=node.name,
        qualified_name=".".join(qualified),
        kind="class",
        line_start=node.lineno,
        line_end=int(getattr(node, "end_lineno", node.lineno)),
        signature=signature,
        docstring=_first_sentence(ast.get_docstring(node)),
        decorators=decorators,
        calls=method_names[:MAX_CALLS],
        raises=(),
        returns_value=False,
        mutates_state=False,
        has_io=False,
    )


def _python_function_facts(
    path: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    qualified: tuple[str, ...],
) -> SymbolFacts:
    visitor = _PythonFactsVisitor()
    visitor.visit(node)
    decorators = tuple(_expr_name(decorator) for decorator in node.decorator_list)
    calls = tuple(name for name, _count in visitor.calls.most_common(MAX_CALLS))
    return SymbolFacts(
        path=path,
        name=node.name,
        qualified_name=".".join(qualified),
        kind="async function" if isinstance(node, ast.AsyncFunctionDef) else "function",
        line_start=node.lineno,
        line_end=int(getattr(node, "end_lineno", node.lineno)),
        signature=_python_signature(node),
        docstring=_first_sentence(ast.get_docstring(node)),
        decorators=decorators,
        calls=calls,
        raises=tuple(visitor.raises[:MAX_CALLS]),
        returns_value=visitor.returns_value,
        mutates_state=visitor.mutates_state,
        has_io=visitor.has_io,
    )


def _python_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    try:
        args = ast.unparse(node.args)
    except Exception:
        args = "..."
    returns = ""
    if node.returns is not None:
        try:
            returns = f" -> {ast.unparse(node.returns)}"
        except Exception:
            returns = " -> ..."
    return f"{prefix} {node.name}({args}){returns}"


def _extract_typescript_facts(path: str, text: str) -> list[SymbolFacts]:
    lines = text.splitlines()
    facts: list[SymbolFacts] = []
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
    for index, line in enumerate(lines, start=1):
        for kind, pattern in patterns:
            match = pattern.search(line)
            if match is None:
                continue
            if pattern is const_pattern and not _typescript_const_is_arrow(
                lines, index
            ):
                continue
            name = match.group(1)
            line_end = _typescript_block_end(lines, index)
            body = "\n".join(lines[index - 1 : line_end])
            calls = tuple(_typescript_calls(body, name)[:MAX_CALLS])
            facts.append(
                SymbolFacts(
                    path=path,
                    name=name,
                    qualified_name=name,
                    kind=_typescript_kind(kind, path, name, body),
                    line_start=index,
                    line_end=line_end,
                    signature=line.strip()[:220],
                    docstring=None,
                    decorators=(),
                    calls=calls,
                    raises=(),
                    returns_value="return " in body or "=>" in body,
                    mutates_state=any(
                        token in body
                        for token in ("setState", "useState", ".current", "dispatch(")
                    ),
                    has_io=any(
                        token in body
                        for token in (
                            "fetch(",
                            "axios.",
                            "localStorage",
                            "sessionStorage",
                        )
                    ),
                )
            )
            break
    return facts


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


def _typescript_calls(body: str, own_name: str) -> list[str]:
    skip = {
        "if",
        "for",
        "while",
        "switch",
        "catch",
        "return",
        "typeof",
        "Math",
        own_name,
    }
    ordered: list[str] = []
    for name in re.findall(r"\b([A-Za-z_$][A-Za-z0-9_$]*)\s*\(", body):
        if name in skip or name in ordered:
            continue
        ordered.append(name)
    return ordered


def _typescript_kind(kind: str, path: str, name: str, body: str) -> str:
    if kind == "class":
        return "class"
    if (
        path.endswith(".tsx")
        and name[:1].isupper()
        and ("<" in body or "jsx" in path.lower())
    ):
        return "React component"
    return "function"


def _record_for_symbol(facts: SymbolFacts, reference_count: int) -> dict[str, object]:
    role = _role_sentence(facts)
    data_notes = _data_notes(facts)
    calls = f" Các lời gọi đáng chú ý: {', '.join(facts.calls)}." if facts.calls else ""
    doc = f" Docstring nói: {facts.docstring}" if facts.docstring else ""
    decorators = (
        f" Decorator: {', '.join(facts.decorators)}." if facts.decorators else ""
    )
    answer = (
        f"[Auto draft] {facts.kind} `{facts.qualified_name}` nằm trong `{facts.path}`. "
        f"{role}{data_notes}{calls}{decorators}{doc}"
    )
    return {
        "id": _auto_id(facts),
        "type": AUTO_TYPE,
        "auto_generated": True,
        "line_start": facts.line_start,
        "line_end": facts.line_end,
        "symbol": facts.qualified_name,
        "kind": facts.kind,
        "question": f"{facts.qualified_name} để làm gì?",
        "answer": answer,
        "short_explanation": _short_explanation(facts),
        "removal_impact": _removal_impact(facts, reference_count),
        "created_at": GENERATED_AT,
        "source": "scripts/build_code_symbol_ledger.py",
    }


def _role_sentence(facts: SymbolFacts) -> str:
    name = facts.name.strip("_")
    tokens = _name_tokens(name)
    if facts.kind == "class":
        return _class_role_sentence(facts, tokens)
    if facts.kind == "React component":
        return "Nó render một phần UI và nối state/props với event handler hoặc API liên quan."
    if not tokens:
        return "Nó đóng vai trò helper nội bộ trong luồng của file."

    first = tokens[0]
    if first in {"build", "make", "create", "construct", "assemble", "compose"}:
        return "Nó dựng hoặc đóng gói dữ liệu trung gian từ input đã có."
    if first in {"load", "read", "fetch", "get", "extract", "collect", "parse"}:
        return "Nó đọc, lấy hoặc trích xuất dữ liệu để bước sau dùng tiếp."
    if first in {"validate", "check", "ensure", "assert", "is", "has", "can", "should"}:
        return "Nó kiểm tra điều kiện và trả kết quả/raise lỗi để chặn dữ liệu không hợp lệ."
    if first in {"normalize", "clean", "sanitize", "canonicalize", "resolve", "coerce"}:
        return "Nó chuẩn hoá dữ liệu về dạng ổn định để các bước sau không phải xử lý nhiều biến thể."
    if first in {"filter", "select", "sort", "group", "rank", "score"}:
        return "Nó chọn, lọc, xếp hạng hoặc chấm điểm dữ liệu theo rule của module."
    if first in {"solve", "search", "plan", "place", "layout"}:
        return "Nó tham gia tìm phương án/plan/layout thoả các ràng buộc."
    if first in {"render", "format", "serialize", "to", "from", "export"}:
        return "Nó chuyển dữ liệu sang dạng hiển thị, payload hoặc format khác."
    if first in {"apply", "update", "sync", "merge", "patch", "set"}:
        return "Nó áp dụng thay đổi vào payload/state hiện có và giữ dữ liệu đồng bộ."
    if first in {"repair", "fallback", "retarget", "relax"}:
        return (
            "Nó xử lý nhánh cứu lỗi hoặc biến thể khi phương án ban đầu chưa phù hợp."
        )
    if first in {"run", "main", "generate", "handle"}:
        return "Nó là điểm điều phối một luồng xử lý lớn hơn trong file."
    if first in {"on"}:
        return "Nó là event handler, chạy khi UI hoặc callback phát sinh sự kiện."
    return "Nó là helper chuyên biệt; tên hàm cho thấy nó xử lý phần dữ liệu/ngữ cảnh tương ứng."


def _class_role_sentence(facts: SymbolFacts, tokens: Sequence[str]) -> str:
    decorators = " ".join(facts.decorators)
    if "dataclass" in decorators:
        return "Class này là data container: gom các field liên quan để truyền trạng thái/kết quả giữa các helper."
    if "config" in tokens or "settings" in tokens:
        return "Class này mô hình hoá cấu hình/settings để các module đọc cùng một contract."
    if "schema" in tokens or "model" in tokens or "payload" in tokens:
        return "Class này mô hình hoá shape dữ liệu/payload để validate và truyền giữa các bước."
    if "service" in tokens:
        return "Class này gom logic nghiệp vụ và thường đứng giữa API layer với adapter/database."
    if "client" in tokens or "adapter" in tokens:
        return "Class này bọc một dependency/IO bên ngoài để phần còn lại gọi qua interface ổn định."
    if facts.calls:
        return f"Class này gom state/logic liên quan; các method chính gồm {', '.join(facts.calls[:5])}."
    return "Class này gom dữ liệu hoặc hành vi có cùng trách nhiệm trong module."


def _data_notes(facts: SymbolFacts) -> str:
    notes: list[str] = []
    if facts.signature:
        notes.append(f"Chữ ký: `{facts.signature}`.")
    if facts.returns_value:
        notes.append("Có trả dữ liệu cho caller.")
    if facts.mutates_state:
        notes.append("Có cập nhật state/object hiện có.")
    if facts.has_io:
        notes.append("Có dấu hiệu IO như đọc/ghi file, network hoặc database.")
    if facts.raises:
        notes.append(f"Có thể raise: {', '.join(facts.raises[:3])}.")
    return " " + " ".join(notes) if notes else ""


def _short_explanation(facts: SymbolFacts) -> str:
    target = "class" if facts.kind == "class" else "hàm"
    return (
        f"Bản tự động: {target} này nằm ở dòng {facts.line_start}-{facts.line_end}; "
        f"hãy xem cùng source để xác nhận chi tiết nghiệp vụ trước khi refactor/xoá."
    )


def _removal_impact(facts: SymbolFacts, reference_count: int) -> str:
    name = facts.name
    if name in {
        "main",
        "run_case",
        "generate_object_layout",
        "solve_object_level_layout",
    }:
        return "Đây có vẻ là entry point quan trọng; xoá sẽ dễ làm CLI/API/pipeline hoặc caller chính lỗi ngay."
    decorator_text = " ".join(facts.decorators)
    if facts.kind == "class" and "dataclass" in decorator_text:
        return "Xoá data container này làm các helper/caller đang tạo hoặc đọc field của nó lỗi; cần thay contract dữ liệu tương ứng."
    if facts.decorators:
        return "Hàm/class có decorator nên có thể được framework đăng ký tự động; xoá cần kiểm tra router, DI, Pydantic/dataclass hoặc callback liên quan."
    if facts.kind == "React component":
        return "Xoá component này làm nơi import/render component lỗi hoặc làm mất một phần UI."
    if facts.kind == "class":
        return "Xoá class này làm mọi nơi instantiate, import, type-check hoặc validate theo class này lỗi; cần tìm caller và schema liên quan trước."
    if name.startswith("_"):
        return "Đây là helper private theo convention; xoá trực tiếp sẽ làm caller nội bộ bị NameError nếu chưa xoá/sửa các lời gọi tương ứng."
    if reference_count <= 2:
        return "Tên này ít xuất hiện trong mã nguồn tĩnh, nhưng vẫn cần kiểm tra dynamic call, test và framework registration trước khi xoá."
    return f"Tên `{name}` xuất hiện khoảng {reference_count} lần trong source; xoá cần sửa các caller/import/test liên quan, không chỉ xoá định nghĩa."


def _auto_id(facts: SymbolFacts) -> str:
    raw = f"{facts.path}:{facts.qualified_name}:{facts.line_start}"
    slug = re.sub(r"[^A-Za-z0-9_.:-]+", "-", raw)
    return f"{AUTO_ID_PREFIX}{slug}"


def _name_tokens(name: str) -> list[str]:
    snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return [token.lower() for token in re.split(r"[_\W]+", snake) if token]


def _first_sentence(value: str | None) -> str | None:
    if not value:
        return None
    normalized = " ".join(value.strip().split())
    if not normalized:
        return None
    match = re.search(r"([.!?])\s+", normalized)
    if match is None:
        return normalized[:260]
    return normalized[: match.end()].strip()


def _call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parent = _call_name(func.value)
        return f"{parent}.{func.attr}" if parent else func.attr
    return None


def _expr_name(expr: ast.AST) -> str:
    try:
        return ast.unparse(expr)
    except Exception:
        return expr.__class__.__name__


def _target_mutates_state(target: ast.expr) -> bool:
    if isinstance(target, ast.Attribute):
        return True
    if isinstance(target, ast.Subscript):
        return True
    if isinstance(target, (ast.Tuple, ast.List)):
        return any(_target_mutates_state(item) for item in target.elts)
    return False


if __name__ == "__main__":
    raise SystemExit(main())
