from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from glob import glob as glob_paths
from pathlib import Path
from types import TracebackType
from typing import TypeAlias, cast
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from pipeline.account_layout_autosave import build_saved_layout_option_drafts

JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
JSONObject: TypeAlias = dict[str, JSONValue]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE_IDS: tuple[str, ...] = (
    "living_01_rect_sofa_tv",
    "living_02_solid_wall_tv_window_clear",
    "living_03_l_shape_circulation",
    "living_04_large_sectional",
    "bedroom_01_rect_door_right_window_left",
    "bedroom_02_square_two_nightstands",
    "bedroom_03_compact_study_window",
    "bedroom_05_l_shape_storage",
)


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    room_type: str
    input_payload: Mapping[str, object]
    raw: Mapping[str, object]


@dataclass(frozen=True)
class LayoutArtifact:
    case: BenchmarkCase
    case_output_dir: Path
    layout_path: Path
    pipeline_output_path: Path


@dataclass(frozen=True)
class CameraCandidate:
    preset: Mapping[str, object]
    score: float
    details: JSONObject
    snapshot_artifact: Mapping[str, object]
    clean_snapshot_image_data_url: str
    depth_map_image_data_url: str | None = None
    edge_map_image_data_url: str | None = None
    segmentation_map_image_data_url: str | None = None
    instance_outline_image_data_url: str | None = None
    instance_outline_overlay_image_data_url: str | None = None


class CdpError(RuntimeError):
    pass


class SkippedLayoutCase(RuntimeError):
    pass


class CdpWebSocket:
    def __init__(self, websocket_url: str, *, timeout_s: float = 30.0) -> None:
        self._url = websocket_url
        self._timeout_s = timeout_s
        self._socket: socket.socket | None = None
        self._next_id = 1

    def __enter__(self) -> CdpWebSocket:
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def connect(self) -> None:
        parsed = urlparse(self._url)
        if parsed.scheme != "ws" or not parsed.hostname or not parsed.port:
            raise CdpError(f"Unsupported DevTools websocket URL: {self._url}")

        sock = socket.create_connection(
            (parsed.hostname, parsed.port),
            timeout=self._timeout_s,
        )
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {parsed.hostname}:{parsed.port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        )
        sock.sendall(request.encode("ascii"))
        response = _read_until(sock, b"\r\n\r\n")
        status_line = response.split(b"\r\n", maxsplit=1)[0].decode(
            "latin-1",
            errors="replace",
        )
        if " 101 " not in status_line:
            sock.close()
            raise CdpError(f"DevTools websocket handshake failed: {status_line}")
        self._socket = sock

    def close(self) -> None:
        if self._socket is None:
            return
        try:
            self._socket.close()
        finally:
            self._socket = None

    def call(
        self,
        method: str,
        params: Mapping[str, object] | None = None,
        *,
        timeout_s: float = 30.0,
    ) -> Mapping[str, object]:
        message_id = self._next_id
        self._next_id += 1
        payload: dict[str, object] = {"id": message_id, "method": method}
        if params is not None:
            payload["params"] = dict(params)
        self._send_json(payload)

        deadline = time.monotonic() + timeout_s
        while True:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                raise TimeoutError(f"Timed out waiting for CDP method {method}.")
            message = self._recv_json(timeout_s=remaining_s)
            if int(message.get("id") or 0) != message_id:
                continue
            error = _mapping(message.get("error"))
            if error:
                text = str(error.get("message") or error)
                raise CdpError(f"CDP method {method} failed: {text}")
            return _mapping(message.get("result"))

    def _send_json(self, payload: Mapping[str, object]) -> None:
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self._send_frame(opcode=0x1, payload=text.encode("utf-8"))

    def _recv_json(self, *, timeout_s: float) -> Mapping[str, object]:
        sock = self._require_socket()
        sock.settimeout(timeout_s)
        chunks: list[bytes] = []
        while True:
            first = _read_exact(sock, 1)[0]
            second = _read_exact(sock, 1)[0]
            fin = bool(first & 0x80)
            opcode = first & 0x0F
            masked = bool(second & 0x80)
            length = second & 0x7F
            if length == 126:
                length = struct.unpack("!H", _read_exact(sock, 2))[0]
            elif length == 127:
                length = struct.unpack("!Q", _read_exact(sock, 8))[0]
            mask = _read_exact(sock, 4) if masked else b""
            payload = _read_exact(sock, length) if length else b""
            if mask:
                payload = bytes(
                    value ^ mask[index % 4] for index, value in enumerate(payload)
                )

            if opcode == 0x8:
                raise CdpError("DevTools websocket closed.")
            if opcode == 0x9:
                self._send_frame(opcode=0xA, payload=payload)
                continue
            if opcode in {0x1, 0x0}:
                chunks.append(payload)
                if fin:
                    message = json.loads(b"".join(chunks).decode("utf-8"))
                    return _mapping(message)

    def _send_frame(self, *, opcode: int, payload: bytes) -> None:
        sock = self._require_socket()
        header = bytearray([0x80 | opcode])
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        elif length <= 0xFFFF:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", length))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", length))
        mask = os.urandom(4)
        masked_payload = bytes(
            value ^ mask[index % 4] for index, value in enumerate(payload)
        )
        sock.sendall(bytes(header) + mask + masked_payload)

    def _require_socket(self) -> socket.socket:
        if self._socket is None:
            raise CdpError("DevTools websocket is not connected.")
        return self._socket


class ManagedProcess(AbstractContextManager["ManagedProcess"]):
    def __init__(self, process: subprocess.Popen[object] | None) -> None:
        self._process = process

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._process is None:
            return
        if self._process.poll() is not None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=8)


class ChromeSession(AbstractContextManager["ChromeSession"]):
    def __init__(
        self,
        chrome_path: Path,
        *,
        device_scale_factor: float,
        headless: bool,
    ) -> None:
        self._chrome_path = chrome_path
        self._device_scale_factor = device_scale_factor
        self._headless = headless
        self._port = _find_free_port()
        self._profile_dir = tempfile.TemporaryDirectory(prefix="tknt_eval_chrome_")
        self._process: subprocess.Popen[object] | None = None
        self._client: CdpWebSocket | None = None

    def __enter__(self) -> ChromeSession:
        command = [
            str(self._chrome_path),
            "--disable-background-networking",
            "--disable-default-apps",
            "--disable-dev-shm-usage",
            "--disable-extensions",
            "--disable-popup-blocking",
            "--enable-webgl",
            "--ignore-gpu-blocklist",
            "--no-default-browser-check",
            "--no-first-run",
            "--remote-allow-origins=*",
            f"--remote-debugging-port={self._port}",
            f"--user-data-dir={self._profile_dir.name}",
            f"--force-device-scale-factor={self._device_scale_factor:g}",
            "--window-size=1440,1000",
            "about:blank",
        ]
        if self._headless:
            command.insert(1, "--headless=new")
            command.insert(2, "--use-gl=angle")
            command.insert(3, "--use-angle=swiftshader")
        self._process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            _wait_for_url(
                f"http://127.0.0.1:{self._port}/json/version",
                timeout_s=20,
            )
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=8)
        self._profile_dir.cleanup()

    def open(self, url: str) -> None:
        target = _create_chrome_target(self._port, url)
        websocket_url = _string(target.get("webSocketDebuggerUrl"))
        if not websocket_url:
            raise CdpError("Chrome did not return a page websocket URL.")
        self._client = CdpWebSocket(websocket_url)
        self._client.connect()
        self._client.call("Page.enable")
        self._client.call("Runtime.enable")
        self._client.call("Page.navigate", {"url": url})
        _wait_for_document_ready(self._require_client(), timeout_s=30)

    def wait_for_eval_bridge(self, *, timeout_s: float) -> None:
        started_at = time.monotonic()
        while time.monotonic() - started_at <= timeout_s:
            value = self.evaluate(
                "Boolean(window.__TKNT_EVAL__)",
                timeout_s=5,
            )
            if value is True:
                return
            time.sleep(0.2)
        raise TimeoutError(
            "Frontend is reachable but window.__TKNT_EVAL__ is missing. "
            "Start the frontend with VITE_ENABLE_EVAL_HOOKS=true, or let this "
            "command auto-start the frontend on a free port."
        )

    def evaluate(self, expression: str, *, timeout_s: float = 30.0) -> object:
        result = self._require_client().call(
            "Runtime.evaluate",
            {
                "expression": expression,
                "awaitPromise": True,
                "returnByValue": True,
            },
            timeout_s=timeout_s,
        )
        if result.get("exceptionDetails"):
            details = _mapping(result.get("exceptionDetails"))
            exception = _mapping(details.get("exception"))
            message = _string(exception.get("description")) or _string(
                details.get("text")
            )
            raise CdpError(message or "Runtime.evaluate raised an exception.")
        remote = _mapping(result.get("result"))
        return remote.get("value")

    def _require_client(self) -> CdpWebSocket:
        if self._client is None:
            raise CdpError("Chrome page is not open.")
        return self._client


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    cases_path = _resolve_repo_path(args.cases)
    output_root = _resolve_repo_path(args.output_root)
    layout_cases_root = _resolve_repo_path(args.layout_cases_root)
    output_root.mkdir(parents=True, exist_ok=True)
    layout_cases_root.mkdir(parents=True, exist_ok=True)

    if args.api_url and not _url_reachable(args.api_url):
        print(
            f"Warning: API URL is not reachable: {args.api_url}. "
            "Snapshot capture can continue, but image-render API calls may fail later."
        )

    cases: list[BenchmarkCase] = []
    if not args.only_benchmark_runs:
        cases.extend(
            _select_cases(
                _load_cases(cases_path),
                include_case_ids=_parse_csv(args.include) or list(DEFAULT_CASE_IDS),
            )
        )
    cases.extend(_load_benchmark_run_cases(args.benchmark_run_glob))
    if not cases:
        raise SystemExit("No layout snapshot cases were selected.")
    print(f"Preparing {len(cases)} layout snapshot case(s).")

    frontend_url = args.frontend_url.rstrip("/")
    frontend_process = _ensure_frontend(
        frontend_url=frontend_url,
        api_url=args.api_url,
        auto_start=not args.no_auto_start_frontend,
    )
    chrome_path = _resolve_chrome_path(args.chrome_path)
    layout_artifacts: list[LayoutArtifact] = []
    failed_cases: list[str] = []
    with (
        frontend_process,
        ChromeSession(
            chrome_path,
            device_scale_factor=max(1.0, args.device_scale_factor),
            headless=not args.show_browser,
        ) as chrome,
    ):
        chrome.open(frontend_url)
        chrome.wait_for_eval_bridge(timeout_s=args.frontend_timeout_s)
        for benchmark_case in cases:
            try:
                artifact = _prepare_layout_artifact(
                    benchmark_case=benchmark_case,
                    output_root=output_root,
                    layout_cases_root=layout_cases_root,
                    user=args.user,
                    timeout_s=args.layout_timeout_s,
                    reuse_existing=args.reuse_existing_layout,
                    skip_missing_layout=args.skip_missing_layout,
                    option_index=max(1, args.option_index),
                )
                layout_artifacts.append(artifact)
                print(f"Layout ready: {benchmark_case.case_id}")
            except SkippedLayoutCase as exc:
                print(f"Layout skipped: {benchmark_case.case_id}: {exc}")
                continue
            except Exception as exc:
                failed_cases.append(benchmark_case.case_id)
                print(f"Layout failed: {benchmark_case.case_id}: {exc}")
                if not args.continue_on_error:
                    raise
                continue

            try:
                _capture_layout_snapshot(
                    chrome=chrome,
                    artifact=artifact,
                    max_camera_presets=max(0, args.max_camera_presets),
                    capture_timeout_s=args.capture_timeout_s,
                    reuse_selected_camera_preset=args.reuse_selected_camera_preset,
                )
                print(f"Snapshot ready: {artifact.case.case_id}")
            except Exception as exc:
                failed_cases.append(artifact.case.case_id)
                print(f"Snapshot failed: {artifact.case.case_id}: {exc}")
                if not args.continue_on_error:
                    raise

    if not layout_artifacts:
        raise SystemExit("No layout artifacts were prepared; nothing to summarize.")

    manifest_path = _write_generated_manifest(
        output_root=output_root,
        artifacts=[
            artifact
            for artifact in layout_artifacts
            if (artifact.case_output_dir / "snapshot_payload.json").exists()
        ],
    )
    summary_path = _write_run_summary(
        output_root=output_root,
        artifacts=layout_artifacts,
        failed_cases=failed_cases,
        manifest_path=manifest_path,
    )
    print(f"Wrote image eval manifest: {manifest_path}")
    print(f"Wrote snapshot prep summary: {summary_path}")
    return 0 if not failed_cases else 2


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate TKNT layout cases, load each layout in the real 3D UI, "
            "capture camera preset snapshots, and save snapshot artifacts."
        )
    )
    parser.add_argument(
        "--cases",
        default="evaluation/layout_benchmark/cases.json",
        help="Benchmark cases JSON.",
    )
    parser.add_argument(
        "--include",
        "--include-cases",
        default=",".join(DEFAULT_CASE_IDS),
        help="Comma-separated case IDs. Defaults to the 8 living/bedroom eval cases.",
    )
    parser.add_argument(
        "--benchmark-run-glob",
        action="append",
        default=[],
        help=(
            "Glob for existing layout benchmark run folders to snapshot, e.g. "
            "'evaluation/layout_benchmark/runs/manual/runs/tknt_new_*'. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--only-benchmark-runs",
        action="store_true",
        help="Skip --cases/--include and snapshot only folders from --benchmark-run-glob.",
    )
    parser.add_argument(
        "--output-root",
        default="experiments/image_flow_eval/cases",
        help="Folder where per-case snapshot artifacts are written.",
    )
    parser.add_argument(
        "--layout-cases-root",
        default="experiments/image_flow_eval/layout_artifacts",
        help="Folder where backend pipeline case artifacts are written.",
    )
    parser.add_argument("--frontend-url", default="http://127.0.0.1:5173")
    parser.add_argument("--api-url", default="http://127.0.0.1:8001")
    parser.add_argument("--user", default="image_eval_user")
    parser.add_argument("--option-index", type=int, default=1)
    parser.add_argument("--layout-timeout-s", type=int, default=3600)
    parser.add_argument("--capture-timeout-s", type=float, default=45.0)
    parser.add_argument("--frontend-timeout-s", type=float, default=30.0)
    parser.add_argument(
        "--max-camera-presets",
        type=int,
        default=0,
        help=(
            "Optional cap for camera presets captured/scored per layout. "
            "0 means all camera presets."
        ),
    )
    parser.add_argument(
        "--reuse-selected-camera-preset",
        action="store_true",
        help=(
            "Capture only the camera preset saved in each case's existing "
            "snapshot_meta.json selected_preset.id."
        ),
    )
    parser.add_argument(
        "--device-scale-factor",
        type=float,
        default=2.0,
        help=(
            "Chrome device scale factor for snapshot capture. "
            "2.0 matches Retina-like UI snapshots."
        ),
    )
    parser.add_argument("--chrome-path", default="")
    parser.add_argument(
        "--no-auto-start-frontend",
        action="store_true",
        help="Fail if --frontend-url is not already serving the eval-enabled UI.",
    )
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Run Chrome visibly instead of headless for debugging.",
    )
    parser.add_argument(
        "--reuse-existing-layout",
        action="store_true",
        help="Reuse each case's layout.json/pipeline_output.json when present.",
    )
    parser.add_argument(
        "--skip-missing-layout",
        action="store_true",
        help=(
            "Skip regular cases without an existing output-root layout.json "
            "instead of running layout generation. Imported benchmark runs still "
            "use their output.json as the layout source."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining cases after a layout or snapshot failure.",
    )
    return parser.parse_args(argv)


def _prepare_layout_artifact(
    *,
    benchmark_case: BenchmarkCase,
    output_root: Path,
    layout_cases_root: Path,
    user: str,
    timeout_s: int,
    reuse_existing: bool,
    skip_missing_layout: bool,
    option_index: int,
) -> LayoutArtifact:
    case_output_dir = output_root / benchmark_case.case_id
    case_output_dir.mkdir(parents=True, exist_ok=True)
    input_path = case_output_dir / "input_payload.json"
    pipeline_output_path = case_output_dir / "pipeline_output.json"
    layout_path = case_output_dir / "layout.json"
    _write_json(input_path, benchmark_case.input_payload)
    benchmark_run_dir = _benchmark_run_dir_for_case(benchmark_case)
    has_existing_layout = layout_path.exists()
    has_existing_pipeline_output = pipeline_output_path.exists()

    if (
        reuse_existing
        and has_existing_layout
        and (has_existing_pipeline_output or skip_missing_layout)
    ):
        pass
    elif benchmark_run_dir is not None:
        source_output_path = benchmark_run_dir / "output.json"
        if skip_missing_layout and not source_output_path.exists():
            raise SkippedLayoutCase(
                f"missing benchmark layout source: {source_output_path}"
            )
        _prepare_benchmark_run_layout(
            benchmark_case=benchmark_case,
            run_dir=benchmark_run_dir,
            case_output_dir=case_output_dir,
            pipeline_output_path=pipeline_output_path,
            layout_path=layout_path,
            option_index=option_index,
        )
    elif skip_missing_layout:
        raise SkippedLayoutCase(f"missing reusable layout.json: {layout_path}")
    elif not reuse_existing or not (
        has_existing_pipeline_output and has_existing_layout
    ):
        _run_layout_pipeline(
            input_path=input_path,
            output_path=pipeline_output_path,
            case_output_dir=case_output_dir,
            case_id=benchmark_case.case_id,
            cases_root=layout_cases_root,
            user=user,
            timeout_s=timeout_s,
        )
        pipeline_payload = _read_json_mapping(pipeline_output_path)
        layout_payload = _build_layout_payload(
            benchmark_case=benchmark_case,
            pipeline_payload=pipeline_payload,
            option_index=option_index,
        )
        _write_json(layout_path, layout_payload)

    return LayoutArtifact(
        case=benchmark_case,
        case_output_dir=case_output_dir,
        layout_path=layout_path,
        pipeline_output_path=pipeline_output_path,
    )


def _prepare_benchmark_run_layout(
    *,
    benchmark_case: BenchmarkCase,
    run_dir: Path,
    case_output_dir: Path,
    pipeline_output_path: Path,
    layout_path: Path,
    option_index: int,
) -> None:
    source_output_path = run_dir / "output.json"
    if not source_output_path.exists():
        raise FileNotFoundError(f"Benchmark run is missing output.json: {run_dir}")
    pipeline_payload = _read_json_mapping(source_output_path)
    _write_json(pipeline_output_path, pipeline_payload)
    layout_payload = _build_layout_payload(
        benchmark_case=benchmark_case,
        pipeline_payload=pipeline_payload,
        option_index=option_index,
        module_outputs_dir=run_dir / "module_outputs",
    )
    inventory_report = _layout_inventory_integrity_report(
        pipeline_payload=pipeline_payload,
        layout_payload=layout_payload,
    )
    _write_json(case_output_dir / "inventory_integrity_report.json", inventory_report)
    if inventory_report.get("status") == "failed":
        raise RuntimeError(
            "Imported benchmark layout inventory drifted from output.json. "
            f"See {case_output_dir / 'inventory_integrity_report.json'}."
        )
    _write_json(layout_path, layout_payload)

    score = (
        _read_json_mapping(run_dir / "score.json")
        if (run_dir / "score.json").exists()
        else {}
    )
    meta: JSONObject = {
        "case_id": benchmark_case.case_id,
        "source": "layout_benchmark_run",
        "source_run_dir": str(run_dir),
        "source_output_path": str(source_output_path),
        "runtime_s": _json_value(
            pipeline_payload.get("runtime_s") or score.get("runtime_s")
        ),
        "input_tokens": _json_value(
            pipeline_payload.get("input_tokens") or score.get("input_tokens")
        ),
        "output_tokens": _json_value(
            pipeline_payload.get("output_tokens") or score.get("output_tokens")
        ),
    }
    _write_json(case_output_dir / "layout_run_meta.json", meta)


def _run_layout_pipeline(
    *,
    input_path: Path,
    output_path: Path,
    case_output_dir: Path,
    case_id: str,
    cases_root: Path,
    user: str,
    timeout_s: int,
) -> None:
    stdout_path = case_output_dir / "layout_stdout.log"
    stderr_path = case_output_dir / "layout_stderr.log"
    command = [
        sys.executable,
        str(REPO_ROOT / "run_case_backend_cli.py"),
        "--input",
        str(input_path),
        "--user",
        user,
        "--cases-root",
        str(cases_root),
        "--case-id",
        case_id,
        "--stdout",
        "wrapped",
        "--out",
        str(output_path),
        "--ablation",
        "full",
    ]
    env = os.environ.copy()
    env.setdefault("TKNT_DISABLE_MODEL_FALLBACK", "1")
    started_at = time.perf_counter()
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_file,
        stderr_path.open(
            "w",
            encoding="utf-8",
        ) as stderr_file,
    ):
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=stdout_file,
            stderr=stderr_file,
            timeout=timeout_s,
            check=False,
        )
    runtime_s = time.perf_counter() - started_at
    meta = {
        "case_id": case_id,
        "command": command,
        "exit_code": completed.returncode,
        "runtime_s": runtime_s,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "output_path": str(output_path),
    }
    _write_json(case_output_dir / "layout_run_meta.json", meta)
    if completed.returncode != 0:
        stderr_tail = _read_text_tail(stderr_path, max_chars=4000)
        raise RuntimeError(
            f"layout pipeline exited {completed.returncode}. "
            f"See {stderr_path}. Tail: {stderr_tail}"
        )
    if not output_path.exists():
        raise RuntimeError(f"layout pipeline did not write {output_path}.")
    pipeline_payload = _read_json_mapping(output_path)
    if pipeline_payload.get("error"):
        raise RuntimeError(str(pipeline_payload.get("error")))


def _build_layout_payload(
    *,
    benchmark_case: BenchmarkCase,
    pipeline_payload: Mapping[str, object],
    option_index: int,
    module_outputs_dir: Path | None = None,
) -> JSONObject:
    case_dir = _pipeline_case_dir(pipeline_payload, benchmark_case.case_id)
    artifact_dir = module_outputs_dir or case_dir / "module_outputs"
    absolute_layout = _first_json_mapping(
        (
            artifact_dir / "absolute_layout.json",
            case_dir / "07_absolute_layout.json",
        )
    )
    styled_result = _first_json_mapping(
        (
            artifact_dir / "stylist.json",
            case_dir / "08_stylist.json",
        )
    )
    if not styled_result:
        styled_result = _mapping(pipeline_payload.get("final_output"))
    layout_variants = _first_json_mapping((artifact_dir / "layout_variants.json",))
    room_payload = _room_payload_for_case(benchmark_case)
    drafts = build_saved_layout_option_drafts(
        case_id=benchmark_case.case_id,
        case_name=benchmark_case.case_id,
        room_payload=room_payload,
        stylist_payload=styled_result,
        absolute_layout_payload=absolute_layout,
        layout_variants_payload=layout_variants,
    )
    if not drafts:
        raise RuntimeError(
            "Could not convert pipeline output into a UI layout payload."
        )
    draft = drafts[min(option_index, len(drafts)) - 1]
    design_json = draft.design_json or {
        "floorplan_geometry": draft.floorplan_json,
        "placements": [],
    }
    payload: JSONObject = {
        "case_id": benchmark_case.case_id,
        "room_type": benchmark_case.room_type,
        "option_id": draft.option_id,
        "option_index": draft.option_index,
        "name": draft.name,
        "pipeline_case_dir": str(case_dir),
        "pipeline_output": _json_value(pipeline_payload),
        "floorplan": _json_value(draft.floorplan_json),
        "design": _json_value(design_json),
        "styled_result": _json_value(draft.styled_result_json),
        "absolute_layout": _json_value(absolute_layout),
        "layout_variants": _json_value(layout_variants),
        "meta": _json_value(draft.meta),
        "user_input": _json_value(_user_input_for_ui(benchmark_case)),
    }
    return payload


def _layout_inventory_integrity_report(
    *,
    pipeline_payload: Mapping[str, object],
    layout_payload: Mapping[str, object],
) -> JSONObject:
    final_output = _mapping(pipeline_payload.get("final_output"))
    styled_result = _mapping(layout_payload.get("styled_result"))
    design = _mapping(layout_payload.get("design"))
    final_ids = _object_inventory_ids(final_output.get("objects"))
    styled_ids = _object_inventory_ids(styled_result.get("objects"))
    design_ids = _object_inventory_ids(design.get("placements"))
    if not final_ids or not styled_ids or not design_ids:
        return {
            "status": "failed",
            "reason": "inventory_ids_missing",
            "final_output_ids": final_ids,
            "styled_result_ids": styled_ids,
            "design_placement_ids": design_ids,
        }
    final_set = set(final_ids)
    styled_set = set(styled_ids)
    design_set = set(design_ids)
    duplicate_ids = sorted(
        {
            object_id
            for object_ids in (final_ids, styled_ids, design_ids)
            for object_id in object_ids
            if object_ids.count(object_id) > 1
        }
    )
    extra_styled_ids = sorted(styled_set - final_set)
    missing_styled_ids = sorted(final_set - styled_set)
    extra_design_ids = sorted(design_set - styled_set)
    missing_design_ids = sorted(styled_set - design_set)
    passed = not (
        duplicate_ids
        or extra_styled_ids
        or missing_styled_ids
        or extra_design_ids
        or missing_design_ids
    )
    return {
        "status": "passed" if passed else "failed",
        "reason": "inventory_match" if passed else "inventory_drift",
        "final_output_ids": final_ids,
        "styled_result_ids": styled_ids,
        "design_placement_ids": design_ids,
        "duplicate_ids": duplicate_ids,
        "extra_styled_ids": extra_styled_ids,
        "missing_styled_ids": missing_styled_ids,
        "extra_design_ids": extra_design_ids,
        "missing_design_ids": missing_design_ids,
    }


def _object_inventory_ids(value: object) -> list[str]:
    object_ids: list[str] = []
    for item in _sequence(value):
        payload = _mapping(item)
        object_id = (
            _string(payload.get("instance_id"))
            or _string(payload.get("placement_id"))
            or _string(payload.get("object_id"))
            or _string(payload.get("id"))
        )
        if object_id:
            object_ids.append(object_id)
    return object_ids


def _capture_layout_snapshot(
    *,
    chrome: ChromeSession,
    artifact: LayoutArtifact,
    max_camera_presets: int,
    capture_timeout_s: float,
    reuse_selected_camera_preset: bool,
) -> None:
    layout = _read_json_mapping(artifact.layout_path)
    eval_payload = {
        "caseId": artifact.case.case_id,
        "variantId": _string(layout.get("option_id")) or "option_1",
        "floorplan": layout.get("floorplan"),
        "design": layout.get("design"),
        "styledResult": layout.get("styled_result"),
        "userInput": layout.get("user_input"),
    }
    chrome.evaluate(
        _async_eval_expression(
            f"window.__TKNT_EVAL__.loadLayout({_json_for_js(eval_payload)})"
        ),
        timeout_s=capture_timeout_s,
    )
    presets_value = chrome.evaluate(
        "window.__TKNT_EVAL__.listCameraPresets()",
        timeout_s=10,
    )
    presets = [
        _mapping(item)
        for item in _sequence(presets_value)
        if _string(_mapping(item).get("id"))
    ]
    selected_preset_id = (
        _load_selected_camera_preset_id(artifact.case_output_dir)
        if reuse_selected_camera_preset
        else None
    )
    if reuse_selected_camera_preset and not selected_preset_id:
        raise RuntimeError(
            "No selected camera preset found in existing snapshot_meta.json."
        )
    presets = _camera_presets_for_capture(
        presets=presets,
        selected_preset_id=selected_preset_id,
        max_camera_presets=max_camera_presets,
    )
    if not presets:
        if selected_preset_id:
            raise RuntimeError(
                f"Selected camera preset is not available: {selected_preset_id}"
            )
        raise RuntimeError("No 3D camera presets are available for this layout.")

    candidates: list[CameraCandidate] = []
    for preset in presets:
        preset_id = _string(preset.get("id"))
        if not preset_id:
            continue
        capture_value = chrome.evaluate(
            _async_eval_expression(
                "window.__TKNT_EVAL__.applyCameraPreset("
                f"{_json_for_js(preset_id)}"
                ").then(() => window.__TKNT_EVAL__.captureSnapshot())"
            ),
            timeout_s=capture_timeout_s,
        )
        capture = _mapping(capture_value)
        snapshot_artifact = _mapping(capture.get("snapshotArtifact"))
        data_url = _string(capture.get("cleanSnapshotImageDataUrl"))
        if not snapshot_artifact or not data_url:
            continue
        score, details = _score_snapshot_candidate(
            preset=preset,
            snapshot_artifact=snapshot_artifact,
        )
        candidates.append(
            CameraCandidate(
                preset=preset,
                score=score,
                details=details,
                snapshot_artifact=snapshot_artifact,
                clean_snapshot_image_data_url=data_url,
                depth_map_image_data_url=_string(capture.get("depthMapImageDataUrl")),
                edge_map_image_data_url=_string(capture.get("edgeMapImageDataUrl")),
                segmentation_map_image_data_url=_string(
                    capture.get("segmentationMapImageDataUrl")
                ),
                instance_outline_image_data_url=_string(
                    capture.get("instanceOutlineImageDataUrl")
                ),
                instance_outline_overlay_image_data_url=_string(
                    capture.get("instanceOutlineOverlayImageDataUrl")
                ),
            )
        )

    if not candidates:
        raise RuntimeError("No camera preset produced a valid snapshot.")
    best = max(candidates, key=lambda item: item.score)
    visible_count = int(_number(best.snapshot_artifact.get("visibleObjectCount")) or 0)
    if visible_count <= 0:
        raise RuntimeError("No camera preset has visible layout objects.")

    camera_index_path = _write_camera_snapshot_gallery(
        artifact=artifact,
        candidates=candidates,
        selected_candidate=best,
    )
    _write_data_url_png(
        best.clean_snapshot_image_data_url,
        artifact.case_output_dir / "snapshot.png",
    )
    selected_geometry_maps = _write_snapshot_geometry_maps(
        candidate=best,
        output_dir=artifact.case_output_dir,
        file_prefix=None,
    )
    _write_json(
        artifact.case_output_dir / "snapshot_payload.json",
        best.snapshot_artifact,
    )
    _write_json(
        artifact.case_output_dir / "snapshot_meta.json",
        {
            "case_id": artifact.case.case_id,
            "selected_preset": _json_value(best.preset),
            "score": best.score,
            "score_details": best.details,
            "candidate_count": len(candidates),
            "camera_snapshots_index": _relative_path(
                camera_index_path,
                artifact.case_output_dir,
            ),
            "geometry_maps": selected_geometry_maps,
            "candidates": [
                {
                    "preset": _json_value(candidate.preset),
                    "score": candidate.score,
                    "score_details": candidate.details,
                }
                for candidate in candidates
            ],
        },
    )


def _load_selected_camera_preset_id(case_output_dir: Path) -> str | None:
    meta_path = case_output_dir / "snapshot_meta.json"
    if not meta_path.exists():
        return None
    snapshot_meta = _read_json_mapping(meta_path)
    selected_preset = _mapping(snapshot_meta.get("selected_preset"))
    return _string(selected_preset.get("id"))


def _camera_presets_for_capture(
    *,
    presets: Sequence[Mapping[str, object]],
    selected_preset_id: str | None,
    max_camera_presets: int,
) -> list[Mapping[str, object]]:
    if selected_preset_id:
        return [
            preset
            for preset in presets
            if _string(preset.get("id")) == selected_preset_id
        ]
    if max_camera_presets:
        return list(presets[:max_camera_presets])
    return list(presets)


def _write_camera_snapshot_gallery(
    *,
    artifact: LayoutArtifact,
    candidates: Sequence[CameraCandidate],
    selected_candidate: CameraCandidate,
) -> Path:
    camera_dir = artifact.case_output_dir / "camera_snapshots"
    if camera_dir.exists():
        shutil.rmtree(camera_dir)
    camera_dir.mkdir(parents=True, exist_ok=True)

    entries: list[JSONObject] = []
    digit_count = max(2, len(str(len(candidates))))
    for index, candidate in enumerate(candidates, start=1):
        preset_id = _string(candidate.preset.get("id")) or f"preset_{index}"
        file_stem = f"{index:0{digit_count}d}_{_safe_file_stem(preset_id)}"
        image_path = camera_dir / f"{file_stem}.png"
        payload_path = camera_dir / f"{file_stem}.snapshot_payload.json"
        _write_data_url_png(candidate.clean_snapshot_image_data_url, image_path)
        _write_json(payload_path, candidate.snapshot_artifact)
        geometry_maps = _write_snapshot_geometry_maps(
            candidate=candidate,
            output_dir=camera_dir,
            file_prefix=file_stem,
        )
        entries.append(
            {
                "preset": _json_value(candidate.preset),
                "score": candidate.score,
                "score_details": candidate.details,
                "snapshot_payload": _relative_path(payload_path, camera_dir),
                "snapshot_image": _relative_path(image_path, camera_dir),
                "geometry_maps": geometry_maps,
                "selected": candidate is selected_candidate,
            }
        )

    index_path = camera_dir / "index.json"
    _write_json(
        index_path,
        {
            "case_id": artifact.case.case_id,
            "selected_preset_id": _string(selected_candidate.preset.get("id")),
            "snapshot_count": len(entries),
            "snapshots": entries,
        },
    )
    return index_path


def _score_snapshot_candidate(
    *,
    preset: Mapping[str, object],
    snapshot_artifact: Mapping[str, object],
) -> tuple[float, JSONObject]:
    total_count = max(0, int(_number(snapshot_artifact.get("totalObjectCount")) or 0))
    visible_count = max(
        0,
        int(_number(snapshot_artifact.get("visibleObjectCount")) or 0),
    )
    canvas = _mapping(snapshot_artifact.get("canvas"))
    canvas_width = max(1.0, _number(canvas.get("widthPx")) or 1.0)
    canvas_height = max(1.0, _number(canvas.get("heightPx")) or 1.0)
    canvas_area = canvas_width * canvas_height
    visible_objects = [
        _mapping(item) for item in _sequence(snapshot_artifact.get("visibleObjects"))
    ]
    boxes = [
        _clip_screen_box(
            _mapping(item.get("screenBboxPx")), canvas_width, canvas_height
        )
        for item in visible_objects
    ]
    boxes = [box for box in boxes if box is not None]
    union_area_ratio = _union_box_area_ratio(boxes, canvas_area)
    object_area_ratio = min(1.5, sum(box["area"] for box in boxes) / canvas_area)
    visible_ratio = visible_count / max(1, total_count)
    coverage_balance = _coverage_balance(union_area_ratio)
    diagnostics = _mapping(snapshot_artifact.get("visibilityDiagnostics"))
    diagnostics_status = _string(diagnostics.get("status")) or "unknown"
    preset_id = _string(preset.get("id"))
    group = _string(preset.get("group"))

    score = 0.0
    if visible_count <= 0:
        score -= 1000.0
    score += visible_count * 12.0
    score += visible_ratio * 28.0
    score += min(union_area_ratio / 0.6, 1.0) * 18.0
    score += min(object_area_ratio / 0.85, 1.0) * 12.0
    score += coverage_balance * 16.0
    score += 10.0 if group == "mid" else 2.0
    if "corner" in preset_id:
        score += 4.0
    elif "wall" in preset_id:
        score += 1.5
    min_visible = min(4, total_count)
    if visible_count < min_visible:
        score -= (min_visible - visible_count) * 32.0
    if diagnostics_status != "ok":
        score -= 8.0

    details: JSONObject = {
        "visible_object_count": visible_count,
        "total_object_count": total_count,
        "visible_ratio": visible_ratio,
        "union_area_ratio": union_area_ratio,
        "object_area_ratio": object_area_ratio,
        "coverage_balance": coverage_balance,
        "preset_group": group,
        "visibility_status": diagnostics_status,
    }
    return score, details


def _ensure_frontend(
    *,
    frontend_url: str,
    api_url: str,
    auto_start: bool,
) -> ManagedProcess:
    if _url_reachable(frontend_url):
        return ManagedProcess(None)
    if not auto_start:
        raise RuntimeError(
            f"Frontend is not reachable: {frontend_url}. Start it with "
            "`VITE_ENABLE_EVAL_HOOKS=true npm run dev` from frontend/."
        )
    parsed = urlparse(frontend_url)
    port = parsed.port or 5173
    host = parsed.hostname or "127.0.0.1"
    npm_path = shutil.which("npm")
    if npm_path is None:
        raise RuntimeError("npm was not found; cannot auto-start the frontend.")
    env = os.environ.copy()
    env["VITE_ENABLE_EVAL_HOOKS"] = "true"
    env["VITE_API_BASE"] = api_url
    command = [
        npm_path,
        "run",
        "dev",
        "--",
        "--host",
        host,
        "--port",
        str(port),
    ]
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT / "frontend",
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_url(frontend_url, timeout_s=45)
    except Exception:
        process.terminate()
        raise RuntimeError(
            f"Could not auto-start frontend at {frontend_url}. Run manually: "
            "cd frontend && VITE_ENABLE_EVAL_HOOKS=true "
            f"VITE_API_BASE={api_url} npm run dev"
        ) from None
    return ManagedProcess(process)


def _write_generated_manifest(
    *,
    output_root: Path,
    artifacts: Sequence[LayoutArtifact],
) -> Path:
    cases: list[JSONObject] = []
    for artifact in artifacts:
        case_dir = artifact.case_output_dir
        relative_snapshot = _relative_path(
            case_dir / "snapshot_payload.json", output_root
        )
        relative_image = _relative_path(case_dir / "snapshot.png", output_root)
        relative_camera_index = _relative_path(
            case_dir / "camera_snapshots" / "index.json",
            output_root,
        )
        description = _string(
            _mapping(artifact.case.input_payload.get("user_input")).get("description")
        )
        cases.append(
            {
                "id": artifact.case.case_id,
                "snapshot_payload": relative_snapshot,
                "snapshot_image": relative_image,
                "camera_snapshots": relative_camera_index,
                "render_prompt": description,
                "variants": [
                    {
                        "id": "baseline",
                        "provider": "tknt_snapshot",
                    }
                ],
            }
        )
    manifest: JSONObject = {
        "defaults": {
            "metrics": ["blank", "visual_diff", "psnr", "ssim", "nrmse"],
            "align_mode": "resize_to_reference",
        },
        "output_dir": "runs/generated_snapshot_eval",
        "cases": cases,
    }
    manifest_path = output_root / "manifest.generated.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def _write_run_summary(
    *,
    output_root: Path,
    artifacts: Sequence[LayoutArtifact],
    failed_cases: Sequence[str],
    manifest_path: Path,
) -> Path:
    rows: list[JSONObject] = []
    for artifact in artifacts:
        meta_path = artifact.case_output_dir / "snapshot_meta.json"
        layout_meta_path = artifact.case_output_dir / "layout_run_meta.json"
        snapshot_meta = _read_json_mapping(meta_path) if meta_path.exists() else {}
        layout_meta = (
            _read_json_mapping(layout_meta_path) if layout_meta_path.exists() else {}
        )
        rows.append(
            {
                "case_id": artifact.case.case_id,
                "layout_path": str(artifact.layout_path),
                "pipeline_output_path": str(artifact.pipeline_output_path),
                "snapshot_payload_path": str(
                    artifact.case_output_dir / "snapshot_payload.json"
                ),
                "snapshot_image_path": str(artifact.case_output_dir / "snapshot.png"),
                "camera_snapshots_index_path": str(
                    artifact.case_output_dir / "camera_snapshots" / "index.json"
                ),
                "layout_runtime_s": _json_value(layout_meta.get("runtime_s")),
                "selected_camera_preset": _json_value(
                    _mapping(snapshot_meta.get("selected_preset")).get("id")
                ),
                "camera_score": _json_value(snapshot_meta.get("score")),
                "camera_snapshot_count": _json_value(
                    snapshot_meta.get("candidate_count")
                ),
            }
        )
    summary: JSONObject = {
        "manifest_path": str(manifest_path),
        "failed_cases": list(failed_cases),
        "cases": rows,
    }
    summary_path = output_root / "snapshot_prep_summary.json"
    _write_json(summary_path, summary)
    return summary_path


def _load_cases(path: Path) -> list[BenchmarkCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases: list[BenchmarkCase] = []
    for item in _sequence(payload):
        row = _mapping(item)
        case_id = _string(row.get("case_id"))
        input_payload = _mapping(row.get("input_payload"))
        if not case_id or not input_payload:
            continue
        cases.append(
            BenchmarkCase(
                case_id=case_id,
                room_type=_string(row.get("room_type")) or "room",
                input_payload=input_payload,
                raw=row,
            )
        )
    if not cases:
        raise ValueError(f"No benchmark cases found in {path}")
    return cases


def _load_benchmark_run_cases(patterns: Sequence[str]) -> list[BenchmarkCase]:
    run_dirs = _discover_benchmark_run_dirs(patterns)
    cases: list[BenchmarkCase] = []
    for run_dir in run_dirs:
        input_payload = _read_json_mapping(run_dir / "input.json")
        user_input = _mapping(input_payload.get("user_input"))
        output_payload = _read_json_mapping(run_dir / "output.json")
        score_payload = (
            _read_json_mapping(run_dir / "score.json")
            if (run_dir / "score.json").exists()
            else {}
        )
        source_case_id = _string(output_payload.get("case_id")) or _string(
            score_payload.get("case_id")
        )
        case_id = _safe_file_stem(
            f"{run_dir.parent.name}__{source_case_id or run_dir.name}"
        )
        cases.append(
            BenchmarkCase(
                case_id=case_id,
                room_type=_string(user_input.get("room_type")) or "room",
                input_payload=input_payload,
                raw={
                    "case_id": case_id,
                    "benchmark_run_dir": str(run_dir),
                    "source_case_id": source_case_id or run_dir.name,
                    "source_run_group": run_dir.parent.name,
                },
            )
        )
    return cases


def _discover_benchmark_run_dirs(patterns: Sequence[str]) -> list[Path]:
    run_dirs: list[Path] = []
    seen: set[Path] = set()
    for raw_pattern in patterns:
        for pattern in _parse_csv(raw_pattern):
            resolved_pattern = _resolve_glob_pattern(pattern)
            for match in sorted(glob_paths(resolved_pattern)):
                candidate = Path(match).expanduser().resolve()
                for run_dir in _candidate_benchmark_run_dirs(candidate):
                    if run_dir in seen:
                        continue
                    seen.add(run_dir)
                    run_dirs.append(run_dir)
    return run_dirs


def _candidate_benchmark_run_dirs(candidate: Path) -> list[Path]:
    if _is_benchmark_run_dir(candidate):
        return [candidate]
    if not candidate.is_dir():
        return []
    run_dirs = [
        path.parent
        for path in candidate.rglob("input.json")
        if _is_benchmark_run_dir(path.parent)
    ]
    return sorted(set(run_dirs))


def _is_benchmark_run_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "input.json").exists()
        and (path / "output.json").exists()
    )


def _benchmark_run_dir_for_case(benchmark_case: BenchmarkCase) -> Path | None:
    raw_run_dir = _string(benchmark_case.raw.get("benchmark_run_dir"))
    if not raw_run_dir:
        return None
    return Path(raw_run_dir).expanduser().resolve()


def _select_cases(
    cases: Sequence[BenchmarkCase],
    *,
    include_case_ids: Sequence[str],
) -> list[BenchmarkCase]:
    requested = [case_id for case_id in include_case_ids if case_id]
    if not requested:
        return list(cases)
    by_id = {benchmark_case.case_id: benchmark_case for benchmark_case in cases}
    missing = [case_id for case_id in requested if case_id not in by_id]
    if missing:
        raise ValueError(f"Unknown requested case(s): {', '.join(missing)}")
    return [by_id[case_id] for case_id in requested]


def _room_payload_for_case(benchmark_case: BenchmarkCase) -> JSONObject:
    user_input = _mapping(benchmark_case.input_payload.get("user_input"))
    points = _json_value(_sequence(user_input.get("shape_points")))
    return {
        "room": {
            "room_id": "room_1",
            "name": benchmark_case.case_id,
            "polygon_ccw": points,
            "polygon_mm": points,
        },
        "openings": {
            "doors": _json_value(_sequence(benchmark_case.input_payload.get("doors"))),
            "windows": _json_value(
                _sequence(benchmark_case.input_payload.get("windows"))
            ),
        },
    }


def _user_input_for_ui(benchmark_case: BenchmarkCase) -> JSONObject:
    user_input = _mapping(benchmark_case.input_payload.get("user_input"))
    return {
        "description": _string(user_input.get("description")),
        "room_type": _string(user_input.get("room_type")) or benchmark_case.room_type,
        "style": _string(user_input.get("style")) or "modern",
        "height": _number(user_input.get("height")) or 2800,
        "floor_area_m2": _number(user_input.get("floor_area_m2")) or 0,
        "shape_points": _json_value(_sequence(user_input.get("shape_points"))),
        "windows": int(_number(user_input.get("windows")) or 0),
        "window_direction": _string(user_input.get("window_direction")),
    }


def _pipeline_case_dir(
    pipeline_payload: Mapping[str, object],
    case_id: str,
) -> Path:
    raw_case_dir = _string(pipeline_payload.get("case_dir"))
    if raw_case_dir:
        raw_path = Path(raw_case_dir).expanduser()
        if raw_path.is_absolute():
            return raw_path.resolve()
        return (REPO_ROOT / raw_path).resolve()
    return (
        REPO_ROOT / "experiments" / "image_flow_eval" / "layout_artifacts" / case_id
    ).resolve()


def _resolve_chrome_path(raw_path: str) -> Path:
    if raw_path:
        candidate = Path(raw_path).expanduser()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Chrome executable not found: {candidate}")
    candidates = (
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        Path(
            "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary"
        ),
        Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
        Path("/usr/bin/google-chrome"),
        Path("/usr/bin/chromium"),
        Path("/usr/bin/chromium-browser"),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find Chrome/Chromium. Pass --chrome-path to the browser executable."
    )


def _create_chrome_target(port: int, url: str) -> Mapping[str, object]:
    endpoint = f"http://127.0.0.1:{port}/json/new?{quote(url, safe='')}"
    for method in ("PUT", "GET"):
        try:
            with urlopen(Request(endpoint, method=method), timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
                return _mapping(payload)
        except (HTTPError, URLError):
            continue
    with urlopen(f"http://127.0.0.1:{port}/json/list", timeout=10) as response:
        pages = json.loads(response.read().decode("utf-8"))
    for page in _sequence(pages):
        row = _mapping(page)
        if _string(row.get("webSocketDebuggerUrl")):
            return row
    raise CdpError("Could not create or find a Chrome DevTools page target.")


def _wait_for_document_ready(client: CdpWebSocket, *, timeout_s: float) -> None:
    started_at = time.monotonic()
    while time.monotonic() - started_at <= timeout_s:
        result = client.call(
            "Runtime.evaluate",
            {
                "expression": "document.readyState",
                "returnByValue": True,
            },
            timeout_s=5,
        )
        value = _mapping(result.get("result")).get("value")
        if value in {"interactive", "complete"}:
            return
        time.sleep(0.1)
    raise TimeoutError("Timed out waiting for the frontend document to load.")


def _async_eval_expression(expression: str) -> str:
    return f"(async () => await ({expression}))()"


def _json_for_js(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _write_data_url_png(data_url: str, output_path: Path) -> None:
    if not data_url.startswith("data:image/") or "," not in data_url:
        raise ValueError("Expected an image data URL from the 3D eval bridge.")
    header, image_base64 = data_url.split(",", maxsplit=1)
    mime_type = header.removeprefix("data:").split(";", maxsplit=1)[0]
    if mime_type != "image/png":
        raise ValueError(f"Expected image/png data URL, got {mime_type}.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(base64.b64decode(image_base64))


def _write_snapshot_geometry_maps(
    *,
    candidate: CameraCandidate,
    output_dir: Path,
    file_prefix: str | None,
) -> JSONObject:
    entries: JSONObject = {}
    for key, data_url in (
        ("depth_map", candidate.depth_map_image_data_url),
        ("edge_map", candidate.edge_map_image_data_url),
        ("segmentation_map", candidate.segmentation_map_image_data_url),
        ("instance_outline_guide", candidate.instance_outline_image_data_url),
        ("instance_outline_overlay", candidate.instance_outline_overlay_image_data_url),
    ):
        filename = f"{file_prefix}.{key}.png" if file_prefix else f"{key}.png"
        output_path = output_dir / filename
        if _write_optional_data_url_png(data_url, output_path):
            entries[key] = filename
        else:
            entries[key] = None
    return entries


def _write_optional_data_url_png(data_url: str | None, output_path: Path) -> bool:
    if not data_url:
        return False
    _write_data_url_png(data_url, output_path)
    return True


def _clip_screen_box(
    box: Mapping[str, object],
    canvas_width: float,
    canvas_height: float,
) -> dict[str, float] | None:
    min_x = _number(box.get("minX"))
    min_y = _number(box.get("minY"))
    max_x = _number(box.get("maxX"))
    max_y = _number(box.get("maxY"))
    if min_x is None or min_y is None or max_x is None or max_y is None:
        return None
    left = max(0.0, min(canvas_width, min(min_x, max_x)))
    right = max(0.0, min(canvas_width, max(min_x, max_x)))
    top = max(0.0, min(canvas_height, min(min_y, max_y)))
    bottom = max(0.0, min(canvas_height, max(min_y, max_y)))
    width = max(0.0, right - left)
    height = max(0.0, bottom - top)
    if width <= 0 or height <= 0:
        return None
    return {
        "min_x": left,
        "min_y": top,
        "max_x": right,
        "max_y": bottom,
        "area": width * height,
    }


def _union_box_area_ratio(
    boxes: Sequence[Mapping[str, float]], canvas_area: float
) -> float:
    if not boxes:
        return 0.0
    min_x = min(box["min_x"] for box in boxes)
    min_y = min(box["min_y"] for box in boxes)
    max_x = max(box["max_x"] for box in boxes)
    max_y = max(box["max_y"] for box in boxes)
    return max(0.0, (max_x - min_x) * (max_y - min_y)) / max(1.0, canvas_area)


def _coverage_balance(union_area_ratio: float) -> float:
    if 0.2 <= union_area_ratio <= 0.85:
        return 1.0
    if union_area_ratio < 0.2:
        return max(0.0, union_area_ratio / 0.2)
    return max(0.0, (1.0 - union_area_ratio) / 0.15)


def _wait_for_url(url: str, *, timeout_s: float) -> None:
    started_at = time.monotonic()
    while time.monotonic() - started_at <= timeout_s:
        if _url_reachable(url):
            return
        time.sleep(0.25)
    raise TimeoutError(f"Timed out waiting for {url}.")


def _url_reachable(url: str) -> bool:
    try:
        with urlopen(url, timeout=3):
            return True
    except (HTTPError, URLError, TimeoutError, OSError):
        return False


def _find_free_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
    except PermissionError:
        return 9222


def _resolve_repo_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _resolve_glob_pattern(pattern: str) -> str:
    candidate = Path(pattern).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str(REPO_ROOT / candidate)


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _first_json_mapping(paths: Sequence[Path]) -> Mapping[str, object]:
    for path in paths:
        if path.exists():
            return _read_json_mapping(path)
    return {}


def _read_json_mapping(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(payload)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _read_text_tail(path: Path, *, max_chars: int) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def _read_exact(sock: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise CdpError("Socket closed while reading.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_until(sock: socket.socket, marker: bytes) -> bytes:
    data = bytearray()
    while marker not in data:
        chunk = sock.recv(4096)
        if not chunk:
            raise CdpError("Socket closed while reading.")
        data.extend(chunk)
    return bytes(data)


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _safe_file_stem(value: str) -> str:
    text = value.strip().lower()
    out = []
    previous_was_separator = False
    for char in text:
        if char.isalnum():
            out.append(char)
            previous_was_separator = False
        elif not previous_was_separator:
            out.append("_")
            previous_was_separator = True
    return "".join(out).strip("_") or "item"


def _mapping(value: object) -> Mapping[str, object]:
    return cast(Mapping[str, object], value) if isinstance(value, dict) else {}


def _sequence(value: object) -> Sequence[object]:
    return cast(Sequence[object], value) if isinstance(value, list) else ()


def _string(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _json_value(value: object) -> JSONValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
