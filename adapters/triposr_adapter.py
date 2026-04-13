from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class TripoSRRunConfig:
    triposr_dir: Path
    python_executable: Path = Path(sys.executable)
    pretrained_model_name_or_path: str = "stabilityai/TripoSR"
    device: str = "auto"
    chunk_size: int | None = None
    mc_resolution: int = 256
    mc_threshold: float = 12.0
    output_format: Literal["glb", "obj"] = "glb"
    bake_texture_when_available: bool = True
    texture_resolution: int = 2048


@dataclass(frozen=True)
class TripoSRRunResult:
    mesh_path: Path
    preferred_mesh_path: Path
    textured_mesh_path: Path | None
    textured_mesh_source_path: Path | None
    texture_image_path: Path | None
    stdout_path: Path
    stderr_path: Path
    command: list[str]
    texture_stdout_path: Path | None
    texture_stderr_path: Path | None
    texture_command: list[str] | None
    texture_bake_error: str | None


@dataclass(frozen=True)
class _TripoSRCommandResult:
    mesh_path: Path
    stdout_path: Path
    stderr_path: Path
    command: list[str]


def run_triposr(
    *,
    image_path: Path,
    output_dir: Path,
    config: TripoSRRunConfig,
) -> TripoSRRunResult:
    resolved_triposr_dir = config.triposr_dir.expanduser().resolve()
    resolved_python = config.python_executable.expanduser().resolve()
    resolved_image_path = image_path.expanduser().resolve()
    resolved_output_dir = output_dir.expanduser().resolve()

    if not resolved_image_path.exists():
        raise FileNotFoundError(f"TripoSR input image not found: {resolved_image_path}")
    if not (resolved_triposr_dir / "run.py").exists():
        raise FileNotFoundError(
            f"TripoSR entrypoint not found in {resolved_triposr_dir}."
        )

    plain_result = _run_triposr_command(
        triposr_dir=resolved_triposr_dir,
        python_executable=resolved_python,
        image_path=resolved_image_path,
        output_dir=resolved_output_dir,
        output_format=config.output_format,
        pretrained_model_name_or_path=config.pretrained_model_name_or_path,
        device=config.device,
        chunk_size=config.chunk_size,
        mc_resolution=config.mc_resolution,
        mc_threshold=config.mc_threshold,
        bake_texture=False,
        texture_resolution=None,
        stdout_filename="triposr_stdout.log",
        stderr_filename="triposr_stderr.log",
    )

    preferred_mesh_path = plain_result.mesh_path
    textured_mesh_path: Path | None = None
    textured_mesh_source_path: Path | None = None
    texture_image_path: Path | None = None
    texture_stdout_path: Path | None = None
    texture_stderr_path: Path | None = None
    texture_command: list[str] | None = None
    texture_bake_error: str | None = None

    if config.bake_texture_when_available and _python_module_available(
        resolved_python, "xatlas"
    ):
        textured_output_dir = resolved_output_dir / "textured"
        try:
            textured_result = _run_triposr_command(
                triposr_dir=resolved_triposr_dir,
                python_executable=resolved_python,
                image_path=resolved_image_path,
                output_dir=textured_output_dir,
                output_format="obj",
                pretrained_model_name_or_path=config.pretrained_model_name_or_path,
                device=config.device,
                chunk_size=config.chunk_size,
                mc_resolution=config.mc_resolution,
                mc_threshold=config.mc_threshold,
                bake_texture=True,
                texture_resolution=config.texture_resolution,
                stdout_filename="triposr_baked_stdout.log",
                stderr_filename="triposr_baked_stderr.log",
            )
            textured_mesh_source_path = textured_result.mesh_path
            texture_image_path = textured_mesh_source_path.with_name("texture.png")
            if not texture_image_path.exists():
                raise RuntimeError(
                    f"TripoSR texture bake finished without producing {texture_image_path}."
                )
            _ensure_textured_obj_material_bindings(
                obj_path=textured_mesh_source_path,
                texture_path=texture_image_path,
            )
            textured_mesh_path = textured_mesh_source_path.with_suffix(".glb")
            _convert_textured_obj_to_glb(
                obj_path=textured_mesh_source_path,
                output_glb_path=textured_mesh_path,
            )
            preferred_mesh_path = textured_mesh_path
            texture_stdout_path = textured_result.stdout_path
            texture_stderr_path = textured_result.stderr_path
            texture_command = textured_result.command
        except Exception as exc:
            texture_bake_error = str(exc)

    return TripoSRRunResult(
        mesh_path=plain_result.mesh_path,
        preferred_mesh_path=preferred_mesh_path,
        textured_mesh_path=textured_mesh_path,
        textured_mesh_source_path=textured_mesh_source_path,
        texture_image_path=texture_image_path,
        stdout_path=plain_result.stdout_path,
        stderr_path=plain_result.stderr_path,
        command=plain_result.command,
        texture_stdout_path=texture_stdout_path,
        texture_stderr_path=texture_stderr_path,
        texture_command=texture_command,
        texture_bake_error=texture_bake_error,
    )


def _run_triposr_command(
    *,
    triposr_dir: Path,
    python_executable: Path,
    image_path: Path,
    output_dir: Path,
    output_format: Literal["glb", "obj"],
    pretrained_model_name_or_path: str,
    device: str,
    chunk_size: int | None,
    mc_resolution: int,
    mc_threshold: float,
    bake_texture: bool,
    texture_resolution: int | None,
    stdout_filename: str,
    stderr_filename: str,
) -> _TripoSRCommandResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_output_dir = output_dir / "0"
    mesh_output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / stdout_filename
    stderr_path = output_dir / stderr_filename

    command = [
        str(python_executable),
        "run.py",
        str(image_path),
        "--device",
        device,
        "--pretrained-model-name-or-path",
        pretrained_model_name_or_path,
        "--output-dir",
        str(output_dir),
        "--model-save-format",
        output_format,
        "--no-remove-bg",
        "--mc-resolution",
        str(mc_resolution),
        "--mc-threshold",
        str(mc_threshold),
    ]
    if chunk_size is not None:
        command.extend(["--chunk-size", str(chunk_size)])
    if bake_texture:
        command.append("--bake-texture")
        if texture_resolution is not None and texture_resolution > 0:
            command.extend(["--texture-resolution", str(texture_resolution)])

    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0" if device == "cpu" else "1"

    completed = subprocess.run(
        command,
        cwd=triposr_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")

    if completed.returncode != 0:
        stderr_excerpt = completed.stderr.strip()[-1200:]
        raise RuntimeError(
            f"TripoSR failed. See {stderr_path}. Error excerpt:\n{stderr_excerpt}"
        )

    mesh_path = mesh_output_dir / f"mesh.{output_format}"
    if not mesh_path.exists():
        raise RuntimeError(
            f"TripoSR finished without producing {mesh_path}. "
            f"See {stdout_path} and {stderr_path}."
        )

    return _TripoSRCommandResult(
        mesh_path=mesh_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        command=command,
    )


def _python_module_available(python_executable: Path, module_name: str) -> bool:
    completed = subprocess.run(
        [
            str(python_executable),
            "-c",
            (
                "import importlib.util, sys; "
                "sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)"
            ),
            module_name,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0


def _ensure_textured_obj_material_bindings(
    *,
    obj_path: Path,
    texture_path: Path,
) -> None:
    material_path = obj_path.with_suffix(".mtl")
    material_name = "material_0"
    material_path.write_text(
        "\n".join(
            [
                f"newmtl {material_name}",
                "Ka 1.000 1.000 1.000",
                "Kd 1.000 1.000 1.000",
                "Ks 0.000 0.000 0.000",
                "d 1.0",
                "illum 2",
                f"map_Kd {texture_path.name}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    lines = obj_path.read_text(encoding="utf-8").splitlines()
    first_face_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.lstrip().startswith("f ")
        ),
        len(lines),
    )

    normalized_lines: list[str] = []
    has_mtllib = False
    has_usemtl = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("mtllib "):
            normalized_lines.append(f"mtllib {material_path.name}")
            has_mtllib = True
            continue
        if stripped.startswith("usemtl "):
            normalized_lines.append(f"usemtl {material_name}")
            has_usemtl = True
            continue
        normalized_lines.append(line)

    if not has_mtllib:
        normalized_lines.insert(0, f"mtllib {material_path.name}")
        first_face_index += 1
    if not has_usemtl:
        normalized_lines.insert(first_face_index, f"usemtl {material_name}")

    obj_path.write_text("\n".join(normalized_lines) + "\n", encoding="utf-8")


def _convert_textured_obj_to_glb(*, obj_path: Path, output_glb_path: Path) -> None:
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "trimesh is required to convert textured OBJ output into GLB."
        ) from exc

    loaded = trimesh.load(
        obj_path,
        force="scene",
        process=False,
        skip_materials=False,
    )
    scene = loaded.scene() if isinstance(loaded, trimesh.Trimesh) else loaded
    exported = scene.export(file_type="glb")
    if not isinstance(exported, (bytes, bytearray)):
        raise RuntimeError("Trimesh did not return GLB bytes for the textured mesh.")
    output_glb_path.parent.mkdir(parents=True, exist_ok=True)
    output_glb_path.write_bytes(bytes(exported))
