from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.layout_benchmark.cache_env import (  # noqa: E402
    clear_run_cache,
    configure_fresh_cache_env,
)
from evaluation.layout_benchmark.external_runners import _external_command  # noqa: E402


class LayoutBenchmarkCacheEnvTest(unittest.TestCase):
    def test_configure_fresh_cache_env_removes_previous_cache(self) -> None:
        with TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            old_cache_file = run_dir / ".cache" / "old_cache.txt"
            old_cache_file.parent.mkdir()
            old_cache_file.write_text("cached", encoding="utf-8")
            env = {
                "XDG_CACHE_HOME": "/tmp/shared-cache",
                "MPLCONFIGDIR": "/tmp/shared-matplotlib",
            }

            configure_fresh_cache_env(env, run_dir)

            self.assertEqual(env["XDG_CACHE_HOME"], str(run_dir / ".cache"))
            self.assertEqual(
                env["MPLCONFIGDIR"],
                str(run_dir / ".cache" / "matplotlib"),
            )
            self.assertFalse(old_cache_file.exists())
            self.assertTrue((run_dir / ".cache" / "matplotlib").is_dir())

    def test_clear_run_cache_removes_cache_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            cache_path = run_dir / ".cache"
            cache_path.write_text("cached", encoding="utf-8")

            clear_run_cache(run_dir)

            self.assertFalse(cache_path.exists())

    def test_idesign_external_command_disables_autogen_cache(self) -> None:
        with TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            _, _, env = _external_command(
                system_id="idesign_openai",
                input_path=run_dir / "input.json",
                output_path=run_dir / "output.json",
                benchmark_case={
                    "case_id": "cache_case",
                    "input_payload": {
                        "user_input": {
                            "description": "Design a compact bedroom.",
                        },
                    },
                },
                model="gpt-5.4-mini",
                idesign_objects=8,
            )

            self.assertEqual(env["IDESIGN_DISABLE_CACHE"], "1")


if __name__ == "__main__":
    unittest.main()
