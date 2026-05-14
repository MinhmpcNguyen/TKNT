# Benchmark Requirements

Use separate Python environments for the three systems. The dependency sets are
not intended to be installed into the same environment.

Create or refresh all three environments with one command:

```bash
./evaluation/layout_benchmark/setup_envs.sh
```

The default env prefixes are configured in `evaluation/layout_benchmark/.env`:

```env
TKNT_ENV_PREFIX=.conda_envs/benchmark-tknt
IDESIGN_ENV_PREFIX=.conda_envs/benchmark-idesign
FLAIRGPT_ENV_PREFIX=.conda_envs/benchmark-flairgpt
```

`run_one.sh` uses TKNT's Python for the benchmark runner, then dispatches
IDesign and FlairGPT subprocesses through their own Python executables.

```bash
python -m pip install -r evaluation/layout_benchmark/requirements/tknt.txt
python -m pip install -r evaluation/layout_benchmark/requirements/idesign.txt
python -m pip install -r evaluation/layout_benchmark/requirements/flairgpt.txt
```

Recommended environment split:

- TKNT: Python 3.11, derived from the local `prj-tknt` conda environment.
- IDesign: Python 3.9, matching the original IDesign README.
- FlairGPT: Python 3.12 or a compatible modern Python version.
