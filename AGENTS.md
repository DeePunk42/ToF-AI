# Repository Guidelines

## Project Structure & Module Organization
- Firmware lives in `main/`: `main.ino` is the sketch entry, `ToF.*` handles VL53L5CX sampling, `ml.*` wires TensorFlow Lite Micro inference, and `model.*` embeds the quantized weights.
- Pretrained assets sit in `model/` (`CNN2D_ST_HandPosture_8classes.h5` plus config) and support files in `MODEL_DATASET_ANALYSIS.md` and `CONVERSION_REPORT.md`; keep large binaries out of reviews unless they change.
- Top-level docs (`README.md`, `SUMMARY_CN.md`, `PROJECT_COMPLETE.md`) describe the conversion pipeline and hardware expectations; update them when behavior changes.

## Build, Test, and Development Commands
- Create a local env for any Python-side work (model inspection, future conversion scripts): `python -m venv .venv && source .venv/bin/activate && pip install -e .` (deps come from `pyproject.toml`).
- Firmware compile with Arduino CLI (replace the board FQBN for your target, e.g., STM32 Nucleo or Arduino Due): `arduino-cli compile --fqbn <board-fqbn> main`.
- Upload after a successful build: `arduino-cli upload -p <serial-port> --fqbn <board-fqbn> main`.
- Confirm runtime via serial: `arduino-cli monitor -p <serial-port> -b 115200` and watch for `[+] Predicted:` lines while exercising gestures.

## Coding Style & Naming Conventions
- C/C++ in `main/` uses 2-space indents per scope, braces on the same line, and snake_case for variables/functions; keep logging concise and prefix status with `[+]`, `[*]`, or `[-]` as already used.
- Avoid dynamic allocation in firmware paths; favor `constexpr`, stack buffers, and fixed-size arrays to stay within microcontroller limits.
- If you add Python utilities, follow PEP 8 with 4-space indents and keep scripts idempotent so they can run on constrained environments.

## Testing Guidelines
- There is no automated test harness; validate changes by compiling, flashing, and confirming stable predictions on hardware with the VL53L5CX attached.
- When modifying preprocessing, log tensor shapes/scales and ensure `ml::quantize_float_to_int8` still matches the model’s zero-point/scale; capture a short serial trace showing inference time and top-1 output.
- For new utilities, prefer lightweight tests (e.g., `pytest` with tiny fixtures) and document how to run them in-line with the added script.

## Commit & Pull Request Guidelines
- Use short, imperative commit subjects (e.g., “Improve ToF quantization path”), and keep related changes in a single commit when possible.
- PRs should state target board, key changes, and tests performed (build, flash, serial output). Attach serial logs or screenshots of inference results when behavior changes.
- Update relevant docs when the data pipeline, model format, or runtime parameters change; flag any new dependencies (Arduino libraries, Python packages) in the PR description.
