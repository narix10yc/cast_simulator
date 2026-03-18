#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

DEFAULT_WORKSPACE = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = Path(".vscode/compile_commands.json")
COMMON_CPP_FLAGS = ["-std=c++17", "-fPIC", "-fexceptions"]
FILTERED_LLVM_CXXFLAGS = {"-std=c++17", "-fno-exceptions"}
LLVM_CONFIG_SETTINGS_KEY = "rust-analyzer.cargo.extraEnv"
LLVM_CONFIG_ENV_VARS = ("LLVM_CONFIG",)
CUDA_ROOT_ENV_VARS = ("CUDA_PATH", "CUDA_HOME")
DEFAULT_CXX = "c++"

CPU_SOURCES = [
    "src/cpp/cpu.cpp",
    "src/cpp/cpu_gen.cpp",
    "src/cpp/cpu_jit.cpp",
]

CUDA_SOURCES = [
    "src/cpp/cuda.cpp",
    "src/cpp/cuda_gen.cpp",
    "src/cpp/cuda_jit.cpp",
    "src/cpp/cuda_exec.cpp",
]

CUDA_ROOT_CANDIDATES = [
    "/usr/local/cuda",
    "/usr/cuda",
]


def load_vscode_settings(workspace: Path) -> dict:
    settings_path = workspace / ".vscode" / "settings.json"
    if not settings_path.is_file():
        return {}

    try:
        return json.loads(settings_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"failed to parse {settings_path}: {exc}") from exc


def resolve_llvm_config(workspace: Path) -> str:
    for env_var in LLVM_CONFIG_ENV_VARS:
        env_value = os.environ.get(env_var)
        if env_value:
            return env_value

    settings = load_vscode_settings(workspace)
    container = settings.get(LLVM_CONFIG_SETTINGS_KEY)
    if isinstance(container, dict):
        value = container.get("LLVM_CONFIG")
        if isinstance(value, str) and value:
            return value

    raise SystemExit(
        "LLVM_CONFIG is not set and no fallback was found in "
        ".vscode/settings.json"
    )


def split_shell_flags(output: str) -> list[str]:
    return shlex.split(output.strip())


def llvm_config_flags(llvm_config: str, *args: str) -> list[str]:
    proc = subprocess.run(
        [llvm_config, *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"`{llvm_config} {' '.join(args)}` failed with {proc.returncode}: "
            f"{proc.stderr.strip()}"
        )
    return split_shell_flags(proc.stdout)


def llvm_cxxflags(llvm_config: str) -> list[str]:
    return [
        flag
        for flag in llvm_config_flags(llvm_config, "--cxxflags")
        if flag not in FILTERED_LLVM_CXXFLAGS
    ]


def existing_dirs(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = str(path.resolve(strict=False))
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.is_dir():
            result.append(path)
    return result


def candidate_cuda_roots() -> list[Path]:
    candidates: list[Path] = []

    for env_var in CUDA_ROOT_ENV_VARS:
        value = os.environ.get(env_var)
        if value:
            candidates.append(Path(value))

    candidates.extend(Path(path) for path in CUDA_ROOT_CANDIDATES)
    for candidate in sorted(Path("/usr/local").glob("cuda-*")):
        candidates.append(candidate)

    return candidates


def cuda_toolkit_root() -> Path | None:
    candidates = candidate_cuda_roots()
    for candidate in existing_dirs(candidates):
        if (candidate / "include" / "cuda.h").is_file():
            return candidate
        if any(candidate.glob("targets/*/include/cuda.h")):
            return candidate
    return None


def cuda_include_dirs(toolkit_root: Path | None) -> list[str]:
    if toolkit_root is None:
        return []

    candidates = [
        toolkit_root / "include",
        *sorted(toolkit_root.glob("targets/*/include")),
    ]
    return [f"-I{path}" for path in existing_dirs(candidates)]


def cuda_compile_flags() -> list[str]:
    toolkit_root = cuda_toolkit_root()
    if toolkit_root is None:
        return []

    return [
        f"--cuda-path={toolkit_root}",
        *cuda_include_dirs(toolkit_root),
    ]


def compile_command(
    workspace: Path,
    compiler: str,
    source: str,
    llvm_flags: list[str],
    extra_flags: list[str],
) -> dict:
    source_path = workspace / source
    return {
        "directory": str(workspace),
        "file": str(source_path),
        "arguments": [
            compiler,
            *COMMON_CPP_FLAGS,
            *llvm_flags,
            *extra_flags,
            "-c",
            str(source_path),
        ],
    }


def generate_database(workspace: Path) -> list[dict]:
    compiler = os.environ.get("CXX", DEFAULT_CXX)
    llvm_flags = llvm_cxxflags(resolve_llvm_config(workspace))
    cuda_flags = cuda_compile_flags()

    source_groups = (
        (CPU_SOURCES, []),
        (CUDA_SOURCES, cuda_flags),
    )
    return [
        compile_command(workspace, compiler, source, llvm_flags, extra_flags)
        for sources, extra_flags in source_groups
        for source in sources
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate compile_commands.json for VS Code clangd."
    )
    parser.add_argument(
        "--workspace",
        default=DEFAULT_WORKSPACE,
        type=Path,
        help="workspace root",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help="output compile_commands.json path",
    )
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    output = args.output or workspace / DEFAULT_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)

    database = generate_database(workspace)
    output.write_text(json.dumps(database, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
