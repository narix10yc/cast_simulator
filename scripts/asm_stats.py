#!/usr/bin/env python3
"""Parse JIT-emitted assembly files and report kernel statistics.

Usage:
    python3 scripts/asm_stats.py dump/*.s
    python3 scripts/asm_stats.py dump/auto_k6.s dump/straight_k6.s   # side-by-side

Detects architecture from instruction patterns (ARM64 vs x86-64) and reports:
  - Total instructions
  - Stack frame size (bytes)
  - Spill stores / loads (stack-relative vector ops)
  - FMA / multiply / add+sub counts
  - Branch count
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AsmStats:
    path: str = ""
    arch: str = "unknown"
    total_insns: int = 0
    stack_frame: int = 0
    spill_stores: int = 0
    spill_loads: int = 0
    fma: int = 0
    mul: int = 0
    add_sub: int = 0
    branches: int = 0
    other: int = 0


# ---------------------------------------------------------------------------
# ARM64 patterns
# ---------------------------------------------------------------------------

_ARM64_FRAME = re.compile(r"sub\s+sp,\s*sp,\s*#(\d+)")
_ARM64_STP_FRAME = re.compile(r"stp\s+.*\[sp,\s*#-(\d+)\]!")
_ARM64_SPILL_STORE = re.compile(r"\bst[rp]\s+[qvd]\d+.*\[sp")
_ARM64_SPILL_LOAD = re.compile(r"\bld[rp]\s+[qvd]\d+.*\[sp")
_ARM64_FMA = re.compile(r"\bfml[as]\b")
_ARM64_MUL = re.compile(r"\bfmul\b")
_ARM64_ADDSUB = re.compile(r"\bf(?:add|sub)\b")
_ARM64_BRANCH = re.compile(r"\b(?:b|b\.\w+|bl|blr|ret)\b")

# ---------------------------------------------------------------------------
# x86-64 patterns
# ---------------------------------------------------------------------------

_X86_FRAME = re.compile(r"sub[ql]?\s+\$(\d+),\s*%rsp")
_X86_SPILL_STORE = re.compile(r"\bvmov[au]p[sd]\s+%[xyz]mm\d+.*\(%rsp")
_X86_SPILL_LOAD = re.compile(r"\bvmov[au]p[sd]\s+[^,]*\(%rsp.*%[xyz]mm")
_X86_FMA = re.compile(r"\bvfn?m(?:add|sub)")
_X86_MUL = re.compile(r"\bvmulp[sd]\b")
_X86_ADDSUB = re.compile(r"\bv(?:add|sub)p[sd]\b")
_X86_BRANCH = re.compile(r"\b(?:jmp|je|jne|jg|jge|jl|jle|ja|jae|jb|jbe|call|ret)\b")


def detect_arch(lines: list[str]) -> str:
    for line in lines[:200]:
        stripped = line.strip()
        if stripped.startswith(".") or stripped.startswith("//") or not stripped:
            continue
        if re.search(r"\b(stp|ldp|fmla|fadd|fsub|fmul)\b", stripped):
            return "arm64"
        if re.search(r"%rsp|%[xyz]mm|vmov", stripped):
            return "x86_64"
    return "unknown"


def parse_file(path: Path) -> AsmStats:
    lines = path.read_text().splitlines()
    arch = detect_arch(lines)
    stats = AsmStats(path=str(path), arch=arch)

    for raw_line in lines:
        line = raw_line.strip()
        # Skip labels, directives, comments, blank lines.
        if not line or line.startswith(".") or line.startswith("//") or line.startswith("#"):
            continue
        if line.endswith(":"):
            continue

        stats.total_insns += 1

        if arch == "arm64":
            _classify_arm64(line, stats)
        elif arch == "x86_64":
            _classify_x86(line, stats)

    return stats


def _classify_arm64(line: str, s: AsmStats) -> None:
    m = _ARM64_FRAME.search(line)
    if m:
        s.stack_frame = max(s.stack_frame, int(m.group(1)))
        return
    m = _ARM64_STP_FRAME.search(line)
    if m:
        s.stack_frame = max(s.stack_frame, int(m.group(1)))
        return
    if _ARM64_SPILL_STORE.search(line):
        s.spill_stores += 1
        return
    if _ARM64_SPILL_LOAD.search(line):
        s.spill_loads += 1
        return
    if _ARM64_FMA.search(line):
        s.fma += 1
        return
    if _ARM64_MUL.search(line):
        s.mul += 1
        return
    if _ARM64_ADDSUB.search(line):
        s.add_sub += 1
        return
    if _ARM64_BRANCH.search(line):
        s.branches += 1
        return
    s.other += 1


def _classify_x86(line: str, s: AsmStats) -> None:
    m = _X86_FRAME.search(line)
    if m:
        s.stack_frame = max(s.stack_frame, int(m.group(1)))
        return
    if _X86_SPILL_STORE.search(line):
        s.spill_stores += 1
        return
    if _X86_SPILL_LOAD.search(line):
        s.spill_loads += 1
        return
    if _X86_FMA.search(line):
        s.fma += 1
        return
    if _X86_MUL.search(line):
        s.mul += 1
        return
    if _X86_ADDSUB.search(line):
        s.add_sub += 1
        return
    if _X86_BRANCH.search(line):
        s.branches += 1
        return
    s.other += 1


def print_table(all_stats: list[AsmStats]) -> None:
    hdr = (
        f"{'file':<40s} {'arch':<7s} {'insns':>7s} {'frame':>7s} "
        f"{'spill_st':>8s} {'spill_ld':>8s} {'fma':>6s} {'mul':>6s} "
        f"{'add/sub':>7s} {'branch':>7s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for s in all_stats:
        name = Path(s.path).name
        print(
            f"{name:<40s} {s.arch:<7s} {s.total_insns:>7d} {s.stack_frame:>7d} "
            f"{s.spill_stores:>8d} {s.spill_loads:>8d} {s.fma:>6d} {s.mul:>6d} "
            f"{s.add_sub:>7d} {s.branches:>7d}"
        )


def main() -> None:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} FILE.s [FILE2.s ...]", file=sys.stderr)
        sys.exit(1)

    all_stats = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if not p.exists():
            print(f"warning: {p} not found, skipping", file=sys.stderr)
            continue
        all_stats.append(parse_file(p))

    print_table(all_stats)


if __name__ == "__main__":
    main()
