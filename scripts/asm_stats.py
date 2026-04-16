#!/usr/bin/env python3
"""Parse JIT-emitted assembly files and report kernel statistics.

Usage:
    python3 scripts/asm_stats.py dump/*.s
    python3 scripts/asm_stats.py dump/auto_k6.s dump/straight_k6.s   # side-by-side

Detects architecture from instruction patterns (ARM64 vs x86-64) and reports
per-file instruction breakdown with percentages.

Caveats (known, acceptable for our use case):
  - Prologue callee-saves (stp d-regs) counted as spills (~4 extra each way).
  - Block-mode volatile stores are indistinguishable from LLVM spills.
  - Stack frame from max(sub sp, stp pre-decrement), not sum (~5% under).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
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
    const_mat: int = 0
    shuffle: int = 0
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
_ARM64_CONST = re.compile(r"\b(?:movk?|dup\.\d+[sd])\b")
_ARM64_SHUFFLE = re.compile(r"\b(?:zip[12]|uzp[12]|trn[12]|ext|tbl|mov\.16b)\b")
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
_X86_CONST = re.compile(r"\b(?:vmovq|vpbroadcastq|vbroadcastsd)\b")
_X86_SHUFFLE = re.compile(r"\b(?:vshuf|vperm|vunpck|vblend|vinser)")
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
    if _ARM64_SHUFFLE.search(line):
        s.shuffle += 1
        return
    if _ARM64_CONST.search(line):
        s.const_mat += 1
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
    if _X86_SHUFFLE.search(line):
        s.shuffle += 1
        return
    if _X86_CONST.search(line):
        s.const_mat += 1
        return
    if _X86_BRANCH.search(line):
        s.branches += 1
        return
    s.other += 1


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "  -"
    return f"{100 * n / total:3.0f}%"


def print_report(all_stats: list[AsmStats]) -> None:
    for s in all_stats:
        n = s.total_insns
        compute = s.fma + s.mul + s.add_sub
        spills = s.spill_stores + s.spill_loads

        print(f"  {Path(s.path).name}  ({s.arch}, {n:,} insns, stack {s.stack_frame:,} B)")
        print(f"    compute   {compute:>7,}  {_pct(compute, n)}   (fma {s.fma:,}, mul {s.mul:,}, add/sub {s.add_sub:,})")
        print(f"    const     {s.const_mat:>7,}  {_pct(s.const_mat, n)}   matrix constant materialization")
        print(f"    spills    {spills:>7,}  {_pct(spills, n)}   (st {s.spill_stores:,}, ld {s.spill_loads:,})")
        print(f"    shuffle   {s.shuffle:>7,}  {_pct(s.shuffle, n)}   lane permutations")
        print(f"    other     {s.other:>7,}  {_pct(s.other, n)}   (branch {s.branches:,})")
        print()


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

    print_report(all_stats)


if __name__ == "__main__":
    main()
