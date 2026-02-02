#!/usr/bin/env python3
"""
qe_analyze_pw.py

Analyze Quantum ESPRESSO pw.x output (pw.out) and optionally pw.in.

Checks:
- whether the run finished ("JOB DONE") and whether relax/vc-relax converged
- number of ionic/relaxation steps (estimated from "Forces acting on atoms" blocks)
- total force (after "Total force =") in the LAST force block
- presence of non-converged SCF cycles
- other errors/fatal messages

Usage examples:
  python qe_analyze_pw.py pw.out
  python qe_analyze_pw.py pw.out --in pw.in
  python qe_analyze_pw.py pw.out -o report.txt
  python qe_analyze_pw.py pw.out --json -o report.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

# Unit conversion: QE forces are typically Ry/au (Ry/bohr). Convert to eV/Å.
RY_TO_EV = 13.605693122994  # CODATA-ish
BOHR_TO_ANG = 0.529177210903
RY_PER_BOHR_TO_EV_PER_ANG = RY_TO_EV / BOHR_TO_ANG  # ~25.711


def ffloat(x: str) -> float:
    """Parse Fortran-style float with d/D exponents."""
    return float(x.strip().replace("D", "E").replace("d", "e"))


def read_text(path: str) -> str:
    with open(path, "r", errors="replace") as f:
        return f.read()


def split_lines(text: str) -> List[str]:
    return text.splitlines()


def parse_pw_in(path: str) -> Dict[str, object]:
    """
    Very lightweight parser for QE input: extracts key = value pairs from namelists.
    Handles strings in quotes and Fortran floats with d exponent.
    """
    if not path:
        return {}
    if not os.path.isfile(path):
        return {}

    txt = read_text(path)
    # Remove comments starting with !
    cleaned_lines = []
    for line in split_lines(txt):
        line = line.split("!")[0]
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines)

    # Regex for key = value (value can be 'string', .true., number, etc.)
    kv = {}
    for m in re.finditer(r"(?im)^\s*([A-Za-z_]\w*)\s*=\s*([^,\n/]+)\s*(?:,|$)", cleaned):
        key = m.group(1).strip()
        val_raw = m.group(2).strip()

        # strip quotes
        if (val_raw.startswith("'") and val_raw.endswith("'")) or (val_raw.startswith('"') and val_raw.endswith('"')):
            kv[key] = val_raw[1:-1]
            continue

        # booleans
        if val_raw.lower() in (".true.", "true"):
            kv[key] = True
            continue
        if val_raw.lower() in (".false.", "false"):
            kv[key] = False
            continue

        # numeric
        try:
            kv[key] = ffloat(val_raw)
        except Exception:
            kv[key] = val_raw  # fallback as string

    return kv


def guess_calculation_from_out(out_text: str) -> Optional[str]:
    # QE usually echoes namelists; try to find calculation = '...'
    m = re.search(r"(?i)\bcalculation\s*=\s*['\"]([^'\"]+)['\"]", out_text)
    if m:
        return m.group(1).strip()
    # Sometimes printed as "calculation  = relax" without quotes in some contexts
    m = re.search(r"(?i)\bcalculation\s*=\s*([A-Za-z_-]+)", out_text)
    if m:
        return m.group(1).strip()
    return None


@dataclass
class ForceSummary:
    total_force_ry_bohr: Optional[float] = None
    total_force_ev_ang: Optional[float] = None


@dataclass
class QEAnalysis:
    pw_out: str
    pw_in: Optional[str]

    calculation: Optional[str]
    job_done: bool
    geometry_converged: Optional[bool]  # None if not applicable/unknown

    ionic_steps_estimate: int

    scf_converged_count: int
    scf_not_converged_count: int
    scf_not_converged_examples: List[str]

    last_force: ForceSummary

    errors: List[str]
    warnings: List[str]

    thresholds: Dict[str, object]


ERROR_PATTERNS = [
    re.compile(r"(?i)^\s*error in routine\b"),
    re.compile(r"(?i)\bterminated\b"),
    re.compile(r"(?i)\baborting\b"),
    re.compile(r"(?i)\bfatal\b"),
    re.compile(r"(?i)\bsegmentation fault\b"),
    re.compile(r"(?i)\bsigsegv\b"),
    re.compile(r"(?i)\bmpi_abort\b"),
    re.compile(r"(?i)\bexceeded.*cpu time\b"),
    re.compile(r"(?i)\bstopping\b"),
    re.compile(r"(?i)\bnan\b"),
    re.compile(r"(?i)\binf\b"),
]

WARNING_PATTERNS = [
    re.compile(r"(?i)^\s*warning\b"),
    re.compile(r"(?i)\bdeprecated\b"),
]


def detect_geometry_convergence(out_text: str, calculation: Optional[str]) -> Optional[bool]:
    """
    Best-effort geometry convergence detection for relax/vc-relax.
    """
    if not calculation:
        return None
    calc = calculation.lower()

    if calc not in ("relax", "vc-relax"):
        return None

    # Common success indicators
    success_markers = [
        r"(?i)\bbfgs converged\b",
        r"(?i)\bend of bfgs geometry optimization\b",
        r"(?i)\bgeometry optimization.*converged\b",
        r"(?i)\bbegin final coordinates\b",
    ]
    # Common failure / non-convergence indicators
    fail_markers = [
        r"(?i)\bmaximum number of iterations\b",
        r"(?i)\bnot converged\b",
        r"(?i)\bconvergence not achieved\b",
    ]

    has_success = any(re.search(p, out_text) for p in success_markers)
    has_fail = any(re.search(p, out_text) for p in fail_markers)

    if has_success and not has_fail:
        return True
    if has_fail and not has_success:
        return False
    if has_success and has_fail:
        # Mixed signals (e.g., SCF not converged at some step but final reached) -> unknown
        return None
    return None


def find_force_block_indices(lines: List[str]) -> List[int]:
    idx = []
    for i, line in enumerate(lines):
        if "Forces acting on atoms" in line:
            idx.append(i)
    return idx


_FORCE_LINE_RE = re.compile(
    r"(?i)^\s*atom\s+(\d+)\s+type\s+\d+\s+force\s*=\s*([-\d.eEdD+]+)\s+([-\d.eEdD+]+)\s+([-\d.eEdD+]+)"
)


def parse_last_force_block(lines: List[str]) -> ForceSummary:
    """
    Parse the LAST 'Forces acting on atoms' block and return the total force.
    Assumes forces are in Ry/bohr (QE prints Ry/au, i.e., Ry/bohr).
    """
    indices = find_force_block_indices(lines)
    if not indices:
        return ForceSummary()

    start = indices[-1]
    total_force = None

    # Scan until blank line streak or until we hit a clearly different section.
    for j in range(start + 1, min(start + 5000, len(lines))):
        line = lines[j].rstrip("\n")
        if "Total force" in line:
            m = re.search(r"(?i)Total force\s*=\s*([-\d.eEdD+]+)", line)
            if m:
                total_force = ffloat(m.group(1))
                break

        # Stop conditions: once we've started parsing forces, a non-force line after
        # seeing at least one force and then a blank or next header.
        if _FORCE_LINE_RE.match(line):
            continue
        if total_force is not None:
            if line.strip() == "":
                break
            if "Entering Dynamics" in line or "Self-consistent Calculation" in line:
                break

    if total_force is None:
        return ForceSummary()

    return ForceSummary(
        total_force_ry_bohr=total_force,
        total_force_ev_ang=total_force * RY_PER_BOHR_TO_EV_PER_ANG,
    )


def count_scf_convergence(lines: List[str]) -> Tuple[int, int, List[str]]:
    """
    Count SCF converged / not converged messages.
    QE typical messages include:
      - "convergence has been achieved in XX iterations"
      - "convergence NOT achieved"
      - "not converged" (various contexts)
    """
    ok = 0
    bad = 0
    examples: List[str] = []

    for i, line in enumerate(lines):
        l = line.strip()
        if re.search(r"(?i)\bconvergence has been achieved\b", l):
            ok += 1
        if re.search(r"(?i)\bconvergence not achieved\b", l) or re.search(r"(?i)\bconvergence NOT achieved\b", l):
            bad += 1
            if len(examples) < 8:
                examples.append(f"line {i+1}: {l}")
        elif re.search(r"(?i)\bscf.*not converged\b", l):
            bad += 1
            if len(examples) < 8:
                examples.append(f"line {i+1}: {l}")
        elif re.search(r"(?i)\btoo many iterations\b", l) and "iteration" in l.lower():
            # Could be SCF or ionic; still useful as a hint
            bad += 1
            if len(examples) < 8:
                examples.append(f"line {i+1}: {l}")

    return ok, bad, examples


def collect_errors_warnings(lines: List[str]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    def add_unique(target: List[str], s: str, maxn: int) -> None:
        if s not in target and len(target) < maxn:
            target.append(s)

    for i, line in enumerate(lines):
        s = line.rstrip("\n")
        for pat in ERROR_PATTERNS:
            if pat.search(s):
                add_unique(errors, f"line {i+1}: {s.strip()}", 40)
                break
        for pat in WARNING_PATTERNS:
            if pat.search(s):
                add_unique(warnings, f"line {i+1}: {s.strip()}", 40)
                break

    # QE common "Error in routine ..." spans multiple lines; keep a bit more context if present
    # If we captured such a line, try to add the next couple lines too.
    extra: List[str] = []
    for e in errors:
        m = re.match(r"line (\d+):", e)
        if not m:
            continue
        ln = int(m.group(1))
        # add next 2 lines as context if they look informative
        for k in range(ln, min(ln + 2, len(lines))):
            s2 = lines[k].strip()
            if s2 and not any(s2 in x for x in errors) and not any(s2 in x for x in extra):
                extra.append(f"line {k+1}: {s2}")
    for x in extra:
        if x not in errors and len(errors) < 60:
            errors.append(x)

    return errors, warnings


def analyze(pw_out_path: str, pw_in_path: Optional[str]) -> QEAnalysis:
    out_text = read_text(pw_out_path)
    lines = split_lines(out_text)

    # thresholds from input (optional)
    inp = parse_pw_in(pw_in_path) if pw_in_path else {}
    thresholds = {k: inp.get(k) for k in ("conv_thr", "forc_conv_thr", "etot_conv_thr", "forc_conv_thr", "nstep")}
    thresholds = {k: v for k, v in thresholds.items() if v is not None}

    calc = None
    if "calculation" in inp and isinstance(inp["calculation"], str):
        calc = inp["calculation"]
    else:
        calc = guess_calculation_from_out(out_text)

    job_done = bool(re.search(r"(?i)\bJOB DONE\b", out_text))

    geom_conv = detect_geometry_convergence(out_text, calc)

    force_block_indices = find_force_block_indices(lines)
    ionic_steps_est = len(force_block_indices)

    last_force = parse_last_force_block(lines)

    scf_ok, scf_bad, scf_bad_examples = count_scf_convergence(lines)

    errors, warnings = collect_errors_warnings(lines)

    return QEAnalysis(
        pw_out=pw_out_path,
        pw_in=pw_in_path,
        calculation=calc,
        job_done=job_done,
        geometry_converged=geom_conv,
        ionic_steps_estimate=ionic_steps_est,
        scf_converged_count=scf_ok,
        scf_not_converged_count=scf_bad,
        scf_not_converged_examples=scf_bad_examples,
        last_force=last_force,
        errors=errors,
        warnings=warnings,
        thresholds=thresholds,
    )


def format_report(res: QEAnalysis) -> str:
    lines: List[str] = []
    lines.append("Quantum ESPRESSO pw.x analysis")
    lines.append("=" * 32)
    lines.append(f"pw.out: {res.pw_out}")
    if res.pw_in:
        lines.append(f"pw.in : {res.pw_in}")
    if res.calculation:
        lines.append(f"calculation: {res.calculation}")
    if res.thresholds:
        lines.append(f"thresholds (from pw.in): {res.thresholds}")

    lines.append("")
    lines.append(f"Finished (JOB DONE): {'YES' if res.job_done else 'NO'}")

    if res.geometry_converged is True:
        lines.append("Geometry convergence (relax/vc-relax): YES")
    elif res.geometry_converged is False:
        lines.append("Geometry convergence (relax/vc-relax): NO")
    elif res.geometry_converged is None and (res.calculation or "").lower() in ("relax", "vc-relax"):
        lines.append("Geometry convergence (relax/vc-relax): UNKNOWN (markers ambiguous/not found)")

    lines.append(f"Ionic/relaxation steps (estimate): {res.ionic_steps_estimate}")

    if res.last_force.total_force_ry_bohr is not None:
        lines.append("")
        lines.append("Last force block:")
        lines.append(
            f"  total force = {res.last_force.total_force_ry_bohr:.6e} Ry/bohr"
            f"  = {res.last_force.total_force_ev_ang:.6e} eV/Å"
        )
    else:
        lines.append("")
        lines.append("Last force block: total force not found (tprnfor may be off, or no forces printed).")

    lines.append("")
    lines.append("SCF convergence summary:")
    lines.append(f"  'convergence has been achieved' count: {res.scf_converged_count}")
    lines.append(f"  non-converged SCF indicators count:     {res.scf_not_converged_count}")
    if res.scf_not_converged_examples:
        lines.append("  examples:")
        for ex in res.scf_not_converged_examples:
            lines.append(f"    - {ex}")

    lines.append("")
    if res.errors:
        lines.append(f"Errors detected ({len(res.errors)}):")
        for e in res.errors[:40]:
            lines.append(f"  - {e}")
        if len(res.errors) > 40:
            lines.append(f"  ... ({len(res.errors) - 40} more)")
    else:
        lines.append("Errors detected: none")

    if res.warnings:
        lines.append("")
        lines.append(f"Warnings detected ({len(res.warnings)}):")
        for w in res.warnings[:25]:
            lines.append(f"  - {w}")
        if len(res.warnings) > 25:
            lines.append(f"  ... ({len(res.warnings) - 25} more)")

    # A simple "overall" status (best-effort)
    lines.append("")
    overall_ok = res.job_done and not res.errors and res.scf_not_converged_count == 0
    if (res.calculation or "").lower() in ("relax", "vc-relax") and res.geometry_converged is False:
        overall_ok = False
    lines.append(f"Overall status (best-effort): {'OK' if overall_ok else 'CHECK'}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Analyze Quantum ESPRESSO pw.out (and optionally pw.in).")
    p.add_argument("pw_out", help="Path to pw.out (pw.x output file).")
    p.add_argument("--in", dest="pw_in", default=None, help="Optional path to pw.in (input file).")
    p.add_argument("-o", "--output", default=None, help="Write report to this file (default: stdout).")
    p.add_argument("--json", action="store_true", help="Output JSON instead of text.")
    p.add_argument("--exit-nonzero", action="store_true",
                   help="Exit with code 1 if overall status is CHECK (useful in scripts).")

    args = p.parse_args()

    if not os.path.isfile(args.pw_out):
        print(f"ERROR: pw.out not found: {args.pw_out}", file=sys.stderr)
        return 2
    if args.pw_in and (not os.path.isfile(args.pw_in)):
        print(f"WARNING: pw.in not found: {args.pw_in} (continuing without it)", file=sys.stderr)
        args.pw_in = None

    res = analyze(args.pw_out, args.pw_in)

    if args.json:
        payload = asdict(res)
        # dataclasses inside dataclasses are already converted by asdict
        out_str = json.dumps(payload, indent=2, sort_keys=False)
    else:
        out_str = format_report(res)

    if args.output:
        with open(args.output, "w") as f:
            f.write(out_str)
            if not out_str.endswith("\n"):
                f.write("\n")
    else:
        print(out_str)

    # Determine "overall" like in report
    overall_ok = res.job_done and not res.errors and res.scf_not_converged_count == 0
    if (res.calculation or "").lower() in ("relax", "vc-relax") and res.geometry_converged is False:
        overall_ok = False

    if args.exit_nonzero and not overall_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
