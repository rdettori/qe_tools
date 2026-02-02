#!/usr/bin/env python3
"""
qe_pw_to_pwin.py

Update a Quantum ESPRESSO pw.in using the last geometry from pw.out.
The script preserves the original pw.in structure while replacing:
  - ATOMIC_POSITIONS coordinates (keeps constraints if present)
  - CELL_PARAMETERS vectors (if present in pw.in and in pw.out)
Optionally updates pseudo_dir in &CONTROL.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

BOHR_TO_ANG = 0.529177210903


def ffloat(x: str) -> float:
    """Parse Fortran-style float with d/D exponents."""
    return float(x.strip().replace("D", "E").replace("d", "e"))


@dataclass
class CellBlock:
    vectors: List[List[float]]
    units: str


@dataclass
class PositionsBlock:
    units: str
    symbols: List[str]
    positions: List[List[float]]


def read_lines(path: str) -> List[str]:
    with open(path, "r", errors="replace") as f:
        return f.read().splitlines()


def parse_alat_bohr(lines: List[str]) -> Optional[float]:
    for line in lines:
        m = re.search(r"(?i)lattice parameter \(alat\)\s*=\s*([\d.eEdD+-]+)\s*a\.u\.", line)
        if m:
            return ffloat(m.group(1))
    return None


def parse_alat_bohr_from_input(lines: List[str]) -> Optional[float]:
    for line in lines:
        m = re.search(r"(?i)celldm\(1\)\s*=\s*([\d.eEdD+-]+)", line)
        if m:
            return ffloat(m.group(1))
        m = re.search(r"(?i)\bA\s*=\s*([\d.eEdD+-]+)", line)
        if m:
            angstrom = ffloat(m.group(1))
            return angstrom / BOHR_TO_ANG
    return None


def parse_cell_parameters(lines: List[str]) -> List[CellBlock]:
    blocks: List[CellBlock] = []
    for i, line in enumerate(lines):
        if "CELL_PARAMETERS" in line:
            m = re.search(r"CELL_PARAMETERS\s*\(([^)]+)\)", line)
            if m:
                units = m.group(1).strip()
            else:
                parts = line.split()
                units = parts[1].strip() if len(parts) > 1 else "alat"
            vectors: List[List[float]] = []
            for j in range(i + 1, min(i + 4, len(lines))):
                parts = lines[j].split()
                if len(parts) < 3:
                    break
                vectors.append([ffloat(parts[0]), ffloat(parts[1]), ffloat(parts[2])])
            if len(vectors) == 3:
                blocks.append(CellBlock(vectors=vectors, units=units))
    return blocks


def parse_crystal_axes(lines: List[str]) -> Optional[List[List[float]]]:
    axes_start = None
    for i, line in enumerate(lines):
        if "crystal axes" in line.lower():
            axes_start = i
            break
    if axes_start is None:
        return None

    vectors: List[List[float]] = []
    for j in range(axes_start + 1, min(axes_start + 4, len(lines))):
        m = re.search(r"\(([^)]+)\)", lines[j])
        if not m:
            break
        parts = m.group(1).split()
        if len(parts) < 3:
            break
        vectors.append([ffloat(parts[0]), ffloat(parts[1]), ffloat(parts[2])])
    if len(vectors) != 3:
        return None
    return vectors


def parse_atomic_positions(lines: List[str]) -> List[PositionsBlock]:
    blocks: List[PositionsBlock] = []
    pos_re = re.compile(r"^\s*([A-Za-z][A-Za-z0-9]*)\s+([\d.eEdD+-]+)\s+([\d.eEdD+-]+)\s+([\d.eEdD+-]+)")
    for i, line in enumerate(lines):
        if "ATOMIC_POSITIONS" in line:
            m = re.search(r"ATOMIC_POSITIONS\s*\(([^)]+)\)", line)
            units = m.group(1).strip() if m else "alat"
            symbols: List[str] = []
            positions: List[List[float]] = []
            for j in range(i + 1, len(lines)):
                if not lines[j].strip():
                    break
                mpos = pos_re.match(lines[j])
                if not mpos:
                    break
                symbols.append(mpos.group(1))
                positions.append([
                    ffloat(mpos.group(2)),
                    ffloat(mpos.group(3)),
                    ffloat(mpos.group(4)),
                ])
            if symbols:
                blocks.append(PositionsBlock(units=units, symbols=symbols, positions=positions))
    return blocks


def convert_cell_to_angstrom(cell: CellBlock, alat_bohr: Optional[float]) -> List[List[float]]:
    units = cell.units.lower()
    scale = 1.0
    if "angstrom" in units:
        scale = 1.0
    elif "bohr" in units or "a.u." in units:
        scale = BOHR_TO_ANG
    elif "alat" in units:
        m = re.search(r"alat\s*=\s*([\d.eEdD+-]+)", units)
        alat_val = ffloat(m.group(1)) if m else alat_bohr
        if alat_val is None:
            raise ValueError("Missing alat value for CELL_PARAMETERS (alat).")
        scale = alat_val * BOHR_TO_ANG
    else:
        if alat_bohr is None:
            raise ValueError(f"Unknown CELL_PARAMETERS units: {cell.units}")
        scale = alat_bohr * BOHR_TO_ANG
    return [[v * scale for v in row] for row in cell.vectors]


def positions_to_cartesian_angstrom(
    block: PositionsBlock,
    alat_bohr: Optional[float],
    cell_ang: Optional[List[List[float]]],
) -> List[List[float]]:
    units = block.units.lower()
    if "crystal" in units:
        if cell_ang is None:
            raise ValueError("CELL_PARAMETERS required to convert crystal coordinates.")
        return [
            [
                pos[0] * cell_ang[0][0] + pos[1] * cell_ang[1][0] + pos[2] * cell_ang[2][0],
                pos[0] * cell_ang[0][1] + pos[1] * cell_ang[1][1] + pos[2] * cell_ang[2][1],
                pos[0] * cell_ang[0][2] + pos[1] * cell_ang[1][2] + pos[2] * cell_ang[2][2],
            ]
            for pos in block.positions
        ]

    scale = 1.0
    if "angstrom" in units:
        scale = 1.0
    elif "bohr" in units or "a.u." in units:
        scale = BOHR_TO_ANG
    elif "alat" in units:
        if alat_bohr is None:
            raise ValueError("Missing alat value for ATOMIC_POSITIONS (alat).")
        scale = alat_bohr * BOHR_TO_ANG
    else:
        if alat_bohr is None:
            raise ValueError(f"Unknown ATOMIC_POSITIONS units: {block.units}")
        scale = alat_bohr * BOHR_TO_ANG

    return [[v * scale for v in row] for row in block.positions]


def invert_3x3(matrix: List[List[float]]) -> List[List[float]]:
    a, b, c = matrix
    det = (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )
    if abs(det) < 1e-14:
        raise ValueError("CELL_PARAMETERS matrix is singular.")
    inv_det = 1.0 / det
    return [
        [
            (b[1] * c[2] - b[2] * c[1]) * inv_det,
            (a[2] * c[1] - a[1] * c[2]) * inv_det,
            (a[1] * b[2] - a[2] * b[1]) * inv_det,
        ],
        [
            (b[2] * c[0] - b[0] * c[2]) * inv_det,
            (a[0] * c[2] - a[2] * c[0]) * inv_det,
            (a[2] * b[0] - a[0] * b[2]) * inv_det,
        ],
        [
            (b[0] * c[1] - b[1] * c[0]) * inv_det,
            (a[1] * c[0] - a[0] * c[1]) * inv_det,
            (a[0] * b[1] - a[1] * b[0]) * inv_det,
        ],
    ]


def cartesian_to_fractional(cart: List[List[float]], cell_ang: List[List[float]]) -> List[List[float]]:
    inv = invert_3x3(cell_ang)
    return [
        [
            pos[0] * inv[0][0] + pos[1] * inv[1][0] + pos[2] * inv[2][0],
            pos[0] * inv[0][1] + pos[1] * inv[1][1] + pos[2] * inv[2][1],
            pos[0] * inv[0][2] + pos[1] * inv[1][2] + pos[2] * inv[2][2],
        ]
        for pos in cart
    ]


def cartesian_to_units(
    cart: List[List[float]],
    target_units: str,
    cell_ang: Optional[List[List[float]]],
    alat_bohr: Optional[float],
) -> List[List[float]]:
    units = target_units.lower()
    if "crystal" in units:
        if cell_ang is None:
            raise ValueError("CELL_PARAMETERS required to convert to crystal coordinates.")
        return cartesian_to_fractional(cart, cell_ang)
    if "angstrom" in units:
        return cart
    if "bohr" in units or "a.u." in units:
        return [[v / BOHR_TO_ANG for v in row] for row in cart]
    if "alat" in units:
        if alat_bohr is None:
            raise ValueError("Missing alat value to convert to alat coordinates.")
        scale = alat_bohr * BOHR_TO_ANG
        return [[v / scale for v in row] for row in cart]
    if alat_bohr is None:
        raise ValueError(f"Unknown target units: {target_units}")
    scale = alat_bohr * BOHR_TO_ANG
    return [[v / scale for v in row] for row in cart]


def parse_units_from_card(line: str, keyword: str) -> str:
    m = re.search(rf"{keyword}\s*\(([^)]+)\)", line, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    parts = line.split()
    if len(parts) > 1:
        return parts[1].strip()
    return "alat"


def find_atomic_positions_block(
    lines: List[str],
) -> Tuple[int, int, str, str, List[str], List[str]]:
    pos_re = re.compile(r"^\s*[A-Za-z][A-Za-z0-9]*\s+[\d.eEdD+-]+\s+[\d.eEdD+-]+\s+[\d.eEdD+-]+")
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("atomic_positions"):
            units = parse_units_from_card(line, "ATOMIC_POSITIONS")
            start = i + 1
            end = start
            suffixes: List[str] = []
            symbols: List[str] = []
            while end < len(lines) and pos_re.match(lines[end]):
                tokens = lines[end].split()
                symbols.append(tokens[0])
                suffixes.append(" ".join(tokens[4:]) if len(tokens) > 4 else "")
                end += 1
            return i, end, line, units, suffixes, symbols
    raise ValueError("ATOMIC_POSITIONS block not found in pw.in.")


def find_cell_parameters_block(lines: List[str]) -> Optional[Tuple[int, int, str, str]]:
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("cell_parameters"):
            units = parse_units_from_card(line, "CELL_PARAMETERS")
            return i, i + 4, line, units
    return None


def update_pseudo_dir(lines: List[str], pseudo_dir: str) -> List[str]:
    if pseudo_dir is None:
        return lines
    updated = list(lines)
    in_control = False
    inserted = False
    indent = " "
    for i, line in enumerate(updated):
        stripped = line.strip()
        if stripped.lower().startswith("&control"):
            in_control = True
            continue
        if in_control and stripped.startswith("/"):
            if not inserted:
                updated.insert(i, f"{indent}pseudo_dir = '{pseudo_dir}'")
                inserted = True
            break
        if in_control:
            if stripped:
                indent = re.match(r"^\s*", line).group(0)
            if re.search(r"(?i)^\s*pseudo_dir\s*=", line):
                updated[i] = re.sub(
                    r"(?i)(^\s*pseudo_dir\s*=\s*)(.*)",
                    rf"\1'{pseudo_dir}'",
                    line,
                )
                inserted = True
                break
    if not inserted and not in_control:
        raise ValueError("CONTROL namelist not found; cannot update pseudo_dir.")
    return updated


def format_positions(
    symbols: List[str],
    positions: List[List[float]],
    suffixes: List[str],
) -> List[str]:
    lines: List[str] = []
    for idx, (sym, pos) in enumerate(zip(symbols, positions)):
        suffix = suffixes[idx] if idx < len(suffixes) else ""
        suffix_part = f" {suffix}" if suffix else ""
        lines.append(f"{sym:<2} {pos[0]:.16f} {pos[1]:.16f} {pos[2]:.16f}{suffix_part}")
    return lines


def format_cell(vectors: List[List[float]]) -> List[str]:
    return [f"   {vec[0]:.16f} {vec[1]:.16f} {vec[2]:.16f}" for vec in vectors]


def convert_cell_from_angstrom(cell_ang: List[List[float]], units: str, alat_bohr: Optional[float]) -> List[List[float]]:
    units_lower = units.lower()
    if "angstrom" in units_lower:
        return cell_ang
    if "bohr" in units_lower or "a.u." in units_lower:
        return [[v / BOHR_TO_ANG for v in row] for row in cell_ang]
    if "alat" in units_lower:
        if alat_bohr is None:
            raise ValueError("Missing alat value to convert CELL_PARAMETERS (alat).")
        scale = alat_bohr * BOHR_TO_ANG
        return [[v / scale for v in row] for row in cell_ang]
    if alat_bohr is None:
        raise ValueError(f"Unknown CELL_PARAMETERS units: {units}")
    scale = alat_bohr * BOHR_TO_ANG
    return [[v / scale for v in row] for row in cell_ang]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Update pw.in with last geometry from pw.out."
    )
    parser.add_argument("pw_out", help="Path to pw.out")
    parser.add_argument("pw_in", help="Path to previous pw.in")
    parser.add_argument("-o", "--output", default="pw_updated.in", help="Output pw.in path")
    parser.add_argument("--pseudo-dir", default=None, help="Override pseudo_dir path")
    args = parser.parse_args()

    if not os.path.isfile(args.pw_out):
        print(f"pw.out not found: {args.pw_out}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.pw_in):
        print(f"pw.in not found: {args.pw_in}", file=sys.stderr)
        return 1

    out_lines = read_lines(args.pw_out)
    in_lines = read_lines(args.pw_in)

    positions_blocks = parse_atomic_positions(out_lines)
    if not positions_blocks:
        print("No ATOMIC_POSITIONS blocks found in pw.out.", file=sys.stderr)
        return 1
    last_positions = positions_blocks[-1]

    alat_bohr = parse_alat_bohr(out_lines)
    if alat_bohr is None:
        alat_bohr = parse_alat_bohr_from_input(in_lines)

    out_cell_blocks = parse_cell_parameters(out_lines)
    cell_ang: Optional[List[List[float]]] = None
    if out_cell_blocks:
        cell_ang = convert_cell_to_angstrom(out_cell_blocks[-1], alat_bohr)
    else:
        axes = parse_crystal_axes(out_lines)
        if axes is not None:
            if alat_bohr is None:
                print("Missing lattice parameter (alat) for crystal axes conversion.", file=sys.stderr)
                return 1
            cell_ang = [[v * alat_bohr * BOHR_TO_ANG for v in row] for row in axes]

    input_cell_blocks = parse_cell_parameters(in_lines)
    input_cell_ang = None
    if input_cell_blocks:
        input_cell_ang = convert_cell_to_angstrom(input_cell_blocks[-1], alat_bohr)
        if cell_ang is None:
            cell_ang = input_cell_ang

    cell_for_positions = cell_ang or input_cell_ang
    cart_positions = positions_to_cartesian_angstrom(last_positions, alat_bohr, cell_for_positions)

    pos_start, pos_end, pos_header, pos_units, pos_suffixes, input_symbols = find_atomic_positions_block(in_lines)
    updated_positions = cartesian_to_units(cart_positions, pos_units, cell_for_positions, alat_bohr)

    if input_symbols and input_symbols != last_positions.symbols:
        print("Warning: atom symbols in pw.out differ from pw.in; using pw.out order.", file=sys.stderr)

    new_position_lines = format_positions(last_positions.symbols, updated_positions, pos_suffixes)

    updated_lines = list(in_lines)
    updated_lines[pos_start + 1:pos_end] = new_position_lines

    cell_block_info = find_cell_parameters_block(updated_lines)
    if cell_block_info and cell_ang is not None:
        cell_start, cell_end, cell_header, cell_units = cell_block_info
        updated_cell = convert_cell_from_angstrom(cell_ang, cell_units, alat_bohr)
        updated_lines[cell_start + 1:cell_end] = format_cell(updated_cell)

    if args.pseudo_dir:
        updated_lines = update_pseudo_dir(updated_lines, args.pseudo_dir)

    with open(args.output, "w") as f:
        f.write("\n".join(updated_lines) + "\n")

    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
