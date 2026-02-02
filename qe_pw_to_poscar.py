#!/usr/bin/env python3
"""
qe_pw_to_poscar.py

Read a Quantum ESPRESSO pw.x output (pw.out) and write the last geometry
found into a POSCAR file. If no relaxation steps were completed (geometry
was not updated), the script stops without writing.
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


def guess_calculation(lines: List[str]) -> Optional[str]:
    text = "\n".join(lines)
    m = re.search(r"(?i)\bcalculation\s*=\s*['\"]([^'\"]+)['\"]", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?i)\bcalculation\s*=\s*([A-Za-z_-]+)", text)
    if m:
        return m.group(1).strip()
    return None


def count_relax_steps(lines: List[str]) -> int:
    """
    Count completed ionic steps by tracking SCF convergence markers.
    QE prints 'convergence has been achieved in' at the end of each SCF cycle,
    and in relax/vc-relax this corresponds to the end of an ionic step.
    """
    count = 0
    for line in lines:
        if re.search(r"(?i)\bconvergence has been achieved in\b", line):
            count += 1
    return count


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


def parse_cell_parameters_from_input(lines: List[str]) -> List[CellBlock]:
    return parse_cell_parameters(lines)


def parse_cell_parameters(lines: List[str]) -> List[CellBlock]:
    blocks: List[CellBlock] = []
    for i, line in enumerate(lines):
        if "CELL_PARAMETERS" in line:
            m = re.search(r"CELL_PARAMETERS\s*\(([^)]+)\)", line)
            units = m.group(1).strip() if m else "alat"
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


def convert_positions(
    block: PositionsBlock,
    alat_bohr: Optional[float],
) -> Tuple[str, List[List[float]]]:
    units = block.units.lower()
    if "crystal" in units:
        return "Direct", block.positions

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

    converted = [[v * scale for v in row] for row in block.positions]
    return "Cartesian", converted


def build_species_counts(symbols: List[str]) -> Tuple[List[str], List[int]]:
    order: List[str] = []
    counts: List[int] = []
    for sym in symbols:
        if sym not in order:
            order.append(sym)
            counts.append(1)
        else:
            idx = order.index(sym)
            counts[idx] += 1
    return order, counts


def reorder_positions(order: List[str], symbols: List[str], positions: List[List[float]]) -> List[List[float]]:
    grouped: List[List[float]] = []
    for sym in order:
        for idx, atom_sym in enumerate(symbols):
            if atom_sym == sym:
                grouped.append(positions[idx])
    return grouped


def write_poscar(
    path: str,
    cell_ang: List[List[float]],
    symbols: List[str],
    positions: List[List[float]],
    coord_type: str,
) -> None:
    order, counts = build_species_counts(symbols)
    ordered_positions = reorder_positions(order, symbols, positions)

    with open(path, "w") as f:
        f.write("Generated from pw.out\n")
        f.write("1.0\n")
        for vec in cell_ang:
            f.write(f"{vec[0]:.16f} {vec[1]:.16f} {vec[2]:.16f}\n")
        f.write(" ".join(order) + "\n")
        f.write(" ".join(str(c) for c in counts) + "\n")
        f.write(f"{coord_type}\n")
        for pos in ordered_positions:
            f.write(f"{pos[0]:.16f} {pos[1]:.16f} {pos[2]:.16f}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert last QE pw.out geometry to POSCAR."
    )
    parser.add_argument("pw_out", help="Path to pw.out")
    parser.add_argument("--in", dest="pw_in", default=None, help="Optional pw.in to read CELL_PARAMETERS.")
    parser.add_argument("-o", "--output", default="POSCAR", help="Output POSCAR path")
    args = parser.parse_args()

    lines = read_lines(args.pw_out)
    pw_in_lines: List[str] = []
    if args.pw_in:
        if not os.path.isfile(args.pw_in):
            print(f"pw.in not found: {args.pw_in}", file=sys.stderr)
            return 1
        pw_in_lines = read_lines(args.pw_in)
    calc = guess_calculation(lines)
    steps = count_relax_steps(lines)
    positions_blocks = parse_atomic_positions(lines)

    if steps == 0:
        print("No relaxation steps completed (step count = 0).", file=sys.stderr)
        return 1

    if not positions_blocks:
        print("No ATOMIC_POSITIONS blocks found in pw.out.", file=sys.stderr)
        return 1

    if (calc or "").lower() in ("relax", "vc-relax") and len(positions_blocks) <= 1:
        print("No updated geometry found (only one positions block).", file=sys.stderr)
        return 1

    last_positions = positions_blocks[-1]
    cell_blocks = parse_cell_parameters(lines)
    alat_bohr = parse_alat_bohr(lines)
    if alat_bohr is None and pw_in_lines:
        alat_bohr = parse_alat_bohr_from_input(pw_in_lines)

    if cell_blocks:
        cell_ang = convert_cell_to_angstrom(cell_blocks[-1], alat_bohr)
    else:
        input_cell_blocks = parse_cell_parameters_from_input(pw_in_lines) if pw_in_lines else []
        if input_cell_blocks:
            cell_ang = convert_cell_to_angstrom(input_cell_blocks[-1], alat_bohr)
        else:
            axes = parse_crystal_axes(lines)
            if axes is None:
                print("No CELL_PARAMETERS or crystal axes found for lattice vectors.", file=sys.stderr)
                return 1
            if alat_bohr is None:
                print("Missing lattice parameter (alat) for crystal axes conversion.", file=sys.stderr)
                return 1
            cell_ang = [[v * alat_bohr * BOHR_TO_ANG for v in row] for row in axes]

    coord_type, positions = convert_positions(last_positions, alat_bohr)

    write_poscar(args.output, cell_ang, last_positions.symbols, positions, coord_type)
    print(f"Wrote {args.output} ({coord_type} coordinates).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
