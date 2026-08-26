#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd


def is_atom_line(line):
    return line.startswith(("ATOM", "HETATM"))


def atom_name(line):
    return line[12:16].strip()


def is_hydrogen(line):
    name = atom_name(line).upper()
    return (
        name.startswith("H")
        or (len(name) >= 2 and name[0].isdigit() and name[1] == "H")
    )


def residue_key(line):
    """
    用 PDB 中的 chain + resSeq + iCode 判断是否是同一个残基。
    只是用于识别残基顺序，不要求和 CSV 里的编号一致。
    """
    chain = line[21].strip()
    resseq = line[22:26].strip()
    icode = line[26].strip()
    return chain, resseq, icode


def set_bfactor(line, value):
    """
    把 score 写入 PDB B-factor 列，保留两位小数。
    """
    line = line.rstrip("\n")

    if len(line) < 66:
        line = line.ljust(66)

    value = float(value)

    if value > 999.99:
        value = 999.99
    if value < -99.99:
        value = -99.99

    return line[:60] + f"{value:6.2f}" + line[66:] + "\n"


def project_atom_score(
    pdb_file,
    atom_csv,
    name,
    score_col,
    out_pdb,
    ignore_h=False,
):
    df = pd.read_csv(atom_csv)

    df = df[df["Name"].astype(str) == str(name)].copy()

    if df.empty:
        raise ValueError(f"No atom scores found for Name = {name}")

    if score_col not in df.columns:
        raise KeyError(f"{score_col} not found in atom csv.")

    scores = df[score_col].astype(float).tolist()

    with open(pdb_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    atom_indices = []

    for i, line in enumerate(lines):
        if not is_atom_line(line):
            continue

        if ignore_h and is_hydrogen(line):
            continue

        atom_indices.append(i)

    if len(atom_indices) != len(scores):
        raise ValueError(
            f"Atom number mismatch:\n"
            f"  PDB atoms = {len(atom_indices)}\n"
            f"  Scores    = {len(scores)}\n"
            f"If PDB has hydrogens but graph does not, use --ignore_h."
        )

    for idx, score in zip(atom_indices, scores):
        lines[idx] = set_bfactor(lines[idx], score)

    out_pdb = Path(out_pdb)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)

    with open(out_pdb, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"[Saved] atom score PDB -> {out_pdb}")


def project_residue_score(
    pdb_file,
    res_csv,
    name,
    score_col,
    out_pdb,
    ignore_h=False,
):
    df = pd.read_csv(res_csv)

    df = df[df["Name"].astype(str) == str(name)].copy()

    if df.empty:
        raise ValueError(f"No residue scores found for Name = {name}")

    if score_col not in df.columns:
        raise KeyError(f"{score_col} not found in residue csv.")

    scores = df[score_col].astype(float).tolist()

    with open(pdb_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # 按 PDB 中残基出现顺序收集残基
    residue_order = []
    seen = set()

    for line in lines:
        if not is_atom_line(line):
            continue

        if ignore_h and is_hydrogen(line):
            continue

        key = residue_key(line)

        if key not in seen:
            seen.add(key)
            residue_order.append(key)

    if len(residue_order) != len(scores):
        raise ValueError(
            f"Residue number mismatch:\n"
            f"  PDB residues = {len(residue_order)}\n"
            f"  Scores       = {len(scores)}\n"
            f"Please check whether this peptide PDB matches the selected Name."
        )

    score_by_residue = {
        key: score for key, score in zip(residue_order, scores)
    }

    for i, line in enumerate(lines):
        if not is_atom_line(line):
            continue

        if ignore_h and is_hydrogen(line):
            continue

        key = residue_key(line)

        if key in score_by_residue:
            lines[i] = set_bfactor(line, score_by_residue[key])

    out_pdb = Path(out_pdb)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)

    with open(out_pdb, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"[Saved] residue score PDB -> {out_pdb}")


def main():
    parser = argparse.ArgumentParser(
        description="Project atom/residue HITPep scores to PDB B-factor column by order."
    )

    parser.add_argument("--pdb", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)

    parser.add_argument("--atom_csv", type=str, default=None)
    parser.add_argument("--res_csv", type=str, default=None)

    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["atom", "residue", "both"],
    )

    parser.add_argument(
        "--atom_score_col",
        type=str,
        default="Atom_Score",
    )

    parser.add_argument(
        "--res_score_col",
        type=str,
        default="Res_Geom_Score",
        help="For example: Res_Geom_Score or Res_Int_Score",
    )

    parser.add_argument(
        "--out_prefix",
        type=str,
        default="projected_score",
    )

    parser.add_argument(
        "--ignore_h",
        action="store_true",
        help="Ignore hydrogen atoms in PDB when matching scores.",
    )

    args = parser.parse_args()

    if args.mode in ["atom", "both"]:
        if args.atom_csv is None:
            raise ValueError("--atom_csv is required for atom mode.")

        project_atom_score(
            pdb_file=args.pdb,
            atom_csv=args.atom_csv,
            name=args.name,
            score_col=args.atom_score_col,
            out_pdb=f"{args.out_prefix}_atom_{args.atom_score_col}.pdb",
            ignore_h=args.ignore_h,
        )

    if args.mode in ["residue", "both"]:
        if args.res_csv is None:
            raise ValueError("--res_csv is required for residue mode.")

        project_residue_score(
            pdb_file=args.pdb,
            res_csv=args.res_csv,
            name=args.name,
            score_col=args.res_score_col,
            out_pdb=f"{args.out_prefix}_residue_{args.res_score_col}.pdb",
            ignore_h=args.ignore_h,
        )


if __name__ == "__main__":
    main()