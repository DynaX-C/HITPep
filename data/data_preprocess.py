import argparse
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from tqdm import tqdm

from .pocket import extract_fixed_pocket_by_mode, extract_fixed_pocket
from .utils import (
    _ensure_dir,
    _fix_single_pdb,
    _fix_decoy_batch,
    split_if_multimodel,
    save_systems,
)
import warnings
from rdkit import RDLogger
warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')


def prepare_single_system(
    target: str,
    work_root: str,
    protein_pdb: str,
    decoy_pdb: str,
    native_pdb: str,
    pocket_cutoff: float,
) -> Dict:

    workdir = _ensure_dir(Path(work_root) / target)

    receptor_raw = str(workdir / protein_pdb)
    decoy_raw = str(workdir / decoy_pdb)
    native_raw = str(workdir / native_pdb)


    decoy_split_list = split_if_multimodel(
        pdb_path=decoy_raw,
        out_dir=str(workdir / "decoys_split"),
        prefix="decoy",
        overwrite=False,
    )
    if len(decoy_split_list) == 0:
        raise ValueError(f"[{target}] No decoys found after splitting.")

    receptor_fixed = _fix_single_pdb(
        receptor_raw,
        str(workdir / "receptor_fixed.pdb"),
    )


    native_fixed = None
    if native_raw is not None:
        native_fixed = _fix_single_pdb(
            native_raw,
            str(workdir / "native_fixed.pdb"),
        )

    decoy_fixed_list = _fix_decoy_batch(
        decoy_split_list,
        out_dir=str(workdir / "decoys_fixed"),
    )

    pocket_pdb_list = []
    pocket_sequence_list = []
    pocket_seq_indices_list = []

    native_pocket_pdb = str(workdir / f"pocket_{pocket_cutoff}A.pdb")
    native_pocket_info = extract_fixed_pocket(
        protein=receptor_fixed,
        peptide=native_fixed,
        cutoff=pocket_cutoff,
        save_path=native_pocket_pdb,
        )
    pocket_pdb_list.append(native_pocket_info["pocket_pdb"])
    pocket_sequence_list.append(native_pocket_info["pocket_sequence"])
    pocket_seq_indices_list.append(native_pocket_info["pocket_seq_indices"])

    for i, decoy in enumerate(decoy_fixed_list):
        decoy_pocket_pdb = str(workdir / f"pocket_{pocket_cutoff}A_decoy_{i+1}.pdb")
        decoy_pocket_info = extract_fixed_pocket(
            protein=receptor_fixed,
            peptide=decoy,
            save_path=decoy_pocket_pdb,
            cutoff=pocket_cutoff,
        )
        pocket_pdb_list.append(decoy_pocket_info["pocket_pdb"])
        pocket_sequence_list.append(decoy_pocket_info["pocket_sequence"])
        pocket_seq_indices_list.append(decoy_pocket_info["pocket_seq_indices"])


    system = {
        "target": target,
        "workdir": str(workdir),

        "receptor_fixed": receptor_fixed,
        "native_fixed": native_fixed,
        "decoy_list": decoy_fixed_list,
        "pocket_pdb": pocket_pdb_list,

        "protein_sequence": native_pocket_info["protein_sequence"],
        "pocket_sequence": pocket_sequence_list,
        "protein_resindices": native_pocket_info["protein_resindices"].tolist(),
        "pocket_seq_indices": pocket_seq_indices_list,

        "peptide_sequence": native_pocket_info["peptide_sequence"],
        "peptide_resindices": native_pocket_info["peptide_resindices"].tolist(),
    }

    return system


def _worker(args):
    return prepare_single_system(*args)

def build_systems_from_csv(
    csv_path: str,
    work_root: str,
    protein_pdb: str,
    decoy_pdb: str,
    native_pdb: str,
    pocket_cutoff: float,
    num_workers: int,
) -> List[Dict]:

    df = pd.read_csv(csv_path)
    targets = sorted(df["TARGET"].unique().tolist())

    worker_args = [
        (
            target,
            work_root,
            protein_pdb,
            decoy_pdb,
            native_pdb,
            pocket_cutoff,
        )
        for target in targets
    ]

    systems = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(_worker, worker_args)

        for system in tqdm(results, total=len(worker_args), desc="Preparing systems"):
            systems.append(system)

    return systems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)

    parser.add_argument("--protein_pdb", type=str, default="receptor.pdb")
    parser.add_argument("--decoy_pdb", type=str, default="decoy.pdb")
    parser.add_argument("--native_pdb", type=str, default="peptide.pdb")

    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    systems = build_systems_from_csv(
        csv_path=args.csv_path,
        work_root=args.work_dir,
        protein_pdb=args.protein_pdb,
        decoy_pdb=args.decoy_pdb,
        native_pdb=args.native_pdb,
        pocket_cutoff=args.cutoff,
        num_workers=args.num_workers,
    )

    save_systems(systems, args.save_path)

    print(f"\n[Done] systems saved to: {args.save_path}")
    print(f"Total systems: {len(systems)}")


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    main()
