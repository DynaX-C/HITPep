import argparse
from typing import Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path

from .graph_geometry import build_geometry_graphs
from .utils import (
    load_systems,
    load_tau,
    resolve_peptide_path,
    systems_to_dict,
    load_precomputed_plm,
)


# =========================
# Worker: supervised
# =========================
def _worker_supervised(args):
    system, rows, tau_dict, cutoff = args

    system_dir = Path(system["workdir"])
    protein_pt = system_dir / "protein.pt"
    peptide_pt = system_dir / "peptide.pt"

    if not protein_pt.exists():
        raise FileNotFoundError(f"Missing protein.pt: {protein_pt}")
    if not peptide_pt.exists():
        raise FileNotFoundError(f"Missing peptide.pt: {peptide_pt}")

    protein_plm = torch.load(protein_pt, map_location="cpu")
    peptide_plm = torch.load(peptide_pt, map_location="cpu")


    # esm_res = load_precomputed_plm(system)
    results = []

    for row in rows:
        decoy_id = int(row["DECOY_ID"])
        peptide_path = resolve_peptide_path(system, decoy_id)

        pocket_seq_indices = torch.tensor(system["pocket_seq_indices"][decoy_id], dtype=torch.long)
        protein_pocket_plm = protein_plm[pocket_seq_indices]
        esm_res = torch.cat([protein_pocket_plm, peptide_plm], dim=0)

        if decoy_id == 0:
            complex_name = f"{system['target']}_native"
        else:
            complex_name = f"{system['target']}_decoy_{decoy_id}"

        kwargs = dict(
            protein_pdb=system["receptor_fixed"],
            pocket_pdb=system["pocket_pdb"][decoy_id],
            peptide_pdb=peptide_path,
            esm_res=esm_res,
            cutoff=cutoff,
            complex_name=complex_name,
        )

        if tau_dict is not None and system["native_fixed"] is not None:
            kwargs.update(
                native_peptide_pdb=system["native_fixed"],
                tau_atom=tau_dict["tau_atom"],
                tau_res=tau_dict["tau_res"],
                tau_global=tau_dict["tau_global"],
                contact_threshold=5.0,
            )

        graphs = build_geometry_graphs(**kwargs)
        results.append(graphs)

    return results


# =========================
# Supervised dataset
# =========================
def build_supervised_dataset(
    systems: List[Dict],
    csv_path: str,
    save_path: str,
    tau_dict: Optional[Dict] = None,
    cutoff: float = 6.0,
    num_workers: int = 4,
):
    df = pd.read_csv(csv_path)
    system_map = systems_to_dict(systems)

    grouped_args = []
    for target, group in df.groupby("TARGET", sort=False):
        system = system_map[target]
        rows = group.to_dict("records")
        grouped_args.append((system, rows, tau_dict, cutoff))

    dataset = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(_worker_supervised, grouped_args)

        for res in tqdm(results, total=len(grouped_args), desc="Building supervised dataset"):
            dataset.extend(res)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, save_path)

    print(f"[Saved] dataset -> {save_path} (N={len(dataset)})")


# =========================
# main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Build graph dataset from systems manifest")
    parser.add_argument("--systems_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--tau_path", type=str, default=None)
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    systems = load_systems(args.systems_path)
    tau_dict = load_tau(args.tau_path)

    build_supervised_dataset(
        systems=systems,
        csv_path=args.csv_path,
        save_path=args.save_path,
        tau_dict=tau_dict,
        cutoff=args.cutoff,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
