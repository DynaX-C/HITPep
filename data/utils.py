import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import MDAnalysis as mda

from .fix_structure import fix_pdb


def _ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def split_if_multimodel(
    pdb_path: str,
    out_dir: str,
    prefix: str = "decoy",
    overwrite: bool = False,
) -> List[str]:
    """
    Split multi-model PDB into single-model PDB files with caching.

    If single model, return original path as a list.
    If multi model and cached files exist with matching count, reuse them unless overwrite=True.
    """
    pdb_path = str(Path(pdb_path).resolve())
    out_dir = _ensure_dir(out_dir)

    u = mda.Universe(pdb_path)
    n_frames = len(u.trajectory)

    if n_frames <= 1:
        return [pdb_path]

    expected_files = [out_dir / f"{prefix}_{i+1}.pdb" for i in range(n_frames)]

    if (not overwrite) and all(p.exists() for p in expected_files):
        return [str(p.resolve()) for p in expected_files]

    for p in out_dir.glob(f"{prefix}_*.pdb"):
        p.unlink()

    pdb_list = []
    for i, _ in enumerate(u.trajectory):
        u.trajectory[i]
        out_path = out_dir / f"{prefix}_{i+1}.pdb"
        u.atoms.write(str(out_path))
        pdb_list.append(str(out_path.resolve()))

    return pdb_list


def _fix_single_pdb(input_pdb: str, output_pdb: str) -> str:
    """
    Cached fixer: if output exists, reuse it.
    """
    output_pdb = str(Path(output_pdb).resolve())
    if Path(output_pdb).exists():
        return output_pdb

    return fix_pdb(
        input_pdb=input_pdb,
        output_pdb=output_pdb,
        remove_hydrogens=True,
        keep_water=False,
        add_missing_residues=True,
        add_missing_atoms=True,
        replace_nonstandard=True,
        reorder=True,
    )


def _fix_decoy_batch(decoy_list: List[str], out_dir: str) -> List[str]:
    out_dir = _ensure_dir(out_dir)
    decoy_fixed_list = []

    for decoy_pdb in decoy_list:
        out_path = out_dir / f"{Path(decoy_pdb).stem}_fixed.pdb"
        decoy_fixed = _fix_single_pdb(decoy_pdb, str(out_path))
        decoy_fixed_list.append(decoy_fixed)

    return decoy_fixed_list


def resolve_peptide_path(system: Dict, decoy_id: int) -> str:
    """
    Convention
    ----------
    DECOY_ID = 0 -> native_fixed
    DECOY_ID = k -> decoy_list[k-1]
    """
    if decoy_id == 0:
        if system["native_fixed"] is None:
            raise ValueError(f"[{system['target']}] DECOY_ID=0 but native_fixed is missing.")
        return system["native_fixed"]

    idx = decoy_id - 1
    if idx < 0 or idx >= len(system["decoy_list"]):
        raise IndexError(
            f"[{system['target']}] decoy_id={decoy_id} out of range, "
            f"available decoys={len(system['decoy_list'])}"
        )
    return system["decoy_list"][idx]


def systems_to_dict(systems: List[Dict]) -> Dict[str, Dict]:
    return {s["target"]: s for s in systems}


def save_systems(systems: List[Dict], save_path: str):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(systems, save_path)


def load_systems(path: str) -> List[Dict]:
    return torch.load(path, map_location="cpu", weights_only=False)


def save_tau(tau_dict: Dict, save_path: str):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(tau_dict, f, indent=2)


def load_tau(path: Optional[str]) -> Optional[Dict]:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_precomputed_plm(system: Dict) -> torch.Tensor:
    """
    Expected files in each target directory
    --------------------------------------
    protein.pt : [N_protein_res, D]
    peptide.pt : [N_peptide_res, D]
    """
    system_dir = Path(system["workdir"])
    protein_pt = system_dir / "protein.pt"
    peptide_pt = system_dir / "peptide.pt"

    if not protein_pt.exists():
        raise FileNotFoundError(f"Missing protein.pt: {protein_pt}")
    if not peptide_pt.exists():
        raise FileNotFoundError(f"Missing peptide.pt: {peptide_pt}")

    protein_plm = torch.load(protein_pt, map_location="cpu", weights_only=False)
    peptide_plm = torch.load(peptide_pt, map_location="cpu", weights_only=False)

    pocket_seq_indices = torch.tensor(system["pocket_seq_indices"], dtype=torch.long)
    protein_pocket_plm = protein_plm[pocket_seq_indices]

    return torch.cat([protein_pocket_plm, peptide_plm], dim=0)
