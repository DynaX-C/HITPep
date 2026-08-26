import json
import argparse
from pathlib import Path
from typing import Optional, Union, List, Dict

import numpy as np
import torch
import MDAnalysis as mda
from MDAnalysis import Merge

from data.graph_geometry import build_geometry_graphs
from data.utils import load_precomputed_plm, split_if_multimodel, _fix_single_pdb, _ensure_dir
from data.pocket import _resname_to_aa, _extract_sequence_and_mapping, _first_frame_atoms, _to_universe, extract_fixed_pocket
from data.plm_features import ESM2Extractor, save_plm_for_system

import warnings
warnings.filterwarnings("ignore")

# =========================================================
# basic helpers
# =========================================================
def _as_list(x):
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def _resolve_path(p: str, base_dir: Optional[Union[str, Path]] = None) -> str:
    """
    Robust path resolver.

    Priority:
    1. use p as-is if it exists
    2. if p is absolute, return it directly
    3. if base_dir is provided, try base_dir / p
    """
    p = Path(p)

    if p.exists():
        return str(p.resolve())

    if p.is_absolute():
        return str(p)

    if base_dir is not None:
        candidate = Path(base_dir) / p
        if candidate.exists():
            return str(candidate.resolve())
        return str(candidate)

    return str(p)


def infer_output_dir(
    complex_pdb: Optional[List[str]] = None,
    protein_pdb: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> str:
    """
    Determine output directory automatically.
    """
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        return str(out_path.resolve())

    ref = None
    if protein_pdb is not None:
        ref = protein_pdb
    elif complex_pdb is not None and len(complex_pdb) > 0:
        ref = complex_pdb[0]

    if ref is None:
        raise ValueError("Cannot infer output directory: no protein_pdb or complex_pdb provided.")

    ref_path = Path(_resolve_path(ref)).resolve()
    out_path = ref_path.parent / "inference"
    out_path.mkdir(parents=True, exist_ok=True)
    return str(out_path)


# =========================================================
# split complex
# =========================================================
def split_complex_to_protein_peptide(
    complex_pdb: str,
    protein_out: str,
    peptide_out: str,
    protein_chain: str = "A",
    peptide_chain: str = "B",
):
    """
    Split complex pdb into protein and peptide.
    Assumption: protein_chain='A', peptide_chain='B'
    """
    u = mda.Universe(complex_pdb)

    prot_atoms = u.select_atoms(f"chainID {protein_chain}")
    pep_atoms = u.select_atoms(f"chainID {peptide_chain}")

    if len(prot_atoms) == 0:
        raise ValueError(f"No atoms found in protein chain {protein_chain}: {complex_pdb}")
    if len(pep_atoms) == 0:
        raise ValueError(f"No atoms found in peptide chain {peptide_chain}: {complex_pdb}")

    Path(protein_out).parent.mkdir(parents=True, exist_ok=True)
    Path(peptide_out).parent.mkdir(parents=True, exist_ok=True)

    prot_atoms.write(protein_out)
    pep_atoms.write(peptide_out)

    return protein_out, peptide_out


# =========================================================
# pocket by reference resid
# =========================================================
def extract_fixed_pocket_by_resid(
    protein,
    peptide,
    pocket_resids,
    save_path: Optional[str] = None,
    protein_chain: Optional[str] = None,
):
    """
    Extract pocket from protein by PDB residue ids (resids).
    """
    protein_u = _to_universe(protein)
    peptide_u = _to_universe(peptide)

    pocket_resids = np.asarray(pocket_resids, dtype=np.int64)
    if len(pocket_resids) == 0:
        raise ValueError("pocket_resids is empty.")

    if protein_chain is None:
        sel = " or ".join([f"resid {int(r)}" for r in pocket_resids])
    else:
        sel = " or ".join(
            [f"(chainID {protein_chain} and resid {int(r)})" for r in pocket_resids]
        )

    pocket_atoms = protein_u.select_atoms(sel)
    if len(pocket_atoms) == 0:
        raise ValueError("No pocket atoms found by given pocket_resids.")

    pocket_atom_indices = pocket_atoms.indices.astype(np.int64)
    pocket_atom_mask = np.zeros(len(protein_u.atoms), dtype=bool)
    pocket_atom_mask[pocket_atom_indices] = True

    full_sequence, full_resindices, resindex_to_seqidx = _extract_sequence_and_mapping(protein_u)

    pocket_resindices = pocket_atoms.residues.resindices.astype(np.int64)
    pocket_seq_indices = np.array(
        [resindex_to_seqidx[int(r)] for r in pocket_resindices],
        dtype=np.int64
    )
    pocket_sequence = "".join(full_sequence[i] for i in pocket_seq_indices)

    pep_atoms = _first_frame_atoms(peptide_u)
    pep_residues = pep_atoms.residues
    peptide_sequence = "".join(_resname_to_aa(res.resname) for res in pep_residues)
    peptide_resindices = pep_residues.resindices.astype(np.int64)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        pocket_atoms.write(save_path)

    return {
        "pocket_pdb": save_path,
        "pocket_atoms": pocket_atoms,
        "pocket_resids": pocket_atoms.residues.resids.astype(np.int64),
        "pocket_resindices": pocket_resindices,
        "pocket_atom_indices": pocket_atom_indices,
        "pocket_atom_mask": pocket_atom_mask,

        "protein_sequence": full_sequence,
        "pocket_sequence": pocket_sequence,
        "protein_resindices": full_resindices,
        "pocket_seq_indices": pocket_seq_indices,

        "peptide_sequence": peptide_sequence,
        "peptide_resindices": peptide_resindices,
    }


def get_reference_pocket_resids(
    protein_pdb: str,
    peptide_pdb: str,
    cutoff: float,
    save_path: Optional[str] = None,
):
    """
    Use one protein-peptide pair to define pocket, then convert pocket_resindices
    to stable PDB resids.
    """
    pocket_info = extract_fixed_pocket(
        protein=protein_pdb,
        peptide=peptide_pdb,
        cutoff=cutoff,
        save_path=save_path,
    )

    protein_u = _to_universe(protein_pdb)
    pocket_resindices = np.asarray(pocket_info["pocket_resindices"], dtype=np.int64)
    pocket_resids = protein_u.residues[pocket_resindices].resids.astype(np.int64)

    pocket_info["pocket_resids"] = pocket_resids
    return pocket_info


# =========================================================
# fixed protein preparation
# =========================================================
def prepare_fixed_protein_system(
    out_dir: str,
    pocket_cutoff: float,
    native_protein_pdb: Optional[str] = None,
    native_peptide_pdb: Optional[str] = None,
    decoy_protein_pdb: Optional[str] = None,
    decoy_peptide_pdb: Optional[str] = None,
    use_native_pocket: bool = False,
):
    """
    Fixed protein mode:
    - extract pocket only once
    - native name -> native
    - decoy names -> decoy_i
    """
    out_dir = str(_ensure_dir(Path(out_dir)))

    native_protein_raw = _resolve_path(native_protein_pdb) if native_protein_pdb is not None else None
    native_peptide_raw = _resolve_path(native_peptide_pdb) if native_peptide_pdb is not None else None
    decoy_protein_raw = _resolve_path(decoy_protein_pdb) if decoy_protein_pdb is not None else None
    decoy_peptide_raw = _resolve_path(decoy_peptide_pdb) if decoy_peptide_pdb is not None else None

    if native_protein_raw is None and decoy_protein_pdb is None:
        raise ValueError("Need native_pdb or decoy_pdb in fixed protein mode.")
    if native_peptide_raw is None and decoy_peptide_raw is None:
        raise ValueError("Need native_pdb or decoy_pdb in fixed protein mode.")

    native_receptor_fixed = None
    if native_protein_raw is not None:
        native_receptor_fixed = _fix_single_pdb(
            native_protein_raw,
            str(Path(out_dir) / "native_receptor_fixed.pdb"),
        )

    native_peptide_fixed = None
    if native_peptide_raw is not None:
        native_peptide_fixed = _fix_single_pdb(
            native_peptide_raw,
            str(Path(out_dir) / "native_peptide_fixed.pdb"),
        )

    decoy_protein_fixed = None
    if decoy_protein_raw is not None:
        decoy_protein_fixed = _fix_single_pdb(
            decoy_protein_raw,
            str(Path(out_dir) / "decoy_receptor_fixed.pdb"),
        )

    decoy_fixed_list = []
    if decoy_peptide_raw is not None:
        decoy_split_list = split_if_multimodel(
            pdb_path=decoy_peptide_raw,
            out_dir=str(Path(out_dir) / "decoys_split"),
            prefix="decoy",
            overwrite=False,
        )
        if len(decoy_split_list) == 0:
            raise ValueError("No decoys found after splitting.")

        decoys_fixed_dir = _ensure_dir(Path(out_dir) / "decoys_fixed")
        for p in decoy_split_list:
            dst = str(Path(decoys_fixed_dir) / Path(p).name)
            fixed_p = _fix_single_pdb(p, dst)
            decoy_fixed_list.append(fixed_p)

    pocket_pdb = str(Path(out_dir) / f"pocket_{pocket_cutoff}A.pdb")

    decoy_pocket_pdb_list = []
    decoy_pocket_sequence_list = []
    decoy_pocket_seq_indices_list = []
    if use_native_pocket:
        pocket_pdb = str(Path(out_dir) / f"pocket_{pocket_cutoff}A.pdb")
        pocket_info = get_reference_pocket_resids(
            protein=native_receptor_fixed,
            peptide=native_peptide_fixed,
            cutoff=pocket_cutoff,
            save_path=pocket_pdb,
        )
        pocket_reference = "native"
        decoy_pocket_pdb_list.append(pocket_info["pocket_pdb"])
        decoy_pocket_sequence_list.append(pocket_info["pocket_sequence"])
        decoy_pocket_seq_indices_list.append(pocket_info["pocket_seq_indices"])
    else:
        if decoy_protein_fixed is not None:
            for i, decoy_peptide in enumerate(decoy_fixed_list):
                pocket_pdb = str(Path(out_dir) / f"pocket_{pocket_cutoff}A_decoy_{i+1}.pdb")
                pocket_info = extract_fixed_pocket(
                    protein=decoy_protein_fixed,
                    peptide=decoy_peptide,
                    cutoff=pocket_cutoff,
                    save_path=pocket_pdb,
                )
                pocket_reference = "decoy"
                decoy_pocket_pdb_list.append(pocket_info["pocket_pdb"])
                decoy_pocket_sequence_list.append(pocket_info["pocket_sequence"])
                decoy_pocket_seq_indices_list.append(pocket_info["pocket_seq_indices"])
        else:
            for i, decoy_peptide in enumerate(decoy_fixed_list):
                pocket_pdb = str(Path(out_dir) / f"pocket_{pocket_cutoff}A_decoy_{i+1}.pdb")
                pocket_info = extract_fixed_pocket(
                    protein=native_receptor_fixed,
                    peptide=decoy_peptide,
                    cutoff=pocket_cutoff,
                    save_path=pocket_pdb,
                )
                pocket_reference = "decoy"
                decoy_pocket_pdb_list.append(pocket_info["pocket_pdb"])
                decoy_pocket_sequence_list.append(pocket_info["pocket_sequence"])
                decoy_pocket_seq_indices_list.append(pocket_info["pocket_seq_indices"])

    # if native_peptide_fixed is not None and native_receptor_fixed is not None:
    #     pocket_info = extract_fixed_pocket(
    #         protein=native_receptor_fixed,
    #         peptide=native_peptide_fixed,
    #         cutoff=pocket_cutoff,
    #         save_path=pocket_pdb,
    #     )
    #     pocket_reference = "native"
    # elif native_receptor_fixed is not None and native_peptide_fixed is None and len(decoy_fixed_list) > 0:
    #     pocket_info = extract_fixed_pocket(
    #         protein=native_receptor_fixed,
    #         peptide=decoy_fixed_list[0],
    #         cutoff=pocket_cutoff,
    #         save_path=pocket_pdb,
    #     )
    #     pocket_reference = "first_decoy"
    # else:
    #     pocket_info = extract_fixed_pocket(
    #         protein=decoy_protein_fixed,
    #         peptide=decoy_fixed_list[0],
    #         cutoff=pocket_cutoff,
    #         save_path=pocket_pdb,
    #     )
    #     pocket_reference = "first_decoy"

    system = {
        "mode": "eval" if native_peptide_fixed is not None else "infer",
        "input_style": "fixed_protein",
        "workdir": out_dir,

        "receptor_fixed": native_receptor_fixed,
        "native_fixed": native_peptide_fixed,
        "decoy_list": decoy_fixed_list,
        "pocket_pdb": decoy_pocket_pdb_list,

        "protein_sequence": pocket_info["protein_sequence"],
        "pocket_sequence": decoy_pocket_sequence_list,
        "protein_resindices": pocket_info["protein_resindices"].tolist(),
        "pocket_seq_indices": decoy_pocket_seq_indices_list,

        "peptide_sequence": pocket_info["peptide_sequence"],
        "peptide_resindices": pocket_info["peptide_resindices"].tolist(),

        # "pocket_resids": pocket_info["pocket_resids"].tolist() if "pocket_resids" in pocket_info else None,
        # "pocket_resindices": pocket_info["pocket_resindices"].tolist(),

        "pocket_reference": pocket_reference,
    }
    return system


# =========================================================
# complex preparation
# =========================================================
def prepare_complex_system(
    out_dir: str,
    pocket_cutoff: float,
    complex_pdb: Union[str, List[str]],
    protein_pdb: Optional[str] = None,
    native_pdb: Optional[str] = None,
    protein_chain: str = "A",
    peptide_chain: str = "B",
    use_native_pocket: bool = False,
):
    """
    Complex mode:
    - if protein_pdb + native_pdb are provided:
      use them to define reference pocket_resids
    - else:
      use first complex to define reference pocket_resids
    - each complex extracts its own pocket pdb
    """
    out_dir = str(_ensure_dir(Path(out_dir)))
    complex_list = _as_list(complex_pdb)

    if len(complex_list) == 0:
        raise ValueError("complex_pdb is empty in complex mode.")

    split_dir = _ensure_dir(Path(out_dir) / "split_complex")
    fixed_dir = _ensure_dir(Path(out_dir) / "fixed_complex")
    pocket_dir = _ensure_dir(Path(out_dir) / "pockets")

    # ---------- determine reference pocket_resids ----------
    if use_native_pocket:
        if protein_pdb is None or native_pdb is None:
            raise ValueError(
                "--user_native_pocket was set, but native_protein_pdb/native_peptide_pdb is missing."
            )
        ref_protein_raw = _resolve_path(protein_pdb)
        ref_native_raw = _resolve_path(native_pdb)

        ref_protein_fixed = _fix_single_pdb(
            ref_protein_raw,
            str(Path(fixed_dir) / "ref_protein_fixed.pdb"),
        )
        ref_native_fixed = _fix_single_pdb(
            ref_native_raw,
            str(Path(fixed_dir) / "ref_native_fixed.pdb"),
        )

        ref_pocket = get_reference_pocket_resids(
            ref_protein_fixed,
            ref_native_fixed,
            cutoff=pocket_cutoff,
            save_path=str(Path(pocket_dir) / f"reference_pocket_{pocket_cutoff}A.pdb"),
        )
        pocket_resids = ref_pocket["pocket_resids"]
        pocket_reference = "protein_pdb + native_pdb"
        mode = "eval"

    else:
        pocket_reference = "self"
        mode = "infer"

    # ---------- process each complex ----------
    systems = []
    for comp in complex_list:
        comp_abs = _resolve_path(comp)
        name = Path(comp).stem

        protein_raw = str(Path(split_dir) / f"{name}_protein_raw.pdb")
        peptide_raw = str(Path(split_dir) / f"{name}_peptide_raw.pdb")
        protein_fixed = str(Path(fixed_dir) / f"{name}_protein_fixed.pdb")
        peptide_fixed = str(Path(fixed_dir) / f"{name}_peptide_fixed.pdb")
        pocket_pdb = str(Path(pocket_dir) / f"{name}_pocket_{pocket_cutoff}A.pdb")

        split_complex_to_protein_peptide(
            comp_abs,
            protein_raw,
            peptide_raw,
            protein_chain=protein_chain,
            peptide_chain=peptide_chain,
        )

        protein_fixed = _fix_single_pdb(protein_raw, protein_fixed)
        peptide_fixed = _fix_single_pdb(peptide_raw, peptide_fixed)

        protein_u = mda.Universe(protein_fixed)
        peptide_u = mda.Universe(peptide_fixed)
        merged = Merge(protein_u.atoms, peptide_u.atoms)
        merged_dir = _ensure_dir(Path(fixed_dir) / "complex")
        merged_complex = str(Path(merged_dir) /f"{name}.pdb")
        merged.atoms.write(str(merged_complex))

        if use_native_pocket:
            pocket_info = extract_fixed_pocket_by_resid(
                protein=protein_fixed,
                peptide=peptide_fixed,
                pocket_resids=pocket_resids,
                save_path=pocket_pdb,
                protein_chain=protein_chain,
            )
        else:
            pocket_info = extract_fixed_pocket(
                protein=protein_fixed,
                peptide=peptide_fixed,
                cutoff=pocket_cutoff,
                save_path=pocket_pdb,
            )

        systems.append({
            "sample_id": name,
            "mode": mode,
            "input_style": "complex",
            "workdir": out_dir,

            "complex_pdb": comp_abs,
            "receptor_fixed": protein_fixed,
            "peptide_fixed": peptide_fixed,
            "pocket_pdb": pocket_info["pocket_pdb"],

            "protein_sequence": pocket_info["protein_sequence"],
            "pocket_sequence": pocket_info["pocket_sequence"],
            "protein_resindices": pocket_info["protein_resindices"].tolist(),
            "pocket_seq_indices": pocket_info["pocket_seq_indices"].tolist(),

            "peptide_sequence": pocket_info["peptide_sequence"],
            "peptide_resindices": pocket_info["peptide_resindices"].tolist(),

            # "pocket_resids": pocket_info["pocket_resids"].tolist(),
            "pocket_resindices": pocket_info["pocket_resindices"].tolist(),

            "pocket_reference": pocket_reference,
        })

    first_sample = systems[0]

    return {
        "mode": mode,
        "input_style": "complex",
        "workdir": out_dir,

        # 让 complex 模式也能只生成一次 pt
        "protein_sequence": first_sample["protein_sequence"],
        "pocket_sequence": first_sample["pocket_sequence"],
        "protein_resindices": first_sample["protein_resindices"],
        "pocket_seq_indices": first_sample["pocket_seq_indices"],
        "peptide_sequence": first_sample["peptide_sequence"],
        "peptide_resindices": first_sample["peptide_resindices"],

        "pocket_reference": pocket_reference,
        # "pocket_resids": pocket_resids.tolist(),
        "systems": systems,
    }


# =========================================================
# unified preparation
# =========================================================
def prepare_inference_system(
    out_dir: str,
    pocket_cutoff: float,
    complex_pdb: Optional[Union[str, List[str]]] = None,
    native_protein_pdb: Optional[str] = None,
    native_peptide_pdb: Optional[str] = None,
    decoy_protein_pdb: Optional[str] = None,
    decoy_peptide_pdb: Optional[str] = None,
    protein_chain: str = "A",
    peptide_chain: str = "B",
    use_native_pocket: bool = False,
):
    complex_list = _as_list(complex_pdb)

    if len(complex_list) == 0:
        if native_protein_pdb is None and decoy_protein_pdb is None:
            raise ValueError("native_protein_pdb or decoy_protein_pdb is required when complex_pdb is not provided.")
        return prepare_fixed_protein_system(
            out_dir=out_dir,
            pocket_cutoff=pocket_cutoff,
            native_protein_pdb=native_protein_pdb,
            native_peptide_pdb=native_peptide_pdb,
            decoy_protein_pdb=decoy_protein_pdb,
            decoy_peptide_pdb=decoy_peptide_pdb,
            use_native_pocket=use_native_pocket,
        )

    return prepare_complex_system(
        out_dir=out_dir,
        pocket_cutoff=pocket_cutoff,
        complex_pdb=complex_list,
        protein_pdb=native_protein_pdb,
        native_pdb=native_peptide_pdb,
        protein_chain=protein_chain,
        peptide_chain=peptide_chain,
        use_native_pocket=use_native_pocket,
    )


# =========================================================
# build inference graphs (no labels)
# =========================================================
def build_inference_graphs(
    system: Dict,
    cutoff: float = 6.0,
):
    """
    Build graph dataset for inference only.
    Requires PLM features to be precomputed already.
    """
    dataset = []
    input_style = system.get("input_style", "fixed_protein")

    # fixed protein mode
    if input_style == "fixed_protein":
        pocket_reference = system.get("pocket_reference", "")
        if system.get("native_fixed") is not None and len(system.get("decoy_list", [])) == 0:
            esm_res = load_precomputed_plm(system)
            graphs = build_geometry_graphs(
                pocket_pdb=system["pocket_pdb"][0],
                peptide_pdb=system["native_fixed"],
                esm_res=esm_res,
                cutoff=cutoff,
                complex_name="native",
            )
            dataset.append(graphs)
            return dataset

        system_dir = Path(system["workdir"])
        protein_pt = system_dir / "protein.pt"
        peptide_pt = system_dir / "peptide.pt"

        if not protein_pt.exists():
            raise FileNotFoundError(f"Missing protein.pt: {protein_pt}")
        if not peptide_pt.exists():
            raise FileNotFoundError(f"Missing peptide.pt: {peptide_pt}")

        protein_plm = torch.load(protein_pt, map_location="cpu", weights_only=False)
        peptide_plm = torch.load(peptide_pt, map_location="cpu", weights_only=False)

        if pocket_reference == 'decoy':
            for i, peptide_path in enumerate(system["decoy_list"]):
                pocket_seq_indices = torch.tensor(system["pocket_seq_indices"][i], dtype=torch.long)
                protein_pocket_plm = protein_plm[pocket_seq_indices]

                esm_res = torch.cat([protein_pocket_plm, peptide_plm], dim=0)

                graphs = build_geometry_graphs(
                    pocket_pdb=system["pocket_pdb"][i],
                    peptide_pdb=peptide_path,
                    esm_res=esm_res,
                    cutoff=cutoff,
                    complex_name=f"decoy_{i+1}",
                )
                dataset.append(graphs)

            return dataset
        else:
            esm_res = load_precomputed_plm(system)
            for i, peptide_path in enumerate(system["decoy_list"]):
                graphs = build_geometry_graphs(
                    pocket_pdb=system["pocket_pdb"][0],
                    peptide_pdb=peptide_path,
                    esm_res=esm_res,
                    cutoff=cutoff,
                    complex_name=f"decoy_{i+1}",                  
                )
                dataset.append(graphs)

            return dataset

    if input_style == "complex":
        pocket_reference = system.get("pocket_reference", "")

        if pocket_reference == "protein_pdb + native_pdb":
            esm_res = load_precomputed_plm(system["systems"][0])

            for sample in system["systems"]:
                graphs = build_geometry_graphs(
                    pocket_pdb=sample["pocket_pdb"],
                    peptide_pdb=sample["peptide_fixed"],
                    esm_res=esm_res,
                    cutoff=cutoff,
                    complex_name=sample["sample_id"],
                )
                dataset.append(graphs)

        else:
            for sample in system["systems"]:
                esm_res = load_precomputed_plm(sample)
                graphs = build_geometry_graphs(
                    pocket_pdb=sample["pocket_pdb"],
                    peptide_pdb=sample["peptide_fixed"],
                    esm_res=esm_res,
                    cutoff=cutoff,
                    complex_name=sample["sample_id"],
                )
                dataset.append(graphs)

        return dataset

    raise ValueError(f"Unsupported input_style: {input_style}")


# =========================================================
# main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Unified inference preparation + graph building")

    parser.add_argument("--complex_pdb", nargs="*", default=None)
    parser.add_argument("--native_protein_pdb", type=str, default=None)
    parser.add_argument("--native_peptide_pdb", type=str, default=None)
    parser.add_argument("--decoy_protein_pdb", type=str, default=None)
    parser.add_argument("--decoy_peptide_pdb", type=str, default=None)

    parser.add_argument("--use_native_pocket", action="store_true")

    parser.add_argument("--out_dir", type=str, default=None)

    parser.add_argument("--pocket_cutoff", type=float, default=6.0)
    parser.add_argument("--graph_cutoff", type=float, default=6.0)
    parser.add_argument("--protein_chain", type=str, default="A")
    parser.add_argument("--peptide_chain", type=str, default="B")

    parser.add_argument("--save_system_json", type=str, default=None)
    parser.add_argument("--save_graph_pt", type=str, default=None)
    parser.add_argument("--prepare_only", action="store_true")

    parser.add_argument("--esm_model", type=str, default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    out_dir = infer_output_dir(
        complex_pdb=args.complex_pdb,
        protein_pdb=args.native_protein_pdb,
        out_dir=args.out_dir,
    )

    system = prepare_inference_system(
        out_dir=out_dir,
        pocket_cutoff=args.pocket_cutoff,
        complex_pdb=args.complex_pdb,
        native_protein_pdb=args.native_protein_pdb,
        native_peptide_pdb=args.native_peptide_pdb,
        decoy_protein_pdb=args.decoy_protein_pdb,
        decoy_peptide_pdb=args.decoy_peptide_pdb,
        protein_chain=args.protein_chain,
        peptide_chain=args.peptide_chain,
        use_native_pocket=args.use_native_pocket,
    )

    # print(json.dumps(system, indent=2, ensure_ascii=False))

    if args.save_system_json is not None:
        Path(args.save_system_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_system_json, "w", encoding="utf-8") as f:
            json.dump(system, f, indent=2, ensure_ascii=False)

    if args.prepare_only:
        return

    extractor = ESM2Extractor(model_name=args.esm_model, device=args.device)
    save_plm_for_system(system, extractor, overwrite=False)

    dataset = build_inference_graphs(
        system=system,
        cutoff=args.graph_cutoff,
    )

    print(f"[Built] inference graphs: N={len(dataset)}")

    if args.save_graph_pt is not None:
        Path(args.save_graph_pt).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, args.save_graph_pt)
        print(f"[Saved] graph dataset -> {args.save_graph_pt}")


if __name__ == "__main__":
    main()