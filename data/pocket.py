from typing import Literal, Optional, Union
import numpy as np
import MDAnalysis as mda

try:
    from Bio.Data.IUPACData import protein_letters_3to1
except ImportError:
    protein_letters_3to1 = {}


def _to_universe(x: Union[str, mda.Universe]) -> mda.Universe:
    """Convert input to MDAnalysis Universe."""
    if isinstance(x, mda.Universe):
        return x
    elif isinstance(x, str):
        return mda.Universe(x)
    else:
        raise TypeError(f"Unsupported input type: {type(x)}")


def heavy_atoms(ag):
    """Return heavy atoms only."""
    return ag.select_atoms("not name H*")


def _first_frame_atoms(u: mda.Universe):
    """
    Return atoms at the first frame of a Universe.
    For single-structure PDBs this is just u.atoms.
    For multi-model/trajectory inputs, this uses frame 0.
    """
    if hasattr(u, "trajectory") and len(u.trajectory) > 0:
        u.trajectory[0]
    return u.atoms


def _resname_to_aa(resname: str) -> str:
    """
    Convert 3-letter residue name to 1-letter code.
    Unknown residues are mapped to 'X'.
    """
    key = resname.capitalize()
    return protein_letters_3to1.get(key, "X")


def _extract_sequence_and_mapping(protein_u: mda.Universe):
    """
    Extract full protein sequence and mapping:
        resindex -> sequence index
    """
    residues = protein_u.residues
    full_sequence = "".join(_resname_to_aa(res.resname) for res in residues)
    full_resindices = residues.resindices.astype(np.int64)
    resindex_to_seqidx = {int(residx): i for i, residx in enumerate(full_resindices)}
    return full_sequence, full_resindices, resindex_to_seqidx


def extract_fixed_pocket(
    protein: Union[str, mda.Universe],
    peptide: Union[str, mda.Universe],
    cutoff: float = 6.0,
    save_path: Optional[str] = None,
) -> dict:
    """
    Extract fixed pocket using a reference peptide conformation.

    Parameters
    ----------
    protein : str or MDAnalysis.Universe
        Protein structure.
    peptide : str or MDAnalysis.Universe
        Reference peptide structure. Can be a single-structure file or
        a multi-model/trajectory file.
    cutoff : float
        Pocket cutoff in Å.
    save_path : str or None
        If provided, save the pocket protein structure to this path.

    Returns
    -------
    dict
        {
            "pocket_atoms": AtomGroup,
            "pocket_resindices": np.ndarray,
            "pocket_atom_indices": np.ndarray,
            "pocket_atom_mask": np.ndarray,
            "full_sequence": str,
            "pocket_sequence": str,
            "full_resindices": np.ndarray,
            "pocket_seq_indices": np.ndarray,
        }
    """
    protein_u = _to_universe(protein)
    peptide_u = _to_universe(peptide)

    prot_atoms = protein_u.atoms
    pep_atoms = _first_frame_atoms(peptide_u)

    prot_heavy = heavy_atoms(prot_atoms)
    pep_heavy = heavy_atoms(pep_atoms)

    if len(prot_heavy) == 0:
        raise ValueError("Protein has no heavy atoms.")
    if len(pep_heavy) == 0:
        raise ValueError("Peptide has no heavy atoms.")

    dmat = mda.lib.distances.distance_array(
        prot_heavy.positions,
        pep_heavy.positions
    )

    close_atom_mask = (dmat.min(axis=1) <= cutoff)

    close_resindices = np.unique(
        prot_heavy[close_atom_mask].residues.resindices
    ).astype(np.int64)

    if len(close_resindices) == 0:
        atom_min_dists = dmat.min(axis=1)
        nearest_atom_idx = np.argmin(atom_min_dists)
        nearest_resindex = prot_heavy[nearest_atom_idx].residue.resindex
        close_resindices = np.array([nearest_resindex], dtype=np.int64)

    pocket_atoms = protein_u.atoms[
        np.isin(protein_u.atoms.resindices, close_resindices)
    ]

    pocket_atom_indices = pocket_atoms.indices.astype(np.int64)

    pocket_atom_mask = np.zeros(len(protein_u.atoms), dtype=bool)
    pocket_atom_mask[pocket_atom_indices] = True

    # ===== sequence + mapping =====
    full_sequence, full_resindices, resindex_to_seqidx = _extract_sequence_and_mapping(protein_u)
    pocket_seq_indices = np.array(
        [resindex_to_seqidx[int(r)] for r in pocket_atoms.residues.resindices],
        dtype=np.int64
    )
    pocket_sequence = "".join(full_sequence[i] for i in pocket_seq_indices)

    # ===== peptide sequence =====
    pep_residues = pep_atoms.residues
    peptide_sequence = "".join(_resname_to_aa(res.resname) for res in pep_residues)
    peptide_resindices = pep_residues.resindices.astype(np.int64)

    if save_path is not None:
        pocket_atoms.write(save_path)

    return {
        "pocket_pdb": save_path,
        "pocket_atoms": pocket_atoms,
        "pocket_resindices": close_resindices,
        "pocket_atom_indices": pocket_atom_indices,
        "pocket_atom_mask": pocket_atom_mask,

        "protein_sequence": full_sequence,
        "pocket_sequence": pocket_sequence,
        "protein_resindices": full_resindices,
        "pocket_seq_indices": pocket_seq_indices,

        "peptide_sequence": peptide_sequence,
        "peptide_resindices": peptide_resindices,        
    }


def extract_fixed_pocket_by_mode(
    protein: Union[str, mda.Universe],
    native_peptide: Optional[Union[str, mda.Universe]],
    decoy_pdb: Optional[Union[str, mda.Universe]],
    mode: Literal["train", "infer"],
    cutoff: float = 6.0,
    save_path: Optional[str] = None,
) -> dict:
    """
    Training:
        use native peptide to define pocket
    Inference:
        use the first frame/model in decoy_pdb to define pocket
    """
    if save_path is None:
        save_path = f"pocket_{cutoff}A.pdb"
        
    if mode == "train":
        if native_peptide is None:
            raise ValueError("native_peptide is required in training mode.")
        out = extract_fixed_pocket(
            protein=protein,
            peptide=native_peptide,
            cutoff=cutoff,
            save_path=save_path,
        )
        out["pocket_mode"] = mode
        out["pocket_reference"] = "native_peptide"

    elif mode == "infer":
        if decoy_pdb is None:
            raise ValueError("decoy_pdb is required in inference mode.")
        out = extract_fixed_pocket(
            protein=protein,
            peptide=decoy_pdb,
            cutoff=cutoff,
            save_path=save_path,
        )
        out["pocket_mode"] = mode
        out["pocket_reference"] = "first_decoy_from_decoy_pdb"

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return out