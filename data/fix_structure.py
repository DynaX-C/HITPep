from pathlib import Path
from typing import Optional, Dict

import parmed as pmd
from pdbfixer import PDBFixer
from openmm.app import PDBFile


def reorder_pdb(
    input_pdb: str, 
    output_pdb: Optional[str] = None, 
    remove_hydrogens: bool = False
) -> str:
    """
    Reorder atoms using parmed and optionally remove hydrogens.

    Parameters
    ----------
    input_pdb : str
    output_pdb : str or None
    remove_hydrogens : bool

    Returns
    -------
    output_pdb : str
    """
    input_pdb = str(input_pdb)
    if output_pdb is None:
        output_pdb = input_pdb.replace(".pdb", "_reorder.pdb")

    structure = pmd.load_file(input_pdb)

    if remove_hydrogens:
        structure.strip("@H=")

    structure.save(output_pdb, overwrite=True)
    return output_pdb


def fix_pdb(
    input_pdb: str,
    output_pdb: Optional[str] = None,
    remove_hydrogens: bool = True,
    keep_water: bool = False,
    add_missing_residues: bool = True,
    add_missing_atoms: bool = True,
    replace_nonstandard: bool = True,
    reorder: bool = True,
) -> str:
    """
    Fix structure using PDBFixer.

    Strategy
    --------
    - optionally add missing residues
    - replace nonstandard residues
    - remove heterogens
    - add missing atoms
    - do NOT add hydrogens
    - do NOT minimize

    Parameters
    ----------
    input_pdb : str
    output_pdb : str or None
        Final output pdb path.
    remove_hydrogens : bool
        Remove hydrogens after fixing.
    keep_water : bool
        Whether to keep water molecules when removing heterogens.
    add_missing_residues : bool
    add_missing_atoms : bool
    replace_nonstandard : bool
    reorder : bool
        Reorder final structure using parmed.

    Returns
    -------
    final_pdb : str
    """
    input_pdb = str(input_pdb)

    if output_pdb is None:
        output_pdb = input_pdb.replace(".pdb", "_fixed.pdb")
    output_pdb = str(output_pdb)

    fixer = PDBFixer(filename=input_pdb)

    # --------------------------------------------------
    # missing residues
    # --------------------------------------------------
    fixer.findMissingResidues()
    if not add_missing_residues:
        fixer.missingResidues = {}
    else:
        # optional: remove terminal missing residues only
        chains = list(fixer.topology.chains())
        keys = list(fixer.missingResidues.keys())
        for key in keys:
            chain = chains[key[0]]
            residues = list(chain.residues())
            if key[1] == 0 or key[1] == len(residues):
                del fixer.missingResidues[key]

    # --------------------------------------------------
    # nonstandard residues
    # --------------------------------------------------
    if replace_nonstandard:
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()

    # --------------------------------------------------
    # heterogens
    # --------------------------------------------------
    fixer.removeHeterogens(keepWater=keep_water)

    # --------------------------------------------------
    # missing atoms
    # --------------------------------------------------
    if add_missing_atoms:
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()

    # --------------------------------------------------
    # write temporary fixed pdb
    # --------------------------------------------------
    with open(output_pdb, "w") as f:
        PDBFile.writeFile(
            fixer.topology,
            fixer.positions,
            f,
            keepIds=True,
        )

    # --------------------------------------------------
    # reorder / strip hydrogens
    # --------------------------------------------------
    if reorder:
        final_pdb = reorder_pdb(
            input_pdb=output_pdb,
            remove_hydrogens=remove_hydrogens,
        )
        return final_pdb
    else:
        return output_pdb