import numpy as np
import torch
import MDAnalysis as mda
from MDAnalysis.analysis.rms import rmsd
from tqdm import tqdm


# =========================================================
# utils
# =========================================================
def _round_tensor(x, decimals=4): 
    return torch.round(x * (10 ** decimals)) / (10 ** decimals)


# def _load_decoy_universe(decoy_peptide_pdb, native_peptide_pdb, decoy_id):
#     if decoy_id == 0:
#         return mda.Universe(native_peptide_pdb)
#     u = mda.Universe(decoy_peptide_pdb)
#     u.trajectory[decoy_id - 1]
#     return u


def compute_tau_from_dataset(values, percentile=75.0):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        raise ValueError("Empty values for tau.")
    return float(np.percentile(values, percentile) + 1e-8)


def exp_normalize(values, tau):
    values = np.asarray(values, dtype=np.float32)
    return np.exp(-values / tau).astype(np.float32)


# =========================================================
# raw geometry
# =========================================================

def extract_raw_label_values(native_peptide_pdb, decoy_peptide_pdb):
    u_nat = mda.Universe(native_peptide_pdb)
    u_dec = mda.Universe(decoy_peptide_pdb)

    # ----- atom displacement -----
    nat_atoms = u_nat.select_atoms("not name H*")
    dec_atoms = u_dec.select_atoms("not name H*")

    atom_disp = np.linalg.norm(
        dec_atoms.positions - nat_atoms.positions, axis=1
    ).astype(np.float32)

    # ----- residue + global BRMSD -----
    nat_res = u_nat.residues
    dec_res = u_dec.residues

    res_brmsd = []
    nat_all = []
    dec_all = []

    for i in range(len(nat_res)):
        nat_bb = nat_res[i].atoms.select_atoms("name N or name CA or name C")
        dec_bb = dec_res[i].atoms.select_atoms("name N or name CA or name C")

        if len(nat_bb) == len(dec_bb) and len(nat_bb) > 0:
            br = rmsd(dec_bb.positions, nat_bb.positions,
                      center=False, superposition=False)
            nat_all.append(nat_bb.positions)
            dec_all.append(dec_bb.positions)
        else:
            br = 0.0

        res_brmsd.append(br)

    res_brmsd = np.asarray(res_brmsd, dtype=np.float32)

    # global BRMSD
    nat_all = np.concatenate(nat_all, axis=0)
    dec_all = np.concatenate(dec_all, axis=0)

    global_brmsd = float(
        rmsd(dec_all, nat_all, center=False, superposition=False)
    )

    return {
        "atom_disp": atom_disp,
        "res_brmsd": res_brmsd,
        "global_brmsd": global_brmsd,
    }


# =========================================================
# FNAT
# =========================================================

def _compute_global_fnat(
    receptor_pdb,
    native_peptide_pdb,
    decoy_peptide_pdb,
    cutoff=5.0,
):
    u_rec = mda.Universe(receptor_pdb)
    u_nat = mda.Universe(native_peptide_pdb)
    u_dec = mda.Universe(decoy_peptide_pdb)

    rec_res = u_rec.residues
    nat_res = u_nat.residues
    dec_res = u_dec.residues

    def get_res_contacts(pep_res, rec_res):
        contacts = set()

        for i, r_pep in enumerate(pep_res):
            pep_atoms = r_pep.atoms.select_atoms("not name H*").positions

            for j, r_rec in enumerate(rec_res):
                rec_atoms = r_rec.atoms.select_atoms("not name H*").positions

                found = False
                for a in pep_atoms:
                    d = np.linalg.norm(rec_atoms - a, axis=1)
                    if np.any(d <= cutoff):
                        contacts.add((i, j))
                        found = True
                        break
                if found:
                    continue

        return contacts

    nat_contacts = get_res_contacts(nat_res, rec_res)
    dec_contacts = get_res_contacts(dec_res, rec_res)

    if len(nat_contacts) == 0:
        return 0.0

    return len(nat_contacts & dec_contacts) / len(nat_contacts)


def _compute_per_residue_fnat(
    receptor_pdb,
    native_peptide_pdb,
    decoy_peptide_pdb,
    num_res_total,
    peptide_res_ids,
    cutoff=5.0,
):
    u_rec = mda.Universe(receptor_pdb)
    u_nat = mda.Universe(native_peptide_pdb)
    u_dec = mda.Universe(decoy_peptide_pdb)

    rec = u_rec.select_atoms("not name H*").positions
    nat_res = u_nat.residues
    dec_res = u_dec.residues

    res_int = np.zeros(num_res_total, dtype=np.float32)

    for i, rid in enumerate(peptide_res_ids):
        nat_atoms = nat_res[i].atoms.select_atoms("not name H*").positions
        dec_atoms = dec_res[i].atoms.select_atoms("not name H*").positions

        def contacts(coords):
            s = set()
            for ai, a in enumerate(coords):
                d = np.linalg.norm(rec - a, axis=1)
                for j in np.where(d <= cutoff)[0]:
                    s.add((ai, int(j)))
            return s

        nat_c = contacts(nat_atoms)
        dec_c = contacts(dec_atoms)

        if len(nat_c) > 0:
            res_int[int(rid)] = len(nat_c & dec_c) / len(nat_c)

    return res_int


# =========================================================
# tau
# =========================================================

def collect_dataset_tau(samples, percentile=75.0):
    atom_all = []
    res_all = []
    global_all = []

    for item in tqdm(samples, total=len(samples), desc="Collecting tau samples"):
        raw = extract_raw_label_values(
            item["native_peptide_pdb"],
            item["decoy_peptide_pdb"],
        )
        atom_all.append(raw["atom_disp"])
        res_all.append(raw["res_brmsd"])
        global_all.append(raw["global_brmsd"])

    return {
        "tau_atom": compute_tau_from_dataset(np.concatenate(atom_all), percentile),
        "tau_res": compute_tau_from_dataset(np.concatenate(res_all), percentile),
        "tau_global": compute_tau_from_dataset(np.array(global_all), percentile),
    }


# =========================================================
# main label builder
# =========================================================

def build_labels(
    receptor_pdb,
    native_peptide_pdb,
    decoy_peptide_pdb,
    atom_graph,
    res_graph,
    tau_atom,
    tau_res,
    tau_global,
    contact_threshold=5.0,
):
    raw = extract_raw_label_values(
        native_peptide_pdb,
        decoy_peptide_pdb,
    )

    # ----- normalize -----
    atom_score = exp_normalize(raw["atom_disp"], tau_atom)
    res_geom = exp_normalize(raw["res_brmsd"], tau_res)
    global_geom = float(np.exp(-raw["global_brmsd"] / tau_global))

    # ----- atom -----
    mask_atom = atom_graph.is_peptide.bool()
    atom_label = np.zeros(atom_graph.x.size(0), dtype=np.float32)
    atom_label[mask_atom.cpu().numpy()] = atom_score

    # ----- residue -----
    mask_res = res_graph.peptide_mask.bool()
    res_ids = np.where(mask_res.cpu().numpy())[0]

    res_geom_full = np.zeros(res_graph.x.size(0), dtype=np.float32)
    res_geom_full[res_ids] = res_geom

    res_int = _compute_per_residue_fnat(
        receptor_pdb,
        native_peptide_pdb,
        decoy_peptide_pdb,
        res_graph.x.size(0),
        res_ids,
        contact_threshold,
    )

    # ----- global FNAT -----
    global_fnat = _compute_global_fnat(
        receptor_pdb,
        native_peptide_pdb,
        decoy_peptide_pdb,
        contact_threshold,
    )

    # global_score = 0.5 * global_geom + 0.5 * global_fnat

    out = {
        "atom": torch.tensor(atom_label, dtype=torch.float),
        "res_geom": torch.tensor(res_geom_full, dtype=torch.float),
        "res_int": torch.tensor(res_int, dtype=torch.float),
        # "global": torch.tensor(global_score, dtype=torch.float),
        "global_geom": torch.tensor(global_geom, dtype=torch.float),
        "global_int": torch.tensor(global_fnat, dtype=torch.float),
        "peptide_mask_atom": mask_atom,
        "peptide_mask_res": mask_res,
    }

    for k in ["atom", "res_geom", "res_int", "global_geom", "global_int"]:
        out[k] = _round_tensor(out[k], 4)

    return out