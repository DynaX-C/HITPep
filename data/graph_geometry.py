import numpy as np
import parmed as pmd
import torch
from typing import Optional
from torch_geometric.data import Data
from itertools import combinations
from .label import build_labels


# =========================================================
# geometry helpers
# =========================================================

def _cross(v1, v2):
    x = v1[..., 1] * v2[..., 2] - v1[..., 2] * v2[..., 1]
    y = v1[..., 2] * v2[..., 0] - v1[..., 0] * v2[..., 2]
    z = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
    return torch.stack([x, y, z], dim=-1)


def calc_dihedral(a0, a1, a2, a3, eps=1e-8):
    v1 = a1 - a0
    v2 = a1 - a2
    v3 = a3 - a2

    v1xv2 = _cross(v1, v2)
    v2xv3 = _cross(v2, v3)
    l1 = torch.sqrt(torch.sum(v1xv2 * v1xv2, dim=-1) + eps)
    l2 = torch.sqrt(torch.sum(v2xv3 * v2xv3, dim=-1) + eps)
    cosa = torch.sum(v1xv2 * v2xv3, dim=-1) / (l1 * l2 + eps)
    cosa = torch.clamp(cosa, -1.0, 1.0)

    angle = torch.acos(cosa)
    sign = torch.where(torch.sum(v3 * v1xv2, dim=-1) <= 0, 1.0, -1.0)
    return angle * sign


def calc_angle(a1, a2, a3, eps=1e-8):
    v1 = a2 - a1
    v2 = a2 - a3

    l1 = torch.sqrt(torch.sum(v1 * v1, dim=-1) + eps)
    l2 = torch.sqrt(torch.sum(v2 * v2, dim=-1) + eps)

    cos_angle = torch.sum(v1 * v2, dim=-1) / (l1 * l2 + eps)
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)

    angle = torch.acos(cos_angle)
    return angle


# =========================================================
# feature helpers
# =========================================================

AA_ORDER = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
    "UNK",
]

ELEMENT_ORDER = ["C", "N", "O", "S", "UNK"]

ATOM_ROLE_ORDER = [
    "BB_N", "BB_CA", "BB_C", "BB_O",
    "SC_C", "SC_N", "SC_O", "SC_S",
    "OTHER"
]


def one_hot_encoding(value, choices):
    if value not in choices:
        value = choices[-1]
    return [value == c for c in choices]


def normalize_resname(resname: str) -> str:
    name = resname.strip().upper()

    if name in {"HID", "HIE", "HIP", "HIS"}:
        return "HIS"
    if name in {"CYS", "CYX", "CYM"}:
        return "CYS"
    if name in {"ARG", "ARN"}:
        return "ARG"
    if name in {"LYS", "LYN"}:
        return "LYS"
    if name in {"ASP", "ASH"}:
        return "ASP"
    if name in {"GLU", "GLH"}:
        return "GLU"

    return name if name in AA_ORDER[:-1] else "UNK"


def get_atom_element(atom):
    elem = atom.element_name.upper()
    if elem not in {"C", "N", "O", "S"}:
        return "UNK"
    return elem


def is_heavy_atom(atom):
    return atom.element_name.upper() != "H"


def is_backbone_atom(atom_name: str) -> int:
    return int(atom_name.strip().upper() in {"N", "CA", "C", "O", "OXT"})


def atom_role_class(atom):
    name = atom.name.strip().upper()
    elem = atom.element_name.upper()

    if name == "N":
        return "BB_N"
    if name == "CA":
        return "BB_CA"
    if name == "C":
        return "BB_C"
    if name in {"O", "OXT"}:
        return "BB_O"

    if elem == "C":
        return "SC_C"
    if elem == "N":
        return "SC_N"
    if elem == "O":
        return "SC_O"
    if elem == "S":
        return "SC_S"

    return "OTHER"


def atom_feature_from_parmed(atom, is_protein: int, is_peptide: int):
    elem = get_atom_element(atom)
    role = atom_role_class(atom)
    resname = normalize_resname(atom.residue.name)

    feat = []
    feat += one_hot_encoding(elem, ELEMENT_ORDER)
    feat += one_hot_encoding(role, ATOM_ROLE_ORDER)
    feat += one_hot_encoding(resname, AA_ORDER)
    feat += [is_backbone_atom(atom.name)]
    feat += [is_protein]
    feat += [is_peptide]
    return feat


# =========================================================
# structure loading / merging
# =========================================================

def load_structure(pdb_path: str):
    return pmd.load_file(pdb_path)


def build_residue_id_maps(pocket_structure, peptide_structure):
    """
    Unified residue ids:
    - pocket protein residues: 0 ... Np-1
    - peptide residues: Np ... Np+Nl-1
    """
    protein_resid_map = {}
    cur = 0
    for res in pocket_structure.residues:
        protein_resid_map[res.idx] = cur
        cur += 1

    peptide_resid_map = {}
    for res in peptide_structure.residues:
        peptide_resid_map[res.idx] = cur
        cur += 1

    return protein_resid_map, peptide_resid_map, cur


def build_heavy_atom_views(pocket_structure, peptide_structure):
    prot_atoms = [a for a in pocket_structure.atoms if is_heavy_atom(a)]
    pep_atoms = [a for a in peptide_structure.atoms if is_heavy_atom(a)]

    prot_old_to_new = {}
    pep_old_to_new = {}

    for i, atom in enumerate(prot_atoms):
        prot_old_to_new[atom.idx] = i

    offset = len(prot_atoms)
    for j, atom in enumerate(pep_atoms):
        pep_old_to_new[atom.idx] = offset + j

    return prot_atoms, pep_atoms, prot_old_to_new, pep_old_to_new


def build_node_features_and_coords(
    prot_atoms,
    pep_atoms,
    protein_resid_map,
    peptide_resid_map,
):
    x = []
    coords = []
    atom2res = []
    is_protein = []
    is_peptide = []

    for atom in prot_atoms:
        x.append(atom_feature_from_parmed(atom, is_protein=1, is_peptide=0))
        coords.append([float(atom.xx), float(atom.xy), float(atom.xz)])
        atom2res.append(protein_resid_map[atom.residue.idx])
        is_protein.append(1)
        is_peptide.append(0)

    for atom in pep_atoms:
        x.append(atom_feature_from_parmed(atom, is_protein=0, is_peptide=1))
        coords.append([float(atom.xx), float(atom.xy), float(atom.xz)])
        atom2res.append(peptide_resid_map[atom.residue.idx])
        is_protein.append(0)
        is_peptide.append(1)

    x = torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float)
    coords = torch.tensor(np.asarray(coords, dtype=np.float32), dtype=torch.float)
    atom2res = torch.tensor(np.asarray(atom2res, dtype=np.int64), dtype=torch.long)
    is_protein = torch.tensor(np.asarray(is_protein, dtype=np.int64), dtype=torch.long)
    is_peptide = torch.tensor(np.asarray(is_peptide, dtype=np.int64), dtype=torch.long)

    return x, coords, atom2res, is_protein, is_peptide


# =========================================================
# graph builders
# =========================================================

def build_nb_atom_graph(
    prot_atoms,
    pep_atoms,
    coords,
    x,
    atom2res,
    is_protein,
    is_peptide,
    cutoff=6.0,
    complex_name=None,
):
    """
    Nonbonded atom graph:
    - nodes = heavy atoms in pocket protein + peptide
    - edges = ONLY cross-interface heavy-atom pairs within cutoff
    """
    n_prot = len(prot_atoms)
    n_pep = len(pep_atoms)

    edge_src, edge_dst, edge_attr, edge_type = [], [], [], []

    if n_prot > 0 and n_pep > 0:
        prot_xyz = coords[:n_prot].numpy()
        pep_xyz = coords[n_prot:n_prot + n_pep].numpy()

        diff = prot_xyz[:, None, :] - pep_xyz[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))

        row, col = np.where(dist <= cutoff)
        for r, c in zip(row, col):
            i = int(r)
            j = int(n_prot + c)
            d = float(dist[r, c])

            edge_src.extend([i, j])
            edge_dst.extend([j, i])
            edge_attr.extend([[d], [d]])

            t = [0, 0, 1] # cross-interface edge
            edge_type.extend([t, t])

    if len(edge_attr) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(np.asarray(edge_attr, dtype=np.float32), dtype=torch.float)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    graph.coord = coords
    graph.edge_type = torch.tensor(edge_type, dtype=torch.float)
    graph.atom2res = atom2res
    graph.is_protein = is_protein
    graph.is_peptide = is_peptide
    graph.complex_name = complex_name
    return graph

def build_bd_atom_graph(
    pocket_structure,
    peptide_structure,
    prot_atoms,
    pep_atoms,
    prot_old_to_new,
    pep_old_to_new,
    x,
    coords,
    atom2res,
    is_protein,
    is_peptide,
    complex_name=None
):
    """
    Bonded atom graph:
    - nodes = heavy atoms in pocket protein + peptide
    - edges = covalent bonds inside each molecule only

    Important
    ---------
    Bond ordering is canonicalized to ensure stable downstream indexing:
        1. each bond is represented as (min(i, j), max(i, j))
        2. all bonds are globally sorted by (i, j)
        3. directed edges are then generated in a deterministic order
    """
    neighbors = {i: [] for i in range(len(prot_atoms) + len(pep_atoms))}

    # store undirected canonical bonds first
    bond_records = []

    # protein bonds
    for bond in pocket_structure.bonds:
        i_old = bond.atom1.idx
        j_old = bond.atom2.idx
        if i_old not in prot_old_to_new or j_old not in prot_old_to_new:
            continue

        i = prot_old_to_new[i_old]
        j = prot_old_to_new[j_old]

        u, v = sorted((i, j))

        xyz_i = np.array([bond.atom1.xx, bond.atom1.xy, bond.atom1.xz], dtype=np.float32)
        xyz_j = np.array([bond.atom2.xx, bond.atom2.xy, bond.atom2.xz], dtype=np.float32)
        dist = float(np.linalg.norm(xyz_i - xyz_j))

        bond_records.append((u, v, dist, [1, 0, 0]))  # intra-protein

        neighbors[i].append(j)
        neighbors[j].append(i)

    # peptide bonds
    for bond in peptide_structure.bonds:
        i_old = bond.atom1.idx
        j_old = bond.atom2.idx
        if i_old not in pep_old_to_new or j_old not in pep_old_to_new:
            continue

        i = pep_old_to_new[i_old]
        j = pep_old_to_new[j_old]

        u, v = sorted((i, j))

        xyz_i = np.array([bond.atom1.xx, bond.atom1.xy, bond.atom1.xz], dtype=np.float32)
        xyz_j = np.array([bond.atom2.xx, bond.atom2.xy, bond.atom2.xz], dtype=np.float32)
        dist = float(np.linalg.norm(xyz_i - xyz_j))

        bond_records.append((u, v, dist, [0, 1, 0]))  # intra-peptide

        neighbors[i].append(j)
        neighbors[j].append(i)

    # remove duplicates if any, then sort globally
    # key = (u, v)
    uniq = {}
    for u, v, dist, etype in bond_records:
        uniq[(u, v)] = (dist, etype)

    sorted_bonds = sorted(uniq.items(), key=lambda kv: kv[0])  # sort by (u, v)

    edge_src, edge_dst, edge_attr, edge_type = [], [], [], []

    # deterministic directed-edge expansion
    for (u, v), (dist, etype) in sorted_bonds:
        edge_src.extend([u, v])
        edge_dst.extend([v, u])
        edge_attr.extend([[dist], [dist]])
        edge_type.extend([etype, etype])

    if len(edge_attr) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
        edge_type = torch.empty((0, 3), dtype=torch.float)
    else:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(np.asarray(edge_attr, dtype=np.float32), dtype=torch.float)
        edge_type = torch.tensor(np.asarray(edge_type, dtype=np.float32), dtype=torch.float)

    # directed bond -> node id, now deterministic
    bond_to_node = {}
    for idx, (src, dst) in enumerate(edge_index.t().tolist()):
        bond_to_node[(src, dst)] = idx

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    graph.coord = coords
    graph.edge_type = edge_type
    graph.atom2res = atom2res
    graph.is_protein = is_protein
    graph.is_peptide = is_peptide
    graph.complex_name = complex_name

    return graph, bond_to_node, neighbors

def build_atom_graph(nb_graph: Data, bd_graph: Data):
    """
    Unified atom graph:
    - edges = NBA ∪ BDA
    - used for atom refinement MP
    """

    # concat edges
    edge_index = torch.cat(
        [nb_graph.edge_index, bd_graph.edge_index],
        dim=1
    )

    edge_attr = torch.cat(
        [nb_graph.edge_attr, bd_graph.edge_attr],
        dim=0
    )

    edge_type = torch.cat([
        nb_graph.edge_type,
        bd_graph.edge_type
    ], dim=0)

    graph = Data(
        x=nb_graph.x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )

    graph.edge_type = edge_type
    graph.coord = nb_graph.coord
    graph.atom2res = nb_graph.atom2res
    graph.is_protein = nb_graph.is_protein
    graph.is_peptide = nb_graph.is_peptide
    graph.complex_name = nb_graph.complex_name

    return graph

def build_bond_bond_graph(bd_atom_graph: Data, neighbors: dict, bond_to_node: dict):
    """
    Bond-bond graph:
    - nodes = directed bonds in bd_atom_graph.edge_index
    - edges = directed angles
    - edge_attr = angle value
    """
    coords = bd_atom_graph.coord

    edge_src, edge_dst, edge_attr = [], [], []
    angle_three_index = []
    angle_to_node = {}

    angle_node_id = 0

    for center_idx, neighs in neighbors.items():
        if len(neighs) < 2:
            continue

        for i in range(len(neighs)):
            for j in range(i + 1, len(neighs)):
                a = neighs[i]
                c = neighs[j]

                # a-center-c
                b1 = (a, center_idx)
                b2 = (c, center_idx)
                if b1 in bond_to_node and b2 in bond_to_node:
                    angle = calc_angle(
                        coords[a].unsqueeze(0),
                        coords[center_idx].unsqueeze(0),
                        coords[c].unsqueeze(0),
                    )[0].item()

                    edge_src.append(bond_to_node[b1])
                    edge_dst.append(bond_to_node[b2])
                    edge_attr.append([angle])
                    angle_three_index.append((a, center_idx, c))
                    angle_to_node[(a, center_idx, c)] = angle_node_id
                    angle_node_id += 1

                # c-center-a
                b1r = (c, center_idx)
                b2r = (a, center_idx)
                if b1r in bond_to_node and b2r in bond_to_node:
                    angle = calc_angle(
                        coords[c].unsqueeze(0),
                        coords[center_idx].unsqueeze(0),
                        coords[a].unsqueeze(0),
                    )[0].item()

                    edge_src.append(bond_to_node[b1r])
                    edge_dst.append(bond_to_node[b2r])
                    edge_attr.append([angle])
                    angle_three_index.append((c, center_idx, a))
                    angle_to_node[(c, center_idx, a)] = angle_node_id
                    angle_node_id += 1

    if len(edge_attr) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
        angle_face = torch.empty((3, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(np.asarray(edge_attr, dtype=np.float32), dtype=torch.float)
        angle_face = torch.tensor(np.asarray(angle_three_index, dtype=np.int64), dtype=torch.long).t().contiguous()

    graph = Data(x=bd_atom_graph.edge_attr ,edge_index=edge_index, edge_attr=edge_attr)
    graph.complex_name = bd_atom_graph.complex_name
    return graph, angle_face, angle_to_node


def build_angle_angle_graph(bond_bond_graph: Data, bd_atom_graph: Data, neighbors: dict, angle_to_node: dict):
    """
    Angle-angle graph:
    - nodes = directed angles
    - edges = proper dihedrals + impropers between directed angles
    - edge_attr = dihedral / improper value
    """
    coords = bd_atom_graph.coord

    edge_src, edge_dst, edge_attr = [], [], []
    seen_edges = set()

    # proper dihedrals
    for b in neighbors:
        for c in neighbors.get(b, []):
            for a in neighbors.get(b, []):
                if a == c:
                    continue
                for d in neighbors.get(c, []):
                    if d == b or d == a:
                        continue

                    ang1 = (a, b, c)
                    ang2 = (d, c, b)

                    if ang1 in angle_to_node and ang2 in angle_to_node:
                        n1 = angle_to_node[ang1]
                        n2 = angle_to_node[ang2]

                        key = (n1, n2, a, b, c, d)
                        if key in seen_edges:
                            continue
                        seen_edges.add(key)

                        dih = calc_dihedral(
                            coords[a].unsqueeze(0),
                            coords[b].unsqueeze(0),
                            coords[c].unsqueeze(0),
                            coords[d].unsqueeze(0),
                        )[0].item()

                        edge_src.extend([n1, n2])
                        edge_dst.extend([n2, n1])
                        edge_attr.extend([[dih], [dih]])

    # impropers
    for c_idx, neighs in neighbors.items():
        if len(neighs) < 3:
            continue

        for a, b, d in combinations(neighs, 3):
            ang1 = (a, c_idx, b)
            ang2 = (d, c_idx, b)

            if ang1 in angle_to_node and ang2 in angle_to_node:
                n1 = angle_to_node[ang1]
                n2 = angle_to_node[ang2]

                key = (n1, n2, a, b, c_idx, d, "improper")
                if key in seen_edges:
                    continue
                seen_edges.add(key)

                improper = calc_dihedral(
                    coords[a].unsqueeze(0),
                    coords[b].unsqueeze(0),
                    coords[c_idx].unsqueeze(0),
                    coords[d].unsqueeze(0),
                )[0].item()

                edge_src.extend([n1, n2])
                edge_dst.extend([n2, n1])
                edge_attr.extend([[improper], [improper]])

    if len(edge_attr) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(np.asarray(edge_attr, dtype=np.float32), dtype=torch.float)

    graph = Data(x=bond_bond_graph.edge_attr, edge_index=edge_index, edge_attr=edge_attr)
    graph.complex_name = bond_bond_graph.complex_name
    return graph

def build_res_graph(
    coords: torch.Tensor,
    atom2res: torch.Tensor,
    is_protein: torch.Tensor,
    is_peptide: torch.Tensor,
    num_res: int,
    esm_res: Optional[torch.Tensor] = None,
    cutoff: float = 6.0,
    complex_name: Optional[str] = None
):
    """
    Residue graph:
    - nodes = residues
    - node features = ESM residue embeddings (optional)
    - edges = residue pairs within cutoff (using minimum heavy-atom distance)
    - edge_attr = minimum heavy-atom distance
    - edge_type:
        [1,0,0] protein-protein
        [0,1,0] peptide-peptide
        [0,0,1] protein-peptide
    """

    device = coords.device

    # ===== residue-level masks =====
    peptide_mask = torch.zeros(num_res, dtype=torch.bool, device=device)
    protein_mask = torch.zeros(num_res, dtype=torch.bool, device=device)

    for r in range(num_res):
        atom_mask = (atom2res == r)
        if atom_mask.any():
            if bool(is_peptide[atom_mask][0].item()):
                peptide_mask[r] = True
            if bool(is_protein[atom_mask][0].item()):
                protein_mask[r] = True

    # ===== residue atom lists =====
    res_atoms = []
    for r in range(num_res):
        idx = torch.where(atom2res == r)[0]
        res_atoms.append(idx)

    edge_src, edge_dst, edge_attr, edge_type = [], [], [], []

    # ===== all residue pairs =====
    for i in range(num_res):
        atoms_i = res_atoms[i]
        if atoms_i.numel() == 0:
            continue

        xyz_i = coords[atoms_i]   # [Ni, 3]

        for j in range(i + 1, num_res):
            atoms_j = res_atoms[j]
            if atoms_j.numel() == 0:
                continue

            xyz_j = coords[atoms_j]   # [Nj, 3]

            diff = xyz_i[:, None, :] - xyz_j[None, :, :]
            dist = torch.sqrt(torch.sum(diff * diff, dim=-1))
            min_dist = float(dist.min().item())

            if min_dist > cutoff:
                continue

            # edge type
            if protein_mask[i] and protein_mask[j]:
                t = [1, 0, 0]   # protein-protein
            elif peptide_mask[i] and peptide_mask[j]:
                t = [0, 1, 0]   # peptide-peptide
            else:
                t = [0, 0, 1]   # protein-peptide

            edge_src.extend([i, j])
            edge_dst.extend([j, i])
            edge_attr.extend([[min_dist], [min_dist]])
            edge_type.extend([t, t])

    if len(edge_attr) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.empty((0, 1), dtype=torch.float, device=device)
        edge_type = torch.empty((0, 3), dtype=torch.float, device=device)
    else:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=device)
        edge_attr = torch.tensor(np.asarray(edge_attr, dtype=np.float32), dtype=torch.float, device=device)
        edge_type = torch.tensor(np.asarray(edge_type, dtype=np.float32), dtype=torch.float, device=device)

    # ===== node feature =====
    if esm_res is None:
        x = torch.zeros((num_res, 1), dtype=torch.float, device=device)
    else:
        if esm_res.size(0) != num_res:
            raise ValueError(
                f"esm_res first dimension ({esm_res.size(0)}) must equal num_res ({num_res})."
            )
        x = esm_res

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    graph.edge_type = edge_type
    graph.peptide_mask = peptide_mask
    graph.protein_mask = protein_mask
    graph.complex_name = complex_name

    return graph

# =========================================================
# total builder
# =========================================================

def build_geometry_graphs(
    protein_pdb: str = None,
    pocket_pdb: str = None,
    peptide_pdb: str = None,
    esm_res: Optional[torch.Tensor] = None,
    cutoff: float = 6.0,
    complex_name: Optional[str] = None,
    native_peptide_pdb: Optional[str] = None,
    tau_atom: Optional[float] = None,
    tau_res: Optional[float] = None,
    tau_global: Optional[float] = None,
    contact_threshold: float = 5.0,
):
    """
    Build all four hierarchical graphs for the no-force-field HITPep version.
    """
    pocket_structure = load_structure(pocket_pdb)
    peptide_structure = load_structure(peptide_pdb)

    protein_resid_map, peptide_resid_map, num_res = build_residue_id_maps(
        pocket_structure, peptide_structure
    )

    prot_atoms, pep_atoms, prot_old_to_new, pep_old_to_new = build_heavy_atom_views(
        pocket_structure, peptide_structure
    )

    x, coords, atom2res, is_protein, is_peptide = build_node_features_and_coords(
        prot_atoms,
        pep_atoms,
        protein_resid_map,
        peptide_resid_map,
    )

    nb_atom_graph = build_nb_atom_graph(
        prot_atoms=prot_atoms,
        pep_atoms=pep_atoms,
        coords=coords,
        x=x,
        atom2res=atom2res,
        is_protein=is_protein,
        is_peptide=is_peptide,
        cutoff=cutoff,
        complex_name=complex_name,
    )

    bd_atom_graph, bond_to_node, neighbors = build_bd_atom_graph(
        pocket_structure=pocket_structure,
        peptide_structure=peptide_structure,
        prot_atoms=prot_atoms,
        pep_atoms=pep_atoms,
        prot_old_to_new=prot_old_to_new,
        pep_old_to_new=pep_old_to_new,
        x=x,
        coords=coords,
        atom2res=atom2res,
        is_protein=is_protein,
        is_peptide=is_peptide,
        complex_name=complex_name,
    )

    atom_graph = build_atom_graph(
        nb_graph=nb_atom_graph,
        bd_graph=bd_atom_graph,
    )

    bond_bond_graph, angle_face, angle_to_node = build_bond_bond_graph(
        bd_atom_graph=bd_atom_graph,
        neighbors=neighbors,
        bond_to_node=bond_to_node,
    )

    bd_atom_graph.face = angle_face

    angle_angle_graph = build_angle_angle_graph(
        bond_bond_graph=bond_bond_graph,
        bd_atom_graph=bd_atom_graph,
        neighbors=neighbors,
        angle_to_node=angle_to_node,
    )

    res_graph = build_res_graph(
        coords=coords,
        atom2res=atom2res,
        is_protein=is_protein,
        is_peptide=is_peptide,
        num_res=num_res,
        esm_res=esm_res,
        cutoff=cutoff,
        complex_name=complex_name,
    )

    label = None
    if (
        native_peptide_pdb is not None
        and tau_atom is not None
        and tau_res is not None
        and tau_global is not None
    ):
        label = build_labels(
            receptor_pdb=protein_pdb,
            native_peptide_pdb=native_peptide_pdb,
            decoy_peptide_pdb=peptide_pdb,
            atom_graph=atom_graph,
            res_graph=res_graph,
            tau_atom=tau_atom,
            tau_res=tau_res,
            tau_global=tau_global,
            contact_threshold=contact_threshold,
        )
        label["complex_name"] = complex_name

    return {
        "atom_graph": atom_graph,
        "nba_graph": nb_atom_graph,
        "bda_graph": bd_atom_graph,
        "bb_graph": bond_bond_graph,
        "aa_graph": angle_angle_graph,
        "res_graph": res_graph,
        "label": label,
    }
