from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F


def atom_loss(
    atom_score: torch.Tensor,
    atom_label: torch.Tensor,
    peptide_atom_mask: Optional[torch.Tensor] = None,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Atom-level local consistency loss.

    Parameters
    ----------
    atom_score : [N_atom]
        Predicted atom-level score in [0, 1].
    atom_label : [N_atom]
        Target atom-level label in [0, 1].
    peptide_atom_mask : [N_atom], optional
        Boolean mask selecting peptide atoms only.
    beta : float
        Huber transition point for smooth_l1_loss.

    Returns
    -------
    loss : scalar
    """
    if peptide_atom_mask is not None:
        atom_score = atom_score[peptide_atom_mask]
        atom_label = atom_label[peptide_atom_mask]

    # return F.smooth_l1_loss(atom_score, atom_label, beta=beta)
    return F.mse_loss(atom_score, atom_label)


def residue_geom_loss(
    res_geom: torch.Tensor,
    geom_label: torch.Tensor,
    peptide_res_mask: Optional[torch.Tensor] = None,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Residue-level geometry restoration loss.

    Parameters
    ----------
    res_geom : [N_res]
        Predicted residue geometry score in [0, 1].
    geom_label : [N_res]
        Target residue geometry label in [0, 1].
    peptide_res_mask : [N_res], optional
        Boolean mask selecting peptide residues only.
    beta : float
        Huber transition point.

    Returns
    -------
    loss : scalar
    """
    if peptide_res_mask is not None:
        res_geom = res_geom[peptide_res_mask]
        geom_label = geom_label[peptide_res_mask]

    # return F.smooth_l1_loss(res_geom, geom_label, beta=beta)
    return F.mse_loss(res_geom, geom_label)


def residue_int_loss(
    res_int: torch.Tensor,
    int_label: torch.Tensor,
    peptide_res_mask: Optional[torch.Tensor] = None,
    use_bce: bool = False,
) -> torch.Tensor:
    """
    Residue-level interaction restoration loss.

    Parameters
    ----------
    res_int : [N_res]
        Predicted residue interaction score in [0, 1].
    int_label : [N_res]
        Target residue interaction label in [0, 1].
        Typically continuous residue-level FNAT.
    peptide_res_mask : [N_res], optional
        Boolean mask selecting peptide residues only.
    use_bce : bool
        If True, use BCE.
        If False, use MSE (recommended for continuous labels).

    Returns
    -------
    loss : scalar
    """
    if peptide_res_mask is not None:
        res_int = res_int[peptide_res_mask]
        int_label = int_label[peptide_res_mask]

    if use_bce:
        return F.binary_cross_entropy(res_int, int_label)
    return F.mse_loss(res_int, int_label)

def global_geom_loss(
    global_geom: torch.Tensor,
    geom_label: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(global_geom, geom_label)

def global_int_loss(
    global_int: torch.Tensor,
    int_label: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(global_int, int_label)

# def global_loss(
#     global_score: torch.Tensor,
#     global_label: torch.Tensor,
#     beta: float = 0.1,
# ) -> torch.Tensor:
#     """
#     Global peptide-level quality loss.

#     Parameters
#     ----------
#     global_score : scalar or [B]
#         Predicted global score in [0, 1].
#     global_label : scalar or [B]
#         Target global score in [0, 1].
#     beta : float
#         Huber transition point.

#     Returns
#     -------
#     loss : scalar
#     """
#     return F.smooth_l1_loss(global_score, global_label, beta=beta)


def total_loss(
    out: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    weights: Optional[Dict[str, float]] = None,
    beta_atom: float = 0.1,
    beta_geom: float = 0.1,
    beta_global: float = 0.1,
    use_bce_for_int: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Total HITPep loss.

    Expected keys in out
    --------------------
    out["atom_score"]   : [N_atom]
    out["res_geom"]     : [N_res]
    out["res_int"]      : [N_res]
    out["global_score"] : scalar or [B]

    Expected keys in label
    ----------------------
    label["atom"]       : [N_atom]
    label["res_geom"]   : [N_res]
    label["res_int"]    : [N_res]
    label["global"]     : scalar or [B]

    Optional mask keys in label
    ---------------------------
    label["peptide_mask_atom"] : [N_atom] bool
    label["peptide_mask_res"]  : [N_res] bool

    Default weights
    ---------------
    atom   : 1.0
    geom   : 1.0
    int    : 1.0
    global : 1.0

    Returns
    -------
    total : scalar
    loss_dict : dict
        Contains tensor losses for logging.
    """
    if weights is None:
        weights = {
            "atom": 1.0,
            "res_geom": 1.0,
            "res_int": 1.0,
            "glb_geom": 1.0,
            "glb_int": 1.0,
        }

    peptide_atom_mask = label.get("peptide_mask_atom", None)
    peptide_res_mask = label.get("peptide_mask_res", None)

    loss_a = atom_loss(
        atom_score=out["atom_score"],
        atom_label=label["atom"],
        peptide_atom_mask=peptide_atom_mask,
        #beta=beta_atom,
    )

    loss_g = residue_geom_loss(
        res_geom=out["res_geom"],
        geom_label=label["res_geom"],
        peptide_res_mask=peptide_res_mask,
        #beta=beta_geom,
    )

    loss_i = residue_int_loss(
        res_int=out["res_int"],
        int_label=label["res_int"],
        peptide_res_mask=peptide_res_mask,
        use_bce=use_bce_for_int,
    )

    # loss_glb = global_loss(
    #     global_score=out["global_score"],
    #     global_label=label["global"],
    #     beta=beta_global,
    # )
    loss_glb_geom = global_geom_loss(
        global_geom=out["global_geom"],
        geom_label=label["global_geom"],
    )

    loss_glb_int = global_int_loss(
        global_int=out["global_int"],
        int_label=label["global_int"],
    )

    total = (
        weights["atom"] * loss_a
        + weights["res_geom"] * loss_g
        + weights["res_int"] * loss_i
        + weights["glb_geom"] * loss_glb_geom
        + weights["glb_int"] * loss_glb_int
    )

    loss_dict = {
        "loss_total": total.detach(),
        "loss_atom": loss_a.detach(),
        "loss_res_geom": loss_g.detach(),
        "loss_res_int": loss_i.detach(),
        "loss_glb_geom": loss_glb_geom.detach(),
        "loss_glb_int": loss_glb_int.detach(),
    }

    return total, loss_dict
