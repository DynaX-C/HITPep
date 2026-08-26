from typing import Dict, Optional
import torch
import torch.nn as nn

from .hit_encoder import HITEncoder, FlatAtomEncoder
from .atom_modules import AtomFusion, AtomRefinement
from .residue_modules import ResidueModule, GlobalHead


class HITPepModel(nn.Module):
    """
    HITPep main model.

    Pipeline
    --------
    1. HITEncoder:
        - encode hierarchical interaction representations
          (NBA / BDA / BB / AA)

    2. AtomFusion:
        - project BB and AA back to atom space
        - fuse NBA / BDA / BB / AA into unified atom representations

    3. AtomRefinement:
        - one light local message passing step
        - atom-level local consistency prediction

    4. ResidueModule:
        - atom-level gating before residue aggregation
        - build residue representations from:
            * gated atom trunk
            * NB / BDA / BB / AA residue-level signals
            * optional ESM
        - residue-level GATv2 context modeling
        - residue-level geometry / interaction prediction

    5. GlobalHead:
        - peptide-level global quality prediction
    """

    def __init__(
        self,
        node_features_dim: int,
        hidden_channels: int,
        edge_dim: int = 16,
        num_layers_nba: int = 2,
        num_layers_bda: int = 2,
        num_layers_bb: int = 2,
        num_layers_aa: int = 2,
        # gine
        eps: float = 0.0,
        train_eps: bool = False,
        # gatv2
        gat_heads: int = 4,
        gat_concat: bool = True,
        gat_negative_slope: float = 0.2,
        gat_dropout: float = 0.1,
        gat_add_self_loops: bool = True,
        gat_fill_value: str = "mean",
        gat_bias: bool = True,
        gat_share_weights: bool = False,
        gat_residual: bool = True,    
        dropout: float = 0.1,
        dist_cutoff: float = 6.0,
        residue_edge_dim: int = 16,
        use_esm: bool = True,
        esm_dim: int = 1280,
        use_orig_emb: bool = True,
        use_hit: bool = True,
        use_atom_gate: bool = True,
        use_res_gate: bool = True,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.edge_dim = edge_dim
        self.use_esm = use_esm
        self.use_hit = use_hit

        # ===== backbone =====
        if use_hit:
            self.encoder = HITEncoder(
                node_features_dim=node_features_dim,
                hidden_channels=hidden_channels,
                edge_dim=edge_dim,
                num_layers_nba=num_layers_nba,
                num_layers_bda=num_layers_bda,
                num_layers_bb=num_layers_bb,
                num_layers_aa=num_layers_aa,
                eps=eps,
                train_eps=train_eps,
                dropout=dropout,
                dist_cutoff=dist_cutoff,
            )
        else:
            self.encoder = FlatAtomEncoder(
                node_features_dim=node_features_dim,
                hidden_channels=hidden_channels,
                edge_dim=edge_dim,
                num_layers=num_layers_nba,
                eps=eps,
                train_eps=train_eps,
                dropout=dropout,
                dist_cutoff=dist_cutoff,
            )
            use_orig_emb = False

        # ===== atom-level =====
        self.atom_fusion = AtomFusion(
            hidden_channels=hidden_channels,
            dropout=dropout,
            use_deg_norm=True,
        )

        self.atom_refinement = AtomRefinement(
            hidden_channels=hidden_channels,
            edge_dim=edge_dim,
            dropout=dropout,
        )

        # ===== residue-level =====
        self.residue_module = ResidueModule(
            hidden_channels=hidden_channels,
            edge_dim=residue_edge_dim,
            use_esm=use_esm,
            esm_dim=esm_dim,
            gat_heads=gat_heads,
            gat_concat=gat_concat,
            gat_negative_slope=gat_negative_slope,
            gat_dropout=gat_dropout,
            gat_add_self_loops=gat_add_self_loops,
            gat_fill_value=gat_fill_value,
            gat_bias=gat_bias,
            gat_share_weights=gat_share_weights,
            gat_residual=gat_residual,
            dropout=dropout,
            dist_cutoff=dist_cutoff,
            use_orig_emb=use_orig_emb,
            use_atom_gate=use_atom_gate,
        )

        # ===== global =====
        self.global_head = GlobalHead(
            hidden_channels=hidden_channels,
            dropout=dropout,
            use_res_gate=use_res_gate,
        )

    def forward(self, data: Dict) -> Dict[str, torch.Tensor]:
        """
        Required keys in data
        ---------------------
        Backbone inputs:
            data["nb_atom_graph"]
            data["bd_atom_graph"]
            data["bond_bond_graph"]
            data["angle_angle_graph"]

        Higher-order -> atom mappings:
            data["bond2atom_index"]   : [2, N_bond_node]
            data["angle2atom_index"]  : [3, N_angle_node]

        Atom refinement graph:
            data["refine_edge_index"] : [2, E_refine]
            data["refine_edge_attr"]  : [E_refine, edge_dim]

        Residue graph:
            data["atom2res"]          : [N_atom]
            data["res_edge_index"]    : [2, E_res]
            data["res_edge_attr"]     : optional [E_res, residue_edge_dim]

        Masks:
            data["peptide_mask_atom"] : [N_atom] bool, optional
            data["peptide_mask_res"]  : [N_res] bool

        Optional:
            data["esm_res"]           : [N_res, esm_dim]

        Returns
        -------
        out : dict with
            "x_atom_refined" : [N_atom, d]
            "atom_score"     : [N_atom]
            "x_res"          : [N_res, d]
            "res_geom"       : [N_res]
            "res_int"        : [N_res]
            "global_score"   : scalar
        """

        if self.use_hit:
            # ==========================================================
            # 1. HIT hierarchical encoding
            # ==========================================================
            x_nba, x_bda, x_bb, x_aa = self.encoder(data)

            # ==========================================================
            # 2. Atom fusion
            # ==========================================================
            x_atom, x_bb_atom, x_aa_atom = self.atom_fusion(
                x_nba=x_nba,
                x_bda=x_bda,
                x_bb=x_bb,
                x_aa=x_aa,
                bond2atom_index=data["bda_graph"].edge_index,
                angle2atom_index=data["bda_graph"].face,
            )
        else:
            x_atom = self.encoder(data)
            x_nba=None
            x_bda=None
            x_bb_atom=None
            x_aa_atom=None

        # ==========================================================
        # 3. atom prediction
        # ==========================================================
        x_atom_refined, atom_score = self.atom_refinement(
            x_atom=x_atom,
            refine_edge_index=data["atom_graph"].edge_index,
            refine_edge_attr=data["atom_graph"].edge_attr,
            refine_edge_type=data["atom_graph"].edge_type if "edge_type" in data["atom_graph"] else None,
        )

        # ==========================================================
        # 4. Residue-level module
        # ==========================================================
        esm_res = data["res_graph"].x if self.use_esm else None
        res_edge_attr = data["res_graph"].edge_attr

        x_res, res_geom, res_int = self.residue_module(
            x_atom=x_atom,
            atom_score=atom_score,
            x_nba=x_nba,
            x_bda=x_bda,
            x_bb_atom=x_bb_atom,
            x_aa_atom=x_aa_atom,
            atom2res=data["atom_graph"].atom2res,
            res_edge_index=data["res_graph"].edge_index,
            res_edge_attr=res_edge_attr,
            res_edge_type=data["res_graph"].edge_type if "edge_type" in data["res_graph"] else None,
            esm_res=esm_res,
        )

        # ==========================================================
        # 5. Global prediction
        # ==========================================================
        global_geom, global_int = self.global_head(
            x_res=x_res,
            res_geom=res_geom,
            res_int=res_int,
            peptide_res_mask=data["res_graph"].peptide_mask,
            batch=data["res_graph"].batch,
        )

        out = {
            "x_atom": x_atom,
            "atom_score": atom_score,
            "x_res": x_res,
            "res_geom": res_geom,
            "res_int": res_int,
            "global_geom": global_geom,
            "global_int": global_int,
        }

        out["atom_score_peptide"] = atom_score[data["atom_graph"].is_peptide.bool()]
        out["res_geom_peptide"] = res_geom[data["res_graph"].peptide_mask]
        out["res_int_peptide"] = res_int[data["res_graph"].peptide_mask]

        return out
