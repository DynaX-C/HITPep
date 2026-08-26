from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max, scatter_add
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.dense.linear import Linear
from .hit_encoder import _rbf

class ResidueModule(nn.Module):
    """
    Final residue-level module for HITPep.

    Design
    ------
    1. Atom-level consistency scores are used to gate atom representations
       BEFORE atom -> residue aggregation.

    2. Residue representations are built from:
        - gated atom-level fused trunk
        - explicit multi-branch residue signals:
            * x_nb_res : interface / non-bonded signal
            * x_bd_res : bonded / covalent signal
            * x_bb_res : angle-level geometry
            * x_aa_res : torsional geometry
        - optional ESM residue embedding

    3. Residue-level context is modeled by GATv2 with edge features
       (e.g. distance RBF + edge type one-hot).

    Outputs
    -------
    x_res : [N_res, d]
        Residue-level contextual representation.
    geom_score : [N_res]
        Residue-level geometry restoration score.
    int_score : [N_res]
        Residue-level interaction restoration score.
    """

    def __init__(
        self,
        hidden_channels: int,
        edge_dim: int = 16,
        use_esm: bool = False,
        esm_dim: int = 1280,
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
        num_res_layers: int = 2,
        use_orig_emb: bool = True,
        use_atom_gate: bool = True,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.use_esm = use_esm
        self.edge_dim = edge_dim
        self.esm_dim = esm_dim
        self.gat_heads = gat_heads
        self.gat_concat = gat_concat
        self.dist_cutoff = dist_cutoff
        self.use_orig_emb = use_orig_emb
        self.use_atom_gate = use_atom_gate

        if use_orig_emb:
            self.lin_struct = nn.Sequential(
                Linear(hidden_channels * 5, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.SiLU(),
                nn.Dropout(dropout),
                Linear(hidden_channels, hidden_channels),
            )
        else:
            self.lin_struct = nn.Sequential(
                Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.SiLU(),
                nn.Dropout(dropout),
                Linear(hidden_channels, hidden_channels),
            )


        # ===== optional ESM projection =====
        if self.use_esm:
            self.lin_esm = nn.Sequential(
                Linear(esm_dim, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.SiLU(),
                nn.Dropout(dropout),
                Linear(hidden_channels, hidden_channels),
            )
            self.lin_fuse = nn.Sequential(
                Linear(hidden_channels * 2, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.SiLU(),
                nn.Dropout(dropout),
                Linear(hidden_channels, hidden_channels),
            )
        else:
            self.lin_esm = None
            self.lin_fuse = None

        # Inputs:
        # x_atom_res : hidden_channels
        # x_nb_res   : hidden_channels
        # x_bd_res   : hidden_channels
        # x_bb_res   : hidden_channels
        # x_aa_res   : hidden_channels
        # x_esm_res  : hidden_channels (optional)

        self.lin_edge = nn.Sequential(
            Linear(edge_dim + 3, hidden_channels),
            nn.SiLU(),
        )

        # ===== GATv2 configuration =====
        if gat_concat:
            if hidden_channels % gat_heads != 0:
                raise ValueError(
                    "When gat_concat=True, hidden_channels must be divisible by gat_heads."
                )
            gat_out_channels = hidden_channels // gat_heads
            gat_final_dim = gat_out_channels * gat_heads
        else:
            gat_out_channels = hidden_channels
            gat_final_dim = hidden_channels

        self.gnns = nn.ModuleList([
            GATv2Conv(
                in_channels=hidden_channels,
                out_channels=gat_out_channels,
                heads=gat_heads,
                concat=gat_concat,
                negative_slope=gat_negative_slope,
                dropout=gat_dropout,
                add_self_loops=gat_add_self_loops,
                edge_dim=hidden_channels,
                fill_value=gat_fill_value,
                bias=gat_bias,
                share_weights=gat_share_weights,
                residual=gat_residual,
            )
            for _ in range(num_res_layers)
        ])

        self.post_gats = nn.ModuleList([
            nn.Sequential(
                Linear(gat_final_dim, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                #nn.LayerNorm(hidden_channels),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_res_layers)
        ])
        
        # ===== residue heads =====
        self.geom_head = nn.Sequential(
            Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            Linear(hidden_channels, 1),
        )

        self.int_head = nn.Sequential(
            Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            Linear(hidden_channels, 1),
        )

    @staticmethod
    def _scatter_mean_by_residue(
        x_atom: torch.Tensor,
        atom2res: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean-pool atom features to residue space.

        Parameters
        ----------
        x_atom : [N_atom, d]
        atom2res : [N_atom]

        Returns
        -------
        x_res : [N_res, d]
        """
        num_res = int(atom2res.max().item()) + 1
        device = x_atom.device

        x_res = torch.zeros(
            num_res,
            x_atom.size(-1),
            device=device,
            dtype=x_atom.dtype,
        )
        x_res.index_add_(0, atom2res, x_atom)

        counts = torch.bincount(atom2res, minlength=num_res).clamp(min=1)
        counts = counts.to(x_atom.dtype).unsqueeze(-1)

        x_res = x_res / counts
        return x_res

    @staticmethod
    def _scatter_add_max_by_residue(x_atom, atom2res):
        x_add = scatter_add(x_atom, atom2res, dim=0)
        x_max, _ = scatter_max(x_atom, atom2res, dim=0)
        return torch.cat([x_add, x_max], dim=-1)

    def forward(
        self,
        x_atom: torch.Tensor,
        atom_score: torch.Tensor,
        x_nba: torch.Tensor,
        x_bda: torch.Tensor,
        x_bb_atom: torch.Tensor,
        x_aa_atom: torch.Tensor,
        atom2res: torch.Tensor,
        res_edge_index: torch.Tensor,
        res_edge_attr: Optional[torch.Tensor] = None,
        res_edge_type: Optional[torch.Tensor] = None,
        esm_res: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x_atom_refined : [N_atom, d]
            Fused + refined atom-level representation.
        atom_score : [N_atom]
            Atom-level local consistency score in [0, 1].
        x_nba : [N_atom, d]
            Non-bonded atom features from HITEncoder.
        x_bda : [N_atom, d]
            Bonded atom features from HITEncoder.
        x_bb_atom : [N_atom, d]
            BB features projected to atom space.
        x_aa_atom : [N_atom, d]
            AA features projected to atom space.
        atom2res : [N_atom]
            Atom-to-residue mapping.
        res_edge_index : [2, E_res]
            Residue-level graph edges.
        res_edge_attr : [E_res, 1], optional
            Residue-level edge features.
        res_edge_type : [E_res, 3], optional
            Residue-level edge types.
        esm_res : [N_res, esm_dim], optional
            Residue-level ESM embedding.

        Returns
        -------
        x_res : [N_res, d]
        geom_score : [N_res]
        int_score : [N_res]
        """

        # ===== 1. Atom-level gating BEFORE residue aggregation =====
        # Use atom-level consistency as confidence reweighting
        if self.use_atom_gate:
            gate = 1.0 + atom_score.detach().unsqueeze(-1)   # [N_atom, 1]
        else:
            gate = torch.ones_like(atom_score).unsqueeze(-1)

        # ===== 2. Atom -> residue pooling =====
        if self.use_orig_emb:
            x_atom_res = scatter_add(x_atom * gate, atom2res, dim=0)
            x_nb_res = scatter_add(x_nba * gate, atom2res, dim=0)
            x_bd_res = scatter_add(x_bda * gate, atom2res, dim=0)
            x_bb_res = scatter_add(x_bb_atom * gate, atom2res, dim=0)
            x_aa_res = scatter_add(x_aa_atom * gate, atom2res, dim=0)

            struct_feats = [
                x_atom_res,   # gated atom-level trunk
                x_nb_res,     # interface source
                x_bd_res,     # bonded/covalent source
                x_bb_res,     # angle-level source
                x_aa_res,     # torsional source
            ]
            x_struct = torch.cat(struct_feats, dim=-1)
            x_struct = self.lin_struct(x_struct)
        else:
            x_atom_res = scatter_add(x_atom * gate, atom2res, dim=0)
            x_struct = self.lin_struct(x_atom_res)

        # ===== 3. Optional ESM =====
        if self.use_esm:
            if esm_res is None:
                raise ValueError("use_esm=True but esm_res is None.")
            x_esm_res = self.lin_esm(esm_res)
            x_res = torch.cat([x_struct, x_esm_res], dim=-1)
            x_res = self.lin_fuse(x_res)
        else:
            x_res = x_struct

        # ===== 4. Residue-level context =====
        device = x_res.device
        edge_rbf = _rbf(
            res_edge_attr.view(-1), D_min=0.0, D_max=self.dist_cutoff, D_count=self.edge_dim, device=device
        )
        edge_attr = torch.cat([edge_rbf, res_edge_type.float()], dim=-1)
        edge_attr = self.lin_edge(edge_attr)

        for gnn, post_gat in zip(self.gnns, self.post_gats):
            h = gnn(x_res, res_edge_index, edge_attr)
            x_res = post_gat(h) + x_res

        # ===== 5. Residue heads =====
        #h_score = self.score_trunk(x_res)

        geom_score = torch.sigmoid(self.geom_head(x_res)).squeeze(-1)
        int_score = torch.sigmoid(self.int_head(x_res)).squeeze(-1)

        return x_res, geom_score, int_score

class GlobalHead(nn.Module):
    """
    Global peptide-level quality prediction from interaction-aware residue representations.

    Design
    ------
    1. Use residue-level geom/int scores as evidence.
    2. Convert evidence into a gate.
    3. Reweight peptide residue features with this gate.
    4. Aggregate gated residue features by add + max pooling.
    5. Predict final global score from pooled peptide features.

    Inputs
    ------
    x_res : [N_res, d]
        Residue contextual embedding.
    res_geom : [N_res]
        Residue-level geometry score in [0, 1].
    res_int : [N_res]
        Residue-level interaction score in [0, 1].
    peptide_res_mask : [N_res] bool
        Mask for peptide residues.
    batch : [N_res]
        Batch index of residues.

    Output
    ------
    global_score : [B]
        Final global score in [0, 1].
    """

    def __init__(
        self,
        hidden_channels: int,
        dropout: float = 0.1,
        use_res_gate: bool = True,
    ):
        super().__init__()
        self.use_res_gate = use_res_gate

        self.global_geom_head = nn.Sequential(
            Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            Linear(hidden_channels, 1),
        )

        self.global_int_head = nn.Sequential(
            Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            Linear(hidden_channels, 1),
        )

    def forward(
        self,
        x_res: torch.Tensor,
        res_geom: torch.Tensor,
        res_int: torch.Tensor,
        peptide_res_mask: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_res : [N_res, d]
        res_geom : [N_res]
        res_int : [N_res]
        peptide_res_mask : [N_res]
        batch : [N_res]

        Returns
        -------
        global_score : [B]
        """
        # ===== peptide residues only =====
        x_pep = x_res[peptide_res_mask]                    # [N_pep_res, d]
        geom_pep = res_geom[peptide_res_mask]             # [N_pep_res]
        int_pep = res_int[peptide_res_mask]               # [N_pep_res]
        batch_pep = batch[peptide_res_mask]               # [N_pep_res]

        if self.use_res_gate:
            gate_geom = 1.0 + geom_pep.detach().unsqueeze(-1)
            gate_int = 1.0 + int_pep.detach().unsqueeze(-1)
        else:
            gate_geom = torch.ones_like(geom_pep).unsqueeze(-1)
            gate_int = torch.ones_like(int_pep).unsqueeze(-1)
        
        # geom branch
        x_pep_geom = x_pep * gate_geom
        pep_add_geom = global_add_pool(x_pep_geom, batch_pep)
        
        # int branch
        x_pep_int = x_pep * gate_int
        pep_add_int = global_add_pool(x_pep_int, batch_pep)
        
        global_geom = torch.sigmoid(self.global_geom_head(pep_add_geom)).squeeze(-1)
        global_int  = torch.sigmoid(self.global_int_head(pep_add_int)).squeeze(-1)

        return global_geom, global_int
