from typing import Tuple
import torch
import torch.nn as nn
from torch_geometric.nn.dense.linear import Linear

from .hit_encoder import _GINEBlock, _rbf


class AtomFusion(nn.Module):
    """
    Fuse hierarchical HIT representations into atom-level features.

    This module explicitly propagates higher-order geometric signals:
        - bond-level (BB) → atoms
        - angle-level (AA) → atoms

    ensuring that all structural constraints are grounded at the atomic level.
    """

    def __init__(
            self, 
            hidden_channels: int, 
            dropout: float = 0.1, 
            use_deg_norm: bool = True
            ):
        super().__init__()

        self.use_deg_norm = use_deg_norm

        self.lin_fuse = nn.Sequential(
            Linear(hidden_channels * 4, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            Linear(hidden_channels, hidden_channels),
        )

    def bond_to_atom_message(
            self, 
            bond_x, 
            bond2atom_index, 
            num_nodes
            ):
        """
        bond_x: [N_bond, d]
        bond2atom_index: [2, N_bond]
        """
        row, col = bond2atom_index

        msg = torch.zeros(num_nodes, bond_x.size(-1), device=bond_x.device)
        msg.index_add_(0, row, bond_x)
        msg.index_add_(0, col, bond_x)

        if self.use_deg_norm:
            deg = torch.zeros(num_nodes, device=bond_x.device)
            one = torch.ones_like(row, dtype=deg.dtype)

            deg.index_add_(0, row, one)
            deg.index_add_(0, col, one)

            msg = msg / deg.clamp(min=1).unsqueeze(-1)

        return msg

    def angle_to_atom_message(
            self, 
            angle_x, 
            angle2atom_index, 
            num_nodes
            ):
        """
        angle_x: [N_angle, d]
        angle2atom_index: [3, N_angle]
        """
        a, b, c = angle2atom_index

        msg = torch.zeros(num_nodes, angle_x.size(-1), device=angle_x.device)
        msg.index_add_(0, a, angle_x)
        msg.index_add_(0, b, angle_x)
        msg.index_add_(0, c, angle_x)

        if self.use_deg_norm:
            deg = torch.zeros(num_nodes, device=angle_x.device)
            one = torch.ones_like(a, dtype=deg.dtype)

            deg.index_add_(0, a, one)
            deg.index_add_(0, b, one)
            deg.index_add_(0, c, one)

            msg = msg / deg.clamp(min=1).unsqueeze(-1)

        return msg

    def forward(
        self,
        x_nba,
        x_bda,
        x_bb,
        x_aa,
        bond2atom_index,
        angle2atom_index,
    ):
        num_atoms = x_nba.size(0)

        # ===== higher-order → atom =====
        x_bb_atom = self.bond_to_atom_message(x_bb, bond2atom_index, num_atoms)
        x_aa_atom = self.angle_to_atom_message(x_aa, angle2atom_index, num_atoms)

        # ===== fusion =====
        x_atom = torch.cat([x_nba, x_bda, x_bb_atom, x_aa_atom], dim=-1)
        x_atom = self.lin_fuse(x_atom)

        return x_atom, x_bb_atom, x_aa_atom

class AtomRefinement(nn.Module):
    """
    Light atom-level refinement after HIT fusion.

    This module performs one additional local message-passing step on the fused
    atom representation, encouraging local physical consistency before atom-level
    prediction and residue aggregation.
    """

    def __init__(
        self,
        hidden_channels: int,
        edge_dim: int,
        dropout: float = 0.1,
        eps: float = 0.0,
        train_eps: bool = False,
        dist_cutoff: float = 6.0,
        num_layers: int = 2,
        refinement: bool = False,
    ):
        super().__init__()

        self.dist_cutoff = dist_cutoff
        self.edge_dim = edge_dim
        self.refinement = refinement

        self.lin_edge = nn.Sequential(
            Linear(edge_dim + 3, hidden_channels),
            nn.SiLU(),
        )

        self.gnns = nn.ModuleList([
            _GINEBlock(
                hidden_channels=hidden_channels,
                dropout=dropout,
                eps=eps,
                train_eps=train_eps,
            )
            for _ in range(num_layers)
        ])

        self.head = nn.Sequential(
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
        x_atom: torch.Tensor,
        refine_edge_index: torch.Tensor,
        refine_edge_attr: torch.Tensor,
        refine_edge_type: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x_atom : [N_atom, d]
        refine_edge_index : [2, E]
        refine_edge_attr : [E, 1]
        refine_edge_type : [E, 3]

        Returns
        -------
        x_atom_refined : [N_atom, d]
        atom_score : [N_atom]
            Atom-level local consistency score.
        """
        device = x_atom.device
        edge_rbf = _rbf(
            refine_edge_attr.view(-1), D_min=0.0, D_max=self.dist_cutoff, D_count=self.edge_dim, device=device
        )
        edge_attr = torch.cat([edge_rbf, refine_edge_type.float()], dim=-1)
        edge_attr = self.lin_edge(edge_attr)

        if self.refinement:
            for layer in self.gnns:
                x_atom = layer(x_atom, refine_edge_index, edge_attr) + x_atom
        atom_score = torch.sigmoid(self.head(x_atom)).squeeze(-1)

        return x_atom, atom_score


