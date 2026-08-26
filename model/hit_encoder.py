from typing import Tuple
import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv
from torch_geometric.nn.dense.linear import Linear


def _rbf(D, D_min=0.0, D_max=20.0, D_count=16, device="cpu"):
    """
    Radial basis expansion for scalar distances.
    D: [E]
    return: [E, D_count]
    """
    D_mu = torch.linspace(D_min, D_max, D_count, device=device).view(1, -1)
    D_sigma = (D_max - D_min) / D_count
    D_expand = D.unsqueeze(-1)
    return torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)


def _fourier_angle(angle: torch.Tensor, num_freq: int) -> torch.Tensor:
    """
    Fourier encoding for angle/dihedral.
    angle: [E]
    return: [E, 2 * num_freq]
    """
    feats = []
    for n in range(1, num_freq + 1):
        feats.append(torch.cos(n * angle))
        feats.append(torch.sin(n * angle))
    return torch.stack(feats, dim=-1)


class _GINEBlock(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        dropout: float = 0.1,
        eps: float = 0.0,
        train_eps: bool = False,
    ):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            #nn.LayerNorm(hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            #nn.Linear(hidden_channels, hidden_channels),
        )
        self.conv = GINEConv(
            nn=mlp,
            eps=eps,
            train_eps=train_eps,
            edge_dim=hidden_channels,
        )

    def forward(self, x, edge_index, edge_attr):
        return self.conv(x, edge_index, edge_attr)


class HITEncoder(nn.Module):
    """
    HIT Encoder.

    Hierarchical representations:
        x_nba: nonbonded atom graph node embedding (interface interactions)
        x_bda: bonded atom graph node embedding (covalent connectivity)
        x_bb : bond-bond graph node embedding (captures angle-level relations)
        x_aa : angle-angle graph node embedding (captures higher-order torsional relations)

    Notes
    -----
    - x_nba and x_bda are independent atom-level branches.
    - x_bb nodes correspond to bond-level nodes.
    - x_aa nodes correspond to angle-level nodes.
    - The AA graph uses dihedral-related edge features, but its nodes are still angle-angle graph nodes.
    """

    def __init__(
        self,
        node_features_dim: int,
        hidden_channels: int,
        edge_dim: int,
        num_layers_nba: int = 2,
        num_layers_bda: int = 2,
        num_layers_bb: int = 2,
        num_layers_aa: int = 2,
        eps: float = 0.0,
        train_eps: bool = False,
        dropout: float = 0.1,
        dist_cutoff: float = 6.0,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.edge_dim = edge_dim
        self.dist_cutoff = dist_cutoff

        # ===== node encoders =====
        self.lin_nba_node = nn.Sequential(
            Linear(node_features_dim, hidden_channels),
            nn.SiLU(),
        )
        self.lin_bda_node = nn.Sequential(
            Linear(node_features_dim, hidden_channels),
            nn.SiLU(),
        )

        # BB node initializer: [atom_pair_feature + bond_distance_rbf]
        self.lin_bb_node = nn.Sequential(
            Linear(node_features_dim + edge_dim, hidden_channels),
            nn.SiLU(),
        )

        # AA node initializer: [bb_pair_feature + angle_fourier]
        self.lin_aa_node = nn.Sequential(
            Linear((node_features_dim + edge_dim) + edge_dim, hidden_channels),
            nn.SiLU(),
        )

        # ===== edge encoders =====
        self.lin_nba_edge = nn.Sequential(
            nn.Linear(edge_dim, hidden_channels),
            nn.SiLU(),
        )
        self.lin_bda_edge = nn.Sequential(
            nn.Linear(edge_dim, hidden_channels),
            nn.SiLU(),
        )
        self.lin_bb_edge = nn.Sequential(
            nn.Linear(edge_dim, hidden_channels),
            nn.SiLU(),
        )
        self.lin_aa_edge = nn.Sequential(
            nn.Linear(edge_dim, hidden_channels),
            nn.SiLU(),
        )

        # ===== graph encoders =====
        self.nba_convs = nn.ModuleList([
            _GINEBlock(hidden_channels, dropout, eps, train_eps)
            for _ in range(num_layers_nba)
        ])
        self.bda_convs = nn.ModuleList([
            _GINEBlock(hidden_channels, dropout, eps, train_eps)
            for _ in range(num_layers_bda)
        ])
        self.bb_convs = nn.ModuleList([
            _GINEBlock(hidden_channels, dropout, eps, train_eps)
            for _ in range(num_layers_bb)
        ])
        self.aa_convs = nn.ModuleList([
            _GINEBlock(hidden_channels, dropout, eps, train_eps)
            for _ in range(num_layers_aa)
        ])

    def forward(
        self, data
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Expected fields in data:
            data['nba_graph'].x
            data['nba_graph'].edge_index
            data['nba_graph'].edge_attr

            data['bda_graph'].x
            data['bda_graph'].edge_index
            data['bda_graph'].edge_attr

            data['bb_graph'].edge_index
            data['bb_graph'].edge_attr

            data['aa_graph'].edge_index
            data['aa_graph'].edge_attr

        Returns
        -------
        x_nba : [N_atom, d]
        x_bda : [N_atom, d]
        x_bb  : [N_bond, d]
        x_aa  : [N_angle, d]
        """

        # ===== raw node inputs =====
        x_atom_nba = data["nba_graph"].x
        x_atom_bda = data["bda_graph"].x

        # In most cases these should be identical
        if x_atom_nba.shape != x_atom_bda.shape:
            raise ValueError("nb_atom_graph.x and bd_atom_graph.x should have the same shape.")

        # ===== raw graph topology =====
        edge_index_nba = data["nba_graph"].edge_index      # interface nonbonded atom graph
        edge_index_bda = data["bda_graph"].edge_index      # bonded atom graph
        edge_index_bb = data["bb_graph"].edge_index          # bond-bond graph
        edge_index_aa = data["aa_graph"].edge_index         # angle-angle graph

        # ===== raw scalar edge attributes =====
        dist_nba = data["nba_graph"].edge_attr.view(-1)
        dist_bda = data["bda_graph"].edge_attr.view(-1)
        angle_bb = data["bb_graph"].edge_attr.view(-1)
        dihedral_aa = data["aa_graph"].edge_attr.view(-1)

        device = x_atom_nba.device

        # ===== edge feature construction =====
        # NBA / BDA: distance RBF
        edge_feat_nba_raw = _rbf(
            dist_nba, D_min=0.0, D_max=self.dist_cutoff, D_count=self.edge_dim, device=device
        )
        edge_feat_bda_raw = _rbf(
            dist_bda, D_min=0.0, D_max=self.dist_cutoff, D_count=self.edge_dim, device=device
        )

        # BB: angle Fourier features
        num_freq = self.edge_dim // 2
        edge_feat_bb_raw = _fourier_angle(angle_bb, num_freq=num_freq)

        # AA: dihedral Fourier features
        edge_feat_aa_raw = _fourier_angle(dihedral_aa, num_freq=num_freq)

        # ===== edge projection =====
        edge_attr_nba = self.lin_nba_edge(edge_feat_nba_raw)
        edge_attr_bda = self.lin_bda_edge(edge_feat_bda_raw)
        edge_attr_bb = self.lin_bb_edge(edge_feat_bb_raw)
        edge_attr_aa = self.lin_aa_edge(edge_feat_aa_raw)

        # ===== atom branch initialization =====
        x_nba = self.lin_nba_node(x_atom_nba)
        x_bda = self.lin_bda_node(x_atom_bda)

        # ===== BB node initialization =====
        # nodes in BB graph correspond to bonded atom pairs
        row_bda, col_bda = edge_index_bda
        bb_node_init = torch.cat(
            [
                x_atom_bda[row_bda] + x_atom_bda[col_bda],
                edge_feat_bda_raw,
            ],
            dim=-1,
        )
        x_bb = self.lin_bb_node(bb_node_init)

        # ===== AA node initialization =====
        # nodes in AA graph are built from neighboring BB nodes + local angle encoding
        row_bb, col_bb = edge_index_bb
        aa_node_init = torch.cat(
            [
                bb_node_init[row_bb] + bb_node_init[col_bb],
                edge_feat_bb_raw,
            ],
            dim=-1,
        )
        x_aa = self.lin_aa_node(aa_node_init)

        # ===== NBA branch =====
        for conv in self.nba_convs:
            x_nba = conv(x_nba, edge_index_nba, edge_attr_nba) + x_nba

        # ===== BDA branch =====
        for conv in self.bda_convs:
            x_bda = conv(x_bda, edge_index_bda, edge_attr_bda) + x_bda

        # ===== BB branch =====
        for conv in self.bb_convs:
            x_bb = conv(x_bb, edge_index_bb, edge_attr_bb) + x_bb

        # ===== AA branch =====
        for conv in self.aa_convs:
            x_aa = conv(x_aa, edge_index_aa, edge_attr_aa) + x_aa

        return x_nba, x_bda, x_bb, x_aa

class FlatAtomEncoder(nn.Module):
    """
    Non-hierarchical atom-level encoder for ablation.

    This encoder only uses data["atom_graph"], without NBA/BDA/BB/AA
    hierarchical decomposition.

    Expected fields:
        data["atom_graph"].x
        data["atom_graph"].edge_index
        data["atom_graph"].edge_attr

    Returns
    -------
    x_atom : [N_atom, hidden_channels]
    """

    def __init__(
        self,
        node_features_dim: int,
        hidden_channels: int,
        edge_dim: int,
        num_layers: int = 2,
        eps: float = 0.0,
        train_eps: bool = False,
        dropout: float = 0.1,
        dist_cutoff: float = 6.0,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.edge_dim = edge_dim
        self.dist_cutoff = dist_cutoff

        # ===== node encoder =====
        self.lin_node = nn.Sequential(
            Linear(node_features_dim, hidden_channels),
            nn.SiLU(),
        )

        # ===== edge encoder =====
        self.lin_edge = nn.Sequential(
            nn.Linear(edge_dim, hidden_channels),
            nn.SiLU(),
        )

        # ===== atom-level GINE encoder =====
        self.convs = nn.ModuleList([
            _GINEBlock(
                hidden_channels=hidden_channels,
                dropout=dropout,
                eps=eps,
                train_eps=train_eps,
            )
            for _ in range(num_layers)
        ])

    def forward(self, data) -> torch.Tensor:
        atom_graph = data["atom_graph"]

        x = atom_graph.x
        edge_index = atom_graph.edge_index
        dist = atom_graph.edge_attr.view(-1)

        device = x.device

        # distance RBF
        edge_feat_raw = _rbf(
            dist,
            D_min=0.0,
            D_max=self.dist_cutoff,
            D_count=self.edge_dim,
            device=device,
        )

        edge_attr = self.lin_edge(edge_feat_raw)

        x_atom = self.lin_node(x)

        for conv in self.convs:
            x_atom = conv(x_atom, edge_index, edge_attr) + x_atom

        return x_atom
