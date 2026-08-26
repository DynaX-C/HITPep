import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch


class GraphDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def collate_label_list(label_list):
    if label_list is None or label_list[0] is None:
        return None
    return label_list


def collate_fn(batch):
    atom_graph_list = []
    res_offset = 0

    for x in batch:
        g = x["atom_graph"].clone()
        n_res = x["res_graph"].x.size(0)

        g.atom2res = g.atom2res + res_offset
        atom_graph_list.append(g)

        res_offset += n_res

    return {
        "atom_graph": Batch.from_data_list(atom_graph_list),
        "nba_graph": Batch.from_data_list(
            [x["nba_graph"] for x in batch]
        ),
        "bda_graph": Batch.from_data_list(
            [x["bda_graph"] for x in batch]
        ),
        "bb_graph": Batch.from_data_list(
            [x["bb_graph"] for x in batch]
        ),
        "aa_graph": Batch.from_data_list(
            [x["aa_graph"] for x in batch]
        ),
        "res_graph": Batch.from_data_list(
            [x["res_graph"] for x in batch]
        ),
        "label": collate_label_list([x["label"] for x in batch]),
    }


def build_dataloader(
    data_list,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    drop_last=False,
):
    dataset = GraphDataset(data_list)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )