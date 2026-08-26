import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import torch
from tqdm import tqdm

from data.dataloader import build_dataloader
from model.hitpep_model import HITPepModel


# =========================================================
# utils
# =========================================================
def move_batch_to_device(batch: Dict[str, Any], device: torch.device):
    for key in ["atom_graph", "nba_graph", "bda_graph", "bb_graph", "aa_graph", "res_graph"]:
        batch[key] = batch[key].to(device)
    return batch


def build_model_from_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "args" not in ckpt:
        raise KeyError("Checkpoint does not contain 'args'. Cannot rebuild model automatically.")

    cfg = ckpt["args"]

    model = HITPepModel(
        node_features_dim=cfg["node_features_dim"],
        hidden_channels=cfg.get("hidden_channels", 256),
        edge_dim=cfg.get("edge_dim", 16),
        num_layers_nba=cfg.get("num_layers_nba", 2),
        num_layers_bda=cfg.get("num_layers_bda", 2),
        num_layers_bb=cfg.get("num_layers_bb", 2),
        num_layers_aa=cfg.get("num_layers_aa", 2),
        eps=cfg.get("eps", 0.0),
        train_eps=cfg.get("train_eps", False),
        gat_heads=cfg.get("gat_heads", 4),
        gat_concat=cfg.get("gat_concat", True),
        gat_negative_slope=cfg.get("gat_negative_slope", 0.1),
        gat_dropout=cfg.get("gat_dropout", 0.1),
        gat_add_self_loops=cfg.get("gat_add_self_loops", True),
        gat_fill_value=cfg.get("gat_fill_value", "mean"),
        gat_bias=cfg.get("gat_bias", True),
        gat_share_weights=cfg.get("gat_share_weights", False),
        gat_residual=cfg.get("gat_residual", True),
        dropout=cfg.get("dropout", 0.1),
        dist_cutoff=cfg.get("dist_cutoff", 6.0),
        residue_edge_dim=cfg.get("residue_edge_dim", 16),
        use_esm=cfg.get("use_esm", True),
        esm_dim=cfg.get("esm_dim", 1280),
        use_orig_emb=cfg.get("use_orig_emb", True),
        use_hit=cfg.get("use_hit", True),
        use_atom_gate = cfg.get("use_atom_gate", True),
        use_res_gate = cfg.get("use_res_gate", True),
    ).to(device)

    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, ckpt


def extract_complex_names(dataset: List[Any]) -> List[str]:
    """
    Recover per-complex names from saved graph dataset.
    """
    names = []
    for i, item in enumerate(dataset):
        name = None

        if isinstance(item, dict):
            if "complex_name" in item:
                name = item["complex_name"]

            if name is None:
                for key in ["atom_graph", "nba_graph", "bda_graph", "bb_graph", "aa_graph", "res_graph"]:
                    if key in item:
                        graph = item[key]
                        if hasattr(graph, "complex_name"):
                            name = graph.complex_name
                            break
                        if isinstance(graph, dict) and "complex_name" in graph:
                            name = graph["complex_name"]
                            break

        if name is None:
            name = f"sample_{i}"

        names.append(str(name))

    return names


def _flatten_pred(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2 and x.size(-1) == 1:
        x = x.squeeze(-1)
    elif x.dim() > 1:
        x = x[..., 0]
    return x.reshape(-1)


def _mean_pool_by_batch(x: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
    """
    x: [N] or [N, 1]
    batch_index: [N]
    return: [B]
    """
    x = _flatten_pred(x)
    batch_index = batch_index.reshape(-1)

    if batch_index.numel() == 0:
        return torch.empty(0, device=x.device, dtype=x.dtype)

    num_graphs = int(batch_index.max().item()) + 1

    out = torch.zeros(num_graphs, device=x.device, dtype=x.dtype)
    cnt = torch.zeros(num_graphs, device=x.device, dtype=x.dtype)

    out.scatter_add_(0, batch_index, x)
    cnt.scatter_add_(0, batch_index, torch.ones_like(x))

    out = out / cnt.clamp_min(1.0)
    return out

def _get_attr_or_none(obj, name):
    """
    Safely get optional attributes from PyG Batch/Data.
    """
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict) and name in obj:
        return obj[name]
    return None


def _to_cpu_list(x):
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    return list(x)

def extract_graph_level_scores(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """
    Convert multi-head outputs into one score per graph.
    """
    score_dict = {}

    # -------- global_score: already graph-level --------
    if "global_geom" not in out:
        raise KeyError(f"'global_geom' not found. Available keys: {list(out.keys())}")
    score_dict["glb_geom"] = _flatten_pred(out["global_geom"])
    if "global_int" not in out:
        raise KeyError(f"'global_int' not found. Available keys: {list(out.keys())}")
    score_dict["glb_int"] = _flatten_pred(out["global_int"])

    # -------- atom_score: prefer peptide-only branch --------
    if "atom_score_peptide" in out:
        atom_score = out["atom_score_peptide"]
        atom_batch = batch["atom_graph"].batch[batch["atom_graph"].is_peptide.bool()]
        score_dict["atom_score"] = _mean_pool_by_batch(atom_score, atom_batch)
    elif "atom_score" in out:
        atom_score = out["atom_score"][batch["atom_graph"].is_peptide.bool()]
        atom_batch = batch["atom_graph"].batch[batch["atom_graph"].is_peptide.bool()]
        score_dict["atom_score"] = _mean_pool_by_batch(atom_score, atom_batch)
    else:
        raise KeyError(f"'atom_score' not found. Available keys: {list(out.keys())}")

    # -------- res_geom: prefer peptide-only branch --------
    if "res_geom_peptide" in out:
        res_geom = out["res_geom_peptide"]
        res_batch = batch["res_graph"].batch[batch["res_graph"].peptide_mask.bool()]
        score_dict["res_geom"] = _mean_pool_by_batch(res_geom, res_batch)
    elif "res_geom" in out:
        res_geom = out["res_geom"][batch["res_graph"].peptide_mask]
        res_batch = batch["res_graph"].batch[batch["res_graph"].peptide_mask.bool()]
        score_dict["res_geom"] = _mean_pool_by_batch(res_geom, res_batch)
    else:
        raise KeyError(f"'res_geom' not found. Available keys: {list(out.keys())}")

    # -------- res_int: prefer peptide-only branch --------
    if "res_int_peptide" in out:
        res_int = out["res_int_peptide"]
        res_batch = batch["res_graph"].batch[batch["res_graph"].peptide_mask.bool()]
        score_dict["res_int"] = _mean_pool_by_batch(res_int, res_batch)
    elif "res_int" in out:
        res_int = out["res_int"][batch["res_graph"].peptide_mask]
        res_batch = batch["res_graph"].batch[batch["res_graph"].peptide_mask.bool()]
        score_dict["res_int"] = _mean_pool_by_batch(res_int, res_batch)
    else:
        raise KeyError(f"'res_int' not found. Available keys: {list(out.keys())}")

    return score_dict

def extract_atom_residue_level_scores(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    batch_names: List[str],
) -> tuple[list, list]:
    """
    Extract per-peptide-atom and per-peptide-residue scores.

    Output:
        atom_rows: one row per peptide atom
        res_rows: one row per peptide residue
    """

    atom_rows = []
    res_rows = []

    # =====================================================
    # Atom-level scores
    # =====================================================
    atom_graph = batch["atom_graph"]
    atom_mask = atom_graph.is_peptide.bool()
    atom_batch = atom_graph.batch[atom_mask]

    if "atom_score_peptide" in out:
        atom_score = _flatten_pred(out["atom_score_peptide"])
    elif "atom_score" in out:
        atom_score = _flatten_pred(out["atom_score"])[atom_mask]
    else:
        raise KeyError(f"'atom_score' not found. Available keys: {list(out.keys())}")

    # atom local index in each graph
    atom_global_idx = torch.arange(
        atom_graph.num_nodes,
        device=atom_score.device,
    )[atom_mask]

    if hasattr(atom_graph, "ptr"):
        atom_ptr = atom_graph.ptr.to(atom_global_idx.device)
        atom_local_idx = atom_global_idx - atom_ptr[atom_batch]
    else:
        atom_local_idx = atom_global_idx

    # optional attributes
    atom_name = _get_attr_or_none(atom_graph, "atom_name")
    res_name_atom = _get_attr_or_none(atom_graph, "res_name")
    resid_atom = _get_attr_or_none(atom_graph, "resid")

    atom_name_list = _to_cpu_list(atom_name) if atom_name is not None else None
    res_name_atom_list = _to_cpu_list(res_name_atom) if res_name_atom is not None else None
    resid_atom_list = _to_cpu_list(resid_atom) if resid_atom is not None else None

    atom_global_idx_cpu = atom_global_idx.detach().cpu().tolist()
    atom_local_idx_cpu = atom_local_idx.detach().cpu().tolist()
    atom_batch_cpu = atom_batch.detach().cpu().tolist()
    atom_score_cpu = atom_score.detach().cpu().tolist()

    for k, score in enumerate(atom_score_cpu):
        g = int(atom_batch_cpu[k])
        global_idx = int(atom_global_idx_cpu[k])

        row = {
            "Name": batch_names[g],
            "Atom_Local_Index": int(atom_local_idx_cpu[k]),
            "Atom_Global_Index_in_Batch": global_idx,
            "Atom_Score": float(score),
        }

        if atom_name_list is not None:
            row["Atom_Name"] = atom_name_list[global_idx]
        if res_name_atom_list is not None:
            row["Residue_Name"] = res_name_atom_list[global_idx]
        if resid_atom_list is not None:
            row["Residue_ID"] = resid_atom_list[global_idx]

        atom_rows.append(row)

    # =====================================================
    # Residue-level scores
    # =====================================================
    res_graph = batch["res_graph"]
    res_mask = res_graph.peptide_mask.bool()
    res_batch = res_graph.batch[res_mask]

    if "res_geom_peptide" in out:
        res_geom = _flatten_pred(out["res_geom_peptide"])
    elif "res_geom" in out:
        res_geom = _flatten_pred(out["res_geom"])[res_mask]
    else:
        raise KeyError(f"'res_geom' not found. Available keys: {list(out.keys())}")

    if "res_int_peptide" in out:
        res_int = _flatten_pred(out["res_int_peptide"])
    elif "res_int" in out:
        res_int = _flatten_pred(out["res_int"])[res_mask]
    else:
        raise KeyError(f"'res_int' not found. Available keys: {list(out.keys())}")

    res_global_idx = torch.arange(
        res_graph.num_nodes,
        device=res_geom.device,
    )[res_mask]

    if hasattr(res_graph, "ptr"):
        res_ptr = res_graph.ptr.to(res_global_idx.device)
        res_local_idx = res_global_idx - res_ptr[res_batch]
    else:
        res_local_idx = res_global_idx

    # optional residue attributes
    res_name = _get_attr_or_none(res_graph, "res_name")
    resid = _get_attr_or_none(res_graph, "resid")

    res_name_list = _to_cpu_list(res_name) if res_name is not None else None
    resid_list = _to_cpu_list(resid) if resid is not None else None

    res_global_idx_cpu = res_global_idx.detach().cpu().tolist()
    res_local_idx_cpu = res_local_idx.detach().cpu().tolist()
    res_batch_cpu = res_batch.detach().cpu().tolist()
    res_geom_cpu = res_geom.detach().cpu().tolist()
    res_int_cpu = res_int.detach().cpu().tolist()

    for k in range(len(res_geom_cpu)):
        g = int(res_batch_cpu[k])
        global_idx = int(res_global_idx_cpu[k])

        res_geom_score = float(res_geom_cpu[k])
        res_int_score = float(res_int_cpu[k])
        res_score = 0.5 * (res_geom_score + res_int_score)

        row = {
            "Name": batch_names[g],
            "Residue_Local_Index": int(res_local_idx_cpu[k]),
            "Residue_Global_Index_in_Batch": global_idx,
            "Res_Geom_Score": res_geom_score,
            "Res_Int_Score": res_int_score,
            "Res_Score": res_score,
        }

        if res_name_list is not None:
            row["Residue_Name"] = res_name_list[global_idx]
        if resid_list is not None:
            row["Residue_ID"] = resid_list[global_idx]

        res_rows.append(row)

    return atom_rows, res_rows

def compute_final_score(row):
    return (
        0.1 * row["Atom_Score"]
        + 0.15 * row["Res_Geom_Score"]
        + 0.15 * row["Res_Int_Score"]
        + 0.3 * row["Global_Geom_Score"]
        + 0.3 * row["Global_Int_Score"]
    )

@torch.no_grad()
def run_inference(model, loader, device: torch.device, names: List[str] = None):
    model.eval()

    all_scores = {
        "atom_score": [],
        "res_geom": [],
        "res_int": [],
        "glb_geom": [],
        "glb_int": [],
    }

    all_atom_rows = []
    all_res_rows = []

    sample_offset = 0

    pbar = tqdm(loader, desc="Inference", leave=False)

    for batch in pbar:
        batch = move_batch_to_device(batch, device)

        out = model(batch)

        # graph-level scores
        score_dict = extract_graph_level_scores(out, batch)

        for key in all_scores:
            all_scores[key].extend(score_dict[key].detach().cpu().tolist())

        # batch size
        batch_size = len(score_dict["glb_geom"])

        if names is not None:
            batch_names = names[sample_offset: sample_offset + batch_size]
        else:
            batch_names = [f"sample_{sample_offset + i}" for i in range(batch_size)]

        # atom/residue-level scores
        atom_rows, res_rows = extract_atom_residue_level_scores(
            out=out,
            batch=batch,
            batch_names=batch_names,
        )

        all_atom_rows.extend(atom_rows)
        all_res_rows.extend(res_rows)

        sample_offset += batch_size

    return all_scores, all_atom_rows, all_res_rows


def save_inference_results(
    names: List[str],
    score_dict: Dict[str, List[float]],
    save_csv: str = None,
    save_json: str = None,
):
    n = len(names)

    rows = []
    for i, name in enumerate(names):
        row = {
            "Name": name,
            "Atom_Score": round(float(score_dict["atom_score"][i]), 6),
            "Res_Geom_Score": round(float(score_dict["res_geom"][i]), 6),
            "Res_Int_Score": round(float(score_dict["res_int"][i]), 6),
            "Global_Geom_Score": round(float(score_dict["glb_geom"][i]), 6),
            "Global_Int_Score": round(float(score_dict["glb_int"][i]), 6),
        }

        row["HITPep_Score"] = round(compute_final_score(row), 6)

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("HITPep_Score", ascending=False).reset_index(drop=True)

    if save_csv is not None:
        save_csv = Path(save_csv)
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv, index=False)

    if save_json is not None:
        save_json = Path(save_json)
        save_json.parent.mkdir(parents=True, exist_ok=True)
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

    return df


# =========================================================
# main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Inference from prebuilt graph dataset")

    parser.add_argument("--graph_pt", type=str, required=True,
                        help="Graph dataset .pt generated by data_pre_workflow.py")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Trained checkpoint, e.g. best_model.pt")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--save_pred_csv", type=str, default=None)
    parser.add_argument("--save_pred_json", type=str, default=None)
    parser.add_argument("--save_atom_csv", type=str, default=None)
    parser.add_argument("--save_res_csv", type=str, default=None)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------- load graph dataset ----------
    dataset = torch.load(args.graph_pt, map_location="cpu", weights_only=False)
    print(f"[Loaded] graph dataset -> {args.graph_pt} (N={len(dataset)})")

    loader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # ---------- load model ----------
    model, ckpt = build_model_from_checkpoint(args.ckpt, device=device)
    print(f"[Loaded] checkpoint -> {args.ckpt}")

    # ---------- inference ----------
    names = extract_complex_names(dataset)
    score_dict, atom_rows, res_rows = run_inference(
        model=model,
        loader=loader,
        device=device,
        names=names,
    )

    for key, vals in score_dict.items():
        if len(names) != len(vals):
            raise RuntimeError(
                f"Prediction count mismatch for {key}: {len(names)} names vs {len(vals)} scores"
            )

    df_pred = save_inference_results(
        names=names,
        score_dict=score_dict,
        save_csv=args.save_pred_csv,
        save_json=args.save_pred_json,
    )

    if args.save_atom_csv is not None:
        atom_df = pd.DataFrame(atom_rows)
        atom_csv = Path(args.save_atom_csv)
        atom_csv.parent.mkdir(parents=True, exist_ok=True)
        atom_df.to_csv(atom_csv, index=False)
        print(f"[Saved] atom-level scores -> {atom_csv}")

    if args.save_res_csv is not None:
        res_df = pd.DataFrame(res_rows)
        res_csv = Path(args.save_res_csv)
        res_csv.parent.mkdir(parents=True, exist_ok=True)
        res_df.to_csv(res_csv, index=False)
        print(f"[Saved] residue-level scores -> {res_csv}")

    print("[Top predictions]")
    print(df_pred.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
