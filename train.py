import os
import time
import json
import random
import math
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from data.dataloader import build_dataloader
from model.hitpep_model import HITPepModel
from model.loss import total_loss


# =========================================================
# basic utils
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def move_batch_to_device(batch: Dict[str, Any], device: torch.device):
    for key in ["atom_graph", "nba_graph", "bda_graph", "bb_graph", "aa_graph", "res_graph"]:
        batch[key] = batch[key].to(device)
    return batch


def merge_label_list(label_list: List[Dict[str, torch.Tensor]], device: torch.device):
    """
    dataloader 返回 list[dict]，在训练时手动合并。
    """
    if label_list is None:
        return None

    out = {}

    node_keys = ["atom", "res_geom", "res_int", "peptide_mask_atom", "peptide_mask_res"]
    for key in node_keys:
        vals = [x[key] for x in label_list if key in x]
        if len(vals) > 0:
            out[key] = torch.cat(vals, dim=0).to(device)

    global_keys = ["global_geom", "global_int"]
    for key in global_keys:
        vals = [x[key] for x in label_list if key in x]
        if len(vals) > 0:
            out[key] = torch.stack(vals, dim=0).to(device)

    return out


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, metric: float):
        if self.best is None:
            self.best = metric
            return True

        if self.mode == "min":
            improved = metric < self.best - self.min_delta
        else:
            improved = metric > self.best + self.min_delta

        if improved:
            self.best = metric
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


# =========================================================
# one epoch
# =========================================================
def run_one_epoch(
    model,
    loader,
    device,
    optimizer=None,
    loss_weights=None,
    beta_atom=0.1,
    beta_geom=0.1,
    beta_global=0.1,
    use_bce_for_int=False,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    meter = {
        "loss_total": 0.0,
        "loss_atom": 0.0,
        "loss_res_geom": 0.0,
        "loss_res_int": 0.0,
        "loss_glb_geom": 0.0,
        "loss_glb_int": 0.0,
    }
    n_batch = 0

    pbar = tqdm(loader, desc="Train" if is_train else "Valid", leave=False)

    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        label = merge_label_list(batch["label"], device)

        if label is None:
            raise ValueError("Training/validation batch has no label.")

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            out = model(batch)

            loss, loss_dict = total_loss(
                out=out,
                label=label,
                weights=loss_weights,
                beta_atom=beta_atom,
                beta_geom=beta_geom,
                beta_global=beta_global,
                use_bce_for_int=use_bce_for_int,
            )

            if is_train:
                loss.backward()
                optimizer.step()


        for k in meter:
            meter[k] += float(loss_dict[k].item())

        n_batch += 1
        pbar.set_postfix(
            total=f"{loss_dict['loss_total'].item():.3f}",
            atom=f"{loss_dict['loss_atom'].item():.3f}",
            res_geom=f"{loss_dict['loss_res_geom'].item():.3f}",
            res_int=f"{loss_dict['loss_res_int'].item():.3f}",
            glb_geom=f"{loss_dict['loss_glb_geom'].item():.3f}",
            glb_int=f"{loss_dict['loss_glb_int'].item():.3f}",
        )

    for k in meter:
        meter[k] /= max(n_batch, 1)

    return meter


def get_stage_config(args, stage: str):
    if stage == "stage1_atom":
        lr = args.lr_stage1
        patience = args.patience_stage1
        loss_weights = {
            "atom": args.stage1_w_atom,
            "res_geom": args.stage1_w_res_geom,
            "res_int": args.stage1_w_res_int,
            "glb_geom": args.stage1_w_glb_geom,
            "glb_int": args.stage1_w_glb_int,
        }

    elif stage == "stage2_residue":
        lr = args.lr_stage2
        patience = args.patience_stage2
        loss_weights = {
            "atom": args.stage2_w_atom,
            "res_geom": args.stage2_w_res_geom,
            "res_int": args.stage2_w_res_int,
            "glb_geom": args.stage2_w_glb_geom,
            "glb_int": args.stage2_w_glb_int,
        }

    elif stage == "stage3_global":
        lr = args.lr_stage3
        patience = args.patience_stage3
        loss_weights = {
            "atom": args.stage3_w_atom,
            "res_geom": args.stage3_w_res_geom,
            "res_int": args.stage3_w_res_int,
            "glb_geom": args.stage3_w_glb_geom,
            "glb_int": args.stage3_w_glb_int,
        }

    else:
        raise ValueError(f"Unknown stage: {stage}")

    return lr, patience, loss_weights


def smooth_alpha(x: float, mode: str = "cosine") -> float:
    """
    x in [0, 1]
    """
    x = max(0.0, min(1.0, x))

    if mode == "linear":
        return x

    if mode == "cosine":
        return 0.5 - 0.5 * math.cos(math.pi * x)

    raise ValueError(f"Unknown transition mode: {mode}")


def interpolate_weights(w0, w1, alpha: float):
    return {
        k: (1.0 - alpha) * w0[k] + alpha * w1[k]
        for k in w0
    }


def get_scheduled_stage_config(
    args,
    stage: str,
    epoch: int,
    stage_start_epoch: int,
    prev_stage_for_transition=None,
):
    """
    Return current lr and loss weights.
    If entering a new stage, interpolate from previous-stage weights
    to current-stage weights within transition_epochs.
    """
    target_lr, patience, target_weights = get_stage_config(args, stage)

    if prev_stage_for_transition is None or args.transition_epochs <= 0:
        return target_lr, patience, target_weights, False, 1.0

    local_epoch = epoch - stage_start_epoch + 1

    if local_epoch > args.transition_epochs:
        return target_lr, patience, target_weights, False, 1.0

    prev_lr, _, prev_weights = get_stage_config(args, prev_stage_for_transition)

    if args.transition_epochs == 1:
        progress = 1.0
    else:
        progress = (local_epoch - 1) / (args.transition_epochs - 1)
    alpha = smooth_alpha(progress, mode=args.transition_mode)

    current_lr = (1.0 - alpha) * prev_lr + alpha * target_lr
    current_weights = interpolate_weights(prev_weights, target_weights, alpha)

    return current_lr, patience, current_weights, True, alpha

def get_stage_monitor_metric(
    val_metrics,
    stage: str,
    loss_weights: dict,
    normalize: bool = True,
):
    """
    Cumulative weighted monitor for MSCO checkpoint selection.

    Stage I:
        monitor atom-level objective only.

    Stage II:
        monitor atom + residue objectives.

    Stage III:
        monitor atom + residue + global objectives.

    Note:
        loss_weights should be the current scheduled weights,
        including possible transition interpolation.
    """

    if stage == "stage1_atom":
        keys = [
            ("atom", "loss_atom"),
        ]

    elif stage == "stage2_residue":
        keys = [
            ("atom", "loss_atom"),
            ("res_geom", "loss_res_geom"),
            ("res_int", "loss_res_int"),
        ]

    elif stage == "stage3_global":
        keys = [
            ("atom", "loss_atom"),
            ("res_geom", "loss_res_geom"),
            ("res_int", "loss_res_int"),
            ("glb_geom", "loss_glb_geom"),
            ("glb_int", "loss_glb_int"),
        ]

    else:
        raise ValueError(f"Unknown stage: {stage}")

    metric = 0.0
    weight_sum = 0.0

    for weight_key, metric_key in keys:
        w = float(loss_weights[weight_key])
        metric += w * float(val_metrics[metric_key])
        weight_sum += w

    if normalize:
        metric = metric / max(weight_sum, 1e-8)

    return metric

# =========================================================
# main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--train_pt", type=str, required=True)
    parser.add_argument("--val_pt", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    # model
    parser.add_argument("--node_features_dim", type=int, required=True)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--edge_dim", type=int, default=16)
    parser.add_argument("--num_layers_nba", type=int, default=2)
    parser.add_argument("--num_layers_bda", type=int, default=2)
    parser.add_argument("--num_layers_bb", type=int, default=2)
    parser.add_argument("--num_layers_aa", type=int, default=2)

    # GINE
    parser.add_argument("--eps", type=float, default=0.0)
    parser.add_argument("--train_eps", action="store_true")

    # GATv2
    parser.add_argument("--gat_heads", type=int, default=4)
    parser.add_argument("--gat_concat", action="store_true")
    parser.add_argument("--gat_negative_slope", type=float, default=0.2)
    parser.add_argument("--gat_dropout", type=float, default=0.1)
    parser.add_argument("--gat_add_self_loops", action="store_true")
    parser.add_argument("--gat_fill_value", type=str, default="mean")
    parser.add_argument("--gat_bias", action="store_true")
    parser.add_argument("--gat_share_weights", action="store_true")
    parser.add_argument("--gat_residual", action="store_true")

    # common
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--dist_cutoff", type=float, default=6.0)
    parser.add_argument("--residue_edge_dim", type=int, default=1)
    parser.add_argument("--use_esm", action="store_true")
    parser.add_argument("--esm_dim", type=int, default=1280)
    parser.add_argument("--use_hit", action="store_true")
    parser.add_argument("--use_orig_emb", action="store_true")

    # train
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    # parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--no_msco",
        action="store_true",
        help=(
            "Disable Multi-Scale Curriculum Optimization. "
            "If set, train directly with stage3_global weights from the beginning."
        ),
    )

    parser.add_argument(
        "--no_atom_gate",
        action="store_false",
        dest="use_atom_gate",
        help="Disable atom-score gate before atom-to-residue aggregation."
    )
    
    parser.add_argument(
        "--no_res_gate",
        action="store_false",
        dest="use_res_gate",
        help="Disable residue-score gate before global pooling."
    )
    
    parser.set_defaults(use_atom_gate=True, use_res_gate=True)

    # early stopping
    parser.add_argument("--patience_stage1", type=int, default=10)
    parser.add_argument("--patience_stage2", type=int, default=20)
    parser.add_argument("--patience_stage3", type=int, default=30)
    parser.add_argument("--min_delta", type=float, default=0.0001)

    # stage-wise lr
    parser.add_argument("--lr_stage1", type=float, default=5e-4)
    parser.add_argument("--lr_stage2", type=float, default=5e-4)
    parser.add_argument("--lr_stage3", type=float, default=5e-4)

    # stage-wise loss weights
    parser.add_argument("--stage1_w_atom", type=float, default=1.0)
    parser.add_argument("--stage1_w_res_geom", type=float, default=0.3)
    parser.add_argument("--stage1_w_res_int", type=float, default=0.3)
    parser.add_argument("--stage1_w_glb_geom", type=float, default=0.1)
    parser.add_argument("--stage1_w_glb_int", type=float, default=0.1)

    parser.add_argument("--stage2_w_atom", type=float, default=0.1)
    parser.add_argument("--stage2_w_res_geom", type=float, default=1.0)
    parser.add_argument("--stage2_w_res_int", type=float, default=1.0)
    parser.add_argument("--stage2_w_glb_geom", type=float, default=0.3)
    parser.add_argument("--stage2_w_glb_int", type=float, default=0.3)

    parser.add_argument("--stage3_w_atom", type=float, default=0.1)
    parser.add_argument("--stage3_w_res_geom", type=float, default=0.3)
    parser.add_argument("--stage3_w_res_int", type=float, default=0.3)
    parser.add_argument("--stage3_w_glb_geom", type=float, default=1.0)
    parser.add_argument("--stage3_w_glb_int", type=float, default=1.0)

    parser.add_argument("--transition_epochs", type=int, default=0)
    parser.add_argument("--transition_mode", type=str, default="cosine", choices=["linear", "cosine"])

    # loss hyperparams
    parser.add_argument("--beta_atom", type=float, default=0.1)
    parser.add_argument("--beta_geom", type=float, default=0.1)
    parser.add_argument("--beta_global", type=float, default=0.1)
    parser.add_argument("--use_bce_for_int", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    stage_ckpt_paths = {
        "stage1_atom": save_dir / "best_stage1_atom.pt",
        "stage2_residue": save_dir / "best_stage2_residue.pt",
        "stage3_global": save_dir / "best_stage3_global.pt",
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------- load data ----------
    train_data = torch.load(args.train_pt, map_location="cpu", weights_only=False)
    val_data = torch.load(args.val_pt, map_location="cpu", weights_only=False)

    train_loader = build_dataloader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = build_dataloader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # ---------- model ----------
    model = HITPepModel(
        node_features_dim=args.node_features_dim,
        hidden_channels=args.hidden_channels,
        edge_dim=args.edge_dim,
        num_layers_nba=args.num_layers_nba,
        num_layers_bda=args.num_layers_bda,
        num_layers_bb=args.num_layers_bb,
        num_layers_aa=args.num_layers_aa,
        eps=args.eps,
        train_eps=args.train_eps,
        gat_heads=args.gat_heads,
        gat_concat=args.gat_concat,
        gat_negative_slope=args.gat_negative_slope,
        gat_dropout=args.gat_dropout,
        gat_add_self_loops=args.gat_add_self_loops,
        gat_fill_value=args.gat_fill_value,
        gat_bias=args.gat_bias,
        gat_share_weights=args.gat_share_weights,
        gat_residual=args.gat_residual,
        dropout=args.dropout,
        dist_cutoff=args.dist_cutoff,
        residue_edge_dim=args.residue_edge_dim,
        use_esm=args.use_esm,
        esm_dim=args.esm_dim,
        use_orig_emb=args.use_orig_emb,
        use_hit=args.use_hit,
        use_atom_gate=args.use_atom_gate,
        use_res_gate=args.use_res_gate,
    ).to(device)

    initial_lr = args.lr_stage3 if args.no_msco else args.lr_stage1

    optimizer = AdamW(
        model.parameters(),
        lr=initial_lr,
        weight_decay=args.weight_decay,
    )

    history = []

    if args.no_msco:
        stage_order = ["stage3_global"]
        print("==> MSCO disabled: train directly with stage3_global weights.")
    else:
        stage_order = ["stage1_atom", "stage2_residue", "stage3_global"]
        print("==> MSCO enabled: stage1_atom -> stage2_residue -> stage3_global.")

    stage_idx = 0
    stage = stage_order[stage_idx]

    stage_start_epoch = 1
    prev_stage_for_transition = None

    # scheduler = ReduceLROnPlateau(
    #     optimizer,
    #     mode="min",
    #     factor=0.5,
    #     patience=20,
    # )

    _, stage_patience, _ = get_stage_config(args, stage)

    early_stopper = EarlyStopping(
       patience=stage_patience,
       min_delta=args.min_delta,
       mode="min",
    )

    best_stage_metric = float("inf")
    print(f"==> Start with {stage} (patience={stage_patience})")

    # ---------- train loop ----------
    for epoch in range(1, args.epochs + 1):
        stage = stage_order[stage_idx]
        stage_lr, stage_patience, loss_weights, in_transition, transition_alpha = get_scheduled_stage_config(
            args=args,
            stage=stage,
            epoch=epoch,
            stage_start_epoch=stage_start_epoch,
            prev_stage_for_transition=prev_stage_for_transition,
        )

        # update optimizer lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = stage_lr

        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            loss_weights=loss_weights,
            beta_atom=args.beta_atom,
            beta_geom=args.beta_geom,
            beta_global=args.beta_global,
            use_bce_for_int=args.use_bce_for_int,
        )

        val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            loss_weights=loss_weights,
            beta_atom=args.beta_atom,
            beta_geom=args.beta_geom,
            beta_global=args.beta_global,
            use_bce_for_int=args.use_bce_for_int,
        )

        # Optional:
        # If you still want ReduceLROnPlateau, don't let it fight with stage-wise lr.
        # Simplest choice: comment it out for now.
        # scheduler.step(val_metrics["loss_total"])
        current_stage_metric = val_metrics["loss_total"]
        #current_stage_metric = get_stage_monitor_metric(val_metrics, stage, loss_weights=loss_weights, normalize=False,)

        transition_info = (
            f"Transition alpha={transition_alpha:.2f}"
            if in_transition else "Stable stage"
        )

        log = {
            "epoch": epoch,
            "stage": stage,
            "use_msco": not args.no_msco,
            "in_transition": in_transition,
            "transition_alpha": transition_alpha,
            "lr": optimizer.param_groups[0]["lr"],
            "w_atom": loss_weights["atom"],
            "w_res_geom": loss_weights["res_geom"],
            "w_res_int": loss_weights["res_int"],
            "w_glb_geom": loss_weights["glb_geom"],
            "w_glb_int": loss_weights["glb_int"],
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(log)

        print(
            f"[Epoch {epoch:03d}] Stage={stage} LR={optimizer.param_groups[0]['lr']:.2e} | {transition_info}\n"
            f"  Weights | atom={loss_weights['atom']:.2f} "
            f"res_geom={loss_weights['res_geom']:.2f} "
            f"res_int={loss_weights['res_int']:.2f} "
            f"glb_geom={loss_weights['glb_geom']:.2f} "
            f"glb_int={loss_weights['glb_int']:.2f}\n"
            f"  Train | total={train_metrics['loss_total']:.4f} "
            f"atom={train_metrics['loss_atom']:.4f} "
            f"res_geom={train_metrics['loss_res_geom']:.4f} "
            f"res_int={train_metrics['loss_res_int']:.4f} "
            f"glb_geom={train_metrics['loss_glb_geom']:.4f} "
            f"glb_int={train_metrics['loss_glb_int']:.4f}\n"
            f"  Val   | total={val_metrics['loss_total']:.4f} "
            f"atom={val_metrics['loss_atom']:.4f} "
            f"res_geom={val_metrics['loss_res_geom']:.4f} "
            f"res_int={val_metrics['loss_res_int']:.4f} "
            f"glb_geom={val_metrics['loss_glb_geom']:.4f} "
            f"glb_int={val_metrics['loss_glb_int']:.4f}"
        )

        if in_transition:
            improved = False
            print(
                f"- Transitioning into {stage}: "
                f"alpha={transition_alpha:.2f}, early stopping skipped."
            )
        else:
            improved = early_stopper.step(current_stage_metric)

        if (not in_transition) and improved:
            best_stage_metric = current_stage_metric
            print(f"- New best for {stage} at epoch {epoch}, monitor_metric={best_stage_metric:.6f}")
        
            torch.save(
                {
                    "epoch": epoch,
                    "stage": stage,
                    "best_stage_metric": best_stage_metric,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                },
                stage_ckpt_paths[stage],
            )
            print(f"- Stage best model saved to {stage_ckpt_paths[stage]}\n")
        
            if stage == "stage3_global":
                torch.save(
                    {
                        "epoch": epoch,
                        "stage": stage,
                        "best_val_loss": best_stage_metric,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "args": vars(args),
                    },
                    save_dir / "best_model.pt",
                )

        with open(save_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if (not in_transition) and early_stopper.should_stop:
            print(f"==> Early stopping triggered for {stage} at epoch {epoch}.")

            # restore best checkpoint of current stage
            if stage_ckpt_paths[stage].exists():
                ckpt = torch.load(stage_ckpt_paths[stage], map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                print(f"==> Restored best checkpoint for {stage} from {stage_ckpt_paths[stage]}")

            # final stage -> finish
            if stage == "stage3_global":
                print("==> Final stage finished.")
                break

            # move to next stage
            prev_stage_for_transition = stage
            stage_idx += 1
            next_stage = stage_order[stage_idx]
            stage_start_epoch = epoch + 1
            next_lr, next_patience, _ = get_stage_config(args, next_stage)

            print(
                f"==> Switching from {stage} to {next_stage} "
                f"(next_lr={next_lr:.2e}, next_patience={next_patience})"
            )

            # reset early stopper for next stage
            early_stopper = EarlyStopping(
                patience=next_patience,
                min_delta=args.min_delta,
                mode="min",
            )
            best_stage_metric = float("inf")

    if (save_dir / "best_model.pt").exists():
        print(f"Training finished. Final best model saved to {save_dir / 'best_model.pt'}")
    else:
        print("Training finished, but no final stage best_model.pt was saved.")


if __name__ == "__main__":
    main()

