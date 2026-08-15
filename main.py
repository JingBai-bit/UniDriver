# Single-GPU cross-validation training entry point.
# Reports accuracy, macro metrics, confusion matrices, GFLOPs, and peak memory.

import argparse
import math
import json
import os
import shutil
import random
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fvcore.nn import FlopCountAnalysis

from utils import *
from models import models_adapter
from video_dataset import VideoDataset
from configs import DATASETS


# -----------------------------
# DataLoader wrapper
# -----------------------------
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


# -----------------------------
# Helpers
# -----------------------------
def _fold_dir_name(fold: int) -> str:
    return f"fold_{fold:02d}"


def _resolve_fold_list_paths(dataset_cfg: dict, fold: int) -> Tuple[str, str]:
    """
    Expect TRAIN_LIST / VAL_LIST to be directories:
      TRAIN_LIST/fold_00/train.txt
      VAL_LIST/fold_00/test.txt  (or val.txt)
    """
    train_root = dataset_cfg["TRAIN_LIST"]
    val_root = dataset_cfg["VAL_LIST"]

    fold_dir = _fold_dir_name(fold)
    train_txt = os.path.join(train_root, fold_dir, "train.txt")

    cand_val = [
        os.path.join(val_root, fold_dir, "test.txt"),
        os.path.join(val_root, fold_dir, "val.txt"),
    ]
    val_txt = next((p for p in cand_val if os.path.exists(p)), cand_val[0])

    if not os.path.exists(train_txt):
        raise FileNotFoundError(f"Train list not found: {train_txt}")
    if not os.path.exists(val_txt):
        raise FileNotFoundError(f"Val/Test list not found (tried {cand_val}): {val_txt}")
    return train_txt, val_txt


def _mkdir_if_needed(path: Optional[str]):
    if path is None:
        return
    os.makedirs(path, exist_ok=True)


def _maybe_copy_code(save_dir: Optional[str]):
    if save_dir is None:
        return
    project_root = os.path.dirname(os.path.abspath(__file__))
    for fn in ["models/models_adapter.py", "utils/__init__.py", "utils/misc.py", "utils/simple_tokenizer.py", "main.py"]:
        src = os.path.join(project_root, *fn.split("/"))
        if os.path.exists(src):
            dst = os.path.join(save_dir, *fn.split("/"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)

def _copy_checkpoint_alias(
    save_dir: Optional[str],
    dst_name: str,
    epoch: Optional[int] = None,
    src_name: Optional[str] = None,
) -> None:
    """
    Create a stable checkpoint alias by copying the checkpoint written by utils.save_model().

    This keeps the checkpoint format exactly the same as save_model():
      - model
      - optimizer
      - lr_sched
      - loss_scaler
      - next_epoch

    It does not serialize args, so it avoids pickle errors from local _ArgsProxy objects.
    """
    if save_dir is None:
        return

    os.makedirs(save_dir, exist_ok=True)

    candidate_names = []
    if src_name is not None:
        candidate_names.append(src_name)
    if epoch is not None:
        candidate_names.append(f"checkpoint-{epoch}.pth")
    candidate_names.append("checkpoint_latest.pth")

    src_path = None
    for name in candidate_names:
        path = os.path.join(save_dir, name)
        if os.path.exists(path):
            src_path = path
            break

    if src_path is None:
        print(
            f"Warning: no source checkpoint found for alias {dst_name}. "
            f"Tried: {candidate_names} in {save_dir}"
        )
        return

    if epoch is not None and os.path.basename(src_path) != f"checkpoint-{epoch}.pth":
        print(
            f"Warning: checkpoint-{epoch}.pth not found; alias {dst_name} will copy "
            f"{os.path.basename(src_path)}. If you need exact per-epoch best/last aliases, "
            f"run with --save_freq 1."
        )

    dst_path = os.path.join(save_dir, dst_name)
    tmp_path = dst_path + ".tmp"
    shutil.copy2(src_path, tmp_path)
    os.replace(tmp_path, dst_path)
    print(f"Saved checkpoint alias: {dst_path} <- {src_path}")

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _is_finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


def _normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def _clamp_logit_scale_if_present(model) -> None:
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "logit_scale"):
        m.logit_scale.data.clamp_(0, math.log(100.0))


def _bytes_to_gib(x: int) -> float:
    return float(x) / (1024 ** 3)


def _compute_confusion_matrix(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    if y_true.numel() == 0:
        return cm

    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        cm[int(t), int(p)] += 1
    return cm


def _plot_confusion_matrix(
    cm: torch.Tensor,
    out_path: str,
    title: str,
    class_names: Optional[list] = None,
    normalize: bool = True,
) -> None:
    if cm is None:
        return

    cm_np = cm.detach().cpu().numpy().astype(np.float64)
    if normalize:
        row_sum = cm_np.sum(axis=1, keepdims=True)
        cm_to_show = np.divide(cm_np, row_sum, out=np.zeros_like(cm_np), where=row_sum > 0)
        colorbar_label = "Row-normalized value"
        annotation_fmt = ".2f"
    else:
        cm_to_show = cm_np
        colorbar_label = "Count"
        annotation_fmt = "d"

    n = cm_np.shape[0]
    fig_size = min(max(8.0, n * 0.28), 24.0)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), constrained_layout=True)
    im = ax.imshow(cm_to_show, interpolation="nearest", cmap="Blues", aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    if class_names is not None and len(class_names) == n and n <= 40:
        ticks = np.arange(n)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(class_names, rotation=90, fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
    else:
        num_ticks = min(n, 10)
        if num_ticks > 0:
            ticks = np.unique(np.linspace(0, n - 1, num=num_ticks, dtype=int))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels([str(int(t)) for t in ticks], rotation=90)
            ax.set_yticklabels([str(int(t)) for t in ticks])

    if n <= 20:
        thresh = cm_to_show.max() / 2.0 if cm_to_show.size else 0.0
        for i in range(n):
            for j in range(n):
                value = cm_to_show[i, j]
                if normalize:
                    text_value = format(value, annotation_fmt)
                else:
                    text_value = str(int(cm_np[i, j]))
                ax.text(
                    j,
                    i,
                    text_value,
                    ha="center",
                    va="center",
                    color="white" if value > thresh else "black",
                    fontsize=8,
                )

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _save_confusion_matrix_bundle(
    save_dir: str,
    fold: Optional[int],
    tag: str,
    confusion_matrices: Dict[str, torch.Tensor],
    class_names: Optional[list] = None,
    epoch: Optional[int] = None,
    prefix: Optional[str] = None,
) -> None:
    if confusion_matrices is None:
        return

    epoch_suffix = "" if epoch is None else f" (epoch {epoch})"
    fold_prefix = f"Fold {fold} " if fold is not None else ""
    file_prefix = "" if prefix is None else f"{prefix}_"
    for head_name, cm in confusion_matrices.items():
        out_path = os.path.join(save_dir, f"{file_prefix}{tag}_confusion_matrix_{head_name}.png")
        title = f"{fold_prefix}{tag} test confusion matrix [{head_name}]{epoch_suffix}"
        _plot_confusion_matrix(
            cm=cm,
            out_path=out_path,
            title=title,
            class_names=class_names,
            normalize=True,
        )


def _sum_confusion_matrix_bundles(
    bundles: List[Optional[Dict[str, torch.Tensor]]]
) -> Optional[Dict[str, torch.Tensor]]:
    valid_bundles = [b for b in bundles if b is not None]
    if not valid_bundles:
        return None

    head_names = sorted(valid_bundles[0].keys())
    ret = {}
    for head_name in head_names:
        ret[head_name] = sum(
            bundle[head_name].to(torch.int64) for bundle in valid_bundles
        )
    return ret

def _count_params(model) -> Dict[str, float]:
    m = model.module if hasattr(model, "module") else model
    n_total = sum(p.numel() for p in m.parameters())
    n_trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return {
        "n_total_params": n_total,
        "n_trainable_params": n_trainable,
        "n_total_params_m": n_total / 1e6,
        "n_trainable_params_m": n_trainable / 1e6,
    }


def _compute_gflops_encode_image(
    model,
    num_frames: int,
    spatial_size: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    Report per-view forward GFLOPs for encode_image with batch size = 1.
    Input shape follows training/eval code: [B, C, T, H, W].
    """
    m = model.module if hasattr(model, "module") else model
    was_training = m.training
    m.eval()

    dummy_video = torch.randn(
        1, 3, num_frames, spatial_size, spatial_size,
        device=device, dtype=torch.float32
    )

    class _EncodeImageWrapper(torch.nn.Module):
        def __init__(self, core):
            super().__init__()
            self.core = core

        def forward(self, x):
            out = self.core.encode_image(x)
            if isinstance(out, (tuple, list)):
                return out[0]
            return out

    wrapped = _EncodeImageWrapper(m).to(device)

    with torch.no_grad():
        flops = FlopCountAnalysis(wrapped, dummy_video)
        total_flops = flops.total()

    if was_training:
        m.train()

    return {
        "flops": float(total_flops),
        "gflops": float(total_flops) / 1e9,
    }


def _multiclass_metrics_from_preds(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Compute:
      - UAR = macro recall
      - Macro-Precision
      - Macro-F1
    """
    if y_true.numel() == 0:
        return {"uar": 0.0, "macro_precision": 0.0, "macro_f1": 0.0}

    cm = _compute_confusion_matrix(y_true, y_pred, num_classes).to(torch.float64)

    tp = cm.diag()
    support = cm.sum(dim=1)
    pred_count = cm.sum(dim=0)

    recall = tp / support.clamp_min(eps)
    precision = tp / pred_count.clamp_min(eps)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(eps)

    uar = recall.mean().item() * 100.0
    macro_precision = precision.mean().item() * 100.0
    macro_f1 = f1.mean().item() * 100.0

    return {
        "uar": uar,
        "macro_precision": macro_precision,
        "macro_f1": macro_f1,
    }


def _format_dual_head_eval_metrics(metrics: Dict[str, float]) -> str:
    return (
        "Acc@1 {acc1:.3f} Acc@5 {acc5:.3f} "
        "UAR {uar:.3f} MacroP {macro_precision:.3f} MacroF1 {macro_f1:.3f} //// "
        "AccFC@1 {acc1_fc:.3f} AccFC@5 {acc5_fc:.3f} "
        "UAR_FC {uar_fc:.3f} MacroP_FC {macro_precision_fc:.3f} MacroF1_FC {macro_f1_fc:.3f}"
    ).format(**metrics)


# -----------------------------
# Fold runner
# -----------------------------
def run_fold(args, dataset_cfg: dict, fold: int) -> Dict:
    train_list, val_list = _resolve_fold_list_paths(dataset_cfg, fold)

    fold_save_dir = None
    if args.save_dir is not None:
        fold_save_dir = os.path.join(args.save_dir, _fold_dir_name(fold))
        _mkdir_if_needed(fold_save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Fold {fold}] Using device: {device}")

    _set_seed(args.seed + fold * 1000)

    classes, num_text_aug, text_dict, classes_dict = text_prompt(args.label_csv, args.mlm_label)
    mlm_label_dict = load_word_index_mapping(args.mlm_label)
    mlm_head_len = len(mlm_label_dict)

    model = models_adapter.__dict__[args.model](
        num_classes=dataset_cfg["NUM_CLASSES"],
        num_frames=args.num_frames,
        mlm_head_len=mlm_head_len,
    ).to(device)
    model = model.float()

    for k, v in model.named_parameters():
        if "adapter" in k:
            v.requires_grad = True
        else:
            v.requires_grad = False

    param_stats = _count_params(model)
    print(
        f"[Fold {fold}] total params: {param_stats['n_total_params']} "
        f"({param_stats['n_total_params_m']:.2f}M)"
    )
    print(
        f"[Fold {fold}] trainable params: {param_stats['n_trainable_params']} "
        f"({param_stats['n_trainable_params_m']:.2f}M)"
    )

    try:
        flop_stats = _compute_gflops_encode_image(
            model=model,
            num_frames=args.num_frames,
            spatial_size=args.spatial_size,
            device=device,
        )
        print(f"[Fold {fold}] per-clip GFLOPs: {flop_stats['gflops']:.3f}")
    except Exception as e:
        flop_stats = {"flops": float("nan"), "gflops": float("nan")}
        print(f"[Fold {fold}] GFLOPs computation skipped: {e}")

    if args.verbose:
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f"  trainable: {n} {tuple(p.shape)} {p.dtype}")

    model_without_ddp = model

    train_root = dataset_cfg.get("TRAIN_ROOT", "")
    val_root = dataset_cfg.get("VAL_ROOT", train_root)

    print(f"[Fold {fold}] train_list={train_list}")
    print(f"[Fold {fold}] val_list={val_list}")
    print(f"[Fold {fold}] train_root={train_root}")
    print(f"[Fold {fold}] val_root={val_root}")
    if fold_save_dir:
        print(f"[Fold {fold}] save_dir={fold_save_dir}")

    if not args.eval_only:
        dataset_train = VideoDataset(
            list_path=train_list,
            data_root=train_root,
            random_sample=True,
            mirror=args.mirror,
            spatial_size=args.spatial_size,
            auto_augment=args.auto_augment,
            num_frames=args.num_frames,
            sampling_rate=args.sampling_rate,
            resize_type=args.resize_type,
            scale_range=args.scale_range,
        )
        print(f"[Fold {fold}] train dataset: {dataset_train}")

    dataset_val = VideoDataset(
        list_path=val_list,
        data_root=val_root,
        random_sample=False,
        spatial_size=args.spatial_size,
        num_frames=args.num_frames,
        sampling_rate=args.sampling_rate,
        num_spatial_views=args.num_spatial_views,
        num_temporal_views=args.num_temporal_views,
    )
    print(f"[Fold {fold}] val dataset: {dataset_val}")

    if not args.eval_only:
        dataloader_train = DataLoaderX(
            dataset_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    dataloader_val = DataLoaderX(
        dataset_val,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    class _ArgsProxy:
        pass

    fold_args = _ArgsProxy()
    for k, v in vars(args).items():
        setattr(fold_args, k, v)
    fold_args.save_dir = fold_save_dir

    _maybe_copy_code(fold_save_dir)

    optimizer = None
    scaler = None
    lr_sched = None

    if not fold_args.eval_only:
        if fold_args.lr is None:
            fold_args.lr = fold_args.blr * fold_args.batch_size / 256.0
            print(f"[Fold {fold}] using blr={fold_args.blr} -> effective lr={fold_args.lr}")
        else:
            print(f"[Fold {fold}] using absolute lr={fold_args.lr}")

        params_with_decay, params_without_decay = [], []
        text_params_with_decay, text_params_without_decay = [], []

        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            is_bias = (".bias" in n)
            is_textad = ("textad_" in n)
            if is_bias:
                (text_params_without_decay if is_textad else params_without_decay).append(p)
            else:
                (text_params_with_decay if is_textad else params_with_decay).append(p)

        optimizer = torch.optim.AdamW(
            [
                {"params": params_with_decay, "lr": fold_args.lr, "weight_decay": fold_args.weight_decay},
                {"params": text_params_with_decay, "lr": fold_args.lr / 2.0, "weight_decay": fold_args.weight_decay},
                {"params": params_without_decay, "lr": fold_args.lr, "weight_decay": 0.0},
                {"params": text_params_without_decay, "lr": fold_args.lr / 2.0, "weight_decay": 0.0},
            ]
        )
        print(optimizer)

        scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 8, growth_interval=2000)

        def lr_func(step: int):
            epoch = step / max(1, len(dataloader_train))
            if epoch < fold_args.warmup_epochs:
                return epoch / max(1e-8, fold_args.warmup_epochs)
            return 0.5 + 0.5 * math.cos(
                (epoch - fold_args.warmup_epochs)
                / max(1e-8, (fold_args.epochs - fold_args.warmup_epochs))
                * math.pi
            )

        lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)

    loss_img = KLLoss()
    loss_txt = KLLoss()

    class_names = None
    if len(classes_dict) == dataset_cfg["NUM_CLASSES"]:
        class_names = [str(classes_dict[i]) for i in range(dataset_cfg["NUM_CLASSES"])]

    @torch.no_grad()
    def evaluate(return_confusion: bool = False):
        metric_logger = MetricLogger(delimiter="  ")
        header = f"Fold {fold} Test:"
        model.eval()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        num_classes = dataset_cfg["NUM_CLASSES"]

        all_labels = []
        all_pred_sim = []
        all_pred_fc = []

        text_inputs = classes.to(device)
        with torch.amp.autocast("cuda", enabled=False):
            text_features, _ = model.encode_text(text_inputs)
            text_features = _normalize(text_features.float(), dim=-1, eps=1e-6)

        for data, labels in metric_logger.log_every(dataloader_val, 10, header):
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # val dataset returns [B, V, C, T, H, W]
            B, V = data.size(0), data.size(1)
            data = data.flatten(0, 1)

            with torch.amp.autocast("cuda", enabled=True):
                image_features, _, logits_fc = model.encode_image(data)

            # FC head
            scores_fc = logits_fc.softmax(dim=-1)
            scores_fc = scores_fc.view(B, V, -1).mean(dim=1)

            top1_fc = scores_fc.topk(1, dim=1)[1]
            top5_fc = scores_fc.topk(5, dim=1)[1]
            acc1_fc = (top1_fc.squeeze(1) == labels).float().mean().item() * 100.0
            acc5_fc = (top5_fc == labels.unsqueeze(1)).any(dim=1).float().mean().item() * 100.0
            pred_fc = scores_fc.argmax(dim=1)

            # similarity head
            with torch.amp.autocast("cuda", enabled=False):
                img = _normalize(image_features.float(), dim=-1, eps=1e-6)
                logits_sim = 100.0 * (img @ text_features.T)
                prob_sim = logits_sim.softmax(dim=-1)
                prob_sim = prob_sim.view(B, V, -1).mean(dim=1)

                top1 = prob_sim.topk(1, dim=1)[1]
                top5 = prob_sim.topk(5, dim=1)[1]
                acc1 = (top1.squeeze(1) == labels).float().mean().item() * 100.0
                acc5 = (top5 == labels.unsqueeze(1)).any(dim=1).float().mean().item() * 100.0
                pred_sim = prob_sim.argmax(dim=1)

            metric_logger.meters["acc1"].update(acc1, n=B)
            metric_logger.meters["acc5"].update(acc5, n=B)
            metric_logger.meters["acc1_fc"].update(acc1_fc, n=B)
            metric_logger.meters["acc5_fc"].update(acc5_fc, n=B)

            all_labels.append(labels.detach().cpu())
            all_pred_sim.append(pred_sim.detach().cpu())
            all_pred_fc.append(pred_fc.detach().cpu())

        metric_logger.synchronize_between_processes()
        metrics = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

        all_labels = torch.cat(all_labels, dim=0)
        all_pred_sim = torch.cat(all_pred_sim, dim=0)
        all_pred_fc = torch.cat(all_pred_fc, dim=0)

        macro_sim = _multiclass_metrics_from_preds(all_labels, all_pred_sim, num_classes)
        macro_fc = _multiclass_metrics_from_preds(all_labels, all_pred_fc, num_classes)

        metrics.update({
            "uar": macro_sim["uar"],
            "macro_precision": macro_sim["macro_precision"],
            "macro_f1": macro_sim["macro_f1"],
            "uar_fc": macro_fc["uar"],
            "macro_precision_fc": macro_fc["macro_precision"],
            "macro_f1_fc": macro_fc["macro_f1"],
        })

        if device.type == "cuda":
            metrics["peak_mem_alloc_gib"] = _bytes_to_gib(torch.cuda.max_memory_allocated(device))
            metrics["peak_mem_reserved_gib"] = _bytes_to_gib(torch.cuda.max_memory_reserved(device))
        else:
            metrics["peak_mem_alloc_gib"] = 0.0
            metrics["peak_mem_reserved_gib"] = 0.0

        print(
            f"* {_format_dual_head_eval_metrics(metrics)} //// "
            f"PeakMem alloc {metrics['peak_mem_alloc_gib']:.3f} GiB "
            f"reserved {metrics['peak_mem_reserved_gib']:.3f} GiB"
        )

        if return_confusion:
            confusion_matrices = {
                "sim": _compute_confusion_matrix(all_labels, all_pred_sim, num_classes),
                "fc": _compute_confusion_matrix(all_labels, all_pred_fc, num_classes),
            }
            return metrics, confusion_matrices

        return metrics

    start_epoch = load_model(fold_args, model_without_ddp, optimizer, lr_sched, scaler)

    if fold_args.eval_only:
        metrics, confusion_matrices = evaluate(return_confusion=True)
        if fold_save_dir is not None:
            _save_confusion_matrix_bundle(
                save_dir=fold_save_dir,
                fold=fold,
                tag="last",
                confusion_matrices=confusion_matrices,
                class_names=class_names,
                epoch=None,
            )
            _save_confusion_matrix_bundle(
                save_dir=fold_save_dir,
                fold=fold,
                tag="best",
                confusion_matrices=confusion_matrices,
                class_names=class_names,
                epoch=None,
            )
        return {
            "fold": fold,
            "best": metrics,
            "best_epoch": None,
            "last": metrics,
            "_last_confusion_matrices": {
                k: v.clone() for k, v in confusion_matrices.items()
            },
            "_best_confusion_matrices": {
                k: v.clone() for k, v in confusion_matrices.items()
            },
            "_class_names": class_names,
        }

    best = None
    best_epoch = -1
    last = None
    last_eval_epoch = None
    last_confusion_matrices = None
    best_confusion_matrices = None

    text_inputs = classes.to(device)

    for epoch in range(start_epoch, fold_args.epochs):
        model.train()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = f"Fold {fold} Epoch: [{epoch}]"

        for step, (data, labels) in enumerate(metric_logger.log_every(dataloader_train, fold_args.print_freq, header)):
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            text_id = np.random.randint(num_text_aug, size=len(labels))
            texts = [text_dict[j][i, :] for i, j in zip(labels.tolist(), text_id)]
            classes_name = [[lab, classes_dict[lab]] for lab in labels.tolist()]

            mlm_input_tokens_ids, mlm_labels = [], []
            for i in range(data.size(0)):
                if random.random() < args.mlm_prob:
                    mlm_inp, mlm_lab = get_masked_sample(
                        classes_name[i][1],
                        mlm_label_dict,
                        masked_rate=args.mlm_mask_rate,
                    )
                    mlm_input_tokens_ids.append(mlm_inp)
                    mlm_labels.append(mlm_lab)
                else:
                    mlm_input_tokens_ids.append(texts[i].tolist())
                    mlm_labels.append([-100 for _ in range(len(texts[i]))])

            mlm_input_tokens_ids = torch.tensor(mlm_input_tokens_ids, dtype=torch.long, device=device)
            mlm_labels = torch.tensor(mlm_labels, dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=True):
                logits_per_image, logits_per_text, image_embedding, text_mlm_inputs, image_mlm_inputs, logits_fc = model(
                    data, mlm_input_tokens_ids
                )

                loss_fc = F.cross_entropy(logits_fc.float(), labels)

                ground_truth = torch.tensor(gen_label(labels), dtype=torch.float32, device=device)

                loss_imgs = loss_img(logits_per_image.float(), ground_truth)
                loss_texts = loss_txt(logits_per_text.float(), ground_truth)
                loss_kl = 0.5 * (loss_imgs + loss_texts)

                if (mlm_labels != -100).any():
                    mlm_loss = model.compute_mlm(text_mlm_inputs, mlm_labels, image_mlm_inputs)["mlm_loss"]
                else:
                    mlm_loss = torch.zeros((), device=device, dtype=torch.float32)

            with torch.amp.autocast("cuda", enabled=False):
                text_features, _ = model.encode_text(text_inputs)
                text_features = _normalize(text_features.float(), dim=-1, eps=1e-6)

                img = _normalize(image_embedding.float(), dim=-1, eps=1e-6)

                logits_sim = 100.0 * (img @ text_features.T)
                loss_sim = F.cross_entropy(logits_sim, labels)

                prob_sim = logits_sim.softmax(dim=-1)
                top1 = prob_sim.topk(1, dim=-1)[1]
                top5 = prob_sim.topk(5, dim=-1)[1]
                acc1 = (top1.squeeze(1) == labels).float().mean().item() * 100.0
                acc5 = (top5 == labels.unsqueeze(1)).any(dim=1).float().mean().item() * 100.0

                prob_fc = logits_fc.softmax(dim=-1)
                top1_fc = prob_fc.topk(1, dim=-1)[1]
                top5_fc = prob_fc.topk(5, dim=-1)[1]
                acc1_fc = (top1_fc.squeeze(1) == labels).float().mean().item() * 100.0
                acc5_fc = (top5_fc == labels.unsqueeze(1)).any(dim=1).float().mean().item() * 100.0

                total_loss = 0.5 * (loss_kl + loss_sim) + loss_fc + (mlm_loss * args.mlm_loss_weight)

            if not _is_finite(total_loss):
                raise RuntimeError(
                    f"Non-finite total_loss at fold={fold}, epoch={epoch}, step={step}: {total_loss.item()}"
                )

            scaler.scale(total_loss).backward()

            scaler.unscale_(optimizer)
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
            lr_sched.step()

            _clamp_logit_scale_if_present(model)

            metric_logger.update(
                loss=float(total_loss.detach().cpu().item()),
                loss_fc=float(loss_fc.detach().cpu().item()),
                loss_kl=float(loss_kl.detach().cpu().item()),
                loss_sim=float(loss_sim.detach().cpu().item()),
                mlm_loss=float(mlm_loss.detach().cpu().item()),
                lr=optimizer.param_groups[0]["lr"],
                acc1=acc1,
                acc5=acc5,
                acc1_fc=acc1_fc,
                acc5_fc=acc5_fc,
            )

        if device.type == "cuda":
            train_peak_alloc_gib = _bytes_to_gib(torch.cuda.max_memory_allocated(device))
            train_peak_reserved_gib = _bytes_to_gib(torch.cuda.max_memory_reserved(device))
        else:
            train_peak_alloc_gib = 0.0
            train_peak_reserved_gib = 0.0

        print(f"[Fold {fold}] Averaged stats: {metric_logger}")
        print(
            f"[Fold {fold}] Epoch {epoch} train peak memory: "
            f"allocated={train_peak_alloc_gib:.3f} GiB, reserved={train_peak_reserved_gib:.3f} GiB"
        )

        log_stats = {"train_" + k: meter.global_avg for k, meter in metric_logger.meters.items()}
        log_stats["train_peak_mem_alloc_gib"] = train_peak_alloc_gib
        log_stats["train_peak_mem_reserved_gib"] = train_peak_reserved_gib

        save_model(fold_args, epoch, model_without_ddp, optimizer, lr_sched, scaler)
        _copy_checkpoint_alias(
            save_dir=fold_save_dir,
            dst_name="checkpoint_last.pth",
            epoch=epoch,
        )

        if epoch >= 30:
            fold_args.eval_freq = 2
        if (epoch + 1) % fold_args.eval_freq == 0 or (epoch + 1) == fold_args.epochs:
            val_metrics, val_confusion_matrices = evaluate(return_confusion=True)
            last = val_metrics
            last_eval_epoch = epoch
            last_confusion_matrices = {
                k: v.clone() for k, v in val_confusion_matrices.items()
            }

            if args.monitor_key not in val_metrics:
                raise KeyError(
                    f"monitor_key={args.monitor_key} not found in val metrics. "
                    f"Available keys: {sorted(val_metrics.keys())}"
                )

            is_best = False
            if (best is None) or (val_metrics[args.monitor_key] > best[args.monitor_key]):
                best = dict(val_metrics)
                best_epoch = epoch
                best_confusion_matrices = {
                    k: v.clone() for k, v in val_confusion_matrices.items()
                }
                is_best = True
                _copy_checkpoint_alias(
                    save_dir=fold_save_dir,
                    dst_name="checkpoint_best.pth",
                    epoch=epoch,
                )

            current_metrics_str = _format_dual_head_eval_metrics(val_metrics)
            best_metrics_str = _format_dual_head_eval_metrics(best)

            print(
                f"[Fold {fold}] Eval current_epoch={epoch} best_epoch={best_epoch} "
                f"monitor={args.monitor_key} "
                f"current_{args.monitor_key}={val_metrics[args.monitor_key]:.4f} "
                f"best_{args.monitor_key}={best[args.monitor_key]:.4f}"
            )
            print(f"[Fold {fold}] Current metrics: {current_metrics_str}")
            print(f"[Fold {fold}] Best metrics   : {best_metrics_str}")

            if fold_save_dir is not None:
                log_stats_to_write = dict(log_stats)

                # current test results
                log_stats_to_write.update({"val_" + k: v for k, v in val_metrics.items()})

                # best-so-far results
                if best is not None:
                    log_stats_to_write.update({"best_" + k: v for k, v in best.items()})

                log_stats_to_write["epoch"] = epoch
                log_stats_to_write["best_epoch"] = best_epoch
                log_stats_to_write["monitor_key"] = args.monitor_key
                log_stats_to_write["is_best"] = is_best
                log_stats_to_write["n_trainable_params"] = param_stats["n_trainable_params"]
                log_stats_to_write["n_total_params"] = param_stats["n_total_params"]
                log_stats_to_write["gflops_per_view_encode_image"] = flop_stats["gflops"]

                with open(os.path.join(fold_save_dir, "metrics.jsonl"), "a") as f:
                    f.write(json.dumps(log_stats_to_write) + "\n")

    if best is None:
        best = last if last is not None else {}
    if best_confusion_matrices is None and last_confusion_matrices is not None:
        best_confusion_matrices = {
            k: v.clone() for k, v in last_confusion_matrices.items()
        }
    if fold_save_dir is not None:
        if last_confusion_matrices is not None:
            _save_confusion_matrix_bundle(
                save_dir=fold_save_dir,
                fold=fold,
                tag="last",
                confusion_matrices=last_confusion_matrices,
                class_names=class_names,
                epoch=last_eval_epoch,
            )
        if best_confusion_matrices is not None:
            _save_confusion_matrix_bundle(
                save_dir=fold_save_dir,
                fold=fold,
                tag="best",
                confusion_matrices=best_confusion_matrices,
                class_names=class_names,
                epoch=best_epoch if best_epoch >= 0 else None,
            )

    return {
        "fold": fold,
        "best": best,
        "best_epoch": best_epoch,
        "last": last if last is not None else best,
        "n_total_params": param_stats["n_total_params"],
        "n_trainable_params": param_stats["n_trainable_params"],
        "gflops_per_view_encode_image": flop_stats["gflops"],
        "_last_confusion_matrices": last_confusion_matrices,
        "_best_confusion_matrices": best_confusion_matrices,
        "_class_names": class_names,
    }


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=2)
    parser.add_argument("--blr", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--eval_only", action="store_true")

    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--auto_remove", action="store_true")
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--label_csv", type=str, required=True)
    parser.add_argument("--mlm_label", type=str, required=True)
    parser.add_argument("--pretrain", type=str, default=None)

    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS.keys())
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--spatial_size", type=int, default=224)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--sampling_rate", type=int, default=0)
    parser.add_argument("--num_spatial_views", type=int, default=1)
    parser.add_argument("--num_temporal_views", type=int, default=1)
    parser.add_argument("--auto_augment", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--resize_type",
        type=str,
        default="random_resized_crop",
        choices=["random_resized_crop", "random_short_side_scale_jitter"],
    )
    parser.add_argument("--scale_range", type=float, nargs=2, default=[0.08, 1.0])
    parser.add_argument("--print_freq", type=int, default=10)
    parser.add_argument("--eval_freq", type=int, default=1)

    parser.add_argument("--fold", type=int, default=-1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    parser.add_argument("--mlm_prob", type=float, default=0.2)
    parser.add_argument("--mlm_mask_rate", type=float, default=0.2)
    parser.add_argument("--mlm_loss_weight", type=float, default=0.1)

    parser.add_argument("--verbose", action="store_true")

    parser.add_argument(
        "--monitor_key",
        type=str,
        default="acc1",
        help=(
            "metric used to select best checkpoint. "
            "e.g. acc1, uar, macro_precision, macro_f1, "
            "acc1_fc, uar_fc, macro_precision_fc, macro_f1_fc"
        ),
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Starting single GPU training")
    print("=" * 80)

    if args.verbose:
        print("Arguments:")
        for k, v in vars(args).items():
            print(f"  {k}: {v}")

    dataset_cfg = DATASETS[args.dataset]
    n_folds = int(dataset_cfg.get("N_FOLDS", 1))

    if args.fold >= 0:
        if args.fold >= n_folds:
            raise ValueError(f"--fold {args.fold} out of range for N_FOLDS={n_folds}")
        folds_to_run = [args.fold]
    else:
        folds_to_run = list(range(n_folds))

    fold_results = []
    aggregate_last_confusion_bundles = []
    aggregate_best_confusion_bundles = []
    aggregate_class_names = None
    for fold in folds_to_run:
        print("=" * 80)
        print(f"Starting fold {fold}/{n_folds - 1}")
        print("=" * 80)

        res = run_fold(args, dataset_cfg, fold)

        if aggregate_class_names is None:
            aggregate_class_names = res.get("_class_names", None)
        aggregate_last_confusion_bundles.append(res.pop("_last_confusion_matrices", None))
        aggregate_best_confusion_bundles.append(res.pop("_best_confusion_matrices", None))
        res.pop("_class_names", None)
        fold_results.append(res)

        torch.cuda.empty_cache()

    keys = set()
    for r in fold_results:
        keys.update(r.get("best", {}).keys())
    keys = sorted(keys)

    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "n_folds": len(fold_results),
        "folds": fold_results,
        "aggregate": {},
        "monitor_key": args.monitor_key,
    }

    print("\n" + "#" * 80)
    print("Cross-validation summary (best per fold)")
    print("#" * 80)

    for r in fold_results:
        f = r["fold"]
        be = r.get("best_epoch", None)
        m = r.get("best", {})
        line = f"Fold {f:02d}  best_epoch={be}  " + "  ".join(
            [f"{k}={m.get(k, float('nan')):.3f}" for k in keys]
        )
        print(line)

    for k in keys:
        vals = [r.get("best", {}).get(k, float("nan")) for r in fold_results]
        arr = np.array(vals, dtype=np.float64)
        mean = float(np.nanmean(arr))
        std = float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0
        summary["aggregate"][k] = {"mean": mean, "std": std, "values": vals}

    # global static info from first fold
    if len(fold_results) > 0:
        summary["n_total_params"] = fold_results[0].get("n_total_params", None)
        summary["n_trainable_params"] = fold_results[0].get("n_trainable_params", None)
        summary["gflops_per_view_encode_image"] = fold_results[0].get("gflops_per_view_encode_image", None)

    print("\nAggregate (mean ± std):")
    for k in keys:
        m = summary["aggregate"][k]["mean"]
        s = summary["aggregate"][k]["std"]
        print(f"{k}: {m:.3f} ± {s:.3f}")

    if summary.get("n_total_params", None) is not None:
        print(f"Total params: {summary['n_total_params']}")
    if summary.get("n_trainable_params", None) is not None:
        print(f"Trainable params: {summary['n_trainable_params']}")
    if summary.get("gflops_per_view_encode_image", None) is not None:
        print(f"GFLOPs per view (encode_image): {summary['gflops_per_view_encode_image']:.3f}")
    aggregate_last_confusion_matrices = _sum_confusion_matrix_bundles(aggregate_last_confusion_bundles)
    aggregate_best_confusion_matrices = _sum_confusion_matrix_bundles(aggregate_best_confusion_bundles)

    if args.save_dir is not None:
        _mkdir_if_needed(args.save_dir)

        if aggregate_last_confusion_matrices is not None:
            _save_confusion_matrix_bundle(
                save_dir=args.save_dir,
                fold=None,
                tag="last_all_folds",
                confusion_matrices=aggregate_last_confusion_matrices,
                class_names=aggregate_class_names,
                epoch=None,
                prefix="aggregate",
            )
        if aggregate_best_confusion_matrices is not None:
            _save_confusion_matrix_bundle(
                save_dir=args.save_dir,
                fold=None,
                tag="best_all_folds",
                confusion_matrices=aggregate_best_confusion_matrices,
                class_names=aggregate_class_names,
                epoch=None,
                prefix="aggregate",
            )

        out_path = os.path.join(args.save_dir, "cv_results.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
