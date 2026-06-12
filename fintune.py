import argparse
import importlib
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


DEFAULT_DATA_DIR = Path("/home/anlab/khai/Probation/Tonichi/ocr_recognition_synth")
DEFAULT_OUTPUT_DIR = Path("/home/anlab/khai/Probation/Tonichi/yomitoku_parseq_large_finetuned")
DEFAULT_OCR_EVAL_DIR = Path("/home/anlab/khai/Probation/Tonichi/ocr")
DEFAULT_YOMITOKU_SRC = Path("/home/anlab/khai/Probation/Tonichi/yomitoku/src")
DEFAULT_MODEL_NAME = "parseq-large-v4_1"
DEFAULT_LOSS_CHARSET = ",01>≧"
DEFAULT_OCR_EVAL_CHARSET = "0123456789,>≧"
DEFAULT_LABEL_MAP = "≥=≧"


class RecognitionLabelDataset(Dataset):
    def __init__(self, data_dir, label_file, img_size, label_map=None):
        self.data_dir = Path(data_dir)
        self.label_map = label_map or {}
        self.samples = self.read_label_file(self.data_dir / label_file)
        self.transform = T.Compose(
            [
                T.Resize(tuple(img_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(0.5, 0.5),
            ]
        )

    def read_label_file(self, path):
        samples = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    image_rel, label = line.split("\t", 1)
                except ValueError as exc:
                    raise ValueError(f"Bad label line {path}:{line_no}: {line}") from exc
                label = apply_label_map(label, self.label_map)
                image_path = self.data_dir / image_rel
                if not image_path.exists():
                    raise FileNotFoundError(image_path)
                samples.append((image_path, label))
        if not samples:
            raise ValueError(f"No samples found in {path}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), label


class MixedDataset(Dataset):
    """Mix finetune data with a replay buffer of general data to prevent catastrophic forgetting.

    replay_dir should contain a train.txt in the same tab-separated format.
    replay_ratio controls expected fraction of steps that include a replay sample.
    Replay samples are trained against the FULL charset (no loss mask) so the model
    keeps producing correct output for characters outside the finetune set.
    """

    def __init__(self, finetune_dataset, replay_dir, replay_ratio=1.0, label_map=None):
        self.ft_ds = finetune_dataset
        self.replay_ratio = replay_ratio
        self.replay_samples = []
        self.replay_transform = finetune_dataset.transform

        if replay_dir is not None:
            replay_path = Path(replay_dir) / "train.txt"
            if replay_path.exists():
                replay_ds = RecognitionLabelDataset(
                    replay_dir, "train.txt",
                    finetune_dataset.transform.transforms[0].size,
                    label_map=label_map,
                )
                self.replay_samples = replay_ds.samples
                self.replay_transform = replay_ds.transform
                print(f"Replay buffer loaded: {len(self.replay_samples)} samples from {replay_dir}")
            else:
                print(f"[WARN] --replay-dir set but {replay_path} not found — running without replay.")

    def __len__(self):
        return len(self.ft_ds)

    def __getitem__(self, idx):
        img_ft, label_ft = self.ft_ds[idx]
        if self.replay_samples and random.random() < self.replay_ratio:
            rep_path, rep_label = random.choice(self.replay_samples)
            img_rep = Image.open(rep_path).convert("RGB")
            img_rep = self.replay_transform(img_rep)
            return img_ft, label_ft, img_rep, rep_label
        return img_ft, label_ft, None, None


def collate_mixed(batch):
    ft_imgs, ft_labels, rep_imgs, rep_labels = zip(*batch)
    ft_tensor = torch.stack(ft_imgs, dim=0)
    valid = [(img, lbl) for img, lbl in zip(rep_imgs, rep_labels) if img is not None]
    if valid:
        rep_tensor = torch.stack([x[0] for x in valid], dim=0)
        rep_label_list = [x[1] for x in valid]
    else:
        rep_tensor = None
        rep_label_list = []
    return ft_tensor, list(ft_labels), rep_tensor, rep_label_list


class EWCRegularizer:
    """Elastic Weight Consolidation — penalises changes to weights important for the original task.

    Compute once before finetuning starts (on general/replay data), then call
    ewc.penalty(model) inside the training loop and add it to the CE loss.
    """

    def __init__(self, model, dataloader, device, n_batches=30):
        print(f"Computing EWC Fisher matrix over {n_batches} batches...")
        self.params_before = {
            n: p.clone().detach()
            for n, p in model.named_parameters() if p.requires_grad
        }
        self.fisher = self._compute_fisher(model, dataloader, device, n_batches)
        print("EWC Fisher matrix ready.")

    def _compute_fisher(self, model, dataloader, device, n_batches):
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        model.eval()
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
            images = batch[0].to(device)
            labels = batch[1]
            logits, target = teacher_forcing_logits(model, images, labels)
            loss = F.cross_entropy(logits.flatten(0, 1), target.flatten(),
                                   ignore_index=model.tokenizer.pad_id)
            model.zero_grad()
            loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach().pow(2)
        for k in fisher:
            fisher[k] /= max(1, n_batches)
        model.zero_grad()
        return fisher

    def penalty(self, model, lam=500.0):
        device = next(model.parameters()).device
        loss = torch.tensor(0.0, device=device)
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss = loss + (self.fisher[n] * (p - self.params_before[n]).pow(2)).sum()
        return lam * loss


def collate_batch(batch):
    images, labels = zip(*batch)
    return torch.stack(images, dim=0), list(labels)


def parse_label_map(spec):
    if not spec:
        return {}
    mapping = {}
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Bad --label-map item {item!r}. Expected SRC=DST")
        src, dst = item.split("=", 1)
        if not src or not dst:
            raise ValueError(f"Bad --label-map item {item!r}. Expected SRC=DST")
        mapping[src] = dst
    return mapping


def apply_label_map(label, label_map):
    for src, dst in label_map.items():
        label = label.replace(src, dst)
    return label


def levenshtein_distance(source, target):
    if len(source) < len(target):
        source, target = target, source
    previous = list(range(len(target) + 1))
    for i, source_item in enumerate(source, start=1):
        current = [i]
        for j, target_item in enumerate(target, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (source_item != target_item)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def word_tokens(text):
    tokens = text.split()
    return tokens if tokens else ([text] if text else [])


def build_allowed_token_mask(tokenizer, loss_charset, num_classes, device):
    if not loss_charset:
        return None

    allowed_ids = {tokenizer.eos_id}
    missing = []
    for ch in loss_charset:
        token_id = tokenizer._stoi.get(ch)
        if token_id is None:
            missing.append(ch)
            continue
        if token_id >= num_classes:
            raise ValueError(
                f"Character {ch!r} maps to token id {token_id}, "
                f"but model output has only {num_classes} classes."
            )
        allowed_ids.add(token_id)

    if missing:
        raise ValueError(f"Characters not found in tokenizer charset: {missing}")

    mask = torch.zeros(num_classes, dtype=torch.bool, device=device)
    mask[list(sorted(allowed_ids))] = True
    return mask


def mask_logits(logits, allowed_token_mask):
    if allowed_token_mask is None:
        return logits
    return logits.masked_fill(~allowed_token_mask.view(1, 1, -1), -1e4)


def choose_decode_mask(args, tokenizer, num_classes, device, allowed_token_mask):
    if args.ocr_eval_full_charset:
        return None
    if args.ocr_eval_charset:
        return build_allowed_token_mask(
            tokenizer,
            args.ocr_eval_charset,
            num_classes,
            device,
        )
    return allowed_token_mask


def validate_labels_in_charset(dataset, loss_charset):
    if not loss_charset:
        return
    allowed = set(loss_charset)
    bad = sorted({ch for _, label in dataset.samples for ch in label if ch not in allowed})
    if bad:
        raise ValueError(f"Labels contain characters outside --loss-charset: {bad}")


def prepare_targets(tokenizer, labels, logits_len, device):
    encoded = tokenizer.encode(labels, device=device)
    target = torch.full(
        (encoded.size(0), logits_len),
        tokenizer.pad_id,
        dtype=torch.long,
        device=device,
    )
    shifted = encoded[:, 1:]
    copy_len = min(logits_len, shifted.size(1))
    target[:, :copy_len] = shifted[:, :copy_len]
    return target


def teacher_forcing_logits(model, images, labels):
    tgt = model.tokenizer.encode(labels, device=images.device)
    tgt_in = tgt[:, :-1]
    target = tgt[:, 1:]
    seq_len = tgt_in.size(1)
    memory = model.encode(images)
    tgt_mask = torch.triu(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=images.device),
        1,
    )
    tgt_padding_mask = tgt_in == model.tokenizer.pad_id
    decoder_out = model.decode(
        tgt_in,
        memory,
        tgt_mask=tgt_mask,
        tgt_padding_mask=tgt_padding_mask,
    )
    logits = model.head(decoder_out)
    return logits, target


@torch.inference_mode()
def evaluate(model, dataloader, device, allowed_token_mask=None, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    total_char_edits = 0
    total_chars = 0
    total_word_edits = 0
    total_words = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        logits, target = teacher_forcing_logits(model, images, labels)
        logits = mask_logits(logits, allowed_token_mask)
        loss = F.cross_entropy(
            logits.flatten(0, 1),
            target.flatten(),
            ignore_index=model.tokenizer.pad_id,
        )

        pred_logits = mask_logits(model(images), allowed_token_mask).softmax(-1)
        preds, _ = model.tokenizer.decode(pred_logits)
        for pred, label in zip(preds, labels):
            correct += int(pred == label)
            total_char_edits += levenshtein_distance(list(pred), list(label))
            total_chars += max(1, len(label))

            pred_words = word_tokens(pred)
            label_words = word_tokens(label)
            total_word_edits += levenshtein_distance(pred_words, label_words)
            total_words += max(1, len(label_words))

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    cer = total_char_edits / max(1, total_chars)
    wer = total_word_edits / max(1, total_words)
    return {
        "loss": total_loss / max(1, total_samples),
        "accuracy": correct / max(1, total_samples),
        "char_accuracy": max(0.0, 1.0 - cer),
        "cer": cer,
        "wer": wer,
        "cer_wer": cer + wer,
    }


def collect_ocr_eval_images(path):
    if path is None:
        return []
    path = Path(path)
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    ignored_dirs = {"ocr_results", "ollama_results"}
    if path.is_file():
        return [path] if path.suffix.lower() in suffixes else []
    if not path.exists():
        raise FileNotFoundError(path)
    return sorted(
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file()
        and file_path.suffix.lower() in suffixes
        and not any(parent.name in ignored_dirs for parent in file_path.parents)
    )


def make_unique_output_name(path, used_names):
    stem = path.stem
    name = stem
    counter = 2
    while name in used_names:
        name = f"{stem}_{counter}"
        counter += 1
    used_names.add(name)
    return name


def load_recognition_image(path, transform):
    image = Image.open(path).convert("RGB")
    return transform(image)


@torch.inference_mode()
def run_ocr_eval_folder(
    model,
    image_dir,
    output_dir,
    img_size,
    device,
    allowed_token_mask=None,
    label_map=None,
):
    image_paths = collect_ocr_eval_images(image_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = T.Compose(
        [
            T.Resize(tuple(img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ]
    )

    model.eval()
    results = []
    used_names = set()
    for image_path in image_paths:
        tensor = load_recognition_image(image_path, transform).unsqueeze(0).to(device)
        logits = mask_logits(model(tensor), allowed_token_mask).softmax(-1)
        preds, scores = model.tokenizer.decode(logits)
        pred = preds[0] if preds else ""
        display_pred = pred
        score = float(scores[0]) if scores else 0.0

        output_name = make_unique_output_name(image_path, used_names)
        txt_path = output_dir / f"{output_name}.txt"
        txt_path.write_text(display_pred + "\n", encoding="utf-8")
        results.append(
            {
                "image": str(image_path),
                "prediction": display_pred,
                "model_prediction": pred,
                "score": score,
                "txt": str(txt_path),
            }
        )

    (output_dir / "ocr_eval.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved OCR eval: {output_dir} ({len(results)} images)")
    return results


def serializable_args(args):
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def save_epoch_outputs(
    epoch_dir,
    model,
    recognizer,
    optimizer,
    epoch,
    train_loss,
    val_metrics,
    args,
    ocr_results=None,
):
    epoch_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(epoch_dir)
    recognizer.save_config(epoch_dir / "config.yaml")
    metrics = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_metrics": val_metrics,
        "best_metric": args.best_metric,
        "ocr_eval_images": len(ocr_results or []),
    }
    (epoch_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_metrics": val_metrics,
            "args": serializable_args(args),
        },
        epoch_dir / "checkpoint.pt",
    )
    print(f"Saved epoch model: {epoch_dir}")


def resolve_existing_path(path, fallback_dir=None):
    if path is None:
        return None
    path = Path(path)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(Path.cwd() / path)
        if fallback_dir is not None:
            candidates.append(Path(fallback_dir) / path)
            candidates.append(Path(fallback_dir) / path.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return path


def train(args):
    script_dir = Path(__file__).resolve().parent
    args.path_cfg = resolve_existing_path(args.path_cfg, script_dir)
    args.data_dir = resolve_existing_path(args.data_dir, script_dir) or args.data_dir
    args.ocr_eval_dir = resolve_existing_path(args.ocr_eval_dir, script_dir) or args.ocr_eval_dir
    args.yomitoku_src = resolve_existing_path(args.yomitoku_src, script_dir) or args.yomitoku_src

    if args.yomitoku_src:
        yomitoku_src = str(args.yomitoku_src.resolve())
        sys.path = [p for p in sys.path if p != yomitoku_src]
        sys.path.insert(0, yomitoku_src)
        loaded = sys.modules.get("yomitoku")
        loaded_file = str(getattr(loaded, "__file__", "")) if loaded else ""
        if loaded_file.endswith("yomitoku.py"):
            del sys.modules["yomitoku"]
        importlib.invalidate_caches()

    from yomitoku.text_recognizer import TextRecognizer

    device = torch.device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    recognizer = TextRecognizer(
        model_name=args.model_name,
        path_cfg=str(args.path_cfg) if args.path_cfg else None,
        device=str(device),
        visualize=False,
        from_pretrained=True,
    )
    model = recognizer.model
    model.tokenizer = recognizer.tokenizer
    model.to(device)

    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen.")

    # Freeze decoder (transformer layers) — only head is trained
    if args.freeze_decoder:
        for param in model.decoder.parameters():
            param.requires_grad = False
        print("Decoder frozen.")

    # Freeze head — useful only for EWC-only runs (rarely needed)
    if args.freeze_head:
        for param in model.head.parameters():
            param.requires_grad = False
        print("Head frozen.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    label_map = parse_label_map(args.label_map)
    if label_map:
        print(f"Using label map: {label_map}")

    train_ds = RecognitionLabelDataset(
        args.data_dir,
        "train.txt",
        recognizer._cfg.data.img_size,
        label_map=label_map,
    )
    val_ds = RecognitionLabelDataset(
        args.data_dir,
        "val.txt",
        recognizer._cfg.data.img_size,
        label_map=label_map,
    )
    validate_labels_in_charset(train_ds, args.loss_charset)
    validate_labels_in_charset(val_ds, args.loss_charset)
    num_classes = model.head.out_features
    allowed_token_mask = build_allowed_token_mask(
        model.tokenizer,
        args.loss_charset,
        num_classes,
        device,
    )
    if allowed_token_mask is not None:
        print(f"Using loss/decode charset: {args.loss_charset!r}")
        print(f"Allowed output classes: {int(allowed_token_mask.sum().item())}/{num_classes}")

    # ── Replay buffer to prevent catastrophic forgetting ──────────────────
    # If --replay-dir is given, MixedDataset interleaves general samples into
    # each batch. Replay samples are trained WITHOUT the loss mask so the model
    # must keep predicting the full charset correctly.
    use_replay = args.replay_dir is not None
    if use_replay:
        mixed_train_ds = MixedDataset(
            train_ds,
            replay_dir=args.replay_dir,
            replay_ratio=args.replay_ratio,
            label_map=label_map,
        )
        train_loader = DataLoader(
            mixed_train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate_mixed,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate_batch,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )

    # ── EWC: compute Fisher before training ────────────────────────────────
    # Use a plain loader over the replay/general data so Fisher reflects the
    # original task distribution, NOT the finetune distribution.
    ewc = None
    if args.ewc_lambda > 0:
        ewc_source_dir = args.replay_dir or args.data_dir
        ewc_ds = RecognitionLabelDataset(
            ewc_source_dir,
            "train.txt",
            recognizer._cfg.data.img_size,
            label_map=label_map,
        )
        ewc_loader = DataLoader(
            ewc_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_batch,
        )
        ewc = EWCRegularizer(model, ewc_loader, device, n_batches=args.ewc_batches)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    best_score = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)

            # ── Unpack batch — plain or mixed ─────────────────────────────
            if use_replay:
                images, labels, rep_images, rep_labels = batch
            else:
                images, labels = batch
                rep_images, rep_labels = None, []

            images = images.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                # 1) Finetune loss — masked to loss_charset only
                logits, target = teacher_forcing_logits(model, images, labels)
                logits = mask_logits(logits, allowed_token_mask)
                ft_loss = F.cross_entropy(
                    logits.flatten(0, 1),
                    target.flatten(),
                    ignore_index=model.tokenizer.pad_id,
                )
                loss = ft_loss

                # 2) Replay loss — NO mask, full charset, weighted by replay_weight
                if rep_images is not None and len(rep_labels) > 0:
                    rep_images = rep_images.to(device, non_blocking=True)
                    rep_logits, rep_target = teacher_forcing_logits(model, rep_images, rep_labels)
                    # No mask on replay — model must predict full charset correctly
                    rep_loss = F.cross_entropy(
                        rep_logits.flatten(0, 1),
                        rep_target.flatten(),
                        ignore_index=model.tokenizer.pad_id,
                    )
                    loss = loss + args.replay_weight * rep_loss

                # 3) EWC penalty — keeps weights close to pretrained values
                if ewc is not None:
                    loss = loss + ewc.penalty(model, lam=args.ewc_lambda)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            batch_size = images.size(0)
            running_loss += ft_loss.item() * batch_size  # log only finetune loss for clarity
            seen += batch_size

            if step % args.log_every == 0:
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} "
                    f"train_loss={running_loss / max(1, seen):.4f}"
                )

        train_loss = running_loss / max(1, seen)
        val_metrics = evaluate(model, val_loader, device, allowed_token_mask=allowed_token_mask)
        row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(row)
        score = val_metrics[args.best_metric]
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_char_acc={val_metrics['char_accuracy']:.4f} "
            f"val_cer={val_metrics['cer']:.4f} "
            f"val_wer={val_metrics['wer']:.4f} "
            f"best_metric={args.best_metric}:{score:.4f}"
        )

        epoch_dir = output_dir / f"epoch_{epoch:03d}"
        ocr_results = []
        if not args.skip_ocr_eval:
            ocr_results = run_ocr_eval_folder(
                model,
                args.ocr_eval_dir,
                epoch_dir / "ocr_eval",
                recognizer._cfg.data.img_size,
                device,
                allowed_token_mask=choose_decode_mask(
                    args,
                    model.tokenizer,
                    num_classes,
                    device,
                    allowed_token_mask,
                ),
                label_map=label_map,
            )
        save_epoch_outputs(
            epoch_dir,
            model,
            recognizer,
            optimizer,
            epoch,
            train_loss,
            val_metrics,
            args,
            ocr_results=ocr_results,
        )

        if score < best_score:
            best_score = score
            best_dir = output_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            recognizer.save_config(best_dir / "config.yaml")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "best_metric": args.best_metric,
                    "best_score": best_score,
                    "args": serializable_args(args),
                },
                best_dir / "checkpoint.pt",
            )
            print(f"Saved best model: {best_dir}")

    last_dir = output_dir / "last"
    last_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(last_dir)
    recognizer.save_config(last_dir / "config.yaml")
    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Done. Output: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune YomiToku PARSeq large v4.1 on synthetic recognition data."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ocr-eval-dir", type=Path, default=DEFAULT_OCR_EVAL_DIR)
    parser.add_argument("--skip-ocr-eval", action="store_true")
    parser.add_argument(
        "--ocr-eval-charset",
        default=DEFAULT_OCR_EVAL_CHARSET,
        help=(
            "Charset allowed during per-epoch OCR eval. "
            "Default allows all digits plus comma/>/≧. Use an empty string to reuse --loss-charset."
        ),
    )
    parser.add_argument(
        "--ocr-eval-full-charset",
        action="store_true",
        default=False,
        help="Decode epoch OCR eval with full charsetv2. This can produce unrelated Japanese text on numeric crops.",
    )
    parser.add_argument(
        "--ocr-eval-masked-charset",
        dest="ocr_eval_full_charset",
        action="store_false",
        help="Decode epoch OCR eval with --ocr-eval-charset/--loss-charset.",
    )
    parser.add_argument("--yomitoku-src", type=Path, default=DEFAULT_YOMITOKU_SRC)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--path-cfg",
        type=Path,
        default=None,
        help="Optional yaml override. Keep charset/num_tokens compatible with pretrained weights.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--loss-charset",
        default=DEFAULT_LOSS_CHARSET,
        help=(
            "Only these characters plus EOS are used in CE loss and validation decode. "
            "Use an empty string to train against the full charsetv2 output space."
        ),
    )
    parser.add_argument(
        "--label-map",
        default=DEFAULT_LABEL_MAP,
        help=(
            "Semicolon-separated label replacements before training, e.g. '≥=≧'. "
            "Use an empty string to disable."
        ),
    )
    parser.add_argument(
        "--best-metric",
        choices=("cer", "wer", "cer_wer"),
        default="cer",
        help="Validation edit-distance objective used to save best checkpoint.",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze encoder — train only decoder + head.",
    )
    parser.add_argument(
        "--freeze-decoder",
        action="store_true",
        help="Freeze decoder — train only head. Use with --freeze-encoder for head-only finetuning.",
    )
    parser.add_argument(
        "--freeze-head",
        action="store_true",
        help="Freeze classification head (rarely useful — mainly for EWC-only experiments).",
    )

    # ── Anti-forgetting: replay buffer ────────────────────────────────────
    parser.add_argument(
        "--replay-dir",
        type=Path,
        default=None,
        help=(
            "Directory with general (non-finetune) train.txt + images. "
            "Samples are mixed into every batch WITHOUT loss-charset masking "
            "so the model keeps predicting the full charset correctly. "
            "Recommended: use the original training data or a random subset of it."
        ),
    )
    parser.add_argument(
        "--replay-ratio",
        type=float,
        default=1.0,
        help="Probability [0,1] that a replay sample is drawn alongside each finetune sample. Default 1.0 = always.",
    )
    parser.add_argument(
        "--replay-weight",
        type=float,
        default=1.0,
        help="Weight applied to replay loss relative to finetune CE loss. Default 1.0 = equal.",
    )

    # ── Anti-forgetting: EWC ─────────────────────────────────────────────
    parser.add_argument(
        "--ewc-lambda",
        type=float,
        default=0.0,
        help=(
            "EWC regularisation strength. 0 = disabled. "
            "Start with 500; increase to 2000 if forgetting is still observed. "
            "Requires --replay-dir or --data-dir to contain a train.txt for Fisher estimation."
        ),
    )
    parser.add_argument(
        "--ewc-batches",
        type=int,
        default=30,
        help="Number of batches used to estimate the EWC Fisher matrix. Default 30.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())