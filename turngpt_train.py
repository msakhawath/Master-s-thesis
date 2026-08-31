#!/usr/bin/env python3
"""
Train a lightweight TurnGPT-style model on free text dialog corpora.

Idea:
- Convert each conversation to one token stream with speaker markers and a
  <ts> turn-shift token after every utterance:
      <s1> Hello there. <ts> <s2> Hi! <ts> <s1> ...
- Fine-tune GPT-2 (small) as a causal LM so it learns where <ts> belongs.

Datasets (auto-loaded; any missing one is skipped cleanly):
  - daily_dialog
  - personachat
  - multi_woz_v22
  - metalwoz
  - samsum
Optional:
  - NPR CSV/TXT with 'dialogue' blocks in "Speaker: text" lines.

Run:
  python train_turngpt.py --out ./turngpt_gpt2_ts --epochs 3 --max_len 512 \
    --batch 2 --accum 16 \
    --include daily_dialog personachat multi_woz_v22 metalwoz samsum \
    --npr_csv ./data/npr_interviews.csv   # optional

This should be K80-friendly (fp32; small batch; gradient accumulation).
"""

import os, re, csv, argparse, random
from typing import List, Tuple, Iterable, Optional

import torch
from datasets import load_dataset, Dataset
from transformers import (
    GPT2LMHeadModel, GPT2TokenizerFast,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)

SPECIAL_TS = "<ts>"   # turn-shift token
SPK1 = "<s1>"
SPK2 = "<s2>"
EXTRA_TOKENS = [SPECIAL_TS, SPK1, SPK2]

# ---------- tiny helpers ------------------------------------------------------

def _clean(text: str) -> str:
    """Basic cleanup; keep it simple & robust."""
    text = (text or "").strip()
    # collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text

def _pairs_from_lines(lines: List[str]) -> List[Tuple[str,str]]:
    """
    Parse ["Speaker: text", ...] into [(speaker, text), ...].
    Unknown/malformed lines are skipped.
    """
    out = []
    for ln in lines:
        if ":" not in ln:
            continue
        spk, utt = ln.split(":", 1)
        spk = _clean(spk)
        utt = _clean(utt)
        if spk and utt:
            out.append((spk, utt))
    return out

def _rebalance_to_two_speakers(pairs: List[Tuple[str,str]]) -> List[Tuple[str,str]]:
    """
    Map arbitrary speaker names to two buckets in order of appearance.
    First unique speaker -> <s1>, second unique (different) -> <s2>.
    Any others also map to <s2> (rare in our corpora).
    """
    if not pairs:
        return pairs
    spk_map = {}
    next_tag = [SPK1, SPK2]  # s1 then s2
    mapped = []
    for spk, utt in pairs:
        if spk not in spk_map:
            spk_map[spk] = next_tag[0] if len(spk_map) == 0 else SPK2
        mapped.append((spk_map[spk], utt))
    return mapped

def _to_stream(pairs: List[Tuple[str,str]]) -> str:
    """
    Convert [(<s1>/<s2>, "utterance"), ...] -> single text with <ts> at ends.
    """
    parts = []
    for tag, utt in pairs:
        utt = _clean(utt)
        if not utt:
            continue
        parts.append(f"{tag} {utt} {SPECIAL_TS}")
    return " ".join(parts).strip()

# ---------- dataset loaders ---------------------------------------------------

def load_dailydialog() -> Iterable[List[Tuple[str,str]]]:
    ds = load_dataset("daily_dialog")
    for ex in ds["train"]:
        dialog = ex.get("dialog", None) or ex.get("dialogue", None)
        if not dialog:  # each example is a list[str] of utterances
            continue
        pairs = []
        cur = SPK1
        for utt in dialog:
            u = _clean(utt)
            if not u:
                continue
            pairs.append((cur, u))
            cur = SPK2 if cur == SPK1 else SPK1
        yield pairs

def load_personachat() -> Iterable[List[Tuple[str,str]]]:
    ds = load_dataset("personachat")
    # Each item has 'utterances' (list). Each 'utterances'[i]['history'] is a list[str]
    for ex in ds["train"]:
        utts = ex.get("utterances", [])
        for u in utts:
            hist = u.get("history", [])
            if len(hist) < 2:
                continue
            pairs = []
            cur = SPK1
            for line in hist:
                line = _clean(line)
                if not line or line.startswith("your persona:"):
                    continue
                pairs.append((cur, line))
                cur = SPK2 if cur == SPK1 else SPK1
            if len(pairs) >= 2:
                yield pairs

def load_multiwoz22() -> Iterable[List[Tuple[str,str]]]:
    ds = load_dataset("multi_woz_v22")
    # Many loaders store turns under something like 'turns' with {'speaker','utterance'}
    for ex in ds["train"]:
        turns = ex.get("turns") or ex.get("dialogue") or []
        pairs = []
        for t in turns:
            spk = str(t.get("speaker", "")) or str(t.get("role", ""))
            utt = t.get("utterance", "") or t.get("text", "")
            spk = _clean(spk)
            utt = _clean(utt)
            if not utt:
                continue
            # map user/system -> s1/s2
            tag = SPK1 if spk.lower() in {"user", "usr", "customer"} else SPK2
            pairs.append((tag, utt))
        if len(pairs) >= 2:
            yield pairs

def load_metalwoz() -> Iterable[List[Tuple[str,str]]]:
    ds = load_dataset("metalwoz")
    for ex in ds["train"]:
        utts = ex.get("utterances", [])
        pairs = []
        for t in utts:
            spk = _clean(str(t.get("speaker", "")))
            utt = _clean(t.get("text", ""))
            if not utt:
                continue
            # map speakers (e.g., "USER"/"ASSISTANT") to s1/s2
            tag = SPK1 if spk.lower() in {"user", "client", "seeker"} else SPK2
            pairs.append((tag, utt))
        if len(pairs) >= 2:
            yield pairs

def load_samsum() -> Iterable[List[Tuple[str,str]]]:
    ds = load_dataset("samsum")
    for ex in ds["train"]:
        block = ex.get("dialogue", "")
        if not block:
            continue
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        pairs = _pairs_from_lines(lines)
        pairs = _rebalance_to_two_speakers(pairs)
        if len(pairs) >= 2:
            yield pairs

def load_npr_csv(path: str) -> Iterable[List[Tuple[str,str]]]:
    """
    CSV with a 'dialogue' column; each cell is a multi-line block "Speaker: text".
    """
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            block = row.get("dialogue", "") or row.get("transcript", "")
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            pairs = _pairs_from_lines(lines)
            pairs = _rebalance_to_two_speakers(pairs)
            if len(pairs) >= 2:
                yield pairs

# ---------- build tokenized training set -------------------------------------

def build_streams(include: List[str], npr_csv: Optional[str]) -> List[str]:
    name2loader = {
        "daily_dialog": load_dailydialog,
        "personachat": load_personachat,
        "multi_woz_v22": load_multiwoz22,
        "metalwoz": load_metalwoz,
        "samsum": load_samsum,
    }
    streams = []

    for name in include:
        loader = name2loader.get(name)
        if not loader:
            print(f"[skip] unknown dataset '{name}'")
            continue
        try:
            cnt = 0
            for pairs in loader():
                streams.append(_to_stream(pairs))
                cnt += 1
            print(f"[ok] {name}: {cnt} conversations")
        except Exception as e:
            print(f"[warn] {name} failed: {e}")

    if npr_csv:
        try:
            cnt = 0
            for pairs in load_npr_csv(npr_csv):
                streams.append(_to_stream(pairs))
                cnt += 1
            print(f"[ok] NPR: {cnt} conversations from {npr_csv}")
        except Exception as e:
            print(f"[warn] NPR CSV failed: {e}")

    random.shuffle(streams)
    return streams

def chunk_and_tokenize(streams: List[str], tokenizer: GPT2TokenizerFast, max_len: int) -> Dataset:
    """
    Tokenize each dialog stream separately, then concat and re-chunk into
    contiguous max_len blocks for causal LM.
    """
    enc = tokenizer("\n\n".join(streams), add_special_tokens=True, return_tensors="pt")
    ids = enc["input_ids"].squeeze(0)  # (N,)
    # chop into blocks
    blocks = [ids[i:i+max_len] for i in range(0, ids.size(0)-max_len, max_len)]
    data = {"input_ids": [b.tolist() for b in blocks], "labels": [b.tolist() for b in blocks]}
    return Dataset.from_dict(data)

# ---------- main -------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include", nargs="+", default=["daily_dialog","personachat","multi_woz_v22","metalwoz","samsum"])
    ap.add_argument("--npr_csv", type=str, default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_len", type=int, default=512)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1) tokenizer + model
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.add_special_tokens({"additional_special_tokens": EXTRA_TOKENS})
    if tok.eos_token is None:
        tok.add_special_tokens({"eos_token": "</s>"})

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tok))

    # 2) build training dataset
    streams = build_streams(args.include, args.npr_csv)
    if not streams:
        raise SystemExit("No conversations gathered. Check dataset names or CSV path.")

    dset = chunk_and_tokenize(streams, tok, max_len=args.max_len)

    # 3) trainer
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
    use_gpu = torch.cuda.is_available()
    print("CUDA:", use_gpu)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.03,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        fp16=False,  # K80 doesn't do fast fp16
        bf16=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=dset,
        data_collator=collator,
    )

    trainer.train()
    # 4) save
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"Saved to: {args.out}")

if __name__ == "__main__":
    main()
