#!/usr/bin/env python3
"""
generate_training_data.py
=========================
Production-ready training data generator for MLXSTMBot.

Pulls from four complementary public HuggingFace datasets to create a
diverse, high-quality training_data.json that matches the exact format
expected by ChatDataProvider: [{"content": "..."}]

Sources used (all Apache-2.0 / CC-licensed, safe for research):
  1. wikipedia      — factual, encyclopedic prose
  2. openwebtext    — web text (filtered CommonCrawl, GPT-2 training set)
  3. bookcorpus     — long-form narrative / literary prose
  4. Anthropic/hh-rlhf — real human–assistant dialogues (helpful half only)

Usage:
  pip install datasets tqdm
  python generate_training_data.py                    # defaults (5 000 samples)
  python generate_training_data.py --samples 20000    # larger run
  python generate_training_data.py --output my.json --samples 2000 --seed 99

Requirements:
  Python >= 3.9
  datasets >= 2.0
  tqdm
"""

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Optional but strongly recommended: pip install datasets tqdm
# ---------------------------------------------------------------------------
try:
    from datasets import load_dataset, DownloadConfig
except ImportError:
    print("ERROR: 'datasets' not installed. Run:  pip install datasets tqdm")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    # Graceful fallback: tqdm is purely cosmetic
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------

# Regex compiled once at module level for performance
_MULTI_SPACE   = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_WIKI_SECTION  = re.compile(r"==+[^=]+=+")          # Wikipedia section headers
_HTML_TAG      = re.compile(r"<[^>]+>")              # residual HTML tags
_URL           = re.compile(r"https?://\S+")         # bare URLs


def clean(text: str) -> str:
    """
    Normalize a raw text sample into clean, model-ready prose.

    Steps applied (in order):
      1. Strip residual HTML tags and bare URLs.
      2. Remove Wikipedia-style section headers (== Header ==).
      3. Collapse runs of whitespace / blank lines.
      4. Strip leading/trailing whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub("", text)
    text = _WIKI_SECTION.sub("", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def is_valid(text: str, min_chars: int = 80, max_chars: int = 2048) -> bool:
    """
    Quality gate: reject samples that are too short, too long, or degenerate.

    Rules:
      - Must have between min_chars and max_chars characters after cleaning.
      - Must contain at least one sentence-ending punctuation mark.
      - Must not be mostly non-ASCII (filters garbled encodings).
    """
    if not text:
        return False
    length = len(text)
    if length < min_chars or length > max_chars:
        return False
    if not any(c in text for c in ".!?"):
        return False
    # Reject if more than 30% of characters are non-ASCII (encoding garbage)
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii / length > 0.30:
        return False
    return True


# ---------------------------------------------------------------------------
# Per-source streaming extractors
# Each yields cleaned strings, one sample at a time.
# Using streaming=True avoids downloading the full dataset to disk.
# ---------------------------------------------------------------------------

def stream_wikipedia(target: int, seed: int) -> Iterator[str]:
    """
    Yield cleaned Wikipedia article introductions (the first paragraph of
    each article before any section header).  Wikipedia 20220301.en.
    """
    log.info("  Streaming Wikipedia (target: %d samples)…", target)
    ds = load_dataset(
        "wikipedia",
        "20220301.en",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    count = 0
    for row in ds:
        if count >= target:
            break
        raw = row.get("text", "")
        # Take only the introductory section (before the first "==" header)
        intro_end = raw.find("\n==")
        intro = raw[:intro_end] if intro_end > 0 else raw
        text = clean(intro)
        if is_valid(text, min_chars=100, max_chars=1500):
            yield text
            count += 1
    log.info("  Wikipedia: collected %d samples.", count)


def stream_openwebtext(target: int, seed: int) -> Iterator[str]:
    """
    Yield cleaned passages from OpenWebText (the open reproduction of WebText,
    used to train GPT-2).  Sourced from Skylion007/openwebtext.
    """
    log.info("  Streaming OpenWebText (target: %d samples)…", target)
    ds = load_dataset(
        "Skylion007/openwebtext",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    count = 0
    for row in ds:
        if count >= target:
            break
        raw = row.get("text", "")
        # Take the first 1 500 characters to keep samples focused
        text = clean(raw[:1500])
        if is_valid(text, min_chars=80, max_chars=1400):
            yield text
            count += 1
    log.info("  OpenWebText: collected %d samples.", count)


def stream_bookcorpus(target: int, seed: int) -> Iterator[str]:
    """
    Yield individual sentences from BookCorpus (bookcorpus/bookcorpus).
    Groups consecutive sentences into ~300-character passages for richer context.
    """
    log.info("  Streaming BookCorpus (target: %d samples)…", target)
    ds = load_dataset(
        "bookcorpus",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    count  = 0
    buffer = []
    buf_len = 0
    TARGET_LEN = 350  # characters per passage
    for row in ds:
        if count >= target:
            break
        sentence = clean(row.get("text", ""))
        if not sentence:
            continue
        buffer.append(sentence)
        buf_len += len(sentence)
        if buf_len >= TARGET_LEN:
            passage = " ".join(buffer)
            if is_valid(passage, min_chars=100, max_chars=1200):
                yield passage
                count += 1
            buffer = []
            buf_len = 0
    # Flush any remaining buffer
    if buffer and count < target:
        passage = " ".join(buffer)
        if is_valid(passage, min_chars=100, max_chars=1200):
            yield passage
    log.info("  BookCorpus: collected %d samples.", count)


def stream_hh_rlhf(target: int, seed: int) -> Iterator[str]:
    """
    Yield human turns from Anthropic's HH-RLHF dataset (helpful split only).
    Extracts the final human message from each conversation so the model
    sees natural question/instruction phrasing without training on the
    assistant's response (which is Llama's job in distillation).
    """
    log.info("  Streaming HH-RLHF helpful dialogues (target: %d samples)…", target)
    ds = load_dataset(
        "Anthropic/hh-rlhf",
        data_dir="helpful-base",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=5_000)
    count = 0
    # Pattern: "\n\nHuman: <text>\n\nAssistant:"
    human_pattern = re.compile(r"\n\nHuman: (.+?)(?:\n\nAssistant:|$)", re.DOTALL)
    for row in ds:
        if count >= target:
            break
        chosen = row.get("chosen", "")
        matches = human_pattern.findall(chosen)
        if not matches:
            continue
        # Use the last human turn for most natural phrasing
        text = clean(matches[-1])
        if is_valid(text, min_chars=30, max_chars=800):
            yield text
            count += 1
    log.info("  HH-RLHF: collected %d samples.", count)


# ---------------------------------------------------------------------------
# Allocation strategy
# ---------------------------------------------------------------------------

def compute_allocations(total: int) -> dict[str, int]:
    """
    Split the total sample budget across sources.

    Rationale:
      - Wikipedia (35%): factual, well-formed prose — good for perplexity.
      - OpenWebText (35%): diverse web language — largest coverage.
      - BookCorpus (20%): long-form narrative — good for coherence.
      - HH-RLHF (10%): instruction-following phrasing — good for chat tasks.

    The weights are intentionally skewed toward web text and Wikipedia because
    those two sources best match the distribution of a general-purpose LLM
    pre-training corpus.
    """
    weights = {
        "wikipedia":   0.35,
        "openwebtext": 0.35,
        "bookcorpus":  0.20,
        "hh_rlhf":     0.10,
    }
    alloc = {k: max(1, round(v * total)) for k, v in weights.items()}
    # Adjust for rounding so the total is exact
    diff = total - sum(alloc.values())
    alloc["openwebtext"] += diff   # absorb rounding error in the largest source
    return alloc


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate(
    output_path: str,
    total_samples: int,
    seed: int,
    min_chars: int,
    max_chars: int,
) -> None:
    rng = random.Random(seed)
    alloc = compute_allocations(total_samples)

    log.info("=" * 60)
    log.info("MLXSTMBot training data generator")
    log.info("  Output:       %s", output_path)
    log.info("  Total target: %d samples", total_samples)
    log.info("  Seed:         %d", seed)
    log.info("  Allocation:")
    for src, n in alloc.items():
        log.info("    %-15s %d", src, n)
    log.info("=" * 60)

    all_samples: list[str] = []

    # Collect from each source
    collectors = {
        "wikipedia":   lambda n: stream_wikipedia(n, seed),
        "openwebtext": lambda n: stream_openwebtext(n, seed),
        "bookcorpus":  lambda n: stream_bookcorpus(n, seed),
        "hh_rlhf":     lambda n: stream_hh_rlhf(n, seed),
    }

    for source_name, n_samples in alloc.items():
        log.info("Collecting from: %s", source_name)
        try:
            samples = list(
                tqdm(
                    collectors[source_name](n_samples),
                    total=n_samples,
                    desc=f"  {source_name}",
                    unit="samples",
                    leave=False,
                )
            )
            all_samples.extend(samples)
            log.info("  ✓ %s: %d samples collected", source_name, len(samples))
        except Exception as exc:
            log.warning(
                "  ✗ %s failed (%s). Skipping this source.", source_name, exc
            )

    if not all_samples:
        log.error("No samples collected from any source. Aborting.")
        sys.exit(1)

    # Final shuffle so sources are interleaved (important for training stability)
    rng.shuffle(all_samples)

    # Deduplicate (exact match)
    seen: set[str] = set()
    deduped: list[str] = []
    for s in all_samples:
        if s not in seen:
            seen.add(s)
            deduped.append(s)

    removed = len(all_samples) - len(deduped)
    if removed:
        log.info("Deduplication removed %d exact duplicates.", removed)

    # Apply final quality gate with caller-specified bounds
    final = [s for s in deduped if is_valid(s, min_chars=min_chars, max_chars=max_chars)]
    log.info("Quality gate: %d → %d samples", len(deduped), len(final))

    # Write output
    output = [{"content": text} for text in final]
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    log.info("=" * 60)
    log.info("Done. Wrote %d samples to: %s", len(output), out_path.resolve())
    log.info("File size: %.1f MB", out_path.stat().st_size / 1_048_576)

    # Print a few samples for quick sanity check
    log.info("--- Sample preview (first 3 entries) ---")
    for i, entry in enumerate(output[:3]):
        preview = entry["content"][:120].replace("\n", " ")
        log.info("  [%d] %s…", i + 1, preview)
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate training_data.json for MLXSTMBot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output", "-o",
        default="training_data.json",
        help="Path to write the output JSON file.",
    )
    p.add_argument(
        "--samples", "-n",
        type=int,
        default=5_000,
        help=(
            "Total number of samples to generate. "
            "Recommended: 2 000 for smoke-test, 10 000 for a real run, "
            "50 000+ for serious training."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    p.add_argument(
        "--min-chars",
        type=int,
        default=80,
        help="Minimum character count per sample (shorter samples are discarded).",
    )
    p.add_argument(
        "--max-chars",
        type=int,
        default=2048,
        help=(
            "Maximum character count per sample. "
            "Keep this ≤ (sequence_length * ~4) to avoid excessive truncation "
            "by ChatDataProvider. Default 2048 ≈ 512 tokens."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(
        output_path=args.output,
        total_samples=args.samples,
        seed=args.seed,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )