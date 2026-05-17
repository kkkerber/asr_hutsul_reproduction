"""Convert the project's HF dataset into the fairseq2 ASR manifest layout.

Output (per the omnilingual-asr ``ManifestAsrDataset`` spec):

    <out_dir>/
    ├── audio/
    │   ├── train/0000000.wav, 0000001.wav, ...
    │   └── dev/0000000.wav, ...
    ├── train.tsv               # line 1: absolute audio root path
    │                           # line 2+: <relpath><TAB><n_samples>
    ├── train.wrd               # one transcript per line, aligned with .tsv
    ├── dev.tsv
    └── dev.wrd

Re-uses the project's existing pipeline:

    * preprocess.load_and_prepare       — same split / cache / duration filter
    * preprocess.decode_audio_entry     — torchcodec-free audio decoding
    * utils.text_normalization          — same Ukrainian/Hutsul normaliser

so the OmniASR manifests stay consistent with the rest of the project.

CLI:

    python scripts/convert_to_omniasr_manifest.py \\
        --out_dir <storage_root>/preprocessed/omniasr/manifest
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    DEFAULT_DATASET_NAME,
    ProjectConfig,
    configure_logging,
    resolve_storage_layout,
)
from preprocess import (  # noqa: E402
    decode_audio_entry,
    load_and_prepare,
)
from utils.text_normalization import build_default_normalizer  # noqa: E402

logger = logging.getLogger(__name__)


def _write_split(
    split: Any,
    *,
    audio_column: str,
    text_column: str,
    out_dir: Path,
    split_name: str,
    sample_rate: int,
    normalizer: Any,
) -> int:
    import soundfile as sf  # imported here so help/--out_dir works without sf

    audio_dir = out_dir / "audio" / split_name
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_root_abs = str(audio_dir.resolve())

    tsv_path = out_dir / f"{split_name}.tsv"
    wrd_path = out_dir / f"{split_name}.wrd"

    n_written = 0
    n_skipped_empty_text = 0
    with open(tsv_path, "w", encoding="utf-8") as ftsv, \
         open(wrd_path, "w", encoding="utf-8") as fwrd:
        # First line of every fairseq2 ASR manifest is the audio root.
        ftsv.write(audio_root_abs + "\n")

        for i, example in enumerate(split):
            text = normalizer(example[text_column] or "")
            if not text.strip():
                n_skipped_empty_text += 1
                continue

            samples, sr = decode_audio_entry(
                example[audio_column], sample_rate
            )
            assert sr == sample_rate, f"resample mismatch: {sr} vs {sample_rate}"
            samples = samples.astype(np.float32, copy=False)

            wav_rel = f"{n_written:07d}.wav"
            wav_path = audio_dir / wav_rel
            sf.write(str(wav_path), samples, sr, subtype="PCM_16")

            ftsv.write(f"{wav_rel}\t{int(len(samples))}\n")
            fwrd.write(text + "\n")
            n_written += 1

    logger.info(
        "Split %s: wrote %d samples (skipped %d empty-text); "
        "manifest=%s audio_root=%s",
        split_name, n_written, n_skipped_empty_text,
        tsv_path, audio_root_abs,
    )
    return n_written


def main() -> None:
    p = argparse.ArgumentParser(
        description="HF dataset -> fairseq2 ASR manifest converter "
                    "for the OmniASR pipeline.",
    )
    p.add_argument(
        "--out_dir",
        required=True,
        help="Manifest output directory (will contain "
             "audio/, train.tsv, train.wrd, dev.tsv, dev.wrd).",
    )
    p.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    p.add_argument("--dataset_config", default=None)
    p.add_argument("--audio_column", default=None)
    p.add_argument("--text_column", default=None)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--include_test",
        action="store_true",
        help="Also emit test.tsv/test.wrd.  The Meta recipe does not "
             "consume a test split — this is for our offline evaluation.",
    )
    p.add_argument(
        "--min_train_audio_duration_sec",
        type=float,
        default=1.0,
        help="Drop samples shorter than this (seconds).  Mirrors the "
             "project's other CTC trainers.",
    )
    p.add_argument(
        "--hf_token",
        default=None,
        help="Hugging Face token (gated datasets).",
    )
    p.add_argument(
        "--trust_remote_code",
        action="store_true",
    )
    args = p.parse_args()

    configure_logging()
    layout = resolve_storage_layout()
    layout.ensure()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ProjectConfig(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        audio_column=args.audio_column,
        text_column=args.text_column,
        sample_rate=args.sample_rate,
        seed=args.seed,
        preprocessed_dir=layout.preprocessed_dir("omniasr"),
        dataset_cache_dir=layout.datasets_cache,
        cache_dir=layout.cache,
        min_train_audio_duration_sec=args.min_train_audio_duration_sec,
    )
    cfg.ensure_dirs()

    dataset, audio_col, text_col = load_and_prepare(
        cfg,
        token=args.hf_token,
        trust_remote_code=args.trust_remote_code,
    )
    normalizer = build_default_normalizer()

    logger.info("Manifest target: %s", out_dir)
    logger.info("Audio column   : %s", audio_col)
    logger.info("Text  column   : %s", text_col)
    logger.info("Splits         : %s",
                {k: len(v) for k, v in dataset.items()})

    _write_split(
        dataset["train"],
        audio_column=audio_col, text_column=text_col,
        out_dir=out_dir, split_name="train",
        sample_rate=args.sample_rate, normalizer=normalizer,
    )
    _write_split(
        dataset["validation"],
        audio_column=audio_col, text_column=text_col,
        out_dir=out_dir, split_name="dev",
        sample_rate=args.sample_rate, normalizer=normalizer,
    )
    if args.include_test:
        _write_split(
            dataset["test"],
            audio_column=audio_col, text_column=text_col,
            out_dir=out_dir, split_name="test",
            sample_rate=args.sample_rate, normalizer=normalizer,
        )

    logger.info("Manifest layout ready under %s", out_dir)


if __name__ == "__main__":
    main()
