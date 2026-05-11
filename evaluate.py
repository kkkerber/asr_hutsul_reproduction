from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from config import (
    DEFAULT_DATASET_NAME,
    DEFAULT_SEED,
    ProjectConfig,
    configure_logging,
    resolve_storage_layout,
    set_global_seed,
)
from metrics import (
    MetricCalculator,
    analyze_substitutions,
)
from preprocess import load_and_prepare
from utils.text_normalization import build_default_normalizer

logger = logging.getLogger(__name__)


WHISPER_ARCHITECTURES = {
    "WhisperForConditionalGeneration",
}
W2V_BERT_ARCHITECTURES = {
    "Wav2Vec2BertForCTC",
    "Wav2Vec2BertForXVector",
}
W2V_ARCHITECTURES = {
    "Wav2Vec2ForCTC",
}


def _read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None


def detect_model_family(
    checkpoint_dir: Path,
    *,
    explicit: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    if explicit:
        return explicit, {}

    config_path = checkpoint_dir / "config.json"
    adapter_config_path = checkpoint_dir / "adapter_config.json"

    if config_path.exists():
        cfg = _read_json_if_exists(config_path) or {}
    elif adapter_config_path.exists():

        adapter_cfg = _read_json_if_exists(adapter_config_path) or {}
        base_id = adapter_cfg.get("base_model_name_or_path")
        if not base_id:
            raise ValueError(
                f"adapter_config.json in {checkpoint_dir} is missing "
                "``base_model_name_or_path``"
            )
        from transformers import AutoConfig

        cfg_obj = AutoConfig.from_pretrained(base_id)
        cfg = cfg_obj.to_dict()
    else:
        raise FileNotFoundError(
            f"Neither config.json nor adapter_config.json found in "
            f"{checkpoint_dir}"
        )

    architectures = set(cfg.get("architectures") or [])
    model_type = (cfg.get("model_type") or "").lower()

    if architectures & WHISPER_ARCHITECTURES or model_type == "whisper":
        return "whisper", cfg
    if architectures & W2V_BERT_ARCHITECTURES or model_type == "wav2vec2-bert":
        return "wav2vec2_bert", cfg
    if architectures & W2V_ARCHITECTURES or model_type == "wav2vec2":
        return "wav2vec2", cfg
    if "omniasr" in model_type or "omniasr" in (
        cfg.get("_name_or_path", "") or ""
    ).lower():
        return "omniasr", cfg

    raise ValueError(
        f"Could not auto-detect model family for {checkpoint_dir} "
        f"(architectures={architectures}, model_type={model_type!r}). "
        "Pass --model_type explicitly."
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@dataclass
class LoadedModel:
    family: str
    model: torch.nn.Module
    processor: Any
    feature_extractor: Any
    tokenizer: Any
    is_peft: bool
    base_model_id: Optional[str] = None


def _is_peft_checkpoint(path: Path) -> bool:
    return (path / "adapter_config.json").exists()


def load_model_and_processor(
    checkpoint_dir: Path,
    *,
    model_family: Optional[str] = None,
    device: Optional[torch.device] = None,
    torch_dtype: Optional[torch.dtype] = None,
    hf_token: Optional[str] = None,
    trust_remote_code: bool = False,
) -> LoadedModel:

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch_dtype is None:
        torch_dtype = (
            torch.float16 if device.type == "cuda" else torch.float32
        )

    family, _cfg = detect_model_family(
        checkpoint_dir, explicit=model_family
    )
    is_peft = _is_peft_checkpoint(checkpoint_dir)

    logger.info(
        "Loading checkpoint: %s (family=%s, peft=%s, device=%s, dtype=%s)",
        checkpoint_dir,
        family,
        is_peft,
        device,
        torch_dtype,
    )

    if family == "whisper":
        loaded = _load_whisper(
            checkpoint_dir,
            is_peft=is_peft,
            torch_dtype=torch_dtype,
            hf_token=hf_token,
            trust_remote_code=trust_remote_code,
        )
    elif family == "wav2vec2":
        loaded = _load_wav2vec2(
            checkpoint_dir,
            torch_dtype=torch_dtype,
            hf_token=hf_token,
            trust_remote_code=trust_remote_code,
        )
    elif family == "wav2vec2_bert":
        loaded = _load_wav2vec2_bert(
            checkpoint_dir,
            torch_dtype=torch_dtype,
            hf_token=hf_token,
            trust_remote_code=trust_remote_code,
        )
    elif family == "omniasr":
        loaded = _load_omniasr(
            checkpoint_dir,
            torch_dtype=torch_dtype,
            hf_token=hf_token,
            trust_remote_code=trust_remote_code,
        )
    else:
        raise ValueError(f"Unknown model family: {family}")

    loaded.model = loaded.model.to(device)
    loaded.model.eval()
    return loaded


def _load_whisper(
    checkpoint_dir: Path,
    *,
    is_peft: bool,
    torch_dtype: torch.dtype,
    hf_token: Optional[str],
    trust_remote_code: bool,
) -> LoadedModel:
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    base_model_id: Optional[str] = None

    if is_peft:
        from peft import PeftModel

        adapter_cfg = _read_json_if_exists(
            checkpoint_dir / "adapter_config.json"
        ) or {}
        base_model_id = adapter_cfg.get("base_model_name_or_path")
        if not base_model_id:
            raise ValueError(
                f"PEFT checkpoint {checkpoint_dir} has no base_model_name_or_path"
            )

        processor = WhisperProcessor.from_pretrained(
            base_model_id,
            language="uk",
            task="transcribe",
            token=hf_token,
            trust_remote_code=trust_remote_code,
        )

        if (checkpoint_dir / "preprocessor_config.json").exists():
            try:
                processor = WhisperProcessor.from_pretrained(
                    str(checkpoint_dir),
                    language="uk",
                    task="transcribe",
                    token=hf_token,
                    trust_remote_code=trust_remote_code,
                )
            except Exception as exc:
                logger.warning(
                    "Could not load processor from adapter dir (%s); "
                    "falling back to base model processor.",
                    exc,
                )

        base_model = WhisperForConditionalGeneration.from_pretrained(
            base_model_id,
            torch_dtype=torch_dtype,
            token=hf_token,
            trust_remote_code=trust_remote_code,
        )

        model = PeftModel.from_pretrained(
            base_model, str(checkpoint_dir)
        )

        try:
            model = model.merge_and_unload()
        except Exception as exc:
            logger.warning(
                "Could not merge LoRA adapters (%s); using PeftModel "
                "directly for inference.",
                exc,
            )
    else:
        processor = WhisperProcessor.from_pretrained(
            str(checkpoint_dir),
            language="uk",
            task="transcribe",
            token=hf_token,
            trust_remote_code=trust_remote_code,
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            str(checkpoint_dir),
            torch_dtype=torch_dtype,
            token=hf_token,
            trust_remote_code=trust_remote_code,
        )

    if model.generation_config is None:
        from transformers import GenerationConfig

        model.generation_config = GenerationConfig.from_pretrained(
            base_model_id or str(checkpoint_dir),
            token=hf_token,
        )
    model.generation_config.language = "uk"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = True

    return LoadedModel(
        family="whisper",
        model=model,
        processor=processor,
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        is_peft=is_peft,
        base_model_id=base_model_id,
    )


def _load_wav2vec2(
    checkpoint_dir: Path,
    *,
    torch_dtype: torch.dtype,
    hf_token: Optional[str],
    trust_remote_code: bool,
) -> LoadedModel:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(
        str(checkpoint_dir),
        token=hf_token,
        trust_remote_code=trust_remote_code,
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        str(checkpoint_dir),
        torch_dtype=torch_dtype,
        token=hf_token,
        trust_remote_code=trust_remote_code,
    )

    return LoadedModel(
        family="wav2vec2",
        model=model,
        processor=processor,
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        is_peft=False,
    )


def _load_wav2vec2_bert(
    checkpoint_dir: Path,
    *,
    torch_dtype: torch.dtype,
    hf_token: Optional[str],
    trust_remote_code: bool,
) -> LoadedModel:
    from transformers import (
        AutoFeatureExtractor,
        AutoModelForCTC,
        Wav2Vec2CTCTokenizer,
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        str(checkpoint_dir),
        token=hf_token,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        str(checkpoint_dir),
        token=hf_token,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCTC.from_pretrained(
        str(checkpoint_dir),
        torch_dtype=torch_dtype,
        token=hf_token,
        trust_remote_code=trust_remote_code,
    )

    from models.wav2vec2_bert_trainer import W2VBertProcessor

    processor = W2VBertProcessor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    return LoadedModel(
        family="wav2vec2_bert",
        model=model,
        processor=processor,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        is_peft=False,
    )


def _load_omniasr(
    checkpoint_dir: Path,
    *,
    torch_dtype: torch.dtype,
    hf_token: Optional[str],
    trust_remote_code: bool,
) -> LoadedModel:
    from transformers import (
        AutoFeatureExtractor,
        AutoModelForCTC,
        AutoTokenizer,
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        str(checkpoint_dir),
        token=hf_token,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint_dir),
        token=hf_token,
        trust_remote_code=True,
    )
    model = AutoModelForCTC.from_pretrained(
        str(checkpoint_dir),
        torch_dtype=torch_dtype,
        token=hf_token,
        trust_remote_code=True,
    )

    from models.omniasr_trainer import OmniASRProcessor

    processor = OmniASRProcessor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    return LoadedModel(
        family="omniasr",
        model=model,
        processor=processor,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        is_peft=False,
    )


@dataclass
class InferenceConfig:
    batch_size: int = 8
    max_new_tokens: int = 225
    num_beams: int = 1
    language: str = "uk"
    task: str = "transcribe"


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}


def _build_input_batch(
    feature_extractor: Any,
    audio_list: List[np.ndarray],
    sample_rate: int,
) -> Dict[str, torch.Tensor]:
    batch = feature_extractor(
        audio_list,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )
    out: Dict[str, torch.Tensor] = {}
    if "input_features" in batch:
        out["input_features"] = batch["input_features"]
    if "input_values" in batch:
        out["input_values"] = batch["input_values"]
    if "attention_mask" in batch:
        out["attention_mask"] = batch["attention_mask"]
    return out


def run_inference(
    loaded: LoadedModel,
    dataset_split: Any,
    *,
    audio_column: str,
    text_column: str,
    sample_rate: int,
    cfg: InferenceConfig,
    device: torch.device,
) -> Tuple[List[str], List[str]]:
    """Return ``(predictions, references)``, both text-normalised."""
    normalizer = build_default_normalizer()

    predictions: List[str] = []
    references: List[str] = []

    n = len(dataset_split)
    logger.info(
        "Running inference on %d samples (batch_size=%d)…",
        n,
        cfg.batch_size,
    )

    autocast_dtype = (
        torch.float16
        if device.type == "cuda" and next(loaded.model.parameters()).dtype == torch.float16
        else None
    )

    for start in range(0, n, cfg.batch_size):
        end = min(start + cfg.batch_size, n)
        rows = [dataset_split[i] for i in range(start, end)]
        audio_arrays: List[np.ndarray] = []
        for row in rows:
            audio = row[audio_column]
            arr = np.asarray(audio["array"], dtype=np.float32)
            audio_arrays.append(arr)

        batch = _build_input_batch(
            loaded.feature_extractor, audio_arrays, sample_rate
        )
        batch = _to_device(batch, device)

        with torch.no_grad():
            ctx = (
                torch.autocast(device_type=device.type, dtype=autocast_dtype)
                if autocast_dtype is not None
                else nullcontext()
            )
            with ctx:
                if loaded.family == "whisper":
                    pred_ids = loaded.model.generate(
                        **batch,
                        max_new_tokens=cfg.max_new_tokens,
                        num_beams=cfg.num_beams,
                        language=cfg.language,
                        task=cfg.task,
                    )
                    decoded = loaded.tokenizer.batch_decode(
                        pred_ids, skip_special_tokens=True
                    )
                else:
                    outputs = loaded.model(**batch)
                    logits = outputs.logits  # (B, T, V)
                    pred_ids = logits.argmax(dim=-1)
                    decoded = loaded.tokenizer.batch_decode(
                        pred_ids, skip_special_tokens=True
                    )

        for row, dec in zip(rows, decoded):
            ref = row[text_column] or ""
            predictions.append(normalizer(dec))
            references.append(normalizer(ref))

    return predictions, references


def write_predictions_csv(
    output_path: Path,
    predictions: List[str],
    references: List[str],
    metric_calculator: MetricCalculator,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["reference", "prediction", "wer", "cer"])
        for ref, pred in zip(references, predictions):
            row_wer = metric_calculator.compute_wer([pred], [ref])
            row_cer = metric_calculator.compute_cer([pred], [ref])
            writer.writerow(
                [
                    ref,
                    pred,
                    f"{row_wer:.6f}",
                    f"{row_cer:.6f}",
                ]
            )
    logger.info("Wrote per-sample predictions to %s", output_path)


def write_results_json(
    output_path: Path,
    payload: Dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    logger.info("Wrote summary to %s", output_path)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved ASR checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Directory containing the saved model (or a checkpoint-* dir).",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        choices=[None, "whisper", "wav2vec2", "wav2vec2_bert", "omniasr"],
        help="Override automatic model-family detection.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Optional directory for legacy single-folder output.  "
            "When omitted, evaluation artefacts are placed under the "
            "Drive layout: predictions.csv → ``evaluations/csv/<run>/``, "
            "test_results.json + error_analysis.json → ``evaluations/json/<run>/``, "
            "raw decoded predictions → ``evaluations/predictions/<run>/``."
        ),
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help=(
            "Identifier used to name the per-run subdirectory inside "
            "the layout's evaluations/{csv,json,predictions} dirs.  "
            "Defaults to the checkpoint dir name."
        ),
    )
    parser.add_argument(
        "--storage_root",
        default=os.environ.get("HUTSUL_ASR_ROOT"),
        help="Override the canonical storage root.",
    )

    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--audio_column", default=None)
    parser.add_argument("--text_column", default=None)
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "validation", "test"],
    )
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=225)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--language", default="uk")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--device", default=None, help="cuda / cpu / cuda:0")
    parser.add_argument(
        "--torch_dtype",
        default=None,
        choices=[None, "float32", "float16", "bfloat16"],
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser


def _parse_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    if name is None:
        return None
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _parse_device(name: Optional[str]) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(argv: Optional[List[str]] = None) -> int:
    configure_logging()
    args = _build_argparser().parse_args(argv)
    set_global_seed(args.seed, deterministic=False)

    checkpoint_dir = Path(args.checkpoint)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    layout = resolve_storage_layout(
        Path(args.storage_root) if args.storage_root else None,
        refresh=bool(args.storage_root),
    )
    layout.ensure()

    run_name = args.run_name or checkpoint_dir.resolve().name or "run"

    if args.output_dir:
        legacy_dir = Path(args.output_dir)
        legacy_dir.mkdir(parents=True, exist_ok=True)
        csv_dir = legacy_dir
        json_dir = legacy_dir
        predictions_dir = legacy_dir
    else:
        csv_dir = layout.evaluations_csv / run_name
        json_dir = layout.evaluations_json / run_name
        predictions_dir = layout.evaluations_predictions / run_name
        for d in (csv_dir, json_dir, predictions_dir):
            d.mkdir(parents=True, exist_ok=True)

    device = _parse_device(args.device)
    torch_dtype = _parse_dtype(args.torch_dtype)

    loaded = load_model_and_processor(
        checkpoint_dir,
        model_family=args.model_type,
        device=device,
        torch_dtype=torch_dtype,
        hf_token=args.hf_token,
        trust_remote_code=args.trust_remote_code,
    )

    project_cfg = ProjectConfig(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        audio_column=args.audio_column,
        text_column=args.text_column,
        seed=args.seed,
        preprocessed_dir=layout.preprocessed_dir("shared"),
        dataset_cache_dir=layout.datasets_cache,
        cache_dir=layout.cache,
    )
    project_cfg.ensure_dirs()

    dataset, audio_col, text_col = load_and_prepare(
        project_cfg,
        token=args.hf_token,
        trust_remote_code=args.trust_remote_code,
    )

    if args.split not in dataset:
        raise KeyError(
            f"Split {args.split!r} not in dataset (have {list(dataset.keys())})"
        )

    split_ds = dataset[args.split]
    if args.max_samples is not None:
        split_ds = split_ds.select(range(min(args.max_samples, len(split_ds))))

    logger.info(
        "Evaluating on split=%s (%d samples), audio=%s, text=%s",
        args.split,
        len(split_ds),
        audio_col,
        text_col,
    )

    inference_cfg = InferenceConfig(
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        language=args.language,
        task=args.task,
    )
    sample_rate = project_cfg.sample_rate

    predictions, references = run_inference(
        loaded,
        split_ds,
        audio_column=audio_col,
        text_column=text_col,
        sample_rate=sample_rate,
        cfg=inference_cfg,
        device=device,
    )

    metric_calculator = MetricCalculator()
    overall_wer = metric_calculator.compute_wer(predictions, references)
    overall_cer = metric_calculator.compute_cer(predictions, references)
    error_report = analyze_substitutions(predictions, references)

    logger.info(
        "Overall: WER=%.4f  CER=%.4f  (n=%d)",
        overall_wer,
        overall_cer,
        len(predictions),
    )

    csv_path = csv_dir / "predictions.csv"
    write_predictions_csv(
        csv_path,
        predictions,
        references,
        metric_calculator,
    )

    results_payload = {
        "checkpoint": str(checkpoint_dir),
        "run_name": run_name,
        "model_family": loaded.family,
        "is_peft": loaded.is_peft,
        "base_model_id": loaded.base_model_id,
        "split": args.split,
        "num_samples": len(predictions),
        "wer": overall_wer,
        "cer": overall_cer,
        "inference": asdict(inference_cfg),
        "device": str(device),
        "dtype": str(next(loaded.model.parameters()).dtype),
        "outputs": {
            "csv": str(csv_path),
            "json": str(json_dir / "test_results.json"),
            "error_analysis": str(json_dir / "error_analysis.json"),
            "predictions": str(predictions_dir / "predictions.txt"),
        },
        "storage_root": str(layout.root),
    }
    write_results_json(json_dir / "test_results.json", results_payload)
    error_report.to_json(json_dir / "error_analysis.json", top_k=50)

    pred_dump_path = predictions_dir / "predictions.txt"
    pred_dump_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pred_dump_path, "w", encoding="utf-8") as fh:
        for ref, pred in zip(references, predictions):
            fh.write(f"REF: {ref}\nHYP: {pred}\n\n")
    logger.info("Wrote raw predictions dump to %s", pred_dump_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
