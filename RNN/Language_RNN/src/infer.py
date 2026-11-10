"""Inferencja — ostatni best_eval z saved_models/ + tokenizer z text_and_tokenizer."""

from __future__ import annotations

import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

from metrics.infer_metrics import (
    INFERENCE_FILENAME,
    append_inference_rows,
    write_inference_reports,
)
from metrics.plot_inference import plot_inference_eval
from text_and_tokenizer import load_tokenizer, tokenizer_dir
from train import (
    _run_tag_suffix,
    checkpoint_run_dir,
    load_checkpoint,
    project_root,
)
from log_print import log_print


def find_latest_run_dir(architecture: str, root: Path | None = None) -> Path:
    root = root or project_root() / "saved_models"
    if not root.is_dir():
        raise FileNotFoundError(f"Brak katalogu saved_models: {root}")

    prefix = f"{architecture}-"
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        raise FileNotFoundError(
            f"Brak runów {prefix}* w {root} — najpierw wytrenuj (train.main architecture={architecture!r})"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_best_eval_pt(run_dir: Path, architecture: str) -> Path:
    path = run_dir / f"{architecture}_best_eval.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Brak {path.name} w {run_dir}")
    return path


def resolve_evaluation_dir(run_tag: str | None = None) -> Path:
    when = datetime.now()
    suffix = _run_tag_suffix(run_tag)
    root = project_root() / "model_evaluations" / f"{when:%Y.%m.%d-%H%M%S}{suffix}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _warmup(net, *, length: int = 8) -> None:
    net.generate(net, seed=".", length=length, temperature=1.0, measure=False)


def _bench_config() -> tuple[int, int, float]:
    repeats = int(os.environ.get("LANGUAGE_RNN_BENCH_REPEATS", "10"))
    repeats = max(repeats, 10)
    discard_first = int(os.environ.get("LANGUAGE_RNN_BENCH_DISCARD_FIRST", "2"))
    iqr_mult = float(os.environ.get("LANGUAGE_RNN_BENCH_IQR_MULT", "1.5"))
    return repeats, discard_first, iqr_mult


def _is_finite(value: object) -> bool:
    if value is None:
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return v == v


def _robust_mean_stats(
    samples: list[float],
    *,
    discard_first: int,
    iqr_mult: float = 1.5,
) -> tuple[float, int, float]:
    """Średnia po odrzuceniu pierwszych próbek i outlierów (IQR)."""
    samples = [float(x) for x in samples if _is_finite(x)]
    if not samples:
        return float("nan"), 0, float("nan")
    rest = list(samples[discard_first:]) if discard_first < len(samples) else [samples[-1]]
    if len(rest) >= 4:
        sorted_rest = sorted(rest)
        n = len(sorted_rest)
        q1 = sorted_rest[n // 4]
        q3 = sorted_rest[(3 * n) // 4]
        if q3 > q1:
            iqr = q3 - q1
            lo = q1 - iqr_mult * iqr
            hi = q3 + iqr_mult * iqr
            kept = [x for x in rest if lo <= x <= hi]
        else:
            kept = rest
    else:
        kept = rest
    if not kept:
        kept = rest
    mean = statistics.mean(kept)
    std = statistics.stdev(kept) if len(kept) > 1 else 0.0
    return mean, len(kept), std


def _generate_with_bench_tps(
    net,
    *,
    prompt: str,
    length: int,
    temperature: float,
) -> tuple[str, dict]:
    """Wielokrotna generacja: średnia tok/s i perplexity (IQR + odrzut początkowych runów)."""
    repeats, discard_first, iqr_mult = _bench_config()

    net.generate(
        net,
        seed=prompt,
        length=min(length, 24),
        temperature=temperature,
        measure=False,
    )

    text = ""
    runs: list[dict] = []
    for i in range(repeats):
        t, s = net.generate(
            net, seed=prompt, length=length, temperature=temperature, measure=True
        )
        runs.append(s)
        if i == 0:
            text = t

    stats = dict(runs[0])
    tps_samples = [
        float(s["tokens_per_second"])
        for s in runs
        if _is_finite(s.get("tokens_per_second")) and float(s["tokens_per_second"]) > 0
    ]
    prompt_ppl = [float(s["prompt_perplexity"]) for s in runs if _is_finite(s.get("prompt_perplexity"))]
    gen_ppl = [float(s["generate_perplexity"]) for s in runs if _is_finite(s.get("generate_perplexity"))]
    samp_ppl = [float(s["sampling_perplexity"]) for s in runs if _is_finite(s.get("sampling_perplexity"))]

    mean_tps, n_used, std_tps = _robust_mean_stats(
        tps_samples, discard_first=discard_first, iqr_mult=iqr_mult
    )
    mean_pp, _, _ = _robust_mean_stats(
        prompt_ppl, discard_first=discard_first, iqr_mult=iqr_mult
    )
    mean_gp, _, _ = _robust_mean_stats(
        gen_ppl, discard_first=discard_first, iqr_mult=iqr_mult
    )
    mean_sp, _, _ = _robust_mean_stats(
        samp_ppl, discard_first=discard_first, iqr_mult=iqr_mult
    )

    stats["tokens_per_second"] = mean_tps
    stats["prompt_perplexity"] = mean_pp
    stats["generate_perplexity"] = mean_gp
    stats["sampling_perplexity"] = mean_sp
    stats["bench_repeats"] = repeats
    stats["bench_tps_n"] = n_used
    stats["bench_tps_std"] = std_tps
    if mean_tps > 0:
        stats["generate_seconds"] = length / mean_tps
        stats["inference_seconds"] = stats.get("prefill_seconds", 0.0) + stats["generate_seconds"]
    return text, stats


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _inference_row(
    *,
    timestamp: str,
    prompt_idx: int,
    prompt: str,
    length: int,
    architecture: str,
    checkpoint: Path,
    payload: dict,
    device: str,
    temperature: float,
    stats: dict,
    response: str,
    model_report: dict | None = None,
) -> dict:
    return {
        "timestamp": timestamp,
        "prompt_idx": prompt_idx,
        "prompt": prompt,
        "length": length,
        "architecture": architecture,
        "checkpoint": str(checkpoint),
        "epoch": payload.get("epoch"),
        "eval_perplexity": payload.get("eval_perplexity"),
        "device": device,
        "temperature": temperature,
        "prompt_tokens": stats.get("prompt_tokens"),
        "generated_tokens": stats.get("generated_tokens"),
        "prefill_seconds": stats.get("prefill_seconds"),
        "generate_seconds": stats.get("generate_seconds"),
        "inference_seconds": stats.get("inference_seconds"),
        "tokens_per_second": stats.get("tokens_per_second"),
        "bench_repeats": stats.get("bench_repeats"),
        "bench_tps_n": stats.get("bench_tps_n"),
        "bench_tps_std": stats.get("bench_tps_std"),
        "prompt_perplexity": stats.get("prompt_perplexity"),
        "generate_perplexity": stats.get("generate_perplexity"),
        "sampling_perplexity": stats.get("sampling_perplexity"),
        "params": (model_report or {}).get("params"),
        "model_bytes": (model_report or {}).get("model_bytes"),
        "model_description": (model_report or {}).get("description"),
        "response": response,
    }


def load_for_inference(
    architecture: str,
    checkpoint_path: Path | None = None,
    device: str | None = None,
):
    if checkpoint_path is None:
        run_dir = find_latest_run_dir(architecture)
        checkpoint_path = resolve_best_eval_pt(run_dir, architecture)
        log_print(f"Run: {run_dir.name}")

    tok = load_tokenizer()
    payload, net = load_checkpoint(checkpoint_path, device=device)
    if net.vocab_size != tok.vocab_size:
        raise RuntimeError(
            f"Tokenizer ({tok.vocab_size}) ≠ checkpoint ({net.vocab_size}). "
            f"Sprawdź {tokenizer_dir()}"
        )
    log_print(
        f"Wczytano: {checkpoint_path.name} (epoka {payload['epoch']}, "
        f"ppl={payload.get('eval_perplexity', float('nan')):.2f})"
    )
    return net, payload, checkpoint_path


def main(
    *,
    architecture: str,
    seed: str = "W polskim",
    length: int = 500,
    temperature: float = 0.85,
    checkpoint_path: Path | str | None = None,
) -> tuple[str, dict]:
    ckpt = Path(checkpoint_path) if checkpoint_path else None
    net, payload, resolved_ckpt = load_for_inference(architecture, checkpoint_path=ckpt)
    _warmup(net)
    text, stats = _generate_with_bench_tps(
        net, prompt=seed, length=length, temperature=temperature
    )
    log_print("--- Generacja (best_eval) ---\n")
    print(text)
    log_print(
        f"infer: {stats['inference_seconds']:.4f}s "
        f"(prefill {stats['prefill_seconds']:.4f}s + generate {stats['generate_seconds']:.4f}s, "
        f"{stats['tokens_per_second']:.1f} tok/s "
        f"— śr. z {stats.get('bench_tps_n')}/{stats.get('bench_repeats')} "
        f"(σ={stats.get('bench_tps_std', 0):.1f}))"
    )
    log_print(
        f"perplexity: prompt={stats['prompt_perplexity']:.2f}, "
        f"generate={stats['generate_perplexity']:.2f}, "
        f"sampling={stats['sampling_perplexity']:.2f}"
    )

    model_report = net._model_report()
    row = _inference_row(
        timestamp=_iso_timestamp(),
        prompt_idx=0,
        prompt=seed,
        length=length,
        architecture=architecture,
        checkpoint=resolved_ckpt,
        payload=payload,
        device=str(net.device),
        temperature=temperature,
        stats=stats,
        response=text,
        model_report=model_report,
    )
    run_dir = checkpoint_run_dir(resolved_ckpt)
    append_inference_rows(run_dir / INFERENCE_FILENAME, [row])
    log_print(f"Zapis inference → {run_dir / INFERENCE_FILENAME}")
    return text, stats


def compare_models(
    *,
    models: list[tuple[str, Path | str | None]],
    prompts: list[tuple[str, int]],
    temperature: float = 0.85,
    run_tag: str | None = None,
) -> list[dict]:
    eval_dir = resolve_evaluation_dir(run_tag)
    csv_path = eval_dir / INFERENCE_FILENAME
    rows: list[dict] = []

    log_print(f"Porównanie modeli → {eval_dir}")

    for architecture, checkpoint_path in models:
        ckpt = Path(checkpoint_path) if checkpoint_path else None
        net, payload, resolved_ckpt = load_for_inference(architecture, checkpoint_path=ckpt)
        _warmup(net)

        for prompt_idx, (prompt, length) in enumerate(prompts):
            ts = _iso_timestamp()
            text, stats = _generate_with_bench_tps(
                net, prompt=prompt, length=length, temperature=temperature
            )
            model_report = net._model_report()
            row = _inference_row(
                timestamp=ts,
                prompt_idx=prompt_idx,
                prompt=prompt,
                length=length,
                architecture=architecture,
                checkpoint=resolved_ckpt,
                payload=payload,
                device=str(net.device),
                temperature=temperature,
                stats=stats,
                response=text,
                model_report=model_report,
            )
            rows.append(row)
            log_print(
                f"[{architecture}] prompt {prompt_idx + 1}/{len(prompts)} "
                f"tok/s={stats['tokens_per_second']:.1f} "
                f"({stats.get('bench_tps_n')}/{stats.get('bench_repeats')}) "
                f"ppl_gen={stats['generate_perplexity']:.2f}"
            )

    header = f"model_evaluations run: {eval_dir.name}, modele={len(models)}, prompty={len(prompts)}"
    append_inference_rows(csv_path, rows)
    report_path, report_simple_path = write_inference_reports(eval_dir, rows, header=header)
    plot_inference_eval(rows, eval_dir)
    log_print(f"CSV → {csv_path}")
    log_print(f"Raport → {report_path}")
    log_print(f"Raport (skrót) → {report_simple_path}")
    return rows
