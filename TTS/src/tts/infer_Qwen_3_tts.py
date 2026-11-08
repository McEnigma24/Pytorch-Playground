import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import soundfile as sf
from huggingface_hub import login, snapshot_download
from qwen_asr import Qwen3ForcedAligner
from qwen_asr.inference.qwen3_forced_aligner import ForcedAlignItem
from qwen_tts import Qwen3TTSModel


def clear_screen() -> None:
    if sys.platform == "win32":
        subprocess.run(["cmd", "/c", "cls"], check=False)
    else:
        subprocess.run(["clear"], check=False)
PROJECT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "output"
QWEN3_HUB_ORG = "Qwen"

# Klucz wyboru → folder pod MODELS_DIR (repo Hub = f"{QWEN3_HUB_ORG}/{folder}").
QWEN3_TTS_MODELS: dict[str, str] = {
    "tokenizer": "Qwen3-TTS-Tokenizer-12Hz",
    "1.7b-custom-voice": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "1.7b-voice-design": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "1.7b-base": "Qwen3-TTS-12Hz-1.7B-Base",
    "0.6b-custom-voice": "Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "0.6b-base": "Qwen3-TTS-12Hz-0.6B-Base",
}

QWEN3_FORCED_ALIGNER = "Qwen3-ForcedAligner-0.6B"

def hub_repo_id(local_name: str) -> str:
    return f"{QWEN3_HUB_ORG}/{local_name}"

def resolve_model_name(model: str) -> str:
    if model in QWEN3_TTS_MODELS:
        return QWEN3_TTS_MODELS[model]
    if model in QWEN3_TTS_MODELS.values():
        return model
    keys = ", ".join(sorted(QWEN3_TTS_MODELS))
    raise ValueError(f"Unknown model {model!r}. Keys: {keys}")

def infer_model_dir(model: str) -> Path:
    local_name = resolve_model_name(model)
    return MODELS_DIR / local_name


def forced_aligner_dir() -> Path:
    return MODELS_DIR / QWEN3_FORCED_ALIGNER

def _local_model_from_tree_manifest(local_dir: Path) -> bool:
    trees_dir = local_dir / ".cache" / "huggingface" / "trees"
    if not trees_dir.is_dir():
        return False
    for tree_path in trees_dir.glob("*.json"):
        try:
            payload = json.loads(tree_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        files = payload.get("files")
        if not isinstance(files, dict) or not files:
            continue
        if all(local_dir.joinpath(*rel.split("/")).is_file() for rel in files):
            return True
    return False


def _local_model_weights_present(local_dir: Path) -> bool:
    """Fallback when snapshot files exist but HF trees/*.json was not written."""
    if not (local_dir / "config.json").is_file():
        return False
    single = local_dir / "model.safetensors"
    if single.is_file():
        return True
    index_path = local_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return False
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return False
    shard_names = set(weight_map.values())
    return all(local_dir.joinpath(name).is_file() for name in shard_names)


def local_model_is_complete(local_dir: Path, repo_id: str) -> bool:
    del repo_id  # tree listing is per local_dir after snapshot_download
    if not local_dir.is_dir() or not any(local_dir.iterdir()):
        return False
    if _local_model_from_tree_manifest(local_dir):
        return True
    return _local_model_weights_present(local_dir)

def download_all_qwen3_tts_models() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    pending: list[tuple[str, Path]] = []
    for local_name in dict.fromkeys(QWEN3_TTS_MODELS.values()):
        repo_id = hub_repo_id(local_name)
        local_dir = MODELS_DIR / local_name
        if local_model_is_complete(local_dir, repo_id):
            print(f"OK (already downloaded): {repo_id} -> {local_dir}")
        else:
            pending.append((repo_id, local_dir))

    aligner_repo = hub_repo_id(QWEN3_FORCED_ALIGNER)
    aligner_dir = forced_aligner_dir()
    if not local_model_is_complete(aligner_dir, aligner_repo):
        pending.append((aligner_repo, aligner_dir))

    if not pending:
        print("All Qwen3 TTS models and forced aligner are present.")
        clear_screen()
        return

    login()
    for repo_id, local_dir in pending:
        print(f"Downloading {repo_id} -> {local_dir}")
        snapshot_download(repo_id=repo_id, local_dir=local_dir)

    clear_screen()

def load_qwen_tts_model(model: str) -> Qwen3TTSModel:
    local_name = resolve_model_name(model)
    model_dir = infer_model_dir(model)
    if not local_model_is_complete(model_dir, hub_repo_id(local_name)):
        raise FileNotFoundError(
            f"Incomplete or missing model at {model_dir}. "
            "Finish snapshot_download (run this script once) or pick another model from QWEN3_TTS_MODELS."
        )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_dir}")
    print(f"Device: {device} (cuda available={torch.cuda.is_available()})")

    return Qwen3TTSModel.from_pretrained(
        str(model_dir),
        device_map=device,
        dtype=torch.bfloat16,
    )


def load_forced_aligner() -> Qwen3ForcedAligner:
    model_dir = forced_aligner_dir()
    if not local_model_is_complete(model_dir, hub_repo_id(QWEN3_FORCED_ALIGNER)):
        raise FileNotFoundError(
            f"Incomplete or missing forced aligner at {model_dir}. "
            "Run download_all_qwen3_tts_models() once to fetch Qwen3-ForcedAligner-0.6B."
        )

    use_cuda = torch.cuda.is_available()
    device = "cuda:0" if use_cuda else "cpu"
    dtype = torch.bfloat16 if use_cuda else torch.float32
    print(f"Loading forced aligner {model_dir} ({device}, {dtype})")
    return Qwen3ForcedAligner.from_pretrained(
        str(model_dir),
        device_map=device,
        dtype=dtype,
    )


def _format_vtt_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def write_webvtt(path: Path, text: str, duration_sec: float) -> None:
    """Jeden cue: cały tekst + czas trwania wygenerowanego WAV (brak word-level align)."""
    body = (
        "WEBVTT\n\n"
        f"00:00:00.000 --> {_format_vtt_timestamp(duration_sec)}\n"
        f"{text.strip()}\n"
    )
    path.write_text(body, encoding="utf-8")


def write_webvtt_word_aligned(path: Path, items: list[ForcedAlignItem]) -> None:
    """Jeden cue na token/słowo z Qwen3-ForcedAligner."""
    chunks = ["WEBVTT", ""]
    cue_idx = 0
    for word in items:
        if not word.text.strip():
            continue
        cue_idx += 1
        chunks.append(str(cue_idx))
        chunks.append(
            f"{_format_vtt_timestamp(float(word.start_time))} --> "
            f"{_format_vtt_timestamp(float(word.end_time))}"
        )
        chunks.append(word.text)
        chunks.append("")
    path.write_text("\n".join(chunks), encoding="utf-8")


@dataclass(frozen=True)
class InferOutputs:
    wav_path: Path
    vtt_path: Path
    generation_sec: float
    gpu_peak_bytes: int | None
    alignment_sec: float | None = None
    align_token_count: int | None = None


def strip_delimited_segments(text: str, bracket_pairs: list[tuple[str, str]]) -> str:
    """Usuwa otwierający znacznik, zamykający oraz całą zawartość między nimi (dla każdej pary)."""
    if not bracket_pairs:
        return text
    if len({open_ch for open_ch, _ in bracket_pairs}) != len(bracket_pairs):
        raise ValueError("bracket_pairs: duplicate opening delimiters")
    open_to_close = dict(bracket_pairs)

    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        close = open_to_close.get(ch)
        if close is not None:
            j = text.find(close, i + 1)
            if j == -1:
                out.append(ch)
                i += 1
            else:
                i = j + 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def print_available_speakers(model: Qwen3TTSModel) -> list[str]:
    speakers = model.get_supported_speakers()
    if not speakers:
        print("Ten checkpoint nie ma listy predefiniowanych głosów (np. Base / VoiceDesign).")
        return []
    print("Dostępni speakerzy:", ", ".join(speakers))
    return list(speakers)

def infer(
    text: str,
    *,
    model: str,
    speaker: str | None = None,
    language: str = "English",
    qwen_model: Qwen3TTSModel | None = None,
    forced_aligner: Qwen3ForcedAligner | None = None,
    use_forced_alignment: bool = True,
    strip_delimited_pairs: list[tuple[str, str]] | None = None,
) -> InferOutputs:
    if not text.strip():
        raise ValueError("text must not be empty")

    pairs = strip_delimited_pairs or []
    if pairs:
        text = strip_delimited_segments(text, pairs)
        if not text.strip():
            raise ValueError("text is empty after strip_delimited_pairs")

    local_name = resolve_model_name(model)
    if qwen_model is None:
        qwen_model = load_qwen_tts_model(model)

    speakers = print_available_speakers(qwen_model)
    if qwen_model.model.tts_model_type != "custom_voice":
        raise ValueError(
            f"{local_name} nie obsługuje generate_custom_voice "
            f"(typ: {qwen_model.model.tts_model_type}). Użyj klucza *-custom-voice."
        )

    if speaker is None:
        if not speakers:
            raise ValueError("speaker is required for this model")
        speaker = speakers[0]
        print(f"Nie podano speakera — używam: {speaker!r}")

    preview = text if len(text) <= 120 else f"{text[:117]}..."
    print(f"Synteza | speaker={speaker!r} | language={language!r} | text: {preview!r}")

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    wavs, sr = qwen_model.generate_custom_voice(
        text=text,
        language=language,
        speaker=speaker,
    )
    if use_cuda:
        torch.cuda.synchronize()
    generation_sec = time.perf_counter() - t0
    gpu_peak_bytes = torch.cuda.max_memory_allocated() if use_cuda else None

    duration_sec = len(wavs[0]) / float(sr)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d---%H-%M-%S")
    wav_path = OUTPUT_DIR / f"{stamp}.wav"
    vtt_path = OUTPUT_DIR / f"{stamp}.vtt"

    sf.write(wav_path, wavs[0], sr)

    alignment_sec: float | None = None
    align_token_count: int | None = None
    if use_forced_alignment:
        if forced_aligner is None:
            forced_aligner = load_forced_aligner()
        if use_cuda:
            torch.cuda.synchronize()
        t_align = time.perf_counter()
        align_results = forced_aligner.align(
            audio=str(wav_path),
            text=text,
            language=language,
        )
        if use_cuda:
            torch.cuda.synchronize()
        alignment_sec = time.perf_counter() - t_align
        items = align_results[0].items
        align_token_count = len(items)
        write_webvtt_word_aligned(vtt_path, items)
    else:
        write_webvtt(vtt_path, text, duration_sec)

    shutil.copy2(wav_path, OUTPUT_DIR / "_latest_.wav")
    shutil.copy2(vtt_path, OUTPUT_DIR / "_latest_.vtt")

    print(f"Zapisano: {wav_path}")
    print(f"Zapisano: {vtt_path}")
    print(f"Czas generacji: {generation_sec:.2f} s | długość audio: {duration_sec:.2f} s")
    if alignment_sec is not None:
        print(
            f"Czas forced alignment: {alignment_sec:.2f} s | tokenów w VTT: {align_token_count}"
        )
    if gpu_peak_bytes is not None:
        print(f"GPU peak (VRAM od startu generacji): {gpu_peak_bytes / 1024**3:.2f} GiB")

    return InferOutputs(
        wav_path=wav_path,
        vtt_path=vtt_path,
        generation_sec=generation_sec,
        gpu_peak_bytes=gpu_peak_bytes,
        alignment_sec=alignment_sec,
        align_token_count=align_token_count,
    )


def main() -> None:
    download_all_qwen3_tts_models()

    # list = [
    #     "aiden",
    #     "dylan",
    #     "eric",
    #     "ono_anna",
    #     "ryan",
    #     "serena",
    #     "sohee",
    #     "uncle_fu",
    #     "vivian",
    # ]

    # for voice in list:
    infer(

        # text="Zuza is the best person in the world. I love her....... Jack Sparrow",
        text="Self attention mechanisms, represented by Transformers [1], have driven the advancement of various machine learning problems, including language understanding [2, 3] and computer vision applications [4–6]. Different from classic model architectures such as Convolutional Neural Nets (CNNs) or Recurrent Neural Nets (RNNs), Transformers enable direct interaction between every pair of elements within a sequence, which makes them especially powerful at capturing long term dependencies. However, Transformers require high computational costs. The cause of this challenge is the need to perform attention operations that have quadratic time and space complexity w.r.t. the context size. This makes it difficult for Transformers to scale to inputs with large context sizes. A number of recent works have been dedicated to addressing the scalability issue of Transformers [7–13]. The common idea here is to approximate the full attention operation, with the techniques ranging from sparsity, locality sensitive hashing, low rank decomposition, kernel approximation, etc..",

        model="1.7b-custom-voice",
        # model="1.7b-voice-design",
        # model="1.7b-base",
        # model="0.6b-custom-voice",
        # model="0.6b-base",
        # model="tokenizer",

        # speaker=voice,

        speaker="ryan",
        # speaker="uncle_fu",
        # speaker="sohee",

        language="English",

        strip_delimited_pairs=[("[", "]")],
    )












if __name__ == "__main__":
    main()
