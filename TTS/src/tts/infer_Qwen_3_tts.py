import subprocess
import sys
from pathlib import Path

from huggingface_hub import get_cached_repo_tree, login, snapshot_download
from huggingface_hub.errors import CachedRepoTreeNotFoundError


def clear_screen() -> None:
    if sys.platform == "win32":
        subprocess.run(["cmd", "/c", "cls"], check=False)
    else:
        subprocess.run(["clear"], check=False)
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
QWEN3_TTS_REPOS: tuple[str, ...] = (
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
)
def local_model_is_complete(local_dir: Path, repo_id: str) -> bool:
    if not local_dir.is_dir() or not any(local_dir.iterdir()):
        return False
    try:
        expected_files = get_cached_repo_tree(repo_id, local_dir=local_dir)
    except CachedRepoTreeNotFoundError:
        return False
    return all(local_dir.joinpath(*file.path.split("/")).is_file() for file in expected_files)
def download_all_qwen3_tts_models() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    pending: list[tuple[str, Path]] = []
    for repo_id in QWEN3_TTS_REPOS:
        name = repo_id.split("/", 1)[1]
        local_dir = MODELS_DIR / name
        if local_model_is_complete(local_dir, repo_id):
            print(f"OK (already downloaded): {repo_id} -> {local_dir}")
        else:
            pending.append((repo_id, local_dir))

    if not pending:
        print("All Qwen3 TTS models are present.")
        clear_screen()
        return

    login()
    for repo_id, local_dir in pending:
        print(f"Downloading {repo_id} -> {local_dir}")
        snapshot_download(repo_id=repo_id, local_dir=local_dir)

    clear_screen()



def main() -> None:
    download_all_qwen3_tts_models()
    print("Hello from tts!")











if __name__ == "__main__":
    main()
