# Model weights: https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors
# Model Config: https://huggingface.co/openai-community/gpt2/resolve/main/config.json
# Vocab: https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json
# Merges: https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt
# Tokenizer: https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json
#
# download all of these in `gpt2_model` directory
# skip if already present
#
#
from pathlib import Path
import requests


def download_data(url: str, save_path: Path):
    if save_path.exists():
        print(f"Skipping download of {save_path.name} as it already exists.")
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {save_path.name}")
    response = requests.get(url)
    with open(save_path, "wb") as f:
        f.write(response.content)


TO_DOWNLOAD = {
    "model.safetensors": "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors",
    "config.json": "https://huggingface.co/openai-community/gpt2/resolve/main/config.json",
    "vocab.json": "https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json",
    "merges.txt": "https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt",
    "tokenizer.json": "https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json",
}

if __name__ == "__main__":
    for filename, url in TO_DOWNLOAD.items():
        download_data(url, Path("gpt2_model") / filename)
