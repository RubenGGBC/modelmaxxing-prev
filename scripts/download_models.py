"""Descarga los tres checkpoints en las revisiones exactas usadas en el spike (models_cache/)."""
from huggingface_hub import snapshot_download

MODELS = {
    "Qwen2.5-0.5B": ("Qwen/Qwen2.5-0.5B", None),
    "Qwen2.5-Coder-0.5B": ("Qwen/Qwen2.5-Coder-0.5B", None),
    "Med-Receiver": ("Rumiii/Qwen2.5-0.5B-Med-Pre-Trained-92k", "915fc914926974078c123044cefbf9c12d639877"),
    "Code-Donor": ("qgallouedec/Qwen2.5-0.5B-codeforces-SFT", "febca9694f5a908e37199740cb92bfe3d70895a1"),
}
for name, (repo, rev) in MODELS.items():
    snapshot_download(repo, revision=rev, local_dir=f"models_cache/{name}",
                      allow_patterns=["*.safetensors", "*.json", "merges.txt", "vocab.json", "LICENSE*", "README.md"])
    print("ok", name)
