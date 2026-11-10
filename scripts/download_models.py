"""Descarga checkpoints en revisiones exactas a models_cache/.

Uso:  python scripts/download_models.py [v03|v04|hito_a|hito_b]   (por defecto v04)
  v03: par 0.5B del spike (banco de pruebas barato).
  v04: par 1.5B del experimento principal (docs/MODEL_SELECTION.md)."""
import sys

from huggingface_hub import snapshot_download

SETS = {
    "v03": {
        "Qwen2.5-0.5B": ("Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987"),
        "Qwen2.5-Coder-0.5B": ("Qwen/Qwen2.5-Coder-0.5B", "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301"),
        "Med-Receiver": ("Rumiii/Qwen2.5-0.5B-Med-Pre-Trained-92k", "915fc914926974078c123044cefbf9c12d639877"),
        "Code-Donor": ("qgallouedec/Qwen2.5-0.5B-codeforces-SFT", "febca9694f5a908e37199740cb92bfe3d70895a1"),
    },
    "v04": {
        "Qwen2.5-1.5B": ("Qwen/Qwen2.5-1.5B", "8faed761d45a263340a0528343f099c05c9a4323"),
        "Qwen2.5-1.5B-Instruct": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
        "Qwen2.5-Coder-1.5B": ("Qwen/Qwen2.5-Coder-1.5B", "df3ce67c0e24480f20468b6ef2894622d69eb73b"),
    },
    # Hito A (APM): ~18,3 GB. Comprobar antes `scripts/hito_a_apm.py configs/hito_a_apm.yaml resources`.
    "hito_a": {
        "Qwen2.5-1.5B": ("Qwen/Qwen2.5-1.5B", "8faed761d45a263340a0528343f099c05c9a4323"),
        "Qwen2.5-7B": ("Qwen/Qwen2.5-7B", "d149729398750b98c0af14eb82c78cfe92750796"),
    },
    # Hito B (Theseus): ~33,6 GB. Comprobar antes `scripts/hito_b_theseus.py configs/hito_b_theseus.yaml resources`.
    "hito_b": {
        "Qwen2.5-1.5B": ("Qwen/Qwen2.5-1.5B", "8faed761d45a263340a0528343f099c05c9a4323"),
        "Qwen2.5-7B": ("Qwen/Qwen2.5-7B", "d149729398750b98c0af14eb82c78cfe92750796"),
        "Qwen2.5-Coder-7B": ("Qwen/Qwen2.5-Coder-7B", "0396a76181e127dfc13e5c5ec48a8cee09938b02"),
    },
}
which = sys.argv[1] if len(sys.argv) > 1 else "v04"
for name, (repo, rev) in SETS[which].items():
    snapshot_download(repo, revision=rev, local_dir=f"models_cache/{name}",
                      allow_patterns=["*.safetensors", "*.json", "merges.txt", "vocab.json", "LICENSE*", "README.md"])
    print("ok", name, flush=True)
