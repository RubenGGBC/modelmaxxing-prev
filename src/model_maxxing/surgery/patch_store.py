from __future__ import annotations

import hashlib
import json
from pathlib import Path

from model_maxxing.surgery.patch_schema import PatchSpec, spec_from_dict, spec_to_dict


def patch_hash(spec: PatchSpec) -> str:
    return hashlib.sha256(json.dumps(spec_to_dict(spec), sort_keys=True).encode()).hexdigest()[:16]


def save_patch(spec: PatchSpec, directory: str = "results/patches") -> str:
    pid = patch_hash(spec)
    Path(directory).mkdir(parents=True, exist_ok=True)
    (Path(directory) / f"{pid}.json").write_text(json.dumps(spec_to_dict(spec), indent=1, sort_keys=True))
    return pid


def load_patch(pid: str, directory: str = "results/patches") -> PatchSpec:
    return spec_from_dict(json.loads((Path(directory) / f"{pid}.json").read_text()))
