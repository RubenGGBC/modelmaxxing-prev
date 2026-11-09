"""Uso: python scripts/inspect_models.py BASE_DIR RECEIVER_DIR DONOR_DIR
Escribe results/reports/compatibility.json (solo lee cabeceras, no carga pesos)."""
import json
import sys
from pathlib import Path

from model_maxxing.models.compatibility import check_pair

base, recv, donor = sys.argv[1:4]
report = {
    "receiver_vs_donor": check_pair(recv, donor),
    "base_vs_donor": check_pair(base, donor),
    "base_vs_receiver": check_pair(base, recv),
}
out = Path("results/reports/compatibility.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2))
print(json.dumps({k: v["compatible"] for k, v in report.items()}))
