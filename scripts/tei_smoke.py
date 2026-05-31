"""Validate the TEI embedding backend and its relationship to the local one.

Run AFTER starting the server (scripts/tei_run.sh):
    /home/amax/.conda/envs/rpt/bin/python scripts/tei_smoke.py
    python scripts/tei_smoke.py --base-url http://localhost:8080/v1 --no-local

Checks: (1) TEI responds via the OpenAI client and returns the expected dim;
(2) how TEI vectors relate to the local SentenceEmbedder — for MiniLM, TEI
L2-normalizes while local does not, so the *directions* should match (cosine
~1.0) while magnitudes differ.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sap_rpt_oss.data.sentence_embedder import RemoteEmbedder  # noqa: E402

SAMPLES = [
    "customer_age",
    "United States",
    "2024-01-15",
    "The quick brown fox jumps over the lazy dog.",
    "42.7",
    "",  # empty — RemoteEmbedder substitutes a space
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8080/v1")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--no-local", action="store_true", help="Skip local comparison.")
    args = ap.parse_args()

    remote = RemoteEmbedder(args.model, base_url=args.base_url)
    try:
        tei = remote.embed(SAMPLES)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"TEI request failed (is scripts/tei_run.sh running?): {exc}")

    tei = np.asarray(tei, dtype=np.float32)
    print(f"TEI: shape={tei.shape} dtype set to float16 in pipeline")
    print(f"TEI per-vector L2 norms: {np.round(np.linalg.norm(tei, axis=1), 4)}")

    if args.no_local:
        print("OK: TEI reachable, returns embeddings.")
        return

    from sap_rpt_oss.data.sentence_embedder import SentenceEmbedder

    local = SentenceEmbedder(args.model, device="cpu").embed(SAMPLES)
    local = np.asarray(local, dtype=np.float32)
    print(f"local per-vector L2 norms: {np.round(np.linalg.norm(local, axis=1), 4)}")

    cos = np.sum(tei * local, axis=1) / (
        np.linalg.norm(tei, axis=1) * np.linalg.norm(local, axis=1) + 1e-9
    )
    print(f"cosine(TEI, local) per sample: {np.round(cos, 4)}")
    print(f"mean cosine = {cos.mean():.4f}")
    if cos.mean() > 0.999:
        print(
            "OK: TEI == L2-normalized(local). Same underlying embedding; the only "
            "difference is normalization. Use one backend consistently for "
            "train + inference."
        )
    else:
        print(
            "WARNING: directions differ — pooling/model mismatch. Check the TEI "
            "--pooling flag and model id."
        )


if __name__ == "__main__":
    main()
