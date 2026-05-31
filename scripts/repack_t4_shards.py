"""Repack the millions of tiny t4 parquet files into a few large shards.

The t4 pretraining data (`datasets/t4/datas/chunk-*/*.parquet`) is ~4-6 million
files of ~80KB each. One file == one logical table == one in-context task, and
tables have HETEROGENEOUS schemas (different column sets), so they cannot be
concatenated column-wise. The per-file open + parquet-footer cost, multiplied by
millions, dominates training I/O, and enumerating them (rglob) at startup is slow.

This tool packs each chunk's tables into a small number of large **Arrow IPC
("feather") shards**. Each shard is an Arrow IPC file whose rows are individual
tables stored as an opaque blob:

    table_id (str) | source_path (str) | source_chunk (str) |
    num_rows (int) | num_cols (int) | table_bytes (binary, feather of the table)

Reading a shard streams these records and deserializes each `table_bytes` back to
the exact same Arrow table (and thus the same pandas DataFrame) the loader would
have gotten from the original parquet file. File count drops ~1000x.

Properties:
  * Parallel over chunks (one task per chunk-*/ directory).
  * Resumable / idempotent: a chunk whose `_parts/<chunk>.json` exists is skipped.
    Shards and part-manifests are written to a temp name then atomically renamed.
  * Default keeps ALL tables (matches the raw loader treating every file as an
    independent task). `--dedup-by-hash` keeps one file per table-hash *within a
    chunk* (filenames are `<hash>=__<uuid>.parquet`); this CHANGES the data
    distribution, so it is opt-in.

Usage:
    python scripts/repack_t4_shards.py \
        --input-root datasets/t4/datas \
        --output-root datasets/t4/shards \
        --shard-size-mb 384 --workers 16
    # smoke test on the first 2 chunks:
    python scripts/repack_t4_shards.py --input-root datasets/t4/datas \
        --output-root /tmp/t4_shards_smoke --workers 8 --limit-chunks 2

The reader side lives in sap_rpt_oss/data/ds.py (RPTShardDataset), driven by
`<output-root>/manifest.json`.
"""

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

BLOB_FORMAT = "feather"
MANIFEST_VERSION = 1

# Shard record schema. large_binary so a shard's blob column can exceed 2GB.
SHARD_SCHEMA = pa.schema(
    [
        ("table_id", pa.string()),
        ("source_path", pa.string()),
        ("source_chunk", pa.string()),
        ("num_rows", pa.int64()),
        ("num_cols", pa.int64()),
        ("table_bytes", pa.large_binary()),
    ]
)


def serialize_table_to_feather(table: pa.Table) -> bytes:
    """Serialize an Arrow table to Arrow-IPC-file ("feather") bytes."""
    sink = pa.BufferOutputStream()
    with pa.ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _write_shard(shard_path: Path, records: list[dict]) -> None:
    """Atomically write one shard (tmp file + rename)."""
    arrays = {name: [r[name] for r in records] for name in SHARD_SCHEMA.names}
    batch = pa.table(arrays, schema=SHARD_SCHEMA)
    tmp_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
    with pa.ipc.new_file(tmp_path, SHARD_SCHEMA) as writer:
        writer.write_table(batch)
    os.replace(tmp_path, shard_path)


def process_chunk(
    chunk_dir_str: str,
    output_root_str: str,
    shard_size_bytes: int,
    dedup_by_hash: bool,
) -> dict:
    """Repack one chunk directory into shards; return its part-manifest dict.

    Idempotent: if the part-manifest already exists it is reused verbatim.
    """
    chunk_dir = Path(chunk_dir_str)
    output_root = Path(output_root_str)
    chunk_name = chunk_dir.name
    parts_dir = output_root / "_parts"
    part_path = parts_dir / f"{chunk_name}.json"

    if part_path.exists():
        try:
            part = json.loads(part_path.read_text())
            part["_reused"] = True  # transient (not persisted) — for logging only
            return part
        except json.JSONDecodeError:
            pass  # corrupt part — reprocess below

    parts_dir.mkdir(parents=True, exist_ok=True)
    # Clean any leftover temp files from an interrupted prior run of this chunk.
    for stale in output_root.glob(f"shard-{chunk_name}-*.arrow.tmp"):
        stale.unlink(missing_ok=True)

    files = sorted(chunk_dir.glob("*.parquet"))
    seen_hashes: set[str] = set()
    shards_info: list[dict] = []
    buf: list[dict] = []
    buf_bytes = 0
    shard_idx = 0
    n_tables = 0
    n_skipped = 0

    def flush() -> None:
        nonlocal buf, buf_bytes, shard_idx
        if not buf:
            return
        shard_name = f"shard-{chunk_name}-{shard_idx:04d}.arrow"
        _write_shard(output_root / shard_name, buf)
        shards_info.append({"path": shard_name, "num_tables": len(buf)})
        shard_idx += 1
        buf = []
        buf_bytes = 0

    for f in files:
        if dedup_by_hash:
            table_hash = f.name.split("=__", 1)[0]
            if table_hash in seen_hashes:
                continue
            seen_hashes.add(table_hash)
        try:
            table = pq.read_table(f)
        except Exception:  # noqa: BLE001 — mirror loader's skip-on-read-error
            n_skipped += 1
            continue
        if table.num_rows == 0 or table.num_columns == 0:
            n_skipped += 1
            continue
        try:
            blob = serialize_table_to_feather(table)
        except Exception:  # noqa: BLE001
            n_skipped += 1
            continue
        buf.append(
            {
                "table_id": f.stem,
                "source_path": str(f.relative_to(chunk_dir.parent)),
                "source_chunk": chunk_name,
                "num_rows": int(table.num_rows),
                "num_cols": int(table.num_columns),
                "table_bytes": blob,
            }
        )
        buf_bytes += len(blob)
        n_tables += 1
        if buf_bytes >= shard_size_bytes:
            flush()
    flush()

    part = {
        "chunk": chunk_name,
        "shards": shards_info,
        "num_tables": n_tables,
        "num_skipped": n_skipped,
        "dedup_by_hash": dedup_by_hash,
    }
    tmp_part = part_path.with_suffix(".json.tmp")
    tmp_part.write_text(json.dumps(part))
    os.replace(tmp_part, part_path)  # atomic "done" signal
    return part


def assemble_manifest(output_root: Path, input_root: Path, args) -> dict:
    """Aggregate all _parts/*.json into a single manifest.json."""
    parts_dir = output_root / "_parts"
    shards: list[dict] = []
    total_tables = 0
    total_skipped = 0
    for part_file in sorted(parts_dir.glob("chunk-*.json")):
        part = json.loads(part_file.read_text())
        for s in part["shards"]:
            shards.append({**s, "chunk": part["chunk"]})
        total_tables += part["num_tables"]
        total_skipped += part.get("num_skipped", 0)

    manifest = {
        "version": MANIFEST_VERSION,
        "blob_format": BLOB_FORMAT,
        "input_root": str(input_root),
        "shard_size_mb": args.shard_size_mb,
        "dedup_by_hash": args.dedup_by_hash,
        "num_shards": len(shards),
        "num_tables": total_tables,
        "num_skipped": total_skipped,
        "shards": shards,
    }
    manifest_path = output_root / "manifest.json"
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="datasets/t4/datas")
    parser.add_argument("--output-root", default="datasets/t4/shards")
    parser.add_argument(
        "--shard-size-mb",
        type=int,
        default=384,
        help="Approximate uncompressed blob bytes per shard before flushing.",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--dedup-by-hash",
        action="store_true",
        help="Keep one file per table-hash within each chunk (changes the data "
        "distribution; off by default).",
    )
    parser.add_argument(
        "--limit-chunks",
        type=int,
        default=None,
        help="Process only the first N chunks (for smoke tests).",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    if not input_root.is_dir():
        sys.exit(f"input root does not exist: {input_root}")

    chunk_dirs = sorted(
        d for d in input_root.iterdir() if d.is_dir() and d.name.startswith("chunk-")
    )
    if not chunk_dirs:
        sys.exit(f"no chunk-* directories found under {input_root}")
    if args.limit_chunks is not None:
        chunk_dirs = chunk_dirs[: args.limit_chunks]

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "_parts").mkdir(exist_ok=True)
    shard_size_bytes = args.shard_size_mb * 1024 * 1024

    print(
        f"[repack] {len(chunk_dirs)} chunks → {output_root} "
        f"(shard ~{args.shard_size_mb}MB, workers={args.workers}, "
        f"dedup_by_hash={args.dedup_by_hash})"
    )

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                process_chunk,
                str(d),
                str(output_root),
                shard_size_bytes,
                args.dedup_by_hash,
            ): d
            for d in chunk_dirs
        }
        for fut in as_completed(futures):
            chunk_dir = futures[fut]
            try:
                part = fut.result()
            except Exception:  # noqa: BLE001
                print(f"[repack] FAILED chunk {chunk_dir.name}:\n{traceback.format_exc()}")
                continue
            done += 1
            reused = " (reused)" if part.get("_reused") else ""
            print(
                f"[repack] ({done}/{len(chunk_dirs)}) {part['chunk']}: "
                f"{len(part['shards'])} shards, {part['num_tables']} tables, "
                f"{part['num_skipped']} skipped{reused}"
            )

    manifest = assemble_manifest(output_root, input_root, args)
    print(
        f"[repack] DONE: {manifest['num_shards']} shards, "
        f"{manifest['num_tables']} tables, {manifest['num_skipped']} skipped → "
        f"{output_root / 'manifest.json'}"
    )


if __name__ == "__main__":
    main()
