"""Export the Python implementation's rankings, so the Java port can be held to them.

This is the only thing that defines "correct" for the Java code in this
repository: whatever `insureassist-rag-mlops` produces is the reference, and the
JUnit parity test fails if Java disagrees. Nothing here reimplements retrieval -
it imports the real modules from a checkout of that repository and records what
they return.

    python tools/export_python_reference.py --insureassist ../insureassist-rag-mlops

Writes two files under src/test/resources/parity/:

  chunks.json     the chunked corpus at 800/120, so Java chunks the same input
  rankings.json   per question: the BM25 ranking, the dense ranking, and the
                  RRF fusion of the two

Why the dense ranking is exported as data rather than recomputed in Java: it
needs a sentence-transformer, and there is no intention of running a transformer
on the JVM here. Java recomputes BM25 and RRF - the two deterministic, pure
pieces - and treats the dense list as a fixed input, which is all RRF needs
since it reads rank positions only.

Scores are written with repr() so the comparison is against the exact double
Python produced, not a rounded decimal.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "src" / "test" / "resources" / "parity"

#: The configuration eval/retrieval_config.json freezes in insureassist.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60
CANDIDATE_DEPTH = 20
FUSED_LIMIT = 10


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--insureassist",
        required=True,
        type=Path,
        help="path to an insureassist-rag-mlops checkout",
    )
    parser.add_argument(
        "--skip-dense",
        action="store_true",
        help="omit the dense rankings, leaving only BM25 to compare",
    )
    args = parser.parse_args(argv)

    repo = args.insureassist.resolve()
    if not (repo / "src" / "retrieval.py").exists():
        print(f"no src/retrieval.py under {repo}", file=sys.stderr)
        return 2

    # Prepended, not appended: this repository has no src/ of its own, but an
    # appended path would let any same-named module already importable win.
    sys.path.insert(0, str(repo))

    from src.corpus import chunk_corpus, corpus_hash, load_corpus, verify_document_hashes
    from src.retrieval import BM25Index, Hit, reciprocal_rank_fusion, tokenize

    documents = load_corpus(repo / "data" / "corpus")

    # The corpus is the input to every number below. If it has changed, the
    # fixture would silently describe a different corpus than its manifest says.
    mismatched = verify_document_hashes(documents)
    if mismatched:
        print(f"corpus does not match its recorded hashes: {mismatched}", file=sys.stderr)
        return 1

    chunks = chunk_corpus(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    index = BM25Index(chunks, k1=BM25_K1, b=BM25_B)

    questions = [
        json.loads(line)
        for line in (repo / "eval" / "ground_truth" / "nfip_questions.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    ]

    # Everything the dense arm needs, or None when it is skipped. One optional
    # bundle rather than several separately-optional names, so there is no way to
    # reach the embedding path with only some of them bound.
    dense = None
    if not args.skip_dense:
        import numpy as np

        from src.config import cfg
        from src.providers import get_embedder

        embedder = get_embedder()
        dense = {
            "cfg": cfg,
            "embedder": embedder,
            "passages": np.asarray(
                embedder.encode([c.text for c in chunks], normalize_embeddings=True),
                dtype=np.float64,
            ),
        }

    rankings = []
    for question in questions:
        text = question["question"]
        bm25_hits = index.search(text, CANDIDATE_DEPTH)

        entry = {
            "question_id": question["question_id"],
            "question": text,
            "bm25": [{"chunk_id": h.chunk_id, "score": repr(h.score)} for h in bm25_hits],
        }

        if dense is not None:
            import numpy as np

            query = np.asarray(
                dense["embedder"].encode(
                    [dense["cfg"].BGE_QUERY_PREFIX + text], normalize_embeddings=True
                ),
                dtype=np.float64,
            )[0]
            similarities = dense["passages"] @ query
            # Same tie-break as BM25Index.search: score descending, then
            # chunk_id ascending, so a tie cannot reorder between runs.
            order = sorted(
                range(len(chunks)),
                key=lambda i: (-float(similarities[i]), chunks[i].chunk_id),
            )[:CANDIDATE_DEPTH]
            dense_hits = [
                Hit(chunk_id=chunks[i].chunk_id, score=float(similarities[i])) for i in order
            ]
            entry["dense"] = [{"chunk_id": h.chunk_id, "score": repr(h.score)} for h in dense_hits]
            fused = reciprocal_rank_fusion([bm25_hits, dense_hits], k=RRF_K, limit=FUSED_LIMIT)
            entry["rrf"] = [{"chunk_id": h.chunk_id, "score": repr(h.score)} for h in fused]

        rankings.append(entry)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stamp = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": platform.python_version(),
        "corpus_hash": corpus_hash(documents),
        "chunking": {"size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP},
        "bm25": {"k1": BM25_K1, "b": BM25_B},
        "rrf": {"k": RRF_K, "candidate_depth": CANDIDATE_DEPTH, "limit": FUSED_LIMIT},
        "tokenizer_probe": tokenize("Up to $30,000 under Coverage D (ICC); see 44 CFR 61."),
    }
    if dense is not None:
        import sentence_transformers

        stamp["dense"] = {
            "embedding_model": dense["cfg"].EMBEDDING_MODEL,
            "query_prefix": dense["cfg"].BGE_QUERY_PREFIX,
            "sentence_transformers": sentence_transformers.__version__,
            "note": (
                "Exported as a fixture, not recomputed on the JVM. The model is pinned by "
                "name only, so re-exporting on a different stack can change these lists; "
                "that is why they are committed."
            ),
        }

    (OUT_DIR / "chunks.json").write_text(
        json.dumps(
            {
                "stamp": stamp,
                "chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "document_id": c.document_id,
                        "chunk_index": c.chunk_index,
                        "start": c.start,
                        "end": c.end,
                        "text": c.text,
                    }
                    for c in chunks
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (OUT_DIR / "rankings.json").write_text(
        json.dumps({"stamp": stamp, "rankings": rankings}, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"{len(chunks)} chunks, {len(rankings)} questions -> {OUT_DIR}")
    print(f"corpus_hash {stamp['corpus_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
