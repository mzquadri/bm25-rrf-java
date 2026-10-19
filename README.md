# bm25-rrf-java

Okapi BM25 and reciprocal rank fusion on the JVM, held to the output of an
existing Python implementation by a parity test over 50 real queries.

Built with AI assistance (Claude Code).

The point of this repository is not that BM25 can be written in Java. It is that
**two independent implementations of the same retrieval pipeline return the same
rankings on the same corpus**, and that the test suite will say so or fail. The
Python side lives in [insureassist-rag-mlops](https://github.com/mzquadri/insureassist-rag-mlops)
and is the reference; nothing here defines what "correct" means.

## What it does

| Class | What it is |
|---|---|
| `Tokenizer` | lowercase, join thousands separators inside numbers, split on runs of non-alphanumerics |
| `PythonText` | Python's `str.strip()` whitespace set, which is not Java's |
| `Chunker` | fixed-size overlapping character windows, with offsets that address the source and a content-addressed chunk id |
| `Bm25Index` | Okapi BM25, k1=1.5, b=0.75, Robertson/Sparck-Jones idf with a positive floor |
| `ReciprocalRankFusion` | combines ranked lists by position, `1/(k+rank)`, k=60 |

No dependencies in `src/main`. BM25 and RRF need arithmetic, a hash and a
regex; a retrieval library here would hide the thing the repository exists to
show. JUnit 5 and Jackson are test-scoped only.

## Run it

Needs JDK 21 and Maven.

```bash
mvn test
```

Or without a local JDK:

```bash
docker run --rm -v "$PWD":/work -w /work maven:3-eclipse-temurin-21 mvn -q test
```

## The parity test

`ParityTest` reads two fixtures under `src/test/resources/parity/`, produced by
`tools/export_python_reference.py` run against a checkout of the Python
repository:

```bash
python tools/export_python_reference.py --insureassist ../insureassist-rag-mlops
```

Java does **not** read the exported chunks and trust them. It loads the three
corpus files, verifies each against the SHA-256 in the manifest, chunks them
itself, builds its own index, and compares. A chunking difference therefore
fails as a chunk-level mismatch rather than surfacing later as a confusing BM25
discrepancy.

| Compared | How |
|---|---|
| Chunk count, ids, offsets, text | exactly, all 314 chunks |
| BM25 ranking order and ids | exactly, 50 questions at depth 20 |
| RRF fusion order and ids | exactly, 50 questions at depth 10 |
| Scores | within a relative tolerance of 1e-12 |

### Why scores get a tolerance and rankings do not

Both languages use IEEE-754 binary64, so the arithmetic is reproducible, and the
implementation goes out of its way to keep it that way:

- The BM25 loops are term-major then document-major, matching the Python loop
  order. Floating-point addition is not associative, so summing the same
  contributions in a different sequence can land a few units in the last place
  away, which is enough to swap two near-tied chunks.
- An inverted index is deliberately **not** used. It would visit documents in
  posting-list order rather than index order, reordering the additions. Parity
  was the goal, so the slower loop is the correct one here.
- `StrictMath.log`, not `Math.log`. `Math.log` is permitted 1 ulp of error and
  may be swapped for a JIT intrinsic, so it can differ between JVMs and even
  between interpreted and compiled runs of the same code.
- Repeated query terms are not collapsed, because the Python side does not
  collapse them.

What remains is the logarithm itself: Python's `math.log` calls the platform
libm and Java uses `StrictMath.log`. Both are correct to within an ulp and
neither is guaranteed to match the other's bits. That residue is the only reason
scores are compared with a tolerance. **The ranking order is asserted with no
tolerance at all**, so a score difference large enough to reorder two chunks
fails the test; the tolerance cannot hide a real divergence.

### Two cross-language traps the tests pin

Both were found by writing the port, not by it breaking later.

**Whitespace.** `Chunker` trims each window and shifts its offsets by what came
off the front, so which characters count as whitespace decides every offset and
every chunk id. Java's `String.strip()` uses `Character.isWhitespace`, which
disagrees with Python's `str.isspace()`: U+00A0 no-break space and U+0085 next
line are whitespace to Python and not to Java, and U+2007 figure space is the
reverse for `Character.isSpaceChar`. `PythonText` implements Python's set. The
NFIP corpus happens to contain only U+0020 and U+000A, so `String.strip()` would
have passed — the class exists so parity is a property of the code rather than
of one input.

**String indexing.** Python indexes by code point, Java by UTF-16 code unit.
They agree only while every character is in the Basic Multilingual Plane. The
corpus satisfies that — its only non-ASCII characters are the em dash and curly
quotes — and `ParityTest` asserts it rather than assuming it, because a
surrogate pair anywhere would shift every later offset and the failure would
appear as an unexplained chunk-id mismatch.

## The corpus

The three NFIP Standard Flood Insurance Policy forms, as published in
**44 CFR Part 61**, in `src/test/resources/corpus/`.

| | |
|---|---|
| Source | [eCFR, title 44, part 61](https://www.ecfr.gov/current/title-44/chapter-I/subchapter-B/part-61) |
| API used | `https://www.ecfr.gov/api/versioner/v1/full/2026-06-22/title-44.xml?part=61` |
| eCFR issue date | 2026-06-22 |
| Retrieved | 2026-08-22 |
| Licence | 17 U.S.C. 105 — a work of the United States Government, not subject to copyright protection in the United States |
| Attribution | Federal Emergency Management Agency, 44 CFR Part 61, Appendices A(1), A(2) and A(3) |
| Size | 3 documents, 212,856 characters, 314 chunks at 800/120 |

Each file's SHA-256 is recorded in `manifest.json` and checked by the test
before anything is measured, so a silently edited corpus fails rather than
quietly changing the numbers. Extraction was done by `scripts/fetch_nfip_corpus.py`
in the Python repository; no text is paraphrased, reordered or altered.

The 50 queries are the hand-authored NFIP question set from the same repository.

**Why these documents are a useful test case:** the three forms are
near-duplicates by design. They share a skeleton and much verbatim wording but
differ in substance, so a retriever matching on topic finds the right provision
in the wrong form. That is also why `Chunker` hashes the document id and the
offset into the chunk id: hashing the text alone collides across these three
documents and silently merges distinct chunks.

## What this is not

- Not a search engine. There is no index persistence, no incremental update and
  no query parser.
- Not the dense half of the pipeline. The dense ranking is exported from Python
  as a fixture; no transformer runs on the JVM. RRF reads rank positions only,
  so a fixed list is a faithful input to it, but the port covers the two
  deterministic stages, not the embedding.
- Not a performance claim. The BM25 loop is deliberately the slower of two
  correct shapes, for the reason given above.

See [LEARNING.md](LEARNING.md) for how BM25 and RRF work and why each constant
is the value it is.
