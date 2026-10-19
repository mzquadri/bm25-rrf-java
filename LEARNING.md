# How this works

Written so the code can be defended in a conversation: what each piece does,
why each constant is the value it is, and where it breaks.

---

## 1. Tokenizing

```java
String lowered = text.toLowerCase(Locale.ROOT);
String joined  = THOUSANDS.matcher(lowered).replaceAll("");  // (?<=\d),(?=\d{3}\b)
// then every match of [a-z0-9]+
```

Three decisions, each with a reason.

**Lowercasing with `Locale.ROOT`, not the default locale.** Under a Turkish
locale, `"I".toLowerCase()` is a dotless `ı`, which is not in `[a-z]` and would
be dropped entirely. Every token containing a capital I would vanish, on that
machine only. `TokenizerTest.isLocaleIndependent` sets the default locale to
`tr-TR` and asserts the tokens anyway.

**Joining thousands separators.** The NFIP forms write the Increased Cost of
Compliance limit as `$30,000`. A question asking about `30000` would otherwise
share no term with the clause that answers it, because the comma would split
`30,000` into `30` and `000`. The lookahead requires exactly three digits
followed by a word boundary, so `44,61` stays two tokens — it is a list of two
numbers, not one number.

**No stopword list and no stemming.** Both are deliberate omissions inherited
from the reference implementation. Stopwords would need a list to maintain and
BM25 already discounts common terms through idf. Stemming would need a stemmer,
which is a dependency and a source of disagreement between languages.

---

## 2. Chunking

Documents are split into 800-character windows overlapping by 120. The cursor
advances by `size - overlap`, so a non-positive step would never terminate —
`Chunker.validate` rejects that rather than hanging, because in the Python
original both numbers come from environment variables.

**Why character windows rather than sentences or paragraphs.** It is the crude
option, and it was chosen on evidence: 800/120 beat 400/80, 600/100, 600/150 and
1000/150 on a dev sweep. Larger chunks keep a provision and the text that
identifies which form it belongs to inside the same window, which is what the
dominant failure mode needed. The three forms are near-duplicates, so a chunk
containing only the provision is ambiguous between them.

**Offsets are shifted when a window is trimmed.** Each window is stripped of
leading and trailing whitespace, and `start` moves forward by however much came
off the front. The invariant is:

```java
document.text().substring(chunk.start(), chunk.end()).equals(chunk.text())
```

That is what lets a citation point back at a character range in the source. The
easy bug is trimming the text and keeping the untrimmed offsets, which produces
citations that are off by a few characters — plausible enough to survive review.

**The chunk id hashes three things:** document id, start offset, and the text.

```java
documentId + "#" + sha256(documentId + "|" + start + "|" + text).substring(0, 12)
```

The document id and offset are in there because the three forms share passages
word for word. Hashing the text alone would give two identical chunks the same
id, and an index keyed by id would silently keep one and lose the other — losing
a whole document's evidence for that passage. `ChunkerTest.idsDoNotCollideAcrossDocuments`
pins this.

---

## 3. BM25

For a query term `t` and chunk `d`:

```
score(d) += idf(t) * ( tf(t,d) * (k1 + 1) ) / ( tf(t,d) + k1 * (1 - b + b * len(d)/avgLen) )
```

Read it as three ideas multiplied together.

**`idf(t)` — how surprising the term is.**

```
idf(t) = max( ln( (N - df + 0.5) / (df + 0.5) + 1 ), 1e-9 )
```

`df` is how many chunks contain the term, `N` the total. A term in few chunks
scores high. The `+0.5`s are the Robertson/Sparck-Jones smoothing, which keeps
the expression finite when `df` is 0 or `N`.

**The `+1` inside the logarithm is what keeps this positive — not the floor.**
Classic Robertson/Sparck-Jones idf, without that `+1`, goes **negative** once a
term appears in more than half the documents, and then containing the term
actively pushes a document down the ranking, which is absurd. With the `+1` the
log's argument always exceeds 1, so the result is always above zero.

That makes the `max(..., 1e-9)` floor **defensive and very nearly dead code**.
This was found by writing a test that asserted the floor *was* reached and
watching it fail:

```
expected: <1.0E-9> but was: <0.13353139262452257>
```

With 3 chunks all containing the term, idf is `ln(1 + 0.5/3.5) = 0.1335`.
The smallest value the expression can take is at `df == N`, which is
`ln(1 + 0.5/(N + 0.5))`, and solving for when that drops below `1e-9` gives
**N > 499,999,999** — about half a billion chunks. Checked rather than reasoned
about:

| N (term in every chunk) | idf | floored? |
|---|---|---|
| 3 | 0.133531392625 | no |
| 314 | 0.00158856268514 | no |
| 1,000,000 | 0.000000499999625 | no |
| 1,000,000,000 | 0.000000000500000041 | **yes** |

The floor is kept anyway, because the Python original has it and parity is the
point. But the reason idf never penalises a document is the `+1`, and the Python
source's comment credits the floor for it. `ubiquitousTermStaysPositive` now
asserts the actual value, and `idfShrinksMonotonically` pins the property the
formula really does guarantee: more documents sharing a term means less
discriminating power, approaching zero from above and never crossing it.

**`tf * (k1+1) / (tf + k1 * ...)` — saturation.** Term frequency has
diminishing returns. The tenth mention of "flood" says far less than the second.
This fraction rises steeply and then flattens, approaching `k1 + 1` as `tf`
grows. With `k1 = 1.5`, saturation sets in after a handful of mentions. A plain
`tf` count would let one long repetitive chunk dominate every query.

**`1 - b + b * len(d)/avgLen` — length normalisation.** One mention in a short
passage is stronger evidence than one mention buried in a long one. `b = 0.75`
applies three quarters of the full correction; `b = 0` would disable it and
`b = 1` would apply it fully. `Bm25IndexTest.lengthNormalisationFavoursShorterChunks`
shows the effect directly.

**k1 = 1.5 and b = 0.75 are the standard defaults and were not tuned.** The
reference repository checked on its dev set that tuning them was not worth it
here, and this port keeps the same values because a different constant would
make the parity test meaningless.

### Chunks with no matching term are absent, not zero

`search` only emits chunks with a score strictly greater than zero. A chunk
sharing no term with the query does not appear at all. This matters for RRF
downstream: a list of ten real hits and a list padded with zero-scored
irrelevant chunks fuse differently, because RRF reads positions and a padded
list hands out rank positions to chunks that matched nothing.

---

## 4. Reciprocal rank fusion

```
total(d) = sum over lists of  1 / (k + rank(d))      k = 60, rank from 1
```

**Why not add the scores.** Dense cosine similarity lives in roughly
`[-1, 1]`; BM25 scores are unbounded positives that depend on corpus
statistics. Adding or averaging them requires inventing a weighting, and any
weighting chosen is a hidden hyperparameter fitted to one corpus. RRF never
looks at a score. `ReciprocalRankFusionTest.ignoresInputScores` multiplies every
input score by a thousand and asserts the output is unchanged.

**What k = 60 does.** It damps the difference between top positions. Rank 1
contributes `1/61` and rank 2 `1/62` — nearly the same. So a chunk that both
retrievers rank second (`1/62 + 1/62`) beats a chunk one retriever ranks first
and the other does not return at all (`1/61`). That is the whole behaviour worth
wanting: **agreement between two different retrievers outweighs confidence from
one.** `agreementBeatsASingleStrongOpinion` is exactly that case.

k = 60 is the value from Cormack et al.'s original paper. It is not tuned here,
and that is a deliberate restraint: with 50 questions, any k fitted on them
would be fitted to noise.

**Ranks start at 1.** Starting at 0 would make the first position worth `1/k`
instead of `1/(k+1)` and change every total. It is a one-character mistake with
no visible symptom, so `ranksAreOneBased` pins the exact value.

---

## 5. Getting two languages to agree

The interesting engineering is here rather than in the algorithms.

**Floating-point addition is not associative.** `(a + b) + c` can differ from
`a + (b + c)` in the last bits. BM25 accumulates one contribution per query
term per document, so the *order* of those additions is part of the result. Two
consequences:

- The loops are term-major then document-major, matching the Python original,
  rather than the document-major shape that would be more natural to write.
- **An inverted index is not used.** It is the obvious optimisation — visit only
  the documents containing a term instead of all of them — and it changes the
  order of the additions, because posting-list order is not index order. Parity
  was the goal, so the slower loop is the correct one. This is a real trade:
  the code is O(terms x documents) when it could be O(postings).

**`StrictMath.log`, not `Math.log`.** `Math.log` is specified to within 1 ulp
and may be replaced by a JIT intrinsic, so it can give different answers on
different JVMs, and even between the interpreted and compiled runs of the same
method. `StrictMath` is defined to be reproducible everywhere.

That still leaves Python's `math.log`, which calls the platform libm and is also
only accurate to an ulp. The two are not guaranteed to agree bit for bit, and
this is the single reason the parity test compares scores with a tolerance of
1e-12 relative. **Ranking order is compared with no tolerance**, which is the
assertion that actually matters — a score difference big enough to reorder two
chunks fails it.

**Whitespace is not the same set in both languages.** See the README section;
the short version is that `Character.isWhitespace` and Python's `str.isspace()`
disagree on U+00A0 and U+0085, and chunk offsets depend on which definition is
used. `PythonText` implements Python's.

**String indices are not the same unit.** Python counts code points, Java
counts UTF-16 code units. For anything outside the BMP they diverge, and every
offset after the first such character would be wrong. The corpus has none, and
the test asserts that instead of relying on it.

---

## 6. What would break this

- **A corpus with a no-break space.** Handled, and the reason `PythonText`
  exists. Without it every offset after the first one shifts.
- **A corpus with an emoji or any non-BMP character.** Not handled. Offsets
  would diverge between the two languages. `ParityTest` fails loudly rather than
  producing wrong citations, which is the intended behaviour, but fixing it
  properly would mean indexing by code point throughout.
- **Re-exporting the fixture on a different machine.** The dense rankings come
  from a sentence-transformer pinned by *name* only, with no revision. A
  different snapshot of `BAAI/bge-small-en-v1.5` produces different dense lists
  and therefore different RRF results. This is not hypothetical: re-running the
  Python chunking sweep on a newer stack moved its metrics while leaving the
  chunking identical. The fixtures are committed for precisely this reason.
- **Adding stopword removal or stemming to one side only.** Rankings would
  diverge immediately. The parity test is what makes that a failure rather than
  a slow drift.
