package de.mzquadri.retrieval;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Okapi BM25 over the corpus chunks.
 *
 * <p>A port of {@code src/retrieval.py:BM25Index}. Defaults are the standard
 * k1=1.5 and b=0.75, untuned, as in the original.
 *
 * <h2>Why this is bit-exact against Python, and what it cost</h2>
 *
 * <p>Both sides use IEEE-754 binary64, so identical operations in an identical
 * order give identical results. Floating-point addition is not associative, so
 * "identical order" is a real constraint rather than a stylistic one: summing
 * the same per-term contributions in a different sequence can land a few
 * units in the last place away, which is enough to swap two near-tied chunks
 * and change the ranking.
 *
 * <p>Two things therefore follow the Python loop structure exactly rather than
 * the shape a JVM author would reach for first:
 *
 * <ul>
 *   <li>The outer loop is over query terms and the inner loop over documents,
 *       not the other way round. Scores accumulate per document across terms in
 *       query order.</li>
 *   <li>Repeated query terms are not de-duplicated. A query containing "flood"
 *       twice adds its contribution twice, because that is what the original
 *       does, and collapsing it would change both the score and the sum order.</li>
 * </ul>
 *
 * <p>An inverted index would be the natural optimisation - iterating only the
 * documents that contain a term, instead of all of them. It is deliberately not
 * used: it visits documents in posting-list order rather than index order, which
 * reorders the additions. Parity was the goal here, so the slower loop is the
 * correct one. {@code ParityTest} is what would catch the change.
 */
public final class Bm25Index {

    public static final double DEFAULT_K1 = 1.5;
    public static final double DEFAULT_B = 0.75;

    private final double k1;
    private final double b;
    private final List<String> chunkIds;
    private final List<Map<String, Integer>> termFrequencies;
    private final int[] lengths;
    private final double averageLength;
    private final Map<String, Double> idf;

    public Bm25Index(List<Chunk> chunks) {
        this(chunks, DEFAULT_K1, DEFAULT_B);
    }

    public Bm25Index(List<Chunk> chunks, double k1, double b) {
        this.k1 = k1;
        this.b = b;
        this.chunkIds = new ArrayList<>(chunks.size());
        this.termFrequencies = new ArrayList<>(chunks.size());
        this.lengths = new int[chunks.size()];

        long totalLength = 0;
        for (int i = 0; i < chunks.size(); i++) {
            Chunk chunk = chunks.get(i);
            chunkIds.add(chunk.chunkId());

            List<String> tokens = Tokenizer.tokenize(chunk.text());
            // LinkedHashMap, not HashMap: the counts are iterated when scoring,
            // and a deterministic iteration order keeps a run reproducible even
            // though the final sort does not depend on it.
            Map<String, Integer> counts = new LinkedHashMap<>();
            for (String token : tokens) {
                counts.merge(token, 1, Integer::sum);
            }
            termFrequencies.add(counts);

            lengths[i] = tokens.size();
            totalLength += tokens.size();
        }

        // Guarded for an empty corpus, which would otherwise be 0/0 = NaN and
        // poison every score silently rather than failing.
        this.averageLength = chunks.isEmpty() ? 0.0 : (double) totalLength / chunks.size();

        Map<String, Integer> documentFrequency = new HashMap<>();
        for (Map<String, Integer> counts : termFrequencies) {
            for (String term : counts.keySet()) {
                documentFrequency.merge(term, 1, Integer::sum);
            }
        }

        int total = termFrequencies.size();
        this.idf = new HashMap<>(documentFrequency.size() * 2);
        for (Map.Entry<String, Integer> entry : documentFrequency.entrySet()) {
            int count = entry.getValue();
            // Robertson/Sparck-Jones idf with the +0.5 smoothing and a +1 inside
            // the logarithm, then floored at a small positive value.
            //
            // The +1 is what keeps this positive. Classic RSJ idf goes negative
            // once a term appears in more than half the documents, which would
            // make containing the term actively push a document down the
            // ranking. With the +1 the log's argument always exceeds 1, so the
            // result is always above zero.
            //
            // The floor is therefore defensive and very nearly dead code: the
            // smallest value this can take is at df == N, which is
            // ln(1 + 0.5/(N + 0.5)), and that only falls below 1e-9 past about
            // 500 million chunks. It is kept because the Python original has it
            // and parity is the point, not because it is reachable here.
            //
            // StrictMath, not Math. Math.log is permitted 1 ulp of error and may
            // be replaced by a JIT intrinsic, so it can differ between JVMs and
            // even between interpreted and compiled runs of this same code.
            // StrictMath is defined to give the same answer everywhere, which is
            // what a parity test needs. Python's math.log calls the platform
            // libm and so may still sit an ulp away; that residue is why
            // ParityTest compares scores within a tolerance while requiring the
            // ranking itself to match exactly.
            double value = StrictMath.log((total - count + 0.5) / (count + 0.5) + 1.0);
            this.idf.put(entry.getKey(), Math.max(value, 1e-9));
        }
    }

    public int size() {
        return chunkIds.size();
    }

    public double averageLength() {
        return averageLength;
    }

    /** The inverse document frequency of a term, or 0 if it is not in the index. */
    public double idfOf(String term) {
        return idf.getOrDefault(term, 0.0);
    }

    /**
     * @param query the raw query text, tokenised the same way documents are
     * @param limit how many hits to return
     * @return hits with a positive score, best first, ties broken by chunk id
     */
    public List<Hit> search(String query, int limit) {
        double[] scores = new double[chunkIds.size()];

        for (String term : Tokenizer.tokenize(query)) {
            Double termIdf = idf.get(term);
            if (termIdf == null) {
                continue;
            }
            for (int index = 0; index < termFrequencies.size(); index++) {
                Integer frequency = termFrequencies.get(index).get(term);
                if (frequency == null || frequency == 0) {
                    continue;
                }
                double normalisation = 1 - b
                        + b * (averageLength == 0.0 ? 0.0 : lengths[index] / averageLength);
                scores[index] += termIdf
                        * (frequency * (k1 + 1) / (frequency + k1 * normalisation));
            }
        }

        List<Hit> ranked = new ArrayList<>();
        for (int i = 0; i < scores.length; i++) {
            // Strictly greater than zero, matching the original. A chunk sharing
            // no term with the query is absent rather than present with 0.
            if (scores[i] > 0) {
                ranked.add(new Hit(chunkIds.get(i), scores[i]));
            }
        }

        // Score descending, then chunk id ascending, so ties resolve identically
        // on every run and on both implementations. Chunk ids are ASCII, so
        // Java's UTF-16 ordering and Python's code-point ordering agree.
        ranked.sort(Comparator.comparingDouble(Hit::score).reversed()
                .thenComparing(Hit::chunkId));

        // Copied, not a subList view. A view keeps a reference to the full
        // ranking and writes through to it, so a caller trimming or sorting the
        // result would be mutating state this method still owns.
        return List.copyOf(ranked.subList(0, Math.min(limit, ranked.size())));
    }
}
