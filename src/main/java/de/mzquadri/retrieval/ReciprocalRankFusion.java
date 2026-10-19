package de.mzquadri.retrieval;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Combines ranked lists by rank position rather than by score.
 *
 * <p>A port of {@code src/retrieval.py:reciprocal_rank_fusion}.
 *
 * <p>Dense cosine similarity and BM25 scores are on different, unnormalised
 * scales, so they cannot be added or averaged without inventing a weighting.
 * RRF reads positions only: each list contributes {@code 1 / (k + rank)}, which
 * needs no calibration between the two systems and no per-corpus tuning.
 *
 * <p>k = 60 is the value from Cormack et al. and is left alone. It damps the
 * difference between the top positions: with k this large, rank 1 and rank 2
 * contribute 1/61 and 1/62, so a chunk needs support from more than one
 * retriever to climb rather than winning on a single list's confidence.
 */
public final class ReciprocalRankFusion {

    public static final int DEFAULT_K = 60;

    private ReciprocalRankFusion() {
    }

    public static List<Hit> fuse(List<List<Hit>> rankings, int limit) {
        return fuse(rankings, DEFAULT_K, limit);
    }

    /**
     * @param rankings ranked lists, best first, in a fixed order
     * @param k        the rank damping constant
     * @param limit    how many fused hits to return
     */
    public static List<Hit> fuse(List<List<Hit>> rankings, int k, int limit) {
        // LinkedHashMap so the accumulation order is the lists' order. The final
        // sort makes the output order independent of this, but the order the
        // additions happen in is not cosmetic: floating-point addition is not
        // associative, so a chunk appearing in several lists must receive its
        // contributions in the same sequence as the Python original for the
        // totals to agree bit for bit.
        Map<String, Double> totals = new LinkedHashMap<>();

        for (List<Hit> ranking : rankings) {
            int position = 1;
            for (Hit hit : ranking) {
                // Ranks are 1-based. Starting at 0 would make the first
                // position contribute 1/k instead of 1/(k+1) and quietly
                // change every fused total.
                totals.merge(hit.chunkId(), 1.0 / (k + position), Double::sum);
                position++;
            }
        }

        List<Hit> ranked = new ArrayList<>(totals.size());
        for (Map.Entry<String, Double> entry : totals.entrySet()) {
            ranked.add(new Hit(entry.getKey(), entry.getValue()));
        }

        // Same tie-break as BM25: total descending, then chunk id ascending.
        ranked.sort(Comparator.comparingDouble(Hit::score).reversed()
                .thenComparing(Hit::chunkId));

        // Copied rather than returned as a subList view, for the same reason as
        // in Bm25Index.search: a view writes through to the full ranking.
        return List.copyOf(ranked.subList(0, Math.min(limit, ranked.size())));
    }
}
