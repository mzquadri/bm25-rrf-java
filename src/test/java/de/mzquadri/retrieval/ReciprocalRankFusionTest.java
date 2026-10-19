package de.mzquadri.retrieval;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class ReciprocalRankFusionTest {

    private static List<String> ids(List<Hit> hits) {
        List<String> out = new ArrayList<>(hits.size());
        for (Hit hit : hits) {
            out.add(hit.chunkId());
        }
        return out;
    }

    @Test
    @DisplayName("a chunk ranked by both retrievers beats one ranked first by only one")
    void agreementBeatsASingleStrongOpinion() {
        // The property RRF exists for. "both" is second on each list; "solo" is
        // first on one and absent from the other. With k = 60 the two seconds
        // (1/62 + 1/62) outweigh one first (1/61).
        List<Hit> dense = List.of(new Hit("solo", 0.9), new Hit("both", 0.8));
        List<Hit> bm25 = List.of(new Hit("other", 12.0), new Hit("both", 11.0));

        List<Hit> fused = ReciprocalRankFusion.fuse(List.of(dense, bm25), 10);

        assertEquals("both", fused.get(0).chunkId());
        assertEquals(2.0 / 62.0, fused.get(0).score(), 1e-15);
    }

    @Test
    @DisplayName("ranks are 1-based, so the first position contributes 1/(k+1)")
    void ranksAreOneBased() {
        // Starting at 0 would make the first position worth 1/k and change
        // every total. Pinned because it is an easy off-by-one to reintroduce.
        List<Hit> only = List.of(new Hit("a", 5.0));

        List<Hit> fused = ReciprocalRankFusion.fuse(List.of(only), 60, 10);

        assertEquals(1.0 / 61.0, fused.get(0).score(), 1e-15);
    }

    @Test
    @DisplayName("input scores are ignored entirely")
    void ignoresInputScores() {
        // The whole point: dense cosine and BM25 scores are on incomparable
        // scales, so only positions are read. Inflating a score must change
        // nothing.
        List<Hit> modest = List.of(new Hit("a", 0.01), new Hit("b", 0.009));
        List<Hit> enormous = List.of(new Hit("a", 9999.0), new Hit("b", 9998.0));

        assertEquals(
                ReciprocalRankFusion.fuse(List.of(modest), 10),
                ReciprocalRankFusion.fuse(List.of(enormous), 10));
    }

    @Test
    @DisplayName("ties are broken by chunk id")
    void tiesBreakOnChunkId() {
        List<Hit> first = List.of(new Hit("zzz", 1.0));
        List<Hit> second = List.of(new Hit("aaa", 1.0));

        List<Hit> fused = ReciprocalRankFusion.fuse(List.of(first, second), 10);

        assertEquals(List.of("aaa", "zzz"), ids(fused));
        assertEquals(fused.get(0).score(), fused.get(1).score(), 0.0);
    }

    @Test
    @DisplayName("the order the lists are given in does not change the result")
    void isOrderIndependentAcrossLists() {
        // Fusion is a sum, and the final sort is total-then-id, so swapping the
        // lists must not move anything. Worth pinning: if it ever does, the
        // accumulation has grown a dependence on list order that the Python
        // side does not have.
        List<Hit> dense = List.of(new Hit("a", 0.9), new Hit("b", 0.8), new Hit("c", 0.7));
        List<Hit> bm25 = List.of(new Hit("c", 9.0), new Hit("a", 8.0));

        assertEquals(
                ids(ReciprocalRankFusion.fuse(List.of(dense, bm25), 10)),
                ids(ReciprocalRankFusion.fuse(List.of(bm25, dense), 10)));
    }

    @Test
    @DisplayName("limit caps the output")
    void limitTruncates() {
        List<Hit> ranking = List.of(new Hit("a", 3.0), new Hit("b", 2.0), new Hit("c", 1.0));

        List<Hit> fused = ReciprocalRankFusion.fuse(List.of(ranking), 2);

        assertEquals(List.of("a", "b"), ids(fused));
    }

    @Test
    @DisplayName("empty input fuses to nothing")
    void handlesEmptyInput() {
        assertTrue(ReciprocalRankFusion.fuse(List.of(), 10).isEmpty());
        assertTrue(ReciprocalRankFusion.fuse(List.of(List.of(), List.of()), 10).isEmpty());
    }

    @Test
    @DisplayName("a larger k flattens the difference between positions")
    void kControlsRankDamping() {
        // k = 60 is Cormack et al.'s value and is not tuned here. This shows
        // what it does: the gap between rank 1 and rank 2 shrinks as k grows.
        List<Hit> ranking = List.of(new Hit("a", 1.0), new Hit("b", 1.0));

        List<Hit> tight = ReciprocalRankFusion.fuse(List.of(ranking), 1, 10);
        List<Hit> loose = ReciprocalRankFusion.fuse(List.of(ranking), 1000, 10);

        double tightGap = tight.get(0).score() - tight.get(1).score();
        double looseGap = loose.get(0).score() - loose.get(1).score();

        assertTrue(tightGap > looseGap, "a smaller k should separate the positions more");
    }
}
