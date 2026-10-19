package de.mzquadri.retrieval;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class Bm25IndexTest {

    private static Chunk chunk(String id, String text) {
        return new Chunk(id, "doc", 0, 0, text.length(), text);
    }

    private static List<String> ids(List<Hit> hits) {
        List<String> out = new ArrayList<>(hits.size());
        for (Hit hit : hits) {
            out.add(hit.chunkId());
        }
        return out;
    }

    @Test
    @DisplayName("a chunk sharing no term with the query is absent, not scored zero")
    void omitsUnmatchedChunks() {
        Bm25Index index = new Bm25Index(List.of(
                chunk("a", "flood insurance policy"),
                chunk("b", "entirely unrelated wording")));

        List<Hit> hits = index.search("flood", 10);

        assertEquals(List.of("a"), ids(hits));
    }

    @Test
    @DisplayName("ties are broken by chunk id, so a run is reproducible")
    void tiesBreakOnChunkId() {
        // Identical text means identical scores. Without the secondary sort the
        // order would depend on iteration order and move between runs.
        Bm25Index index = new Bm25Index(List.of(
                chunk("zzz", "flood"),
                chunk("aaa", "flood"),
                chunk("mmm", "flood")));

        List<Hit> hits = index.search("flood", 10);

        assertEquals(List.of("aaa", "mmm", "zzz"), ids(hits));
        assertEquals(hits.get(0).score(), hits.get(2).score(), 0.0, "scores should be tied");
    }

    @Test
    @DisplayName("a term in every chunk still scores positively, and the floor is not what does it")
    void ubiquitousTermStaysPositive() {
        // The `+ 1.0` inside the logarithm is what keeps idf positive, not the
        // max(..., 1e-9) floor. This test originally asserted the floor was
        // reached here and failed: with 3 chunks all containing the term,
        // idf is ln(0.5/3.5 + 1) = 0.1335, nowhere near 1e-9.
        //
        // Worth being exact, because the Python original's comment credits the
        // floor with preventing a negative idf. Classic Robertson/Sparck-Jones
        // idf does go negative once a term is in more than half the documents,
        // but this formula adds 1 inside the log, so the argument always exceeds
        // 1 and the result is always positive. The floor is defensive, and
        // effectively dead: the smallest possible value is at df == N, which is
        // ln(1 + 0.5/(N + 0.5)), and that only drops below 1e-9 beyond about
        // 500 million chunks.
        List<Chunk> chunks = List.of(
                chunk("a", "flood"),
                chunk("b", "flood"),
                chunk("c", "flood"));
        Bm25Index index = new Bm25Index(chunks);

        double idf = index.idfOf("flood");
        assertEquals(Math.log(1 + 0.5 / 3.5), idf, 1e-12, "idf at df == N");
        assertTrue(idf > 0, "a term in every chunk must not penalise the chunks containing it");
        assertTrue(idf > 1e-9, "the floor is not reached at this corpus size");

        for (Hit hit : index.search("flood", 10)) {
            assertTrue(hit.score() > 0, "every matching chunk scores positively");
        }
    }

    @Test
    @DisplayName("idf shrinks towards zero as a term becomes more common, without crossing it")
    void idfShrinksMonotonically() {
        // The property the formula actually guarantees: more documents sharing a
        // term means less discriminating power, approaching zero from above.
        double previous = Double.MAX_VALUE;
        for (int total : new int[] {3, 10, 50, 200}) {
            List<Chunk> chunks = new ArrayList<>();
            for (int i = 0; i < total; i++) {
                chunks.add(chunk("c" + i, "flood"));
            }
            double idf = new Bm25Index(chunks).idfOf("flood");

            assertTrue(idf > 0, "idf stays positive at N=" + total);
            assertTrue(idf < previous, "idf should shrink as N grows, at N=" + total);
            previous = idf;
        }
    }

    @Test
    @DisplayName("a repeated query term is counted twice")
    void repeatedQueryTermsCountTwice() {
        // The tokenizer keeps duplicates and search does not collapse them, so a
        // term given twice contributes twice. This matches the Python original;
        // de-duplicating would change both the score and the summation order.
        Bm25Index index = new Bm25Index(List.of(
                chunk("a", "flood damage claim"),
                chunk("b", "unrelated text here")));

        double once = index.search("flood", 10).get(0).score();
        double twice = index.search("flood flood", 10).get(0).score();

        assertEquals(once * 2, twice, Math.abs(once) * 1e-12);
    }

    @Test
    @DisplayName("a shorter chunk outranks a longer one containing the term equally often")
    void lengthNormalisationFavoursShorterChunks() {
        // What b = 0.75 is for: one mention in a short passage is stronger
        // evidence than one mention buried in a long one.
        StringBuilder padding = new StringBuilder();
        for (int i = 0; i < 100; i++) {
            padding.append("filler ");
        }
        Bm25Index index = new Bm25Index(List.of(
                chunk("short", "flood"),
                chunk("long", "flood " + padding)));

        assertEquals(List.of("short", "long"), ids(index.search("flood", 10)));
    }

    @Test
    @DisplayName("limit caps the result without changing the order")
    void limitTruncates() {
        Bm25Index index = new Bm25Index(List.of(
                chunk("a", "flood"),
                chunk("b", "flood water"),
                chunk("c", "flood water damage")));

        List<Hit> all = index.search("flood", 10);
        List<Hit> capped = index.search("flood", 2);

        assertEquals(2, capped.size());
        assertEquals(ids(all).subList(0, 2), ids(capped));
    }

    @Test
    @DisplayName("an empty index answers rather than dividing by zero")
    void handlesEmptyCorpus() {
        // Average chunk length is 0/0 on an empty corpus. Unguarded that is NaN,
        // which would propagate into every score silently.
        Bm25Index index = new Bm25Index(List.of());

        assertEquals(0, index.size());
        assertEquals(0.0, index.averageLength(), 0.0);
        assertTrue(index.search("flood", 10).isEmpty());
    }

    @Test
    @DisplayName("a query with no known term returns nothing")
    void unknownTermsReturnNothing() {
        Bm25Index index = new Bm25Index(List.of(chunk("a", "flood insurance")));

        assertTrue(index.search("helicopter", 10).isEmpty());
        assertTrue(index.search("", 10).isEmpty());
    }
}
