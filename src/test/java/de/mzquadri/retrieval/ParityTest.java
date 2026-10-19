package de.mzquadri.retrieval;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Holds this port to the rankings of the Python implementation it came from.
 *
 * <p>The fixtures under {@code src/test/resources/parity/} were produced by
 * {@code tools/export_python_reference.py} running against a checkout of
 * insureassist-rag-mlops. Nothing in this repository defines what "correct"
 * means; that repository does, and these tests fail if Java disagrees with it.
 *
 * <p>Java chunks the corpus files itself rather than reading the exported
 * chunks, so the chunker is under test too. If it were trusted instead, a
 * chunking difference would show up as a confusing BM25 mismatch rather than as
 * the chunk-level failure it is.
 *
 * <h2>What is compared exactly, and what within a tolerance</h2>
 *
 * <p>Chunk ids, chunk offsets, chunk text and the ORDER of every ranking are
 * compared exactly. Scores are compared within a relative tolerance of 1e-12.
 *
 * <p>The tolerance is there for one reason: the inverse document frequency goes
 * through a natural logarithm, and Python's {@code math.log} calls the platform
 * libm while Java uses {@code StrictMath.log}. Both are correct to within an
 * ulp but are not guaranteed to produce the same bits. Everything else -
 * additions, multiplications, divisions, the hash, the sort - is exactly
 * reproducible, which is why the ranking order is asserted without any
 * tolerance at all. A score difference large enough to reorder two chunks would
 * fail the order assertion, so the tolerance cannot hide a real divergence.
 */
class ParityTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    /** Relative tolerance, justified in the class comment. */
    private static final double SCORE_TOLERANCE = 1e-12;

    private static JsonNode chunksFixture;
    private static JsonNode rankingsFixture;
    private static List<Chunk> chunks;
    private static Bm25Index index;

    @BeforeAll
    static void loadEverything() throws IOException {
        chunksFixture = readJson("/parity/chunks.json");
        rankingsFixture = readJson("/parity/rankings.json");

        JsonNode stamp = chunksFixture.get("stamp");
        int size = stamp.get("chunking").get("size").asInt();
        int overlap = stamp.get("chunking").get("overlap").asInt();

        List<Document> documents = loadCorpus();
        chunks = Chunker.chunkCorpus(documents, size, overlap);
        index = new Bm25Index(
                chunks,
                stamp.get("bm25").get("k1").asDouble(),
                stamp.get("bm25").get("b").asDouble());
    }

    // ----------------------------------------------------------------- corpus

    /**
     * Loads the documents in manifest order, checking each against its recorded
     * hash first.
     *
     * <p>Order matters: it decides chunk order, which decides every document
     * frequency in the index. Globbing the directory would make the ranking
     * depend on filesystem iteration order.
     */
    private static List<Document> loadCorpus() throws IOException {
        JsonNode manifest = readJson("/corpus/manifest.json");
        List<Document> documents = new ArrayList<>();

        for (JsonNode entry : manifest.get("documents")) {
            String id = entry.get("document_id").asText();
            String text = readText("/corpus/" + entry.get("filename").asText());

            assertEquals(
                    entry.get("sha256").asText(),
                    sha256Hex(text),
                    id + " does not match the hash recorded in its manifest, so every number "
                            + "below would describe a corpus the fixture was not built from");

            documents.add(new Document(id, text));
        }
        return documents;
    }

    // ------------------------------------------------------------------ tests

    @Test
    @DisplayName("the corpus is the one the fixture was exported from")
    void corpusMatchesTheFixture() {
        // The exporter stamps a hash over the whole document set. Checking it
        // here means a fixture regenerated against a different corpus cannot be
        // compared against this one by accident.
        JsonNode manifest;
        try {
            manifest = readJson("/corpus/manifest.json");
        } catch (IOException e) {
            throw new AssertionError(e);
        }

        StringBuilder joined = new StringBuilder();
        boolean first = true;
        for (JsonNode entry : manifest.get("documents")) {
            if (!first) {
                joined.append("|");
            }
            joined.append(entry.get("document_id").asText())
                    .append(":")
                    .append(entry.get("sha256").asText());
            first = false;
        }

        assertEquals(
                chunksFixture.get("stamp").get("corpus_hash").asText(),
                sha256Hex(joined.toString()),
                "corpus_hash differs from the fixture's");
    }

    @Test
    @DisplayName("the tokenizer agrees on a probe with a thousands separator")
    void tokenizerMatchesTheProbe() {
        // The exporter runs one awkward string through Python's tokenize and
        // records the result: a dollar amount with a thousands separator, a
        // parenthesised abbreviation and a CFR citation.
        JsonNode probe = chunksFixture.get("stamp").get("tokenizer_probe");
        List<String> expected = new ArrayList<>();
        probe.forEach(node -> expected.add(node.asText()));

        List<String> actual =
                Tokenizer.tokenize("Up to $30,000 under Coverage D (ICC); see 44 CFR 61.");

        assertEquals(expected, actual);
        // The point of the rule, stated as its own expectation so a regression
        // names itself.
        assertTrue(actual.contains("30000"), "the thousands separator should be joined");
        assertFalse(actual.contains("30"), "30,000 should not split into 30 and 000");
    }

    @Test
    @DisplayName("chunking reproduces every chunk id, offset and body")
    void chunkingMatchesExactly() {
        JsonNode expected = chunksFixture.get("chunks");

        assertEquals(expected.size(), chunks.size(), "chunk count");

        for (int i = 0; i < expected.size(); i++) {
            JsonNode want = expected.get(i);
            Chunk got = chunks.get(i);
            String where = "chunk " + i + " (" + want.get("chunk_id").asText() + ")";

            assertEquals(want.get("chunk_id").asText(), got.chunkId(), where + " id");
            assertEquals(want.get("document_id").asText(), got.documentId(), where + " document");
            assertEquals(want.get("chunk_index").asInt(), got.chunkIndex(), where + " index");
            assertEquals(want.get("start").asInt(), got.start(), where + " start");
            assertEquals(want.get("end").asInt(), got.end(), where + " end");
            assertEquals(want.get("text").asText(), got.text(), where + " text");
        }
    }

    @Test
    @DisplayName("BM25 ranks all 50 NFIP questions identically to Python")
    void bm25RankingsMatch() {
        JsonNode rankings = rankingsFixture.get("rankings");
        int depth = rankingsFixture.get("stamp").get("rrf").get("candidate_depth").asInt();

        assertTrue(rankings.size() > 0, "the fixture holds no questions");

        for (JsonNode entry : rankings) {
            String questionId = entry.get("question_id").asText();
            List<Hit> actual = index.search(entry.get("question").asText(), depth);
            assertRankingMatches(entry.get("bm25"), actual, "BM25 " + questionId);
        }
    }

    @Test
    @DisplayName("RRF fuses the exported dense and BM25 lists identically to Python")
    void rrfFusionMatches() {
        JsonNode stamp = rankingsFixture.get("stamp");
        int k = stamp.get("rrf").get("k").asInt();
        int limit = stamp.get("rrf").get("limit").asInt();
        int depth = stamp.get("rrf").get("candidate_depth").asInt();

        int fused = 0;
        for (JsonNode entry : rankingsFixture.get("rankings")) {
            if (!entry.has("dense") || !entry.has("rrf")) {
                // The exporter can be run with --skip-dense, which leaves BM25
                // as the only comparable arm.
                continue;
            }
            String questionId = entry.get("question_id").asText();

            // BM25 is recomputed here rather than read back, so this test covers
            // the whole chain. The dense list is a fixture by necessity: it needs
            // a sentence-transformer, which is not run on the JVM. RRF reads only
            // positions, so a fixed list is a faithful input for it.
            List<Hit> bm25 = index.search(entry.get("question").asText(), depth);
            List<Hit> dense = readHits(entry.get("dense"));

            List<Hit> actual = ReciprocalRankFusion.fuse(List.of(bm25, dense), k, limit);
            assertRankingMatches(entry.get("rrf"), actual, "RRF " + questionId);
            fused++;
        }

        assertTrue(fused > 0, "no question carried a dense ranking, so RRF was never compared");
    }

    @Test
    @DisplayName("the corpus holds no character where Python and Java indices diverge")
    void corpusStaysInsideTheBasicMultilingualPlane() {
        // Chunk offsets are Python code points and Java UTF-16 code units. They
        // agree only while every character is in the BMP. Asserted rather than
        // assumed, because a surrogate pair anywhere in the corpus would shift
        // every offset after it and the failure would surface as a chunk id
        // mismatch with no obvious cause.
        for (Chunk chunk : chunks) {
            String text = chunk.text();
            assertEquals(
                    text.length(),
                    text.codePointCount(0, text.length()),
                    "chunk " + chunk.chunkId() + " contains a non-BMP character");
        }
    }

    // ------------------------------------------------------------- assertions

    private static void assertRankingMatches(JsonNode expected, List<Hit> actual, String label) {
        assertEquals(expected.size(), actual.size(), label + ": ranking length");

        for (int i = 0; i < expected.size(); i++) {
            JsonNode want = expected.get(i);
            Hit got = actual.get(i);

            // Order and identity exactly. This is the assertion that matters:
            // a ranking is what the system returns, and scores are an internal
            // detail of how it got there.
            assertEquals(
                    want.get("chunk_id").asText(),
                    got.chunkId(),
                    label + ": chunk at position " + (i + 1));

            // Parsed from Python's repr(), which round-trips a double exactly,
            // so the comparison is against the value Python held rather than a
            // rounded decimal.
            double wanted = Double.parseDouble(want.get("score").asText());
            double tolerance = Math.max(Math.abs(wanted) * SCORE_TOLERANCE, Double.MIN_NORMAL);
            assertEquals(
                    wanted,
                    got.score(),
                    tolerance,
                    label + ": score at position " + (i + 1));
        }
    }

    // ----------------------------------------------------------------- plumbing

    private static JsonNode readJson(String resource) throws IOException {
        try (InputStream stream = ParityTest.class.getResourceAsStream(resource)) {
            if (stream == null) {
                throw new IOException("missing test resource " + resource);
            }
            return MAPPER.readTree(stream);
        }
    }

    private static String readText(String resource) throws IOException {
        try (InputStream stream = ParityTest.class.getResourceAsStream(resource)) {
            if (stream == null) {
                throw new IOException("missing test resource " + resource);
            }
            // UTF-8 explicitly. The corpus has em dashes and curly quotes, and
            // reading it under a Windows code page changes the bytes, the hash
            // and every offset.
            return new String(stream.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    private static List<Hit> readHits(JsonNode array) {
        List<Hit> hits = new ArrayList<>(array.size());
        for (JsonNode node : array) {
            hits.add(new Hit(
                    node.get("chunk_id").asText(),
                    Double.parseDouble(node.get("score").asText())));
        }
        return hits;
    }

    private static String sha256Hex(String value) {
        MessageDigest digest;
        try {
            digest = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
        byte[] hash = digest.digest(value.getBytes(StandardCharsets.UTF_8));
        StringBuilder out = new StringBuilder(hash.length * 2);
        for (byte b : hash) {
            out.append(Character.forDigit((b >> 4) & 0xF, 16));
            out.append(Character.forDigit(b & 0xF, 16));
        }
        return out.toString();
    }
}
