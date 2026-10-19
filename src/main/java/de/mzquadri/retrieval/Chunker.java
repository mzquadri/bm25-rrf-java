package de.mzquadri.retrieval;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.List;

/**
 * Fixed-size overlapping windows over a document's characters.
 *
 * <p>A port of {@code src/corpus.py:chunk_document}. Chunking has to match the
 * Python side before any score can, because BM25 statistics are computed over
 * the chunks: a different number of chunks changes every inverse document
 * frequency in the index.
 *
 * <p>Offsets are character indices. Python indexes by code point and Java by
 * UTF-16 code unit, so the two diverge on any text outside the Basic
 * Multilingual Plane. The NFIP corpus contains none - its only non-ASCII
 * characters are the em dash and the curly quotes, all in the BMP - and
 * {@code ChunkerTest} asserts that rather than leaving it as an assumption.
 */
public final class Chunker {

    private Chunker() {
    }

    /**
     * @param size    window length in characters, greater than the overlap
     * @param overlap how much each window repeats of the one before
     * @throws IllegalArgumentException on a configuration that cannot terminate
     */
    public static void validate(int size, int overlap) {
        // The cursor advances by size - overlap. A non-positive step never
        // terminates, and in the Python original both values come from the
        // environment, so this is reachable by configuration alone.
        if (size <= 0) {
            throw new IllegalArgumentException("chunk size must be positive, got " + size);
        }
        if (overlap < 0) {
            throw new IllegalArgumentException("chunk overlap cannot be negative, got " + overlap);
        }
        if (overlap >= size) {
            throw new IllegalArgumentException(
                    "chunk overlap (" + overlap + ") must be smaller than size (" + size + ")");
        }
    }

    /** Chunks every document, keeping document order and chunk order within a document. */
    public static List<Chunk> chunkCorpus(List<Document> documents, int size, int overlap) {
        List<Chunk> all = new ArrayList<>();
        for (Document document : documents) {
            all.addAll(chunkDocument(document, size, overlap));
        }
        return all;
    }

    public static List<Chunk> chunkDocument(Document document, int size, int overlap) {
        validate(size, overlap);

        List<Chunk> chunks = new ArrayList<>();
        String text = document.text();
        int cursor = 0;
        int index = 0;

        while (cursor < text.length()) {
            int windowEnd = Math.min(cursor + size, text.length());
            String window = text.substring(cursor, windowEnd);

            int lead = PythonText.firstNonSpace(window);
            int trailEnd = PythonText.lastNonSpaceEnd(window);

            // An all-whitespace window produces no chunk and does not advance
            // the chunk index, so indices stay contiguous.
            if (lead < trailEnd) {
                String stripped = window.substring(lead, trailEnd);
                int start = cursor + lead;
                chunks.add(new Chunk(
                        makeChunkId(document.documentId(), start, stripped),
                        document.documentId(),
                        index,
                        start,
                        start + stripped.length(),
                        stripped));
                index++;
            }

            // Checked after appending, so the final window is kept. Advancing
            // first would drop the tail of every document.
            if (windowEnd == text.length()) {
                break;
            }
            cursor = windowEnd - overlap;
        }

        return chunks;
    }

    /**
     * A stable id derived from where the text is and what it says.
     *
     * <p>The document id and the offset are both inside the hash on purpose.
     * The three NFIP forms share passages word for word, so hashing the text
     * alone would collide across documents and silently merge distinct chunks.
     */
    public static String makeChunkId(String documentId, int start, String text) {
        String payload = documentId + "|" + start + "|" + text;
        return documentId + "#" + sha256Hex(payload).substring(0, 12);
    }

    private static String sha256Hex(String value) {
        MessageDigest digest;
        try {
            digest = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            // SHA-256 is required of every Java platform implementation.
            throw new IllegalStateException("SHA-256 unavailable", e);
        }
        byte[] hash = digest.digest(value.getBytes(StandardCharsets.UTF_8));
        StringBuilder out = new StringBuilder(hash.length * 2);
        for (byte b : hash) {
            // Lowercase hex, two digits per byte, to match Python's hexdigest().
            out.append(Character.forDigit((b >> 4) & 0xF, 16));
            out.append(Character.forDigit(b & 0xF, 16));
        }
        return out.toString();
    }
}
